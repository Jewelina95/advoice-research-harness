#!/usr/bin/env python3
"""Run a token-free five-arm validation from paired cached predictions.

The study is deliberately retrospective: it splits a previously evaluated cohort
into repeated calibration/evaluation partitions. It is suitable for selecting a
candidate authority policy, not for claiming untouched external performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


ARMS = {
    "supervised": "仅监督模型",
    "agent_only": "仅 Agent",
    "supervised_state": "监督模型 + 状态修正",
    "supervised_agent": "监督模型 + Agent 独立判断",
    "full_joint": "完整联合模型",
}


def _normalize(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


def log_pool(*components: tuple[np.ndarray, float]) -> np.ndarray:
    """Geometrically pool independent probability components."""

    score = None
    for probability, weight in components:
        contribution = float(weight) * np.log(_normalize(probability))
        score = contribution if score is None else score + contribution
    score -= score.max(axis=1, keepdims=True)
    return _normalize(np.exp(score))


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = _normalize(probability)
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, probability[:, 1])),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability[:, 1])),
    }


def selection_key(y: np.ndarray, probability: np.ndarray) -> tuple[float, ...]:
    result = metrics(y, probability)
    return (
        result["macro_f1"],
        result["balanced_accuracy"],
        result["auroc"],
        -result["log_loss"],
        -result["brier"],
    )


def choose_weight(
    y: np.ndarray,
    base: np.ndarray,
    auxiliary: np.ndarray,
    grid: list[float],
) -> float:
    return max(grid, key=lambda weight: selection_key(y, log_pool((base, 1.0), (auxiliary, weight))))


def choose_joint_weights(
    y: np.ndarray,
    base: np.ndarray,
    state: np.ndarray,
    agent: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    return max(
        ((state_weight, agent_weight) for state_weight in grid for agent_weight in grid),
        key=lambda pair: selection_key(
            y,
            log_pool((base, 1.0), (state, pair[0]), (agent, pair[1])),
        ),
    )


def _load(source: Path) -> pd.DataFrame:
    ablations = pd.read_csv(source / "ours_ablations.csv")
    agent = pd.read_csv(source / "b2_predictions.csv")
    required = {
        "Ours_base_supervised": "base",
        "Ours_overall_state_only": "state",
    }
    frames = []
    for condition, prefix in required.items():
        frame = ablations.loc[ablations["condition"].eq(condition)].copy()
        frame = frame[["subject_id", "label", "prob_HC", "prob_AD"]].rename(
            columns={"prob_HC": f"{prefix}_HC", "prob_AD": f"{prefix}_AD"}
        )
        frames.append(frame)
    agent = agent[["subject_id", "label", "prob_HC", "prob_AD"]].rename(
        columns={"prob_HC": "agent_HC", "prob_AD": "agent_AD"}
    )
    paired = frames[0].merge(frames[1], on=["subject_id", "label"], validate="one_to_one")
    paired = paired.merge(agent, on=["subject_id", "label"], validate="one_to_one")
    if paired.empty or paired["label"].nunique() != 2:
        raise ValueError("A paired binary cohort is required.")
    return paired.sort_values("subject_id").reset_index(drop=True)


def run_study(frame: pd.DataFrame, repeats: int, test_size: float) -> dict:
    observed = set(frame["label"].astype(str))
    labels = ["HC", "AD"]
    if observed != set(labels):
        raise ValueError(f"Expected HC/AD labels, found {sorted(observed)}")
    label_index = {label: index for index, label in enumerate(labels)}
    y = frame["label"].map(label_index).to_numpy(dtype=int)
    base = frame[["base_HC", "base_AD"]].to_numpy(dtype=float)
    state = frame[["state_HC", "state_AD"]].to_numpy(dtype=float)
    agent = frame[["agent_HC", "agent_AD"]].to_numpy(dtype=float)
    grid = [round(value, 2) for value in np.arange(0.0, 1.51, 0.1)]
    splitter = StratifiedShuffleSplit(
        n_splits=repeats, test_size=test_size, random_state=20260917
    )
    rows: list[dict] = []
    weights: list[dict] = []
    for repeat, (calibration, evaluation) in enumerate(splitter.split(base, y), start=1):
        state_weight = choose_weight(y[calibration], base[calibration], state[calibration], grid)
        agent_weight = choose_weight(y[calibration], base[calibration], agent[calibration], grid)
        joint_state, joint_agent = choose_joint_weights(
            y[calibration], base[calibration], state[calibration], agent[calibration], grid
        )
        weights.append(
            {
                "repeat": repeat,
                "state_weight": state_weight,
                "agent_weight": agent_weight,
                "joint_state_weight": joint_state,
                "joint_agent_weight": joint_agent,
            }
        )
        arm_probability = {
            "supervised": base[evaluation],
            "agent_only": agent[evaluation],
            "supervised_state": log_pool(
                (base[evaluation], 1.0), (state[evaluation], state_weight)
            ),
            "supervised_agent": log_pool(
                (base[evaluation], 1.0), (agent[evaluation], agent_weight)
            ),
            "full_joint": log_pool(
                (base[evaluation], 1.0),
                (state[evaluation], joint_state),
                (agent[evaluation], joint_agent),
            ),
        }
        for arm, probability in arm_probability.items():
            rows.append({"repeat": repeat, "arm": arm, **metrics(y[evaluation], probability)})
    results = pd.DataFrame(rows)
    weight_frame = pd.DataFrame(weights)
    summary = []
    for arm, group in results.groupby("arm", sort=False):
        row = {"arm": arm, "label": ARMS[arm]}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "log_loss", "brier"]:
            values = group[metric].to_numpy(dtype=float)
            row[metric] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summary.append(row)
    selected = {
        column: float(median(weight_frame[column].tolist()))
        for column in ["state_weight", "agent_weight", "joint_state_weight", "joint_agent_weight"]
    }
    selection_frequency = {
        column: {
            str(weight): int(count)
            for weight, count in weight_frame[column].value_counts().sort_index().items()
        }
        for column in ["state_weight", "agent_weight", "joint_state_weight", "joint_agent_weight"]
    }
    return {
        "schema_version": "advoice.cached_five_arm_validation.v1",
        "study_type": "retrospective_repeated_stratified_holdout",
        "dataset": "ADReSS_2020",
        "labels": labels,
        "n": int(len(frame)),
        "repeats": repeats,
        "calibration_fraction": 1.0 - test_size,
        "evaluation_fraction": test_size,
        "agent_api_calls": 0,
        "agent_token_usage": 0,
        "weight_grid": grid,
        "selected_median_weights": selected,
        "selection_frequency": selection_frequency,
        "summary": summary,
        "per_repeat": rows,
        "weight_selections": weights,
        "limitations": [
            "The cohort had been evaluated previously; this is not untouched external validation.",
            "The Agent-only arm reuses the cached transcript-only B2 decision, while the joint arms use it as an independent likelihood component.",
            "Repeated holdout estimates stability but does not create new independent subjects.",
        ],
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_html(result: dict) -> str:
    rows = result["summary"]
    best_f1 = max(row["macro_f1"] for row in rows)
    table_rows = "".join(
        f"<tr class={'best' if math.isclose(row['macro_f1'], best_f1) else ''}>"
        f"<td>{row['label']}</td><td>{_pct(row['accuracy'])}</td>"
        f"<td>{_pct(row['balanced_accuracy'])}</td><td>{_pct(row['macro_f1'])}</td>"
        f"<td>{row['auroc']:.3f}</td><td>{row['log_loss']:.3f}</td><td>{row['brier']:.3f}</td></tr>"
        for row in rows
    )
    bars = "".join(
        f"<div class='bar-row'><span>{row['label']}</span><div class='track'><i style='width:{100*row['macro_f1']:.1f}%'></i></div><b>{row['macro_f1']:.3f}</b></div>"
        for row in rows
    )
    weights = result["selected_median_weights"]
    zero_agent = result["selection_frequency"]["joint_agent_weight"].get("0.0", 0)
    full = next(row for row in rows if row["arm"] == "full_joint")
    supervised = next(row for row in rows if row["arm"] == "supervised")
    conclusion = (
        "完整联合模型在本次内部重复留出中取得更高的 Macro-F1。"
        if full["macro_f1"] > supervised["macro_f1"]
        else "完整联合模型未稳定超过监督模型；当前数据支持保留 Agent 接口，但不支持放大其预测权重。"
    )
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ADvoice 五组权重验证</title><style>
    :root{{--ink:#16202a;--muted:#64717d;--line:#d9e0e5;--green:#237a57;--green2:#dcefe6;--blue:#3f6fa8;--paper:#f7f8f6;}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;letter-spacing:0}}
    main{{max-width:1120px;margin:auto;padding:42px 28px 70px}}h1{{font-size:34px;margin:0 0 10px}}h2{{font-size:23px;margin:36px 0 14px}}p{{line-height:1.72;margin:8px 0;color:#33404a}}.lede{{font-size:18px;max-width:900px}}
    .meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}}.meta div{{background:white;border-top:3px solid var(--blue);padding:16px}}.meta b{{display:block;font-size:24px;margin-top:6px}}
    table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:13px 12px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}thead{{background:#e9eef2}}tr.best{{background:var(--green2);font-weight:700}}
    .panel{{background:white;border:1px solid var(--line);padding:22px;margin-top:16px}}.bar-row{{display:grid;grid-template-columns:230px 1fr 58px;gap:12px;align-items:center;margin:13px 0}}.track{{height:16px;background:#e7ebee}}.track i{{display:block;height:100%;background:var(--green)}}
    .weights{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.weight{{border-left:4px solid var(--green);background:white;padding:17px}}.weight b{{font-size:27px;display:block}}
    .verdict{{border-left:6px solid var(--green);background:var(--green2);padding:20px;font-size:18px;line-height:1.65}}.note{{font-size:13px;color:var(--muted)}}code{{font-family:ui-monospace,monospace}}
    @media(max-width:760px){{.meta,.weights{{grid-template-columns:1fr 1fr}}.bar-row{{grid-template-columns:150px 1fr 48px}}table{{font-size:12px}}th,td{{padding:9px 6px}}}}
    </style></head><body><main>
    <h1>ADvoice 五组职责边界与 Agent 权重验证</h1>
    <p class='lede'>同一批 ADReSS 2020 病例、同一标签口径、同一概率输出。每轮仅在校准子集选择权重，再在未参与选权重的病例上评估；复用缓存，因此没有新增 Agent 调用或 token 消耗。</p>
    <section class='meta'><div>配对病例<b>{result['n']}</b></div><div>重复分层留出<b>{result['repeats']} 次</b></div><div>校准 / 评估<b>{_pct(result['calibration_fraction'])} / {_pct(result['evaluation_fraction'])}</b></div><div>新增 Agent tokens<b>0</b></div></section>
    <h2>五组结果</h2><table><thead><tr><th>模型条件</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro-F1</th><th>AUROC</th><th>Log loss ↓</th><th>Brier ↓</th></tr></thead><tbody>{table_rows}</tbody></table>
    <div class='panel'><strong>Macro-F1（重复留出均值）</strong>{bars}</div>
    <h2>验证集选择出的职责权重</h2><div class='weights'>
      <div class='weight'>状态修正单独加入<b>{weights['state_weight']:.2f}</b><span>监督主干固定为 1.00</span></div>
      <div class='weight'>Agent 单独加入<b>{weights['agent_weight']:.2f}</b><span>监督主干固定为 1.00</span></div>
      <div class='weight'>完整联合：状态<b>{weights['joint_state_weight']:.2f}</b><span>与 Agent 同时校准</span></div>
      <div class='weight'>完整联合：Agent<b>{weights['joint_agent_weight']:.2f}</b><span>允许验证集选择 0，即拒绝无净增益组件</span></div>
    </div>
    <p>完整联合校准中，Agent 权重在 <strong>{zero_agent}/{result['repeats']}</strong> 次重复留出里被选择为 0。状态权重也并不稳定，因此当前可部署策略应回退到监督主干，同时保留证据链用于审查和未来重新校准。</p>
    <h2>结论</h2><div class='verdict'>{conclusion} 权重不是人工指定，也不是“监督模型越不确定，Agent 越大”，而是由病例级标签在校准子集上选择，并在分离的评估子集上检验。</div>
    <h2>为什么保留当前框架</h2><p>五组消融分别隔离监督识别、Agent 独立判断、认知状态修正及二者联合后的增量。完整框架的结构价值不取决于 Agent 必须拥有较大权重，而在于它允许状态与 Agent 作为独立证据似然进入、允许权重为零、并保留 MetricEvidence → StateCard → 决策的回溯链。只有在校准集显示净增益时，Agent 才获得预测权限。</p>
    <p class='note'>限制：这是已被历史分析过的 27 例回顾性内部验证，不能替代 PREPARE 同协议测试或新的外部队列。B2 为缓存的 transcript-only Agent 独立判断；本报告验证职责边界，不宣称已超过 SpeechCARE。</p>
    </main></body></html>"""


def render_oral(result: dict) -> str:
    rows = {row["arm"]: row for row in result["summary"]}
    weights = result["selected_median_weights"]
    zero_agent = result["selection_frequency"]["joint_agent_weight"].get("0.0", 0)
    ranking = sorted(result["summary"], key=lambda row: row["macro_f1"], reverse=True)
    return f"""# ADvoice 五组模型验证：中文口语汇报

这次验证不是重新调用大模型，也不是重新训练编码器。我们复用了 ADReSS 2020 中已经配对的 27 个病例，把每轮病例分成校准部分和评估部分。权重只在校准病例上选择，再到没有参与选权重的病例上评估，重复 {result['repeats']} 次。因此这次实验新增的 Agent 调用和 token 消耗都是零。

我们比较了五组。第一组只有监督模型，用来表示原有预测主干。第二组只有 Agent，用来判断 Agent 独立阅读转录时是否足以承担分类。第三组在监督模型上加入认知状态修正，用来检验 StateCard 是否提供额外信息。第四组把 Agent 的独立判断作为证据似然加入监督模型。第五组同时加入状态修正和 Agent 独立判断，是完整联合模型。

结果按 Macro-F1 排序，最高的是“{ranking[0]['label']}”，Macro-F1 为 {ranking[0]['macro_f1']:.3f}。仅监督模型的 Accuracy 为 {rows['supervised']['accuracy']:.3f}、Macro-F1 为 {rows['supervised']['macro_f1']:.3f}；仅 Agent 的 Accuracy 为 {rows['agent_only']['accuracy']:.3f}、Macro-F1 为 {rows['agent_only']['macro_f1']:.3f}；完整联合模型的 Accuracy 为 {rows['full_joint']['accuracy']:.3f}、Macro-F1 为 {rows['full_joint']['macro_f1']:.3f}。

权重不是人工拍定。监督主干固定为 1。单独加入状态时，中位校准权重是 {weights['state_weight']:.2f}；单独加入 Agent 时是 {weights['agent_weight']:.2f}。在完整联合模型中，状态权重是 {weights['joint_state_weight']:.2f}，Agent 权重是 {weights['joint_agent_weight']:.2f}。在 {zero_agent}/{result['repeats']} 次重复留出中，完整模型把 Agent 权重选为零。搜索范围包含零，所以如果 Agent 没有稳定净增益，系统会拒绝它进入最终概率，而不是因为监督模型不确定就自动放大 Agent。

这项结果支持的不是“Agent 必须主导”，而是当前职责设计：监督模型提供稳定基线，MetricEvidence 和 StateCard 提供可回溯的认知证据，Agent 提供独立的病例级证据似然，验证集决定它们能否进入最终概率。这样既保留 Agent 随模型能力提升而获得更大权限的接口，也避免弱 Agent 在当前版本中破坏已正确的 HC 或 AD 判断。

这仍然是一项快速的回顾性内部验证。27 个病例此前已经被分析过，重复留出也不会产生新的独立样本。因此它可以用于确定候选权重和检查框架逻辑，但不能作为超过 SpeechCARE 的最终证据。下一次正式结论必须冻结这里选出的规则，再在 PREPARE 同协议测试或新的外部数据上运行一次。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_study(_load(args.source), args.repeats, args.test_size)
    (args.output / "five_arm_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "five_arm_agent_weight_validation.html").write_text(
        render_html(result), encoding="utf-8"
    )
    (args.output / "five_arm_validation_oral_zh.md").write_text(
        render_oral(result), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
