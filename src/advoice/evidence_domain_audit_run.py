"""Build non-Agent acoustic, language, dialogue and construct-coverage audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REMEDIATION = [
    {"construct": "pause_duration_variability", "state_id": "S01", "historical_metrics": "StddevUnvoicedSegmentLength", "implementation": "Add pause_sd_sec and pause_iqr_sec from the existing VAD run-length vector.", "data_requirement": "patient-role audio", "readiness": "candidate_implemented_validation_pending", "validation": "VAD sensitivity; test-retest stability; redundancy with pause_mean_sec/pause_p90_sec"},
    {"construct": "response_latency", "state_id": "S01;S13", "historical_metrics": "response_latency", "implementation": "Compute prompt-end to patient-onset latency per turn, then robust median and upper quantile.", "data_requirement": "speaker roles plus aligned turn timestamps", "readiness": "dataset_conditional", "validation": "exclude interviewer overlap, hearing/protocol effects and missing prompt boundaries"},
    {"construct": "noun_ratio", "state_id": "S07", "historical_metrics": "noun_rate", "implementation": "Language-specific POS tagging with a shared universal-POS output contract.", "data_requirement": "reliable transcript and language-specific POS model", "readiness": "needs_multilingual_model", "validation": "manual POS audit by language; ASR perturbation; task calibration"},
    {"construct": "mtld", "state_id": "S08", "historical_metrics": "MTLD", "implementation": "Add bidirectional MTLD with a minimum-token validity rule.", "data_requirement": "tokenized transcript", "readiness": "candidate_implemented_validation_pending", "validation": "short-sample bias; correlation with MATTR/TTR; bootstrap stability"},
    {"construct": "clause_complexity", "state_id": "S11", "historical_metrics": "clause_per_utterance", "implementation": "Universal-dependency clause counts per patient utterance.", "data_requirement": "utterance boundaries and multilingual dependency parser", "readiness": "needs_multilingual_model", "validation": "parser accuracy and transcript-error sensitivity by language"},
    {"construct": "dependency_length", "state_id": "S11", "historical_metrics": "dependency_length", "implementation": "Mean dependency distance and parse-depth summaries on patient utterances.", "data_requirement": "multilingual dependency parse", "readiness": "needs_multilingual_model", "validation": "parser confidence; utterance length adjustment; cross-language comparability"},
    {"construct": "repetition_rate", "state_id": "S12", "historical_metrics": "repeat_token_rate;repeat_bigram_rate", "implementation": "Token and bigram recurrence after removing immediate CHAT repair annotations.", "data_requirement": "language-conditioned tokens", "readiness": "candidate_implemented_validation_pending", "validation": "separate pathological repetition from task-required repetition"},
    {"construct": "semantic_coherence", "state_id": "S09", "historical_metrics": "semantic_coherence_tfidf;embedding_coherence;topic_drift;task_relevance", "implementation": "Utterance adjacency coherence plus task-reference relevance, with separate drift and omission outputs.", "data_requirement": "utterance transcript, task reference and multilingual sentence encoder", "readiness": "research_validation", "validation": "human discourse ratings; no pooled threshold across tasks/languages"},
    {"construct": "task_specific_performance", "state_id": "S14", "historical_metrics": "story_recall_units;naming_correct;semantic_fluency_count;reading_error_rate", "implementation": "Use a separate scoring adapter and answer key for each cognitive task.", "data_requirement": "task identity, scoring key and aligned response", "readiness": "dataset_conditional", "validation": "score against task gold standard; never impute across incompatible tasks"},
]


def _load_yaml(path: Path, key: str) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[key]


def state_coverage(
    history: pd.DataFrame, mapping: pd.DataFrame, states: list[dict[str, Any]], metrics: list[dict[str, Any]],
) -> pd.DataFrame:
    metric_role = {metric["id"]: metric.get("role", "") for metric in metrics}
    rows = []
    for state in states:
        state_id = state["id"]
        historical = history.loc[history.state_id.eq(state_id)]
        mapped = mapping.loc[mapping.state_id.eq(state_id) & mapping.disposition.eq("mapped_to_runtime_registry")]
        configured = list(state.get("metrics", []))
        unresolved = mapping.loc[
            mapping.state_id.eq(state_id)
            & mapping.disposition.eq("legacy_core_without_confirmed_runtime_successor")
        ]
        rows.append({
            "state_id": state_id,
            "state_name": state.get("name_zh", state.get("name", "")),
            "historical_rows": len(historical),
            "historical_unique_metrics": historical.metric_name.nunique(),
            "mapped_historical_rows": len(mapped),
            "current_configured_metrics": len(configured),
            "current_report_permitted_metrics": sum(metric_role.get(metric_id) in {"clinical_support", "cautious_support"} for metric_id in configured),
            "unresolved_core_rows": len(unresolved),
            "unresolved_core_metrics": ";".join(unresolved.historical_metric.astype(str).unique()),
            "state_status": state.get("status", "active"),
        })
    return pd.DataFrame(rows)


def collect_observability(
    source: Path, dataset_ids: list[str], metrics: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definition = {metric["id"]: metric for metric in metrics}
    coverage_rows, pair_rows, dimension_rows, dialogue_rows = [], [], [], []
    dialogue_ids = {"role_coverage_fraction", "patient_turn_share", "mean_utterance_words", "repair_rate_100w"}
    for dataset_id in dataset_ids:
        folder = source / dataset_id
        coverage = pd.read_csv(folder / "coverage.csv")
        coverage.insert(0, "dataset_id", dataset_id)
        overall = coverage.loc[coverage.task_scope.eq("overall")].copy()
        coverage_rows.append(overall)
        dialogue_rows.append(overall.loc[overall.metric_id.isin(dialogue_ids)])

        pairs = pd.read_csv(folder / "pairwise_overlap.csv")
        pairs = pairs.loc[pairs.task_scope.eq("overall") & pairs.review_candidate.fillna(False)].copy()
        pairs.insert(0, "dataset_id", dataset_id)
        pair_rows.append(pairs)

        dimensions = pd.read_csv(folder / "dimension_summary.csv")
        dimensions = dimensions.loc[dimensions.task_scope.eq("overall") & dimensions.status.eq("ok")].copy()
        dimensions.insert(0, "dataset_id", dataset_id)
        dimension_rows.append(dimensions)
    all_coverage = pd.concat(coverage_rows, ignore_index=True)
    all_pairs = pd.concat(pair_rows, ignore_index=True)
    all_dimensions = pd.concat(dimension_rows, ignore_index=True)
    dialogue = pd.concat(dialogue_rows, ignore_index=True)
    all_coverage["configured_state"] = all_coverage.metric_id.map(lambda value: definition.get(value, {}).get("state", ""))
    all_coverage["configured_branch"] = all_coverage.metric_id.map(lambda value: definition.get(value, {}).get("branch", ""))
    return all_coverage, all_pairs, all_dimensions, dialogue


def state_observability(coverage: pd.DataFrame) -> pd.DataFrame:
    state_rows = coverage.loc[coverage.configured_state.str.match(r"S\d+")].copy()
    grouped = state_rows.groupby(["dataset_id", "language", "configured_state", "status"]).size().unstack(fill_value=0)
    for column in ("variable", "constant", "unobserved"):
        if column not in grouped:
            grouped[column] = 0
    result = grouped.reset_index()
    result["configured_metrics_present"] = result[["variable", "constant", "unobserved"]].sum(axis=1)
    result["observability_score"] = result.variable / result.configured_metrics_present.clip(lower=1)
    result["observability_status"] = np.select(
        [result.variable.eq(0), result.variable.lt(result.configured_metrics_present)],
        ["not_observable", "partially_observable"],
        default="observable",
    )
    return result


def branch_observability(coverage: pd.DataFrame) -> pd.DataFrame:
    return (
        coverage.groupby(["dataset_id", "language", "configured_branch", "status"])
        .size().unstack(fill_value=0).reset_index()
    )


def _plot_state_coverage(table: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(table))
    width = .36
    fig, axis = plt.subplots(figsize=(13, 5.6))
    axis.bar(x - width / 2, table.historical_unique_metrics, width, label="Historical candidates", color="#b8c4c0")
    axis.bar(x + width / 2, table.current_configured_metrics, width, label="Current configured metrics", color="#2b7a78")
    for index, count in enumerate(table.unresolved_core_rows):
        if count:
            axis.text(index + width / 2, table.current_configured_metrics.iloc[index] + .5, f"gap {count}", ha="center", fontsize=9, color="#a23b3b")
    axis.set_xticks(x, table.state_id)
    axis.set_ylabel("Metric count")
    axis.set_title("Historical candidate breadth versus current governed state coverage", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_observability(table: pd.DataFrame, path: Path) -> None:
    table = table.copy()
    table["dataset_language"] = table.dataset_id + " / " + table.language
    matrix = table.pivot_table(index="dataset_language", columns="configured_state", values="observability_score", aggfunc="mean")
    matrix = matrix.reindex(columns=[f"S{i:02d}" for i in range(1, 15)])
    fig, axis = plt.subplots(figsize=(14, max(4, .7 * len(matrix))))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            axis.text(column, row, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if pd.notna(value) and value > .58 else "#263238")
    axis.set_title("State observability by dataset and language (1 = variable evidence available)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=.025, pad=.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_effective_dimension(table: pd.DataFrame, path: Path) -> None:
    overall = table.loc[table.metric_subset.eq("all_variable")].copy()
    labels = overall.dataset_id + " / " + overall.language
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.scatter(overall.n_variable_metrics, overall.effective_rank, s=90, color="#286f6c")
    for x, y, label in zip(overall.n_variable_metrics, overall.effective_rank, labels, strict=True):
        axis.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axis.plot([0, 40], [0, 40], linestyle="--", color="#a7b0b5", linewidth=1)
    axis.set_xlim(0, max(40, overall.n_variable_metrics.max() + 3))
    axis.set_ylim(0, max(20, overall.effective_rank.max() + 3))
    axis.set_xlabel("Variable governed metrics")
    axis.set_ylabel("Effective rank")
    axis.set_title("Observed information dimension is smaller than the metric count", loc="left", fontweight="bold")
    axis.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _oral_report(path: Path) -> None:
    text = """# 中文口语汇报

这次我们没有继续增加模型，也没有使用 Agent 的判断结果。我们先回答一个更基础的问题：当前证据体系本身是否完整、是否重复，以及它在不同数据通道中是否真的能够被观察到。

第一部分是声学证据。当前停顿、发声连续性和部分谱特征确实能够从现有音频中稳定计算，但它们并不是彼此独立的。例如静音比例和发声比例来自同一个 VAD 掩码，两者相加接近一；平均停顿长度和停顿九十分位数在多个数据库中也高度相关。因此，后续不应该按指标行数累计证据，而要先形成信息家族，再由家族产生一次贡献。

第二部分是语言证据。英语图片描述中的十项语言指标基本都可观察；普通话和西班牙语中，通用词汇指标可以计算，但图片任务指标只有在任务类型和评分词表匹配时才成立。当前系统已经使用分语言词表，但名词比例、MTLD、从句复杂度和依存距离没有完整继承。MTLD可以直接补充；名词比例和句法指标则需要经过各语言的词性和句法分析质量审查，不能把英文分析器直接复制到其他语言。

第三部分是对话证据。现在只有带可靠说话人角色和轮次标记的数据，才能真正支持患者轮次占比、修正行为和回答启动延迟。审计发现，patient_turn_share 在部分非对话数据中是常数，这说明“列存在”并不等于“证据可用”。回答启动延迟也不能从普通整段音频推断，必须知道访谈者提示结束和患者开始作答的时间。因此，对话状态必须由可观察性门控，不能在所有数据库中默认启用。

第四部分是认知状态覆盖。历史体系中有一百七十九个名称，但大量项目是同一家族的低层声学统计。真正值得恢复的不是全部旧指标，而是七类遗失构念：非发声段变异、回答启动延迟、名词比例、MTLD、从句复杂度、依存距离和重复率。此外，语义连贯性和多任务表现仍然需要单独的任务参考和验证标准。

这次分析得到的核心结论是：证据体系不能用一个固定数字来定义。正确结构应当是一个较完整的候选库，加上任务、语言、角色、质量和算法可靠性的可观察性门控。每次病例只激活当前数据真正支持的证据；同源指标先合并，缺少前提条件的指标保持不可用，而不是补零或者由大模型猜测。

停顿时长变异、MTLD 和重复率已经实现为候选提取量，但尚未进入疾病风险或医生报告。回答启动延迟需要先核查带时间戳的访谈数据；名词比例与句法复杂度需要建立多语言分析器审计；语义连贯性和任务表现要按具体任务建立评分器。下一轮重提取后，再重新运行数量—性能、外部验证和证据回放实验，决定哪些候选正式进入运行注册表。
"""
    path.write_text(text, encoding="utf-8")


def _html(path: Path, coverage: pd.DataFrame, state_table: pd.DataFrame, dialogue: pd.DataFrame, remediation: pd.DataFrame, pairs: pd.DataFrame) -> None:
    dialogue_summary = dialogue.groupby(["dataset_id", "metric_id", "status"]).size().reset_index(name="blocks")
    dialogue_rows = "".join(f"<tr><td>{r.dataset_id}</td><td>{r.metric_id}</td><td>{r.status}</td></tr>" for r in dialogue_summary.itertuples())
    remediation_rows = "".join(f"<tr><td>{r.state_id}</td><td>{r.construct}</td><td>{r.readiness}</td><td>{r.data_requirement}</td><td>{r.validation}</td></tr>" for r in remediation.itertuples())
    exact = pairs.loc[pairs.equal_on_observed_rows.fillna(False) | pairs.sum_one_on_observed_rows.fillna(False)]
    html = f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Evidence domain audit</title><style>
body{{font-family:Arial,'Noto Sans SC',sans-serif;margin:0;background:#f4f6f7;color:#243238}}main{{max-width:1180px;margin:auto;background:white;padding:44px}}h1{{font-size:34px}}h2{{margin-top:42px;padding-top:24px;border-top:1px solid #d9e0e2}}.lead{{font-size:18px;line-height:1.7;max-width:900px}}.note{{border-left:5px solid #2b7774;background:#edf7f5;padding:16px}}img{{width:100%;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dfe4e7;text-align:left;vertical-align:top}}th{{background:#eef2f3}}code{{background:#eef2f3;padding:2px 5px}}</style></head><body><main>
<h1>Evidence Governance：非 Agent 证据结构审计</h1><p class='lead'>本报告只检查音频、文本、角色、任务和当前状态配置，不读取 Agent 的风险判断。目的不是再定义一套概念，而是验证证据是否可观察、是否重复、缺失构念能否实现。</p>
<p class='note'>分析覆盖 PREPARE、NCMMSC 长音频、IAEAV 和 ADReSS 的训练分区。列存在不等于证据有效；常数列、任务不匹配和缺失角色信息均被单独标记。</p>
<h2>一、14 个状态的覆盖差距</h2><img src='state_metric_coverage.png'><p>S05/S06 的历史数量主要由成组低层声学统计造成；S09 没有当前正式指标。红色 gap 标记的是历史核心证据没有确认运行时继承者的状态。</p>
<h2>二、不同数据通道实际能观察到什么</h2><img src='state_observability_heatmap.png'><p>1 表示该状态的当前配置指标均有变化；0 表示只有常数或不可观察值。该图证明统一启用 37 项不成立，证据需要任务和语言门控。</p>
<h2>三、指标数量不等于独立信息量</h2><img src='effective_dimension.png'><p>各数据块中约 28--36 个可变指标通常只有约 8--13 的有效秩。当前共发现 {len(pairs)} 个 |Spearman ρ|≥0.90 的复核候选，其中 {len(exact)} 个属于观察到的精确相等或互补关系。</p>
<h2>四、对话与沟通分析</h2><p>只有角色和轮次真正变化的数据才能支持互动状态。以下表格显示当前对话相关指标在各数据集中的状态。</p><table><thead><tr><th>数据集</th><th>指标</th><th>状态</th></tr></thead><tbody>{dialogue_rows}</tbody></table>
<h2>五、遗漏构念如何恢复</h2><table><thead><tr><th>状态</th><th>构念</th><th>准备度</th><th>数据要求</th><th>进入注册表前的验证</th></tr></thead><tbody>{remediation_rows}</tbody></table>
<h2>输出</h2><p><code>state_construct_coverage.csv</code>、<code>state_dataset_observability.csv</code>、<code>branch_dataset_observability.csv</code>、<code>dialogue_observability.csv</code>、<code>high_overlap_pairs.csv</code> 和 <code>missing_construct_remediation.csv</code> 保留全部审计细节。</p>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    recipe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    output = root / recipe["output_dir"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    metrics = _load_yaml(root / recipe["metrics_config"], "metrics")
    states = _load_yaml(root / recipe["states_config"], "states")
    history = pd.read_csv(root / recipe["historical_dictionary"])
    mapping = pd.read_csv(root / recipe["mapping_table"])
    source = root / recipe["feature_structure_output"]
    dataset_ids = list(recipe["datasets"])

    state_table = state_coverage(history, mapping, states, metrics)
    coverage, pairs, dimensions, dialogue = collect_observability(source, dataset_ids, metrics)
    state_obs = state_observability(coverage)
    branch_obs = branch_observability(coverage)
    remediation = pd.DataFrame(REMEDIATION)

    state_table.to_csv(output / "state_construct_coverage.csv", index=False)
    state_obs.to_csv(output / "state_dataset_observability.csv", index=False)
    branch_obs.to_csv(output / "branch_dataset_observability.csv", index=False)
    dialogue.to_csv(output / "dialogue_observability.csv", index=False)
    pairs.to_csv(output / "high_overlap_pairs.csv", index=False)
    dimensions.to_csv(output / "effective_dimension_summary.csv", index=False)
    remediation.to_csv(output / "missing_construct_remediation.csv", index=False)

    _plot_state_coverage(state_table, output / "state_metric_coverage.png")
    _plot_observability(state_obs, output / "state_observability_heatmap.png")
    _plot_effective_dimension(dimensions, output / "effective_dimension.png")
    _oral_report(output / "evidence_domain_audit_oral_presentation_zh.md")
    _html(output / "evidence_domain_audit_report_zh.html", coverage, state_table, dialogue, remediation, pairs)

    summary = {
        "datasets": dataset_ids,
        "agent_outputs_used": False,
        "high_overlap_review_pairs": len(pairs),
        "exact_or_complement_pairs": int((pairs.equal_on_observed_rows.fillna(False) | pairs.sum_one_on_observed_rows.fillna(False)).sum()),
        "states_with_unresolved_core_rows": state_table.loc[state_table.unresolved_core_rows.gt(0), "state_id"].tolist(),
        "implemented_candidate_constructs": remediation.loc[remediation.readiness.eq("candidate_implemented_validation_pending"), "construct"].tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
