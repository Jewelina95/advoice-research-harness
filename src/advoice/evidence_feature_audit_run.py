"""Run the historical-to-runtime evidence mapping and feature-count audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .evidence_feature_audit import (
    candidate_metric_ids, choose_feature_counts, feature_inference,
    map_historical_metrics, stratified_bootstrap_indices,
)


def _resolve(config: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config.parent / path).resolve()


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _pipeline(k: int, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0,
            solver="lbfgs", random_state=seed,
        )),
    ])


def _auc(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _scoring(n_classes: int) -> dict[str, str]:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "auroc": "roc_auc" if n_classes == 2 else "roc_auc_ovr",
    }


def _valid_features(train: pd.DataFrame, candidates: list[str]) -> list[str]:
    output = []
    for column in candidates:
        if column not in train:
            continue
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() >= 5 and values.nunique(dropna=True) >= 2:
            output.append(column)
    return output


def _feature_curve(
    frame: pd.DataFrame, candidates: list[str], seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    split = frame.split.astype(str).str.strip().str.lower()
    train, test = frame.loc[split.eq("train")].copy(), frame.loc[split.eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("Both explicit train and test rows are required")
    features = _valid_features(train, candidates)
    if not features:
        raise ValueError("No variable governed metrics in training data")
    encoder = LabelEncoder().fit(train.label.astype(str))
    unseen = set(test.label.astype(str)) - set(encoder.classes_)
    if unseen:
        raise ValueError(f"Test labels absent from training: {sorted(unseen)}")
    y_train = encoder.transform(train.label.astype(str))
    y_test = encoder.transform(test.label.astype(str))
    min_class = int(pd.Series(y_train).value_counts().min())
    folds = min(5, min_class)
    if folds < 2:
        raise ValueError("Insufficient training examples per class")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    x_train = train[features].apply(pd.to_numeric, errors="coerce")
    x_test = test[features].apply(pd.to_numeric, errors="coerce")
    rows = []
    ranking_rows = []
    for k in choose_feature_counts(len(features)):
        estimator = _pipeline(k, seed)
        scores = cross_validate(
            estimator, x_train, y_train, cv=cv, scoring=_scoring(len(encoder.classes_)),
            return_train_score=False, error_score=np.nan,
        )
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        probabilities = estimator.predict_proba(x_test)
        selected = np.asarray(features)[estimator.named_steps["select"].get_support()]
        for metric_id in selected:
            ranking_rows.append({"feature_count": k, "metric_id": metric_id})
        rows.append({
            "feature_count": k,
            "n_available_features": len(features),
            "n_train": len(train),
            "n_test": len(test),
            "n_classes": len(encoder.classes_),
            "cv_folds": folds,
            "cv_accuracy_mean": float(np.nanmean(scores["test_accuracy"])),
            "cv_accuracy_sd": float(np.nanstd(scores["test_accuracy"], ddof=1)),
            "cv_balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
            "cv_balanced_accuracy_sd": float(np.nanstd(scores["test_balanced_accuracy"], ddof=1)),
            "cv_macro_f1_mean": float(np.nanmean(scores["test_macro_f1"])),
            "cv_macro_f1_sd": float(np.nanstd(scores["test_macro_f1"], ddof=1)),
            "cv_auroc_mean": float(np.nanmean(scores["test_auroc"])),
            "cv_auroc_sd": float(np.nanstd(scores["test_auroc"], ddof=1)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "test_macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "test_auroc": _auc(y_test, probabilities, estimator.named_steps["model"].classes_),
            "selected_metrics": ";".join(selected),
            "selection_scope": "anova_f_fit_on_training_partition_only",
            "test_scope": "locked_existing_split_not_used_for_feature_ranking",
        })

    full = SimpleImputer(strategy="median").fit_transform(x_train)
    scores, _ = f_classif(full, y_train)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)[::-1]
    full_ranking = pd.DataFrame({
        "rank": np.arange(1, len(features) + 1),
        "metric_id": np.asarray(features)[order],
        "anova_f_train": scores[order],
    })
    return pd.DataFrame(rows), pd.DataFrame(ranking_rows), full_ranking, features


def _bootstrap_stability(
    frame: pd.DataFrame, features: list[str], *, seed: int, repeats: int, top_k: int = 8,
) -> pd.DataFrame:
    train = frame.loc[frame.split.astype(str).str.lower().eq("train")].copy()
    encoder = LabelEncoder().fit(train.label.astype(str))
    labels = encoder.transform(train.label.astype(str))
    values = train[features].apply(pd.to_numeric, errors="coerce")
    rng = np.random.default_rng(seed)
    counts = Counter()
    score_sum = Counter()
    score_square = Counter()
    successful = 0
    for _ in range(repeats):
        indices = stratified_bootstrap_indices(labels, rng)
        sample = values.iloc[indices]
        sample_labels = labels[indices]
        transformed = SimpleImputer(strategy="median").fit_transform(sample)
        if transformed.shape[1] != len(features):
            continue
        scores, _ = f_classif(transformed, sample_labels)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        successful += 1
        for index, score in enumerate(scores):
            score_sum[features[index]] += float(score)
            score_square[features[index]] += float(score * score)
        for index in np.argsort(scores)[::-1][: min(top_k, len(features))]:
            counts[features[index]] += 1
    rows = []
    for metric_id in features:
        mean = score_sum[metric_id] / max(successful, 1)
        variance = score_square[metric_id] / max(successful, 1) - mean * mean
        rows.append({
            "metric_id": metric_id,
            "top_k": min(top_k, len(features)),
            "selection_frequency": counts[metric_id] / max(successful, 1),
            "anova_f_mean": mean,
            "anova_f_sd": float(np.sqrt(max(variance, 0.0))),
            "bootstrap_repeats_requested": repeats,
            "bootstrap_repeats_successful": successful,
            "scope": "training_partition_ranking_stability_not_clinical_validity",
        })
    return pd.DataFrame(rows).sort_values(["selection_frequency", "anova_f_mean"], ascending=False)


def _learning_curve(
    frame: pd.DataFrame, features: list[str], *, seed: int,
) -> pd.DataFrame:
    train = frame.loc[frame.split.astype(str).str.lower().eq("train")].copy()
    encoder = LabelEncoder().fit(train.label.astype(str))
    labels = encoder.transform(train.label.astype(str))
    min_class = int(pd.Series(labels).value_counts().min())
    folds = min(5, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    estimator = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed)),
    ])
    sizes, train_scores, validation_scores = learning_curve(
        estimator,
        train[features].apply(pd.to_numeric, errors="coerce"),
        labels,
        train_sizes=np.asarray([0.5, 0.7, 0.85, 1.0]),
        cv=cv,
        scoring="balanced_accuracy",
        shuffle=True,
        random_state=seed,
        error_score=np.nan,
    )
    return pd.DataFrame({
        "training_rows": sizes,
        "train_balanced_accuracy_mean": np.nanmean(train_scores, axis=1),
        "train_balanced_accuracy_sd": np.nanstd(train_scores, axis=1, ddof=1),
        "validation_balanced_accuracy_mean": np.nanmean(validation_scores, axis=1),
        "validation_balanced_accuracy_sd": np.nanstd(validation_scores, axis=1, ddof=1),
        "scope": "cross_validated_training_learning_curve",
    })


def _plot_feature_curves(curves: pd.DataFrame, path: Path) -> None:
    datasets = list(curves.dataset_id.unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(4, 3.4 * len(datasets))), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets, strict=True):
        part = curves.loc[curves.dataset_id.eq(dataset)].sort_values("feature_count")
        axis.plot(part.feature_count, part.cv_balanced_accuracy_mean, marker="o", label="Training CV balanced accuracy")
        axis.fill_between(
            part.feature_count,
            part.cv_balanced_accuracy_mean - part.cv_balanced_accuracy_sd,
            part.cv_balanced_accuracy_mean + part.cv_balanced_accuracy_sd,
            alpha=.16,
        )
        axis.plot(part.feature_count, part.test_balanced_accuracy, marker="s", label="Locked test balanced accuracy")
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Number of governed non-QC metrics")
        axis.set_ylabel("Balanced accuracy")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=.2)
        axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_learning_curves(curves: pd.DataFrame, path: Path) -> None:
    datasets = list(curves.dataset_id.unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(4, 3.4 * len(datasets))), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets, strict=True):
        part = curves.loc[curves.dataset_id.eq(dataset)].sort_values("training_rows")
        axis.errorbar(part.training_rows, part.train_balanced_accuracy_mean, yerr=part.train_balanced_accuracy_sd, marker="o", label="Train")
        axis.errorbar(part.training_rows, part.validation_balanced_accuracy_mean, yerr=part.validation_balanced_accuracy_sd, marker="s", label="Validation")
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Training rows used inside cross-validation")
        axis.set_ylabel("Balanced accuracy")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _html_report(output: Path, summary: dict[str, Any]) -> None:
    dataset_rows = "".join(
        f"<tr><td>{item['dataset_id']}</td><td>{item['n_train']}</td><td>{item['n_test']}</td>"
        f"<td>{item['n_available_features']}</td><td>{item['best_cv_feature_count']}</td>"
        f"<td>{item['best_cv_balanced_accuracy']:.3f}</td><td>{item['locked_test_balanced_accuracy_at_cv_choice']:.3f}</td></tr>"
        for item in summary["datasets"]
    )
    html = f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Evidence feature audit</title>
<style>body{{font-family:Arial,'Noto Sans SC',sans-serif;color:#1f2933;background:#f7f8fa;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:42px}}h1{{font-size:32px}}h2{{margin-top:38px;border-top:1px solid #dce2e8;padding-top:24px}}.note{{border-left:5px solid #2b6f6d;background:#edf7f5;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #d8dee4;padding:10px;text-align:left}}th{{background:#eef2f4}}img{{width:100%;margin-top:14px}}code{{background:#eef2f4;padding:2px 5px}}</style></head><body><main>
<h1>Evidence Governance 指标映射与数量校准</h1>
<p class='note'>本报告把历史 180 行（179 个唯一名称）与当前 37 个运行指标分开处理。映射关系表示来源或构念关系，不自动证明数值等价或临床有效。</p>
<h2>映射结果</h2><p>当前注册表共 37 项，其中 {summary['runtime_mapped']} 项存在经规则审查的历史前身，{summary['runtime_new']} 项为当前管线新增。历史行中 {summary['historical_mapped']} 行已映射，未映射行按未提取、未实现、辅助指标、方向待审查或遗留核心待人工确认分别保留。</p>
<p>完整结果：<code>historical_to_runtime_mapping.csv</code>、<code>runtime_metric_lineage.csv</code>、<code>mapping_summary.csv</code>。</p>
<h2>特征数量与性能</h2><table><thead><tr><th>Dataset</th><th>Train</th><th>Test</th><th>可用非 QC 指标</th><th>CV 选择数量</th><th>CV balanced accuracy</th><th>锁定测试集</th></tr></thead><tbody>{dataset_rows}</tbody></table>
<img src='feature_count_performance.png' alt='Feature count performance curves'>
<h2>抽样与样本充分性</h2><p>学习曲线只在训练分区内部交叉验证。训练与验证曲线差距大表示方差或样本不足，不能通过增加指标数量解决。</p>
<img src='sample_size_learning_curves.png' alt='Sample size learning curves'>
<h2>解释边界</h2><p>当前性能分析只覆盖仓库中已有 subject-level 特征表的数据集。未提取的历史指标只能完成设计映射，不能获得虚构的性能分数。测试集从未参与指标排序；但该报告仍是探索性证据，不替代跨数据库外部验证。</p>
</main></body></html>"""
    (output / "evidence_feature_audit_report_zh.html").write_text(html, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    recipe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = _resolve(config_path, recipe["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    metrics_path = _resolve(config_path, recipe["metrics_config"])
    history_path = _resolve(config_path, recipe["historical_dictionary"])
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))["metrics"]
    history = pd.read_csv(history_path)
    mapping, lineage, mapping_summary = map_historical_metrics(history, metrics)
    mapping.to_csv(output / "historical_to_runtime_mapping.csv", index=False)
    lineage.to_csv(output / "runtime_metric_lineage.csv", index=False)
    mapping_summary.to_csv(output / "mapping_summary.csv", index=False)

    seed = int(recipe.get("random_seed", 20260917))
    repeats = int(recipe.get("bootstrap_repeats", 100))
    candidates = candidate_metric_ids(metrics)
    all_curves, all_learning, datasets_summary = [], [], []
    inputs = [_fingerprint(config_path), _fingerprint(metrics_path), _fingerprint(history_path)]
    for item in recipe["datasets"]:
        dataset_id = item["id"]
        path = _resolve(config_path, item["features_path"])
        inputs.append(_fingerprint(path))
        frame = pd.read_csv(path, dtype={"subject_id": "string"})
        dataset_dir = output / dataset_id
        dataset_dir.mkdir()
        inference = feature_inference(frame, [metric["id"] for metric in metrics], kruskal_fn=kruskal)
        inference.to_csv(dataset_dir / "train_only_statistical_inference.csv", index=False)
        curves, selections, ranking, features = _feature_curve(frame, candidates, seed)
        curves.insert(0, "dataset_id", dataset_id)
        curves.to_csv(dataset_dir / "feature_count_performance.csv", index=False)
        selections.to_csv(dataset_dir / "selected_metrics_by_count.csv", index=False)
        ranking.to_csv(dataset_dir / "full_train_feature_ranking.csv", index=False)
        stability = _bootstrap_stability(frame, features, seed=seed, repeats=repeats)
        stability.to_csv(dataset_dir / "bootstrap_feature_stability.csv", index=False)
        learning = _learning_curve(frame, features, seed=seed)
        learning.insert(0, "dataset_id", dataset_id)
        learning.to_csv(dataset_dir / "sample_size_learning_curve.csv", index=False)
        all_curves.append(curves)
        all_learning.append(learning)
        best = curves.sort_values(
            ["cv_balanced_accuracy_mean", "feature_count"], ascending=[False, True]
        ).iloc[0]
        datasets_summary.append({
            "dataset_id": dataset_id,
            "n_train": int(best.n_train),
            "n_test": int(best.n_test),
            "n_available_features": int(best.n_available_features),
            "best_cv_feature_count": int(best.feature_count),
            "best_cv_balanced_accuracy": float(best.cv_balanced_accuracy_mean),
            "locked_test_balanced_accuracy_at_cv_choice": float(best.test_balanced_accuracy),
            "locked_test_auroc_at_cv_choice": float(best.test_auroc),
        })
    combined_curves = pd.concat(all_curves, ignore_index=True)
    combined_learning = pd.concat(all_learning, ignore_index=True)
    combined_curves.to_csv(output / "all_dataset_feature_count_performance.csv", index=False)
    combined_learning.to_csv(output / "all_dataset_sample_size_learning_curves.csv", index=False)
    pd.DataFrame(datasets_summary).to_csv(output / "dataset_summary.csv", index=False)
    _plot_feature_curves(combined_curves, output / "feature_count_performance.png")
    _plot_learning_curves(combined_learning, output / "sample_size_learning_curves.png")

    summary = {
        "runtime_metrics": len(metrics),
        "runtime_mapped": int(lineage.provenance_status.eq("mapped_predecessor").sum()),
        "runtime_new": int(lineage.provenance_status.eq("new_runtime_metric").sum()),
        "historical_rows": len(mapping),
        "historical_unique_names": int(mapping.historical_metric.nunique()),
        "historical_mapped": int(mapping.disposition.eq("mapped_to_runtime_registry").sum()),
        "datasets": datasets_summary,
        "random_seed": seed,
        "bootstrap_repeats": repeats,
        "inputs": inputs,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        "python": platform.python_version(),
        "scope": "research_only_no_production_registry_changes",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _html_report(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
