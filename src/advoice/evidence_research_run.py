"""Run a private, train-only descriptive evidence-overlap study.

Usage: python -m advoice.evidence_research_run --config local-recipe.yaml
No training, feature selection, inference, or patient-level export is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .evidence_research import (
    effective_dimension_statistics, historical_declared_reuse,
    historical_inventory, metric_column, overlap_components,
    overlap_statistics, state_overlap_tables, task_scope_reuse_statistics,
)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def load_subject_features(path: Path, metric_ids: set[str], metadata_path: Path | None) -> pd.DataFrame:
    keep = {"subject_id", "split", "language", "dataset_id", "task_type"}
    frame = pd.read_csv(
        path, dtype={"subject_id": "string"},
        usecols=lambda c: c in keep or metric_column(c, metric_ids) is not None,
    )
    if metadata_path:
        metadata = pd.read_csv(metadata_path, dtype={"subject_id": "string"}, usecols=lambda c: c in keep)
        if "subject_id" not in metadata or metadata.subject_id.isna().any() or metadata.subject_id.duplicated().any():
            raise ValueError("Metadata must have unique, non-null subject identities")
        if not set(frame.subject_id).issubset(set(metadata.subject_id)):
            raise ValueError("Metadata does not cover all feature subjects")
        joined = frame.merge(metadata, on="subject_id", how="left", validate="one_to_one", suffixes=("", "_metadata"))
        for col in ("split", "language", "dataset_id", "task_type"):
            if f"{col}_metadata" in joined:
                both = joined[col].notna() & joined[f"{col}_metadata"].notna()
                if not joined.loc[both, col].astype(str).str.lower().equals(
                    joined.loc[both, f"{col}_metadata"].astype(str).str.lower()
                ):
                    raise ValueError(f"Feature/metadata {col} disagreement")
                joined[col] = joined[col].fillna(joined[f"{col}_metadata"])
                joined = joined.drop(columns=f"{col}_metadata")
        frame = joined
    return frame


def run_study(config_path: Path) -> dict:
    config_path = config_path.resolve()
    recipe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    output = resolve(recipe["output_dir"])
    if not output.is_relative_to(root / ".local"):
        raise ValueError("Private study outputs must stay under the repository .local directory")
    metrics_path = resolve(recipe["metrics_config"])
    states_path = resolve(recipe["states_config"])
    history_path = resolve(recipe["historical_dictionary"])
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))["metrics"]
    states = yaml.safe_load(states_path.read_text(encoding="utf-8"))["states"]
    ids = {m["id"] for m in metrics}
    datasets = recipe["datasets"]
    names = [item["id"] for item in datasets]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Use unique, filesystem-safe dataset identifiers")
    sources = [config_path, metrics_path, states_path, history_path, Path(__file__), Path(__file__).with_name("evidence_research.py")]
    for item in datasets:
        sources.append(resolve(item["features_path"]))
        if item.get("metadata_path"):
            sources.append(resolve(item["metadata_path"]))
    inputs = [fingerprint(path) for path in dict.fromkeys(sources)]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pyyaml")},
        "inputs": inputs, "recipe": recipe, "clinical_validation": False,
    }
    manifest_path = output / "manifest.json"
    try:
        history = pd.read_csv(history_path)
        inventory = historical_inventory(history, metrics)
        inventory.to_csv(output / "historical_inventory.csv", index=False)
        historical_edges, historical_reuse = historical_declared_reuse(history)
        historical_edges.to_csv(output / "historical_declared_edges.csv", index=False)
        historical_reuse.to_csv(output / "historical_declared_reuse.csv", index=False)
        edges, incidence, reuse = state_overlap_tables(states, ids)
        edges.to_csv(output / "state_metric_edges.csv", index=False)
        incidence.to_csv(output / "state_metric_incidence.csv", index=False)
        reuse.to_csv(output / "state_reuse.csv", index=False)
        summary = {
            "historical_rows": len(history), "historical_unique_names": history.metric_name.nunique(),
            "runtime_metrics": len(metrics), "runtime_states": len(states),
            "exact_historical_name_candidates": int(inventory.runtime_name_candidate.ne("").sum()),
            "name_mapping_is_not_equivalence": True, "datasets": {},
            "reused_historical_declared_columns": len(historical_reuse),
        }
        for item in datasets:
            frame = load_subject_features(resolve(item["features_path"]), ids,
                resolve(item["metadata_path"]) if item.get("metadata_path") else None)
            coverage, pairs, metadata = overlap_statistics(
                frame, metrics, min_pairs=int(recipe.get("min_pairs", 20)),
                threshold=float(recipe.get("review_threshold", .9)),
            )
            components = overlap_components(
                coverage, pairs, threshold=float(recipe.get("review_threshold", .9))
            )
            dimensions = effective_dimension_statistics(
                frame, metrics, min_rows=int(recipe.get("dimension_min_rows", 20))
            )
            scope_reuse = task_scope_reuse_statistics(
                frame, metrics, min_pairs=int(recipe.get("min_pairs", 20))
            )
            folder = output / item["id"]
            folder.mkdir()
            coverage.to_csv(folder / "coverage.csv", index=False)
            pairs.to_csv(folder / "pairwise_overlap.csv", index=False)
            components.to_csv(folder / "overlap_components.csv", index=False)
            dimensions.to_csv(folder / "dimension_summary.csv", index=False)
            scope_reuse.to_csv(folder / "task_scope_reuse.csv", index=False)
            metadata.update({
                "scope_note": item.get("scope_note", "Extraction provenance requires independent review"),
                "variable_metric_blocks": int(coverage.status.eq("variable").sum()),
                "constant_metric_blocks": int(coverage.status.eq("constant").sum()),
                "unobserved_metric_blocks": int(coverage.status.eq("unobserved").sum()),
                "pairs_evaluable": int(pairs.status.eq("ok").sum()) if len(pairs) else 0,
                "high_overlap_review_candidates": int(pairs.review_candidate.sum()) if len(pairs) else 0,
                "affine_relation_candidates": int(pairs.affine_on_observed_rows.sum()) if len(pairs) else 0,
                "multi_metric_overlap_components": int(components.n_members.gt(1).sum()) if len(components) else 0,
                "dimension_blocks_evaluable": int(dimensions.status.eq("ok").sum()) if len(dimensions) else 0,
                "task_scope_exact_copy_candidates": int(scope_reuse.equal_on_observed_rows.sum()) if len(scope_reuse) else 0,
                "dimension_estimates_are_not_clinical_factor_counts": True,
            })
            summary["datasets"][item["id"]] = metadata
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["status"] = "completed_descriptive_analysis_only"
        return summary
    except Exception as exc:
        manifest["status"] = "failed_do_not_interpret_partial_outputs"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_study(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
