"""Inventory existing experiments without relabelling them as new runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from advoice.config import paths, load_yaml


def inspect_dataset(directory: Path) -> dict:
    source = directory / "subject_features.csv"
    result = {"dataset": directory.name, "source": str(directory), "available": source.is_file()}
    if not source.is_file():
        return result
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits: dict[str, set[str]] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        splits.setdefault(row.get("split", "missing"), set()).add(row.get("subject_id", ""))
        label = row.get("label", "missing")
        label_counts[label] = label_counts.get(label, 0) + 1
    overlap = set()
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap |= splits[left] & splits[right]
    ids = [row.get("subject_id", "") for row in rows]
    result.update({
        "rows": len(rows), "subjects": len(set(ids)),
        "splits": {key: len(value) for key, value in splits.items()},
        "label_counts": label_counts, "cross_split_subject_overlap": len(overlap),
        "duplicate_subject_rows": len(ids) - len(set(ids)),
        "missing_subject_ids": sum(not value.strip() for value in ids),
        "feature_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prediction_available": (directory / "ours_predictions.csv").is_file(),
        "evaluation_available": (directory / "layer_a_metrics.csv").is_file(),
        "required_missing": [name for name in (
            "manifest.csv", "recording_features.csv", "segments.csv",
            "subject_transcripts.csv", "metric_evidence.csv",
            "b1_predictions.csv", "b2_predictions.csv",
        ) if not (directory / name).is_file()],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = load_yaml(paths().configs / "datasets/registry.yaml")["datasets"]
    ids = [item["id"] for item in registry if item.get("active")]
    records = [inspect_dataset(root / dataset) for root in args.source for dataset in ids]
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "existing_artifact_inventory_not_retraining", "registered_tasks": ids,
               "records": records}
    (args.output / "inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    body = []
    for row in records:
        if not row["available"]:
            continue
        issues = []
        for key in ("cross_split_subject_overlap", "duplicate_subject_rows", "missing_subject_ids"):
            if row[key]:
                issues.append(f"{key}: {row[key]}")
        issues.extend(row["required_missing"])
        values = [row["dataset"], str(Path(row["source"]).parent), str(row["subjects"]),
                  json.dumps(row["splits"]), "; ".join(issues) or "No identity/schema issue detected"]
        body.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice experiment inventory</title><style>
body{font:16px/1.6 system-ui;color:#242a2d;margin:40px auto;max-width:1250px;padding:0 20px}
h1{font-size:28px}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:12px;text-align:left;border-bottom:1px solid #ccd3d5;overflow-wrap:anywhere}
th{background:#eaf2f3}td:nth-child(2){max-width:350px} .table{overflow:auto}
</style><h1>ADvoice experiment inventory</h1>
<p>Existing files only. This inventory does not retrain a model or establish clinical validity.
Different source directories are separate historical runs. Absence of subject overlap does not
exclude recording, family, site, or preprocessing leakage.</p><div class="table"><table>
<thead><tr><th>Dataset / task</th><th>Artifact source</th><th>Subjects</th><th>Splits</th><th>Input checks</th></tr></thead><tbody>"""
    (args.output / "inventory.html").write_text(document + "".join(body) + "</tbody></table></div></html>", encoding="utf-8")
    print(args.output / "inventory.html")


if __name__ == "__main__":
    main()
