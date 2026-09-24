"""Offline rendering for locked pilot aggregate results.

This module is deliberately a presentation boundary.  It reads one immutable
``aggregate.json`` payload and never reads labels, case records, or model
artifacts.  In particular, it does not calculate statistics, fit models, or
resample subjects.
"""
from __future__ import annotations

import json
import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape


ARMS = ("B_raw", "B", "J-A", "J-S", "J-AS")
AGGREGATE_SCHEMA = "advoice.pilot.aggregate.v1"
AGGREGATE_LOCK_SCHEMA = "advoice.pilot.aggregate-lock.v1"
ARM_COLORS = {
    "B_raw": "#7d8790",
    "B": "#5d6770",
    "J-A": "#c97a52",
    "J-S": "#4f8c89",
    "J-AS": "#276f72",
}
ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = ROOT / "templates" / "pilot"


class ReportInputError(ValueError):
    """The locked aggregate is malformed or contains unsafe report data."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Cannot read locked aggregate JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReportInputError("The locked aggregate must be a JSON object.")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def aggregate_lock(payload: Mapping[str, Any]) -> dict[str, str]:
    """Create the content lock that report rendering verifies before publication."""

    unlocked = {key: value for key, value in payload.items() if key != "lock"}
    return {
        "schema_version": AGGREGATE_LOCK_SCHEMA,
        "status": "locked",
        "aggregate_hash": _hash(unlocked),
    }


def _validate_metric_rows(raw: Any, *, name: str) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        raise ReportInputError(f"{name} must be a list of locked metric rows.")
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ReportInputError(f"{name}[{index}] is not a metric object.")
        required = {"dataset_id", "arm", "metric", "value", "n", "denominator"}
        missing = required - set(row)
        if missing:
            raise ReportInputError(f"{name}[{index}] lacks required metric fields: {', '.join(sorted(missing))}.")
        if not isinstance(row["dataset_id"], str) or not isinstance(row["arm"], str) or not isinstance(row["metric"], str):
            raise ReportInputError(f"{name}[{index}] has malformed metric identifiers.")
        if not isinstance(row["n"], int) or not isinstance(row["denominator"], int) or row["n"] < 0 or row["denominator"] < row["n"]:
            raise ReportInputError(f"{name}[{index}] has an invalid denominator.")
        if row["value"] is not None and not _finite(row["value"]):
            raise ReportInputError(f"{name}[{index}] has a non-finite value.")
        if row["value"] is None and not isinstance(row.get("undefined_reason"), str):
            raise ReportInputError(f"{name}[{index}] has an undefined value without a reason.")


def _validate_paired_rows(raw: Any) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        raise ReportInputError("paired must be a list of locked paired rows.")
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ReportInputError(f"paired[{index}] is not an object.")
        required = {"dataset_id", "arm", "baseline_arm", "n", "helped", "harmed", "per_class"}
        if required - set(row):
            raise ReportInputError(f"paired[{index}] lacks the paired scoring schema.")
        if not all(isinstance(row[key], str) and row[key] for key in ("dataset_id", "arm", "baseline_arm")):
            raise ReportInputError(f"paired[{index}] has malformed arm identifiers.")
        if not all(isinstance(row[key], int) and row[key] >= 0 for key in ("n", "helped", "harmed")) or row["helped"] + row["harmed"] > row["n"]:
            raise ReportInputError(f"paired[{index}] has invalid help/harm counts.")
        if not isinstance(row["per_class"], Mapping):
            raise ReportInputError(f"paired[{index}] must include per-class help/harm.")


def _validate_aggregate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != AGGREGATE_SCHEMA:
        raise ReportInputError("Locked aggregate schema_version is missing or unsupported.")
    lock = payload.get("lock")
    if not isinstance(lock, Mapping):
        raise ReportInputError("Locked aggregate has no lock manifest.")
    expected = aggregate_lock(payload)
    for field, value in expected.items():
        if lock.get(field) != value:
            raise ReportInputError(f"Locked aggregate {field} mismatch.")
    status = payload.get("status")
    if status not in {"complete", "incomplete", "blocked", "failed"}:
        raise ReportInputError("Locked aggregate status is missing or invalid.")
    _validate_metric_rows(payload.get("layer_a", payload.get("metrics")), name="layer_a")
    _validate_metric_rows(payload.get("layer_b"), name="layer_b")
    _validate_paired_rows(payload.get("paired"))
    has_metrics = bool(payload.get("layer_a", payload.get("metrics", [])) or payload.get("layer_b", []))
    if not has_metrics and status != "incomplete":
        raise ReportInputError("An aggregate without metric rows must be explicitly incomplete.")


def _aggregate_path(run_dir: Path) -> Path:
    candidates = (run_dir / "aggregate.json", run_dir / "aggregate" / "aggregate.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ReportInputError("No locked aggregate.json found in the run directory.")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _display(value: Any) -> str:
    if value is None or (isinstance(value, str) and value.lower() in {"na", "n/a", "undefined", "null"}):
        return "Undefined"
    if _finite(value):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return str(value)


def _reason(row: Mapping[str, Any]) -> str:
    return str(row.get("undefined_reason") or row.get("reason") or "Not reported")


def _metric_rows(raw: Any, *, section: str) -> list[dict[str, Any]]:
    """Normalize only presentation fields; values remain those in the JSON."""
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row["section"] = section
        row["dataset"] = row.get("dataset") or row.get("dataset_id") or "Unspecified dataset"
        row["arm"] = row.get("arm") or row.get("condition") or ""
        row["metric"] = row.get("metric") or row.get("name") or "Unspecified metric"
        row["value_display"] = _display(row.get("value"))
        row["ci_display"] = (
            f"[{_display(row.get('ci_low'))}, {_display(row.get('ci_high'))}]"
            if row.get("ci_low") is not None and row.get("ci_high") is not None
            else "Undefined"
        )
        row["n_display"] = _display(row.get("n", row.get("denominator")))
        row["denominator_display"] = _display(row.get("denominator", row.get("n")))
        row["undefined_display"] = _reason(row)
        row["is_stress"] = str(row.get("partition") or row.get("split") or "").lower() == "stress"
        rows.append(row)
    return rows


def _sections(payload: Mapping[str, Any], key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = payload.get(key, payload.get("metrics", []) if key == "layer_a" else [])
    if isinstance(raw, Mapping):
        raw = raw.get("metrics", raw.get("rows", []))
    rows = _metric_rows(raw, section=key)
    return ([row for row in rows if not row["is_stress"]], [row for row in rows if row["is_stress"]])


def _failures(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("failures", payload.get("failure_modes", []))
    if isinstance(raw, Mapping):
        raw = [{"name": key, "detail": value} for key, value in raw.items()]
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, Mapping):
            result.append({"name": item.get("name") or item.get("code") or "Failure", "detail": item.get("detail") or item.get("reason") or "Unspecified"})
        else:
            result.append({"name": "Failure", "detail": str(item)})
    return result


def _trace(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    value = payload.get("trace_example") or payload.get("trace")
    if not isinstance(value, Mapping):
        return None
    # Keep this projection strictly de-identified and free of transcript text.
    allowed = ("case_id", "snapshot_hash", "state_version", "reference_fit_id", "evidence_ids", "state_ids", "source_segment_ids", "trace_ids", "status")
    return {key: value[key] for key in allowed if key in value}


def _plot(rows: list[dict[str, Any]], path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4.0, min(10.0, 2.2 + len(rows) * 0.34))), constrained_layout=True)
    valid = [row for row in rows if _finite(row.get("value"))]
    if not valid:
        ax.axis("off")
        ax.text(0.5, 0.5, "No defined aggregate values", ha="center", va="center", fontsize=14, color="#5e6a70")
    else:
        labels = [f"{row['dataset']} | {row['arm']} | {row['metric']}" for row in valid]
        values = [float(row["value"]) for row in valid]
        colors = [ARM_COLORS.get(str(row.get("arm")), "#276f72") for row in valid]
        y = list(range(len(valid)))
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        for index, value in enumerate(values):
            ax.text(value, index, f" {value:.3f}", va="center", ha="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=select_autoescape(["html", "xml"]))
    env.filters["display"] = _display
    return env


def render_run(run_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Render both fixed pilot reports from one locked aggregate JSON."""
    run_path, out = Path(run_dir), Path(output_dir)
    aggregate_path = _aggregate_path(run_path)
    payload = _json(aggregate_path)
    _validate_aggregate(payload)
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)

    layer_a, layer_a_stress = _sections(payload, "layer_a")
    layer_b, layer_b_stress = _sections(payload, "layer_b")
    failures = _failures(payload)
    paired = [dict(row) for row in payload.get("paired", [])]
    plot_a = assets / "layer_a_summary.png"
    plot_b = assets / "layer_b_summary.png"
    _plot(layer_a, plot_a, "Layer A | Locked prediction metrics", "Aggregate value")
    _plot(layer_b, plot_b, "Layer B | Technical proxy metrics", "Aggregate value")
    context = {
        "run": payload.get("run", payload.get("metadata", {})),
        "metadata": payload.get("metadata", {}),
        "datasets": payload.get("datasets", payload.get("dataset_inputs", [])),
        "pipeline": payload.get("pipeline", payload.get("system", {})),
        "versions": payload.get("versions", {}),
        "limits": payload.get("limits", payload.get("unresolved_limits", [])),
        "status": payload.get("status", "incomplete" if failures else "complete"),
        "failures": failures,
        "trace": _trace(payload),
        "layer_a": layer_a,
        "layer_a_stress": layer_a_stress,
        "layer_b": layer_b,
        "layer_b_stress": layer_b_stress,
        "paired": paired,
        "arms": list(dict.fromkeys((*ARMS, *(str(row.get("arm")) for row in layer_a), *(str(row.get("baseline_arm")) for row in paired)))),
        "source_aggregate": aggregate_path.name,
        "aggregate_hash": payload["lock"]["aggregate_hash"],
        "assets": "assets",
    }
    env = _environment()
    system_path = out / "system_report.html"
    evaluation_path = out / "evaluation_report.html"
    system_path.write_text(env.get_template("system_report.html.j2").render(**context), encoding="utf-8")
    evaluation_path.write_text(env.get_template("evaluation_report.html.j2").render(**context), encoding="utf-8")
    return {"system_report": system_path, "evaluation_report": evaluation_path, "layer_a_figure": plot_a, "layer_b_figure": plot_b}


__all__ = ["AGGREGATE_LOCK_SCHEMA", "AGGREGATE_SCHEMA", "ReportInputError", "aggregate_lock", "render_run"]
