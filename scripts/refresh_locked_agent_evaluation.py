from __future__ import annotations

import argparse
from pathlib import Path

from advoice.cognitive_agent import run_cognitive_diagnostic_agent
from advoice.config import load_all, paths
from advoice.diagnostic_agent_report import run_diagnostic_agent_reports
from advoice.models import train_negative_controls
from advoice.pipeline import (
    _artifact_paths,
    _resolved_configs,
    reevaluate_dataset,
)
from advoice.aggregate_reporting import build_aggregate_report
from advoice.report_scoring_agent import run_report_scoring_agent


def refresh_dataset(dataset_id: str, provider: str) -> Path:
    project_paths = paths()
    configs = _resolved_configs(dataset_id)
    artifact = project_paths.artifacts / dataset_id
    output = _artifact_paths(artifact)

    run_cognitive_diagnostic_agent(
        project_paths.root,
        output["b3_supervised_predictions"],
        output["diagnostic_agent_workspaces"],
        configs["agents"],
        provider,
        output["ours_predictions"],
        output["cognitive_agent_decisions"],
        output["cognitive_agent_audit"],
        output["locked_agent_workspaces"],
        output["cognitive_agent_status"],
        output["cognitive_agent_prompt"],
        output["agent_calibration_predictions"],
        output["agent_calibration_workspaces"],
        output["agent_correction_calibration"],
    )
    train_negative_controls(
        output["subject_features"],
        output["negative_control_predictions"],
        configs["models"],
    )
    run_diagnostic_agent_reports(
        project_paths.root,
        output["ours_predictions"],
        output["locked_agent_workspaces"],
        configs["agents"],
        provider,
        output["ours_reports"],
        output["ours_report_status"],
        output["ours_report_prompt"],
    )
    scoring_provider = (
        str(configs["agents"].get("report_scoring_provider", "disabled"))
        if provider != "disabled"
        else "disabled"
    )
    run_report_scoring_agent(
        project_paths.root,
        output["b2_reports"],
        output["ours_reports"],
        configs["agents"],
        scoring_provider,
        output["report_scores"],
        output["report_scoring_status"],
        output["report_scoring_prompt"],
    )
    return reevaluate_dataset(dataset_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh Agent decisions, controls, reports and evaluation from locked upstream artifacts."
    )
    parser.add_argument(
        "--provider",
        choices=["disabled", "codex_cli", "openai_api"],
        default="openai_api",
    )
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    project_paths = paths()
    default_datasets = [
        str(value)
        for value in load_all("NCMMSC2021_AD")["project"].get("default_datasets", [])
    ]
    selected = args.datasets or default_datasets
    for dataset_id in selected:
        print(f"[refresh] {dataset_id}", flush=True)
        print(refresh_dataset(dataset_id, args.provider), flush=True)
    print(build_aggregate_report(project_paths, selected), flush=True)


if __name__ == "__main__":
    main()
