from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def paths() -> ProjectPaths:
    return ProjectPaths(project_root())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_all(dataset_id: str) -> dict[str, Any]:
    p = paths()
    dataset = load_yaml(p.configs / "datasets" / f"{dataset_id}.yaml")
    metrics = load_yaml(p.configs / "metrics" / "audio_metrics.yaml")
    states = load_yaml(p.configs / "states" / "audio_states.yaml")
    profile_name = dataset.get("channel_profile", "audio_only")
    profile = load_yaml(p.configs / "channels" / f"{profile_name}.yaml")
    enabled_states = set(profile.get("enabled_states", []))
    overrides = profile.get("metric_overrides", {})

    selected_metrics = []
    for definition in metrics.get("metrics", []):
        if definition["state"] != "QC" and definition["state"] not in enabled_states:
            continue
        item = dict(definition)
        override = overrides.get(item["id"], {})
        item.update({key: value for key, value in override.items() if key != "confounds_add"})
        item["reliability"] = float(item["reliability"]) * float(override.get("reliability_multiplier", 1.0))
        item.pop("reliability_multiplier", None)
        item["confounds"] = sorted(set(item.get("confounds", [])) | set(override.get("confounds_add", [])))
        selected_metrics.append(item)

    selected_states = [definition for definition in states.get("states", []) if definition["id"] in enabled_states]
    unavailable_states = [
        {
            "id": definition["id"],
            "name_zh": definition["name_zh"],
            "reason": profile.get("unavailable_reasons", {}).get(
                definition["id"], definition.get("planned_reason", "This channel does not currently support the required evidence.")
            ),
        }
        for definition in states.get("states", [])
        if definition["id"] not in enabled_states
    ]

    return {
        "project": load_yaml(p.configs / "project.yaml"),
        "dataset": dataset,
        "channel_profile": {"id": profile_name, **profile},
        "metrics": {"metrics": selected_metrics},
        "states": {"states": selected_states, "unavailable_states": unavailable_states},
        "models": load_yaml(p.configs / "models" / "default.yaml"),
        "agents": load_yaml(p.configs / "agents" / "default.yaml"),
        "evaluation": load_yaml(p.configs / "evaluation" / "default.yaml"),
    }
