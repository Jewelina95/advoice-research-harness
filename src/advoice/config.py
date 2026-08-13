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
    return {
        "project": load_yaml(p.configs / "project.yaml"),
        "dataset": load_yaml(p.configs / "datasets" / f"{dataset_id}.yaml"),
        "metrics": load_yaml(p.configs / "metrics" / "audio_metrics.yaml"),
        "states": load_yaml(p.configs / "states" / "audio_states.yaml"),
        "models": load_yaml(p.configs / "models" / "default.yaml"),
        "agents": load_yaml(p.configs / "agents" / "default.yaml"),
        "evaluation": load_yaml(p.configs / "evaluation" / "default.yaml"),
    }

