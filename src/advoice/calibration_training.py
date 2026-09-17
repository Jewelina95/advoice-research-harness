"""Reserve Agent calibration participants before any supervised model selection."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .agent_runtime import case_pseudonym
from .utils import json_dump


def train_with_dedicated_calibration(train: Callable, arguments: dict[str, Any]) -> None:
    arguments = dict(arguments)
    models = deepcopy(arguments["models_config"])
    config = dict(models["condition_c"]["dedicated_calibration"])
    models["condition_c"]["dedicated_calibration"]["enabled"] = False
    arguments["models_config"] = models
    features = pd.read_csv(arguments["subject_features_path"], dtype={"subject_id": str})
    eligible = features[features.split.eq("train")]
    if (len(eligible) < int(config.get("minimum_training_subjects", 120))
            or eligible.label.value_counts().min() < 4):
        train(**arguments)
        return
    fraction = float(config.get("fraction", .2))
    if not 0 < fraction < .5:
        raise ValueError("Dedicated calibration fraction must be between zero and 0.5.")
    fit_ids, calibration_ids = train_test_split(
        eligible.subject_id.astype(str).to_numpy(), test_size=fraction,
        random_state=int(config.get("seed", 20260917)), stratify=eligible.label.astype(str),
    )
    fit_ids, calibration_ids = set(fit_ids), set(calibration_ids)
    test_ids = set(features.loc[features.split.eq("test"), "subject_id"].astype(str))
    if fit_ids & calibration_ids or (fit_ids | calibration_ids) & test_ids:
        raise ValueError("Supervised fit, Agent calibration, and test subjects must be disjoint.")
    partition = {
        "status": "dedicated_holdout", "fit_subject_ids": sorted(fit_ids),
        "calibration_subject_ids": sorted(calibration_ids), "test_subject_ids": sorted(test_ids),
        "seed": int(config.get("seed", 20260917)), "fraction": fraction,
        "selection_independent": True,
    }
    outputs = ["predictions_path", "base_predictions_path", "ablations_path", "interventions_path",
               "workspaces_path", "contributions_path", "model_path", "metadata_path"]
    with TemporaryDirectory(prefix="advoice-calibration-") as directory:
        root = Path(directory)
        cache_root = Path(arguments["metadata_path"]).parent
        embedding_caches = ("multilingual_text_embeddings.npz", "multilingual_audio_embeddings.npz")
        # Frozen encoders validate their own input/model fingerprints before reuse.
        for name in embedding_caches:
            if (cache_root / name).is_file():
                shutil.copy2(cache_root / name, root / name)
        inner = dict(arguments)
        inner["analysis_manifest_path"] = (
            arguments.get("analysis_manifest_path")
            or Path(arguments["subject_features_path"]).parent / "analysis_manifest.csv"
        )
        for key in ("subject_features_path", "state_wide_path", "metric_evidence_path", "state_cards_path"):
            frame = pd.read_csv(arguments[key], dtype={"subject_id": str})
            frame.loc[frame.subject_id.astype(str).isin(calibration_ids), "split"] = "test"
            inner[key] = root / f"input_{key}.csv"
            frame.to_csv(inner[key], index=False)
        for key in outputs:
            inner[key] = root / Path(arguments[key]).name
        inner["agent_calibration_predictions_path"] = None
        inner["agent_calibration_workspaces_path"] = None
        train(**inner)
        cache_root.mkdir(parents=True, exist_ok=True)
        for name in embedding_caches:
            if (root / name).is_file():
                shutil.copy2(root / name, cache_root / name)
        for key in outputs:
            destination = Path(arguments[key])
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = inner[key]
            if key in {"model_path", "metadata_path", "workspaces_path"}:
                continue
            frame = pd.read_csv(source, dtype={"subject_id": str})
            if "subject_id" in frame:
                frame = frame[frame.subject_id.astype(str).isin(test_ids)]
            frame.to_csv(destination, index=False)
        prior = pd.read_csv(inner["predictions_path"], dtype={"subject_id": str})
        prior = prior[prior.subject_id.astype(str).isin(calibration_ids)].copy()
        prior["condition"] = "B3_agent_calibration_prior"
        prior["oof_status"] = "dedicated_calibration_holdout"
        prior["selection_independent"] = True
        prior["dedicated_calibration_holdout"] = True
        prior["probability_calibration_scope"] = "supervised_fit_partition_only"
        if arguments.get("agent_calibration_predictions_path") is not None:
            Path(arguments["agent_calibration_predictions_path"]).parent.mkdir(parents=True, exist_ok=True)
            prior.to_csv(arguments["agent_calibration_predictions_path"], index=False)
        workspaces = [json.loads(line) for line in inner["workspaces_path"].read_text().splitlines() if line.strip()]
        test_keys = {case_pseudonym(i) for i in test_ids}
        calibration_keys = {case_pseudonym(i) for i in calibration_ids}
        with Path(arguments["workspaces_path"]).open("w") as handle:
            for workspace in workspaces:
                if workspace["case_id"] in test_keys:
                    handle.write(json.dumps(workspace, ensure_ascii=False) + "\n")
        if arguments.get("agent_calibration_workspaces_path") is not None:
            Path(arguments["agent_calibration_workspaces_path"]).parent.mkdir(parents=True, exist_ok=True)
            with Path(arguments["agent_calibration_workspaces_path"]).open("w") as handle:
                for workspace in workspaces:
                    if workspace["case_id"] in calibration_keys:
                        workspace["workspace_role"] = "agent_correction_calibration"
                        workspace["oof_provenance"] = {
                            "selection_independent": True, "dedicated_calibration_holdout": True,
                            "status": "dedicated_calibration_holdout",
                        }
                        handle.write(json.dumps(workspace, ensure_ascii=False) + "\n")
        bundle = joblib.load(inner["model_path"])
        bundle["dedicated_calibration"] = partition
        joblib.dump(bundle, arguments["model_path"])
        metadata = json.loads(inner["metadata_path"].read_text())
        metadata["dedicated_calibration"] = partition
        metadata["agent_workspace_count"] = len(test_ids)
        json_dump(metadata, arguments["metadata_path"])
