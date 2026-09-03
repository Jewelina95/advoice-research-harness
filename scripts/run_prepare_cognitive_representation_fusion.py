from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from advoice.cognitive_representation_fusion import (
    LABELS,
    ProtocolData,
    active_task_state_matrix,
    apply_logit_offsets,
    fit_class_logit_offsets,
    fit_protocol_model,
    predict_protocol_model,
    protocol_metrics,
)
from advoice.deep_audio_embeddings import (
    encode_multilingual_audio,
    load_audio_window_sequences,
)
from advoice.config import load_all
from advoice.states import build_fold_calibrated_state_frame
from advoice.transcripts import repair_utf8_mojibake


def load_data(
    root: Path,
    audio_sequence_mode: str,
    transcript_source: str = "speechcare",
    repair_mojibake: bool = False,
    state_reference_partitions: tuple[str, ...] = ("train",),
) -> ProtocolData:
    artifact = root / "artifacts" / "PREPARE_DrivenData"
    reference = pd.read_csv(
        root / "references" / "speechcare" / "prepare_protocol_inputs.csv",
        dtype={"uid": str},
    )
    features = pd.read_csv(artifact / "subject_features.csv", dtype={"subject_id": str})
    manifest = pd.read_csv(artifact / "analysis_manifest.csv", dtype={"subject_id": str})
    frame = features[["subject_id", "label"]].merge(
        reference, left_on="subject_id", right_on="uid", validate="one_to_one"
    )
    if len(frame) != len(features):
        raise ValueError("SpeechCARE protocol snapshot does not cover PREPARE.")
    subject_ids = frame["subject_id"].tolist()
    task_by_subject = (
        manifest.groupby("subject_id", sort=False)["task_type"].first().astype(str).to_dict()
    )
    reference_subject_ids = set(
        frame.loc[
            frame["reference_partition"].isin(state_reference_partitions), "subject_id"
        ].astype(str)
    )
    if not reference_subject_ids:
        raise ValueError("No subjects are available for fold-local state calibration.")
    states = build_fold_calibrated_state_frame(
        pd.read_csv(artifact / "metric_evidence.csv", dtype={"subject_id": str}),
        load_all("PREPARE_DrivenData")["states"],
        reference_subject_ids,
        "HC",
    )
    states = frame[["subject_id"]].merge(states, on="subject_id", validate="one_to_one")
    state_values, state_reliability, state_names = active_task_state_matrix(
        states, task_by_subject
    )
    if audio_sequence_mode == "window_mean":
        audio_cache = artifact / "multilingual_audio_embeddings.npz"
    elif audio_sequence_mode in {"temporal_bins", "window_stats"}:
        audio_cache = artifact / (
            "multilingual_audio_temporal_embeddings.npz"
            if audio_sequence_mode == "temporal_bins"
            else "multilingual_audio_window_stats_embeddings.npz"
        )
        encode_multilingual_audio(
            artifact / "analysis_manifest.csv",
            subject_ids,
            audio_cache,
            {
                "model": "utter-project/mHuBERT-147",
                "revision": "7ad3fc0bc5106c58c9c13526abccad527150d135",
                "sample_rate": 16000,
                "lowpass_hz": 7999.0,
                "window_seconds": 5.0,
                "overlap": 0.20,
                "max_subject_seconds": 30.0,
                "batch_size": 8,
                "temporal_bins_per_window": (
                    8 if audio_sequence_mode == "temporal_bins" else 1
                ),
                "token_pooling": (
                    "temporal_bins"
                    if audio_sequence_mode == "temporal_bins"
                    else "window_stats"
                ),
                "device": "auto",
            },
        )
    else:
        raise ValueError(f"Unsupported audio_sequence_mode={audio_sequence_mode!r}")
    audio, audio_mask = load_audio_window_sequences(audio_cache, subject_ids)
    task = frame["subject_id"].map(task_by_subject).fillna("other")
    language = frame["language"].fillna("unknown").str.lower()
    if transcript_source == "speechcare":
        texts = frame["transcription"].fillna("[no_transcript]").astype(str)
        if repair_mojibake:
            texts = texts.map(repair_utf8_mojibake)
    elif transcript_source == "asr":
        transcript_by_subject = (
            manifest.groupby("subject_id", sort=False)["transcript_path"].first().to_dict()
        )
        texts = frame["subject_id"].map(
            lambda subject_id: (
                Path(str(transcript_by_subject.get(subject_id, ""))).read_text(
                    encoding="utf-8", errors="replace"
                )
                if Path(str(transcript_by_subject.get(subject_id, ""))).is_file()
                else "[no_transcript]"
            )
        )
    else:
        raise ValueError(f"Unsupported transcript_source={transcript_source!r}")
    task_categories = sorted(task.unique())
    language_categories = sorted(language.unique())
    task_map = {value: index for index, value in enumerate(task_categories)}
    language_map = {value: index for index, value in enumerate(language_categories)}
    return ProtocolData(
        subject_ids=subject_ids,
        labels=frame["label"].to_numpy(dtype=str),
        partitions=frame["reference_partition"].to_numpy(dtype=str),
        texts=texts.tolist(),
        audio_sequence=audio,
        audio_mask=audio_mask,
        states=state_values,
        reliability=state_reliability,
        task_index=task.map(task_map).to_numpy(dtype=np.int64),
        language_index=language.map(language_map).to_numpy(dtype=np.int64),
        age=pd.to_numeric(frame["age"], errors="coerce").to_numpy(dtype=np.float32),
        task_categories=task_categories,
        language_categories=language_categories,
        state_features=state_names,
    )


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_trainable_checkpoint(model: object, path: Path) -> None:
    """Persist the learned delta; frozen encoders are pinned by model revision."""
    import torch

    learned = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(learned, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260830])
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument(
        "--text-backbone",
        choices=("gte", "e5"),
        default="gte",
        help="Pinned multilingual text encoder; validation runs are isolated.",
    )
    parser.add_argument("--trainable-text-layers", type=int, default=2)
    parser.add_argument("--text-learning-rate", type=float, default=1e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.10)
    parser.add_argument("--hierarchical-loss-weight", type=float, default=0.0)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument(
        "--transcript-source",
        choices=("speechcare", "asr"),
        default="speechcare",
        help="Explicit text provenance for the protocol ablation.",
    )
    parser.add_argument(
        "--repair-mojibake",
        action="store_true",
        help="Repair double-decoded UTF-8 in the released SpeechCARE CSV.",
    )
    parser.add_argument(
        "--class-weighting", choices=("none", "balanced"), default="none"
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--run-name",
        help="Stable run identifier. Validation runs never replace promoted test artifacts.",
    )
    parser.add_argument(
        "--audio-sequence-mode",
        choices=("window_mean", "window_stats", "temporal_bins"),
        default="window_mean",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = root / "artifacts" / "PREPARE_DrivenData" / "cognitive_fusion_protocol"
    output.mkdir(parents=True, exist_ok=True)
    run_kind = "validation" if arguments.validation_only else "official_test"
    run_name = arguments.run_name or (
        f"{run_kind}_{arguments.audio_sequence_mode}_"
        + "-".join(str(seed) for seed in arguments.seeds)
    )
    run_output = output / "runs" / run_name
    run_output.mkdir(parents=True, exist_ok=True)
    data = load_data(
        root,
        arguments.audio_sequence_mode,
        transcript_source=arguments.transcript_source,
        repair_mojibake=arguments.repair_mojibake,
        state_reference_partitions=("train",),
    )
    train = np.flatnonzero(data.partitions == "train")
    validation = np.flatnonzero(data.partitions == "validation")
    test = np.flatnonzero(data.partitions == "test")
    assert not set(np.concatenate([train, validation])).intersection(test)
    text_backbones = {
        "gte": {
            "text_model": "Alibaba-NLP/gte-multilingual-base",
            "text_revision": "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
            "text_code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4",
            "text_prefix": "",
            "text_pooling": "cls",
            "explicit_position_ids": True,
        },
        "e5": {
            "text_model": "intfloat/multilingual-e5-base",
            "text_revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
            "text_code_revision": None,
            "text_prefix": "passage: ",
            "text_pooling": "mean",
            "explicit_position_ids": False,
        },
    }
    config = {
        **text_backbones[arguments.text_backbone],
        "trainable_text_layers": arguments.trainable_text_layers,
        "max_length": 512,
        "hidden_dimension": 128,
        "maximum_windows": int(data.audio_sequence.shape[1]),
        "dropout": 0.15,
        "batch_size": 16,
        "epochs": arguments.epochs,
        "patience": 3,
        "text_learning_rate": arguments.text_learning_rate,
        "head_learning_rate": arguments.head_learning_rate,
        "weight_decay": 0.005,
        "auxiliary_loss_weight": arguments.auxiliary_loss_weight,
        "hierarchical_loss_weight": arguments.hierarchical_loss_weight,
        "selection_metric": "micro_f1",
        "class_weighting": arguments.class_weighting,
        "audio_sequence_mode": arguments.audio_sequence_mode,
        "content_in_gate": True,
        "gate_floor": arguments.gate_floor,
        "transcript_source": arguments.transcript_source,
        "repair_mojibake": arguments.repair_mojibake,
        "state_reference_partitions": ["train"],
        "device": "auto",
    }
    runs: list[dict[str, object]] = []
    test_probabilities: list[np.ndarray] = []
    test_weights: list[np.ndarray] = []
    for seed in arguments.seeds:
        model, metadata = fit_protocol_model(data, train, validation, config, seed)
        validation_probability, _ = predict_protocol_model(
            model, data, validation, config
        )
        offsets, adjusted_validation = fit_class_logit_offsets(
            data.labels[validation], validation_probability
        )
        metadata["class_logit_offsets"] = offsets.tolist()
        metadata["threshold_adjusted_validation"] = adjusted_validation
        runs.append(metadata)
        print(json.dumps({"seed": seed, **metadata["validation"]}, indent=2))
        if arguments.validation_only:
            del model
            release_memory()
            continue
        selected_epochs = max(1, int(metadata["best_epoch"]))
        del model
        release_memory()
        # Select epoch count on the released validation set, then refit on all
        # development subjects. Test labels are evaluation-only.
        full_development = np.concatenate([train, validation])
        final_data = load_data(
            root,
            arguments.audio_sequence_mode,
            transcript_source=arguments.transcript_source,
            repair_mojibake=arguments.repair_mojibake,
            state_reference_partitions=("train", "validation"),
        )
        final_model, _ = fit_protocol_model(
            final_data,
            full_development,
            validation,
            config,
            seed + 1000,
            fixed_epochs=selected_epochs,
        )
        probability, weights = predict_protocol_model(final_model, final_data, test, config)
        test_probabilities.append(probability)
        test_weights.append(weights)
        save_trainable_checkpoint(
            final_model, run_output / f"trainable_delta_seed_{seed}.pt"
        )
        del final_model
        release_memory()
    result: dict[str, object] = {
        "protocol": "SpeechCARE official train/validation/test split",
        "test_labels_used_for_training_or_selection": False,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "config": config,
        "validation_runs": runs,
        "state_features": data.state_features,
        "run_name": run_name,
        "artifact_policy": (
            "validation_isolated" if arguments.validation_only else "promoted_official_test"
        ),
        "input_sha256": {
            "protocol_inputs": sha256(
                root / "references" / "speechcare" / "prepare_protocol_inputs.csv"
            ),
            "subject_features": sha256(
                root / "artifacts" / "PREPARE_DrivenData" / "subject_features.csv"
            ),
            "analysis_manifest": sha256(
                root / "artifacts" / "PREPARE_DrivenData" / "analysis_manifest.csv"
            ),
            "state_wide": sha256(
                root / "artifacts" / "PREPARE_DrivenData" / "state_wide.csv"
            ),
            "metric_evidence": sha256(
                root / "artifacts" / "PREPARE_DrivenData" / "metric_evidence.csv"
            ),
        },
    }
    if test_probabilities:
        probability = np.mean(test_probabilities, axis=0)
        weights = np.mean(test_weights, axis=0)
        result["official_test"] = protocol_metrics(data.labels[test], probability)
        mean_offsets = np.mean(
            [np.asarray(run["class_logit_offsets"], dtype=float) for run in runs], axis=0
        )
        adjusted_probability = apply_logit_offsets(probability, mean_offsets)
        result["official_test_threshold_adjusted"] = protocol_metrics(
            data.labels[test], adjusted_probability
        )
        result["official_test_mean_gate_weights"] = weights.mean(axis=0).tolist()
        predictions = pd.DataFrame(
            {
                "subject_id": np.asarray(data.subject_ids)[test],
                "true_label": data.labels[test],
                "predicted_label": np.asarray(LABELS)[probability.argmax(axis=1)],
                **{f"prob_{label}": probability[:, i] for i, label in enumerate(LABELS)},
                "gate_text": weights[:, 0],
                "gate_audio": weights[:, 1],
                "gate_state": weights[:, 2],
            }
        )
        run_prediction_path = run_output / "official_test_predictions.csv"
        predictions.to_csv(run_prediction_path, index=False)
        result["prediction_sha256"] = sha256(run_prediction_path)
        print(json.dumps(result["official_test"], indent=2))
    run_result_path = run_output / "result.json"
    run_result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if test_probabilities:
        shutil.copy2(run_prediction_path, output / "official_test_predictions.csv")
        shutil.copy2(run_result_path, output / "result.json")
        (output / "official_test_predictions.meta.json").write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "prediction_sha256": result["prediction_sha256"],
                    "result_path": str(run_result_path.relative_to(root)),
                    "result_sha256": sha256(run_result_path),
                    "config": config,
                    "input_sha256": result["input_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(run_result_path)


if __name__ == "__main__":
    main()
