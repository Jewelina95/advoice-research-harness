from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import advoice.condition_c as condition_c
from test_training_evaluation_regressions import _training_inputs


class _ManifestInspected(Exception):
    """Stop at the encoder boundary, before any model or audio loading."""


@pytest.mark.parametrize("explicit_override", [False, True], ids=["sibling", "override"])
@pytest.mark.parametrize("relative_paths", [False, True], ids=["absolute", "relative"])
def test_dedicated_training_preserves_audio_manifest_location(
    tmp_path: Path, monkeypatch, explicit_override: bool, relative_paths: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    paths, states, models = _training_inputs(input_root, ["HC", "AD"], train_size=40)
    models["condition_c"].update({
        "dedicated_calibration": {
            "enabled": True, "fraction": 0.25,
            "minimum_training_subjects": 30, "seed": 17,
        },
        "deep_audio": {"enabled": True, "minimum_training_subjects": 1},
        "deep_text": {"enabled": False},
        "qc_shortcut_guard": {"enabled": False},
    })
    media = tmp_path / "media" / "clip.wav"
    media.parent.mkdir()
    media.write_bytes(b"encoder boundary fixture; not decoded")
    manifest = pd.DataFrame({
        "subject_id": [str(index) for index in range(46)],
        "audio_path": ["media/clip.wav"] * 46,
    })
    sibling = input_root / "analysis_manifest.csv"
    manifest.to_csv(sibling, index=False)
    manifest_path = sibling
    if explicit_override:
        manifest_path = tmp_path / "override" / "analysis_manifest.csv"
        manifest_path.parent.mkdir()
        manifest.to_csv(manifest_path, index=False)
        # A discoverable sibling must not take precedence over the explicit path.
        manifest.assign(audio_path="wrong-sibling.wav").to_csv(sibling, index=False)
    original_manifest = manifest_path.read_bytes()
    if relative_paths:
        paths = [path.relative_to(tmp_path) for path in paths]
        manifest_path = manifest_path.relative_to(tmp_path)

    output_root = tmp_path / "run"
    output_root.mkdir()
    outputs = [output_root / name for name in (
        "predictions.csv", "base.csv", "ablations.csv", "interventions.csv",
        "workspaces.jsonl", "contributions.csv", "model.joblib", "metadata.json",
    )]
    arguments = dict(inspect.signature(condition_c.train_condition_c).bind(
        *paths, states, models, *outputs,
    ).arguments)
    if explicit_override:
        arguments["analysis_manifest_path"] = manifest_path

    def inspect_manifest(received_path, subject_ids, cache_path, config):
        assert received_path == manifest_path, "Forward the original path, not a relocated copy"
        assert Path(received_path).is_file()
        assert Path(received_path).read_bytes() == original_manifest
        received = pd.read_csv(received_path, dtype={"subject_id": str})
        pd.testing.assert_frame_equal(received, manifest)
        assert set(subject_ids) == set(manifest.subject_id)
        # The encoder resolves audio_path relative to the working directory.
        assert received.audio_path.eq("media/clip.wav").all()
        assert Path(received.audio_path.iloc[0]).resolve() == media.resolve()
        assert Path(received.audio_path.iloc[0]).read_bytes() == media.read_bytes()
        assert Path(cache_path).parent != output_root
        assert config["enabled"] is True
        raise _ManifestInspected

    monkeypatch.setattr(condition_c, "encode_multilingual_audio", inspect_manifest)
    with pytest.raises(_ManifestInspected):
        condition_c.train_condition_c(**arguments)
    assert manifest_path.read_bytes() == original_manifest
