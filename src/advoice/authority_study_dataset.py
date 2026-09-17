"""Label-isolated dataset preparation for authority-review pilot studies.

This module defines the boundary before an authority-review runtime or a
provider receives a case.  It loads immutable historical artifacts, fits a
new state-only Module A expert on declared training rows, and prepares only
test cases.  Ground-truth labels are deliberately available through a
separate evaluation accessor and are never read by ``prepare_test_cases``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .condition_c_advisor import FrozenConditionCAdvisor
from .conditional_authority import ConditionalAuthorityExecutor, PreparedAuthorityCase
from .conditional_authority_pilot import FrozenAuthorityDataset, load_frozen_authority_dataset
from .config import load_yaml, paths
from .evidence import MetricEvidenceV2
from .evidence_replay import build_state_graph_v2
from .module_a import TaskConditionedStatisticalExpert
from .module_b import ConditionalArbitrator


class AuthorityStudyDatasetError(ValueError):
    """Raised when a frozen artifact cannot safely prepare a review cohort."""


_DIAGNOSIS_TARGET_VALUES = frozenset({"diagnosis", "cross_sectional_diagnosis"})
_TARGET_ROUTE_COLUMNS = ("target_route", "target", "target_type", "endpoint")
_PROGRESSION_LABEL_VALUES = frozenset({
    "progression", "progressor", "nonprogressor", "decline", "no_decline", "stable",
})


@dataclass(frozen=True, slots=True)
class PreparedAuthorityStudyCase:
    """One provider-ready case plus its transcript, kept outside metadata.

    ``prepared_case.case_metadata`` intentionally contains no label, split,
    filesystem path, or transcript path.  The transcript is a separate value
    so a provider payload can be assembled explicitly by the caller.
    """

    prepared_case: PreparedAuthorityCase
    transcript: str


@dataclass(slots=True)
class AuthorityStudyDataset:
    """Frozen dataset and label-isolated authority-review preparation state."""

    frozen: FrozenAuthorityDataset
    advisor: FrozenConditionCAdvisor
    executor: ConditionalAuthorityExecutor
    _transcripts: Mapping[str, str]
    _subject_routing: Mapping[str, Mapping[str, str]]
    _evaluation_truth: Mapping[str, str]

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        states_config: Mapping[str, Any] | None = None,
    ) -> "AuthorityStudyDataset":
        """Load one artifact directory and fit Module A on ``split=train`` only."""

        root = Path(artifact_dir).expanduser().resolve()
        frozen = load_frozen_authority_dataset(root)
        _require_diagnosis_target(frozen)
        advisor = FrozenConditionCAdvisor.from_artifact_dir(
            root,
            expected_dataset_id=_dataset_id_from_frozen(frozen),
        )
        _validate_advisor_contract(frozen, advisor)
        transcripts = _load_transcripts(root / "subject_transcripts.csv")
        routing = _subject_routing(frozen.manifest)
        _validate_test_transcript_coverage(frozen, transcripts)

        state_config = (
            dict(states_config)
            if states_config is not None
            else load_yaml(paths().configs / "states" / "audio_states.yaml")
        )
        train_frame = frozen.state_wide.loc[
            frozen.state_wide["split"].astype(str).eq("train")
        ].copy()
        if train_frame.empty:
            raise AuthorityStudyDatasetError("state_wide.csv has no split=train rows.")
        whitelist = _replayable_state_feature_whitelist(
            frozen,
            train_frame=train_frame,
            states_config=state_config,
        )
        train_labels = tuple(train_frame["label"].astype(str))
        if set(train_labels) != set(advisor.class_order):
            raise AuthorityStudyDatasetError(
                "Training labels and frozen Condition C class order differ: "
                f"train={sorted(set(train_labels))}, advisor={list(advisor.class_order)}."
            )

        train_routing = _routing_columns_for_frame(train_frame, routing)
        train_frame = train_frame.assign(**train_routing)
        module_a = TaskConditionedStatisticalExpert(
            advisor.class_order,
            task_column="__authority_task_type",
            language_column="__authority_language",
            require_explicit_feature_whitelist=True,
        ).fit(
            train_frame,
            train_labels,
            feature_columns=whitelist,
            artifact_snapshot={
                "dataset_id": advisor.dataset_id,
                "training_split": "train",
                "training_subject_ids": sorted(train_frame["subject_id"].astype(str).tolist()),
                "state_feature_whitelist": list(whitelist),
            },
        )
        executor = ConditionalAuthorityExecutor(
            states_config=state_config,
            module_a=module_a,
            module_b=ConditionalArbitrator(advisor.class_order),
            module_a_state_feature_whitelist=whitelist,
            target_route_config={
                "diagnosis": {
                    "endpoint": "cross_sectional_diagnosis",
                    "labels": list(advisor.class_order),
                    "requires_paired_visits": False,
                    "minimum_visits": 1,
                }
            },
        )
        return cls(
            frozen=frozen,
            advisor=advisor,
            executor=executor,
            _transcripts=transcripts,
            _subject_routing=routing,
            _evaluation_truth=dict(frozen.subject_labels),
        )

    @property
    def class_order(self) -> tuple[str, ...]:
        """Fixed diagnostic class order shared by advisor, Module A, and route."""

        return self.advisor.class_order

    def prepare_test_cases(
        self,
        *,
        max_cases: int | None = None,
        order: str = "longest_first",
        selection_salt: str = "authority-pilot-v1",
    ) -> tuple[PreparedAuthorityStudyCase, ...]:
        """Prepare deterministic test cases without consulting their labels.

        ``order`` is ``longest_first`` (descending transcript character count),
        ``subject_id`` (ascending subject ID), or ``stable_hash``. The hash
        order binds dataset, channel, case ID, and an explicit salt; it never
        consults test truth.
        """

        if max_cases is not None and max_cases < 1:
            raise AuthorityStudyDatasetError("max_cases must be positive when supplied.")
        normalized_order = _normalize_order(order)
        normalized_salt = str(selection_salt).strip()
        if not normalized_salt:
            raise AuthorityStudyDatasetError("selection_salt must be non-empty.")
        subject_ids = [
            str(subject_id)
            for subject_id, split in self.frozen.subject_splits.items()
            if str(split) == "test"
        ]
        if not subject_ids:
            raise AuthorityStudyDatasetError("Frozen artifact has no split=test subjects.")
        missing_evidence = [
            subject_id
            for subject_id in subject_ids
            if not self.frozen.evidence_for_subject(subject_id)
        ]
        if missing_evidence:
            raise AuthorityStudyDatasetError(
                "Test subjects have no typed MetricEvidenceV2: " + ", ".join(sorted(missing_evidence))
            )
        missing_transcripts = [subject_id for subject_id in subject_ids if subject_id not in self._transcripts]
        if missing_transcripts:
            raise AuthorityStudyDatasetError(
                "Test subjects are missing transcripts: " + ", ".join(sorted(missing_transcripts))
            )

        if normalized_order == "longest_first":
            subject_ids.sort(key=lambda subject_id: (-len(self._transcripts[subject_id]), subject_id))
        elif normalized_order == "stable_hash":
            subject_ids.sort(
                key=lambda subject_id: (
                    hashlib.sha256(
                        "|".join((
                            self.advisor.dataset_id,
                            self._subject_routing[subject_id]["channel"],
                            subject_id,
                            normalized_salt,
                        )).encode("utf-8")
                    ).hexdigest(),
                    subject_id,
                )
            )
        else:
            subject_ids.sort()
        if max_cases is not None:
            subject_ids = subject_ids[:max_cases]

        prepared: list[PreparedAuthorityStudyCase] = []
        for subject_id in subject_ids:
            routing = self._subject_routing[subject_id]
            case_metadata = {
                "case_id": subject_id,
                "subject_id": subject_id,
                "dataset_id": self.advisor.dataset_id,
                "channel": routing["channel"],
                "task_type": routing["task_type"],
                "language": routing["language"],
                "target_route": "diagnosis",
                "target_labels": list(self.advisor.class_order),
            }
            prepared_case = self.executor.prepare_case(
                case_metadata=case_metadata,
                evidence=self.frozen.evidence_for_subject(subject_id),
            )
            prepared.append(
                PreparedAuthorityStudyCase(
                    prepared_case=prepared_case,
                    transcript=self._transcripts[subject_id],
                )
            )
        return tuple(prepared)

    def evaluation_truth(self, subject_ids: Sequence[str] | None = None) -> dict[str, str]:
        """Return frozen labels only after provider-free preparation is complete."""

        requested = (
            tuple(str(subject_id) for subject_id in subject_ids)
            if subject_ids is not None
            else tuple(str(subject_id) for subject_id in self.frozen.subject_splits)
        )
        if len(set(requested)) != len(requested):
            raise AuthorityStudyDatasetError("evaluation_truth subject_ids must be unique.")
        unknown = sorted(set(requested) - set(self._evaluation_truth))
        if unknown:
            raise AuthorityStudyDatasetError(f"Unknown subjects requested for evaluation truth: {unknown}")
        return {subject_id: self._evaluation_truth[subject_id] for subject_id in requested}


def _dataset_id_from_frozen(frozen: FrozenAuthorityDataset) -> str:
    values = {
        str(value).strip()
        for value in frozen.manifest.get("dataset_id", pd.Series(dtype="string")).tolist()
        if str(value).strip()
    }
    if len(values) != 1:
        raise AuthorityStudyDatasetError(
            f"manifest.csv must declare one non-empty dataset_id; received {sorted(values)}."
        )
    return next(iter(values))


def _normalize_target_value(value: Any) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _explicit_manifest_values(manifest: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in manifest.columns:
        return ()
    return tuple(
        value
        for value in (_normalize_target_value(item) for item in manifest[column].tolist())
        if value and value not in {"nan", "none", "null"}
    )


def _require_diagnosis_target(frozen: FrozenAuthorityDataset) -> None:
    """Reject an artifact that declares an endpoint this study cannot evaluate.

    Historical artifacts without target metadata predate explicit routing and
    remain diagnosis artifacts for backward compatibility.  Once an artifact
    declares a route, silently mapping it to ``diagnosis`` would turn a
    longitudinal endpoint into a false cross-sectional label.
    """

    manifest = frozen.manifest
    for column in _TARGET_ROUTE_COLUMNS:
        values = set(_explicit_manifest_values(manifest, column))
        unsupported = sorted(values - _DIAGNOSIS_TARGET_VALUES)
        if unsupported:
            raise AuthorityStudyDatasetError(
                "AuthorityStudyDataset only supports diagnosis targets; "
                f"manifest column {column!r} declares {unsupported}."
            )

    label_values = set(_explicit_manifest_values(manifest, "label"))
    label_values.update(_normalize_target_value(value) for value in frozen.subject_labels.values())
    progression = sorted(label_values & _PROGRESSION_LABEL_VALUES)
    if progression:
        raise AuthorityStudyDatasetError(
            "AuthorityStudyDataset only supports diagnosis targets; "
            f"labels declare progression semantics: {progression}."
        )


def _validate_advisor_contract(
    frozen: FrozenAuthorityDataset,
    advisor: FrozenConditionCAdvisor,
) -> None:
    dataset_id = _dataset_id_from_frozen(frozen)
    if advisor.dataset_id != dataset_id:
        raise AuthorityStudyDatasetError(
            f"Frozen advisor dataset_id {advisor.dataset_id!r} does not match manifest {dataset_id!r}."
        )
    test_subjects = {
        str(subject_id)
        for subject_id, split in frozen.subject_splits.items()
        if str(split) == "test"
    }
    missing = sorted(test_subjects - set(advisor.subject_ids))
    extra = sorted(set(advisor.subject_ids) - test_subjects)
    if missing or extra:
        raise AuthorityStudyDatasetError(
            "Frozen advisor subjects do not match authority artifact test subjects: "
            f"missing={missing}, extra={extra}."
        )


def _replayable_state_feature_whitelist(
    frozen: FrozenAuthorityDataset,
    *,
    train_frame: pd.DataFrame,
    states_config: Mapping[str, Any],
) -> tuple[str, ...]:
    train_subjects = tuple(train_frame["subject_id"].astype(str))
    training_evidence = _representative_training_evidence(
        frozen,
        train_subjects=train_subjects,
        states_config=states_config,
    )
    graph = build_state_graph_v2(
        training_evidence,
        states_config,
        dataset_id=_dataset_id_from_frozen(frozen),
        label="unknown",
        split="train_graph_contract",
    )
    allowed_prefixes = ("state_", "rel_", "available_")
    stored_features = {
        str(column) for column in train_frame.columns if str(column).startswith(allowed_prefixes)
    }
    replayable_features = {
        str(column) for column in graph.wide.columns if str(column).startswith(allowed_prefixes)
    }
    whitelist = tuple(sorted(stored_features & replayable_features))
    if not whitelist:
        raise AuthorityStudyDatasetError(
            "state_wide.csv and replayed training StateGraphV2 have no common "
            "state_/rel_/available_ features."
        )
    return whitelist


def _representative_training_evidence(
    frozen: FrozenAuthorityDataset,
    *,
    train_subjects: Sequence[str],
    states_config: Mapping[str, Any],
) -> tuple[MetricEvidenceV2, ...]:
    """Select a deterministic minimum cover of replayable state/task scopes.

    StateGraphV2 feature names depend on configured state definitions and the
    task scopes present in typed evidence, not on cohort size or labels.  A
    greedy set cover therefore discovers the same feature-name contract while
    avoiding a full graph rebuild over every training subject.
    """

    metric_states: dict[str, set[str]] = {}
    for definition in states_config.get("states", ()):
        state_id = str(definition.get("id", "")).strip()
        if not state_id:
            continue
        for metric_id in definition.get("metrics", ()):
            metric_states.setdefault(str(metric_id), set()).add(state_id)
    if not metric_states:
        raise AuthorityStudyDatasetError(
            "states_config has no state metrics from which StateGraphV2 features can be rebuilt."
        )

    evidence_by_subject: dict[str, tuple[MetricEvidenceV2, ...]] = {}
    signatures_by_subject: dict[str, frozenset[tuple[str, str]]] = {}
    missing_evidence: list[str] = []
    for subject_id in sorted(set(str(value) for value in train_subjects)):
        evidence = tuple(
            item
            for item in frozen.evidence_for_subject(subject_id)
            if item.consumed_by_supervised
        )
        signatures = frozenset(
            (state_id, str(item.task_id or "overall"))
            for item in evidence
            for state_id in metric_states.get(str(item.metric_id), ())
        )
        if not evidence or not signatures:
            missing_evidence.append(subject_id)
            continue
        evidence_by_subject[subject_id] = evidence
        signatures_by_subject[subject_id] = signatures
    if missing_evidence:
        raise AuthorityStudyDatasetError(
            "Training subjects have no supervised evidence supported by states_config: "
            + ", ".join(missing_evidence)
        )

    uncovered = set().union(*signatures_by_subject.values())
    selected: list[str] = []
    while uncovered:
        ranked = sorted(
            signatures_by_subject,
            key=lambda subject_id: (
                -len(signatures_by_subject[subject_id] & uncovered),
                subject_id,
            ),
        )
        subject_id = ranked[0]
        covered = signatures_by_subject[subject_id] & uncovered
        if not covered:
            raise AuthorityStudyDatasetError(
                "Could not cover all replayable state/task signatures from training evidence."
            )
        selected.append(subject_id)
        uncovered -= covered
    return tuple(item for subject_id in selected for item in evidence_by_subject[subject_id])


def _load_transcripts(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AuthorityStudyDatasetError("Missing required subject_transcripts.csv.")
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError) as exc:
        raise AuthorityStudyDatasetError("Cannot read subject_transcripts.csv.") from exc
    required = {"subject_id", "transcript"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuthorityStudyDatasetError(
            f"subject_transcripts.csv is missing required columns: {missing}"
        )
    subject_ids = frame["subject_id"].astype(str).str.strip()
    if subject_ids.eq("").any():
        raise AuthorityStudyDatasetError("subject_transcripts.csv contains blank subject_id values.")
    duplicates = sorted(subject_ids[subject_ids.duplicated()].unique().tolist())
    if duplicates:
        raise AuthorityStudyDatasetError(
            "subject_transcripts.csv has duplicate subject IDs: " + ", ".join(duplicates)
        )
    transcripts = frame["transcript"].astype(str).str.strip()
    empty = sorted(subject_ids[transcripts.eq("")].tolist())
    if empty:
        raise AuthorityStudyDatasetError(
            "subject_transcripts.csv contains empty transcripts: " + ", ".join(empty)
        )
    return dict(zip(subject_ids.tolist(), transcripts.tolist(), strict=True))


def _subject_routing(manifest: pd.DataFrame) -> dict[str, dict[str, str]]:
    required = {"subject_id", "channel"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise AuthorityStudyDatasetError(f"manifest.csv is missing routing columns: {missing}")
    output: dict[str, dict[str, str]] = {}
    for raw_subject_id, group in manifest.groupby("subject_id", sort=False):
        subject_id = str(raw_subject_id).strip()
        if not subject_id:
            raise AuthorityStudyDatasetError("manifest.csv contains blank subject_id values.")
        channel_values = _unique_manifest_values(group, "channel")
        if len(channel_values) != 1:
            raise AuthorityStudyDatasetError(
                f"Manifest subject {subject_id!r} must have exactly one channel; got {channel_values}."
            )
        task_values = _unique_manifest_values(group, "task_type") or ("unknown_task",)
        language_values = _unique_manifest_values(group, "language") or ("unknown",)
        output[subject_id] = {
            "channel": channel_values[0],
            "task_type": "|".join(sorted(task_values)),
            "language": "|".join(sorted(language_values)),
        }
    return output


def _unique_manifest_values(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame:
        return ()
    values = tuple(
        dict.fromkeys(
            value
            for value in frame[column].astype(str).str.strip().tolist()
            if value and value.lower() not in {"nan", "none", "null"}
        )
    )
    return values


def _routing_columns_for_frame(
    frame: pd.DataFrame,
    routing: Mapping[str, Mapping[str, str]],
) -> dict[str, list[str]]:
    subject_ids = frame["subject_id"].astype(str).tolist()
    missing = sorted(set(subject_ids) - set(routing))
    if missing:
        raise AuthorityStudyDatasetError(f"Training subjects missing manifest routing metadata: {missing}")
    return {
        "__authority_task_type": [routing[subject_id]["task_type"] for subject_id in subject_ids],
        "__authority_language": [routing[subject_id]["language"] for subject_id in subject_ids],
    }


def _validate_test_transcript_coverage(
    frozen: FrozenAuthorityDataset,
    transcripts: Mapping[str, str],
) -> None:
    test_subjects = {
        str(subject_id)
        for subject_id, split in frozen.subject_splits.items()
        if str(split) == "test"
    }
    missing = sorted(test_subjects - set(transcripts))
    if missing:
        raise AuthorityStudyDatasetError(
            "subject_transcripts.csv is missing test subjects: " + ", ".join(missing)
        )


def _normalize_order(order: str) -> str:
    normalized = str(order).strip().lower().replace("-", "_")
    if normalized not in {"longest_first", "subject_id", "stable_hash"}:
        raise AuthorityStudyDatasetError(
            "order must be 'longest_first', 'subject_id', or 'stable_hash'."
        )
    return normalized
