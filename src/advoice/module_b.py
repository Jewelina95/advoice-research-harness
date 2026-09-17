"""Conditional, bounded calibration of Agent evidence after Module A replay.

Module B is deliberately *not* a global blend between a statistical model and
an Agent.  The Agent's main effect is an evidence revision followed by a
numerical Module A replay.  This module can add a small residual only when the
case declares validated, incremental evidence that Module A did not consume.

The fitted object is intentionally small: a deterministic ridge map from
validated ordinal scores and route/action interactions to a zero-sum logit
correction.  It is designed for cross-fitted development rows; it must never
be fitted on evaluation predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODULE_B_VERSION = "advoice.module_b.conditional_ridge.v2"
MODULE_B_SCHEMA_VERSION = "advoice.module_b.contract.v2"
_FALLBACK_POST_REPLAY = "module_a_post_replay"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exponentiated = np.exp(shifted)
    return exponentiated / float(exponentiated.sum())


def _ordered_labels(labels: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(label) for label in labels)
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("labels must contain at least two unique entries in a deterministic order.")
    return result


def _probabilities(value: Any, labels: tuple[str, ...], field: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping keyed by the fixed class order.")
    keys = tuple(str(key) for key in value)
    if set(keys) != set(labels):
        raise ValueError(f"{field} must contain exactly the configured class labels.")
    array = np.asarray([float(value[label]) for label in labels], dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{field} must be finite, non-negative, and sum to one.")
    return array


def _unit_interval(value: Any, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a finite value in [0, 1].")
    return result


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a boolean.")
    return bool(value)


def _evidence_ids(row: Mapping[str, Any], field: str) -> tuple[str, ...]:
    """Canonicalize a traceable evidence-ID collection without accepting text."""

    value = row.get(field, ())
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise ValueError(f"{field} must be a sequence of non-empty evidence IDs.")
    try:
        ids = tuple(str(item).strip() for item in value)
    except TypeError as error:
        raise ValueError(f"{field} must be a sequence of non-empty evidence IDs.") from error
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{field} must contain unique, non-empty evidence IDs.")
    return tuple(sorted(ids))


@dataclass(frozen=True)
class ConditionalArbitrationInput:
    """One complete, label-free Module B input contract.

    ``module_a_post_replay`` is always the numerical baseline.  The pre-replay
    vector is retained for audit only; using it as a second voting signal would
    double-count a reviewed state revision.
    """

    module_a_pre_replay: Mapping[str, float]
    module_a_post_replay: Mapping[str, float]
    agent_ordinal_scores: Mapping[str, int]
    agent_scores_validated: bool
    eligible: bool
    revision_type: str
    action_type: str
    agreement: bool
    evidence_coverage: float
    evidence_reliability: float
    confound_burden: float
    route: str
    language: str
    ood: float
    incremental_evidence_declared: bool
    evidence_consumed_by_module_a: bool
    route_supported: bool
    replay_performed: bool
    incremental_evidence_ids: tuple[str, ...] = ()
    consumed_evidence_ids: tuple[str, ...] = ()
    cross_fit_fold: str | int | None = None

    @classmethod
    def from_mapping(
        cls, row: Mapping[str, Any], labels: Sequence[str]) -> "ConditionalArbitrationInput":
        order = _ordered_labels(labels)
        required = {
            "module_a_pre_replay", "module_a_post_replay", "agent_ordinal_scores",
            "agent_scores_validated", "eligible", "revision_type", "action_type", "agreement",
            "evidence_coverage", "evidence_reliability", "confound_burden", "route", "language",
            "ood", "incremental_evidence_declared", "evidence_consumed_by_module_a",
            "route_supported", "replay_performed",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Module B input is missing required fields: {missing}")
        pre = _probabilities(row["module_a_pre_replay"], order, "module_a_pre_replay")
        post = _probabilities(row["module_a_post_replay"], order, "module_a_post_replay")
        raw_scores = row["agent_ordinal_scores"]
        if not isinstance(raw_scores, Mapping) or set(map(str, raw_scores)) != set(order):
            raise ValueError("agent_ordinal_scores must contain exactly the configured class labels.")
        scores: dict[str, int] = {}
        for label in order:
            value = raw_scores[label]
            if isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 4:
                raise ValueError("agent_ordinal_scores must be ordinal integers from 0 through 4.")
            scores[label] = int(value)
        revision_type, action_type = str(row["revision_type"]).strip(), str(row["action_type"]).strip()
        route, language = str(row["route"]).strip(), str(row["language"]).strip()
        if not revision_type or not action_type or not route or not language:
            raise ValueError("revision_type, action_type, route, and language must be non-empty strings.")
        replay_performed = _bool(row["replay_performed"], "replay_performed")
        # Any revision needs an executable replay.  The unchanged/no-revision
        # case still supplies the post-replay field as the auditable baseline.
        if revision_type not in {"none", "no_revision", "unchanged"} and not replay_performed:
            raise ValueError("An Agent evidence revision requires Module A numerical replay before Module B.")
        incremental_declared = _bool(
            row["incremental_evidence_declared"], "incremental_evidence_declared"
        )
        consumed_by_module_a = _bool(
            row["evidence_consumed_by_module_a"], "evidence_consumed_by_module_a"
        )
        incremental_ids = _evidence_ids(row, "incremental_evidence_ids")
        consumed_ids = _evidence_ids(row, "consumed_evidence_ids")
        # Older callers that omit both ID fields can still submit a no-op
        # correction.  They cannot activate Module B on a bare boolean.
        if incremental_declared != bool(incremental_ids):
            raise ValueError(
                "incremental_evidence_declared must exactly match non-empty incremental_evidence_ids."
            )
        if consumed_by_module_a != bool(consumed_ids):
            raise ValueError(
                "evidence_consumed_by_module_a must exactly match non-empty consumed_evidence_ids."
            )
        overlap = sorted(set(incremental_ids) & set(consumed_ids))
        if overlap:
            raise ValueError(f"Incremental and Module A-consumed evidence IDs must be disjoint: {overlap}")
        return cls(
            module_a_pre_replay={label: float(pre[index]) for index, label in enumerate(order)},
            module_a_post_replay={label: float(post[index]) for index, label in enumerate(order)},
            agent_ordinal_scores=scores,
            agent_scores_validated=_bool(row["agent_scores_validated"], "agent_scores_validated"),
            eligible=_bool(row["eligible"], "eligible"),
            revision_type=revision_type,
            action_type=action_type,
            agreement=_bool(row["agreement"], "agreement"),
            evidence_coverage=_unit_interval(row["evidence_coverage"], "evidence_coverage"),
            evidence_reliability=_unit_interval(row["evidence_reliability"], "evidence_reliability"),
            confound_burden=_unit_interval(row["confound_burden"], "confound_burden"),
            route=route,
            language=language,
            ood=_unit_interval(row["ood"], "ood"),
            incremental_evidence_declared=incremental_declared,
            evidence_consumed_by_module_a=consumed_by_module_a,
            route_supported=_bool(row["route_supported"], "route_supported"),
            replay_performed=replay_performed,
            incremental_evidence_ids=incremental_ids,
            consumed_evidence_ids=consumed_ids,
            cross_fit_fold=row.get("cross_fit_fold"),
        )

    def to_dict(self, labels: Sequence[str]) -> dict[str, Any]:
        order = _ordered_labels(labels)
        return {
            "module_a_pre_replay": {label: float(self.module_a_pre_replay[label]) for label in order},
            "module_a_post_replay": {label: float(self.module_a_post_replay[label]) for label in order},
            "agent_ordinal_scores": {label: int(self.agent_ordinal_scores[label]) for label in order},
            "agent_scores_validated": self.agent_scores_validated,
            "eligible": self.eligible,
            "revision_type": self.revision_type,
            "action_type": self.action_type,
            "agreement": self.agreement,
            "evidence_coverage": self.evidence_coverage,
            "evidence_reliability": self.evidence_reliability,
            "confound_burden": self.confound_burden,
            "route": self.route,
            "language": self.language,
            "ood": self.ood,
            "incremental_evidence_declared": self.incremental_evidence_declared,
            "evidence_consumed_by_module_a": self.evidence_consumed_by_module_a,
            "incremental_evidence_ids": list(self.incremental_evidence_ids),
            "consumed_evidence_ids": list(self.consumed_evidence_ids),
            "route_supported": self.route_supported,
            "replay_performed": self.replay_performed,
            "cross_fit_fold": self.cross_fit_fold,
        }


@dataclass(frozen=True)
class ModuleBPrediction:
    """Auditable output of one conditional arbitration decision."""

    class_order: tuple[str, ...]
    probabilities: Mapping[str, float]
    module_a_post_replay: Mapping[str, float]
    additive_logit_correction: Mapping[str, float]
    additive_correction_applied: bool
    fallback: str | None
    authority: float
    reason: str
    incremental_evidence_ids: tuple[str, ...] = ()
    consumed_evidence_ids: tuple[str, ...] = ()

    @property
    def predicted_label(self) -> str:
        return self.class_order[int(np.argmax([self.probabilities[label] for label in self.class_order]))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULE_B_SCHEMA_VERSION,
            "class_order": list(self.class_order),
            "predicted_label": self.predicted_label,
            "probabilities": {label: float(self.probabilities[label]) for label in self.class_order},
            "module_a_post_replay": {label: float(self.module_a_post_replay[label]) for label in self.class_order},
            "additive_logit_correction": {
                label: float(self.additive_logit_correction[label]) for label in self.class_order
            },
            "additive_correction_applied": self.additive_correction_applied,
            "fallback": self.fallback,
            "authority": float(self.authority),
            "reason": self.reason,
            "incremental_evidence_ids": list(self.incremental_evidence_ids),
            "consumed_evidence_ids": list(self.consumed_evidence_ids),
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())


class ConditionalArbitrator:
    """Low-capacity residual calibrator, bound to cross-fitted development rows.

    The design matrix has no intercept.  It only contains centred Agent ordinal
    scores multiplied by an evidence-quality authority and a few predeclared
    revision/action/route/language interactions.  Therefore it cannot inject a generic
    class prior or move a case with no validated incremental Agent evidence.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        ridge_alpha: float = 8.0,
        max_logit_correction: float = 0.35,
        minimum_route_language_rows: int = 1,
        module_version: str = MODULE_B_VERSION,
    ) -> None:
        self.labels = _ordered_labels(labels)
        if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive.")
        if not np.isfinite(max_logit_correction) or max_logit_correction <= 0:
            raise ValueError("max_logit_correction must be finite and positive.")
        if int(minimum_route_language_rows) < 1:
            raise ValueError("minimum_route_language_rows must be at least one.")
        self.ridge_alpha = float(ridge_alpha)
        self.max_logit_correction = float(max_logit_correction)
        self.minimum_route_language_rows = int(minimum_route_language_rows)
        self.module_version = str(module_version)

    @staticmethod
    def _is_incremental(item: ConditionalArbitrationInput) -> bool:
        return bool(
            item.eligible
            and item.agent_scores_validated
            and item.incremental_evidence_declared
            and bool(item.incremental_evidence_ids)
            and not item.evidence_consumed_by_module_a
        )

    @staticmethod
    def _authority(item: ConditionalArbitrationInput) -> float:
        # OOD and confounds reduce authority; disagreement is preserved as a
        # review signal but receives a small, fixed conservatism factor.
        agreement_factor = 1.0 if item.agreement else 0.70
        return float(
            item.evidence_coverage
            * item.evidence_reliability
            * (1.0 - item.confound_burden)
            * (1.0 - item.ood)
            * agreement_factor
        )

    def _feature_names_for(
        self,
        *,
        revisions: Sequence[str],
        actions: Sequence[str],
        routes: Sequence[str],
        languages: Sequence[str],
    ) -> tuple[str, ...]:
        names = [f"score:{label}" for label in self.labels]
        names.extend(f"score:{label}|revision={revision}" for revision in revisions for label in self.labels)
        names.extend(f"score:{label}|action={action}" for action in actions for label in self.labels)
        names.extend(f"score:{label}|route={route}" for route in routes for label in self.labels)
        names.extend(f"score:{label}|language={language}" for language in languages for label in self.labels)
        return tuple(names)

    def _features(self, item: ConditionalArbitrationInput) -> np.ndarray:
        if not self._is_incremental(item):
            return np.zeros(len(self.feature_names_), dtype=float)
        score = np.asarray([item.agent_ordinal_scores[label] for label in self.labels], dtype=float)
        score -= score.mean()
        score *= self._authority(item)
        values: list[float] = list(score)
        values.extend(
            component
            for revision in self.revision_types_
            for component in (score if item.revision_type == revision else np.zeros(len(score)))
        )
        values.extend(
            component
            for action in self.actions_
            for component in (score if item.action_type == action else np.zeros(len(score)))
        )
        values.extend(
            component
            for route in self.routes_
            for component in (score if item.route == route else np.zeros(len(score)))
        )
        values.extend(
            component
            for language in self.languages_
            for component in (score if item.language == language else np.zeros(len(score)))
        )
        return np.asarray(values, dtype=float)

    def fit(
        self,
        rows: Iterable[Mapping[str, Any] | ConditionalArbitrationInput],
        y: Iterable[str] | None = None,
    ) -> "ConditionalArbitrator":
        """Fit only cross-fitted development predictions and their labels.

        Every mapping must include ``cross_fit_fold``.  A caller holding a
        final evaluation set cannot accidentally fit it without declaring an
        out-of-fold development provenance field.
        """

        raw_rows = list(rows)
        if not raw_rows:
            raise ValueError("At least one cross-fitted development row is required.")
        items = [
            ConditionalArbitrationInput.from_mapping(
                value.to_dict(self.labels) if isinstance(value, ConditionalArbitrationInput) else value,
                self.labels,
            )
            for value in raw_rows
        ]
        if any(item.cross_fit_fold is None or str(item.cross_fit_fold) == "" for item in items):
            raise ValueError("Module B fit requires cross_fit_fold on every development row.")
        if y is None:
            inferred = []
            for row in raw_rows:
                if isinstance(row, ConditionalArbitrationInput):
                    raise ValueError("y is required when fitting ConditionalArbitrationInput objects.")
                label = row.get("true_label", row.get("label"))
                if label is None:
                    raise ValueError("Development rows require true_label (or pass y explicitly).")
                inferred.append(str(label))
            target = np.asarray(inferred, dtype=object)
        else:
            target = np.asarray([str(value) for value in y], dtype=object)
        if len(target) != len(items) or set(target) - set(self.labels):
            raise ValueError("y must have one configured class label for each development row.")

        self.revision_types_ = tuple(sorted({item.revision_type for item in items}))
        self.actions_ = tuple(sorted({item.action_type for item in items}))
        self.routes_ = tuple(sorted({item.route for item in items}))
        self.languages_ = tuple(sorted({item.language for item in items}))
        self.feature_names_ = self._feature_names_for(
            revisions=self.revision_types_, actions=self.actions_, routes=self.routes_, languages=self.languages_
        )
        design = np.vstack([self._features(item) for item in items])
        one_hot = np.zeros((len(items), len(self.labels)), dtype=float)
        for index, label in enumerate(target):
            one_hot[index, self.labels.index(label)] = 1.0
        baseline = np.vstack([
            [item.module_a_post_replay[label] for label in self.labels] for item in items
        ])
        # Fitting residuals rather than raw classes stops Module B from learning
        # a second global classifier over Module A's already-calibrated prior.
        residual = one_hot - baseline
        gram = design.T @ design + self.ridge_alpha * np.eye(design.shape[1])
        self.coefficients_ = np.linalg.solve(gram, design.T @ residual)
        self.coefficients_ -= self.coefficients_.mean(axis=1, keepdims=True)
        pair_counts: dict[str, int] = {}
        for item in items:
            key = f"{item.route}\u0000{item.language}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        self.supported_route_languages_ = tuple(sorted(
            key for key, count in pair_counts.items() if count >= self.minimum_route_language_rows
        ))
        self.cross_fit_folds_ = tuple(sorted({str(item.cross_fit_fold) for item in items}))
        self.fitted_rows_ = len(items)
        return self

    def _require_fit(self) -> None:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("fit must be called before Module B prediction.")

    def predict_one(self, row: Mapping[str, Any] | ConditionalArbitrationInput) -> ModuleBPrediction:
        self._require_fit()
        item = ConditionalArbitrationInput.from_mapping(
            row.to_dict(self.labels) if isinstance(row, ConditionalArbitrationInput) else row,
            self.labels,
        )
        baseline = np.asarray([item.module_a_post_replay[label] for label in self.labels], dtype=float)
        zero = {label: 0.0 for label in self.labels}
        base_mapping = {label: float(baseline[index]) for index, label in enumerate(self.labels)}
        if not item.eligible:
            # Return the original float values, rather than an equivalent
            # softmax reconstruction: eligibility false has a bitwise identity
            # invariant with the replayed Module A probability vector.
            return ModuleBPrediction(self.labels, base_mapping, base_mapping, zero, False, _FALLBACK_POST_REPLAY, 0.0, "ineligible", item.incremental_evidence_ids, item.consumed_evidence_ids)
        pair = f"{item.route}\u0000{item.language}"
        if not item.route_supported or pair not in self.supported_route_languages_:
            return ModuleBPrediction(self.labels, base_mapping, base_mapping, zero, False, _FALLBACK_POST_REPLAY, 0.0, "unsupported_route_language", item.incremental_evidence_ids, item.consumed_evidence_ids)
        if not self._is_incremental(item):
            return ModuleBPrediction(self.labels, base_mapping, base_mapping, zero, False, None, 0.0, "no_validated_incremental_evidence", item.incremental_evidence_ids, item.consumed_evidence_ids)
        authority = self._authority(item)
        raw = self._features(item) @ self.coefficients_
        correction = np.asarray(raw, dtype=float)
        correction -= correction.mean()
        maximum = float(np.max(np.abs(correction)))
        if maximum > self.max_logit_correction:
            correction *= self.max_logit_correction / maximum
        correction -= correction.mean()  # Preserve the zero-sum logit invariant after floating point scaling.
        if not np.any(correction):
            return ModuleBPrediction(self.labels, base_mapping, base_mapping, zero, False, None, authority, "zero_fitted_correction", item.incremental_evidence_ids, item.consumed_evidence_ids)
        final = _softmax(np.log(np.clip(baseline, np.finfo(float).tiny, 1.0)) + correction)
        return ModuleBPrediction(
            self.labels,
            {label: float(final[index]) for index, label in enumerate(self.labels)},
            base_mapping,
            {label: float(correction[index]) for index, label in enumerate(self.labels)},
            True,
            None,
            authority,
            "validated_incremental_evidence",
            item.incremental_evidence_ids,
            item.consumed_evidence_ids,
        )

    def predict(self, rows: Iterable[Mapping[str, Any] | ConditionalArbitrationInput]) -> list[ModuleBPrediction]:
        return [self.predict_one(row) for row in rows]

    def predict_proba(self, rows: Iterable[Mapping[str, Any] | ConditionalArbitrationInput]) -> np.ndarray:
        predictions = self.predict(rows)
        return np.asarray([[item.probabilities[label] for label in self.labels] for item in predictions], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        return {
            "schema_version": MODULE_B_SCHEMA_VERSION,
            "module_version": self.module_version,
            "class_order": list(self.labels),
            "ridge_alpha": self.ridge_alpha,
            "max_logit_correction": self.max_logit_correction,
            "minimum_route_language_rows": self.minimum_route_language_rows,
            "revision_types": list(self.revision_types_),
            "actions": list(self.actions_),
            "routes": list(self.routes_),
            "languages": list(self.languages_),
            "feature_names": list(self.feature_names_),
            "coefficients": self.coefficients_.tolist(),
            "supported_route_languages": list(self.supported_route_languages_),
            "cross_fit_folds": list(self.cross_fit_folds_),
            "fitted_rows": self.fitted_rows_,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionalArbitrator":
        if payload.get("schema_version") != MODULE_B_SCHEMA_VERSION:
            raise ValueError("Unsupported Module B serialization schema.")
        result = cls(
            payload["class_order"],
            ridge_alpha=float(payload["ridge_alpha"]),
            max_logit_correction=float(payload["max_logit_correction"]),
            minimum_route_language_rows=int(payload["minimum_route_language_rows"]),
            module_version=str(payload["module_version"]),
        )
        result.revision_types_ = tuple(str(value) for value in payload["revision_types"])
        result.actions_ = tuple(str(value) for value in payload["actions"])
        result.routes_ = tuple(str(value) for value in payload["routes"])
        result.languages_ = tuple(str(value) for value in payload["languages"])
        result.feature_names_ = tuple(str(value) for value in payload["feature_names"])
        expected_names = result._feature_names_for(
            revisions=result.revision_types_, actions=result.actions_, routes=result.routes_, languages=result.languages_
        )
        if result.feature_names_ != expected_names:
            raise ValueError("Serialized Module B feature contract is inconsistent.")
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        if coefficients.shape != (len(result.feature_names_), len(result.labels)) or not np.isfinite(coefficients).all():
            raise ValueError("Serialized Module B coefficients have an invalid shape or values.")
        result.coefficients_ = coefficients
        result.supported_route_languages_ = tuple(sorted(str(value) for value in payload["supported_route_languages"]))
        result.cross_fit_folds_ = tuple(str(value) for value in payload["cross_fit_folds"])
        result.fitted_rows_ = int(payload["fitted_rows"])
        return result

    @classmethod
    def from_json(cls, value: str) -> "ConditionalArbitrator":
        return cls.from_dict(json.loads(value))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalArbitrator":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# Concise public aliases keep integration code readable without obscuring the
# conditional-arbitration boundary in the implementation above.
ModuleBConditionalArbitrator = ConditionalArbitrator
ConditionalModuleB = ConditionalArbitrator
