"""Typed, independent routing contracts for observations and prediction targets.

Routing is deliberately label-free.  An observation route describes what a
recording/task can measure; a target route describes what endpoint a model may
predict.  Keeping the two objects separate prevents a longitudinal endpoint
from being mistaken for an observation channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class RouteValidationError(ValueError):
    """Raised when case metadata cannot satisfy a requested route contract."""


@dataclass(frozen=True, slots=True)
class ObservationRoute:
    """The measurement channel selected for one case."""

    id: str
    family: str
    task_id: str | None = None
    language: str | None = None
    required_roles: tuple[str, ...] = ()
    allowed_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.family:
            raise ValueError("ObservationRoute requires non-empty id and family.")
        object.__setattr__(self, "required_roles", tuple(self.required_roles))
        object.__setattr__(self, "allowed_states", tuple(self.allowed_states))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "task_id": self.task_id,
            "language": self.language,
            "required_roles": list(self.required_roles),
            "allowed_states": list(self.allowed_states),
        }


@dataclass(frozen=True, slots=True)
class TargetRoute:
    """The prediction endpoint selected independently of observation routing."""

    id: str
    endpoint: str
    labels: tuple[str, ...]
    requires_paired_visits: bool = False
    minimum_visits: int = 1
    interval_field: str = "visit_interval_days"

    def __post_init__(self) -> None:
        if not self.id or not self.endpoint:
            raise ValueError("TargetRoute requires non-empty id and endpoint.")
        labels = tuple(self.labels)
        if not labels:
            raise ValueError("TargetRoute requires at least one target label.")
        if self.minimum_visits < 1:
            raise ValueError("minimum_visits must be positive.")
        if self.requires_paired_visits and self.minimum_visits < 2:
            object.__setattr__(self, "minimum_visits", 2)
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "endpoint": self.endpoint,
            "labels": list(self.labels),
            "requires_paired_visits": self.requires_paired_visits,
            "minimum_visits": self.minimum_visits,
            "interval_field": self.interval_field,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """A case's independent observation and target route selection."""

    observation_route: ObservationRoute
    target_route: TargetRoute
    visit_ids: tuple[str, ...] = ()
    interval_days: float | None = None
    validation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "visit_ids", tuple(self.visit_ids))
        object.__setattr__(self, "validation", tuple(self.validation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_route": self.observation_route.to_dict(),
            "target_route": self.target_route.to_dict(),
            "visit_ids": list(self.visit_ids),
            "interval_days": self.interval_days,
            "validation": list(self.validation),
        }


OBSERVATION_ROUTE_DEFAULTS: dict[str, dict[str, Any]] = {
    "clinical_interview": {"family": "structured_clinical_interview", "required_roles": ("patient", "interviewer")},
    "picture_description": {"family": "picture_description"},
    "picture_description_audio_only": {"family": "picture_description"},
    "structured_multitask": {"family": "structured_cognitive_multitask"},
    "structured_task_audio": {"family": "structured_cognitive_multitask"},
    "spontaneous_multilingual": {"family": "spontaneous_speech"},
    "public_speech": {"family": "public_speech"},
    "longitudinal_progression_audio": {"family": "picture_description"},
    "audio_only": {"family": "audio_only"},
}

TARGET_ROUTE_DEFAULTS: dict[str, dict[str, Any]] = {
    "cross_sectional_diagnosis": {
        "endpoint": "cross_sectional_diagnosis",
        "labels": ("HC", "MCI", "AD"),
    },
    "diagnosis": {
        "endpoint": "cross_sectional_diagnosis",
        "labels": ("HC", "MCI", "AD"),
    },
    "longitudinal_progression": {
        "endpoint": "longitudinal_progression",
        "labels": ("no_decline", "decline"),
        "requires_paired_visits": True,
        "minimum_visits": 2,
    },
    "progression": {
        "endpoint": "longitudinal_progression",
        "labels": ("no_decline", "decline"),
        "requires_paired_visits": True,
        "minimum_visits": 2,
    },
}


def _value(metadata: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return default


def resolve_observation_route(
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> ObservationRoute:
    """Resolve only measurement metadata; target fields are ignored."""

    channel = str(_value(metadata, "observation_route", "channel", "channel_profile", default="audio_only"))
    route_config = config or OBSERVATION_ROUTE_DEFAULTS
    if "observation_routes" in route_config:
        route_config = route_config["observation_routes"]  # type: ignore[assignment]
    configured = route_config.get(channel)
    if configured is None:
        raise RouteValidationError(f"Unknown observation route: {channel!r}.")
    task_id = _value(metadata, "task_id", "task", "task_type")
    family = str(configured.get("family", channel))
    if channel == "structured_task_audio" and task_id is not None:
        normalized_task = str(task_id).strip().lower()
        if "picture" in normalized_task and "description" in normalized_task:
            family = "picture_description"
    roles = _value(metadata, "required_roles", default=configured.get("required_roles", ()))
    states = _value(metadata, "allowed_states", default=configured.get("allowed_states", ()))
    return ObservationRoute(
        id=channel,
        family=family,
        task_id=str(task_id) if task_id is not None else None,
        language=str(_value(metadata, "language", default="unknown")),
        required_roles=tuple(str(value) for value in roles),
        allowed_states=tuple(str(value) for value in states),
    )


def resolve_target_route(
    metadata: Mapping[str, Any],
    *,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> TargetRoute:
    """Resolve only the prediction endpoint and its data requirements."""

    target = str(_value(metadata, "target_route", "target", "target_type", default="diagnosis"))
    route_config = config or TARGET_ROUTE_DEFAULTS
    if "target_routes" in route_config:
        route_config = route_config["target_routes"]  # type: ignore[assignment]
    configured = route_config.get(target)
    if configured is None:
        raise RouteValidationError(f"Unknown target route: {target!r}.")
    return TargetRoute(
        id=target,
        endpoint=str(configured.get("endpoint", target)),
        labels=tuple(str(value) for value in _value(metadata, "target_labels", "labels", default=configured.get("labels", ()))),
        requires_paired_visits=bool(configured.get("requires_paired_visits", False)),
        minimum_visits=int(configured.get("minimum_visits", 1)),
        interval_field=str(configured.get("interval_field", "visit_interval_days")),
    )


def _visit_ids(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _value(metadata, "visit_ids", "visits", default=())
    if isinstance(raw, Mapping):
        raw = list(raw)
    if isinstance(raw, str):
        raw = (raw,)
    values = []
    for value in raw:
        if isinstance(value, Mapping):
            value = _value(value, "visit_id", "id", "session_id")
        if value is not None:
            values.append(str(value))
    if not values:
        baseline = _value(metadata, "baseline_visit_id", "baseline_session_id")
        followup = _value(metadata, "followup_visit_id", "follow_up_visit_id", "followup_session_id")
        values = [str(value) for value in (baseline, followup) if value is not None]
    return tuple(dict.fromkeys(values))


def validate_target_observability(
    target_route: TargetRoute,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return deterministic validation messages for target-route requirements."""

    visits = _visit_ids(metadata)
    if target_route.requires_paired_visits and len(visits) < target_route.minimum_visits:
        raise RouteValidationError(
            f"Target route {target_route.id!r} requires at least "
            f"{target_route.minimum_visits} paired visits; received {len(visits)}."
        )
    interval = _value(metadata, target_route.interval_field, "interval_days", "visit_interval_days")
    if target_route.requires_paired_visits and interval is None:
        raise RouteValidationError(
            f"Target route {target_route.id!r} requires interval metadata."
        )
    if interval is not None:
        try:
            if float(interval) <= 0:
                raise RouteValidationError("Visit interval must be positive.")
        except (TypeError, ValueError) as exc:
            raise RouteValidationError("Visit interval must be numeric.") from exc
    return ()


def route_case(
    metadata: Mapping[str, Any],
    *,
    observation_config: Mapping[str, Mapping[str, Any]] | None = None,
    target_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> RouteDecision:
    """Build both routes while preserving their independent contracts."""

    observation = resolve_observation_route(metadata, config=observation_config)
    target = resolve_target_route(metadata, config=target_config)
    validate_target_observability(target, metadata)
    visits = _visit_ids(metadata)
    interval = _value(metadata, target.interval_field, "interval_days", "visit_interval_days")
    return RouteDecision(
        observation_route=observation,
        target_route=target,
        visit_ids=visits,
        interval_days=float(interval) if interval is not None else None,
    )


# Explicit aliases make the contract discoverable to callers that use the
# architecture vocabulary rather than the implementation name.
ObservationRouteContract = ObservationRoute
TargetRouteContract = TargetRoute
RoutingDecision = RouteDecision
resolve_routes = route_case
validate_target_route = validate_target_observability
