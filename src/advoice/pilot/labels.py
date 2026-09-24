"""Explicitly imported, scorer-only label records and JSONL helpers."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from .contracts import PilotContractError, SubjectRow


SCHEMA_VERSION = "advoice.pilot.v1"
_PURPOSES = frozenset({"scoring", "training"})
_FIELDS = frozenset({
    "schema_version", "subject", "source_label", "label", "label_mapping", "source",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


def _nonempty_string(value: Any, field: str) -> None:
    _require(type(value) is str and bool(value), f"{field} must be a non-empty string.")


def _purpose(purpose: Any) -> None:
    _require(type(purpose) is str and purpose in _PURPOSES,
             "Labels are restricted to scoring or training.")


def _strict_json_loads(value: str | bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            _require(key not in result, f"Duplicate JSON key: {key}.")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=unique_object)
    except PilotContractError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PilotContractError("Malformed label JSON.") from exc


def _label_identity(row: "LabelRow") -> tuple[str, str, str, str]:
    return (row.subject.dataset_id, row.subject.task, row.subject.subject_id, row.source)


@dataclass(frozen=True, kw_only=True)
class LabelRow:
    """A truth-bearing record that never crosses a provider-facing boundary."""

    subject: SubjectRow
    source_label: str
    label: str
    label_mapping: Mapping[str, str]
    source: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.schema_version == SCHEMA_VERSION, "Unsupported label schema version.")
        _require(type(self.subject) is SubjectRow, "Label subject must be a SubjectRow.")
        _nonempty_string(self.source_label, "source_label")
        _nonempty_string(self.label, "label")
        _nonempty_string(self.source, "source")
        _require(isinstance(self.label_mapping, Mapping) and bool(self.label_mapping),
                 "label_mapping must be non-empty.")
        _require(all(type(key) is str and bool(key) for key in self.label_mapping),
                 "label_mapping keys must be non-empty strings.")
        _require(all(type(value) is str and bool(value) for value in self.label_mapping.values()),
                 "label_mapping values must be non-empty strings.")
        _require(self.source_label in self.label_mapping,
                 "source_label must be present in label_mapping.")
        _require(self.label_mapping[self.source_label] == self.label,
                 "label must equal the mapped source label.")
        _require(self.label in self.subject.class_order,
                 "Label is unsupported for this task.")
        _require(all(value in self.subject.class_order for value in self.label_mapping.values()),
                 "Label mapping cannot fabricate an unsupported class.")
        object.__setattr__(self, "label_mapping", MappingProxyType(
            {key: self.label_mapping[key] for key in sorted(self.label_mapping)}
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_mapping": dict(self.label_mapping),
            "schema_version": self.schema_version,
            "source": self.source,
            "source_label": self.source_label,
            "subject": self.subject.to_dict(),
        }

    def to_json(self) -> str:
        try:
            return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise PilotContractError("Malformed label JSON.") from exc

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        return int(self.content_hash[:15], 16)

    def to_inference_dict(self) -> dict[str, Any]:
        raise PilotContractError("LabelRow is scorer/training-only.")

    def to_publishable_dict(self) -> dict[str, Any]:
        raise PilotContractError("LabelRow is scorer/training-only.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LabelRow":
        _require(isinstance(value, Mapping), "Expected a label record mapping.")
        _require(set(value) == _FIELDS, "Unknown or missing label record fields.")
        subject = value["subject"]
        if type(subject) is not SubjectRow:
            subject = SubjectRow.from_mapping(subject)
        return cls(
            subject=subject,
            source_label=value["source_label"],
            label=value["label"],
            label_mapping=value["label_mapping"],
            source=value["source"],
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "LabelRow":
        return cls.from_mapping(_strict_json_loads(value))


def serialize_labels_jsonl(rows: Iterable[LabelRow], *, purpose: str) -> str:
    _purpose(purpose)
    _require(not isinstance(rows, (str, bytes, Mapping)),
             "Rows must be an iterable of LabelRow records.")
    serialized: list[str] = []
    identities: set[tuple[str, str, str, str]] = set()
    try:
        iterator = iter(rows)
        for row in iterator:
            _require(type(row) is LabelRow, "Rows must contain only LabelRow records.")
            identity = _label_identity(row)
            _require(identity not in identities, "Duplicate dataset/task/subject/source label.")
            identities.add(identity)
            serialized.append(row.to_json())
    except PilotContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise PilotContractError("Malformed label rows.") from exc
    return "\n".join(serialized)


def read_labels_jsonl(lines: Iterable[str | bytes], *, purpose: str) -> tuple[LabelRow, ...]:
    _purpose(purpose)
    _require(not isinstance(lines, (str, bytes, Mapping)),
             "Lines must be an iterable of JSONL records.")
    rows: list[LabelRow] = []
    identities: set[tuple[str, str, str, str]] = set()
    try:
        iterator = iter(lines)
        for line in iterator:
            _require(type(line) in (str, bytes) and bool(line.strip()),
                     "Malformed JSONL label row.")
            row = LabelRow.from_json(line)
            identity = _label_identity(row)
            _require(identity not in identities, "Duplicate dataset/task/subject/source label.")
            identities.add(identity)
            rows.append(row)
    except PilotContractError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PilotContractError("Malformed label rows.") from exc
    return tuple(rows)


__all__ = ["LabelRow", "read_labels_jsonl", "serialize_labels_jsonl"]
