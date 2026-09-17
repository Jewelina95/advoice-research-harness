import json
from pathlib import Path

from jsonschema import Draft202012Validator

from advoice.evidence import MetricEvidenceV2


def test_metric_evidence_runtime_payload_matches_checked_in_schema() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "metric_evidence_v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    evidence = MetricEvidenceV2.from_mapping(
        {
            "evidence_id": "dataset:session:metric",
            "metric_id": "pause_rate",
            "subject_id": "subject-1",
            "session_id": "session-1",
            "case_id": "case-1",
            "metric_instance_id": "pause_rate:overall",
            "state_id": "S01",
            "value": 1.25,
            "reliability_components": {"source": 0.8},
            "consumed_by_supervised": True,
            "incremental_for_agent": False,
        }
    )

    Draft202012Validator(schema).validate(evidence.to_dict())
