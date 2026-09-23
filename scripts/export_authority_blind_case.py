"""Export one label-blind authority-review payload and its JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advoice.authority_review_runtime import (
    _blind_schema,
    _policy_documents,
    build_blind_payload,
)
from advoice.authority_study_dataset import AuthorityStudyDataset
from advoice.decision_lock import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection-salt", default="evidence-bound-v5-agent-smoke")
    args = parser.parse_args()

    dataset = AuthorityStudyDataset.from_artifact_dir(args.artifact_dir)
    selected = dataset.prepare_test_cases(
        max_cases=1,
        order="stable_hash",
        selection_salt=args.selection_salt,
    )[0]
    prepared = selected.prepared_case
    root = Path(__file__).resolve().parents[1]
    policy = _policy_documents(root, None)
    payload = build_blind_payload(
        prepared,
        policy_documents=policy,
        transcript=selected.transcript,
    )
    state_ids = sorted({str(card["state_id"]) for card in prepared.pre_state_cards})
    evidence_ids_by_state = {
        state_id: tuple(sorted(
            item.evidence_id
            for item in prepared.evidence
            if item.state_id == state_id and item.inference_permission
        ))
        for state_id in state_ids
    }
    bundle = {
        "dataset_id": dataset.advisor.dataset_id,
        "payload": json.loads(canonical_json(payload)),
        "response_schema": _blind_schema(
            prepared.route.target_route.labels,
            evidence_ids_by_state,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "passed",
        "dataset_id": dataset.advisor.dataset_id,
        "case_id": prepared.case_id,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
