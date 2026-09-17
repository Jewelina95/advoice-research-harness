"""Replay saved fusion inputs without training, labels, or provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from advoice.authority_joint_fusion import AuthorityJointFusionConfig, fuse_authority_joint


def replay_audit(source: Path, *, channel: str) -> dict:
    payload = source.read_bytes()
    original = json.loads(payload)
    records = []
    skipped = []
    for case in original["cases"]:
        if case.get("status") != "completed":
            skipped.append(case["case_id"])
            continue
        old = case["fusion"]
        result = fuse_authority_joint(
            case["frozen"]["probabilities"],
            case["pre_state"]["probabilities"],
            case["post_state"]["probabilities"],
            old["blind_ordinal_scores"],
            class_order=case["prepared"]["class_order"],
            config=AuthorityJointFusionConfig(**old["config"]),
            channel=channel,
            provenance=old.get("provenance", {}),
        )
        records.append({
            "case_id": case["case_id"],
            "previous_audit_hash": old["audit_hash"],
            "previous_probabilities": old["probabilities"],
            "previous_predicted_label": old["predicted_label"],
            "class_changed": old["predicted_label"] != result.predicted_label,
            "replayed_fusion": result.to_dict(),
        })
    return {
        "schema_version": "advoice.offline_fusion_replay.v1",
        "purpose": "Engineering replay of previously inspected cases; not a fresh performance estimate.",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_study_hash": original.get("study_hash"),
        "channel": channel,
        "provider_calls": 0,
        "training_performed": False,
        "labels_consumed": False,
        "case_count": len(records),
        "skipped_case_ids": skipped,
        "class_changed_count": sum(row["class_changed"] for row in records),
        "state_agent_conflict_count": sum(row["replayed_fusion"]["state_agent_conflict"] for row in records),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing case_audit.json")
    parser.add_argument("--channel", required=True, help="Original observation channel; never guessed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay_audit(args.source, channel=args.channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve every original audit and refuse accidental replacement.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
