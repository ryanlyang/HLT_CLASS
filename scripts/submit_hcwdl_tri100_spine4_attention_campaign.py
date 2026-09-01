#!/usr/bin/env python3
"""Dry-run or submit the gated persistent-HLT attention four-spine DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_exact_dag_submission import (  # noqa: E402
    submit_exact_dag,
)
from hlt_classification.scouting.hcwdl_recovery import (  # noqa: E402
    build_submission_ledger, task_attestation_path,
    validate_submission_ledger, validate_task_attestation,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_campaign import (  # noqa: E402
    GATE_SUBMISSION_PHRASE, SCIENCE_SUBMISSION_PHRASE, validate_campaign,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_contracts import (  # noqa: E402
    PLAN_CONTRACT, validate_artifact,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_attention_execution import (  # noqa: E402
    validate_attention_execution_acceptance, validate_parameter_lock,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_persistent_support import (  # noqa: E402
    validate_support_audit,
)


GATE_TASKS = ("authenticate", "support_audit", "preflight")


def _validated_gate(spec: Mapping[str, object]) -> dict[str, object]:
    root = Path(str(spec["campaign_root"]))
    ledger = load_json(spec["artifact_paths"]["gate_submission_ledger"])
    validate_submission_ledger(ledger)
    if (
        ledger.get("campaign_spec_sha256") != spec["content_hash"]
        or ledger.get("dry_run") is not False
        or set(ledger.get("jobs", {})) != set(GATE_TASKS)
    ):
        raise PermissionError("attention gate ledger differs")
    for task_id in GATE_TASKS:
        attestation = load_json(task_attestation_path(root, task_id, None))
        validate_task_attestation(
            attestation,
            campaign_spec_sha256=str(spec["content_hash"]),
            task_id=task_id,
            array_index=None,
        )
    validate_support_audit(
        load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
    )
    validate_parameter_lock(
        load_json(spec["artifact_paths"]["parameter_lock"]), spec=spec,
    )
    validate_attention_execution_acceptance(
        load_json(spec["artifact_paths"]["execution_acceptance"]), spec=spec,
    )
    return ledger


def _publish_combined_ledger(
    spec: Mapping[str, object], gate: Mapping[str, object],
    science: Mapping[str, object],
) -> None:
    jobs = {**gate["jobs"], **science["jobs"]}
    commands = {**gate["commands"], **science["commands"]}
    if len(jobs) != 61 or len(commands) != 61:
        raise RuntimeError("attention combined ledger coverage differs")
    combined = build_submission_ledger(
        campaign_spec_sha256=str(spec["content_hash"]),
        jobs=jobs,
        commands=commands,
        dry_run=False,
    )
    destination = Path(spec["artifact_paths"]["submission_ledger"])
    if destination.exists():
        if load_json(destination) != combined:
            raise FileExistsError("attention combined ledger differs")
    else:
        write_immutable_json(destination, combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("gate", "science"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()

    spec = load_json(args.spec)
    validate_campaign(spec, executable=args.execute)
    root = Path(spec["campaign_root"])
    plan_path = Path(spec["artifact_paths"][f"{args.phase}_command_plan"])
    plan = load_json(plan_path)
    validate_artifact(plan, contract=PLAN_CONTRACT)
    canonical_dry_run = root / f"dry_run_{args.phase}_submission_ledger.json"
    canonical_live = Path(
        spec["artifact_paths"][f"{args.phase}_submission_ledger"]
    )
    expected_output = canonical_live if args.execute else canonical_dry_run
    if args.output.resolve() != expected_output.resolve():
        raise ValueError(
            f"attention {args.phase} output must be {expected_output}"
        )
    expected_phrase = (
        GATE_SUBMISSION_PHRASE
        if args.phase == "gate" else SCIENCE_SUBMISSION_PHRASE
    )
    if args.execute:
        if args.authorization_phrase != expected_phrase:
            raise PermissionError("attention four-spine submission phrase differs")
        if ROOT.resolve() != Path(spec["project_dir"]).resolve():
            raise PermissionError(
                "attention four-spine submitter is outside bound worktree"
            )
        validate_source_checkout(ROOT, expected_commit=spec["source_commit"])

    gate = None
    if args.phase == "science" and args.execute:
        gate = _validated_gate(spec)
    ledger = submit_exact_dag(
        identity=spec["content_hash"],
        plan=plan,
        output=args.output,
        canonical_dry_run=canonical_dry_run,
        execute=args.execute,
    )
    if args.phase == "science" and args.execute:
        _publish_combined_ledger(spec, gate, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
