#!/usr/bin/env python3
"""Resume only authenticated absent HCWDL-RKD outputs under a chained ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish, scheduler
from hlt_classification.scouting.hcwdl_representation_recovery import (
    build_recovery_plan, build_recovery_submission_ledger,
)
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec, validate_command_plan, validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    load_runtime_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--recovery-ledger", type=Path, action="append", default=[])
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--recovery-plan-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = artifact(args.campaign_spec)
    command_plan = artifact(args.command_plan)
    original_ledger = artifact(args.submission_ledger)
    validate_campaign_spec(spec, executable=args.execute)
    validate_command_plan(command_plan, spec=spec)
    validate_submission_ledger(
        original_ledger, spec=spec, command_plan=command_plan,
    )
    runtime_binding = load_runtime_binding(spec)
    plan = build_recovery_plan(
        monitor_report=artifact(args.monitor_report),
        spec=spec, runtime_binding=runtime_binding,
    )
    publish(args.recovery_plan_output, plan)
    if not args.execute:
        return 0
    prior = [artifact(path) for path in args.recovery_ledger]
    ledger = build_recovery_submission_ledger(
        recovery_plan=plan, command_plan=command_plan,
        original_ledger=original_ledger, prior_recovery_ledgers=prior,
        spec=spec, runtime_binding=runtime_binding,
        scheduler=scheduler, execute=True,
    )
    publish(args.ledger_output, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
