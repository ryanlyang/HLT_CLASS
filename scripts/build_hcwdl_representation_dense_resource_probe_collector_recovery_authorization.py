#!/usr/bin/env python3
"""Bind one replacement dense accounting collector to completed probe jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    build_dense_resource_probe_collector_recovery_authorization,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--original-ledger", type=Path, required=True)
    parser.add_argument("--failed-collector-log", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = artifact(args.plan)
    root = Path(str(plan["collector"]["output_root"])).resolve().parents[1]
    expected_output = (
        root / "review"
        / "dense_resource_probe_collector_recovery_authorization.json"
    )
    if args.output.resolve() != expected_output:
        raise PermissionError("dense collector-recovery authorization route differs")
    publish(args.output, build_dense_resource_probe_collector_recovery_authorization(
        plan=plan, authorization=artifact(args.authorization),
        ledger=artifact(args.original_ledger), ledger_path=args.original_ledger,
        failed_collector_log=artifact_reference(args.failed_collector_log),
        authorization_phrase=args.authorization_phrase,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
