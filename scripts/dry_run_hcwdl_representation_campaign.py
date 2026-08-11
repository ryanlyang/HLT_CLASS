#!/usr/bin/env python3
"""Validate all bound rows and publish a nonmutating HCWDL-RKD dry run."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import (
    build_command_plan, validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_runtime_rows import (
    build_runtime_dry_run_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.campaign_spec)
    validate_campaign_spec(spec, executable=False)
    plan = build_command_plan(spec)
    audit = build_runtime_dry_run_audit(
        spec, artifact(args.runtime_binding), plan,
    )
    publish(args.output, plan)
    publish(args.audit_output, audit)
    for row in plan["commands"]:
        print(row["task_key"] + ": " + " ".join(row["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
