#!/usr/bin/env python3
"""Publish a strict, non-submitting HCWDL-RKD executable-candidate audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_candidate import (
    build_executable_candidate_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-spec", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_executable_candidate_audit(
        planning_spec_path=args.planning_spec,
        command_plan_path=args.command_plan,
        runtime_binding_path=args.runtime_binding,
    )
    publish(args.output, audit)
    print(
        "Strict executable candidate reviewed without scheduler mutation: "
        f"{audit['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
