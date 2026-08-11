#!/usr/bin/env python3
"""Validate and print the complete nonmutating HCWDL-U-RKD command plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import build_command_plan, validate_campaign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign(spec, executable=False)
    plan = build_command_plan(spec)
    for row in plan["commands"]:
        print(row["task_id"] + ": " + " ".join(row["command"]))
    print(f"Validated {len(plan['commands'])} nonmutating task commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
