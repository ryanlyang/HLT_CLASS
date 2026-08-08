#!/usr/bin/env python3
"""Build the explicit user authorization artifact for a future exact HCWDL spec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import build_submission_authorization  # noqa: E402
from hlt_classification.scouting.hcwdl_campaign import validate_campaign_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    validate_campaign_spec(spec, executable=False)
    if (
        spec.get("planning_only") is not False
        or spec.get("recipe_status") != "locked"
        or spec.get("resource_profile_status") not in {
            "measured_prelaunch_candidate", "smoke_test_only",
        }
        or spec.get("live_submission_authorized") is not False
    ):
        raise PermissionError(
            "authorization requires a locked smoke or measured prelaunch candidate spec"
        )
    result = build_submission_authorization(
        mode=spec["mode"], source_commit=spec["source_commit"],
        source_manifest_sha256=spec["source_manifest_sha256"],
        split_manifest_sha256=spec["split_manifest_sha256"],
        recipe_sha256=spec["recipe_sha256"],
        resource_request_sha256=spec["resource_request_sha256"],
        command_plan_sha256=spec["command_plan_sha256"],
        authorization_phrase=args.authorization_phrase,
        production_authorization_sha256=spec.get("production_authorization_sha256"),
    )
    write_immutable_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
