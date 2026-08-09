#!/usr/bin/env python3
"""Create a hashed HCWDL smoke, pilot, named-midscale, or production spec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import (  # noqa: E402
    ENDPOINT_CONTINUATION_MODES, EXECUTION_SCOPES, FULL_CAMPAIGN_SCOPE,
    validate_source_checkout,
)
from hlt_classification.scouting.hcwdl_campaign import MODES, create_campaign_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--project-dir", default=str(REPO_ROOT))
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--resource-measurement-sha256")
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--production-authorization-sha256")
    parser.add_argument("--submission-authorization", type=Path)
    parser.add_argument(
        "--endpoint-continuation", choices=ENDPOINT_CONTINUATION_MODES,
        default="manual_posthoc",
    )
    parser.add_argument(
        "--execution-scope", choices=EXECUTION_SCOPES,
        default=FULL_CAMPAIGN_SCOPE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.planning_only:
        if REPO_ROOT.resolve() != Path(args.project_dir).resolve():
            raise PermissionError(
                "executable HCWDL campaign must bind the checkout creating it"
            )
        validate_source_checkout(REPO_ROOT, expected_commit=args.source_commit)
    source = load_json(args.source_manifest); split = load_json(args.split_manifest)
    recipe = None if args.recipe is None else load_json(args.recipe)
    recipe_hash = None if recipe is None else recipe["content_hash"]
    resource_profile = None if args.resource_profile is None else load_json(args.resource_profile)
    submission_authorization = (
        None if args.submission_authorization is None else load_json(args.submission_authorization)
    )
    counts = {role: int(split["roles"][role]["file_count"]) for role in ("train", "validation", "final_test")}
    spec = create_campaign_spec(
        mode=args.mode, campaign_root=args.campaign_root,
        source_manifest_sha256=source["content_hash"],
        split_manifest_sha256=split["content_hash"], source_commit=args.source_commit,
        role_source_counts=counts, recipe_sha256=recipe_hash,
        recipe_path=args.recipe,
        planning_only=args.planning_only, source_manifest_path=args.source_manifest,
        split_manifest_path=args.split_manifest, data_root=args.data_root,
        project_dir=args.project_dir,
        live_submission_authorized=args.authorize_live_submission,
        resource_measurement_sha256=args.resource_measurement_sha256,
        resource_profile=resource_profile,
        production_authorization_sha256=args.production_authorization_sha256,
        submission_authorization=submission_authorization,
        include_label_only_warm_continuation=(
            False if recipe is None
            else bool(recipe["controls"]["include_label_only_warm_continuation"])
        ),
        endpoint_continuation=args.endpoint_continuation,
        execution_scope=args.execution_scope,
    )
    write_immutable_json(args.output, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
