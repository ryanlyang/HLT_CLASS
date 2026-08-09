#!/usr/bin/env python3
"""Create the exact supplemental 300k dense cold HCWDL specification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_dense import (  # noqa: E402
    DENSE_AUTHORIZATION_PHRASE, create_dense_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() != (args.campaign_root / "campaign_spec.json").resolve():
        raise PermissionError("dense cold spec output must be canonical under campaign root")
    if REPO_ROOT.resolve() != args.project_dir.resolve():
        raise PermissionError("dense cold spec must bind the checkout creating it")
    validate_source_checkout(REPO_ROOT, expected_commit=args.source_commit)
    phrase = args.authorization_phrase if args.authorize_live_submission else None
    if not args.authorize_live_submission and args.authorization_phrase is not None:
        raise ValueError("dense cold authorization phrase supplied without live authorization")
    spec = create_dense_spec(
        parent_campaign_spec=args.parent_campaign_spec,
        campaign_root=args.campaign_root,
        project_dir=args.project_dir,
        source_commit=args.source_commit,
        authorization_phrase=phrase,
    )
    write_immutable_json(args.output, spec)
    print(f"Dense graph: {spec['graph_sha256']}")
    print(f"Authorized: {spec['live_submission_authorized']}")
    print(f"Creation phrase: {DENSE_AUTHORIZATION_PHRASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
