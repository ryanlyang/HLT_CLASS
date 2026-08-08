#!/usr/bin/env python3
"""Publish the explicit, lineage-bound HCWDL endpoint diagnostic acknowledgement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_qualification import (  # noqa: E402
    DIAGNOSTIC_ACK_PHRASE, QUALIFIERS, build_diagnostic_acknowledgement,
)


def _named_path(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("qualifier report must be NAME=PATH") from error
    if name not in QUALIFIERS or not raw_path:
        raise argparse.ArgumentTypeError("qualifier report name or path differs")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Required exact phrase: {DIAGNOSTIC_ACK_PHRASE}",
    )
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--cache-miniature", type=Path, required=True)
    parser.add_argument(
        "--qualifier-report", action="append", type=_named_path, required=True,
        help="Repeat exactly once for each of T0,TFS,THC,TSOFT,TSHELL,TOFF as NAME=PATH.",
    )
    parser.add_argument("--acknowledgement-phrase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports: dict[str, str] = {}
    for name, path in args.qualifier_report:
        if name in reports:
            raise ValueError(f"duplicate HCWDL qualifier report {name}")
        reports[name] = load_json(path)["content_hash"]
    acknowledgement = build_diagnostic_acknowledgement(
        campaign_spec_sha256=load_json(args.campaign_spec)["content_hash"],
        assignment_manifest_sha256=load_json(args.assignment_manifest)["content_hash"],
        recipe_sha256=load_json(args.recipe)["content_hash"],
        cache_miniature_sha256=load_json(args.cache_miniature)["content_hash"],
        qualifier_report_sha256=reports,
        acknowledgement_phrase=args.acknowledgement_phrase,
    )
    write_immutable_json(args.output, acknowledgement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
