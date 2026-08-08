#!/usr/bin/env python3
"""Create the immutable exploratory comparison of all distinct PMARD models."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402
from hlt_classification.scouting.exploratory_test import (  # noqa: E402
    create_exploratory_test_spec, validate_exploratory_test_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sweep-root", type=Path, required=True)
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = args.output_root.resolve() / "exploratory_test_spec.json"
    if args.output.resolve() != expected:
        raise ValueError(
            "exploratory specification must be OUTPUT_ROOT/exploratory_test_spec.json"
        )
    spec = create_exploratory_test_spec(
        parent_sweep_root=args.parent_sweep_root,
        followup_root=args.followup_root,
        output_root=args.output_root,
        source_snapshot=capture_source_snapshot(REPO_ROOT, require_clean=True),
        project_dir=REPO_ROOT,
    )
    validate_exploratory_test_inputs(spec)
    write_immutable_json(args.output, spec)
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
