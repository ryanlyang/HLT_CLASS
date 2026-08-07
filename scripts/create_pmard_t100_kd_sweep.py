#!/usr/bin/env python3
"""Create an immutable supplemental T100 KD weight/temperature sweep spec."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.provenance import capture_source_snapshot  # noqa: E402
from hlt_classification.scouting.kd_sweep import (  # noqa: E402
    create_t100_sweep_spec, validate_t100_sweep_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = args.output_root.resolve() / "sweep_spec.json"
    if args.output.resolve() != expected:
        raise ValueError("sweep specification must be OUTPUT_ROOT/sweep_spec.json")
    spec = create_t100_sweep_spec(
        parent_campaign_root=args.parent_campaign_root,
        output_root=args.output_root,
        source_snapshot=capture_source_snapshot(REPO_ROOT, require_clean=True),
        project_dir=REPO_ROOT,
    )
    validate_t100_sweep_inputs(spec)
    write_immutable_json(args.output, spec)
    print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
