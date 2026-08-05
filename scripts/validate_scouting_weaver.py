#!/usr/bin/env python3
"""Run authoritative installed-Weaver FP32 parity for the Scouting ParT."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.models.scouting_particle_transformer import validate_scouting_weaver_fp32_parity  # noqa: E402
from hlt_classification.scouting.config_contracts import validate_vendored_preprocessing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parity = validate_scouting_weaver_fp32_parity(device=args.device)
    if parity["passed"] is not True: raise RuntimeError("installed-Weaver Scouting parity failed")
    report = with_content_hash({
        "contract": "hlt_classification_scouting_weaver_acceptance_v1", "schema_version": 1,
        "parity": parity, "preprocessing": validate_vendored_preprocessing(args.repository),
    })
    write_immutable_json(args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
