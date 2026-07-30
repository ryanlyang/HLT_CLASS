#!/usr/bin/env python3
"""Validate a campaign specification and the active clean source snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import validate_campaign_spec  # noqa: E402
from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.provenance import validate_campaign_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    print(
        json.dumps(
            {
                "campaign_spec_sha256": validate_campaign_spec(spec),
                "source_snapshot_contract_sha256": validate_campaign_source(
                    spec,
                    repository=args.repository,
                ),
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
