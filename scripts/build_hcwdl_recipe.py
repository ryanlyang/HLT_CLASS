#!/usr/bin/env python3
"""Build an immutable HCWDL recipe from an explicit JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_recipe import build_recipe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HCWDL recipe payload must be a JSON object")
    write_immutable_json(args.output, build_recipe(payload, authorized=args.authorize))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
