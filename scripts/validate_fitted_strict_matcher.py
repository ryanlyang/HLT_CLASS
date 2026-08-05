#!/usr/bin/env python3
"""Authenticate and summarize the vendored fitted_strict matcher bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.scouting.fitted_strict import (  # noqa: E402
    ConstituentMatcher, FITTED_STRICT_ARTIFACT_DIR, fitted_strict_artifact_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=FITTED_STRICT_ARTIFACT_DIR)
    args = parser.parse_args()
    matcher = ConstituentMatcher.from_artifacts(
        args.artifact_dir / "fitted_edge_model.json",
        args.artifact_dir / "confidence_models.json",
        independent_audit_json=args.artifact_dir / "independent_validation.json",
    )
    print(json.dumps(fitted_strict_artifact_report(matcher), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
