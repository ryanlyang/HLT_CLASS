#!/usr/bin/env python3
"""Create an immutable HCWDL-U-RKD smoke or 300k pilot candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (  # noqa: E402
    AUTHORIZATION_PHRASE, create_campaign,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (  # noqa: E402
    FIT_COUNT, TARGET_BANK_COUNT,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-homotopy-spec", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--representation-recipe", type=Path, required=True)
    parser.add_argument("--kernel-envelope", type=Path, required=True,
                        help="JSON mapping accepted by the v5 kernel loader")
    parser.add_argument("--architecture-attestation", type=Path, required=True)
    parser.add_argument("--numerical-acceptance", type=Path, required=True)
    parser.add_argument("--integration-attestation", type=Path, required=True)
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--authorize-live-submission", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    envelope = load_json(args.kernel_envelope)
    if "payload" in envelope and "token" not in envelope:
        envelope = envelope["payload"]
    spec = create_campaign(
        parent_homotopy_spec=args.parent_homotopy_spec,
        campaign_root=args.campaign_root, project_dir=args.project_dir,
        source_commit=args.source_commit,
        representation_recipe_path=args.representation_recipe,
        kernel_envelope=envelope,
        architecture_attestation_path=args.architecture_attestation,
        numerical_acceptance_path=args.numerical_acceptance,
        integration_attestation_path=args.integration_attestation,
        resource_profile_path=args.resource_profile,
        authorize_live_submission=args.authorize_live_submission,
        authorization_phrase=args.authorization_phrase,
    )
    print(json.dumps({
        "campaign_root": spec["campaign_root"], "mode": spec["mode"],
        "fit_count": FIT_COUNT, "target_bank_count": TARGET_BANK_COUNT,
        "task_count": len(spec["tasks"]),
        "authorized": spec["live_submission_authorized"],
        "creation_phrase": AUTHORIZATION_PHRASE,
        "content_hash": spec["content_hash"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
