#!/usr/bin/env python3
"""Build bounded HCWDL-RKD action inputs and their non-final authority."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE,
    build_nonfinal_acceptance_action_inputs,
    build_nonfinal_acceptance_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--acceptance-bootstrap", type=Path, required=True)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-import", type=Path, required=True)
    parser.add_argument("--parent-loss-attestation", type=Path, required=True)
    parser.add_argument("--representation-recipe", type=Path, required=True)
    parser.add_argument("--ordinary-worker", type=Path, required=True)
    parser.add_argument("--deterministic-worker", type=Path, required=True)
    parser.add_argument(
        "--derived-root",
        type=Path,
        required=True,
        help="directory for canonically derived bounded action inputs",
    )
    parser.add_argument("--action-inputs-output", type=Path, required=True)
    parser.add_argument(
        "--authorization-phrase",
        required=True,
        help=f"must equal exactly: {NONFINAL_ACCEPTANCE_AUTHORIZATION_PHRASE}",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    derived_root = args.derived_root.resolve()
    if args.action_inputs_output.resolve() != derived_root / "action_inputs.json":
        raise PermissionError("non-final action inputs require their canonical route")
    if args.output.resolve() != derived_root / "authority.json":
        raise PermissionError("non-final authority requires its canonical route")

    action_inputs = build_nonfinal_acceptance_action_inputs(
        acceptance_bootstrap_path=args.acceptance_bootstrap,
        representation_recipe_path=args.representation_recipe,
        derived_root=derived_root,
    )
    publish(args.action_inputs_output, action_inputs)
    authority = build_nonfinal_acceptance_authority(
        project_dir=args.project_dir,
        acceptance_bootstrap_path=args.acceptance_bootstrap,
        action_inputs_path=args.action_inputs_output,
        parent_campaign_spec_path=args.parent_campaign_spec,
        parent_recipe_path=args.parent_recipe,
        parent_import_path=args.parent_import,
        parent_loss_attestation_path=args.parent_loss_attestation,
        representation_recipe_path=args.representation_recipe,
        ordinary_worker_path=args.ordinary_worker,
        deterministic_worker_path=args.deterministic_worker,
        authorization_phrase=args.authorization_phrase,
    )
    publish(args.output, authority)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
