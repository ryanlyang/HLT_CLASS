#!/usr/bin/env python3
"""Create an HCWDL-RKD planning spec or its reviewed, authorized live form."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, publish
from hlt_classification.scouting.hcwdl_representation_campaign import create_campaign_spec
from hlt_classification.scouting.hcwdl_representation_resources import artifact_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "production"), required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--checkpoint-namespace", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--parent-import-sha256", required=True)
    parser.add_argument("--representation-recipe-sha256", required=True)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--disposition-sha256", required=True)
    parser.add_argument(
        "--disposition",
        choices=("combined_confirmatory", "validation_only_parent_claim_consumed"),
        required=True,
    )
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--final-rows", type=int, required=True)
    parser.add_argument("--final-source-partitions", type=int, required=True)
    parser.add_argument("--combined-finalist-count", type=int, required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--planning-only", dest="planning_only", action="store_true",
        help="create the default non-executable planning specification",
    )
    mode_group.add_argument(
        "--authorized-executable", dest="planning_only", action="store_false",
        help=(
            "create the live form only after a strict candidate audit and explicit "
            "submission authorization exist"
        ),
    )
    parser.set_defaults(planning_only=True)
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--storage-estimate", type=Path)
    parser.add_argument("--fixed-size-inventory", type=Path)
    parser.add_argument("--tigris-acceptance", type=Path)
    parser.add_argument("--executable-candidate-audit", type=Path)
    parser.add_argument("--submission-authorization", type=Path)
    parser.add_argument("--source-manifest-path", type=Path)
    parser.add_argument("--split-manifest-path", type=Path)
    parser.add_argument("--parent-import-path", type=Path)
    parser.add_argument("--representation-graph-path", type=Path)
    parser.add_argument("--representation-recipe-path", type=Path)
    parser.add_argument("--final-disposition-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path_arguments = {
        "source_manifest": args.source_manifest_path,
        "split_manifest": args.split_manifest_path,
        "parent_import": args.parent_import_path,
        "representation_graph": args.representation_graph_path,
        "representation_recipe": args.representation_recipe_path,
        "final_disposition": args.final_disposition_path,
        "runtime_binding": args.runtime_binding,
    }
    artifact_paths = (
        None
        if all(value is None for value in path_arguments.values())
        else {
            name: (
                value if value is not None
                else args.campaign_root / {
                    "source_manifest": "inputs/source_manifest.json",
                    "split_manifest": "inputs/split_manifest.json",
                    "parent_import": "import/parent_import.json",
                    "representation_graph": "graph/ascent_graph.json",
                    "representation_recipe": "recipes/representation_recipe.json",
                    "final_disposition": "import/final_disposition.json",
                    "runtime_binding": "runtime/runtime_binding.json",
                }[name]
            )
            for name, value in path_arguments.items()
        }
    )
    runtime = None if args.runtime_binding is None else artifact(args.runtime_binding)
    spec = create_campaign_spec(
        mode=args.mode, campaign_root=args.campaign_root,
        checkpoint_namespace=args.checkpoint_namespace, project_dir=args.project_dir,
        source_commit=args.source_commit, source_manifest_sha256=args.source_manifest_sha256,
        split_manifest_sha256=args.split_manifest_sha256,
        parent_import_sha256=args.parent_import_sha256,
        representation_recipe_sha256=args.representation_recipe_sha256,
        graph_sha256=args.graph_sha256, disposition_sha256=args.disposition_sha256,
        disposition=args.disposition,
        role_counts={
            "train": args.train_rows, "validation": args.validation_rows,
            "final_test": args.final_rows,
        },
        final_source_partitions=args.final_source_partitions,
        combined_finalist_count=args.combined_finalist_count,
        planning_only=args.planning_only,
        executable_candidate_audit=(
            None if args.executable_candidate_audit is None
            else artifact(args.executable_candidate_audit)
        ),
        submission_authorization=(
            None if args.submission_authorization is None
            else artifact(args.submission_authorization)
        ),
        resource_profile=(
            None if args.resource_profile is None else artifact(args.resource_profile)
        ),
        storage_estimate=(
            None if args.storage_estimate is None else artifact(args.storage_estimate)
        ),
        fixed_size_inventory=(
            None
            if args.fixed_size_inventory is None
            else artifact_reference(args.fixed_size_inventory)
        ),
        tigris_acceptance=(
            None if args.tigris_acceptance is None else artifact(args.tigris_acceptance)
        ),
        artifact_paths=artifact_paths,
        runtime_binding_sha256=(None if runtime is None else runtime["content_hash"]),
    )
    publish(args.output, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
