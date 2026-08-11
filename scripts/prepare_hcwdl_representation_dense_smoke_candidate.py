#!/usr/bin/env python3
"""Prepare the exact dense smoke candidate; never invoke the scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _hcwdl_representation_common import REPO_ROOT
from hlt_classification.scouting.hcwdl_representation_dense_preparation import (
    prepare_dense_smoke_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--compatible-resource-profile", type=Path, required=True)
    parser.add_argument("--storage-estimate", type=Path, required=True)
    parser.add_argument("--storage-template", type=Path, required=True)
    parser.add_argument("--dense-teacher-import", type=Path, required=True)
    parser.add_argument("--representation-graph", type=Path, required=True)
    parser.add_argument("--representation-recipe", type=Path, required=True)
    parser.add_argument("--dense-disposition", type=Path, required=True)
    parser.add_argument("--tap-schema", type=Path, required=True)
    parser.add_argument("--surface-parity", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-row-selection", type=Path, required=True)
    parser.add_argument("--train-assignment-manifest", type=Path, required=True)
    parser.add_argument("--validation-assignment-manifest", type=Path, required=True)
    parser.add_argument("--historical-campaign-spec", type=Path, required=True)
    parser.add_argument("--historical-recipe", type=Path, required=True)
    parser.add_argument("--historical-project-dir", type=Path, required=True)
    parser.add_argument("--runtime-measurement-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_dense_smoke_candidate(
        representation_root=args.representation_root,
        project_dir=args.project_dir,
        compatible_resource_profile_path=args.compatible_resource_profile,
        storage_estimate_path=args.storage_estimate,
        storage_template_path=args.storage_template,
        dense_teacher_import_path=args.dense_teacher_import,
        representation_graph_path=args.representation_graph,
        representation_recipe_path=args.representation_recipe,
        dense_disposition_path=args.dense_disposition,
        tap_schema_path=args.tap_schema, surface_parity_path=args.surface_parity,
        source_manifest_path=args.source_manifest,
        split_manifest_path=args.split_manifest,
        train_row_selection_path=args.train_row_selection,
        train_assignment_manifest_path=args.train_assignment_manifest,
        validation_assignment_manifest_path=args.validation_assignment_manifest,
        historical_campaign_spec_path=args.historical_campaign_spec,
        historical_recipe_path=args.historical_recipe,
        historical_project_dir=args.historical_project_dir,
        runtime_measurement_root=args.runtime_measurement_root,
        data_root=args.data_root,
    )
    print("DENSE_SMOKE_CANDIDATE_READY")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
