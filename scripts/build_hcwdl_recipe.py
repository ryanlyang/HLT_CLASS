#!/usr/bin/env python3
"""Build an immutable HCWDL recipe from an explicit JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_recipe import CLASS_WEIGHT_POLICY, build_recipe  # noqa: E402
from hlt_classification.scouting.selective_assignment import validate_row_selection  # noqa: E402
from hlt_classification.scouting.training import sqrt_inverse_class_weights  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument(
        "--train-row-selection", type=Path,
        help="Required for an authorized recipe; supplies authenticated train counts and lineage.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HCWDL recipe payload must be a JSON object")
    if args.authorize and args.train_row_selection is None:
        raise PermissionError("authorized HCWDL recipe requires the train row selection")
    if args.train_row_selection is not None:
        selection = load_json(args.train_row_selection)
        selection_hash = validate_row_selection(
            selection, split_manifest_sha256=selection["split_manifest_sha256"],
        )
        train = selection.get("roles", {}).get("train")
        if not isinstance(train, dict):
            raise ValueError("HCWDL row selection lacks train counts")
        counts = [int(value) for value in train["class_counts"]]
        weights = sqrt_inverse_class_weights(counts)
        payload["class_weighting"] = {
            "policy": CLASS_WEIGHT_POLICY,
            "train_class_counts": counts,
            "train_row_selection_sha256": selection_hash,
        }
        payload["class_weights"] = np.asarray(weights, np.float32).tolist()
        evidence = dict(payload.get("evidence", {}))
        previous = evidence.get("train_row_selection")
        if previous is not None and previous != selection_hash:
            raise ValueError("recipe payload names a different train row selection")
        evidence["train_row_selection"] = selection_hash
        payload["evidence"] = evidence
    write_immutable_json(args.output, build_recipe(payload, authorized=args.authorize))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
