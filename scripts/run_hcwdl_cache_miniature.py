#!/usr/bin/env python3
"""Exercise assignment-backed Shell Exact view construction in bounded RAM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.highcov_cache import DenseAssignmentStore  # noqa: E402
from hlt_classification.scouting.pmard_stream import iterate_pmard_batches  # noqa: E402
from hlt_classification.scouting.selective_assignment import RowSelection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation"), default="validation")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_json(args.split_manifest); selection_raw = load_json(args.selection_manifest)
    selection = RowSelection(
        selection_raw, role=args.role, split_manifest_sha256=split["content_hash"],
    )
    bound = min(args.rows, selection.rows)
    if bound <= 0:
        raise ValueError("HCWDL cache miniature row bound differs")
    store = DenseAssignmentStore(args.assignment_manifest)
    d0_rows = 0; d0_exact = True
    for batch in iterate_pmard_batches(
        split, data_root=args.data_root, role=args.role, matcher_model=None,
        alpha=0.0, repair_family="HIGHCOV_SHELL_EXACT/v1",
        matcher_variant="highcov_empirical_lexicographic_dr0p30_v1", threshold=0.0,
        max_rows=bound, max_rows_policy="stream_prefix",
        batch_size=min(256, bound), assignment_store=store,
        row_selection=selection, sampler_seed=1337, repair_seed=1337,
    ):
        d0_rows += len(batch["labels"])
        hlt = batch["hlt"]; d0 = batch["privileged"]
        d0_exact = d0_exact and all(
            getattr(hlt, name).tobytes() == getattr(d0, name).tobytes()
            for name in ("features", "vectors", "mask", "raw_lengths")
        )
    if d0_rows != bound:
        raise ValueError("HCWDL D0 cache miniature did not cover its bounded population")

    observed = 0; array_bytes = 0; endpoint_audits: list[dict[str, object]] = []
    for batch in iterate_pmard_batches(
        split, data_root=args.data_root, role=args.role, matcher_model=None,
        alpha=1.0, repair_family="HIGHCOV_SHELL_EXACT/v1",
        matcher_variant="highcov_empirical_lexicographic_dr0p30_v1", threshold=0.0,
        max_rows=bound, max_rows_policy="stream_prefix",
        batch_size=min(256, bound), assignment_store=store,
        row_selection=selection, sampler_seed=1337, repair_seed=1337,
        endpoint_audit_collector=endpoint_audits,
    ):
        observed += len(batch["labels"])
        view = batch["privileged"]
        array_bytes += sum(getattr(view, name).nbytes for name in (
            "features", "vectors", "mask", "raw_lengths",
        ))
    if observed != bound:
        raise ValueError("HCWDL cache miniature did not cover its bounded population")
    if not endpoint_audits or sum(int(row["rows"]) for row in endpoint_audits) != bound:
        raise ValueError("HCWDL endpoint audit did not cover its bounded population")
    invariants = {
        "d0_exact_hlt": d0_exact,
        "d100_assigned_exact_offline": all(
            bool(row["d100_assigned_exact_offline"]) for row in endpoint_audits
        ),
        "dustbins_exact_hlt": all(bool(row["dustbins_exact_hlt"]) for row in endpoint_audits),
        "hlt_skeleton_unchanged": all(
            bool(row["hlt_skeleton_unchanged"]) for row in endpoint_audits
        ),
        "all_21_fields_checked": all(
            bool(row["all_21_fields_checked"]) for row in endpoint_audits
        ),
    }
    if not all(invariants.values()):
        raise ValueError("HCWDL cache miniature endpoint invariant failed")
    report = with_content_hash({
        "contract": "HCWDL_CACHE_MINIATURE/v1", "schema_version": 1,
        "role": args.role, "rows": observed, "array_bytes": array_bytes,
        "assignment_manifest_sha256": store.manifest["content_hash"],
        "row_selection_sha256": selection_raw["content_hash"],
        "repair_family": "HIGHCOV_SHELL_EXACT/v1",
        "durable_repaired_dataset": False, "matcher_called": False,
        "endpoint_invariants": invariants,
        "matched_tokens_checked": sum(int(row["matched_tokens"]) for row in endpoint_audits),
        "dustbin_tokens_checked": sum(int(row["dustbin_tokens"]) for row in endpoint_audits),
        "d0_rows_checked": d0_rows,
    })
    write_immutable_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
