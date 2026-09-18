"""Select/inspect a capacity-safe partial snapshot from a frozen inventory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.partial_snapshot import (
    build_partial_snapshot_plan,
    validate_partial_snapshot_destination,
    validate_partial_snapshot_plan,
    verify_inventory_source_checksums,
)
from hlt_classification.jetclass2_delphes.inventory import verify_snapshot


def _print(plan: dict) -> None:
    print(f"Plan: {plan['content_hash']}")
    print(f"Files: {plan['selected_file_count']:,}")
    print(f"Bytes: {plan['selected_size_bytes']:,}")
    print(f"Selected rows: {plan['selected_rows']:,}")
    print(f"Projected role counts: {plan['projected_role_counts']}")
    print(f"Required with headroom: {plan['required_role_capacity']}")
    print("Final test accessed: False (scalar inventory metadata only)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("create")
    create.add_argument("--inventory", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--headroom-fraction", type=float, default=0.05)
    create.add_argument("--split-seed", type=int, default=20260910)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--inventory", type=Path, required=True)
    inspect.add_argument("--plan", type=Path, required=True)
    verify = sub.add_parser("verify-destination")
    verify.add_argument("--parent-inventory", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--destination-inventory", type=Path, required=True)
    verify.add_argument("--destination-splits", type=Path, required=True)
    verify.add_argument("--data-root", type=Path, required=True)
    source = sub.add_parser("verify-source-manifest")
    source.add_argument("--inventory", type=Path, required=True)
    source.add_argument("--sha256sum", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        inventory = load_json(args.inventory)
        if args.output_root.exists():
            parser.error("Output root already exists; plans are immutable")
        plan = build_partial_snapshot_plan(
            inventory,
            headroom_fraction=args.headroom_fraction,
            split_seed=args.split_seed,
        )
        write_immutable_json(args.output_root / "partial_snapshot_plan.json", plan)
        (args.output_root / "selected_files.txt").write_text(
            "".join(path + "\n" for path in plan["selected_paths"]),
            encoding="utf-8",
            newline="\n",
        )
    elif args.mode == "inspect":
        inventory = load_json(args.inventory)
        plan = load_json(args.plan)
        validate_partial_snapshot_plan(plan, inventory)
    elif args.mode == "verify-destination":
        plan = load_json(args.plan)
        destination_inventory = load_json(args.destination_inventory)
        destination_splits = load_json(args.destination_splits)
        validate_partial_snapshot_destination(
            plan,
            load_json(args.parent_inventory),
            destination_inventory,
            destination_splits,
        )
        verify_snapshot(args.data_root, destination_inventory)
        print("Destination bytes, inventory, split, and planned subset: VERIFIED")
    else:
        inventory = load_json(args.inventory)
        verify_inventory_source_checksums(
            inventory,
            args.sha256sum.read_text(encoding="ascii").splitlines(),
        )
        print("Local inventory paths and hashes match the source checksum manifest: VERIFIED")
        return 0
    _print(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
