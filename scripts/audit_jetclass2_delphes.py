"""Freeze or verify a read-only JetClass2 snapshot; never train or submit jobs."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.inventory import build_inventory, verify_snapshot
from hlt_classification.jetclass2_delphes.splits import build_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-inventory", type=Path)
    parser.add_argument("--include-signal-file-qcd", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.verify_inventory:
        if args.output_root or args.include_signal_file_qcd:
            parser.error("Verification consumes the frozen policy; do not supply output/policy switches")
        verify_snapshot(args.data_root, load_json(args.verify_inventory))
        print("Snapshot hashes/cycles/schema: VERIFIED")
        return 0
    if args.output_root is None:
        parser.error("--output-root is required when freezing a snapshot")
    root, data = args.output_root.resolve(), args.data_root.resolve()
    if root.is_relative_to(data):
        parser.error("Audit output must be outside the read-only raw dataset")
    inventory = build_inventory(data, include_signal_file_qcd=args.include_signal_file_qcd)
    # Freeze valid inventory even if requested class coverage makes splits fail.
    write_immutable_json(root / "inventory.json", inventory)
    splits = build_splits(inventory, seed=args.split_seed)
    write_immutable_json(root / "splits.json", splits)
    print(f"Inventory: {inventory['content_hash']}")
    print(f"Selected rows: {sum(inventory['selected_class_counts']):,}")
    print(f"Role counts: {splits['role_counts']}")
    print("Status: provisional producer assumptions; final-test inference remains sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
