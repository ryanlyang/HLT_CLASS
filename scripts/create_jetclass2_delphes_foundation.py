"""Create isolated Delphes preparation spec; publish no jobs or model results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.foundation import build_foundation_spec
from hlt_classification.jetclass2_delphes.split_registry import select_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--splits", type=Path, help="Explicit profile JSON or legacy non-production reservoirs")
    group.add_argument("--split-registry", type=Path)
    p.add_argument("--profile", help="Explicit registry profile, e.g. TRAIN_500K")
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    if bool(a.split_registry) != bool(a.profile):
        p.error("--split-registry and --profile must be supplied together")
    inventory = load_json(a.inventory)
    splits = select_profile(load_json(a.split_registry), inventory, a.profile) if a.split_registry else load_json(a.splits)
    spec = build_foundation_spec(inventory, splits)
    write_immutable_json(a.output_root / "foundation_spec.json", spec)
    print(f"Foundation spec: {spec['content_hash']}")
    print(f"Capacity: {spec['inputs']['capacity']} (no truncation)")
    print(f"Profile: {splits.get('profile', 'LEGACY_FULL_RESERVOIR_NOT_PRODUCTION')}")
    print(f"Role counts: {splits['role_counts']}")
    print(f"Train/validation assignment shards: {len(spec['assignment_tasks'])}")
    print("No jobs submitted; run the bounded audit before full preparation.")


if __name__ == "__main__":
    main()
