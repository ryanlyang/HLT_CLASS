"""Freeze/inspect/verify reusable 500k/1M/1.5M/2M profiles. No jobs or model access."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.split_registry import (
    build_registry, publish_registry, select_profile, validate_registry,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--reservoir-splits", type=Path, required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--step-size", type=int, default=100_000)
    for mode in ("inspect", "verify"):
        q = sub.add_parser(mode)
        q.add_argument("--inventory", type=Path, required=True)
        q.add_argument("--registry", type=Path, required=True)
        if mode == "verify":
            q.add_argument("--data-root", type=Path, required=True)
            q.add_argument("--step-size", type=int, default=100_000)
    a = p.parse_args()
    inventory = load_json(a.inventory)
    if a.mode == "build":
        if a.output_root.exists() or a.output_root.resolve().is_relative_to(a.data_root.resolve()):
            p.error("Choose a new output directory outside the raw dataset")
        registry = build_registry(a.data_root, inventory, load_json(a.reservoir_splits), step_size=a.step_size)
        publish_registry(a.output_root, registry, inventory)
    else:
        registry = load_json(a.registry)
        validate_registry(registry, inventory)
        # Authenticate every exported profile, not only the central registry.
        for item in registry["design"]["profiles"]:
            exported = load_json(a.registry.parent / "profiles" / f"{item['name']}.json")
            if exported != select_profile(registry, inventory, item["name"]):
                raise ValueError("Exported profile differs from registry")
        if a.mode == "verify":
            design = registry["design"]
            replay = build_registry(a.data_root, inventory, registry["reservoirs"],
                                    training_sizes=tuple(x["train_rows"] for x in design["profiles"]),
                                    validation_size=design["evaluation_rows"]["validation"],
                                    test_size=design["evaluation_rows"]["final_test"],
                                    seed=design["seed"], step_size=a.step_size)
            # Producer source provenance may legitimately differ on replay;
            # scientific design and exact membership must be identical.
            if replay["design"] != design or replay["memberships"] != registry["memberships"]:
                raise ValueError("Registry differs from scalar-metadata selection replay")
            print("METADATA REPLAY: PASS")
    print(f"Registry: {registry['content_hash']}")
    print(f"{'profile':<16} {'train':>12} {'validation':>12} {'final test':>12}")
    for item in registry["design"]["profiles"]:
        print(f"{item['name']:<16} {item['train_rows']:>12,} "
              f"{registry['design']['evaluation_rows']['validation']:>12,} "
              f"{registry['design']['evaluation_rows']['final_test']:>12,}")
    for role in ("validation", "final_test"):
        print(f"Shared {role} membership: {registry['memberships'][role]['content_hash']}")
    print("Final-test particles/predictions accessed: False. No matching, training or submission performed.")


if __name__ == "__main__":
    main()
