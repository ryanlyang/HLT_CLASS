"""Create/dry-run or explicitly submit the SPORC matching + A100 readiness gate only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.readiness import create_readiness, submit_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    create = modes.add_parser("create")
    for name in ("inventory", "split-profile", "data-root", "output-root"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for name, default in (("cpus", 8), ("workers", 8), ("memory-mb", 81920),
                          ("array-concurrency", 16), ("assignment-minutes", 120),
                          ("profile-minutes", 240), ("max-train-minutes", 2880)):
        create.add_argument("--" + name, type=int, default=default)
    submit = modes.add_parser("submit")
    submit.add_argument("--spec", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--authorization-phrase")
    args = parser.parse_args()
    if args.mode == "create":
        spec = create_readiness(inventory=load_json(args.inventory), split_profile=load_json(args.split_profile),
                                data_root=args.data_root, output_root=args.output_root, project=ROOT,
                                source_commit=args.source_commit, **{key: getattr(args, key) for key in (
                                    "cpus", "workers", "memory_mb", "array_concurrency", "assignment_minutes",
                                    "profile_minutes", "max_train_minutes")})
        print("Readiness spec:", Path(spec["readiness_root"]) / "readiness_spec.json")
        print("Split:", spec["foundation"]["splits"]["profile"], spec["foundation"]["splits"]["role_counts"])
        print("Resources (unmeasured initial request):", spec["resources"])
        print("Dry run created. No jobs submitted. Gate ends after real-A100 profiling; scientific fits = 0.")
    else:
        spec = load_json(args.spec)
        if args.spec.resolve() != Path(spec["readiness_root"]).resolve() / "readiness_spec.json":
            parser.error("Use the canonical readiness_spec.json path")
        ledger = submit_readiness(spec, execute=args.execute, authorization_phrase=args.authorization_phrase)
        for task, job in ledger["jobs"].items():
            print(f"{task:<12} {job}")
        print("Dry run:" if ledger["dry_run"] else "Readiness queued:", ledger["content_hash"])
        print("No scientific campaign submitted. No other jobs changed.")


if __name__ == "__main__":
    main()
