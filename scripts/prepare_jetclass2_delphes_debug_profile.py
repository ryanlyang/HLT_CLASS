"""One debug profile job reusing a completed readiness foundation; no science submission."""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.profile_attempt import (
    create_profile_attempt, profile_attempt_plan, run_profile_attempt, submit_profile_attempt,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    create = modes.add_parser("create")
    create.add_argument("--readiness-spec", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for name, default in (("cpus", 8), ("workers", 8), ("memory-mb", 73728),
                          ("profile-minutes", 240), ("max-train-minutes", 2880)):
        create.add_argument("--" + name, type=int, default=default)
    submit = modes.add_parser("submit")
    submit.add_argument("--spec", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--authorization-phrase")
    run = modes.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        spec = create_profile_attempt(readiness_spec=args.readiness_spec, output_root=args.output_root,
                                      project=ROOT, source_commit=args.source_commit,
                                      **{k: getattr(args, k) for k in (
                                          "cpus", "workers", "memory_mb", "profile_minutes", "max_train_minutes")})
        print("Profile spec:", Path(spec["attempt_root"]) / "profile_attempt_spec.json")
        print(shlex.join(profile_attempt_plan(spec)["commands"][0]["command"]))
        print("Dry run: ONE debug profile job; zero assignment jobs; zero scientific fits.")
    else:
        spec = load_json(args.spec)
        if (args.spec.resolve() != Path(spec["attempt_root"]).resolve() / "profile_attempt_spec.json"
                or ROOT != Path(spec["project_dir"]).resolve()):
            parser.error("Use the canonical attempt spec and its pinned worktree")
        if args.mode == "run":
            result = run_profile_attempt(spec)
            print("Runtime profile:", result["content_hash"])
        else:
            result = submit_profile_attempt(spec, execute=args.execute, authorization_phrase=args.authorization_phrase)
            print("Dry run" if result["dry_run"] else "Queued debug profile:", result["jobs"]["profile"])
    print("Existing foundation and jobs unchanged. Scientific campaign remains separately gated on tier3.")


if __name__ == "__main__":
    main()
