"""Create/submit/run the additive final-direct node-failure recovery."""
from pathlib import Path
import argparse
import importlib.util
import json
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    create = modes.add_parser("create")
    create.add_argument("--source-spec", type=Path, required=True)
    create.add_argument("--recovery-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for name in ("submit", "run"):
        command = modes.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        if name == "submit":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
        else:
            command.add_argument("--task", required=True)
    args = parser.parse_args()
    recovery = None if args.mode == "create" else json.loads(args.spec.read_text())
    source_path = args.source_spec if recovery is None else Path(recovery["source_spec"])
    source = json.loads(source_path.read_text())
    # Never import scientific packages from the new recovery checkout.
    sys.path.insert(0, str(Path(source["project_dir"]) / "src"))
    filename = ROOT / "src/hlt_classification/jetclass2_delphes/dzfix_fusion_recovery.py"
    module_spec = importlib.util.spec_from_file_location("dzfix_reviewed_recovery", filename)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if args.mode == "create":
        value = module.create(source_spec=source_path, recovery_root=args.recovery_root,
                              project=ROOT, source_commit=args.source_commit)
        print("Recovery:", value["recovery_root"])
        print("Three replacements: final-direct, aggregate, complete. Dry run only.")
        print("Retain fusion branch:", ", ".join(value["original_jobs"][n] for n in module.KEEP))
        print("Retire ONLY these summaries on live submission:",
              value["original_jobs"]["aggregate"], value["original_jobs"]["complete"])
        commands = json.loads((Path(value["recovery_root"]) / "command_plan.json").read_text())
        for row in commands["commands"]:
            print(shlex.join(row["command"]))
    elif args.mode == "submit":
        value = module.submit(recovery, execute=args.execute, authorization=args.authorization_phrase)
        for name, job in value["jobs"].items():
            print(job, name)
    else:
        value = module.run_task(recovery, args.task)
    print(value["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
