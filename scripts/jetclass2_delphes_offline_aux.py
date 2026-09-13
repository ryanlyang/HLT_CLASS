"""Isolated 500k offline-auxiliary study: prepare, audit and explicitly submit stages."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from hlt_classification.jetclass2_delphes.offline_aux.contracts import load_json
    from hlt_classification.jetclass2_delphes.offline_aux import campaign, submission, recovery
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    init = sub.add_parser("create")
    for name in ("root", "project-dir"):
        init.add_argument("--"+name, type=Path, required=True)
    for name in ("data-root", "inventory", "profile", "readiness-spec"):
        init.add_argument("--"+name, type=Path)
    init.add_argument("--source-commit", required=True)
    continuation = sub.add_parser("continue-debug", help="Reuse completed CPU preparation; profile only on debug")
    for name in ("source-study", "root", "project-dir"):
        continuation.add_argument("--"+name, type=Path, required=True)
    continuation.add_argument("--source-commit", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--study", type=Path, required=True)
    stage.add_argument("--name", choices=campaign.STAGES, required=True)
    for name in ("dry-run", "submit", "monitor", "recover", "run", "results"):
        q = sub.add_parser(name)
        q.add_argument("--stage", type=Path, required=True)
        if name in {"dry-run", "submit", "monitor", "recover", "run"}:
            q.add_argument("--attempt", default="initial")
        if name == "submit":
            q.add_argument("--authorization-phrase", required=True)
        if name == "run":
            q.add_argument("--task", required=True)
    a = p.parse_args()
    if a.action == "create":
        if a.readiness_spec is not None:
            if a.inventory is not None or a.profile is not None:
                p.error("Use either --readiness-spec or --inventory/--profile, not both")
            from hlt_classification.jetclass2_delphes.contracts import validate as validate_native
            readiness = load_json(a.readiness_spec)
            validate_native(readiness, "READINESS_SPEC")
            inventory, profile = readiness["foundation"]["inventory"], readiness["foundation"]["splits"]
            data_root = a.data_root or Path(readiness["data_root"])
        else:
            if any(v is None for v in (a.inventory, a.profile, a.data_root)):
                p.error("Supply --readiness-spec, or all of --inventory --profile --data-root")
            inventory, profile, data_root = load_json(a.inventory), load_json(a.profile), a.data_root
        result = campaign.create_study(root=a.root, data_root=data_root, inventory=inventory,
            profile=profile, project_dir=a.project_dir, source_commit=a.source_commit)
    elif a.action == "continue-debug":
        from hlt_classification.jetclass2_delphes.offline_aux.preparation_import import create_debug_continuation
        result = create_debug_continuation(source_study=a.source_study, root=a.root,
            project_dir=a.project_dir, source_commit=a.source_commit)
    elif a.action == "stage":
        result = campaign.create_stage(load_json(a.study), a.name)
    else:
        stage = load_json(a.stage); study = load_json(stage["study_spec_path"])
        attempt = Path(stage["root"]) / "attempts" / getattr(a, "attempt", "initial")
        if a.action in {"dry-run", "submit"}:
            result = submission.submit(stage, study, attempt, execute=a.action == "submit",
                                       phrase=getattr(a, "authorization_phrase", None))
        elif a.action == "recover":
            result = recovery.prepare(stage, study, attempt)
        elif a.action == "monitor":
            result = submission.monitor(stage, load_json(attempt / "submission_ledger.json"))
        elif a.action == "run":
            from hlt_classification.jetclass2_delphes.offline_aux.workflow import run_task
            result = run_task(stage, a.task, attempt)
        else:
            result = {}
            for task in stage["tasks"]:
                item = campaign.completed(stage, task["task_id"])
                if item is None:
                    result[task["task_id"]] = "PENDING"
                elif task["kind"] == "train":
                    report = item[0]["report"]
                    result[task["task_id"]] = dict(selected_pass=report["selected_pass"],
                        completed_passes=report["completed_passes"], validation=report["validation"])
                elif task["kind"] in {"aggregate", "configuration", "complete", "profile"}:
                    result[task["task_id"]] = item[0]
                else:
                    result[task["task_id"]] = "COMPLETE"
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
