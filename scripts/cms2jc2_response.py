#!/usr/bin/env python3
"""CMS2JC2 source preparation and explicitly provisional response development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    header = commands.add_parser("headers",help="CMS metadata only; no particle reads or split creation")
    header.add_argument("--cms-root",type=Path,required=True)
    header.add_argument("--output",type=Path,required=True)
    create = commands.add_parser("create-preparation",help="New, source-pinned, read-only SPORC audit")
    for name in ("campaign-root","cms-root","cms-split-manifest","jc2-root","jc2-inventory","jc2-profile"):
        create.add_argument("--"+name,type=Path,required=True)
    create.add_argument("--source-commit",required=True)
    submit = commands.add_parser("submit-preparation",help="Dry by default; does not submit scientific fits")
    submit.add_argument("--spec",type=Path,required=True)
    submit.add_argument("--execute",action="store_true")
    submit.add_argument("--authorization-phrase")
    run = commands.add_parser("run-preparation",help="Worker-only execution")
    run.add_argument("--spec",type=Path,required=True)
    ready = commands.add_parser("readiness",help="Print honest implementation and evidence boundaries")
    ready.add_argument("--compatibility-review",type=Path)
    ready.add_argument("--spec", type=Path, help="Validate a materialized stage; does not contact Slurm")
    commands.add_parser("graph", help="Inspect the registered 71-task primary-science design")
    acceptance = commands.add_parser("create-acceptance", help="Create the real CPU miniature/profile stage; does not submit")
    for name in ("preparation-spec", "compatibility", "campaign-root"):
        acceptance.add_argument("--"+name, type=Path, required=True)
    acceptance.add_argument("--source-commit", required=True)
    for stage in ("science", "confirmation"):
        p = commands.add_parser("create-"+stage, help="Materialize a separately authorized staged plan")
        p.add_argument("--previous-spec", type=Path, required=True)
        p.add_argument("--execution-lock", type=Path, required=True)
        if stage == "confirmation":
            p.add_argument("--selection-lock", type=Path, required=True)
    for name in ("dry-run", "submit", "run-task", "monitor", "results", "recover", "reconcile"):
        p = commands.add_parser(name)
        p.add_argument("--spec", type=Path, required=True)
        if name in {"run-task", "reconcile"}:
            p.add_argument("--task", required=True)
        if name == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--authorization-phrase")
            p.add_argument("--reviewed-plan-hash")
        if name == "recover":
            p.add_argument("--attempt", required=True)
            p.add_argument("--approved-job-ids", nargs="*", default=[])
        if name == "reconcile":
            p.add_argument("--job-id", required=True)
    assumptions = commands.add_parser("assumptions", help="Publish user-authorized provisional conventions")
    assumptions.add_argument("--cms-inventory", type=Path, required=True)
    assumptions.add_argument("--output", type=Path, required=True)
    preview = commands.add_parser("preview", help="Bounded local fit-role development check; not production evidence")
    preview.add_argument("--preparation-root", type=Path, required=True)
    preview.add_argument("--cms-root", type=Path, required=True)
    preview.add_argument("--compatibility-review", type=Path, required=True)
    preview.add_argument("--output-root", type=Path, required=True)
    preview.add_argument("--jets-per-role", type=int, default=32)
    preview.add_argument("--candidate", choices=("A_L", "B_L", "C_L"), default="A_L")
    args = parser.parse_args(argv)
    from hlt_classification.cms2jc2_response.contracts import load_json, publish, validate_compatibility
    if args.command == "create-acceptance":
        from hlt_classification.cms2jc2_response.campaign import create_acceptance
        result = create_acceptance(preparation_spec=args.preparation_spec, compatibility=args.compatibility,
                                   campaign_root=args.campaign_root, project_dir=ROOT, source_commit=args.source_commit)
    elif args.command in {"create-science", "create-confirmation"}:
        from hlt_classification.cms2jc2_response.campaign import create_followon
        result = create_followon(args.previous_spec, stage=args.command.removeprefix("create-"),
                                 execution_path=args.execution_lock, selection_path=getattr(args, "selection_lock", None))
    elif args.command in {"dry-run", "submit"}:
        from hlt_classification.cms2jc2_response.submission import submit
        result = submit(load_json(args.spec), execute=getattr(args, "execute", False),
                         authorization_phrase=getattr(args, "authorization_phrase", None),
                         reviewed_plan_hash=getattr(args, "reviewed_plan_hash", None))
    elif args.command == "run-task":
        from hlt_classification.cms2jc2_response.orchestration import run
        result = run(load_json(args.spec), args.task)
    elif args.command == "monitor":
        from hlt_classification.cms2jc2_response.submission import monitor
        result = monitor(load_json(args.spec))
    elif args.command == "recover":
        from hlt_classification.cms2jc2_response.submission import create_recovery
        result = create_recovery(args.spec, attempt=args.attempt, approved_job_ids=args.approved_job_ids)
    elif args.command == "reconcile":
        from hlt_classification.cms2jc2_response.submission import reconcile
        result = reconcile(load_json(args.spec), task_id=args.task, job_id=args.job_id)
    elif args.command == "results":
        from hlt_classification.cms2jc2_response.campaign import validate_spec
        from hlt_classification.cms2jc2_response.orchestration import product
        spec = load_json(args.spec)
        validate_spec(spec)
        result = dict(stage=spec["stage"], tasks=[])
        for task in spec["tasks"]:
            try:
                row = product(spec, task["task_id"], "examples" if task["action"].startswith("visual_") else "result")
                result["tasks"].append(dict(task=task["task_id"], report=row))
            except FileNotFoundError:
                result["tasks"].append(dict(task=task["task_id"], state="NO_AUTHENTICATED_OUTPUT"))
    elif args.command == "headers":
        from hlt_classification.cms2jc2_response.audit import inspect_headers
        if args.output.resolve().is_relative_to(args.cms_root.resolve()):
            raise PermissionError("Audit outputs cannot be written inside raw data")
        result = inspect_headers(args.cms_root)
        publish(args.output,result,"HEADER_AUDIT")
    elif args.command == "create-preparation":
        from hlt_classification.cms2jc2_response.preparation import create
        values = vars(args).copy(); values.pop("command")
        result = create(project_dir=ROOT,**values)
    elif args.command == "submit-preparation":
        from hlt_classification.cms2jc2_response.preparation import submit
        result = submit(load_json(args.spec),execute=args.execute,authorization_phrase=args.authorization_phrase)
    elif args.command == "run-preparation":
        from hlt_classification.cms2jc2_response.preparation import run
        result = run(load_json(args.spec))
    elif args.command == "assumptions":
        from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
        from hlt_classification.cms2jc2_response.audit import validate_inventory
        inventory = load_json(args.cms_inventory)
        validate_inventory(inventory)
        result = provisional_compatibility(inventory_hash=inventory["content_hash"])
        publish(args.output, result, "PROVISIONAL_COMPATIBILITY")
    elif args.command == "preview":
        from hlt_classification.cms2jc2_response.development import preview
        result = preview(project_dir=ROOT, preparation_root=args.preparation_root,
                         cms_root=args.cms_root, review_path=args.compatibility_review,
                         output_root=args.output_root, jets_per_role=args.jets_per_role,
                         candidate_id=args.candidate)
        result = {k: result[k] for k in ("contract", "content_hash", "elapsed_seconds", "location_counts",
                                        "residual_counts", "physical_status", "science_queue_ready")}
    elif args.command == "graph":
        from hlt_classification.cms2jc2_response.graph import science_graph
        result = dict(graph=science_graph(), executable=False, measured_resource_lock=False)
    else:
        result = dict(science_queue_ready=False, local_implementation_complete=True,
                      queue_tooling_ready=True, live_submission_stages=["preparation", "acceptance", "science", "confirmation"],
                      automatic_submission=False, remaining_implementation=[],
                      required_evidence=["verified review or explicit provisional assumption contract", "real SPORC CPU miniature",
                                         "measured full-size resources", "clean pushed source", "reviewed stage plan and exact authorization phrase"],
                      note="No remote acceptance is inferred from local tests. With no --spec, full-science readiness is not established.")
        if args.spec:
            from hlt_classification.cms2jc2_response.campaign import validate_spec, command_plan
            spec = load_json(args.spec)
            validate_spec(spec)
            result.update(stage=spec["stage"], spec_hash=spec["content_hash"],
                          plan_hash=command_plan(spec)["content_hash"], stage_ready_for_reviewed_submission=True,
                          science_queue_ready=spec["stage"] == "science")
        if args.compatibility_review:
            try:
                validate_compatibility(load_json(args.compatibility_review))
                from hlt_classification.cms2jc2_response.assumptions import physical_status
                result["physical_status"] = physical_status(load_json(args.compatibility_review))
                result["compatibility_verified"] = result["physical_status"]["producer_confirmed"]
            except (ValueError,PermissionError) as exc:
                result["compatibility_verified"] = False; result["compatibility_error"] = str(exc)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
