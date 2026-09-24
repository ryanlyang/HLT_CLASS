#!/usr/bin/env python3
"""Explicit, independently submitted CPU response development stages."""
from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response import dev_campaign as campaign, dev_submission as submission


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("create")
    for name in ("project-dir", "source-commit", "preparation-spec", "root"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--partition", default="debug")
    p = commands.add_parser("stage")
    p.add_argument("--study", required=True)
    p.add_argument("--stage", choices=("pilot", "confirm", "compare"), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--parent-spec")
    p.add_argument("--policy", choices=tuple(campaign.POLICIES))
    p.add_argument("--b-threads", type=int, choices=(1, 16), default=1)
    for cpus, partition in ((64, "debug"), (36, "tier3")):
        p = commands.add_parser(f"create-compare{cpus}")
        for name in ("parent-spec", "project-dir", "source-commit", "root"):
            p.add_argument("--"+name, required=True)
        p.add_argument("--partition", default=partition)
    for name in ("dry-run", "submit", "run-task", "monitor", "results", "reconcile"):
        p = commands.add_parser(name)
        p.add_argument("--spec", required=True)
        if name == "submit":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--authorization-phrase")
            p.add_argument("--reviewed-plan-hash")
        if name in ("run-task", "reconcile"):
            p.add_argument("--task", required=True)
        if name == "reconcile":
            p.add_argument("--job-id", required=True)
    a = parser.parse_args()
    if a.command == "create":
        result = campaign.create_study(project_dir=a.project_dir, source_commit=a.source_commit,
                                      preparation_spec=a.preparation_spec, root=a.root, partition=a.partition)
    elif a.command == "create-compare64":
        from hlt_classification.cms2jc2_response.dev_restart import create_compare64
        result = create_compare64(parent_spec=a.parent_spec, project_dir=a.project_dir,
                                 source_commit=a.source_commit, root=a.root, partition=a.partition)
    elif a.command == "create-compare36":
        from hlt_classification.cms2jc2_response.dev_restart import create_compare36
        result = create_compare36(parent_spec=a.parent_spec, project_dir=a.project_dir,
                                 source_commit=a.source_commit, root=a.root, partition=a.partition)
    elif a.command == "stage":
        result = campaign.create_stage(a.study, stage=a.stage, name=a.name, parent_spec=a.parent_spec,
                                       policy_id=a.policy, b_threads=a.b_threads)
    else:
        spec = load_json(Path(a.spec))
        if a.command in ("dry-run", "submit"):
            result = submission.submit(spec, execute=getattr(a, "execute", False),
                authorization_phrase=getattr(a, "authorization_phrase", None),
                reviewed_plan_hash=getattr(a, "reviewed_plan_hash", None))
        elif a.command == "run-task":
            from hlt_classification.cms2jc2_response.dev_worker import run
            result = run(spec, a.task)
        elif a.command == "reconcile":
            result = submission.reconcile(spec, task_id=a.task, job_id=a.job_id)
        else:
            result = getattr(submission, a.command)(spec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
