#!/usr/bin/env python3
"""Create, submit, or execute the staged learned-handoff autolaunch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_learned_autolaunch import (
    create_autolaunch, run, schedule,
)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("create")
    create.add_argument("--screen-spec", type=Path, required=True)
    create.add_argument("--screen-ledger", type=Path, required=True)
    create.add_argument("--screen-complete-job-id", required=True)
    create.add_argument("--data-root", type=Path, required=True)
    create.add_argument("--campaign-root", type=Path, required=True)
    create.add_argument("--launch-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    for mode in ("submit", "run"):
        command = commands.add_parser(mode)
        command.add_argument("--spec", type=Path, required=True)
        if mode == "submit":
            command.add_argument("--phase", choices=("after_screen", "after_gate"), required=True)
            command.add_argument("--execute", action="store_true")
            command.add_argument("--authorization-phrase")
        else:
            command.add_argument("--phase", choices=("after_screen", "after_gate"), required=True)
    return value


def main():
    arguments = parser().parse_args()
    if arguments.mode == "create":
        result = create_autolaunch(
            screen_spec_path=arguments.screen_spec,
            screen_ledger_path=arguments.screen_ledger,
            screen_complete_job_id=arguments.screen_complete_job_id,
            data_root=arguments.data_root,
            campaign_root=arguments.campaign_root,
            launch_root=arguments.launch_root,
            project=ROOT, source_commit=arguments.source_commit,
        )
    else:
        spec = load_json(arguments.spec)
        if arguments.mode == "submit":
            result = schedule(
                spec, phase=arguments.phase, execute=arguments.execute,
                authorization_phrase=arguments.authorization_phrase,
            )
        else:
            result = run(spec, phase=arguments.phase)
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
