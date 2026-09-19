#!/usr/bin/env python3
"""Create, submit, or execute the staged dz-fix salience continuation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.dzfix_salience_continuation import (
    PHASES, create_continuation, run, schedule,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_subparsers(dest="mode", required=True)
    create = modes.add_parser("create")
    create.add_argument("--readiness-spec", type=Path, required=True)
    create.add_argument("--readiness-recovery-spec", type=Path, required=True)
    create.add_argument("--readiness-recovery-ledger", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    submit = modes.add_parser("submit")
    submit.add_argument("--spec", type=Path, required=True)
    submit.add_argument("--phase", choices=PHASES, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--authorization-phrase")
    worker = modes.add_parser("run")
    worker.add_argument("--spec", type=Path, required=True)
    worker.add_argument("--phase", choices=PHASES, required=True)
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.mode == "create":
        result = create_continuation(
            readiness_spec_path=arguments.readiness_spec,
            readiness_recovery_spec_path=arguments.readiness_recovery_spec,
            readiness_recovery_ledger_path=arguments.readiness_recovery_ledger,
            output_root=arguments.output_root,
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
