#!/usr/bin/env python3
"""Cancel only exact authenticated HCWDL-RKD job IDs, including recoveries."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_recovery import exact_cancellation_commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--recovery-ledger", type=Path, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commands = exact_cancellation_commands(
        artifact(args.submission_ledger), [artifact(path) for path in args.recovery_ledger]
    )
    for command in commands:
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
