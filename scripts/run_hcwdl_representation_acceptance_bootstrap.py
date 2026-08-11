#!/usr/bin/env python3
"""Execute one exact task under a bounded nonfinal HCWDL-RKD bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_bootstrap import (
    load_acceptance_bootstrap_context,
    validate_acceptance_bootstrap_task,
)
from hlt_classification.scouting.hcwdl_representation_task_runtime import (
    execute_registered_task,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--deterministic-worker", action="store_true")
    args = parser.parse_args()
    bootstrap = artifact(args.bootstrap_spec)
    planning, _ = load_acceptance_bootstrap_context(bootstrap)
    validate_acceptance_bootstrap_task(
        bootstrap,
        planning_spec=planning,
        task_key=args.task,
        deterministic_worker=args.deterministic_worker,
    )
    execute_registered_task(
        spec=planning,
        task_key=args.task,
        array_index=None,
        deterministic_worker=args.deterministic_worker,
        acceptance_bootstrap=bootstrap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
