#!/usr/bin/env python3
"""Build an explicitly authorized, bounded, nonfinal HCWDL-RKD bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import publish
from hlt_classification.scouting.hcwdl_representation_bootstrap import (
    BOOTSTRAP_AUTHORIZATION_PHRASE,
    SAFE_BOOTSTRAP_TASK_PREFIX,
    build_acceptance_bootstrap,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-spec", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--ordinary-worker", type=Path, required=True)
    parser.add_argument("--deterministic-worker", type=Path, required=True)
    parser.add_argument(
        "--through-task", choices=SAFE_BOOTSTRAP_TASK_PREFIX, required=True,
        help="authorize the dependency-closed safe prefix through this task",
    )
    parser.add_argument(
        "--authorization-phrase", required=True,
        help=f"must equal exactly: {BOOTSTRAP_AUTHORIZATION_PHRASE}",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stop = SAFE_BOOTSTRAP_TASK_PREFIX.index(args.through_task) + 1
    value = build_acceptance_bootstrap(
        planning_spec_path=args.planning_spec,
        runtime_binding_path=args.runtime_binding,
        ordinary_worker_path=args.ordinary_worker,
        deterministic_worker_path=args.deterministic_worker,
        authorized_tasks=SAFE_BOOTSTRAP_TASK_PREFIX[:stop],
        authorization_phrase=args.authorization_phrase,
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
