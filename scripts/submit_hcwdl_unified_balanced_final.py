#!/usr/bin/env python3
"""Print or submit the one exact authorized HCWDL-UB sealed-test job."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (  # noqa: E402
    validate_execution_lock, validate_finalist_lock, validate_foundation_spec,
)


PHRASE = "SUBMIT HCWDL UB SEALED FINAL TEST EXACT LOCK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--foundation-spec", type=Path, required=True)
    parser.add_argument("--finalist-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    foundation = load_json(args.foundation_spec)
    validate_foundation_spec(foundation)
    finalist = load_json(args.finalist_lock)
    finalist_hash = validate_finalist_lock(finalist)
    execution = load_json(args.execution_lock)
    validate_execution_lock(execution)
    if execution.get("authorized") is not True or execution.get("finalist_lock_sha256") != finalist_hash:
        raise PermissionError("HCWDL-UB final execution/finalist locks differ")
    if args.project_dir.resolve() != Path(foundation["project_dir"]).resolve():
        raise PermissionError("HCWDL-UB final project directory differs")
    if (
        execution.get("source_commit") != foundation["source_commit"]
        or execution.get("split_manifest_sha256") != foundation["parents"]["split_manifest_sha256"]
        or execution.get("selection_manifest_sha256") != foundation["parents"]["selection_manifest_sha256"]
    ):
        raise PermissionError("HCWDL-UB final execution data/source lineage differs")
    command = [
        "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
        "--cpus-per-task=8", "--mem=96G", "--time=06:00:00",
        "--gres=gpu:gh200:1", "--job-name=hcwub_sealed_final",
        "--export=ALL," + ",".join((
            f"PROJECT_DIR={args.project_dir.resolve()}",
            f"HCWDL_UB_FOUNDATION_SPEC={args.foundation_spec.resolve()}",
            f"HCWDL_UB_FINALIST_LOCK={args.finalist_lock.resolve()}",
            f"HCWDL_UB_EXECUTION_LOCK={args.execution_lock.resolve()}",
            f"HCWDL_UB_FINAL_OUTPUT={args.output.resolve()}",
        )),
        str(args.project_dir.resolve() / "sbatch/run_hcwdl_unified_balanced_final.sh"),
    ]
    print(" ".join(command))
    if args.execute:
        if args.authorization_phrase != PHRASE:
            raise PermissionError("HCWDL-UB final submission phrase differs")
        validate_source_checkout(
            args.project_dir.resolve(), expected_commit=foundation["source_commit"],
        )
        if args.output.exists():
            raise FileExistsError("HCWDL-UB final output already exists")
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip()
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
