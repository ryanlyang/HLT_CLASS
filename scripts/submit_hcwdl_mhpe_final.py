#!/usr/bin/env python3
"""Render or submit the one separately locked HCWDL-MHPE final-test job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_mhpe_contracts import (  # noqa: E402
    FINALIST_LOCK_CONTRACT, validate_execution_lock,
)
from hlt_classification.data.cache_contracts import validate_content_hash  # noqa: E402

PHRASE = "SUBMIT HCWDL MHPE SEALED FINAL TEST EXACT JOB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--finalist-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args()
    spec = load_json(args.spec)
    spec_hash = validate_campaign(spec, verify_source_tree=args.execute)
    finalist = load_json(args.finalist_lock)
    finalist_hash = validate_content_hash(
        finalist, expected_contract=FINALIST_LOCK_CONTRACT, expected_schema_version=1,
    )
    execution = load_json(args.execution_lock)
    validate_execution_lock(execution)
    if (execution["campaign_spec_sha256"] != spec_hash
            or execution["finalist_lock_sha256"] != finalist_hash):
        raise PermissionError("HCWDL-MHPE final submission locks differ")
    output = Path(spec["campaign_root"]) / "final_test/evaluation.json"
    command = [
        "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
        "--cpus-per-task=8", "--mem=128G", "--time=12:00:00",
        "--gres=gpu:gh200:1", "--job-name=hcwmhpe_sealed_final",
        "--export=ALL," + ",".join((
            f"PROJECT_DIR={spec['project_dir']}",
            f"HCWDL_MHPE_SPEC={Path(args.spec).resolve()}",
            f"HCWDL_MHPE_FINALIST_LOCK={Path(args.finalist_lock).resolve()}",
            f"HCWDL_MHPE_EXECUTION_LOCK={Path(args.execution_lock).resolve()}",
            f"HCWDL_MHPE_FINAL_OUTPUT={output.resolve()}",
        )),
        str(Path(spec["project_dir"]) / "sbatch/run_hcwdl_mhpe_final_task.sh"),
    ]
    print(json.dumps({"command": command, "final_test_accessed": False}, indent=2))
    if args.execute:
        if args.authorization_phrase != PHRASE:
            raise PermissionError("HCWDL-MHPE final submission phrase differs")
        if ROOT.resolve() != Path(spec["project_dir"]).resolve():
            raise PermissionError("HCWDL-MHPE final submitter is outside bound worktree")
        validate_source_checkout(ROOT, expected_commit=spec["source_commit"])
        print(subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
