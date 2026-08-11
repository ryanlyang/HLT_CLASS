#!/usr/bin/env python3
"""Execute one exact HCWDL-U-RKD task from its immutable campaign spec."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import semantic_source_hashes, validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_workflow import HomotopyRepresentationWorkflow  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import RESOURCE_RECOVERY_CONTRACT, SOURCE_RECOVERY_CONTRACT, validate_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    spec = load_json(args.campaign_spec)
    recovery = None
    recovery_path = os.environ.get("HCWDL_U_RKD_RECOVERY")
    expected_project = Path(spec["project_dir"]).resolve()
    expected_commit = spec["source_commit"]
    if recovery_path:
        validate_campaign(spec, executable=True, verify_source=False)
        recovery = load_json(recovery_path)
        if recovery["contract"] not in {SOURCE_RECOVERY_CONTRACT, RESOURCE_RECOVERY_CONTRACT}:
            raise ValueError("HCWDL-U-RKD recovery contract differs")
        validate_artifact(
            recovery, contract=recovery["contract"],
            required_parents=("campaign_spec", "submission_ledger", "monitor_report"),
            required_fields=("source_commit", "project_dir", "semantic_source_sha256"),
        )
        if recovery["parents"]["campaign_spec"] != spec["content_hash"]:
            raise ValueError("HCWDL-U-RKD recovery campaign differs")
        expected_project = Path(recovery["project_dir"]).resolve()
        expected_commit = recovery["source_commit"]
    else:
        validate_campaign(spec, executable=True)
    if REPO_ROOT.resolve() != expected_project:
        raise PermissionError("HCWDL-U-RKD worker checkout differs")
    if subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise PermissionError("HCWDL-U-RKD worker checkout is dirty")
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise PermissionError("HCWDL-U-RKD worker commit differs")
    expected_hashes = (
        spec["semantic_source_sha256"] if recovery is None
        else recovery["semantic_source_sha256"]
    )
    if semantic_source_hashes(REPO_ROOT) != expected_hashes:
        raise PermissionError("HCWDL-U-RKD worker scientific source differs")
    HomotopyRepresentationWorkflow(
        spec, repository=REPO_ROOT, producer_commit=expected_commit,
        recovery_sha256=None if recovery is None else recovery["content_hash"],
    ).run(
        args.task, device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
