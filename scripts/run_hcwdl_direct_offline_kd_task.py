#!/usr/bin/env python3
"""Run one exact direct offline-to-HLT KD campaign task."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout  # noqa: E402
from hlt_classification.scouting.hcwdl_direct_offline_kd_campaign import validate_campaign  # noqa: E402
from hlt_classification.scouting.hcwdl_direct_offline_kd_workflow import DirectOfflineKdWorkflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    spec = load_json(args.campaign_spec); validate_campaign(spec, executable=True)
    if REPO_ROOT.resolve() != Path(spec["project_dir"]).resolve():
        raise PermissionError("direct KD worker is not running from its bound worktree")
    validate_source_checkout(REPO_ROOT, expected_commit=spec["source_commit"])
    DirectOfflineKdWorkflow(spec, repository=REPO_ROOT).run(args.task, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
