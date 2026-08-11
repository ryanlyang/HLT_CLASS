#!/usr/bin/env python3
"""Authorize and remove only compact HCWDL-U-RKD target payload bytes."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_graph import NODE_REGISTRY  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_targets import build_cleanup_authorization, build_cleanup_completion, validate_target_manifest  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_training import target_output_dir, node_output_dir  # noqa: E402
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-phrase")
    args = parser.parse_args(); spec = load_json(args.campaign_spec)
    root = Path(spec["campaign_root"]); bank_root = target_output_dir(root, args.bank)
    auth_path = bank_root / "cleanup_authorization.json"
    if auth_path.is_file():
        authorization = load_json(auth_path)
    else:
        manifest = load_json(bank_root / "manifest.json"); validate_target_manifest(manifest)
        consumers = {}
        for row in manifest["payload"]["authorized_consumers"]:
            node = row["node_id"]
            report = load_json(node_output_dir(root, node) / "combined_training_report.json")
            consumers[node] = report["content_hash"]
        authorization = build_cleanup_authorization(manifest, completed_consumers=consumers)
        write_immutable_json(auth_path, authorization)
    print(f"Authorized payloads: {len(authorization['authorized_paths'])}")
    if not args.execute:
        return 0
    if args.authorization_phrase != "DELETE HCWDL U RKD AUTHORIZED TARGET PAYLOADS":
        raise PermissionError("HCWDL-U-RKD cleanup phrase differs")
    absent = []
    for raw in authorization["authorized_paths"]:
        path = Path(raw); path.unlink(missing_ok=True); absent.append(str(path))
    completion = build_cleanup_completion(authorization, absent_paths=absent)
    write_immutable_json(bank_root / "cleanup_completion.json", completion)
    return 0
if __name__ == "__main__": raise SystemExit(main())
