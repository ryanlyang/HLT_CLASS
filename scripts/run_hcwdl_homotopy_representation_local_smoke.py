#!/usr/bin/env python3
"""Run a bounded, non-authorizing all-node HCWDL-U-RKD semantic smoke."""

from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json  # noqa: E402
from hlt_classification.scouting.hcwdl_homotopy_representation_graph import NODE_REGISTRY, resolved_base_loss, target_bank_registry, validate_graph  # noqa: E402
from hlt_classification.scouting.hcwdl_representation_losses import scheduled_representation_loss  # noqa: E402
from hlt_classification.scouting.hcwdl_representation_graph import RREL_STRATEGY  # noqa: E402
def main() -> int:
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); rows = []
    for node_id, node in NODE_REGISTRY.items():
        base = resolved_base_loss(node_id)
        schedules = []
        for effective_pass in (0.5, 2.0, 4.0, 6.0, 8.0, 60.0):
            short_strategy = "RREL" if node.strategy == RREL_STRATEGY else "RSET"
            value = scheduled_representation_loss(
                strategy=short_strategy, effective_pass=effective_pass,
                scaled_jet=torch.tensor(1.0, requires_grad=True),
                scaled_set=torch.tensor(2.0, requires_grad=True),
                scaled_relation=(torch.tensor(3.0, requires_grad=True) if short_strategy == "RREL" else None),
                orthogonality=torch.tensor(0.5, requires_grad=True),
            )
            value.total.backward()
            schedules.append({"pass": effective_pass, "loss": float(value.total.detach())})
        rows.append({
            "node_id": node_id, "domain": node.student_domain,
            "teacher": node.teacher.node_id, "temperature": node.temperature,
            "base_loss": asdict(base), "schedule": schedules,
        })
    artifact = with_content_hash({
        "contract": "HCWDL_HOMOTOPY_REPRESENTATION_LOCAL_SMOKE/v1",
        "schema_version": 1, "graph_sha256": validate_graph(),
        "nodes": rows, "fit_count": len(rows),
        "target_bank_count": len(target_bank_registry()),
        "all_nodes_exercised": len(rows) == 42,
        "final_test_accessed": False, "authorizes_tigris_or_pilot": False,
    })
    write_immutable_json(args.output, artifact); print(artifact["content_hash"]); return 0
if __name__ == "__main__": raise SystemExit(main())
