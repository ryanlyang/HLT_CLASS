"""Bounded synthetic end-to-end behavioral smoke for the 45-fit U/J v2 graph."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, sha256_file, with_content_hash, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_homotopy_graph import (
    DOMAINS, GRAPH_SHA256, NODE_REGISTRY, resolved_loss, validate_graph,
)
from .hcwdl_training import select_checkpoint
from .training import derive_seed, pmard_loss


LOCAL_SMOKE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_LOCAL_SMOKE/v1"
LOCAL_NODE_CONTRACT: Final = "HCWDL_STRUCTURAL_FEATURE_LOCAL_SMOKE_NODE/v1"


def _torch_bytes(value: object) -> bytes:
    import torch

    stream = BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def run_local_homotopy_smoke(output_root: str | Path) -> dict[str, Any]:
    """Train every registered edge for two tiny updates without external data.

    This is a behavioral graph/loss/teacher-domain smoke, not a scientific
    result and not a substitute for the production-worker ROOT/Tigris smoke.
    """

    import torch

    if validate_graph() != GRAPH_SHA256:
        raise ValueError("HCWDL-UJ local smoke graph identity differs")
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL-UJ local smoke output must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)

    labels = torch.arange(15, dtype=torch.long).repeat(2)
    generator = torch.Generator().manual_seed(941_177)
    common = torch.randn((len(labels), 12), generator=generator)
    domain_inputs = {}
    for index, domain in enumerate(sorted(DOMAINS)):
        # Fixed, finite domain offsets make wrong-domain routing observable.
        domain_inputs[domain] = common + (index + 1) / 1000.0
    domain_inputs["toff"] = common + 0.25

    class TinyClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(12, 15)

        def forward(self, value):
            return self.classifier(value)

    torch.manual_seed(derive_seed(1337, "hcwdl_uj/local/TOFF"))
    models: dict[str, TinyClassifier] = {"TOFF": TinyClassifier().eval()}
    report_hashes = {}
    for node_id, node in NODE_REGISTRY.items():
        torch.manual_seed(derive_seed(1337, f"hcwdl/init/{node.seed_alias}"))
        model = TinyClassifier()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        history = []
        states = []
        loss_config = resolved_loss(node_id)
        for update in (1, 2):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(domain_inputs[node.student_domain])
            hlt_target = privileged_target = None
            if node.teachers:
                teacher = node.teachers[0]
                teacher_model = models[teacher.node_id]
                teacher_model.eval()
                with torch.no_grad():
                    teacher_logits = teacher_model(domain_inputs[teacher.domain]).detach()
                if teacher.domain == "hlt":
                    hlt_target = teacher_logits
                else:
                    privileged_target = teacher_logits
            parts = pmard_loss(
                logits, labels, class_weights=torch.ones(15),
                configuration=loss_config, hlt_teacher_logits=hlt_target,
                privileged_teacher_logits=privileged_target,
            )
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"local HCWDL-UJ loss is nonfinite for {node_id}")
            parts["total"].backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_logits = model(domain_inputs[node.student_domain]).numpy()
            history.append({
                "update": update,
                **classification_metrics(validation_logits, labels.numpy()),
            })
            states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        selection = select_checkpoint(history)
        selected_index = [row["update"] for row in history].index(selection["selected_update"])
        model.load_state_dict(states[selected_index]); models[node_id] = model.eval()
        node_root = root / "nodes" / node_id
        node_root.mkdir(parents=True, exist_ok=True)
        checkpoint = node_root / "selected_model.pt"
        atomic_publish_bytes(checkpoint, _torch_bytes({"model": states[selected_index]}))
        report = with_content_hash({
            "contract": LOCAL_NODE_CONTRACT, "schema_version": 1,
            "node_id": node_id, "node": node.payload(),
            "resolved_loss": __import__("dataclasses").asdict(loss_config),
            "teacher_domain": None if not node.teachers else node.teachers[0].domain,
            "teacher_evaluated_on_own_domain": True,
            "updates": 2, "validation_history": history,
            "selection": selection,
            "selected_checkpoint_sha256": sha256_file(checkpoint),
            "finite": True, "scientific_result": False,
            "final_test_accessed": False,
        })
        report_path = node_root / "training_report.json"
        write_immutable_json(report_path, report)
        report_hashes[node_id] = report["content_hash"]

    if tuple(report_hashes) != tuple(NODE_REGISTRY):
        raise RuntimeError("HCWDL-UJ local smoke did not complete the exact graph")
    payload = with_content_hash({
        "contract": LOCAL_SMOKE_CONTRACT, "schema_version": 1,
        "graph_sha256": GRAPH_SHA256, "fit_count": len(report_hashes),
        "node_report_sha256": report_hashes,
        "updates_per_node": 2, "all_losses_finite": True,
        "all_students_cold_started": all(
            node.initialization == "fresh" for node in NODE_REGISTRY.values()
        ),
        "teacher_own_domain_routing_exercised": True,
        "production_worker_exercised": False,
        "scientific_result": False, "final_test_accessed": False,
    })
    write_immutable_json(root / "smoke_report.json", payload)
    return payload


__all__ = ["LOCAL_SMOKE_CONTRACT", "run_local_homotopy_smoke"]
