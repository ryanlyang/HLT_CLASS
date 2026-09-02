"""Genuine single-GH200 production acceptance for output handoff."""

from __future__ import annotations

import math
import os
import re
import socket
from typing import Any, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.models.scouting_particle_transformer import (
    build_scouting_particle_transformer,
)

from .hcwdl_adjacent_output_handoff_contracts import (
    EXECUTION_ACCEPTANCE_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_output_handoff_source import validate_source_lock
from .hcwdl_homotopy import PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_homotopy_stream import iterate_unified_balanced_batches
from .hcwdl_tri100_spine4_bottleneck_graph import NODE_REGISTRY as SOURCE_NODES
from .hcwdl_unified_balanced_runner import _load_common
from .training import derive_seed


def run_execution_acceptance(
    *, spec: Mapping[str, Any], source_commit: str, device: str = "cuda",
    require_production: bool = True,
) -> dict[str, Any]:
    """Run the exact view source, production ParT, and C25P75 T=2 loss."""

    import torch

    source = load_json(spec["artifact_paths"]["source_lock"])
    source_hash = validate_source_lock(source)
    source_spec = load_json(source["source_campaign_spec_path"])
    foundation = load_json(source["foundation_spec_path"])
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    node = SOURCE_NODES[source["u100_node_id"]]
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    stream = iterate_unified_balanced_batches(
        split, data_root=foundation["data_root"], role="validation",
        assignment_store=assignments["validation"],
        coupling_store=balanced["validation"],
        row_selection=selections["validation"], coordinate=node.coordinate,
        repair_seed=repair_seed, batch_size=2, workers=1,
        include_training_metadata=True,
        support_policy=PERSISTENT_HLT_SUPPORT_POLICY,
    )
    try:
        batch = next(stream)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    view = batch["privileged"]
    if len(batch["labels"]) < 1 or not np.isfinite(view.features).all():
        raise ValueError("output-handoff production view acceptance differs")

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("output-handoff CUDA acceptance is unavailable")
    visible = torch.cuda.device_count()
    name = torch.cuda.get_device_name(target)
    job = os.environ.get("SLURM_JOB_ID")
    genuine = bool(
        re.fullmatch(r"[0-9]+", job or "")
        and os.environ.get("SLURM_NNODES") == "1"
        and os.environ.get("SLURM_NTASKS") == "1"
        and visible == 1 and "GH200" in name.upper()
    )
    if require_production and not genuine:
        raise RuntimeError("output-handoff production topology differs")

    torch.manual_seed(derive_seed(int(spec["replicate_seed"]), "output-handoff/preflight/model"))
    model = build_scouting_particle_transformer().to(target).train()
    features = torch.as_tensor(view.features, dtype=torch.float32, device=target)
    vectors = torch.as_tensor(view.vectors, dtype=torch.float32, device=target)
    mask = torch.as_tensor(view.mask, dtype=torch.bool, device=target)
    if mask.ndim == 2:
        mask = mask[:, None]
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
    logits = model(features, vectors, mask).float()
    if logits.shape != (len(labels), 15) or not torch.isfinite(logits).all():
        raise RuntimeError("output-handoff production model forward differs")
    teacher_offset = torch.linspace(-.05, .05, 15, device=target)[None, :]
    teacher = torch.softmax(logits.detach() / 2.0 + teacher_offset, dim=1)
    ce = torch.nn.functional.cross_entropy(logits, labels)
    kd = torch.nn.functional.kl_div(
        torch.log_softmax(logits / 2.0, dim=1), teacher,
        reduction="batchmean",
    ) * 4.0
    loss = .25 * ce + .75 * kd
    loss.backward()
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters() if parameter.grad is not None
    ]
    gradient = float(torch.stack(gradients).norm().cpu()) if gradients else float("nan")
    observed = float(loss.detach().cpu())
    if not math.isfinite(observed) or not math.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("output-handoff production backward differs")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "recipe": spec["parents"]["recipe"],
            "source_lock": source_hash,
            "foundation": spec["parents"]["foundation"],
        },
        "source_commit": source_commit,
        "source_campaign_spec_sha256": source_spec["content_hash"],
        "hostname": socket.gethostname(), "slurm_job_id": job,
        "slurm_nodes": os.environ.get("SLURM_NNODES"),
        "slurm_tasks": os.environ.get("SLURM_NTASKS"),
        "visible_cuda_devices": visible, "device_name": name,
        "view_role": "validation", "view_coordinate": "U100",
        "view_rows": len(labels), "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "production_model_factory": "build_scouting_particle_transformer",
        "production_model_output_shape": [len(labels), 15],
        "loss_semantics": "C25P75_T2",
        "view_forward_loss": observed,
        "view_backward_gradient_norm": gradient,
        "installed_weaver_production_model_executed": True,
        "genuine_tigris_single_gh200_worker": genuine if require_production else None,
        "passed": True, "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)


def validate_execution_acceptance(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    source = load_json(spec["artifact_paths"]["source_lock"])
    source_spec = load_json(source["source_campaign_spec_path"])
    if (
        value.get("parents") != {
            "campaign_spec": spec["content_hash"],
            "recipe": spec["parents"]["recipe"],
            "source_lock": validate_source_lock(source),
            "foundation": spec["parents"]["foundation"],
        }
        or value.get("view_role") != "validation"
        or value.get("source_campaign_spec_sha256") != source_spec["content_hash"]
        or value.get("view_coordinate") != "U100"
        or value.get("support_policy") != PERSISTENT_HLT_SUPPORT_POLICY
        or int(value.get("view_rows", 0)) < 1
        or value.get("production_model_factory") != "build_scouting_particle_transformer"
        or value.get("production_model_output_shape") != [int(value["view_rows"]), 15]
        or value.get("loss_semantics") != "C25P75_T2"
        or not math.isfinite(float(value.get("view_forward_loss", float("nan"))))
        or not math.isfinite(float(value.get("view_backward_gradient_norm", float("nan"))))
        or float(value.get("view_backward_gradient_norm", 0.0)) <= 0
        or value.get("installed_weaver_production_model_executed") is not True
        or value.get("genuine_tigris_single_gh200_worker") is not True
        or value.get("visible_cuda_devices") != 1
        or "GH200" not in str(value.get("device_name", "")).upper()
        or value.get("passed") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("output-handoff execution acceptance differs")
    return digest


__all__ = ["run_execution_acceptance", "validate_execution_acceptance"]
