"""Genuine single-GH200 and new-view-source execution acceptance."""

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

from .hcwdl_fullcard_bottleneck_foundation_campaign import validate_foundation
from .hcwdl_homotopy import HomotopyCoordinate, PERSISTENT_HLT_SUPPORT_POLICY
from .hcwdl_homotopy_stream import iterate_unified_balanced_batches
from .hcwdl_tri100_spine4_bottleneck_contracts import (
    EXECUTION_ACCEPTANCE_CONTRACT,
    artifact,
    validate_artifact,
)
from .hcwdl_tri100_spine4_bottleneck_graph import EXECUTION
from .hcwdl_unified_balanced_runner import _load_common
from .repair import PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
from .training import derive_seed
from .hcwdl_tri100_spine4_persistent_support import validate_support_audit


def run_execution_acceptance(
    *, spec: Mapping[str, Any], source_commit: str, device: str = "cuda",
    require_production: bool = True,
) -> dict[str, Any]:
    import torch

    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    validate_foundation(foundation)
    support = load_json(spec["artifact_paths"]["support_audit"])
    support_hash = validate_support_audit(support, spec=spec)
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    repair_seed = derive_seed(int(spec["replicate_seed"]), "tri60/repair/shared_v1")
    batch = next(iterate_unified_balanced_batches(
        split, data_root=foundation["data_root"], role="validation",
        assignment_store=assignments["validation"],
        coupling_store=balanced["validation"],
        row_selection=selections["validation"],
        coordinate=HomotopyCoordinate(1, 2, 0, 1), repair_seed=repair_seed,
        batch_size=2, workers=1, source_index=0,
        include_training_metadata=True,
        support_policy=PERSISTENT_HLT_SUPPORT_POLICY,
    ))
    view = batch["privileged"]
    if len(batch["labels"]) < 1 or not np.isfinite(view.features).all():
        raise ValueError("bottleneck view-source acceptance differs")

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("bottleneck four-spine CUDA acceptance is unavailable")
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    genuine = (
        slurm_job_id is not None
        and re.fullmatch(r"[0-9]+", slurm_job_id) is not None
        and os.environ.get("SLURM_NNODES") == "1"
        and os.environ.get("SLURM_NTASKS") == "1"
        and target.type == "cuda" and visible == 1 and "GH200" in device_name.upper()
    )
    if require_production and not genuine:
        raise RuntimeError("bottleneck four-spine production GPU topology differs")
    torch.manual_seed(derive_seed(int(spec["replicate_seed"]), "spine4b/preflight/model"))
    model = build_scouting_particle_transformer().to(target)
    model.train()
    features = torch.as_tensor(view.features, dtype=torch.float32, device=target)
    vectors = torch.as_tensor(view.vectors, dtype=torch.float32, device=target)
    mask = torch.as_tensor(view.mask, dtype=torch.bool, device=target)
    if mask.ndim == 2:
        mask = mask[:, None]
    labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=target)
    logits = model(features, vectors, mask)
    if logits.shape != (len(labels), 15) or not torch.isfinite(logits).all():
        raise RuntimeError("bottleneck production-model forward pass differs")
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("bottleneck production model produced no gradients")
    gradient = float(torch.stack(gradients).norm().cpu())
    observed = float(loss.detach().cpu())
    if not math.isfinite(observed) or not math.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("bottleneck view-source backward pass is nonfinite")
    return artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
            "assignment_lock": spec["parents"]["assignment_lock"],
            "support_audit": support_hash,
        },
        "source_commit": source_commit, "execution": dict(EXECUTION),
        "hostname": socket.gethostname(), "pid": os.getpid(),
        "slurm_job_id": slurm_job_id,
        "slurm_nodes": os.environ.get("SLURM_NNODES"),
        "slurm_tasks": os.environ.get("SLURM_NTASKS"),
        "visible_cuda_devices": visible, "device_name": device_name,
        "view_role": "validation", "view_coordinate": "U050",
        "view_rows": len(batch["labels"]),
        "view_pairing_provenance": "pairing_validity",
        "matched_unclassified_hlt_policy": (
            PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        ),
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "production_model_factory": "build_scouting_particle_transformer",
        "production_model_output_shape": [len(labels), 15],
        "view_forward_loss": observed, "view_backward_gradient_norm": gradient,
        "installed_weaver_production_model_executed": True,
        "genuine_tigris_single_gh200_worker": genuine if require_production else None,
        "passed": True, "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)


def validate_execution_acceptance(
    value: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    if (
        value.get("parents") != {
            "campaign_spec": spec["content_hash"],
            "recipe": spec["parents"]["recipe"],
            "foundation": spec["parents"]["foundation"],
            "assignment_lock": spec["parents"]["assignment_lock"],
            "support_audit": validate_support_audit(
                load_json(spec["artifact_paths"]["support_audit"]), spec=spec,
            ),
        }
        or value.get("execution") != dict(EXECUTION)
        or value.get("view_role") != "validation"
        or value.get("view_coordinate") != "U050"
        or value.get("view_pairing_provenance") != "pairing_validity"
        or value.get("matched_unclassified_hlt_policy")
        != PAIRING_VALIDITY_UNCLASSIFIED_HLT_POLICY
        or value.get("support_policy") != PERSISTENT_HLT_SUPPORT_POLICY
        or int(value.get("view_rows", 0)) < 1
        or value.get("production_model_factory") != "build_scouting_particle_transformer"
        or value.get("production_model_output_shape") != [int(value["view_rows"]), 15]
        or not math.isfinite(float(value.get("view_forward_loss", float("nan"))))
        or not math.isfinite(float(value.get("view_backward_gradient_norm", float("nan"))))
        or float(value.get("view_backward_gradient_norm", 0.0)) <= 0
        or value.get("installed_weaver_production_model_executed") is not True
        or value.get("genuine_tigris_single_gh200_worker") is not True
        or value.get("passed") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("bottleneck four-spine execution acceptance differs")
    return digest


__all__ = ["run_execution_acceptance", "validate_execution_acceptance"]
