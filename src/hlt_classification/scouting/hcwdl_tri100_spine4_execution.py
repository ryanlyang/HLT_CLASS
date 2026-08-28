"""Single-GH200 execution acceptance for the TRI100 four-spine campaign."""

from __future__ import annotations

import math
import os
import re
import socket
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import require_sha256

from .hcwdl_tri100_spine4_contracts import (
    EXECUTION_ACCEPTANCE_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_graph import EXECUTION


def run_execution_acceptance(
    *, campaign_spec_sha256: str, recipe_sha256: str,
    source_commit: str, device: str = "cuda",
    require_production: bool = True,
) -> dict[str, Any]:
    """Prove the exact one-process CUDA topology and a real backward pass."""

    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TRI100 four-spine CUDA acceptance is unavailable")
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = (
        torch.cuda.get_device_name(target)
        if target.type == "cuda" else "cpu"
    )
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_nodes = os.environ.get("SLURM_NNODES")
    slurm_tasks = os.environ.get("SLURM_NTASKS")
    genuine_tigris = (
        slurm_job_id is not None
        and re.fullmatch(r"[0-9]+", slurm_job_id) is not None
        and slurm_nodes == "1"
        and slurm_tasks == "1"
        and target.type == "cuda"
        and visible == 1
        and "GH200" in device_name.upper()
    )
    if require_production and not genuine_tigris:
        raise RuntimeError("TRI100 four-spine production GPU topology differs")

    model = torch.nn.Linear(1, 1, bias=False, device=target)
    with torch.no_grad():
        model.weight.fill_(1.0)
    value = torch.tensor([[2.0]], dtype=torch.float32, device=target)
    output = model(value).square().mean()
    output.backward()
    observed_output = float(output.detach().cpu().item())
    observed_gradient = float(model.weight.grad.detach().cpu().item())
    if (
        not math.isclose(observed_output, 4.0, rel_tol=0, abs_tol=1.0e-6)
        or not math.isclose(observed_gradient, 8.0, rel_tol=0, abs_tol=1.0e-6)
    ):
        raise RuntimeError("TRI100 four-spine single-GPU backward pass differs")

    return artifact({
        "parents": {
            "campaign_spec": require_sha256(
                campaign_spec_sha256, name="TRI100 four-spine campaign",
            ),
            "recipe": require_sha256(
                recipe_sha256, name="TRI100 four-spine recipe",
            ),
        },
        "source_commit": source_commit,
        "execution": dict(EXECUTION),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": slurm_job_id,
        "slurm_nodes": slurm_nodes,
        "slurm_tasks": slurm_tasks,
        "visible_cuda_devices": visible,
        "device_name": device_name,
        "backward_output_expected": 4.0,
        "backward_output_observed": observed_output,
        "backward_gradient_expected": 8.0,
        "backward_gradient_observed": observed_gradient,
        "genuine_tigris_single_gh200_worker": (
            genuine_tigris if require_production else None
        ),
        "passed": True,
        "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)


def validate_execution_acceptance(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
    recipe_sha256: str,
) -> str:
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)

    def matches(observed: Any, expected: float) -> bool:
        try:
            return math.isclose(
                float(observed), expected, rel_tol=0, abs_tol=1.0e-6,
            )
        except (TypeError, ValueError):
            return False

    if (
        value.get("parents") != {
            "campaign_spec": campaign_spec_sha256, "recipe": recipe_sha256,
        }
        or re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("source_commit", "")),
        ) is None
        or value.get("execution") != dict(EXECUTION)
        or not isinstance(value.get("hostname"), str)
        or not value["hostname"]
        or not isinstance(value.get("pid"), int)
        or value["pid"] <= 0
        or re.fullmatch(r"[0-9]+", str(value.get("slurm_job_id", ""))) is None
        or value.get("slurm_nodes") != "1"
        or value.get("slurm_tasks") != "1"
        or value.get("visible_cuda_devices") != 1
        or "GH200" not in str(value.get("device_name", "")).upper()
        or not matches(value.get("backward_output_expected"), 4.0)
        or not matches(value.get("backward_output_observed"), 4.0)
        or not matches(value.get("backward_gradient_expected"), 8.0)
        or not matches(value.get("backward_gradient_observed"), 8.0)
        or value.get("genuine_tigris_single_gh200_worker") is not True
        or value.get("passed") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI100 four-spine execution acceptance differs")
    return digest


__all__ = ["run_execution_acceptance", "validate_execution_acceptance"]
