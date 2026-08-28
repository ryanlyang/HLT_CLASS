"""Four-rank execution acceptance for the TRI100 four-spine campaign."""

from __future__ import annotations

import math
import os
import re
import socket
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import require_sha256

from .hcwdl_mhpe_tri60_training import Tri60DistributedContext
from .hcwdl_tri100_spine4_contracts import (
    DISTRIBUTED_ACCEPTANCE_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_tri100_spine4_graph import DDP_EXECUTION


def run_distributed_acceptance(
    *, campaign_spec_sha256: str, recipe_sha256: str,
    source_commit: str, context: Tri60DistributedContext,
    require_production: bool = True,
) -> dict[str, Any]:
    """Prove collective topology and a real synchronized DDP backward pass."""

    import torch
    import torch.distributed as dist

    context.validate()
    if require_production and (
        context.world_size != DDP_EXECUTION["world_size"]
        or context.backend != DDP_EXECUTION["backend"]
        or context.local_rank != 0
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("TRI100 four-spine production DDP topology differs")
    device = torch.device("cuda:0" if context.backend == "nccl" else "cpu")
    collective = torch.tensor(
        [float(context.rank + 1)], dtype=torch.float64, device=device,
    )
    dist.all_reduce(collective, op=dist.ReduceOp.SUM)
    expected_collective = context.world_size * (context.world_size + 1) / 2
    if float(collective.item()) != expected_collective:
        raise RuntimeError("TRI100 four-spine collective arithmetic differs")

    model = torch.nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[0] if device.type == "cuda" else None,
        output_device=0 if device.type == "cuda" else None,
        broadcast_buffers=False,
    )
    value = torch.tensor(
        [[float(context.rank + 1)]], dtype=torch.float32, device=device,
    )
    ddp(value).square().mean().backward()
    observed_gradient = float(model.weight.grad.detach().cpu().item())
    expected_gradient = 2.0 * sum(
        float(rank + 1) ** 2 for rank in range(context.world_size)
    ) / context.world_size
    if not math.isclose(
        observed_gradient, expected_gradient, rel_tol=0, abs_tol=1.0e-6,
    ):
        raise RuntimeError("TRI100 four-spine DDP gradient average differs")

    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        ),
        "collective_sum": float(collective.item()),
        "ddp_gradient": observed_gradient,
    }
    ranks: list[Any] = [None] * context.world_size
    dist.all_gather_object(ranks, local)
    if [row["rank"] for row in ranks] != list(range(context.world_size)):
        raise RuntimeError("TRI100 four-spine rank registry differs")
    if require_production and len({row["hostname"] for row in ranks}) != context.world_size:
        raise RuntimeError("TRI100 four-spine requires one rank per node")

    value_out = artifact({
        "parents": {
            "campaign_spec": require_sha256(
                campaign_spec_sha256, name="TRI100 four-spine campaign",
            ),
            "recipe": require_sha256(
                recipe_sha256, name="TRI100 four-spine recipe",
            ),
        },
        "source_commit": source_commit,
        "execution": dict(DDP_EXECUTION),
        "observed_backend": context.backend,
        "observed_world_size": context.world_size,
        "rank_registry": ranks,
        "collective_expected": expected_collective,
        "collective_observed": float(collective.item()),
        "ddp_gradient_expected": expected_gradient,
        "ddp_gradient_observed_by_rank": [
            row["ddp_gradient"] for row in ranks
        ],
        "one_visible_gpu_per_rank": (
            all(row["visible_cuda_devices"] == 1 for row in ranks)
            if require_production else None
        ),
        "rank_zero_only_publication": True,
        "passed": True,
        "final_test_accessed": False,
    }, contract=DISTRIBUTED_ACCEPTANCE_CONTRACT)
    payload: list[Any] = [value_out if context.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    return dict(payload[0])


def validate_distributed_acceptance(
    value: Mapping[str, Any], *, campaign_spec_sha256: str,
    recipe_sha256: str,
) -> str:
    digest = validate_artifact(
        value, contract=DISTRIBUTED_ACCEPTANCE_CONTRACT,
    )
    ranks = value.get("rank_registry", ())
    expected_collective = 10.0
    expected_gradient = 15.0
    observed_gradients = value.get("ddp_gradient_observed_by_rank", ())

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
        or (
            re.fullmatch(
                r"[0-9a-f]{40}", str(value.get("source_commit", "")),
            ) is None
        )
        or value.get("execution") != dict(DDP_EXECUTION)
        or value.get("observed_backend") != "nccl"
        or value.get("observed_world_size") != 4
        or not isinstance(ranks, list) or len(ranks) != 4
        or any(not isinstance(row, Mapping) for row in ranks)
        or [row.get("rank") for row in ranks] != [0, 1, 2, 3]
        or len({row.get("hostname") for row in ranks}) != 4
        or any(row.get("local_rank") != 0 for row in ranks)
        or any(row.get("visible_cuda_devices") != 1 for row in ranks)
        or any(row.get("collective_sum") != expected_collective for row in ranks)
        or any(
            not matches(row.get("ddp_gradient"), expected_gradient)
            for row in ranks
        )
        or value.get("collective_expected") != expected_collective
        or value.get("collective_observed") != expected_collective
        or value.get("ddp_gradient_expected") != expected_gradient
        or not isinstance(observed_gradients, list)
        or len(observed_gradients) != 4
        or any(
            not matches(item, expected_gradient)
            for item in observed_gradients
        )
        or value.get("one_visible_gpu_per_rank") is not True
        or value.get("rank_zero_only_publication") is not True
        or value.get("passed") is not True
        or value.get("final_test_accessed") is not False
    ):
        raise ValueError("TRI100 four-spine distributed acceptance differs")
    return digest


__all__ = ["run_distributed_acceptance", "validate_distributed_acceptance"]
