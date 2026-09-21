"""Explicit execution sites; no inherited defaults from older campaigns."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from .contracts import artifact, validate

DEBUG_PROFILE_TRANSFER = "sporc_debug_to_tier3_same_a100_environment_resources_v1"


def execution_site(name: str) -> dict:
    sites = {
        # Debug is opt-in and must be recorded by a versioned campaign/spec.
        # It is never inferred from a resource profile or used as the default.
        "sporc_a100_debug": dict(cluster="sporc", partition="debug", qos="qos_tier3",
                                 gres="gpu:a100:1", gpu_family="A100", architecture="x86_64",
                                 conda_base="/home/ryreu/miniconda3", conda_env="atlas_kd_sporc",
                                 max_cpus=36, max_memory_mb=340000),
        "sporc_a100": dict(cluster="sporc", partition="tier3", qos="qos_tier3",
                           gres="gpu:a100:1", gpu_family="A100", architecture="x86_64",
                           conda_base="/home/ryreu/miniconda3", conda_env="atlas_kd_sporc",
                           max_cpus=36, max_memory_mb=340000),
        "tigris_gh200": dict(cluster="tigris", partition="tigris", qos=None,
                             gres="gpu:gh200:1", gpu_family="GH200", architecture="aarch64",
                             conda_base="/home/ryreu/miniforge3-aarch64", conda_env="atlas_kd_tigris",
                             max_cpus=72, max_memory_mb=550000),
    }
    if name not in sites:
        raise ValueError("Unknown Delphes execution site")
    return artifact("EXECUTION_SITE", name=name, account="reu-aisocial",
                    nodes=1, tasks=1, gpus=1, **sites[name])


def validate_site(site: dict) -> str:
    digest = validate(site, "EXECUTION_SITE")
    if site != execution_site(site["name"]):
        raise ValueError("Execution site differs from the registered profile")
    return digest


def production_site(measurement_site: dict) -> dict:
    """One explicit profiling exception, not arbitrary cross-site portability."""
    validate_site(measurement_site)
    if measurement_site["name"] == "sporc_a100_debug":
        return execution_site("sporc_a100")
    return measurement_site


def validate_resources(site: dict, cpus: int, memory_mb: int, workers: int):
    validate_site(site)
    if (any(type(v) is not int for v in (cpus, memory_mb, workers))
            or not 1 <= workers <= cpus <= site["max_cpus"]
            or not 1024 <= memory_mb <= site["max_memory_mb"]):
        raise ValueError("CPU/worker/memory request exceeds the execution site envelope")


def slurm_options(site: dict) -> list[str]:
    validate_site(site)
    options = ["sbatch", "--parsable", "--account=" + site["account"],
               "--partition=" + site["partition"], "--nodes=1", "--ntasks=1",
               "--export=ALL", "--no-requeue"]
    if site["qos"] is not None:
        options.append("--qos=" + site["qos"])
    return options


def allocation(site: dict) -> tuple[str, int, int]:
    """Authenticate scheduler, host, isolated environment and visible GPU."""
    import torch

    validate_site(site)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise PermissionError("A real Slurm allocation is required")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    memory_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    validate_resources(site, cpus, memory_mb, 1)
    details = subprocess.run(["scontrol", "show", "job", "-o", job_id],
                             check=True, text=True, capture_output=True).stdout
    fields = dict(token.split("=", 1) for token in details.split() if "=" in token)
    if (fields.get("Account") != site["account"] or fields.get("Partition") != site["partition"]
            or (site["qos"] is not None and fields.get("QOS") != site["qos"])
            or os.environ.get("SLURM_CLUSTER_NAME") != site["cluster"]
            or fields.get("NumNodes") != "1" or fields.get("NumTasks") != "1"
            or fields.get("NumCPUs") != str(cpus)):
        raise PermissionError("Not the registered single-task scheduler allocation")
    expected_prefix = site["conda_base"] + "/envs/" + site["conda_env"]
    if (platform.machine() != site["architecture"] or sys.prefix != expected_prefix
            or os.environ.get("CONDA_PREFIX") != expected_prefix
            or os.environ.get("PYTHONNOUSERSITE") != "1"):
        raise PermissionError("Worker host/Conda environment differs from the execution site")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or site["gpu_family"] not in torch.cuda.get_device_name(0)
            or not torch.cuda.is_bf16_supported()):
        raise PermissionError("Exactly one registered BF16-capable GPU is required")
    return job_id, cpus, memory_mb


def gpu_identity() -> dict:
    import torch
    properties = torch.cuda.get_device_properties(0)
    return dict(name=torch.cuda.get_device_name(0), total_memory_bytes=properties.total_memory,
                compute_capability=list(torch.cuda.get_device_capability(0)))
