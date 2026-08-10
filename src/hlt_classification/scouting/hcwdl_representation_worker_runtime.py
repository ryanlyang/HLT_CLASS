"""Live worker provenance for executable HCWDL-RKD task rows.

Runtime bindings are reviewed commitments, not observations.  This module is
the deliberately small boundary that measures the process which is about to
execute a row and proves that it is the process committed by the binding.  It
must run before any registered scientific input is opened.

The audit-only GPU UUID is intentionally excluded from the reusable row
signature: Slurm may place an otherwise identical job on any equivalent
registered device.  It is nevertheless measured and is written into the
post-build target execution attestation.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import site
import subprocess
import sys
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from hlt_classification.provenance import capture_source_snapshot

from .hcwdl_representation_contracts import (
    LIVE_WORKER_RUNTIME_DOMAIN,
    ROW_RUNTIME_SIGNATURE_DOMAIN,
    WORKER_RUNTIME_MEASUREMENT_CONTRACT,
)


TARGET_DETERMINISTIC_SEED: Final = 1337
_WEAVER_SUFFIXES: Final = frozenset({".py", ".so", ".pyd", ".dll", ".dylib"})


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_import_origin(project_dir: Path) -> None:
    expected = (
        project_dir
        / "src"
        / "hlt_classification"
        / "scouting"
        / "hcwdl_representation_worker_runtime.py"
    ).resolve()
    if Path(__file__).resolve() != expected:
        raise PermissionError(
            "HCWDL-RKD worker imported runtime code outside the frozen project checkout"
        )


def _active_conda_environment(expected: str) -> dict[str, str]:
    active = os.environ.get("CONDA_DEFAULT_ENV")
    raw_prefix = os.environ.get("CONDA_PREFIX")
    if active != expected or not raw_prefix:
        raise PermissionError("HCWDL-RKD worker Conda environment differs")
    prefix = Path(raw_prefix).resolve()
    executable = Path(sys.executable).resolve()
    if prefix.name != expected or not prefix.is_dir() or not _under(
        executable, prefix,
    ):
        raise PermissionError(
            "HCWDL-RKD worker Python is outside the bound Conda environment"
        )
    return {
        "environment": active,
        "prefix": str(prefix),
        "python_executable": str(executable),
    }


def _require_no_user_site() -> None:
    if (
        os.environ.get("PYTHONNOUSERSITE") != "1"
        or int(sys.flags.no_user_site) != 1
        or site.ENABLE_USER_SITE is not False
    ):
        raise PermissionError("HCWDL-RKD worker user-site isolation is not live")


def measure_weaver_runtime_source() -> dict[str, Any]:
    """Hash the actual importable Weaver package bytes, not a version label."""

    specification = importlib.util.find_spec("weaver")
    locations = (
        () if specification is None
        else specification.submodule_search_locations
    )
    roots = tuple(Path(value).resolve() for value in (locations or ()))
    if len(roots) != 1 or not roots[0].is_dir() or roots[0].is_symlink():
        raise RuntimeError("installed Weaver package root is unavailable or ambiguous")
    root = roots[0]
    inventory: list[dict[str, Any]] = []
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise PermissionError("installed Weaver runtime contains a symlink")
        if not member.is_file() or member.suffix.lower() not in _WEAVER_SUFFIXES:
            continue
        inventory.append({
            "path": member.relative_to(root).as_posix(),
            "bytes": int(member.stat().st_size),
            "sha256": sha256_file(member),
        })
    if not inventory:
        raise RuntimeError("installed Weaver runtime has no defining source/binary files")
    distribution_names = sorted(
        importlib.metadata.packages_distributions().get("weaver", ())
    )
    if not distribution_names:
        # Weaver is distributed as weaver-core in the authoritative Tigris
        # environment.  The fallback keeps a conventional non-editable install
        # measurable when packages_distributions has incomplete metadata.
        distribution_names = ["weaver-core"]
    distributions = []
    for name in distribution_names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        distributions.append({"name": name, "version": version})
    if not distributions:
        raise RuntimeError("installed Weaver distribution metadata is unavailable")
    payload = {
        "distributions": distributions,
        "files": inventory,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"required HCWDL-RKD runtime package is absent: {distribution}"
        ) from error


def _package_inventory(torch: Any, weaver: Mapping[str, Any]) -> dict[str, str]:
    cuda = getattr(torch.version, "cuda", None)
    cudnn = torch.backends.cudnn.version()
    if cuda is None or cudnn is None:
        cuda_text = "unavailable"
        cudnn_text = "unavailable"
    else:
        cuda_text = str(cuda)
        cudnn_text = str(cudnn)
    distributions = weaver["distributions"]
    weaver_versions = ",".join(
        f"{row['name']}=={row['version']}" for row in distributions
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": cuda_text,
        "cudnn": cudnn_text,
        "numpy": _package_version("numpy"),
        "awkward": _package_version("awkward"),
        "uproot": _package_version("uproot"),
        "weaver": f"{weaver_versions};source_sha256={weaver['sha256']}",
    }


def _nvidia_smi_rows() -> list[dict[str, str]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi device audit failed: {process.stderr.strip()}"
        )
    rows = []
    for line in process.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or any(not field for field in fields):
            raise RuntimeError("nvidia-smi device audit row differs")
        rows.append(dict(zip(
            ("index", "name", "driver", "uuid"), fields, strict=True,
        )))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU device")
    return rows


def _selected_nvidia_row(
    rows: list[dict[str, str]], *, local_index: int,
) -> dict[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if visible is None else [value.strip() for value in visible.split(",")]
    if tokens and local_index < len(tokens):
        token = tokens[local_index]
        if token.startswith("GPU-"):
            matches = [row for row in rows if row["uuid"] == token]
        elif token.isdigit():
            matches = [row for row in rows if row["index"] == token]
        else:
            matches = []
    elif len(rows) == 1 and local_index == 0:
        matches = rows
    else:
        matches = [row for row in rows if row["index"] == str(local_index)]
    if len(matches) != 1:
        raise RuntimeError("active CUDA device cannot be mapped to one audited GPU UUID")
    return matches[0]


def _cuda_device(torch: Any) -> tuple[dict[str, str], str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("HCWDL-RKD CUDA row requires exactly one visible GPU")
    local_index = int(torch.cuda.current_device())
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(local_index))
    if len(capability) != 2:
        raise RuntimeError("CUDA compute capability differs")
    architecture = {9: "Hopper"}.get(capability[0])
    if architecture is None:
        raise RuntimeError("HCWDL-RKD CUDA device architecture is not registered")
    audited = _selected_nvidia_row(_nvidia_smi_rows(), local_index=local_index)
    model = str(torch.cuda.get_device_name(local_index)).strip()
    if not model or audited["name"] != model:
        raise RuntimeError("PyTorch and nvidia-smi GPU model identities differ")
    runtime = getattr(torch.version, "cuda", None)
    if runtime is None:
        raise RuntimeError("CUDA-enabled PyTorch runtime is unavailable")
    public = {
        "request": "gpu:gh200:1",
        "architecture": architecture,
        "model": model,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver": audited["driver"],
        "runtime": str(runtime),
    }
    return public, audited["uuid"]


def _sha256_tensor(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def configure_deterministic_worker_backend(torch: Any) -> dict[str, Any]:
    """Apply and measure the frozen deterministic worker backend policy."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PermissionError("deterministic worker CUBLAS workspace differs")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.manual_seed(TARGET_DETERMINISTIC_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TARGET_DETERMINISTIC_SEED)
    states = {
        "cpu": _sha256_tensor(torch.get_rng_state()),
        "cuda": [
            _sha256_tensor(state) for state in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
            )
        ],
    }
    reduced = any(
        bool(getattr(torch.backends.cuda.matmul, name))
        for name in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        )
        if hasattr(torch.backends.cuda.matmul, name)
    )
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "reduced_precision_fp32_reduction": reduced,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "rng_states_sha256": canonical_sha256(states),
    }


def _ordinary_backend(torch: Any) -> dict[str, Any]:
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def measure_live_worker_runtime(
    *, project_dir: str | Path, expected_conda_environment: str,
    expected_source_commit: str, expected_source_snapshot_sha256: str,
    expected_weaver_runtime_sha256: str, row_device: str,
    resource_class: str, deterministic_worker: bool,
) -> dict[str, Any]:
    """Measure and validate the live process before registered input access."""

    project = Path(project_dir).resolve()
    if os.environ.get("PROJECT_DIR") is None or Path(
        os.environ["PROJECT_DIR"]
    ).resolve() != project or Path.cwd().resolve() != project:
        raise PermissionError("HCWDL-RKD live project directory differs")
    _validate_import_origin(project)
    _require_no_user_site()
    conda = _active_conda_environment(expected_conda_environment)
    snapshot = capture_source_snapshot(project, require_clean=True)
    if (
        snapshot["git_commit"] != expected_source_commit
        or snapshot["source_snapshot_sha256"]
        != require_sha256(
            expected_source_snapshot_sha256, name="bound source snapshot"
        )
    ):
        raise PermissionError("HCWDL-RKD live clean source checkout differs")
    weaver = measure_weaver_runtime_source()
    if weaver["sha256"] != require_sha256(
        expected_weaver_runtime_sha256, name="bound Weaver runtime"
    ):
        raise PermissionError("HCWDL-RKD live Weaver/runtime-source signature differs")

    import torch

    if row_device == "cuda":
        device, gpu_uuid = _cuda_device(torch)
    elif row_device == "cpu":
        device, gpu_uuid = {"type": "cpu"}, None
    else:
        raise ValueError("HCWDL-RKD row device differs")
    backend = (
        configure_deterministic_worker_backend(torch)
        if deterministic_worker else _ordinary_backend(torch)
    )
    return with_content_hash({
        "contract": LIVE_WORKER_RUNTIME_DOMAIN,
        "schema_version": 1,
        "project_dir": str(project),
        "source_commit": snapshot["git_commit"],
        "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
        "conda": conda,
        "python_no_user_site": True,
        "packages": _package_inventory(torch, weaver),
        "weaver_runtime_sha256": weaver["sha256"],
        "resource_class": str(resource_class),
        "row_device": row_device,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "deterministic_worker": bool(deterministic_worker),
        "backend": backend,
    })


def build_row_runtime_signature(live: Mapping[str, Any]) -> dict[str, Any]:
    """Build the reusable signature; omit only the audit-only physical UUID."""

    validate_live_worker_runtime(live)
    required = {
        "project_dir", "source_commit", "source_snapshot_sha256", "conda",
        "python_no_user_site", "packages", "weaver_runtime_sha256",
        "resource_class", "row_device", "device", "deterministic_worker",
        "backend",
    }
    if not isinstance(live, Mapping) or not required <= set(live):
        raise ValueError("HCWDL-RKD live worker measurement fields differ")
    return with_content_hash({
        "contract": ROW_RUNTIME_SIGNATURE_DOMAIN,
        "schema_version": 1,
        **{name: live[name] for name in sorted(required)},
    })


def validate_live_worker_runtime(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=LIVE_WORKER_RUNTIME_DOMAIN,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "project_dir", "source_commit",
        "source_snapshot_sha256", "conda", "python_no_user_site", "packages",
        "weaver_runtime_sha256", "resource_class", "row_device", "device",
        "gpu_uuid", "deterministic_worker", "backend", "content_hash",
    }
    if set(value) != required:
        raise ValueError("HCWDL-RKD live worker runtime fields differ")
    commit = str(value["source_commit"])
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("HCWDL-RKD live worker source commit differs")
    require_sha256(value["source_snapshot_sha256"], name="live source snapshot")
    require_sha256(value["weaver_runtime_sha256"], name="live Weaver runtime")
    if value.get("python_no_user_site") is not True:
        raise PermissionError("HCWDL-RKD live worker user-site claim differs")
    conda = value.get("conda")
    if not isinstance(conda, Mapping) or set(conda) != {
        "environment", "prefix", "python_executable",
    } or any(not isinstance(item, str) or not item for item in conda.values()):
        raise ValueError("HCWDL-RKD live worker Conda identity differs")
    packages = value.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "python", "torch", "cuda", "cudnn", "numpy", "awkward", "uproot", "weaver",
    } or any(
        not isinstance(item, str) or not item for item in packages.values()
    ):
        raise ValueError("HCWDL-RKD live worker package inventory differs")
    device = value.get("device")
    row_device = value.get("row_device")
    if row_device == "cpu":
        if device != {"type": "cpu"} or value.get("gpu_uuid") is not None:
            raise ValueError("HCWDL-RKD live CPU identity differs")
    elif row_device == "cuda":
        if (
            not isinstance(device, Mapping)
            or set(device) != {
                "request", "architecture", "model", "compute_capability",
                "driver", "runtime",
            }
            or any(not isinstance(item, str) or not item for item in device.values())
            or not isinstance(value.get("gpu_uuid"), str)
            or not value["gpu_uuid"]
        ):
            raise ValueError("HCWDL-RKD live CUDA identity differs")
    else:
        raise ValueError("HCWDL-RKD live worker row device differs")
    if not isinstance(value.get("backend"), Mapping) or not value["backend"]:
        raise ValueError("HCWDL-RKD live worker backend identity differs")
    return digest


def validate_live_task_runtime(
    *, spec: Mapping[str, Any], binding: Mapping[str, Any],
    task: Mapping[str, Any], runtime_row: Mapping[str, Any],
    deterministic_worker: bool,
) -> dict[str, Any]:
    facts = binding.get("runtime_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("HCWDL-RKD runtime binding has no fact registry")
    if bool(task.get("deterministic_worker", False)) is not deterministic_worker:
        raise PermissionError("HCWDL-RKD live worker route differs")
    row_device = str(runtime_row.get("device", ""))
    if row_device == "cuda" and facts.get("device") != "cuda":
        raise PermissionError("HCWDL-RKD CUDA row differs from the bound device policy")
    live = measure_live_worker_runtime(
        project_dir=str(facts["project_dir"]),
        expected_conda_environment=str(facts["conda_environment"]),
        expected_source_commit=str(spec["source_commit"]),
        expected_source_snapshot_sha256=str(facts["source_snapshot_sha256"]),
        expected_weaver_runtime_sha256=str(facts["weaver_runtime_sha256"]),
        row_device=row_device,
        resource_class=str(task["resource_class"]),
        deterministic_worker=deterministic_worker,
    )
    expected = require_sha256(
        runtime_row.get("runtime_signature_sha256"),
        name="registered row runtime signature",
    )
    actual = build_row_runtime_signature(live)["content_hash"]
    if actual != expected:
        raise PermissionError("HCWDL-RKD live row runtime signature differs")
    return live


def build_live_target_runtime_environment(
    forward_spec: Mapping[str, Any], *, live_worker_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Return measured target facts for the post-build execution attestation."""

    from .hcwdl_representation_targets import validate_target_forward_spec

    validate_target_forward_spec(forward_spec)
    validate_live_worker_runtime(live_worker_runtime)
    payload = forward_spec["payload"]
    if live_worker_runtime.get("row_device") != "cuda":
        raise PermissionError("HCWDL-RKD target construction requires a live CUDA worker")
    import torch

    backend = configure_deterministic_worker_backend(torch)
    producer = {
        "source_commit": live_worker_runtime["source_commit"],
        "source_snapshot_sha256": live_worker_runtime["source_snapshot_sha256"],
        "packages": dict(live_worker_runtime["packages"]),
    }
    precision = {
        "parameters": "float32",
        "inputs": "float32",
        "activations": "float32",
        "autocast": False,
        "matmul_tf32": backend["matmul_tf32"],
        "cudnn_tf32": backend["cudnn_tf32"],
        "reduced_precision_fp32_reduction": backend[
            "reduced_precision_fp32_reduction"
        ],
        "output_order": "C",
    }
    determinism = {
        "deterministic_algorithms": backend["deterministic_algorithms"],
        "cudnn_deterministic": backend["cudnn_deterministic"],
        "cudnn_benchmark": backend["cudnn_benchmark"],
        "cublas_workspace_config": backend["cublas_workspace_config"],
        "rng_states_sha256": backend["rng_states_sha256"],
    }
    device = dict(live_worker_runtime["device"])
    device["gpu_uuid"] = live_worker_runtime.get("gpu_uuid")
    measured = {
        "producer": producer,
        "device": device,
        "precision": precision,
        "determinism": determinism,
    }
    expected = {
        "producer": payload["producer"],
        "device": payload["device"],
        "precision": payload["precision"],
        "determinism": payload["determinism"],
    }
    comparable_device = dict(device)
    gpu_uuid = comparable_device.pop("gpu_uuid", None)
    if (
        not isinstance(gpu_uuid, str) or not gpu_uuid
        or producer != expected["producer"]
        or comparable_device != expected["device"]
        or precision != expected["precision"]
        or determinism != expected["determinism"]
    ):
        raise PermissionError(
            "HCWDL-RKD measured target producer/environment/device/backend differs"
        )
    return measured


def target_forward_runtime_commitment_from_measurement(
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one reviewed deterministic GPU measurement into forward facts."""

    validate_worker_runtime_measurement(measurement)
    live = measurement["live_worker_runtime"]
    backend = live["backend"]
    if (
        live.get("row_device") != "cuda"
        or live.get("deterministic_worker") is not True
        or set(backend) != {
            "deterministic_algorithms", "cudnn_deterministic",
            "cudnn_benchmark", "matmul_tf32", "cudnn_tf32",
            "reduced_precision_fp32_reduction", "cublas_workspace_config",
            "rng_states_sha256",
        }
    ):
        raise PermissionError(
            "target forward requires a deterministic measured CUDA worker"
        )
    return {
        "producer": {
            "source_commit": live["source_commit"],
            "source_snapshot_sha256": live["source_snapshot_sha256"],
            "packages": dict(live["packages"]),
        },
        "device": dict(live["device"]),
        "precision": {
            "parameters": "float32", "inputs": "float32",
            "activations": "float32", "autocast": False,
            "matmul_tf32": backend["matmul_tf32"],
            "cudnn_tf32": backend["cudnn_tf32"],
            "reduced_precision_fp32_reduction": backend[
                "reduced_precision_fp32_reduction"
            ],
            "output_order": "C",
        },
        "determinism": {
            "deterministic_algorithms": backend["deterministic_algorithms"],
            "cudnn_deterministic": backend["cudnn_deterministic"],
            "cudnn_benchmark": backend["cudnn_benchmark"],
            "cublas_workspace_config": backend["cublas_workspace_config"],
            "rng_states_sha256": backend["rng_states_sha256"],
        },
    }


def build_worker_runtime_measurement(
    *, campaign_spec_sha256: str, data_root: str | Path,
    resource_class: str, resource_request: Mapping[str, Any],
    live_worker_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Package one live allocated-class measurement without authorizing work."""

    request_fields = {"cpus", "memory", "walltime", "gpu"}
    if not isinstance(resource_request, Mapping) or set(resource_request) != request_fields:
        raise ValueError("HCWDL-RKD measured resource request fields differ")
    cpus = resource_request["cpus"]
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus <= 0:
        raise ValueError("HCWDL-RKD measured resource CPU request differs")
    if not all(
        isinstance(resource_request[name], str) and resource_request[name]
        for name in ("memory", "walltime")
    ) or (
        resource_request["gpu"] is not None
        and (not isinstance(resource_request["gpu"], str) or not resource_request["gpu"])
    ):
        raise ValueError("HCWDL-RKD measured resource request values differ")
    if live_worker_runtime.get("resource_class") != resource_class:
        raise ValueError("HCWDL-RKD measured resource class differs from live runtime")
    row_device = live_worker_runtime.get("row_device")
    if (resource_request["gpu"] is None) != (row_device == "cpu"):
        raise ValueError("HCWDL-RKD measured resource/device route differs")
    raw_data = Path(data_root)
    if not raw_data.is_absolute():
        raise ValueError("HCWDL-RKD measured data root is not absolute")
    data = raw_data.resolve()
    if not data.is_dir() or data.is_symlink():
        raise FileNotFoundError("HCWDL-RKD measured data root is absent or a symlink")
    row_signature = build_row_runtime_signature(live_worker_runtime)
    runtime_facts = {
        "conda_environment": live_worker_runtime["conda"]["environment"],
        "data_root": str(data),
        "device": row_device,
        "project_dir": live_worker_runtime["project_dir"],
        "python_no_user_site": True,
        "source_snapshot_sha256": live_worker_runtime["source_snapshot_sha256"],
        "weaver_runtime_sha256": live_worker_runtime["weaver_runtime_sha256"],
    }
    return with_content_hash({
        "contract": WORKER_RUNTIME_MEASUREMENT_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="runtime-measurement campaign spec"
        ),
        "resource_class": str(resource_class),
        "resource_request": dict(resource_request),
        "resource_request_sha256": canonical_sha256(dict(resource_request)),
        "runtime_facts": runtime_facts,
        "row_runtime_signature": row_signature,
        "row_runtime_signature_sha256": row_signature["content_hash"],
        "live_worker_runtime": dict(live_worker_runtime),
        "scheduler_mutated": False,
        "scientific_authorization": False,
        "authorizes_tigris_or_pilot": False,
        "final_role_accessed": False,
    })


def validate_worker_runtime_measurement(value: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        value, expected_contract=WORKER_RUNTIME_MEASUREMENT_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "campaign_spec_sha256", "resource_class",
        "resource_request", "resource_request_sha256", "runtime_facts",
        "row_runtime_signature", "row_runtime_signature_sha256",
        "live_worker_runtime", "scheduler_mutated", "scientific_authorization",
        "authorizes_tigris_or_pilot", "final_role_accessed", "content_hash",
    }
    if set(value) != required:
        raise ValueError("HCWDL-RKD worker runtime measurement fields differ")
    require_sha256(value["campaign_spec_sha256"], name="runtime-measurement campaign")
    runtime_facts = value.get("runtime_facts")
    data_root = (
        Path(str(runtime_facts.get("data_root")))
        if isinstance(runtime_facts, Mapping) else Path(".")
    )
    if not data_root.is_absolute():
        raise ValueError("HCWDL-RKD worker runtime measurement data root differs")
    if (
        value.get("resource_request_sha256")
        != canonical_sha256(value.get("resource_request"))
        or value.get("row_runtime_signature_sha256")
        != value.get("row_runtime_signature", {}).get("content_hash")
        or build_row_runtime_signature(value.get("live_worker_runtime", {}))
        != value.get("row_runtime_signature")
        or value.get("runtime_facts") != {
            "conda_environment": value["live_worker_runtime"]["conda"]["environment"],
            "data_root": value["runtime_facts"].get("data_root"),
            "device": value["live_worker_runtime"]["row_device"],
            "project_dir": value["live_worker_runtime"]["project_dir"],
            "python_no_user_site": True,
            "source_snapshot_sha256": value["live_worker_runtime"][
                "source_snapshot_sha256"
            ],
            "weaver_runtime_sha256": value["live_worker_runtime"][
                "weaver_runtime_sha256"
            ],
        }
        or any(
            value.get(name) is not False
            for name in (
                "scheduler_mutated", "scientific_authorization",
                "authorizes_tigris_or_pilot", "final_role_accessed",
            )
        )
    ):
        raise PermissionError("HCWDL-RKD worker runtime measurement claims/lineage differ")
    request = value["resource_request"]
    if not isinstance(request, Mapping) or set(request) != {
        "cpus", "memory", "walltime", "gpu",
    }:
        raise ValueError("HCWDL-RKD worker runtime request differs")
    return digest


def measure_registered_worker_runtime(
    *, spec: Mapping[str, Any], data_root: str | Path,
    conda_environment: str, resource_class: str,
    resource_request: Mapping[str, Any], row_device: str,
    deterministic_worker: bool,
) -> dict[str, Any]:
    """Measure one exact planning-spec resource class on its allocated node."""

    resources = spec.get("resources")
    if (
        not isinstance(resources, Mapping)
        or resource_class not in resources
        or dict(resources[resource_class]) != dict(resource_request)
    ):
        raise PermissionError("HCWDL-RKD explicit resource request differs from campaign")
    matching_tasks = [
        row for row in spec.get("tasks", ())
        if row.get("resource_class") == resource_class
    ]
    if not matching_tasks or {
        bool(row.get("deterministic_worker")) for row in matching_tasks
    } != {bool(deterministic_worker)}:
        raise PermissionError("HCWDL-RKD measured worker class has mixed/wrong routing")
    expected_device = "cuda" if resource_request.get("gpu") is not None else "cpu"
    if row_device != expected_device:
        raise PermissionError("HCWDL-RKD measured row device differs from resource request")
    project = Path(str(spec["project_dir"]))
    snapshot = capture_source_snapshot(project, require_clean=True)
    if snapshot["git_commit"] != spec.get("source_commit"):
        raise PermissionError("HCWDL-RKD measurement checkout differs from campaign")
    weaver = measure_weaver_runtime_source()
    live = measure_live_worker_runtime(
        project_dir=project,
        expected_conda_environment=conda_environment,
        expected_source_commit=snapshot["git_commit"],
        expected_source_snapshot_sha256=snapshot["source_snapshot_sha256"],
        expected_weaver_runtime_sha256=weaver["sha256"],
        row_device=row_device, resource_class=resource_class,
        deterministic_worker=deterministic_worker,
    )
    return build_worker_runtime_measurement(
        campaign_spec_sha256=require_sha256(
            spec.get("content_hash"), name="runtime-measurement planning spec"
        ),
        data_root=data_root, resource_class=resource_class,
        resource_request=resource_request, live_worker_runtime=live,
    )


__all__ = [
    "LIVE_WORKER_RUNTIME_DOMAIN", "ROW_RUNTIME_SIGNATURE_DOMAIN",
    "TARGET_DETERMINISTIC_SEED", "WORKER_RUNTIME_MEASUREMENT_CONTRACT",
    "build_live_target_runtime_environment", "build_worker_runtime_measurement",
    "build_row_runtime_signature", "configure_deterministic_worker_backend",
    "measure_live_worker_runtime", "measure_registered_worker_runtime",
    "measure_weaver_runtime_source", "validate_live_task_runtime",
    "validate_live_worker_runtime", "validate_worker_runtime_measurement",
    "target_forward_runtime_commitment_from_measurement",
]
