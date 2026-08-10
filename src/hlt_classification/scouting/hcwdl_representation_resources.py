"""Frozen local planning resource classes and measured-profile validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import io
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import shutil
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .hcwdl_representation_contracts import (
    DENSE_STORAGE_ESTIMATE_CONTRACT,
    DENSE_STORAGE_TEMPLATE_CONTRACT,
    DENSE_RESOURCE_PROFILE_CONTRACT,
    FIXED_SIZE_INVENTORY_CONTRACT,
    MINIATURE_EVIDENCE_CONTRACT,
    NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    SCHEDULER_EVIDENCE_CONTRACT,
    STORAGE_ESTIMATE_CONTRACT,
    WORKER_RUNTIME_MEASUREMENT_CONTRACT,
)

TIGRIS_SITE: Final = "tigris.rc.rit.edu"
TIGRIS_ACCOUNT: Final = "reu-aisocial"
TIGRIS_PARTITION: Final = "tigris"
WORKER_ROLES: Final = ("ordinary", "deterministic")
DENSE_RESOURCE_CLASSES: Final = (
    "cpu_small", "cpu_io", "gpu_target", "gpu_representation",
)
SCHEDULER_EVIDENCE_ORIGINS: Final = (
    "local_fixture/v1",
    "tigris_sacct_raw/v1",
)
SACCT_FIELDS: Final = (
    "JobIDRaw",
    "JobName",
    "Account",
    "Partition",
    "Cluster",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "TimelimitRaw",
    "ReqCPUS",
    "ReqMem",
    "ReqTRES",
    "MaxRSS",
    "Comment",
    "SubmitLine",
)
SACCT_FORMAT: Final = ",".join(SACCT_FIELDS)
SACCT_COMMENT_PREFIX: Final = "hcwdl-rkd-evidence-v1:"
NONFINAL_SACCT_COMMENT_PREFIX: Final = "hcwdl-rkd-nonfinal-v1:"
FIXED_SIZE_KINDS: Final = (
    "retained_resume",
    "selected_checkpoint",
    "final_assignment",
    "fixed_artifact",
)
DENSE_STORAGE_NODE_COUNT: Final = 86
DENSE_STORAGE_TARGET_GENERATION_COUNT: Final = 83
DENSE_STORAGE_FIXED_METADATA_RESERVE_PER_NODE: Final = 16 * 1024 * 1024
DENSE_STORAGE_CAMPAIGN_FIXED_RESERVE: Final = 1024 * 1024 * 1024
NONFINAL_COLLECTOR_WORKER: Final = (
    "run_hcwdl_representation_nonfinal_evidence_collector.sh"
)
DENSE_RESOURCE_COLLECTOR_JOB_NAME: Final = "hcwdlr_resource_probe_collector"
NONFINAL_COLLECTOR_CLI: Final = (
    "build_hcwdl_representation_nonfinal_acceptance_scheduler_evidence.py"
)
NONFINAL_COLLECTOR_JOB_NAME: Final = "hcwdl-rkd-nonfinal-evidence-collector"
_TIGRIS_CAPTURE_HOST: Final = re.compile(
    r"^(?:tigris|g[gh]-[a-z]+-[0-9]+)(?:\.[A-Za-z0-9.-]+)?$"
)
_COLLECTOR_JOB_NAMES: Final = frozenset({
    DENSE_RESOURCE_COLLECTOR_JOB_NAME,
    NONFINAL_COLLECTOR_JOB_NAME,
})


@dataclass(frozen=True)
class ResourceRequest:
    cpus: int
    memory: str
    walltime: str
    gpu: str | None = None


PLANNING_RESOURCES: Final = {
    "cpu_small": ResourceRequest(2, "16G", "02:00:00"),
    "cpu_io": ResourceRequest(8, "128G", "24:00:00"),
    "gpu_target": ResourceRequest(8, "320G", "24:00:00", "gpu:gh200:1"),
    "gpu_representation": ResourceRequest(8, "320G", "48:00:00", "gpu:gh200:1"),
    "gpu_final_prediction": ResourceRequest(8, "192G", "24:00:00", "gpu:gh200:1"),
}

SMOKE_RESOURCES: Final = {
    "cpu_small": ResourceRequest(2, "8G", "00:30:00"),
    "cpu_io": ResourceRequest(2, "24G", "01:00:00"),
    "gpu_target": ResourceRequest(4, "64G", "02:00:00", "gpu:gh200:1"),
    "gpu_representation": ResourceRequest(4, "64G", "02:00:00", "gpu:gh200:1"),
    "gpu_final_prediction": ResourceRequest(4, "64G", "02:00:00", "gpu:gh200:1"),
}


def resource_table(*, mode: str) -> dict[str, dict[str, Any]]:
    source = SMOKE_RESOURCES if mode == "smoke" else PLANNING_RESOURCES
    return {name: asdict(request) for name, request in source.items()}


def _full_source_commit(value: object, *, name: str = "source commit") -> str:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return commit


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _memory_bytes(value: object) -> int:
    memory = str(value)
    if not memory.endswith("G") or not memory[:-1].isdigit():
        raise ValueError("resource memory request must use positive whole GiB")
    gibibytes = int(memory[:-1])
    if gibibytes <= 0:
        raise ValueError("resource memory request must use positive whole GiB")
    return gibibytes * 1024**3


def _walltime_seconds(value: object) -> int:
    fields = str(value).split(":")
    if len(fields) != 3 or any(not field.isdigit() for field in fields):
        raise ValueError("resource walltime request must use HH:MM:SS")
    hours, minutes, seconds = map(int, fields)
    if hours < 0 or minutes not in range(60) or seconds not in range(60):
        raise ValueError("resource walltime request differs")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("resource walltime request must be positive")
    return total


def _slurm_bytes(value: object, *, name: str) -> int:
    """Parse one raw Slurm memory field using its documented binary suffixes."""

    raw = str(value).strip()
    if raw.endswith(("c", "n")):
        raw = raw[:-1]
    if not raw:
        raise ValueError(f"raw sacct {name} is empty")
    suffix = raw[-1].upper() if raw[-1].isalpha() else ""
    number = raw[:-1] if suffix else raw
    multipliers = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if suffix not in multipliers:
        raise ValueError(f"raw sacct {name} unit differs")
    try:
        numeric = float(number)
    except ValueError as error:
        raise ValueError(f"raw sacct {name} is not numeric") from error
    result = numeric * multipliers[suffix]
    if not math.isfinite(result) or result <= 0 or not result.is_integer():
        raise ValueError(f"raw sacct {name} differs")
    return int(result)


def _requested_gpu_from_tres(value: object) -> str | None:
    """Normalize Slurm's current ReqTRES GPU entries to ``gpu:type:count``.

    Tigris removed the legacy ``ReqGRES`` accounting field.  Current
    ``ReqTRES`` rows include ordinary CPU/memory/billing entries plus both a
    generic ``gres/gpu`` count and the typed ``gres/gpu:<type>`` count.  Only
    the typed request is authoritative for this campaign.
    """

    raw = str(value).strip()
    if raw in {"", "(null)"}:
        return None
    generic: int | None = None
    typed: list[tuple[str, int]] = []
    for token in raw.split(","):
        key, separator, count_text = token.strip().partition("=")
        if not separator:
            raise ValueError("raw sacct ReqTRES entry differs")
        if key == "gres/gpu":
            if not count_text.isdigit() or int(count_text) <= 0:
                raise ValueError("raw sacct ReqTRES generic GPU count differs")
            generic = int(count_text)
        elif key.startswith("gres/gpu:"):
            gpu_type = key.removeprefix("gres/gpu:")
            if (
                not gpu_type
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", gpu_type)
                or not count_text.isdigit()
                or int(count_text) <= 0
            ):
                raise ValueError("raw sacct ReqTRES typed GPU count differs")
            typed.append((gpu_type, int(count_text)))
    if not typed:
        if generic is None:
            return None
        raise ValueError("raw sacct ReqTRES lacks the typed GPU request")
    if len(typed) != 1 or (generic is not None and generic != typed[0][1]):
        raise ValueError("raw sacct ReqTRES GPU request is ambiguous")
    return f"gpu:{typed[0][0]}:{typed[0][1]}"


def _scheduler_binding_payload(
    *, task_key: str, resource_class: str, source_commit: str,
    representation_recipe_sha256: str | None, worker_role: str,
    worker_sha256: str, request: Mapping[str, Any],
) -> dict[str, Any]:
    recipe = (
        None
        if representation_recipe_sha256 is None
        else require_sha256(
            representation_recipe_sha256, name="scheduler representation recipe",
        )
    )
    return {
        "hash_domain": "hcwdl-representation-scheduler-binding/v1",
        "schema_version": 1,
        "task_key": str(task_key),
        "resource_class": str(resource_class),
        "source_commit": _full_source_commit(source_commit),
        "representation_recipe_sha256": recipe,
        "worker_role": str(worker_role),
        "worker_sha256": require_sha256(worker_sha256, name="scheduler worker"),
        "request": {
            "cpus": int(request["cpus"]),
            "memory_bytes": _memory_bytes(request["memory"]),
            "walltime_seconds": _walltime_seconds(request["walltime"]),
            "gpu": request["gpu"],
        },
    }


def scheduler_evidence_comment(
    *, task_key: str, resource_class: str, source_commit: str,
    representation_recipe_sha256: str | None, worker_role: str,
    worker_sha256: str, request: Mapping[str, Any],
) -> str:
    """Return the exact pre-submission Slurm comment binding an evidence job."""

    return SACCT_COMMENT_PREFIX + canonical_sha256(_scheduler_binding_payload(
        task_key=task_key,
        resource_class=resource_class,
        source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role,
        worker_sha256=worker_sha256,
        request=request,
    ))


def _nonfinal_scheduler_binding_payload(
    *, authority_sha256: str, action_id: str, resource_class: str,
    source_commit: str, representation_recipe_sha256: str,
    worker_role: str, worker_sha256: str, request: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one bounded non-final action into its pre-submission comment."""

    if not action_id or action_id.startswith("train_") or "final/" in action_id:
        raise PermissionError("non-final acceptance action identity is forbidden")
    return {
        "hash_domain": "hcwdl-representation-nonfinal-scheduler-binding/v1",
        "schema_version": 1,
        "authority_sha256": require_sha256(
            authority_sha256, name="non-final acceptance authority",
        ),
        "action_id": str(action_id),
        **{
            name: value for name, value in _scheduler_binding_payload(
                task_key=f"acceptance-nonfinal-{action_id}",
                resource_class=resource_class,
                source_commit=source_commit,
                representation_recipe_sha256=representation_recipe_sha256,
                worker_role=worker_role,
                worker_sha256=worker_sha256,
                request=request,
            ).items()
            if name not in {"hash_domain", "schema_version"}
        },
    }


def nonfinal_acceptance_scheduler_comment(
    *, authority_sha256: str, action_id: str, resource_class: str,
    source_commit: str, representation_recipe_sha256: str,
    worker_role: str, worker_sha256: str, request: Mapping[str, Any],
) -> str:
    """Return the exact reviewed comment for one non-final acceptance action."""

    return NONFINAL_SACCT_COMMENT_PREFIX + canonical_sha256(
        _nonfinal_scheduler_binding_payload(
            authority_sha256=authority_sha256, action_id=action_id,
            resource_class=resource_class, source_commit=source_commit,
            representation_recipe_sha256=representation_recipe_sha256,
            worker_role=worker_role, worker_sha256=worker_sha256,
            request=request,
        )
    )


def _sacct_capture_command(job_id: int, *, executable: str = "sacct") -> list[str]:
    return [
        executable,
        "--jobs", str(job_id),
        "--starttime", "1970-01-01",
        "--duplicates",
        "--parsable2",
        "--units=K",
        f"--format={SACCT_FORMAT}",
    ]


def _validate_capture_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "site", "cluster", "collector_job_id", "capture_host",
        "account", "partition", "collector_job_name", "sacct_executable",
        "python_no_user_site", "conda_environment", "conda_prefix",
        "python_executable", "ld_library_path_prefix", "platform",
    }:
        raise PermissionError("raw sacct capture runtime fields differ")
    collector_job_id = _positive_integer(
        value["collector_job_id"], name="Slurm evidence-collector job ID",
    )
    conda_prefix = PurePosixPath(str(value["conda_prefix"]))
    expected_library_prefix = str(conda_prefix / "lib")
    if (
        value["site"] != TIGRIS_SITE
        or value["cluster"] != TIGRIS_PARTITION
        or _TIGRIS_CAPTURE_HOST.fullmatch(str(value["capture_host"])) is None
        or value["account"] != TIGRIS_ACCOUNT
        or value["partition"] != TIGRIS_PARTITION
        or value["collector_job_name"] not in _COLLECTOR_JOB_NAMES
        or value["sacct_executable"] != "/usr/bin/sacct"
        or value["python_no_user_site"] is not True
        or value["conda_environment"] != "atlas_kd_tigris"
        or not conda_prefix.is_absolute()
        or not PurePosixPath(str(value["python_executable"])).is_absolute()
        or value["ld_library_path_prefix"] != expected_library_prefix
        or value["platform"] != "posix"
    ):
        raise PermissionError("raw sacct capture did not run in the Tigris worker environment")
    return {**dict(value), "collector_job_id": collector_job_id}


def _live_tigris_capture_runtime() -> dict[str, Any]:
    """Fail closed unless the raw capture builder itself runs in a Tigris job."""

    if os.name != "posix":
        raise PermissionError("authorizing sacct evidence must be captured on Tigris")
    raw_job = os.environ.get("SLURM_JOB_ID", "")
    if not raw_job.isdigit():
        raise PermissionError("authorizing sacct capture lacks its collector Slurm job")
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    raw_sacct = Path(shutil.which("sacct") or "")
    sacct = raw_sacct.resolve()
    if (
        str(sacct) != "/usr/bin/sacct"
        or not sacct.is_file()
        or raw_sacct.is_symlink()
        or sacct.stat().st_uid != 0
        or not os.access(sacct, os.X_OK)
    ):
        raise PermissionError("authorizing sacct capture lacks the trusted Slurm client")
    library_prefix = str(conda_prefix / "lib")
    runtime = {
        "site": TIGRIS_SITE,
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "collector_job_id": int(raw_job),
        "capture_host": socket.getfqdn(),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "collector_job_name": os.environ.get("SLURM_JOB_NAME"),
        "sacct_executable": str(sacct),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": str(conda_prefix),
        "python_executable": str(Path(sys.executable).resolve()),
        "ld_library_path_prefix": (
            os.environ.get("LD_LIBRARY_PATH", "").split(":", 1)[0]
        ),
        "platform": os.name,
    }
    if runtime["ld_library_path_prefix"] != library_prefix:
        raise PermissionError("authorizing sacct capture lacks the Tigris library prefix")
    return _validate_capture_runtime(runtime)


def _parse_sacct_reference(
    reference: Mapping[str, Any], *, expected_worker_path: str,
    expected_comment: str, expected_task_key: str,
    exact_script_argv: bool = False,
) -> dict[str, Any]:
    """Parse an exact raw ``sacct --parsable2`` capture, never a caller summary."""

    path, _ = _validate_file_reference(reference, name="raw sacct accounting record")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("raw sacct accounting record is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    fieldnames = tuple(reader.fieldnames or ())
    expected_header = SACCT_FIELDS + ("",)
    if fieldnames not in (SACCT_FIELDS, expected_header):
        raise ValueError("raw sacct accounting header differs")
    rows = []
    for raw in reader:
        if "" in raw:
            trailing = raw.pop("")
            if trailing not in (None, "", [], [""]):
                raise ValueError("raw sacct accounting row has trailing data")
        if set(raw) != set(SACCT_FIELDS):
            raise ValueError("raw sacct accounting row fields differ")
        rows.append({name: str(raw[name] or "").strip() for name in SACCT_FIELDS})
    if not rows:
        raise ValueError("raw sacct accounting record is empty")
    parents = [row for row in rows if row["JobIDRaw"].isdigit()]
    if len(parents) != 1:
        raise ValueError("raw sacct accounting must contain one allocation row")
    parent = parents[0]
    job_id = _positive_integer(int(parent["JobIDRaw"]), name="Slurm job ID")
    allowed_prefix = f"{job_id}."
    if any(
        row["JobIDRaw"] != str(job_id)
        and not row["JobIDRaw"].startswith(allowed_prefix)
        for row in rows
    ):
        raise ValueError("raw sacct accounting mixes multiple jobs")
    expected_job_name = f"hcwdlr_{expected_task_key}"
    if parent["JobName"] != expected_job_name:
        raise PermissionError("raw sacct job name differs")
    raw_comment = parent["Comment"]
    if raw_comment not in {"", expected_comment}:
        raise PermissionError("raw sacct binding comment differs")
    try:
        submit_tokens = shlex.split(parent["SubmitLine"], posix=True)
    except ValueError as error:
        raise ValueError("raw sacct submit line cannot be parsed") from error
    normalized_worker = str(Path(expected_worker_path))
    if exact_script_argv:
        # The bounded workers consume authority/action through their frozen
        # environment and accept no script arguments.  Requiring the reviewed
        # worker to be the sole positional and final token prevents a command
        # such as ``sbatch evil.sh /path/to/reviewed-worker.sh`` from acquiring
        # the reviewed worker's lineage after the fact.  Long options are
        # deliberately frozen to ``--name=value`` form (plus ``--parsable``),
        # so there is no ambiguous option-value/positional parse.
        if (
            not submit_tokens
            or PurePosixPath(submit_tokens[0]).name != "sbatch"
            or len(submit_tokens) < 2
            or submit_tokens[-1] != normalized_worker
            or submit_tokens.count(normalized_worker) != 1
            or any(
                token != "--parsable"
                and re.fullmatch(r"--[a-z0-9][a-z0-9-]*=.+", token) is None
                for token in submit_tokens[1:-1]
            )
            or any(token.startswith("--wrap") for token in submit_tokens[1:-1])
        ):
            raise PermissionError("raw sacct submit line does not execute the exact worker argv")
        required_options = {
            f"--job-name={expected_job_name}",
            f"--account={TIGRIS_ACCOUNT}",
            f"--partition={TIGRIS_PARTITION}",
            f"--comment={expected_comment}",
        }
        if not required_options.issubset(set(submit_tokens[1:-1])):
            raise PermissionError("raw sacct submit line lacks exact bound Slurm options")
        # Current Tigris preserves the submitted binding in ``SubmitLine`` but
        # returns an empty ``Comment`` accounting column.  The exact comment
        # option above remains mandatory; an empty column never weakens or
        # replaces that authenticated command-line binding.
    elif raw_comment != expected_comment:
        raise PermissionError("raw sacct binding comment differs")
    elif normalized_worker not in submit_tokens:
        raise PermissionError("raw sacct submit line lacks the exact production worker")
    completed = [row for row in rows if row["State"].split("+", 1)[0] == "COMPLETED"]
    if len(completed) != len(rows) or any(row["ExitCode"] != "0:0" for row in rows):
        raise PermissionError("raw sacct job or step did not complete successfully")
    rss_values = [
        _slurm_bytes(row["MaxRSS"], name="MaxRSS")
        for row in rows if row["MaxRSS"]
    ]
    if not rss_values:
        raise PermissionError("raw sacct accounting lacks measured MaxRSS")
    return {
        "job_id": job_id,
        "job_name": parent["JobName"],
        "account": parent["Account"],
        "partition": parent["Partition"],
        "cluster": parent["Cluster"],
        "state": parent["State"].split("+", 1)[0],
        "exit_code": parent["ExitCode"],
        "elapsed_seconds": _positive_integer(
            int(parent["ElapsedRaw"]), name="raw sacct elapsed seconds",
        ),
        "timelimit_minutes": _positive_integer(
            int(parent["TimelimitRaw"]), name="raw sacct timelimit minutes",
        ),
        "requested_cpus": _positive_integer(
            int(parent["ReqCPUS"]), name="raw sacct requested CPUs",
        ),
        "requested_memory_bytes": _slurm_bytes(parent["ReqMem"], name="ReqMem"),
        "requested_gpu": _requested_gpu_from_tres(parent["ReqTRES"]),
        "peak_rss_bytes": max(rss_values),
        "binding_comment": expected_comment,
        "submit_line": parent["SubmitLine"],
        "submit_argv": submit_tokens,
    }


def artifact_reference(path: str | Path) -> dict[str, str]:
    """Return an absolute byte-authenticated reference to an existing file."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_file_reference(
    value: Mapping[str, Any], *, name: str,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must be an exact path/SHA-256 reference")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = require_sha256(value["sha256"], name=f"{name} bytes")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} byte hash differs")
    return path, observed


def load_authenticated_json_reference(
    value: Mapping[str, Any], *, expected_contract: str, name: str,
    expected_schema_version: int = 1,
) -> tuple[dict[str, Any], str]:
    """Open a referenced JSON artifact after authenticating its exact bytes."""

    path, _ = _validate_file_reference(value, name=name)
    artifact = load_json(path)
    digest = validate_content_hash(
        artifact,
        expected_contract=expected_contract,
        expected_schema_version=expected_schema_version,
    )
    return artifact, digest


def build_fixed_size_inventory(
    *, parent_import_sha256: str,
    files_by_kind: Mapping[str, Sequence[str | Path]],
) -> dict[str, Any]:
    """Inventory the exact files behind the four measured fixed-size classes."""

    if set(files_by_kind) != set(FIXED_SIZE_KINDS):
        raise ValueError("fixed-size inventory classes differ")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    totals = {kind: 0 for kind in FIXED_SIZE_KINDS}
    for kind in FIXED_SIZE_KINDS:
        paths = tuple(files_by_kind[kind])
        if not paths:
            raise ValueError(f"fixed-size inventory {kind!r} is empty")
        for raw_path in paths:
            reference = artifact_reference(raw_path)
            path = str(Path(reference["path"]).resolve())
            if path in seen:
                raise ValueError("fixed-size inventory repeats an artifact path")
            seen.add(path)
            size = Path(path).stat().st_size
            if size <= 0:
                raise ValueError("fixed-size inventory contains an empty artifact")
            entries.append({
                "kind": kind,
                "path": path,
                "sha256": reference["sha256"],
                "size_bytes": size,
            })
            totals[kind] += size
    return with_content_hash({
        "contract": FIXED_SIZE_INVENTORY_CONTRACT,
        "schema_version": 1,
        "parent_import_sha256": require_sha256(
            parent_import_sha256, name="parent import",
        ),
        "entries": entries,
        "category_totals_bytes": totals,
    })


def validate_fixed_size_inventory(
    value: Mapping[str, Any], *, parent_import_sha256: str | None = None,
) -> tuple[str, dict[str, int]]:
    digest = validate_content_hash(
        value,
        expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
        expected_schema_version=1,
    )
    parent = require_sha256(value.get("parent_import_sha256"), name="parent import")
    if parent_import_sha256 is not None and parent != parent_import_sha256:
        raise ValueError("fixed-size inventory parent import differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("fixed-size inventory entries differ")
    totals = {kind: 0 for kind in FIXED_SIZE_KINDS}
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, Mapping) or set(row) != {
            "kind", "path", "sha256", "size_bytes",
        }:
            raise ValueError("fixed-size inventory entry differs")
        kind = str(row["kind"])
        if kind not in totals:
            raise ValueError("fixed-size inventory entry class differs")
        path, _ = _validate_file_reference(
            {"path": row["path"], "sha256": row["sha256"]},
            name=f"fixed-size {kind} artifact",
        )
        resolved = str(path.resolve())
        if resolved in seen:
            raise ValueError("fixed-size inventory repeats an artifact path")
        seen.add(resolved)
        size = _positive_integer(row["size_bytes"], name="fixed-size artifact bytes")
        if path.stat().st_size != size:
            raise ValueError("fixed-size inventory artifact size differs")
        totals[kind] += size
    if any(total <= 0 for total in totals.values()):
        raise ValueError("fixed-size inventory lacks a required class")
    if value.get("category_totals_bytes") != totals:
        raise ValueError("fixed-size inventory category totals differ")
    return digest, totals


def _nonnegative_integer(value: int, *, name: str) -> int:
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"storage {name} count must be a nonnegative integer")
    return result


def build_storage_estimate(
    *, train_rows: int, validation_rows: int, final_rows: int,
    parent_import_sha256: str, prediction_finalists: int,
    retained_resume_bytes: int = 0,
    selected_checkpoint_bytes: int = 0,
    final_assignment_bytes: int = 0,
    fixed_artifact_bytes: int = 0,
    interrupted_target_reserve_bytes: int | None = None,
    fixed_size_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inventory_reference: dict[str, Any] | None = None
    inventory_sha256: str | None = None
    if fixed_size_inventory is not None:
        inventory, inventory_sha256 = load_authenticated_json_reference(
            fixed_size_inventory,
            expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
            name="fixed-size inventory",
        )
        _, inventory_totals = validate_fixed_size_inventory(
            inventory, parent_import_sha256=parent_import_sha256,
        )
        explicitly_supplied = (
            retained_resume_bytes,
            selected_checkpoint_bytes,
            final_assignment_bytes,
            fixed_artifact_bytes,
        )
        if any(int(value) != 0 for value in explicitly_supplied):
            raise ValueError(
                "measured fixed sizes must be derived only from their inventory"
            )
        retained_resume_bytes = inventory_totals["retained_resume"]
        selected_checkpoint_bytes = inventory_totals["selected_checkpoint"]
        final_assignment_bytes = inventory_totals["final_assignment"]
        fixed_artifact_bytes = inventory_totals["fixed_artifact"]
        inventory_reference = dict(fixed_size_inventory)
    counts = {
        "train": train_rows, "validation": validation_rows, "final": final_rows,
        "finalists": prediction_finalists,
        "retained_resume_bytes": retained_resume_bytes,
        "selected_checkpoint_bytes": selected_checkpoint_bytes,
        "final_assignment_bytes": final_assignment_bytes,
        "fixed_artifact_bytes": fixed_artifact_bytes,
    }
    normalized = {
        name: _nonnegative_integer(value, name=name) for name, value in counts.items()
    }
    ordinary_bytes = normalized["train"] * 7_815
    toff_bytes = normalized["train"] * 15_021
    # The immutable prediction payload is the 32-byte canonical identity digest
    # plus fifteen FP32 logits.  Probabilities and labels are deliberately absent.
    prediction_row_bytes = 32 + 15 * 4
    prediction_bytes = (
        normalized["final"] * prediction_row_bytes * normalized["finalists"]
    )
    # Only one committed target generation can be live.  The estimator also
    # reserves one interrupted/recovery generation, as required by the plan.
    interrupted = (
        max(ordinary_bytes, toff_bytes)
        if interrupted_target_reserve_bytes is None
        else _nonnegative_integer(
            interrupted_target_reserve_bytes, name="interrupted_target_reserve_bytes",
        )
    )
    committed_target = max(ordinary_bytes, toff_bytes)
    subtotal = sum((
        committed_target, interrupted, prediction_bytes,
        normalized["retained_resume_bytes"], normalized["selected_checkpoint_bytes"],
        normalized["final_assignment_bytes"], normalized["fixed_artifact_bytes"],
    ))
    # The filesystem authorization is based on integer bytes and reserves the
    # exact ceiling of 25% headroom.  No compression saving is assumed.
    headroom = (subtotal + 3) // 4
    total = subtotal + headroom
    return with_content_hash(
        {
            "contract": STORAGE_ESTIMATE_CONTRACT,
            "schema_version": 1,
            "parent_import_sha256": require_sha256(parent_import_sha256, name="parent import"),
            "row_counts": {
                "train": normalized["train"], "validation": normalized["validation"],
                "final": normalized["final"],
            },
            "logical_bytes_per_ordinary_row": 7_815,
            "logical_bytes_per_toff_row": 15_021,
            "ordinary_bank_bytes": ordinary_bytes,
            "toff_bank_bytes": toff_bytes,
            "committed_target_generation_bytes": committed_target,
            "interrupted_target_generation_reserve_bytes": interrupted,
            "peak_target_staging_plus_committed_bytes": committed_target + interrupted,
            "prediction_identity_bytes_per_row": 32,
            "prediction_logit_bytes_per_row": 15 * 4,
            "prediction_payload_bytes_per_row": prediction_row_bytes,
            "prediction_finalists": normalized["finalists"],
            "prediction_logits_bytes": prediction_bytes,
            "retained_resume_bytes": normalized["retained_resume_bytes"],
            "selected_checkpoint_bytes": normalized["selected_checkpoint_bytes"],
            "final_assignment_bytes": normalized["final_assignment_bytes"],
            "fixed_artifact_bytes": normalized["fixed_artifact_bytes"],
            "subtotal_before_filesystem_headroom_bytes": subtotal,
            "filesystem_headroom_numerator": 1,
            "filesystem_headroom_denominator": 4,
            "filesystem_headroom_bytes": headroom,
            "estimated_campaign_peak_durable_bytes": total,
            "simultaneously_committed_target_banks": 1,
            "compression_saving_assumed": False,
            "operational_fixed_sizes_measured": all(
                normalized[name] > 0
                for name in (
                    "retained_resume_bytes", "selected_checkpoint_bytes",
                    "final_assignment_bytes", "fixed_artifact_bytes",
                )
            ),
        }
    )


def validate_storage_estimate(
    value: Mapping[str, Any], *, require_measured_fixed_sizes: bool = False,
    fixed_size_inventory: Mapping[str, Any] | None = None,
) -> str:
    """Recompute every byte count instead of trusting a self-hashed estimate."""

    digest = validate_content_hash(
        value, expected_contract=STORAGE_ESTIMATE_CONTRACT,
        expected_schema_version=1,
    )
    counts = value.get("row_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "train", "validation", "final",
    }:
        raise ValueError("representation storage row counts differ")
    common = dict(
        train_rows=counts["train"], validation_rows=counts["validation"],
        final_rows=counts["final"],
        parent_import_sha256=value.get("parent_import_sha256"),
        prediction_finalists=value.get("prediction_finalists"),
        interrupted_target_reserve_bytes=value.get(
            "interrupted_target_generation_reserve_bytes"
        ),
    )
    expected = build_storage_estimate(
        **common,
        retained_resume_bytes=value.get("retained_resume_bytes"),
        selected_checkpoint_bytes=value.get("selected_checkpoint_bytes"),
        final_assignment_bytes=value.get("final_assignment_bytes"),
        fixed_artifact_bytes=value.get("fixed_artifact_bytes"),
    )
    if dict(value) != expected:
        raise ValueError("representation storage estimate is not canonically derived")
    if fixed_size_inventory is not None:
        inventory, _ = load_authenticated_json_reference(
            fixed_size_inventory,
            expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
            name="fixed-size inventory",
        )
        _, totals = validate_fixed_size_inventory(
            inventory, parent_import_sha256=str(value["parent_import_sha256"]),
        )
        expected_totals = {
            "retained_resume": value["retained_resume_bytes"],
            "selected_checkpoint": value["selected_checkpoint_bytes"],
            "final_assignment": value["final_assignment_bytes"],
            "fixed_artifact": value["fixed_artifact_bytes"],
        }
        if totals != expected_totals or value.get(
            "operational_fixed_sizes_measured"
        ) is not True:
            raise PermissionError(
                "representation storage fixed sizes differ from their inventory"
            )
    elif require_measured_fixed_sizes:
        raise PermissionError(
            "representation storage estimate lacks its authenticated fixed-size inventory"
        )
    return digest


def _dense_template_reference(value: Mapping[str, Any], *, name: str) -> tuple[Path, int]:
    path, _ = _validate_file_reference(value, name=name)
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{name} is empty")
    return path, size


def build_dense_storage_template(
    *, source_commit: str, planning_spec_sha256: str,
    representation_recipe_sha256: str, graph_sha256: str,
    dense_teacher_import_sha256: str,
    resume_state_template: Mapping[str, Any],
    selected_checkpoint_template: Mapping[str, Any],
    deployable_checkpoint_template: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact serialized maximum-topology files measured by one probe job.

    The template deliberately reserves the pilot-sized 300k identity registry,
    even when it is produced during the bounded smoke bootstrap.  The three
    referenced files are immutable worker outputs; the genuine scheduler
    miniature binds this artifact as the ``gpu_representation`` probe result.
    """

    references = {
        "resume_state": dict(resume_state_template),
        "selected_checkpoint": dict(selected_checkpoint_template),
        "deployable_checkpoint": dict(deployable_checkpoint_template),
    }
    sizes = {
        name: _dense_template_reference(reference, name=f"dense {name} template")[1]
        for name, reference in references.items()
    }
    return with_content_hash({
        "contract": DENSE_STORAGE_TEMPLATE_CONTRACT,
        "schema_version": 1,
        "source_commit": _full_source_commit(source_commit),
        "planning_spec_sha256": require_sha256(
            planning_spec_sha256, name="dense storage planning spec",
        ),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="dense storage recipe",
        ),
        "graph_sha256": require_sha256(graph_sha256, name="dense storage graph"),
        "dense_teacher_import_sha256": require_sha256(
            dense_teacher_import_sha256, name="dense storage teacher import",
        ),
        "maximum_topology_execution_id": "RREL_D100",
        "pilot_identity_rows_reserved": 300_000,
        "identity_digest_bytes_per_row": 32,
        "optimizer_state_initialized": True,
        "serialization": "torch_canonical_pickle/v1",
        "templates": references,
        "template_sizes_bytes": sizes,
        "fixed_metadata_reserve_per_node_bytes": (
            DENSE_STORAGE_FIXED_METADATA_RESERVE_PER_NODE
        ),
        "genuine_worker_measurement_required": True,
        "final_role_accessed": False,
    })


def validate_dense_storage_template(
    value: Mapping[str, Any], *, expected_source_commit: str | None = None,
    expected_recipe_sha256: str | None = None,
    expected_graph_sha256: str | None = None,
    expected_dense_teacher_import_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_STORAGE_TEMPLATE_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "source_commit", "planning_spec_sha256",
        "representation_recipe_sha256", "graph_sha256",
        "dense_teacher_import_sha256", "maximum_topology_execution_id",
        "pilot_identity_rows_reserved", "identity_digest_bytes_per_row",
        "optimizer_state_initialized", "serialization", "templates",
        "template_sizes_bytes", "fixed_metadata_reserve_per_node_bytes",
        "genuine_worker_measurement_required", "final_role_accessed",
        "content_hash",
    }
    if set(value) != required:
        raise ValueError("dense storage template fields differ")
    source_commit = _full_source_commit(str(value.get("source_commit")))
    expected = {
        "source_commit": (
            source_commit if expected_source_commit is None
            else _full_source_commit(expected_source_commit)
        ),
        "representation_recipe_sha256": (
            value.get("representation_recipe_sha256")
            if expected_recipe_sha256 is None else require_sha256(
                expected_recipe_sha256, name="expected dense storage recipe",
            )
        ),
        "graph_sha256": (
            value.get("graph_sha256") if expected_graph_sha256 is None
            else require_sha256(expected_graph_sha256, name="expected dense graph")
        ),
        "dense_teacher_import_sha256": (
            value.get("dense_teacher_import_sha256")
            if expected_dense_teacher_import_sha256 is None else require_sha256(
                expected_dense_teacher_import_sha256,
                name="expected dense teacher import",
            )
        ),
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("dense storage template lineage differs")
    require_sha256(value.get("planning_spec_sha256"), name="dense storage planning spec")
    templates = value.get("templates")
    if not isinstance(templates, Mapping) or set(templates) != {
        "resume_state", "selected_checkpoint", "deployable_checkpoint",
    }:
        raise ValueError("dense storage template registry differs")
    sizes = {
        name: _dense_template_reference(reference, name=f"dense {name} template")[1]
        for name, reference in templates.items()
    }
    if value.get("template_sizes_bytes") != sizes:
        raise PermissionError("dense storage template byte sizes differ")
    if (
        value.get("maximum_topology_execution_id") != "RREL_D100"
        or value.get("pilot_identity_rows_reserved") != 300_000
        or value.get("identity_digest_bytes_per_row") != 32
        or value.get("optimizer_state_initialized") is not True
        or value.get("serialization") != "torch_canonical_pickle/v1"
        or value.get("fixed_metadata_reserve_per_node_bytes")
        != DENSE_STORAGE_FIXED_METADATA_RESERVE_PER_NODE
        or value.get("genuine_worker_measurement_required") is not True
        or value.get("final_role_accessed") is not False
    ):
        raise PermissionError("dense storage template semantics differ")
    return digest


def measure_dense_storage_template(
    *, planning_spec: Mapping[str, Any], output_root: str | Path,
) -> dict[str, Any]:
    """Serialize the largest dense state using the installed production model.

    This performs no data read and no scientific optimizer update.  AdamW
    moments are materialized directly so the serialized template has the same
    tensor inventory as a post-update checkpoint.
    """

    from .hcwdl_representation_campaign import (
        DENSE_TRAINING_DISPOSITION, validate_campaign_spec,
    )
    spec_sha256 = validate_campaign_spec(planning_spec, executable=False)
    if (
        planning_spec.get("mode") != "smoke"
        or planning_spec.get("disposition") != DENSE_TRAINING_DISPOSITION
        or planning_spec.get("role_counts")
        != {"train": 512, "validation": 256, "final_test": 0}
    ):
        raise PermissionError("dense storage measurement requires the smoke plan")
    import numpy as np
    import torch
    from .hcwdl_representation_training import (
        _cpu_tree, _head_state, _torch_bytes, initialize_representation_student,
    )

    model = initialize_representation_student("RREL_D100", replicate_seed=1337)
    model.cpu()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    for parameter in parameters:
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(
                parameter, memory_format=torch.preserve_format,
            ),
        }
    deployable = _cpu_tree(model.deployable_model.state_dict())
    state = {
        "contract": "HCWDL_REPRESENTATION_DENSE_STORAGE_STATE_TEMPLATE/v1",
        "execution_id": "RREL_D100",
        "deployable_model": deployable,
        "representation_heads": _head_state(model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "pilot_identity_registry": np.zeros((300_000, 32), dtype=np.uint8),
        # This fixed reserve dominates the JSON sidecars, validation history,
        # calibration records, RNG state and immutable envelope metadata.
        "metadata_reserve": bytes(DENSE_STORAGE_FIXED_METADATA_RESERVE_PER_NODE),
    }
    root = Path(output_root).resolve()
    paths = {
        "resume_state": root / "resume_state_template.pt",
        "selected_checkpoint": root / "selected_checkpoint_template.pt",
        "deployable_checkpoint": root / "deployable_checkpoint_template.pt",
    }
    payloads = {
        "resume_state": _torch_bytes(state),
        "selected_checkpoint": _torch_bytes({
            "contract": "HCWDL_REPRESENTATION_DENSE_SELECTED_TEMPLATE/v1",
            "state": state,
        }),
        "deployable_checkpoint": _torch_bytes(deployable),
    }
    for name, path in paths.items():
        if atomic_publish_bytes(path, payloads[name]) != "published":
            raise FileExistsError(f"dense storage {name} template already exists")
    result = build_dense_storage_template(
        source_commit=str(planning_spec["source_commit"]),
        planning_spec_sha256=spec_sha256,
        representation_recipe_sha256=str(
            planning_spec["representation_recipe_sha256"]
        ),
        graph_sha256=str(planning_spec["graph_sha256"]),
        dense_teacher_import_sha256=str(planning_spec["parent_import_sha256"]),
        resume_state_template=artifact_reference(paths["resume_state"]),
        selected_checkpoint_template=artifact_reference(paths["selected_checkpoint"]),
        deployable_checkpoint_template=artifact_reference(
            paths["deployable_checkpoint"]
        ),
    )
    validate_dense_storage_template(
        result, expected_source_commit=str(planning_spec["source_commit"]),
        expected_recipe_sha256=str(planning_spec["representation_recipe_sha256"]),
        expected_graph_sha256=str(planning_spec["graph_sha256"]),
        expected_dense_teacher_import_sha256=str(
            planning_spec["parent_import_sha256"]
        ),
    )
    return result


def build_dense_storage_estimate(
    *, train_rows: int, validation_rows: int,
    dense_teacher_import_sha256: str,
    storage_template: Mapping[str, Any],
) -> dict[str, Any]:
    template, template_sha256 = load_authenticated_json_reference(
        storage_template, expected_contract=DENSE_STORAGE_TEMPLATE_CONTRACT,
        name="dense storage template",
    )
    validate_dense_storage_template(
        template,
        expected_dense_teacher_import_sha256=dense_teacher_import_sha256,
    )
    train = _positive_integer(train_rows, name="dense storage train rows")
    validation = _positive_integer(
        validation_rows, name="dense storage validation rows",
    )
    if (train, validation) not in {(512, 256), (300_000, 100_000)}:
        raise ValueError("dense storage role populations differ")
    sizes = template["template_sizes_bytes"]
    per_node_persistent = (
        2 * int(sizes["resume_state"])
        + int(sizes["selected_checkpoint"])
        + int(sizes["deployable_checkpoint"])
        + DENSE_STORAGE_FIXED_METADATA_RESERVE_PER_NODE
    )
    all_node_persistent = DENSE_STORAGE_NODE_COUNT * per_node_persistent
    transient_training = int(sizes["resume_state"]) + int(
        sizes["selected_checkpoint"]
    )
    ordinary_bank = train * 7_815
    toff_bank = train * 15_021
    target_peak = 2 * max(ordinary_bank, toff_bank)
    retained_target_metadata = DENSE_STORAGE_TARGET_GENERATION_COUNT * 1024 * 1024
    subtotal = (
        all_node_persistent + transient_training + target_peak
        + retained_target_metadata + DENSE_STORAGE_CAMPAIGN_FIXED_RESERVE
    )
    # A 50% reserve covers filesystem block allocation, immutable JSON,
    # envelope duplication and future failure/recovery staging without relying
    # on compression.
    headroom = (subtotal + 1) // 2
    total = subtotal + headroom
    return with_content_hash({
        "contract": DENSE_STORAGE_ESTIMATE_CONTRACT,
        "schema_version": 1,
        "disposition": "dense_training_only",
        "dense_teacher_import_sha256": require_sha256(
            dense_teacher_import_sha256, name="dense storage teacher import",
        ),
        "storage_template": dict(storage_template),
        "storage_template_sha256": template_sha256,
        "row_counts": {"train": train, "validation": validation, "final": 0},
        "training_node_count": DENSE_STORAGE_NODE_COUNT,
        "target_generation_count": DENSE_STORAGE_TARGET_GENERATION_COUNT,
        "simultaneously_committed_target_banks": 1,
        "interrupted_target_generation_reserve": 1,
        "ordinary_target_bank_bytes": ordinary_bank,
        "toff_target_bank_bytes": toff_bank,
        "peak_target_staging_plus_committed_bytes": target_peak,
        "per_node_persistent_bytes": per_node_persistent,
        "all_nodes_persistent_bytes": all_node_persistent,
        "transient_training_staging_bytes": transient_training,
        "retained_target_metadata_bytes": retained_target_metadata,
        "campaign_fixed_reserve_bytes": DENSE_STORAGE_CAMPAIGN_FIXED_RESERVE,
        "subtotal_before_filesystem_headroom_bytes": subtotal,
        "filesystem_headroom_numerator": 1,
        "filesystem_headroom_denominator": 2,
        "filesystem_headroom_bytes": headroom,
        "estimated_campaign_peak_durable_bytes": total,
        "minimum_free_bytes_at_submission": total,
        "compression_saving_assumed": False,
        "operational_template_measured": True,
        "final_role_storage_bytes": 0,
    })


def validate_dense_storage_estimate(
    value: Mapping[str, Any], *, storage_template: Mapping[str, Any] | None = None,
    expected_source_commit: str | None = None,
    expected_recipe_sha256: str | None = None,
    expected_graph_sha256: str | None = None,
    expected_dense_teacher_import_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=DENSE_STORAGE_ESTIMATE_CONTRACT,
        expected_schema_version=1,
    )
    reference = value.get("storage_template")
    if not isinstance(reference, Mapping):
        raise ValueError("dense storage estimate template reference differs")
    template_reference = dict(reference) if storage_template is None else dict(
        storage_template
    )
    if dict(reference) != template_reference:
        raise PermissionError("dense storage estimate template differs")
    template, template_sha256 = load_authenticated_json_reference(
        template_reference, expected_contract=DENSE_STORAGE_TEMPLATE_CONTRACT,
        name="dense storage template",
    )
    validate_dense_storage_template(
        template, expected_source_commit=expected_source_commit,
        expected_recipe_sha256=expected_recipe_sha256,
        expected_graph_sha256=expected_graph_sha256,
        expected_dense_teacher_import_sha256=(
            expected_dense_teacher_import_sha256
            if expected_dense_teacher_import_sha256 is not None
            else str(value.get("dense_teacher_import_sha256"))
        ),
    )
    counts = value.get("row_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "train", "validation", "final",
    } or counts.get("final") != 0:
        raise ValueError("dense storage estimate row counts differ")
    expected = build_dense_storage_estimate(
        train_rows=int(counts["train"]),
        validation_rows=int(counts["validation"]),
        dense_teacher_import_sha256=str(value.get("dense_teacher_import_sha256")),
        storage_template=template_reference,
    )
    if dict(value) != expected or value.get("storage_template_sha256") != template_sha256:
        raise PermissionError("dense storage estimate is not canonically derived")
    return digest


def validate_dense_storage_availability(
    value: Mapping[str, Any], *, campaign_root: str | Path,
) -> int:
    """Recheck live free bytes immediately before review or scheduler mutation."""

    validate_dense_storage_estimate(value)
    root = Path(campaign_root).resolve()
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    required = _positive_integer(
        value.get("minimum_free_bytes_at_submission"),
        name="dense storage minimum free bytes",
    )
    if free < required:
        raise PermissionError(
            f"dense campaign requires {required} free bytes but only {free} are available"
        )
    return free


def validate_scheduler_evidence(
    value: Mapping[str, Any], *, resource_class: str,
    request: Mapping[str, Any], expected_source_commit: str,
    expected_workers: Mapping[str, Mapping[str, Any]],
    expected_recipe_sha256: str | None = None,
    require_genuine: bool = False,
) -> dict[str, Any]:
    """Validate scheduler evidence, optionally requiring raw Tigris accounting.

    ``local_fixture/v1`` records deliberately remain useful for unit tests but
    can never satisfy a genuine-resource or pilot-acceptance gate.  The
    authorizing form is reconstructed from referenced raw ``sacct`` bytes;
    completion, exit status, requests, RSS, elapsed time, job identity, and
    the pre-submission lineage comment are not accepted as caller booleans.
    """

    validate_content_hash(
        value,
        expected_contract=SCHEDULER_EVIDENCE_CONTRACT,
        expected_schema_version=1,
    )
    if set(value) != {
        "contract", "schema_version", "job_id", "task_key", "resource_class",
        "site", "account", "partition", "source_commit", "python_no_user_site",
        "representation_recipe_sha256", "worker_role", "worker",
        "requested_cpus", "requested_memory_bytes",
        "requested_walltime_seconds", "requested_gpu", "state", "exit_code",
        "peak_rss_bytes", "elapsed_seconds", "evidence_origin",
        "raw_accounting_record", "capture_command", "job_name",
        "binding_comment", "submit_line", "capture_runtime",
        "authorization_capable",
        "content_hash",
    }:
        raise PermissionError("scheduler evidence fields differ")
    job_id = _positive_integer(value["job_id"], name="Slurm job ID")
    if not str(value["task_key"]):
        raise PermissionError("scheduler evidence task key is empty")
    if value["resource_class"] != resource_class:
        raise PermissionError("scheduler evidence resource class differs")
    if (
        value["site"] != TIGRIS_SITE
        or value["account"] != TIGRIS_ACCOUNT
        or value["partition"] != TIGRIS_PARTITION
        or value["python_no_user_site"] is not True
    ):
        raise PermissionError("scheduler evidence Tigris environment differs")
    if _full_source_commit(value["source_commit"]) != _full_source_commit(
        expected_source_commit
    ):
        raise PermissionError("scheduler evidence source commit differs")
    recipe = value["representation_recipe_sha256"]
    if recipe is not None:
        recipe = require_sha256(recipe, name="scheduler representation recipe")
    if expected_recipe_sha256 is not None and recipe != require_sha256(
        expected_recipe_sha256, name="expected scheduler representation recipe",
    ):
        raise PermissionError("scheduler evidence representation recipe differs")
    role = str(value["worker_role"])
    if role not in WORKER_ROLES or role not in expected_workers:
        raise PermissionError("scheduler evidence worker role differs")
    if value["worker"] != expected_workers[role]:
        raise PermissionError("scheduler evidence worker reference differs")
    _validate_file_reference(value["worker"], name=f"{role} production worker")
    if isinstance(request.get("cpus"), bool) or int(request["cpus"]) <= 0:
        raise ValueError("resource CPU request differs")
    if value["requested_cpus"] != int(request["cpus"]):
        raise PermissionError("scheduler evidence CPU request differs")
    requested_memory = _positive_integer(
        value["requested_memory_bytes"], name="requested memory bytes",
    )
    if requested_memory != _memory_bytes(request["memory"]):
        raise PermissionError("scheduler evidence memory request differs")
    requested_walltime = _positive_integer(
        value["requested_walltime_seconds"], name="requested walltime seconds",
    )
    if requested_walltime != _walltime_seconds(request["walltime"]):
        raise PermissionError("scheduler evidence walltime request differs")
    if value["requested_gpu"] != request["gpu"]:
        raise PermissionError("scheduler evidence GPU request differs")
    if value["state"] != "COMPLETED" or value["exit_code"] != "0:0":
        raise PermissionError("scheduler evidence job did not complete successfully")
    peak_rss = _positive_integer(value["peak_rss_bytes"], name="peak RSS bytes")
    elapsed = value["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError("scheduler elapsed seconds must be numeric")
    elapsed = float(elapsed)
    if not 0 < elapsed <= requested_walltime:
        raise PermissionError("scheduler elapsed time exceeds requested walltime")
    if peak_rss > (3 * requested_memory) // 4:
        raise PermissionError("scheduler peak RSS exceeds the 75% memory budget")
    origin = str(value["evidence_origin"])
    if origin not in SCHEDULER_EVIDENCE_ORIGINS:
        raise PermissionError("scheduler evidence origin differs")
    if origin == "local_fixture/v1":
        if (
            value["raw_accounting_record"] is not None
            or value["capture_command"] is not None
            or value["binding_comment"] is not None
            or value["submit_line"] is not None
            or value["capture_runtime"] is not None
            or value["authorization_capable"] is not False
        ):
            raise PermissionError("local scheduler fixture claims external authority")
        if value["job_name"] != f"hcwdlr_{value['task_key']}":
            raise PermissionError("local scheduler fixture job name differs")
        if require_genuine:
            raise PermissionError(
                "scheduler evidence is a nonauthorizing local fixture, not raw sacct"
            )
        return dict(value)

    if value["authorization_capable"] is not True:
        raise PermissionError("raw scheduler evidence is marked nonauthorizing")
    _validate_capture_runtime(value["capture_runtime"])
    worker_path, worker_sha256 = _validate_file_reference(
        value["worker"], name=f"{role} production worker",
    )
    expected_comment = scheduler_evidence_comment(
        task_key=str(value["task_key"]), resource_class=resource_class,
        source_commit=expected_source_commit,
        representation_recipe_sha256=recipe,
        worker_role=role, worker_sha256=worker_sha256, request=request,
    )
    parsed = _parse_sacct_reference(
        value["raw_accounting_record"], expected_worker_path=str(worker_path),
        expected_comment=expected_comment, expected_task_key=str(value["task_key"]),
        exact_script_argv=True,
    )
    expected_capture = _sacct_capture_command(parsed["job_id"])
    if value["capture_command"] != expected_capture:
        raise PermissionError("raw sacct capture command differs")
    extracted = {
        "job_id": parsed["job_id"],
        "job_name": parsed["job_name"],
        "account": parsed["account"],
        "partition": parsed["partition"],
        "state": parsed["state"],
        "exit_code": parsed["exit_code"],
        "requested_cpus": parsed["requested_cpus"],
        "requested_memory_bytes": parsed["requested_memory_bytes"],
        "requested_gpu": parsed["requested_gpu"],
        "peak_rss_bytes": parsed["peak_rss_bytes"],
        "elapsed_seconds": parsed["elapsed_seconds"],
        "binding_comment": parsed["binding_comment"],
        "submit_line": parsed["submit_line"],
    }
    if any(value[name] != observed for name, observed in extracted.items()):
        raise PermissionError("scheduler evidence differs from raw sacct accounting")
    if parsed["cluster"] != TIGRIS_PARTITION:
        raise PermissionError("raw sacct cluster differs")
    if parsed["timelimit_minutes"] != math.ceil(requested_walltime / 60):
        raise PermissionError("raw sacct timelimit differs")
    return dict(value)


def build_scheduler_evidence(
    *, job_id: int, task_key: str, resource_class: str,
    source_commit: str, worker_role: str,
    worker: Mapping[str, Any], request: Mapping[str, Any],
    state: str, exit_code: str, peak_rss_bytes: int,
    elapsed_seconds: int | float,
    representation_recipe_sha256: str | None = None,
) -> dict[str, Any]:
    """Build an explicitly nonauthorizing scheduler fixture for local tests."""

    if resource_class not in PLANNING_RESOURCES:
        raise ValueError("scheduler evidence resource class differs")
    if worker_role not in WORKER_ROLES:
        raise ValueError("scheduler evidence worker role differs")
    worker_reference = dict(worker)
    _validate_file_reference(
        worker_reference, name=f"{worker_role} production worker",
    )
    artifact = with_content_hash({
        "contract": SCHEDULER_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "job_id": _positive_integer(job_id, name="Slurm job ID"),
        "task_key": str(task_key),
        "resource_class": resource_class,
        "site": TIGRIS_SITE,
        "account": TIGRIS_ACCOUNT,
        "partition": TIGRIS_PARTITION,
        "source_commit": _full_source_commit(source_commit),
        "representation_recipe_sha256": (
            None
            if representation_recipe_sha256 is None
            else require_sha256(
                representation_recipe_sha256,
                name="scheduler representation recipe",
            )
        ),
        "python_no_user_site": True,
        "worker_role": worker_role,
        "worker": worker_reference,
        "requested_cpus": int(request["cpus"]),
        "requested_memory_bytes": _memory_bytes(request["memory"]),
        "requested_walltime_seconds": _walltime_seconds(request["walltime"]),
        "requested_gpu": request["gpu"],
        "state": str(state),
        "exit_code": str(exit_code),
        "peak_rss_bytes": _positive_integer(
            peak_rss_bytes, name="peak RSS bytes",
        ),
        "elapsed_seconds": elapsed_seconds,
        "evidence_origin": "local_fixture/v1",
        "raw_accounting_record": None,
        "capture_command": None,
        "job_name": f"hcwdlr_{task_key}",
        "binding_comment": None,
        "submit_line": None,
        "capture_runtime": None,
        "authorization_capable": False,
    })
    validate_scheduler_evidence(
        artifact, resource_class=resource_class, request=request,
        expected_source_commit=source_commit,
        expected_recipe_sha256=representation_recipe_sha256,
        expected_workers={worker_role: worker_reference},
    )
    return artifact


def build_scheduler_evidence_from_sacct(
    *, raw_accounting_record: Mapping[str, Any], task_key: str,
    resource_class: str, source_commit: str,
    representation_recipe_sha256: str | None, worker_role: str,
    worker: Mapping[str, Any], request: Mapping[str, Any],
    expected_collector_job_name: str | None = None,
) -> dict[str, Any]:
    """Derive authorizing scheduler evidence only from immutable raw sacct bytes."""

    if resource_class not in PLANNING_RESOURCES:
        raise ValueError("scheduler evidence resource class differs")
    if worker_role not in WORKER_ROLES:
        raise ValueError("scheduler evidence worker role differs")
    worker_reference = dict(worker)
    worker_path, worker_sha256 = _validate_file_reference(
        worker_reference, name=f"{worker_role} production worker",
    )
    recipe = (
        None
        if representation_recipe_sha256 is None
        else require_sha256(
            representation_recipe_sha256, name="scheduler representation recipe",
        )
    )
    comment = scheduler_evidence_comment(
        task_key=task_key, resource_class=resource_class,
        source_commit=source_commit, representation_recipe_sha256=recipe,
        worker_role=worker_role, worker_sha256=worker_sha256, request=request,
    )
    raw_reference = dict(raw_accounting_record)
    capture_runtime = _live_tigris_capture_runtime()
    if (
        expected_collector_job_name is not None
        and capture_runtime["collector_job_name"] != expected_collector_job_name
    ):
        raise PermissionError("scheduler evidence collector job name differs")
    parsed = _parse_sacct_reference(
        raw_reference, expected_worker_path=str(worker_path),
        expected_comment=comment, expected_task_key=task_key,
        exact_script_argv=True,
    )
    artifact = with_content_hash({
        "contract": SCHEDULER_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "job_id": parsed["job_id"],
        "task_key": str(task_key),
        "resource_class": resource_class,
        "site": TIGRIS_SITE,
        "account": parsed["account"],
        "partition": parsed["partition"],
        "source_commit": _full_source_commit(source_commit),
        "representation_recipe_sha256": recipe,
        "python_no_user_site": True,
        "worker_role": worker_role,
        "worker": worker_reference,
        "requested_cpus": parsed["requested_cpus"],
        "requested_memory_bytes": parsed["requested_memory_bytes"],
        "requested_walltime_seconds": _walltime_seconds(request["walltime"]),
        "requested_gpu": parsed["requested_gpu"],
        "state": parsed["state"],
        "exit_code": parsed["exit_code"],
        "peak_rss_bytes": parsed["peak_rss_bytes"],
        "elapsed_seconds": parsed["elapsed_seconds"],
        "evidence_origin": "tigris_sacct_raw/v1",
        "raw_accounting_record": raw_reference,
        "capture_command": _sacct_capture_command(parsed["job_id"]),
        "job_name": parsed["job_name"],
        "binding_comment": parsed["binding_comment"],
        "submit_line": parsed["submit_line"],
        "capture_runtime": capture_runtime,
        "authorization_capable": True,
    })
    validate_scheduler_evidence(
        artifact, resource_class=resource_class, request=request,
        expected_source_commit=source_commit,
        expected_recipe_sha256=recipe,
        expected_workers={worker_role: worker_reference}, require_genuine=True,
    )
    return artifact


def _nonfinal_scheduler_artifact(
    *, authority_sha256: str, action_id: str, resource_class: str,
    source_commit: str, representation_recipe_sha256: str,
    worker_role: str, worker: Mapping[str, Any], request: Mapping[str, Any],
    parsed: Mapping[str, Any] | None, raw_accounting_record: Mapping[str, Any] | None,
    capture_runtime: Mapping[str, Any] | None, local_job_id: int | None = None,
    local_peak_rss_bytes: int | None = None, local_elapsed_seconds: int | float | None = None,
    collector_produced_raw_bytes: bool = False,
    collector_entrypoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    worker_reference = dict(worker)
    worker_path, worker_sha256 = _validate_file_reference(
        worker_reference, name=f"{worker_role} non-final acceptance worker",
    )
    authority = require_sha256(
        authority_sha256, name="non-final acceptance authority",
    )
    recipe = require_sha256(
        representation_recipe_sha256, name="non-final representation recipe",
    )
    task_key = f"acceptance-nonfinal-{action_id}"
    if parsed is None:
        if local_job_id is None or local_peak_rss_bytes is None or local_elapsed_seconds is None:
            raise ValueError("local non-final scheduler fixture is incomplete")
        observed = {
            "job_id": _positive_integer(local_job_id, name="Slurm job ID"),
            "job_name": f"hcwdlr_{task_key}",
            "account": TIGRIS_ACCOUNT,
            "partition": TIGRIS_PARTITION,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "requested_cpus": int(request["cpus"]),
            "requested_memory_bytes": _memory_bytes(request["memory"]),
            "requested_gpu": request["gpu"],
            "peak_rss_bytes": _positive_integer(
                local_peak_rss_bytes, name="peak RSS bytes",
            ),
            "elapsed_seconds": local_elapsed_seconds,
            "binding_comment": None,
            "submit_line": None,
        }
        origin = "local_fixture/v1"
        authorization_capable = False
        capture_command = None
        raw_accounting_sha256 = None
        collector_identity_sha256 = None
        submit_argv = None
        normalized_collector_entrypoint = None
    else:
        observed = dict(parsed)
        origin = "tigris_sacct_raw/v1"
        authorization_capable = True
        capture_command = _sacct_capture_command(int(parsed["job_id"]))
        if raw_accounting_record is None or capture_runtime is None:
            raise ValueError("genuine non-final scheduler capture is incomplete")
        raw_accounting_sha256 = require_sha256(
            raw_accounting_record.get("sha256"), name="raw sacct accounting bytes",
        )
        collector_identity_sha256 = canonical_sha256({
            "hash_domain": "hcwdl-representation-nonfinal-sacct-collector/v1",
            "schema_version": 1,
            "target_job_id": int(parsed["job_id"]),
            "capture_command": capture_command,
            "raw_accounting_record": dict(raw_accounting_record),
            "capture_runtime": dict(capture_runtime),
            "collector_entrypoint": dict(collector_entrypoint or {}),
        })
        submit_argv = list(parsed["submit_argv"])
        normalized_collector_entrypoint = dict(collector_entrypoint or {})
    artifact = with_content_hash({
        "contract": NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "authority_sha256": authority,
        "action_id": str(action_id),
        "job_id": observed["job_id"],
        "task_key": task_key,
        "resource_class": resource_class,
        "site": TIGRIS_SITE,
        "account": observed["account"],
        "partition": observed["partition"],
        "source_commit": _full_source_commit(source_commit),
        "representation_recipe_sha256": recipe,
        "python_no_user_site": True,
        "worker_role": worker_role,
        "worker": worker_reference,
        "requested_cpus": observed["requested_cpus"],
        "requested_memory_bytes": observed["requested_memory_bytes"],
        "requested_walltime_seconds": _walltime_seconds(request["walltime"]),
        "requested_gpu": observed["requested_gpu"],
        "state": observed["state"],
        "exit_code": observed["exit_code"],
        "peak_rss_bytes": observed["peak_rss_bytes"],
        "elapsed_seconds": observed["elapsed_seconds"],
        "evidence_origin": origin,
        "raw_accounting_record": (
            None if raw_accounting_record is None else dict(raw_accounting_record)
        ),
        "raw_accounting_sha256": raw_accounting_sha256,
        "capture_command": capture_command,
        "job_name": observed["job_name"],
        "binding_comment": observed["binding_comment"],
        "submit_line": observed["submit_line"],
        "submit_argv": submit_argv,
        "capture_runtime": None if capture_runtime is None else dict(capture_runtime),
        "collector_identity_sha256": collector_identity_sha256,
        "collector_produced_raw_bytes": collector_produced_raw_bytes,
        "collector_entrypoint": normalized_collector_entrypoint,
        "authorization_capable": authorization_capable,
        "final_role_accessed": False,
        "pilot_submission_authorized": False,
    })
    validate_nonfinal_acceptance_scheduler_evidence(
        artifact, expected_authority_sha256=authority,
        expected_action_id=action_id, request=request,
        expected_source_commit=source_commit,
        expected_recipe_sha256=recipe,
        expected_worker=worker_reference,
        expected_resource_class=resource_class,
        expected_worker_role=worker_role,
    )
    return artifact


def build_nonfinal_acceptance_scheduler_evidence(
    *, authority_sha256: str, action_id: str, job_id: int,
    resource_class: str, source_commit: str,
    representation_recipe_sha256: str, worker_role: str,
    worker: Mapping[str, Any], request: Mapping[str, Any],
    peak_rss_bytes: int = 1, elapsed_seconds: int | float = 1,
) -> dict[str, Any]:
    """Build an explicitly nonauthorizing local action-scheduler fixture."""

    return _nonfinal_scheduler_artifact(
        authority_sha256=authority_sha256, action_id=action_id,
        resource_class=resource_class, source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role, worker=worker, request=request, parsed=None,
        raw_accounting_record=None, capture_runtime=None, local_job_id=job_id,
        local_peak_rss_bytes=peak_rss_bytes,
        local_elapsed_seconds=elapsed_seconds,
    )


def build_nonfinal_acceptance_scheduler_evidence_from_sacct(
    *, authority_sha256: str, action_id: str,
    raw_accounting_record: Mapping[str, Any], resource_class: str,
    source_commit: str, representation_recipe_sha256: str,
    worker_role: str, worker: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject caller-supplied bytes at the authorizing non-final boundary.

    Genuine bounded-action evidence must use
    :func:`capture_nonfinal_acceptance_scheduler_evidence`, which executes the
    frozen ``sacct`` command itself inside a distinct live collector job.
    """

    raise PermissionError(
        "caller-supplied sacct bytes cannot authorize a non-final action; "
        "use the live collector"
    )


def _validate_nonfinal_submit_request(
    submit_argv: Sequence[str], *, request: Mapping[str, Any],
) -> None:
    options = tuple(submit_argv[1:-1])
    required = {
        f"--cpus-per-task={int(request['cpus'])}",
        f"--mem={request['memory']}",
        f"--time={request['walltime']}",
    }
    if request["gpu"] is not None:
        required.add(f"--gres={request['gpu']}")
    prefixes = ("--cpus-per-task=", "--mem=", "--time=", "--gres=")
    scoped = [token for token in options if token.startswith(prefixes)]
    if not required.issubset(set(options)) or len(scoped) != len(required):
        raise PermissionError("raw sacct submit argv resource request differs")


def capture_nonfinal_acceptance_scheduler_evidence(
    *, authority_sha256: str, action_id: str, job_id: int,
    raw_accounting_output: str | Path, resource_class: str,
    source_commit: str, representation_recipe_sha256: str,
    worker_role: str, worker: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture raw accounting bytes in a distinct live Tigris collector job."""

    if resource_class not in PLANNING_RESOURCES or worker_role not in WORKER_ROLES:
        raise ValueError("non-final scheduler resource or worker role differs")
    worker_reference = dict(worker)
    worker_path, worker_sha256 = _validate_file_reference(
        worker_reference, name=f"{worker_role} non-final acceptance worker",
    )
    expected_comment = nonfinal_acceptance_scheduler_comment(
        authority_sha256=authority_sha256, action_id=action_id,
        resource_class=resource_class, source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role, worker_sha256=worker_sha256, request=request,
    )
    target_job_id = _positive_integer(job_id, name="non-final target Slurm job ID")
    capture_runtime = _live_tigris_capture_runtime()
    if capture_runtime["collector_job_name"] != NONFINAL_COLLECTOR_JOB_NAME:
        raise PermissionError("non-final sacct capture uses another collector role")
    if int(capture_runtime["collector_job_id"]) == target_job_id:
        raise PermissionError("sacct collector job must differ from the target job")
    if os.environ.get("HCWDL_NONFINAL_EVIDENCE_COLLECTOR") != "1":
        raise PermissionError("non-final sacct capture lacks the collector worker marker")
    project_root = worker_path.resolve().parent.parent
    if worker_path.resolve().parent.name != "sbatch":
        raise PermissionError("non-final action worker lacks canonical project context")
    collector_worker = project_root / "sbatch" / NONFINAL_COLLECTOR_WORKER
    collector_cli = project_root / "scripts" / NONFINAL_COLLECTOR_CLI
    collector_entrypoint = {
        "worker": artifact_reference(collector_worker),
        "cli": artifact_reference(collector_cli),
        "environment_marker": "HCWDL_NONFINAL_EVIDENCE_COLLECTOR=1",
    }
    from .hcwdl_representation_campaign import validate_source_checkout

    validate_source_checkout(project_root, expected_commit=source_commit)
    capture_command = _sacct_capture_command(
        target_job_id, executable=str(capture_runtime["sacct_executable"]),
    )
    completed = subprocess.run(
        capture_command, check=True, capture_output=True,
    )
    raw_bytes = bytes(completed.stdout)
    if not raw_bytes:
        raise PermissionError("live sacct collector returned no accounting bytes")
    raw_path = Path(raw_accounting_output).resolve()
    if raw_path.exists():
        raise FileExistsError("live sacct output path already exists")
    if atomic_publish_bytes(raw_path, raw_bytes) != "published":
        raise FileExistsError("live sacct output was not freshly collector-produced")
    raw_accounting_record = artifact_reference(raw_path)
    parsed = _parse_sacct_reference(
        raw_accounting_record, expected_worker_path=str(worker_path),
        expected_comment=expected_comment,
        expected_task_key=f"acceptance-nonfinal-{action_id}",
        exact_script_argv=True,
    )
    if int(parsed["job_id"]) != target_job_id:
        raise PermissionError("live sacct capture returned a different target job")
    _validate_nonfinal_submit_request(parsed["submit_argv"], request=request)
    return _nonfinal_scheduler_artifact(
        authority_sha256=authority_sha256, action_id=action_id,
        resource_class=resource_class, source_commit=source_commit,
        representation_recipe_sha256=representation_recipe_sha256,
        worker_role=worker_role, worker=worker_reference, request=request,
        parsed=parsed, raw_accounting_record=raw_accounting_record,
        capture_runtime=capture_runtime, collector_produced_raw_bytes=True,
        collector_entrypoint=collector_entrypoint,
    )


def validate_nonfinal_acceptance_scheduler_evidence(
    value: Mapping[str, Any], *, expected_authority_sha256: str,
    expected_action_id: str, request: Mapping[str, Any],
    expected_source_commit: str, expected_recipe_sha256: str,
    expected_worker: Mapping[str, Any], expected_resource_class: str,
    expected_worker_role: str, require_genuine: bool = False,
) -> str:
    """Validate exact action, authority, worker and raw Slurm lineage."""

    digest = validate_content_hash(
        value, expected_contract=NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT,
        expected_schema_version=1,
    )
    expected_fields = {
        "contract", "schema_version", "authority_sha256", "action_id",
        "job_id", "task_key", "resource_class", "site", "account",
        "partition", "source_commit", "representation_recipe_sha256",
        "python_no_user_site", "worker_role", "worker", "requested_cpus",
        "requested_memory_bytes", "requested_walltime_seconds", "requested_gpu",
        "state", "exit_code", "peak_rss_bytes", "elapsed_seconds",
        "evidence_origin", "raw_accounting_record", "capture_command",
        "raw_accounting_sha256", "job_name", "binding_comment", "submit_line",
        "submit_argv", "capture_runtime", "collector_identity_sha256",
        "collector_produced_raw_bytes", "collector_entrypoint",
        "authorization_capable", "final_role_accessed",
        "pilot_submission_authorized", "content_hash",
    }
    if set(value) != expected_fields:
        raise PermissionError("non-final scheduler evidence fields differ")
    authority = require_sha256(
        expected_authority_sha256, name="non-final acceptance authority",
    )
    action_id = str(expected_action_id)
    if (
        value["authority_sha256"] != authority
        or value["action_id"] != action_id
        or value["task_key"] != f"acceptance-nonfinal-{action_id}"
        or value["source_commit"] != _full_source_commit(expected_source_commit)
        or value["representation_recipe_sha256"]
        != require_sha256(expected_recipe_sha256, name="expected recipe")
        or value["worker"] != dict(expected_worker)
        or value["resource_class"] != expected_resource_class
        or value["worker_role"] != expected_worker_role
        or value["final_role_accessed"] is not False
        or value["pilot_submission_authorized"] is not False
    ):
        raise PermissionError("non-final scheduler authority or lineage differs")
    _, worker_sha256 = _validate_file_reference(
        value["worker"], name="non-final acceptance worker",
    )
    if (
        value["site"] != TIGRIS_SITE
        or value["account"] != TIGRIS_ACCOUNT
        or value["partition"] != TIGRIS_PARTITION
        or value["python_no_user_site"] is not True
        or value["state"] != "COMPLETED"
        or value["exit_code"] != "0:0"
        or value["requested_cpus"] != int(request["cpus"])
        or value["requested_memory_bytes"] != _memory_bytes(request["memory"])
        or value["requested_walltime_seconds"] != _walltime_seconds(request["walltime"])
        or value["requested_gpu"] != request["gpu"]
    ):
        raise PermissionError("non-final scheduler environment or request differs")
    rss = _positive_integer(value["peak_rss_bytes"], name="peak RSS bytes")
    elapsed = float(value["elapsed_seconds"])
    if not 0 < elapsed <= value["requested_walltime_seconds"] or rss > (
        3 * value["requested_memory_bytes"]
    ) // 4:
        raise PermissionError("non-final scheduler resource measurement differs")
    origin = value["evidence_origin"]
    if origin == "local_fixture/v1":
        if any(value[name] is not None for name in (
            "raw_accounting_record", "capture_command", "binding_comment",
            "raw_accounting_sha256", "submit_line", "submit_argv",
            "capture_runtime", "collector_identity_sha256",
            "collector_entrypoint",
        )) or value["collector_produced_raw_bytes"] is not False or value[
            "authorization_capable"
        ] is not False:
            raise PermissionError("local non-final scheduler fixture claims authority")
        if require_genuine:
            raise PermissionError("non-final scheduler evidence is a local fixture")
        return digest
    if origin != "tigris_sacct_raw/v1" or value["authorization_capable"] is not True:
        raise PermissionError("non-final scheduler evidence origin differs")
    capture_runtime = _validate_capture_runtime(value["capture_runtime"])
    if (
        value["collector_produced_raw_bytes"] is not True
        or int(capture_runtime["collector_job_id"]) == int(value["job_id"])
    ):
        raise PermissionError("non-final scheduler evidence lacks a distinct live collector")
    entrypoint = value["collector_entrypoint"]
    if not isinstance(entrypoint, Mapping) or set(entrypoint) != {
        "worker", "cli", "environment_marker",
    }:
        raise PermissionError("non-final scheduler collector entrypoint differs")
    raw_path, raw_sha256 = _validate_file_reference(
        value["raw_accounting_record"], name="raw sacct accounting record",
    )
    if (
        value["raw_accounting_sha256"] != raw_sha256
        or value["raw_accounting_record"].get("path") != str(raw_path)
    ):
        raise PermissionError("non-final scheduler raw accounting bytes differ")
    expected_comment = nonfinal_acceptance_scheduler_comment(
        authority_sha256=authority, action_id=action_id,
        resource_class=str(value["resource_class"]),
        source_commit=expected_source_commit,
        representation_recipe_sha256=expected_recipe_sha256,
        worker_role=str(value["worker_role"]), worker_sha256=worker_sha256,
        request=request,
    )
    worker_path, _ = _validate_file_reference(
        value["worker"], name="non-final acceptance worker",
    )
    parsed = _parse_sacct_reference(
        value["raw_accounting_record"], expected_worker_path=str(worker_path),
        expected_comment=expected_comment,
        expected_task_key=f"acceptance-nonfinal-{action_id}",
        exact_script_argv=True,
    )
    _validate_nonfinal_submit_request(parsed["submit_argv"], request=request)
    expected_raw = {
        "job_id": parsed["job_id"], "job_name": parsed["job_name"],
        "account": parsed["account"], "partition": parsed["partition"],
        "state": parsed["state"], "exit_code": parsed["exit_code"],
        "requested_cpus": parsed["requested_cpus"],
        "requested_memory_bytes": parsed["requested_memory_bytes"],
        "requested_gpu": parsed["requested_gpu"],
        "peak_rss_bytes": parsed["peak_rss_bytes"],
        "elapsed_seconds": parsed["elapsed_seconds"],
        "binding_comment": parsed["binding_comment"],
        "submit_line": parsed["submit_line"],
        "submit_argv": parsed["submit_argv"],
    }
    if any(value[name] != expected for name, expected in expected_raw.items()):
        raise PermissionError("non-final scheduler evidence differs from raw sacct")
    if value["capture_command"] != _sacct_capture_command(parsed["job_id"]):
        raise PermissionError("non-final scheduler capture command differs")
    expected_collector_identity = canonical_sha256({
        "hash_domain": "hcwdl-representation-nonfinal-sacct-collector/v1",
        "schema_version": 1,
        "target_job_id": int(parsed["job_id"]),
        "capture_command": value["capture_command"],
        "raw_accounting_record": dict(value["raw_accounting_record"]),
        "capture_runtime": dict(value["capture_runtime"]),
        "collector_entrypoint": dict(entrypoint),
    })
    if value["collector_identity_sha256"] != expected_collector_identity:
        raise PermissionError("non-final scheduler collector identity differs")
    if parsed["cluster"] != TIGRIS_PARTITION or parsed["timelimit_minutes"] != math.ceil(
        value["requested_walltime_seconds"] / 60
    ):
        raise PermissionError("non-final scheduler cluster or timelimit differs")
    worker_root = Path(worker_path).resolve().parent.parent
    if Path(worker_path).resolve().parent.name != "sbatch":
        raise PermissionError("non-final scheduler worker is outside canonical project context")
    from .hcwdl_representation_campaign import validate_source_checkout

    collector_worker, _ = _validate_file_reference(
        entrypoint["worker"], name="non-final evidence collector worker",
    )
    collector_cli, _ = _validate_file_reference(
        entrypoint["cli"], name="non-final evidence collector CLI",
    )
    if (
        collector_worker.resolve()
        != worker_root / "sbatch" / NONFINAL_COLLECTOR_WORKER
        or collector_cli.resolve() != worker_root / "scripts" / NONFINAL_COLLECTOR_CLI
        or entrypoint["environment_marker"]
        != "HCWDL_NONFINAL_EVIDENCE_COLLECTOR=1"
    ):
        raise PermissionError("non-final scheduler collector source identity differs")

    validate_source_checkout(worker_root, expected_commit=expected_source_commit)
    return digest


def build_miniature_evidence(
    *, evidence_kind: str, scheduler_evidence: Mapping[str, Any],
    representation_recipe_sha256: str | None, rows: int,
    result_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one immutable result output to its exact scheduler lineage."""

    if not evidence_kind:
        raise ValueError("miniature evidence kind is empty")
    _, worker_sha256 = _validate_file_reference(
        scheduler_evidence["worker"], name="miniature production worker",
    )
    result_reference: dict[str, Any] | None = None
    result_contract: str | None = None
    result_sha256: str | None = None
    if result_artifact is not None:
        result_reference = dict(result_artifact)
        result_path, _ = _validate_file_reference(
            result_reference, name="miniature result artifact",
        )
        result = load_json(result_path)
        result_contract = str(result.get("contract"))
        if not result_contract or result_contract == "None":
            raise ValueError("miniature result contract differs")
        result_sha256 = validate_content_hash(
            result, expected_contract=result_contract, expected_schema_version=1,
        )
    scheduler_authorizing = scheduler_evidence.get("authorization_capable") is True
    authorization_capable = scheduler_authorizing and result_reference is not None
    result_execution_sha256 = (
        None
        if result_reference is None
        else canonical_sha256({
            "hash_domain": "hcwdl-representation-action-result-execution/v1",
            "schema_version": 1,
            "evidence_kind": evidence_kind,
            "job_id": scheduler_evidence["job_id"],
            "task_key": scheduler_evidence["task_key"],
            "resource_class": scheduler_evidence["resource_class"],
            "source_commit": scheduler_evidence["source_commit"],
            "representation_recipe_sha256": representation_recipe_sha256,
            "worker_role": scheduler_evidence["worker_role"],
            "worker_sha256": worker_sha256,
            "scheduler_evidence_sha256": scheduler_evidence["content_hash"],
            "result_artifact": result_reference,
            "result_contract": result_contract,
            "result_sha256": result_sha256,
        })
    )
    artifact = with_content_hash({
        "contract": MINIATURE_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "evidence_kind": evidence_kind,
        "job_id": _positive_integer(
            scheduler_evidence["job_id"], name="miniature Slurm job ID",
        ),
        "task_key": str(scheduler_evidence["task_key"]),
        "source_commit": _full_source_commit(scheduler_evidence["source_commit"]),
        "representation_recipe_sha256": (
            None
            if representation_recipe_sha256 is None
            else require_sha256(
                representation_recipe_sha256, name="miniature representation recipe",
            )
        ),
        "worker_sha256": worker_sha256,
        "worker_role": scheduler_evidence["worker_role"],
        "resource_class": scheduler_evidence["resource_class"],
        "scheduler_evidence_sha256": require_sha256(
            scheduler_evidence.get("content_hash"), name="scheduler evidence",
        ),
        "scheduler_evidence_origin": scheduler_evidence.get("evidence_origin"),
        "result_artifact": result_reference,
        "result_contract": result_contract,
        "result_sha256": result_sha256,
        "result_execution_sha256": result_execution_sha256,
        "rows": _positive_integer(rows, name="miniature measured rows"),
        "semantic_result_authenticated": result_reference is not None,
        "authorization_capable": authorization_capable,
        "final_role_access": (
            "validation_only_proxy"
            if evidence_kind == "validation_only_proxy"
            else "none"
        ),
    })
    validate_miniature_evidence(
        artifact, expected_kind=evidence_kind,
        expected_source_commit=scheduler_evidence["source_commit"],
        expected_recipe_sha256=representation_recipe_sha256,
        scheduler_evidence=scheduler_evidence,
    )
    return artifact


def validate_miniature_evidence(
    value: Mapping[str, Any], *, expected_kind: str,
    expected_source_commit: str, expected_recipe_sha256: str | None,
    scheduler_evidence: Mapping[str, Any],
    require_genuine: bool = False,
) -> str:
    """Validate one measured miniature/stage report against its Slurm record."""

    digest = validate_content_hash(
        value,
        expected_contract=MINIATURE_EVIDENCE_CONTRACT,
        expected_schema_version=1,
    )
    if set(value) != {
        "contract", "schema_version", "evidence_kind", "job_id", "task_key",
        "source_commit", "representation_recipe_sha256", "worker_sha256",
        "worker_role", "resource_class", "scheduler_evidence_sha256",
        "scheduler_evidence_origin", "result_artifact", "result_contract",
        "result_sha256", "result_execution_sha256", "rows",
        "semantic_result_authenticated", "authorization_capable",
        "final_role_access", "content_hash",
    }:
        raise PermissionError("miniature evidence fields differ")
    if value["evidence_kind"] != expected_kind:
        raise PermissionError("miniature evidence kind differs")
    if _positive_integer(value["job_id"], name="miniature Slurm job ID") != int(
        scheduler_evidence["job_id"]
    ):
        raise PermissionError("miniature evidence job ID differs")
    if value["task_key"] != scheduler_evidence["task_key"]:
        raise PermissionError("miniature evidence task key differs")
    source_commit = _full_source_commit(value["source_commit"])
    if (
        source_commit != _full_source_commit(expected_source_commit)
        or source_commit != scheduler_evidence["source_commit"]
    ):
        raise PermissionError("miniature evidence source commit differs")
    recipe = value["representation_recipe_sha256"]
    if expected_recipe_sha256 is None:
        if recipe is not None:
            require_sha256(recipe, name="miniature representation recipe")
    elif require_sha256(recipe, name="miniature representation recipe") != require_sha256(
        expected_recipe_sha256, name="expected representation recipe",
    ):
        raise PermissionError("miniature evidence representation recipe differs")
    if recipe != scheduler_evidence.get("representation_recipe_sha256"):
        raise PermissionError("miniature evidence scheduler recipe lineage differs")
    _, worker_sha256 = _validate_file_reference(
        scheduler_evidence["worker"], name="miniature production worker",
    )
    if value["worker_sha256"] != worker_sha256:
        raise PermissionError("miniature evidence worker hash differs")
    if (
        value["worker_role"] != scheduler_evidence["worker_role"]
        or value["resource_class"] != scheduler_evidence["resource_class"]
        or value["scheduler_evidence_sha256"] != scheduler_evidence["content_hash"]
        or value["scheduler_evidence_origin"] != scheduler_evidence["evidence_origin"]
    ):
        raise PermissionError("miniature evidence scheduler lineage differs")
    _positive_integer(value["rows"], name="miniature measured rows")
    result_reference = value["result_artifact"]
    if result_reference is None:
        if any(value[name] is not None for name in (
            "result_contract", "result_sha256", "result_execution_sha256",
        )) or value["semantic_result_authenticated"] is not False:
            raise PermissionError("miniature evidence has partial result lineage")
    else:
        result_path, _ = _validate_file_reference(
            result_reference, name="miniature result artifact",
        )
        result = load_json(result_path)
        result_contract = str(value["result_contract"])
        result_sha256 = validate_content_hash(
            result, expected_contract=result_contract, expected_schema_version=1,
        )
        if result_sha256 != value["result_sha256"]:
            raise PermissionError("miniature result content identity differs")
        expected_execution = canonical_sha256({
            "hash_domain": "hcwdl-representation-action-result-execution/v1",
            "schema_version": 1,
            "evidence_kind": expected_kind,
            "job_id": scheduler_evidence["job_id"],
            "task_key": scheduler_evidence["task_key"],
            "resource_class": scheduler_evidence["resource_class"],
            "source_commit": scheduler_evidence["source_commit"],
            "representation_recipe_sha256": recipe,
            "worker_role": scheduler_evidence["worker_role"],
            "worker_sha256": worker_sha256,
            "scheduler_evidence_sha256": scheduler_evidence["content_hash"],
            "result_artifact": result_reference,
            "result_contract": result_contract,
            "result_sha256": result_sha256,
        })
        if (
            value["result_execution_sha256"] != expected_execution
            or value["semantic_result_authenticated"] is not True
        ):
            raise PermissionError("miniature result execution lineage differs")
    expected_access = (
        "validation_only_proxy"
        if expected_kind == "validation_only_proxy"
        else "none"
    )
    if value["final_role_access"] != expected_access:
        raise PermissionError("miniature evidence final-role access differs")
    expected_authorizing = (
        scheduler_evidence.get("authorization_capable") is True
        and result_reference is not None
    )
    if value["authorization_capable"] is not expected_authorizing:
        raise PermissionError("miniature evidence authorization class differs")
    if require_genuine and value["authorization_capable"] is not True:
        raise PermissionError(
            "miniature evidence is a nonauthorizing fixture without raw/result lineage"
        )
    return digest


def validate_measured_profile(
    profile: Mapping[str, Any], *, require_genuine_tigris: bool = False,
    expected_source_commit: str | None = None,
) -> str:
    digest = validate_content_hash(profile, expected_contract=RESOURCE_PROFILE_CONTRACT)
    requests = profile.get("requests")
    measurements = profile.get("measurements")
    concurrency = profile.get("array_concurrency_limits", {})
    if not isinstance(requests, Mapping) or set(requests) != set(PLANNING_RESOURCES):
        raise ValueError("representation measured resource classes differ")
    if not isinstance(measurements, Mapping) or set(measurements) != set(requests):
        raise ValueError("representation resource measurements differ")
    genuine_row_shape = all(
        isinstance(row, Mapping)
        and set(row) == {"scheduler_evidence", "miniature_evidence"}
        for row in measurements.values()
    )
    if not isinstance(concurrency, Mapping):
        raise ValueError("representation array-concurrency registry differs")
    for task_key, raw_limit in concurrency.items():
        if (
            not isinstance(task_key, str) or not task_key
            or isinstance(raw_limit, bool) or not isinstance(raw_limit, int)
            or raw_limit <= 0
        ):
            raise ValueError("representation measured array-concurrency limit differs")
    for name, request in requests.items():
        if set(request) != {"cpus", "memory", "walltime", "gpu"}:
            raise ValueError(f"resource request {name!r} differs")
        row = measurements[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"resource measurement {name!r} differs")
        _memory_bytes(request["memory"])
        _walltime_seconds(request["walltime"])
        if int(request["cpus"]) <= 0:
            raise ValueError(f"resource request {name!r} CPU count differs")
        if require_genuine_tigris or genuine_row_shape:
            if set(row) != {"scheduler_evidence", "miniature_evidence"}:
                raise PermissionError(
                    f"resource measurement {name!r} lacks genuine Tigris evidence"
                )
        elif set(row) != {"peak_rss_bytes", "elapsed_seconds"} or not all(
            not isinstance(row[key], bool)
            and isinstance(row[key], (int, float))
            and float(row[key]) > 0
            for key in ("peak_rss_bytes", "elapsed_seconds")
        ):
            raise ValueError(f"resource measurement {name!r} differs")
    if require_genuine_tigris or genuine_row_shape:
        if expected_source_commit is None:
            raise PermissionError(
                "genuine resource validation requires the expected source commit"
            )
        expected_source_commit = _full_source_commit(expected_source_commit)
        environment = profile.get("measurement_environment")
        if not isinstance(environment, Mapping) or set(environment) != {
            "site", "account", "partition", "source_commit",
            "python_no_user_site", "production_workers",
        }:
            raise PermissionError("resource profile lacks its genuine Tigris environment")
        if (
            environment["site"] != TIGRIS_SITE
            or environment["account"] != TIGRIS_ACCOUNT
            or environment["partition"] != TIGRIS_PARTITION
            or environment["python_no_user_site"] is not True
        ):
            raise PermissionError("resource profile Tigris environment differs")
        source_commit = _full_source_commit(environment["source_commit"])
        if source_commit != expected_source_commit:
            raise PermissionError("resource profile source commit differs")
        workers = environment["production_workers"]
        if not isinstance(workers, Mapping) or set(workers) != set(WORKER_ROLES):
            raise PermissionError("resource profile worker registry differs")
        worker_hashes: dict[str, str] = {}
        for worker_role, reference in workers.items():
            _, worker_hashes[worker_role] = _validate_file_reference(
                reference, name=f"{worker_role} production worker",
            )
        for name, request in requests.items():
            row = measurements[name]
            scheduler, _ = load_authenticated_json_reference(
                row["scheduler_evidence"],
                expected_contract=SCHEDULER_EVIDENCE_CONTRACT,
                name=f"{name} scheduler evidence",
            )
            scheduler_data = validate_scheduler_evidence(
                scheduler,
                resource_class=name,
                request=request,
                expected_source_commit=source_commit,
                expected_workers=workers,
                require_genuine=True,
            )
            miniature, _ = load_authenticated_json_reference(
                row["miniature_evidence"],
                expected_contract=MINIATURE_EVIDENCE_CONTRACT,
                name=f"{name} miniature evidence",
            )
            validate_miniature_evidence(
                miniature,
                expected_kind=f"resource_profile:{name}",
                expected_source_commit=source_commit,
                expected_recipe_sha256=None,
                scheduler_evidence=scheduler_data,
                require_genuine=True,
            )
            if miniature["worker_sha256"] != worker_hashes[scheduler["worker_role"]]:
                raise PermissionError(
                    f"resource measurement {name!r} worker lineage differs"
                )
    return digest


def build_measured_profile(
    *, mode: str, source_commit: str,
    production_workers: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
    array_concurrency_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the genuine resource profile from exact scheduler/result files."""

    requests = resource_table(mode=mode)
    if set(production_workers) != set(WORKER_ROLES):
        raise ValueError("resource-profile worker registry differs")
    workers = {
        role: dict(reference) for role, reference in production_workers.items()
    }
    for role, reference in workers.items():
        _validate_file_reference(reference, name=f"{role} production worker")
    if set(measurements) != set(requests):
        raise ValueError("resource-profile measurement registry differs")
    normalized_measurements: dict[str, dict[str, Any]] = {}
    for resource_class, row in measurements.items():
        if not isinstance(row, Mapping) or set(row) != {
            "scheduler_evidence", "miniature_evidence",
        }:
            raise ValueError("resource-profile measurement row differs")
        scheduler, _ = load_authenticated_json_reference(
            row["scheduler_evidence"],
            expected_contract=SCHEDULER_EVIDENCE_CONTRACT,
            name=f"{resource_class} scheduler evidence",
        )
        scheduler = validate_scheduler_evidence(
            scheduler, resource_class=resource_class,
            request=requests[resource_class], expected_source_commit=source_commit,
            expected_workers=workers, require_genuine=True,
        )
        miniature, _ = load_authenticated_json_reference(
            row["miniature_evidence"],
            expected_contract=MINIATURE_EVIDENCE_CONTRACT,
            name=f"{resource_class} miniature evidence",
        )
        validate_miniature_evidence(
            miniature, expected_kind=f"resource_profile:{resource_class}",
            expected_source_commit=source_commit,
            expected_recipe_sha256=None, scheduler_evidence=scheduler,
            require_genuine=True,
        )
        normalized_measurements[resource_class] = {
            "scheduler_evidence": dict(row["scheduler_evidence"]),
            "miniature_evidence": dict(row["miniature_evidence"]),
        }
    artifact = with_content_hash({
        "contract": RESOURCE_PROFILE_CONTRACT,
        "schema_version": 1,
        "requests": requests,
        "measurements": normalized_measurements,
        "array_concurrency_limits": dict(array_concurrency_limits or {}),
        "measurement_environment": {
            "site": TIGRIS_SITE,
            "account": TIGRIS_ACCOUNT,
            "partition": TIGRIS_PARTITION,
            "source_commit": _full_source_commit(source_commit),
            "python_no_user_site": True,
            "production_workers": workers,
        },
    })
    validate_measured_profile(
        artifact, require_genuine_tigris=True,
        expected_source_commit=source_commit,
    )
    return artifact


def validate_dense_measured_profile(
    profile: Mapping[str, Any], *, expected_source_commit: str,
) -> str:
    """Validate the exact four resource classes used by the non-final DAG."""

    digest = validate_content_hash(
        profile, expected_contract=DENSE_RESOURCE_PROFILE_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "disposition", "requests",
        "measurements", "array_concurrency_limits", "measurement_environment",
        "content_hash",
    }
    if set(profile) != required or profile.get("disposition") != "dense_training_only":
        raise PermissionError("dense resource profile schema/disposition differs")
    requests = profile.get("requests")
    measurements = profile.get("measurements")
    expected_requests = resource_table(mode="smoke")
    expected_requests = {
        name: expected_requests[name] for name in DENSE_RESOURCE_CLASSES
    }
    if requests != expected_requests or not isinstance(measurements, Mapping) or set(
        measurements
    ) != set(DENSE_RESOURCE_CLASSES):
        raise ValueError("dense resource profile class registry differs")
    concurrency = profile.get("array_concurrency_limits")
    if not isinstance(concurrency, Mapping) or any(
        not isinstance(key, str) or not key
        or isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for key, value in concurrency.items()
    ):
        raise ValueError("dense resource concurrency registry differs")
    environment = profile.get("measurement_environment")
    if not isinstance(environment, Mapping) or set(environment) != {
        "site", "account", "partition", "source_commit",
        "python_no_user_site", "production_workers",
    }:
        raise PermissionError("dense resource profile environment differs")
    source_commit = _full_source_commit(expected_source_commit)
    if (
        environment.get("site") != TIGRIS_SITE
        or environment.get("account") != TIGRIS_ACCOUNT
        or environment.get("partition") != TIGRIS_PARTITION
        or environment.get("source_commit") != source_commit
        or environment.get("python_no_user_site") is not True
    ):
        raise PermissionError("dense resource profile Tigris lineage differs")
    workers = environment.get("production_workers")
    if not isinstance(workers, Mapping) or set(workers) != set(WORKER_ROLES):
        raise PermissionError("dense resource worker registry differs")
    worker_hashes = {}
    for role, reference in workers.items():
        _, worker_hashes[role] = _validate_file_reference(
            reference, name=f"dense {role} resource worker",
        )
    for resource_class in DENSE_RESOURCE_CLASSES:
        row = measurements[resource_class]
        if not isinstance(row, Mapping) or set(row) != {
            "scheduler_evidence", "miniature_evidence",
        }:
            raise ValueError("dense resource measurement row differs")
        scheduler, _ = load_authenticated_json_reference(
            row["scheduler_evidence"],
            expected_contract=SCHEDULER_EVIDENCE_CONTRACT,
            name=f"dense {resource_class} scheduler evidence",
        )
        scheduler = validate_scheduler_evidence(
            scheduler, resource_class=resource_class,
            request=expected_requests[resource_class],
            expected_source_commit=source_commit,
            expected_workers=workers, require_genuine=True,
        )
        miniature, _ = load_authenticated_json_reference(
            row["miniature_evidence"],
            expected_contract=MINIATURE_EVIDENCE_CONTRACT,
            name=f"dense {resource_class} miniature evidence",
        )
        validate_miniature_evidence(
            miniature, expected_kind=f"resource_profile:{resource_class}",
            expected_source_commit=source_commit,
            expected_recipe_sha256=None, scheduler_evidence=scheduler,
            require_genuine=True,
        )
        result_path, _ = _validate_file_reference(
            miniature["result_artifact"],
            name=f"dense {resource_class} measured result",
        )
        result = load_json(result_path)
        if resource_class == "gpu_representation":
            if miniature.get("result_contract") != DENSE_STORAGE_TEMPLATE_CONTRACT:
                raise PermissionError(
                    "dense representation probe lacks its storage template"
                )
            validate_dense_storage_template(
                result, expected_source_commit=source_commit,
            )
        elif miniature.get("result_contract") != WORKER_RUNTIME_MEASUREMENT_CONTRACT:
            raise PermissionError("dense resource probe result contract differs")
        if miniature["worker_sha256"] != worker_hashes[scheduler["worker_role"]]:
            raise PermissionError("dense resource worker/result lineage differs")
    return digest


def build_dense_measured_profile(
    *, source_commit: str,
    production_workers: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
    array_concurrency_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    requests = resource_table(mode="smoke")
    requests = {name: requests[name] for name in DENSE_RESOURCE_CLASSES}
    if set(production_workers) != set(WORKER_ROLES):
        raise ValueError("dense resource-profile worker registry differs")
    workers = {role: dict(reference) for role, reference in production_workers.items()}
    for role, reference in workers.items():
        _validate_file_reference(reference, name=f"dense {role} resource worker")
    if set(measurements) != set(DENSE_RESOURCE_CLASSES):
        raise ValueError("dense resource-profile measurement registry differs")
    normalized = {
        name: {
            key: dict(row[key]) for key in (
                "scheduler_evidence", "miniature_evidence",
            )
        }
        for name, row in measurements.items()
        if isinstance(row, Mapping) and set(row) == {
            "scheduler_evidence", "miniature_evidence",
        }
    }
    if set(normalized) != set(DENSE_RESOURCE_CLASSES):
        raise ValueError("dense resource-profile measurement row differs")
    result = with_content_hash({
        "contract": DENSE_RESOURCE_PROFILE_CONTRACT,
        "schema_version": 1,
        "disposition": "dense_training_only",
        "requests": requests,
        "measurements": normalized,
        "array_concurrency_limits": dict(array_concurrency_limits or {}),
        "measurement_environment": {
            "site": TIGRIS_SITE, "account": TIGRIS_ACCOUNT,
            "partition": TIGRIS_PARTITION,
            "source_commit": _full_source_commit(source_commit),
            "python_no_user_site": True,
            "production_workers": workers,
        },
    })
    validate_dense_measured_profile(result, expected_source_commit=source_commit)
    return result


__all__ = [
    "DENSE_RESOURCE_CLASSES", "DENSE_RESOURCE_PROFILE_CONTRACT",
    "DENSE_STORAGE_ESTIMATE_CONTRACT", "DENSE_STORAGE_TEMPLATE_CONTRACT",
    "FIXED_SIZE_INVENTORY_CONTRACT", "FIXED_SIZE_KINDS",
    "MINIATURE_EVIDENCE_CONTRACT", "PLANNING_RESOURCES",
    "NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE_CONTRACT",
    "NONFINAL_COLLECTOR_CLI", "NONFINAL_COLLECTOR_WORKER",
    "RESOURCE_PROFILE_CONTRACT", "ResourceRequest", "SCHEDULER_EVIDENCE_CONTRACT",
    "SACCT_FIELDS", "SACCT_FORMAT", "SCHEDULER_EVIDENCE_ORIGINS",
    "SMOKE_RESOURCES", "STORAGE_ESTIMATE_CONTRACT", "TIGRIS_ACCOUNT",
    "TIGRIS_PARTITION", "TIGRIS_SITE", "WORKER_ROLES", "artifact_reference",
    "build_dense_measured_profile", "build_dense_storage_estimate",
    "build_dense_storage_template", "build_fixed_size_inventory", "build_measured_profile",
    "build_miniature_evidence", "build_scheduler_evidence",
    "build_nonfinal_acceptance_scheduler_evidence",
    "build_nonfinal_acceptance_scheduler_evidence_from_sacct",
    "capture_nonfinal_acceptance_scheduler_evidence",
    "build_scheduler_evidence_from_sacct",
    "build_storage_estimate", "measure_dense_storage_template",
    "load_authenticated_json_reference", "resource_table",
    "scheduler_evidence_comment",
    "nonfinal_acceptance_scheduler_comment",
    "validate_dense_measured_profile", "validate_dense_storage_availability",
    "validate_dense_storage_estimate", "validate_dense_storage_template",
    "validate_fixed_size_inventory", "validate_measured_profile",
    "validate_miniature_evidence", "validate_scheduler_evidence",
    "validate_nonfinal_acceptance_scheduler_evidence",
    "validate_storage_estimate",
]
