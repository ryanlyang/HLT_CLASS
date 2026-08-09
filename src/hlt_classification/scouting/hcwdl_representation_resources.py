"""Frozen local planning resource classes and measured-profile validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import io
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import socket
import sys
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .hcwdl_representation_contracts import (
    FIXED_SIZE_INVENTORY_CONTRACT,
    MINIATURE_EVIDENCE_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    SCHEDULER_EVIDENCE_CONTRACT,
    STORAGE_ESTIMATE_CONTRACT,
)

TIGRIS_SITE: Final = "tigris.rc.rit.edu"
TIGRIS_ACCOUNT: Final = "reu-aisocial"
TIGRIS_PARTITION: Final = "tigris"
WORKER_ROLES: Final = ("ordinary", "deterministic")
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
    "ReqGRES",
    "MaxRSS",
    "Comment",
    "SubmitLine",
)
SACCT_FORMAT: Final = ",".join(SACCT_FIELDS)
SACCT_COMMENT_PREFIX: Final = "hcwdl-rkd-evidence-v1:"
FIXED_SIZE_KINDS: Final = (
    "retained_resume",
    "selected_checkpoint",
    "final_assignment",
    "fixed_artifact",
)


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


def _sacct_capture_command(job_id: int) -> list[str]:
    return [
        "sacct",
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
        or not str(value["capture_host"])
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
    library_prefix = str(conda_prefix / "lib")
    runtime = {
        "site": TIGRIS_SITE,
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "collector_job_id": int(raw_job),
        "capture_host": socket.getfqdn(),
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
    expected_job_name = f"hcwdl_rkd_{expected_task_key}"
    if parent["JobName"] != expected_job_name:
        raise PermissionError("raw sacct job name differs")
    if parent["Comment"] != expected_comment:
        raise PermissionError("raw sacct binding comment differs")
    try:
        submit_tokens = shlex.split(parent["SubmitLine"], posix=True)
    except ValueError as error:
        raise ValueError("raw sacct submit line cannot be parsed") from error
    normalized_worker = str(Path(expected_worker_path))
    if normalized_worker not in submit_tokens:
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
        "requested_gpu": None if parent["ReqGRES"] in ("", "(null)") else parent["ReqGRES"],
        "peak_rss_bytes": max(rss_values),
        "binding_comment": parent["Comment"],
        "submit_line": parent["SubmitLine"],
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
        if value["job_name"] != f"hcwdl_rkd_{value['task_key']}":
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
        "job_name": f"hcwdl_rkd_{task_key}",
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
    parsed = _parse_sacct_reference(
        raw_reference, expected_worker_path=str(worker_path),
        expected_comment=comment, expected_task_key=task_key,
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
            "validation_proxy_only"
            if evidence_kind == "final_role_validation_proxy"
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
        "validation_proxy_only"
        if expected_kind == "final_role_validation_proxy"
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


__all__ = [
    "FIXED_SIZE_INVENTORY_CONTRACT", "FIXED_SIZE_KINDS",
    "MINIATURE_EVIDENCE_CONTRACT", "PLANNING_RESOURCES",
    "RESOURCE_PROFILE_CONTRACT", "ResourceRequest", "SCHEDULER_EVIDENCE_CONTRACT",
    "SACCT_FIELDS", "SACCT_FORMAT", "SCHEDULER_EVIDENCE_ORIGINS",
    "SMOKE_RESOURCES", "STORAGE_ESTIMATE_CONTRACT", "TIGRIS_ACCOUNT",
    "TIGRIS_PARTITION", "TIGRIS_SITE", "WORKER_ROLES", "artifact_reference",
    "build_fixed_size_inventory", "build_measured_profile",
    "build_miniature_evidence", "build_scheduler_evidence",
    "build_scheduler_evidence_from_sacct",
    "build_storage_estimate",
    "load_authenticated_json_reference", "resource_table",
    "scheduler_evidence_comment",
    "validate_fixed_size_inventory", "validate_measured_profile",
    "validate_miniature_evidence", "validate_scheduler_evidence",
    "validate_storage_estimate",
]
