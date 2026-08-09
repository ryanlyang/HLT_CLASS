"""Closed production dispatcher for immutable HCWDL-RKD campaign rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import signal
import threading
from typing import Any, Final

from hlt_classification.data.cache_contracts import canonical_sha256, sha256_file

from .hcwdl_representation_runtime_binding import (
    CAMPAIGN_ARTIFACT_BINDING, IMMUTABLE_OUTPUT_ROOT_BINDING,
    PREPUBLISHED_OUTPUT_BINDING, UPSTREAM_OUTPUT_BINDING,
    load_runtime_binding,
    resolve_runtime_row,
    runtime_campaign_identity,
    validate_prepublished_output_binding,
)
from .hcwdl_representation_smoke import FINAL_ROLE_KINDS
from .hcwdl_representation_workflow import RepresentationWorkflow


Handler = Callable[[Mapping[str, Any], Any, int | None, Mapping[str, Any]], Any]


class RepresentationPreemptionMonitor:
    """Convert Slurm USR1/TERM into an exact-resume request.

    The worker uses ``exec python`` so ``B:USR1`` reaches this process.  Only
    the main thread installs handlers, and production training adapters receive
    the read-only callback through their ephemeral runtime context.
    """

    def __init__(self) -> None:
        self.requested = False
        self.previous: dict[int, object] = {}

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("HCWDL-RKD signal handlers require the main thread")
        for name in ("SIGUSR1", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            self.previous[number] = signal.getsignal(number)
            signal.signal(number, self._request)

    def _request(self, _signum, _frame) -> None:
        self.requested = True

    def restore(self) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        self.previous.clear()

    def is_requested(self) -> bool:
        return bool(self.requested)

# This is the complete built-in task-kind surface.  It is deliberately data,
# not import strings; `_production_handlers` below imports each owned adapter
# explicitly and rejects an incomplete map.
TASK_KINDS: Final = frozenset({
    "tap_schema", "surface_parity", "architecture_attestation",
    "parent_loss_attestation", "parent_import", "kernel_resources",
    "representation_recipe", "numerical_acceptance", "target_build",
    "control_registry", "cache_miniature_bank", "cache_miniature",
    "smoke_probe", "zero_coefficient_acceptance", "reservation",
    "train_node", "train_control", "shuffle_map",
    "target_cleanup", "screen_aggregate", "confirmation_registry",
    "confirmation", "confirmation_aggregate", "finalist_lock",
    "shared_final_claim", "final_selection", "assignment_shard",
    "assignment_finalize", "data_attestation", "execution_lock",
    "prediction_shard", "prediction_finalize", "metric_join",
    "final_aggregate", "validation_only_aggregate",
})


def _validate_environment(spec: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    expected_path = str(Path(spec["artifact_paths"]["runtime_binding"]))
    environment_path = os.environ.get("HCWDL_REPRESENTATION_RUNTIME_BINDING")
    if environment_path is not None and str(Path(environment_path)) != expected_path:
        raise ValueError("worker runtime-binding path differs from campaign")
    facts = binding["runtime_facts"]
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PermissionError("HCWDL-RKD worker user-site isolation is absent")
    if Path(str(facts["project_dir"])) != Path(str(spec["project_dir"])):
        raise ValueError("HCWDL-RKD worker project directory differs")


def _directory_identity_candidates(path: Path) -> set[str]:
    """Return the sole accepted identity: the complete directory inventory.

    A manifest/sidecar content hash authenticates that JSON file, not arbitrary
    siblings.  Accepting it as the directory identity would allow an unrelated
    payload member to change without invalidating the registered input.
    """

    from hlt_classification.data.cache_contracts import canonical_sha256

    inventory = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise PermissionError("HCWDL-RKD registered input directory contains a symlink")
        if not member.is_file():
            continue
        inventory.append({
            "path": member.relative_to(path).as_posix(),
            "bytes": member.stat().st_size,
            "sha256": sha256_file(member),
        })
    if not inventory:
        raise ValueError("HCWDL-RKD registered input directory is empty")
    return {canonical_sha256(inventory)}


def _resolve_immutable_output_root(
    descriptor: Mapping[str, Any], *, upstream: Mapping[str, Any] | None = None,
    campaign_identity_sha256: str | None = None,
) -> tuple[Path, str]:
    """Resolve one producer-owned root to its sole authenticated commit.

    The final envelope ID is deliberately not guessed before execution.  A
    consumer accepts exactly one committed child, validates the complete
    binary-envelope contract and bytes, and checks its frozen producer and
    registered-output route.
    """

    required = {
        "root", "artifact_contract", "producer_task_id",
        "registered_output_row", "registered_route", "template",
        "campaign_identity_sha256", "expected_publication_owner",
        "expected_parent_sources",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != required:
        raise ValueError("HCWDL-RKD immutable output-root descriptor differs")
    if descriptor["template"] != "committed/${envelope_id}":
        raise ValueError("HCWDL-RKD immutable output template differs")
    if (
        campaign_identity_sha256 is not None
        and descriptor["campaign_identity_sha256"] != campaign_identity_sha256
    ):
        raise PermissionError("HCWDL-RKD immutable output campaign lineage differs")
    root = Path(str(descriptor["root"]))
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError("HCWDL-RKD immutable output root is absent")
    committed = root / "committed"
    leaves = sorted(
        child for child in committed.iterdir()
        if child.is_dir() and not child.is_symlink()
    ) if committed.is_dir() and not committed.is_symlink() else []
    if len(leaves) != 1:
        raise ValueError(
            "HCWDL-RKD immutable output root must contain exactly one committed envelope"
        )
    leaf = leaves[0]
    from .hcwdl_representation_artifacts import (
        derive_envelope_owner_id, validate_binary_envelope,
    )

    parents = _resolve_expected_parent_sources(
        descriptor["expected_parent_sources"], envelope_directory=leaf,
    )
    expected_owner = descriptor["expected_publication_owner"]
    if not isinstance(expected_owner, Mapping) or not expected_owner:
        raise ValueError("HCWDL-RKD immutable envelope expected owner differs")
    expected_owner_id = derive_envelope_owner_id(
        envelope_id=leaf.name,
        campaign_or_recovery_owner=expected_owner,
    )
    envelope = validate_binary_envelope(
        root, leaf.name,
        expected_contract=str(descriptor["artifact_contract"]),
        expected_parents=parents,
        expected_owner_id=expected_owner_id,
    )
    payload = envelope.commit["payload"]
    if (
        payload["producer_task_id"] != descriptor["producer_task_id"]
        or payload["registered_output_row"] != descriptor["registered_output_row"]
        or payload["campaign_or_recovery_owner"] != expected_owner
    ):
        raise PermissionError(
            "HCWDL-RKD immutable envelope differs from its registered producer route"
        )
    route = descriptor["registered_route"]
    if not isinstance(route, Mapping) or set(route) != {
        "task_key", "array_index", "registered_output",
    }:
        raise ValueError("HCWDL-RKD immutable registered route differs")
    if upstream is not None and (
        route["task_key"] != upstream.get("task_key")
        or route["array_index"] != upstream.get("array_index")
        or route["registered_output"] != upstream.get("registered_output")
    ):
        raise PermissionError("HCWDL-RKD immutable envelope route lineage differs")
    digest = next(iter(_directory_identity_candidates(leaf)))
    return leaf, digest


def _resolve_expected_parent_sources(
    sources: object, *, envelope_directory: Path,
) -> dict[str, str]:
    """Resolve frozen parent derivations without trusting ``commit.parents``."""

    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("HCWDL-RKD immutable parent-source registry differs")
    from hlt_classification.data.cache_contracts import (
        load_json, require_sha256, validate_content_hash,
    )

    parents: dict[str, str] = {}
    for raw_name, source in sorted(sources.items()):
        name = str(raw_name)
        if not name or not isinstance(source, Mapping):
            raise ValueError("HCWDL-RKD immutable parent-source row differs")
        kind = source.get("source_kind")
        if kind == "literal_sha256":
            digest = require_sha256(
                source.get("sha256"), name=f"immutable parent {name}",
            )
        elif kind in {
            "json_artifact", "envelope_json_member",
            "final_task_registry_field",
        }:
            if kind == "envelope_json_member":
                path = envelope_directory / str(source.get("member", ""))
                try:
                    path.resolve(strict=False).relative_to(
                        envelope_directory.resolve(),
                    )
                except ValueError as error:
                    raise ValueError(
                        "HCWDL-RKD immutable parent member escapes its envelope"
                    ) from error
            else:
                path = Path(str(source.get("path", "")))
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(
                    f"HCWDL-RKD immutable parent artifact is absent: {name}"
                )
            artifact = load_json(path)
            artifact_hash = validate_content_hash(
                artifact,
                expected_contract=str(source.get("expected_contract", "")),
                expected_schema_version=int(
                    source.get("expected_schema_version", 0),
                ),
            )
            if kind == "final_task_registry_field":
                rows = artifact.get("tasks")
                matches = [
                    row for row in rows if isinstance(row, Mapping)
                    and row.get("task_id") == source.get("task_id")
                ] if isinstance(rows, list) else []
                if len(matches) != 1:
                    raise ValueError(
                        "HCWDL-RKD immutable parent task-registry selector differs"
                    )
                digest = require_sha256(
                    matches[0].get(str(source.get("field"))),
                    name=f"immutable parent {name}",
                )
            else:
                digest = artifact_hash
        else:
            raise ValueError("HCWDL-RKD immutable parent source kind differs")
        parents[name] = digest
    return parents


def _validate_input_bytes(
    row: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Authenticate concrete byte inputs before a production adapter opens them.

    JSON artifacts are subsequently contract-validated by their owner.  The
    campaign-spec reference is the one acyclic campaign identity reference and
    is validated by the campaign loader rather than as a byte hash.
    """

    materialized: dict[str, dict[str, str]] = {}
    for logical_name, reference in row["inputs"].items():
        if not isinstance(reference, Mapping):
            raise ValueError("HCWDL-RKD registered input reference is not an object")
        campaign_artifact = reference.get(CAMPAIGN_ARTIFACT_BINDING)
        if campaign_artifact is not None:
            if logical_name != "${submission_ledger}" or set(reference) != {
                CAMPAIGN_ARTIFACT_BINDING, "path",
            }:
                raise PermissionError("campaign-produced artifact binding route differs")
            path = Path(reference["path"])
            expected_path = Path(str(spec["campaign_root"])) / "submission_ledger.json"
            if path != expected_path or not path.is_file() or path.is_symlink():
                raise ValueError("submission-ledger late-bound path differs")
            from hlt_classification.data.cache_contracts import load_json
            from .hcwdl_representation_campaign import (
                build_command_plan, validate_submission_ledger,
            )
            command_plan = build_command_plan(spec)
            if (
                campaign_artifact.get("artifact_role") != "submission_ledger"
                or campaign_artifact.get("expected_contract")
                != "HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1"
                or campaign_artifact.get("expected_schema_version") != 1
                or campaign_artifact.get("campaign_identity_sha256")
                != canonical_sha256(runtime_campaign_identity(spec))
                or campaign_artifact.get("command_plan_sha256")
                != command_plan["content_hash"]
            ):
                raise PermissionError("submission-ledger campaign/plan lineage differs")
            ledger = load_json(path)
            jobs = ledger.get("jobs") if isinstance(ledger, Mapping) else None
            ordered_tasks = [row["task_key"] for row in command_plan["commands"]]
            if isinstance(jobs, Mapping) and set(jobs) == set(ordered_tasks):
                # Immutable JSON is serialized with sorted object keys; restore
                # the scientifically meaningful reviewed command order before
                # the campaign validator performs its explicit order check.
                ledger = {**dict(ledger), "jobs": {key: jobs[key] for key in ordered_tasks}}
            validate_submission_ledger(ledger, spec=spec, command_plan=command_plan)
            materialized[logical_name] = {
                "path": str(path), "sha256": sha256_file(path),
            }
            continue
        prepublished = reference.get(PREPUBLISHED_OUTPUT_BINDING)
        if prepublished is not None:
            if set(reference) != {
                PREPUBLISHED_OUTPUT_BINDING, "path", "sha256",
            }:
                raise ValueError(
                    "HCWDL-RKD prepublished input reference fields differ"
                )
            descriptor = validate_prepublished_output_binding(
                prepublished, logical_input=logical_name, spec=spec,
            )
            path = Path(str(reference["path"]))
            expected_path = Path(str(spec["campaign_root"])) / str(
                descriptor["registered_output"]
            )
            if path != expected_path:
                raise PermissionError(
                    "HCWDL-RKD prepublished input is not its exact own output path"
                )
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    "HCWDL-RKD prepublished immutable output is absent"
                )
            observed_sha256 = sha256_file(path)
            if observed_sha256 != reference["sha256"]:
                raise ValueError(
                    "HCWDL-RKD prepublished input byte/hash identity differs"
                )
            from hlt_classification.data.cache_contracts import (
                load_json, validate_content_hash,
            )

            artifact = load_json(path)
            content_hash = validate_content_hash(
                artifact,
                expected_contract=str(descriptor["expected_contract"]),
                expected_schema_version=int(
                    descriptor["expected_schema_version"]
                ),
            )
            if content_hash != descriptor["expected_content_hash"]:
                raise PermissionError(
                    "HCWDL-RKD prepublished input differs from campaign content hash"
                )
            materialized[logical_name] = {
                "path": str(path), "sha256": observed_sha256,
            }
            continue
        immutable_root = reference.get(IMMUTABLE_OUTPUT_ROOT_BINDING)
        if immutable_root is not None:
            if set(reference) != {
                UPSTREAM_OUTPUT_BINDING, IMMUTABLE_OUTPUT_ROOT_BINDING,
            }:
                raise ValueError("HCWDL-RKD immutable upstream reference fields differ")
            upstream = reference.get(UPSTREAM_OUTPUT_BINDING)
            if not isinstance(upstream, Mapping) or upstream.get("artifact_kind") != "directory":
                raise ValueError("HCWDL-RKD immutable upstream artifact kind differs")
            leaf, digest = _resolve_immutable_output_root(
                immutable_root, upstream=upstream,
                campaign_identity_sha256=canonical_sha256(
                    runtime_campaign_identity(spec)
                ),
            )
            materialized[logical_name] = {"path": str(leaf), "sha256": digest}
            continue
        path = Path(reference["path"])
        if logical_name == "${campaign_spec}":
            if UPSTREAM_OUTPUT_BINDING in reference:
                raise PermissionError("campaign specification cannot be late-bound")
            expected = Path(str(spec["campaign_root"])) / "campaign_spec.json"
            if path != expected or not path.is_file():
                raise ValueError("HCWDL-RKD runtime campaign-spec path differs")
            from hlt_classification.data.cache_contracts import load_json

            if canonical_sha256(load_json(path)) != canonical_sha256(dict(spec)):
                raise ValueError("HCWDL-RKD runtime campaign-spec bytes differ")
            materialized[logical_name] = {
                "path": str(path), "sha256": sha256_file(path),
            }
            continue
        if not path.exists():
            raise FileNotFoundError(
                f"HCWDL-RKD registered input is absent: {logical_name} -> {path}"
            )
        if path.is_symlink():
            raise PermissionError(
                f"HCWDL-RKD registered input is a symlink: {logical_name}"
            )
        upstream = reference.get(UPSTREAM_OUTPUT_BINDING)
        if upstream is None:
            if set(reference) != {"path", "sha256"}:
                raise ValueError("HCWDL-RKD static input reference fields differ")
            if path.is_file():
                valid = {sha256_file(path)}
            elif path.is_dir():
                valid = _directory_identity_candidates(path)
            else:
                raise ValueError(
                    f"HCWDL-RKD registered input is not a file/directory: {logical_name}"
                )
            if reference["sha256"] not in valid:
                raise ValueError(
                    f"HCWDL-RKD registered input byte/hash identity differs: {logical_name}"
                )
            materialized[logical_name] = {
                "path": str(path), "sha256": str(reference["sha256"]),
            }
            continue
        if set(reference) != {UPSTREAM_OUTPUT_BINDING, "path"} or not isinstance(
            upstream, Mapping,
        ):
            raise ValueError("HCWDL-RKD late-bound input reference fields differ")
        artifact_kind = str(upstream.get("artifact_kind", ""))
        if artifact_kind == "json":
            if not path.is_file():
                raise ValueError("late-bound JSON output is not a file")
            from hlt_classification.data.cache_contracts import load_json, validate_content_hash

            value = load_json(path)
            validate_content_hash(
                value,
                expected_contract=str(upstream.get("expected_contract", "")),
                expected_schema_version=int(upstream.get("expected_schema_version", 0)),
            )
            digest = sha256_file(path)
        elif artifact_kind == "file":
            if not path.is_file():
                raise ValueError("late-bound file output is not a file")
            digest = sha256_file(path)
        elif artifact_kind == "directory":
            if not path.is_dir():
                raise ValueError("late-bound directory output is not a directory")
            digest = next(iter(_directory_identity_candidates(path)))
        else:
            raise ValueError("late-bound upstream artifact kind differs")
        materialized[logical_name] = {"path": str(path), "sha256": digest}
    return materialized


def _production_handlers() -> dict[str, Handler]:
    # Lazy import preserves CLI `--help` on nodes where Torch/Weaver has not
    # yet been activated.  The module contains only closed built-in adapters.
    from .hcwdl_representation_runtime_adapters import PRODUCTION_ADAPTERS

    handlers = dict(PRODUCTION_ADAPTERS)
    if set(handlers) != TASK_KINDS:
        missing = sorted(TASK_KINDS - set(handlers))
        extra = sorted(set(handlers) - TASK_KINDS)
        raise RuntimeError(
            f"HCWDL-RKD production adapter registry differs: missing={missing}, extra={extra}"
        )
    return handlers


def execute_registered_task(
    *, spec: Mapping[str, Any], task_key: str, array_index: int | None,
    deterministic_worker: bool, local_planning_fixture: bool = False,
    acceptance_bootstrap: Mapping[str, Any] | None = None,
) -> Any:
    """Execute exactly one registered row through its fixed kind adapter."""

    rows = [row for row in spec["tasks"] if row["task_key"] == task_key]
    if len(rows) != 1:
        raise KeyError("HCWDL-RKD task is not one exact campaign registry row")
    task_kind = str(rows[0]["kind"])
    if task_kind not in TASK_KINDS:
        raise KeyError(f"HCWDL-RKD task kind is not built in: {task_kind!r}")
    if local_planning_fixture and acceptance_bootstrap is not None:
        raise PermissionError("local fixture and acceptance bootstrap authorities conflict")
    if acceptance_bootstrap is not None:
        from .hcwdl_representation_bootstrap import (
            validate_acceptance_bootstrap_task,
        )

        validate_acceptance_bootstrap_task(
            acceptance_bootstrap,
            planning_spec=spec,
            task_key=task_key,
            deterministic_worker=deterministic_worker,
        )
    if local_planning_fixture:
        from .hcwdl_representation_runtime_adapters import (
            build_local_planning_handlers,
        )

        if task_kind in FINAL_ROLE_KINDS or task_kind in {"reservation", "finalist_lock"}:
            raise PermissionError(
                "local planning fixture cannot invoke a final/reservation adapter"
            )
        handlers = build_local_planning_handlers(
            TASK_KINDS - FINAL_ROLE_KINDS - {"reservation", "finalist_lock"},
        )
        workflow = RepresentationWorkflow(spec, handlers=handlers, executable=False)
        return workflow.run(
            task_key, array_index=array_index,
            deterministic_worker=deterministic_worker,
        )

    binding = load_runtime_binding(spec)
    _validate_environment(spec, binding)
    runtime_row = resolve_runtime_row(
        binding, spec=spec, task_key=task_key, array_index=array_index,
    )
    # Review-time runtime facts are commitments, not observations.  Measure
    # this exact process and checkout before opening any registered scientific
    # input or importing a production adapter.
    from .hcwdl_representation_worker_runtime import validate_live_task_runtime

    live_worker_runtime = validate_live_task_runtime(
        spec=spec, binding=binding, task=rows[0], runtime_row=runtime_row,
        deterministic_worker=deterministic_worker,
    )
    materialized_inputs = _validate_input_bytes(runtime_row, spec=spec)
    monitor = RepresentationPreemptionMonitor()
    monitor.install()
    runtime_context = dict(runtime_row)
    runtime_context["inputs"] = materialized_inputs
    runtime_context["_live_worker_runtime"] = live_worker_runtime
    runtime_context["_preemption_requested"] = monitor.is_requested
    production = _production_handlers()
    handlers = {
        kind: (
            lambda current_spec, task, index, _kind=kind: production[_kind](
                current_spec, task, index, runtime_context,
            )
        )
        for kind in TASK_KINDS
    }
    try:
        workflow = RepresentationWorkflow(
            spec,
            handlers=handlers,
            executable=acceptance_bootstrap is None,
        )
        return workflow.run(
            task_key, array_index=array_index,
            deterministic_worker=deterministic_worker,
        )
    finally:
        monitor.restore()


__all__ = [
    "RepresentationPreemptionMonitor", "TASK_KINDS", "execute_registered_task",
]
