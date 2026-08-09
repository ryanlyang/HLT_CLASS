"""Genuine bounded production bridge for the non-final validation proxy.

This module owns no campaign task kind and imports no shared-final/final data
surface.  It authenticates a deliberately filtered subset of the frozen
``parent_import`` runtime row, recomputes the canonical 256-row validation
selection in the sole label-bearing pass, and then runs independent
label-free assignment and D0c/D100/TOFF streams.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_contracts import (
    NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
    NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
)
from .hcwdl_representation_runtime_binding import resolve_runtime_row
from .hcwdl_representation_validation_proxy import (
    VALIDATION_ASSIGNMENT_BRANCHES,
    VALIDATION_BRANCH_ALLOWLISTS,
    VALIDATION_HLT_BRANCHES,
    VALIDATION_NATIVE_OFFLINE_BRANCHES,
    VALIDATION_PROXY_MODEL_IDS,
    VALIDATION_PROXY_PATHS,
    VALIDATION_PROXY_VIEW_REGISTRY,
    VALIDATION_SELECTION_BRANCHES,
    VALIDATION_SHELL_EXACT_BRANCHES,
    ValidationAccessRequest,
    ValidationModelRow,
    ValidationReadResult,
    build_validation_proxy_input_lineage,
    publish_validation_proxy_action_result,
    run_validation_proxy_action,
)


_ACTION_ID: Final = "validation_proxy"
_SOURCE_TASK_KEY: Final = "parent_import"
_ROW_LIMIT: Final = 256
_INPUT_LOGICAL: Final = {
    "matcher_resources": "${matcher_resources}",
    "parent_campaign_spec": "${parent_campaign_spec}",
    "parent_import": "${prebuilt_parent_import}",
    "parent_reports": "${parent_reports}",
    "parent_source_manifest": "${parent_source_manifest}",
    "split_manifest": "${split_manifest}",
    "validation_assignment_manifest": "${parent_assignment_manifest:validation}",
}
_FORBIDDEN_TOKENS: Final = (
    "final_test", "shared_final", "reservation", "capability", "escrow",
    "execution_lock", "finalist_lock", "final_selection",
)
_FORBIDDEN_ROUTE_COMPONENTS: Final = frozenset(_FORBIDDEN_TOKENS) | {
    "final",
}


@dataclass(frozen=True)
class ValidationProxyProductionResult:
    semantic_result_path: Path
    semantic_result: Mapping[str, Any]
    workspace: Path
    dependencies: Mapping[str, Mapping[str, str]]
    source_task_key: str


@dataclass(frozen=True)
class _Inputs:
    authority: Mapping[str, Any]
    authority_sha256: str
    action_inputs: Mapping[str, Any]
    source_row: Mapping[str, Any]
    descriptor: Mapping[str, Any]
    bounded_selection: Mapping[str, Any]
    planning: Mapping[str, Any]
    runtime: Mapping[str, Any]
    registered: Mapping[str, Mapping[str, str]]
    parent_campaign: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    split: Mapping[str, Any]
    matcher_resources: Mapping[str, Any]
    assignment_manifest_path: Path
    assignment_manifest: Mapping[str, Any]
    parent_import: Mapping[str, Any]
    parent_report_root: Path
    data_root: Path
    live_worker_runtime: Mapping[str, Any]
    input_lineage: Mapping[str, Any]
    model_bindings: Mapping[str, Mapping[str, str]]
    models: Mapping[str, Any]
    semantic_path: Path
    workspace: Path


def _forbidden_route(path: Path) -> bool:
    return any(
        part.lower() in _FORBIDDEN_ROUTE_COMPONENTS for part in path.parts
    )


def _safe_reference(reference: object, *, name: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"validation proxy {name} reference fields differ")
    path = Path(str(reference["path"]))
    if _forbidden_route(path):
        raise PermissionError(
            f"validation proxy {name} reference names a forbidden final route"
        )
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError(f"validation proxy {name} reference is unsafe")
    expected = require_sha256(reference["sha256"], name=f"{name} bytes")
    if sha256_file(path) != expected:
        raise ValueError(f"validation proxy {name} bytes differ")
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"validation proxy {name} is not a JSON object")
    return path.resolve(), dict(value)


def _reject_forbidden(value: object, *, location: str) -> None:
    if isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        if any(token in lowered for token in _FORBIDDEN_TOKENS):
            raise PermissionError(
                f"validation proxy {location} names a forbidden final surface"
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_forbidden(key, location=f"{location}.key")
            _reject_forbidden(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden(item, location=f"{location}[{index}]")


def _authority_digest_only(authority: Mapping[str, Any]) -> str:
    """Validate only the immutable envelope before any registered data read."""

    digest = validate_content_hash(
        authority,
        expected_contract=NONFINAL_ACCEPTANCE_AUTHORITY_CONTRACT,
        expected_schema_version=1,
    )
    action = authority.get("actions", {}).get(_ACTION_ID)
    if (
        not isinstance(action, Mapping)
        or action.get("kind") != _ACTION_ID
        or action.get("worker_role") != "deterministic"
        or action.get("resource_class") != "gpu_final_prediction"
        or action.get("validation_rows") != _ROW_LIMIT
        or action.get("final_rows") != 0
        or action.get("campaign_task_kind") is not None
    ):
        raise PermissionError("validation proxy immutable action route differs")
    false_claims = (
        "arrays_authorized", "campaign_training_authorized",
        "reservation_authorized", "shared_final_authorized",
        "final_role_access_authorized", "pilot_submission_authorized",
        "scheduler_submission_authorized", "scheduler_mutated",
    )
    if any(authority.get(name) is not False for name in false_claims):
        raise PermissionError("validation proxy immutable authority boundary differs")
    return digest


def _measure_live_runtime(
    authority: Mapping[str, Any], *, project_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Measure the new action's own GPU class; never inherit the CPU source row."""

    _, planning = _safe_reference(authority.get("planning_spec"), name="planning spec")
    _, runtime = _safe_reference(authority.get("runtime_binding"), name="runtime binding")
    from .hcwdl_representation_runtime_binding import validate_runtime_binding

    validate_runtime_binding(runtime, spec=planning)
    facts = runtime.get("runtime_facts")
    snapshot = authority.get("source_snapshot")
    if not isinstance(facts, Mapping) or not isinstance(snapshot, Mapping):
        raise ValueError("validation proxy runtime/source facts are absent")
    from .hcwdl_representation_worker_runtime import measure_live_worker_runtime

    live = measure_live_worker_runtime(
        project_dir=project_dir,
        expected_conda_environment=str(facts["conda_environment"]),
        expected_source_commit=str(authority["source_commit"]),
        expected_source_snapshot_sha256=require_sha256(
            snapshot.get("source_snapshot_sha256"), name="authority source snapshot",
        ),
        expected_weaver_runtime_sha256=require_sha256(
            facts.get("weaver_runtime_sha256"), name="authority Weaver runtime",
        ),
        row_device="cuda",
        resource_class="gpu_final_prediction",
        deterministic_worker=True,
    )
    return dict(live), planning, runtime


def _static_authority(
    authority: Mapping[str, Any], *, project_dir: Path,
) -> str:
    from .hcwdl_representation_nonfinal_acceptance import (
        validate_nonfinal_acceptance_authority_static,
    )

    return validate_nonfinal_acceptance_authority_static(
        authority, project_dir=project_dir,
    )


def _materialize_registered_inputs(
    source_row: Mapping[str, Any], *, planning: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    inputs = source_row.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("validation proxy source runtime inputs are absent")
    required = set(_INPUT_LOGICAL.values())
    if not required <= set(inputs):
        raise ValueError("validation proxy required registered inputs are absent")
    filtered = {logical: inputs[logical] for logical in sorted(required)}
    _reject_forbidden(filtered, location="registered_inputs")
    from .hcwdl_representation_task_runtime import _validate_input_bytes

    materialized = _validate_input_bytes({"inputs": filtered}, spec=planning)
    if set(materialized) != required:
        raise PermissionError("validation proxy materialized input set differs")
    return {
        name: materialized[logical]
        for name, logical in sorted(_INPUT_LOGICAL.items())
    }


def _json_input(
    registered: Mapping[str, Mapping[str, str]], name: str,
) -> dict[str, Any]:
    reference = registered[name]
    path = Path(reference["path"])
    if not path.is_file() or path.is_symlink() or sha256_file(path) != reference["sha256"]:
        raise PermissionError(f"validation proxy registered {name} bytes differ")
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"validation proxy registered {name} is not an object")
    return dict(value)


def _within_registered_directory(path: str | Path, root: Path, *, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"validation proxy {name} is absent")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise PermissionError(
            f"validation proxy {name} lies outside registered parent reports"
        ) from error
    cursor = root.resolve()
    for part in resolved.relative_to(root.resolve()).parts:
        cursor /= part
        if cursor.is_symlink():
            raise PermissionError(f"validation proxy {name} crosses a symlink")
    return resolved


def _validate_data_root(data_root: object) -> Path:
    path = Path(str(data_root))
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PermissionError("validation proxy data root is unsafe")
    if _forbidden_route(path):
        raise PermissionError("validation proxy data root names a final route")
    return path.resolve()


def _source_inventory(source_manifest: Mapping[str, Any]):
    from .audit import SOURCE_MANIFEST_CONTRACT, SOURCE_MANIFEST_VERSION
    from .splits import source_file_record_from_manifest_row

    digest = validate_content_hash(
        source_manifest,
        expected_contract=SOURCE_MANIFEST_CONTRACT,
        expected_schema_version=SOURCE_MANIFEST_VERSION,
    )
    rows = source_manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation proxy source inventory is empty")
    return digest, tuple(source_file_record_from_manifest_row(row) for row in rows)


def _parent_models(
    parent_import: Mapping[str, Any], *, report_root: Path, device: str,
) -> tuple[
    dict[str, Any], dict[str, dict[str, str]], dict[str, str],
    dict[str, dict[str, Any]],
]:
    from .hcwdl_representation_locks import validate_parent_import
    from .hcwdl_representation_production import (
        _load_model_source, _pmard_report_chain,
    )

    validate_parent_import(parent_import)
    teachers = {
        str(row["node_id"]): row
        for row in parent_import["payload"]["teachers"]
    }
    required = set(VALIDATION_PROXY_MODEL_IDS.values())
    if not required <= set(teachers):
        raise ValueError("validation proxy imported model registry is incomplete")
    models: dict[str, Any] = {}
    bindings: dict[str, dict[str, str]] = {}
    reports: dict[str, str] = {}
    source_lineage: dict[str, dict[str, Any]] = {}
    for path in VALIDATION_PROXY_PATHS:
        node_id = VALIDATION_PROXY_MODEL_IDS[path]
        row = teachers[node_id]
        report_path = _within_registered_directory(
            row["report_path"], report_root, name=f"{node_id} report",
        )
        checkpoint_path = _within_registered_directory(
            row["checkpoint_path"], report_root, name=f"{node_id} checkpoint",
        )
        report = load_json(report_path)
        if (
            not isinstance(report, Mapping)
            or report.get("content_hash") != row["report_sha256"]
            or sha256_file(checkpoint_path) != row["checkpoint_sha256"]
            or row["checkpoint_sha256"] != row["checkpoint_byte_sha256"]
        ):
            raise PermissionError(f"validation proxy {node_id} imported bytes differ")
        report_reference = {
            "path": str(report_path), "sha256": sha256_file(report_path),
        }
        source = {"kind": "pmard", "report": report_reference}
        chain = _pmard_report_chain(
            report_reference, name=f"{node_id} validation source",
        )
        engine_path = _within_registered_directory(
            chain["engine_path"], report_root, name=f"{node_id} engine report",
        )
        if (
            chain["wrapper"] is None
            or Path(chain["wrapper_path"]).resolve() != report_path
            or Path(chain["checkpoint"]).resolve() != checkpoint_path
        ):
            raise PermissionError(
                f"validation proxy {node_id} wrapper/engine route differs"
            )
        engine_config = chain["engine"].get("config")
        if not isinstance(engine_config, Mapping):
            raise ValueError(
                f"validation proxy {node_id} engine configuration is absent"
            )
        engine_config_sha256 = canonical_sha256(engine_config)
        wrapper_execution_sha256 = require_sha256(
            chain["wrapper"].get("pmard_execution_config_sha256"),
            name=f"{node_id} wrapper execution config",
        )
        engine_execution_sha256 = require_sha256(
            chain["engine"].get("execution_config_sha256"),
            name=f"{node_id} engine execution config",
        )
        if not (
            wrapper_execution_sha256
            == engine_execution_sha256
            == engine_config_sha256
        ):
            raise PermissionError(
                f"validation proxy {node_id} execution configuration differs"
            )
        extraction_sha256 = canonical_sha256({
            "model_input": engine_config.get("model_input"),
            "representation_arm": engine_config.get("representation_arm", "R0"),
        })
        model, loaded_checkpoint, checkpoint_sha256 = _load_model_source(
            source,
            name=node_id,
            device=device,
        )
        if (
            loaded_checkpoint.resolve() != checkpoint_path
            or checkpoint_sha256 != row["checkpoint_sha256"]
        ):
            raise PermissionError(
                f"validation proxy {node_id} model loader lineage differs"
            )
        models[path] = model
        model_lineage = with_content_hash({
            "kind": "validation_proxy_model_source",
            "schema_version": 1,
            "node_id": node_id,
            "source_kind": "authenticated_pmard_parent_v1",
            "parent_import_row_sha256": canonical_sha256(dict(row)),
            "wrapper_report_content_sha256": require_sha256(
                row["report_sha256"], name=f"{node_id} wrapper report",
            ),
            "wrapper_report_byte_sha256": sha256_file(report_path),
            "engine_report_content_sha256": require_sha256(
                chain["engine"].get("content_hash"),
                name=f"{node_id} engine report",
            ),
            "engine_report_byte_sha256": sha256_file(engine_path),
            "wrapper_execution_config_sha256": wrapper_execution_sha256,
            "engine_execution_config_sha256": engine_execution_sha256,
            "engine_config_sha256": engine_config_sha256,
            "model_extraction_sha256": extraction_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        })
        source_lineage[node_id] = model_lineage
        bindings[path] = {
            "view_id": VALIDATION_PROXY_VIEW_REGISTRY[path],
            "model_id": node_id,
            "checkpoint_sha256": checkpoint_sha256,
            "model_source_lineage_sha256": model_lineage["content_hash"],
        }
        reports[node_id] = require_sha256(
            row["report_sha256"], name=f"{node_id} report",
        )
    return models, bindings, reports, source_lineage


def _resolve_inputs(
    *, authority: Mapping[str, Any], authority_sha256: str,
    planning: Mapping[str, Any], runtime: Mapping[str, Any],
    live_worker_runtime: Mapping[str, Any],
) -> _Inputs:
    return _resolve_inputs_impl(
        authority=authority,
        authority_sha256=authority_sha256,
        planning=planning,
        runtime=runtime,
        live_worker_runtime=live_worker_runtime,
    )


def _slice_arrays(arrays: Mapping[str, Any], indexes: np.ndarray) -> dict[str, Any]:
    return {name: value[indexes] for name, value in arrays.items()}


def _slice_particle(view: Any, index: int):
    values = (
        view.features[index:index + 1],
        view.vectors[index:index + 1],
        view.mask[index:index + 1],
        view.raw_lengths[index:index + 1],
    )
    return type(view)(*values)


def _concat_particles(views: Sequence[Any]):
    if not views:
        raise ValueError("validation proxy cannot concatenate empty inputs")
    return type(views[0])(
        np.ascontiguousarray(np.concatenate([view.features for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.vectors for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.mask for view in views], axis=0)),
        np.ascontiguousarray(np.concatenate([view.raw_lengths for view in views], axis=0)),
    )


def _identity(source_sha256: str, entry: int) -> str:
    return canonical_sha256({
        "source_file_sha256": source_sha256,
        "source_entry": int(entry),
    })


class _ProxyBackend:
    def __init__(self, inputs: _Inputs) -> None:
        self.inputs = inputs
        from .splits import role_records

        records = role_records(inputs.split, "validation")
        self.records = {record.path: record for record in records}
        expected_sources = inputs.bounded_selection["roles"]["validation"]["sources"]
        if [row["path"] for row in expected_sources] != [record.path for record in records]:
            raise PermissionError("validation proxy bounded source order differs")
        self.selected = {
            str(row["path"]): tuple(int(entry) for entry in row["entries"])
            for row in expected_sources
        }
        if sum(map(len, self.selected.values())) != _ROW_LIMIT:
            raise ValueError("validation proxy bounded source rows differ")
        for record in records:
            path = inputs.data_root / record.path
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"validation proxy source file is absent: {record.path}"
                )
            if sha256_file(path) != record.sha256:
                raise PermissionError(
                    f"validation proxy source-file bytes differ: {record.path}"
                )

    @staticmethod
    def _check_request(
        request: ValidationAccessRequest, *, path: str,
        identities: Sequence[str] = (),
    ) -> None:
        if (
            request.role != "validation"
            or request.path != path
            or request.row_limit != _ROW_LIMIT
            or request.projected_branches
            != tuple(sorted(VALIDATION_BRANCH_ALLOWLISTS[path]))
            or request.labels_allowed is not (path == "selection")
            or tuple(request.selected_identities) != tuple(identities)
        ):
            raise PermissionError(f"validation proxy {path} access request differs")

    def _chunks(self, branches: Sequence[str] | set[str] | frozenset[str]):
        from .streaming import iterate_projected_chunks

        for record in self.records.values():
            prior_stop = 0
            for chunk in iterate_projected_chunks(
                (self.inputs.data_root / record.path,),
                branches,
                data_root=self.inputs.data_root,
                role="validation",
                completed_locks=(),
                step_size=4096,
            ):
                if (
                    chunk.source_path != record.path
                    or chunk.entry_start != prior_stop
                    or chunk.entry_stop <= chunk.entry_start
                ):
                    raise ValueError("validation proxy source ranges reorder or gap")
                prior_stop = chunk.entry_stop
                access = {
                    "source_path": record.path,
                    "source_file_sha256": record.sha256,
                    "tree": "tree",
                    "entry_start": chunk.entry_start,
                    "entry_stop": chunk.entry_stop,
                }
                selected = np.asarray([
                    entry for entry in self.selected[record.path]
                    if chunk.entry_start <= entry < chunk.entry_stop
                ], dtype=np.int64)
                indexes = selected - chunk.entry_start
                yield record, chunk, selected, indexes, access
            if prior_stop != record.raw_entries:
                raise ValueError("validation proxy source range coverage differs")

    def selection_reader(
        self, request: ValidationAccessRequest,
    ) -> ValidationReadResult:
        self._check_request(request, path="selection")
        from .labels import baseline_mask, multiclass_labels
        from .selective_assignment import _largest_remainder, _selection_rank

        role = self.inputs.split["roles"]["validation"]
        targets = _largest_remainder(role["class_counts"], _ROW_LIMIT)
        heaps: list[list[tuple[int, str, int]]] = [[] for _ in range(15)]
        observed = np.zeros(15, np.int64)
        accesses = []
        for _record, chunk, _selected, _indexes, access in self._chunks(
            VALIDATION_SELECTION_BRANCHES,
        ):
            accesses.append(access)
            labels = multiclass_labels(chunk.arrays)
            indexes = np.flatnonzero(baseline_mask(chunk.arrays) & (labels >= 0))
            observed += np.bincount(labels[indexes], minlength=15)
            for index in indexes:
                category = int(labels[index])
                target = targets[category]
                if target == 0:
                    continue
                entry = chunk.entry_start + int(index)
                rank = _selection_rank(
                    1337, "validation", chunk.source_path, entry,
                )
                item = (-rank, chunk.source_path, entry)
                heap = heaps[category]
                if len(heap) < target:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        if observed.tolist() != role["class_counts"]:
            raise ValueError("validation proxy observed population classes differ")
        by_source = {path: [] for path in self.records}
        label_by_source_entry: dict[tuple[str, int], int] = {}
        for category, heap in enumerate(heaps):
            if len(heap) != targets[category]:
                raise ValueError(
                    f"validation proxy could not fill class {category}"
                )
            for _negative_rank, path, entry in heap:
                by_source[path].append(entry)
                label_by_source_entry[(path, entry)] = category
        recomputed = {
            "all_rows": False,
            "rows": _ROW_LIMIT,
            "class_counts": [len(heap) for heap in heaps],
            "population_class_counts": list(role["class_counts"]),
            "sources": [
                {
                    "path": record.path,
                    "rows": len(by_source[record.path]),
                    "entries": sorted(by_source[record.path]),
                }
                for record in self.records.values()
            ],
        }
        expected = self.inputs.bounded_selection["roles"]["validation"]
        if recomputed != expected:
            raise PermissionError(
                "validation proxy sole label-bearing scan differs from canonical selection"
            )
        rows = []
        for record in self.records.values():
            for entry in self.selected[record.path]:
                rows.append({
                    "identity_digest": _identity(record.sha256, entry),
                    "source_path": record.path,
                    "source_file_sha256": record.sha256,
                    "source_entry": entry,
                    "label": label_by_source_entry[(record.path, entry)],
                })
        return ValidationReadResult(rows=tuple(rows), source_accesses=tuple(accesses))

    def assignment_reader(
        self, request: ValidationAccessRequest,
    ) -> ValidationReadResult:
        identities = tuple(
            _identity(self.records[path].sha256, entry)
            for path in self.records
            for entry in self.selected[path]
        )
        self._check_request(request, path="assignment", identities=identities)
        from .highcov_cache import (
            DenseAssignmentStore,
            quantize_confidence,
        )
        from .highcov_matcher import (
            HighCoverageMatcher,
            from_scouting_particles,
            model_key_for_role,
        )
        from .highcov_resources import load_highcov_resources
        from .particles import decode_particle_sets

        resources = load_highcov_resources()
        matcher = HighCoverageMatcher(
            resources.empirical,
            resources.calibration,
            model_key=model_key_for_role("validation", None),
        )
        store = DenseAssignmentStore(self.inputs.assignment_manifest_path)
        rows, accesses = [], []
        for record, chunk, entries, indexes, access in self._chunks(
            VALIDATION_ASSIGNMENT_BRANCHES,
        ):
            accesses.append(access)
            if not len(indexes):
                continue
            arrays = _slice_arrays(chunk.arrays, indexes)
            for local, entry in enumerate(entries):
                hlt_raw, offline_raw, _ = decode_particle_sets(arrays, local)
                result = matcher.match(
                    from_scouting_particles(hlt_raw, offline=False),
                    from_scouting_particles(offline_raw, offline=True),
                )
                cached = store.get(record.path, int(entry))
                if (
                    not np.array_equal(
                        np.asarray(result.native_offline_index, np.int16),
                        cached.native_offline_index,
                    )
                    or not np.array_equal(
                        quantize_confidence(
                            result.confidence, result.native_offline_index,
                        ),
                        quantize_confidence(
                            cached.confidence, cached.native_offline_index,
                        ),
                    )
                ):
                    raise PermissionError(
                        "validation proxy label-free matcher/cache recomputation differs"
                    )
                rows.append({
                    "identity_digest": _identity(record.sha256, int(entry)),
                    "assignment": {
                        "native_offline_index": cached.native_offline_index,
                        "confidence": cached.confidence,
                    },
                })
        return ValidationReadResult(rows=tuple(rows), source_accesses=tuple(accesses))

    def stream_reader(
        self, path: str, request: ValidationAccessRequest,
        assignments: Mapping[str, Any] | None,
    ) -> ValidationReadResult:
        identities = tuple(
            _identity(self.records[source].sha256, entry)
            for source in self.records
            for entry in self.selected[source]
        )
        self._check_request(request, path=path, identities=identities)
        if path not in VALIDATION_PROXY_PATHS:
            raise ValueError("validation proxy model stream path differs")
        if (assignments is None) is (path == "shell_exact"):
            raise ValueError("validation proxy Shell-Exact assignment route differs")
        from .inputs import build_hlt_inputs, build_native_offline_inputs
        from .repair import (
            HIGHCOV_SHELL_EXACT_FAMILY,
            build_alpha_repaired_inputs,
            combined_offline_p4,
        )

        branches = {
            "hlt": VALIDATION_HLT_BRANCHES,
            "shell_exact": VALIDATION_SHELL_EXACT_BRANCHES,
            "native_offline": VALIDATION_NATIVE_OFFLINE_BRANCHES,
        }[path]
        rows, accesses = [], []
        for record, chunk, entries, indexes, access in self._chunks(branches):
            accesses.append(access)
            if not len(indexes):
                continue
            arrays = _slice_arrays(chunk.arrays, indexes)
            if path == "hlt":
                view = build_hlt_inputs(arrays)
            elif path == "native_offline":
                view = build_native_offline_inputs(arrays)
            else:
                assert assignments is not None
                canonical = build_hlt_inputs(arrays)
                width = int(canonical.features.shape[2])
                mapping = np.full((len(entries), width), -1, np.int16)
                confidence = np.zeros((len(entries), width), np.float32)
                offline_p4 = []
                identity_keys = []
                for local, entry in enumerate(entries):
                    identity = _identity(record.sha256, int(entry))
                    payload = assignments.get(identity)
                    if not isinstance(payload, Mapping) or set(payload) != {
                        "native_offline_index", "confidence",
                    }:
                        raise ValueError(
                            "validation proxy Shell-Exact assignment payload differs"
                        )
                    native = np.asarray(payload["native_offline_index"], np.int16)
                    probability = np.asarray(payload["confidence"], np.float32)
                    if (
                        native.ndim != 1
                        or probability.shape != native.shape
                        or len(native) > width
                        or not np.isfinite(probability).all()
                    ):
                        raise ValueError(
                            "validation proxy Shell-Exact assignment shape differs"
                        )
                    mapping[local, :len(native)] = native
                    confidence[local, :len(native)] = probability
                    offline_p4.append(combined_offline_p4(arrays, arrays, local))
                    identity_keys.append(f"{record.path}::tree::{int(entry)}")
                view = build_alpha_repaired_inputs(
                    arrays,
                    offline_p4,
                    mapping,
                    alpha=1.0,
                    repair_family=HIGHCOV_SHELL_EXACT_FAMILY,
                    confidence_weights=confidence,
                    offline_arrays=arrays,
                    identity_keys=identity_keys,
                    discrete_seed=1337,
                )
            for local, entry in enumerate(entries):
                identity = _identity(record.sha256, int(entry))
                if path == "native_offline":
                    model_inputs = type(view)(
                        _slice_particle(view.charged, local),
                        _slice_particle(view.neutral, local),
                    )
                else:
                    model_inputs = _slice_particle(view, local)
                rows.append({
                    "identity_digest": identity,
                    "model_inputs": model_inputs,
                })
        return ValidationReadResult(rows=tuple(rows), source_accesses=tuple(accesses))

    def predictor(
        self, path: str, request: ValidationAccessRequest,
        rows: Sequence[ValidationModelRow],
    ) -> np.ndarray:
        if path not in VALIDATION_PROXY_PATHS or request.path != path:
            raise PermissionError("validation proxy predictor route differs")
        binding = self.inputs.model_bindings[path]
        if any((
            request.view_id != binding["view_id"],
            request.model_id != binding["model_id"],
            request.checkpoint_sha256 != binding["checkpoint_sha256"],
            request.model_source_lineage_sha256
            != binding["model_source_lineage_sha256"],
        )):
            raise PermissionError("validation proxy predictor model lineage differs")
        import torch
        from .inputs import NativeOfflineInputs

        model = self.inputs.models[path]
        model.to("cuda").float().eval()
        if model.training or any(
            parameter.dtype != torch.float32 for parameter in model.parameters()
        ):
            raise ValueError("validation proxy model is not eval-mode FP32")
        views = [row.model_inputs for row in rows]
        if path == "native_offline":
            if not all(isinstance(view, NativeOfflineInputs) for view in views):
                raise TypeError("validation proxy TOFF input topology differs")
            charged = _concat_particles([view.charged for view in views])
            neutral = _concat_particles([view.neutral for view in views])
            arguments = (
                charged.features, charged.vectors, charged.mask,
                neutral.features, neutral.vectors, neutral.mask,
            )
        else:
            if any(isinstance(view, NativeOfflineInputs) for view in views):
                raise TypeError("validation proxy ordinary input topology differs")
            ordinary = _concat_particles(views)
            arguments = (ordinary.features, ordinary.vectors, ordinary.mask)
        tensors = []
        for value in arguments:
            tensor = torch.as_tensor(value, device="cuda")
            tensors.append(tensor.float() if tensor.dtype.is_floating_point else tensor)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            output = model(*tensors).float()
        if output.shape != (len(rows), 15) or not torch.isfinite(output).all():
            raise FloatingPointError("validation proxy model emitted invalid logits")
        return np.ascontiguousarray(output.cpu().numpy(), dtype=np.float32)


def execute_validation_proxy_production_action(
    *, authority: Mapping[str, Any], authority_path: str | Path,
    project_dir: str | Path, deterministic_worker: bool,
) -> ValidationProxyProductionResult:
    """Execute and publish exactly the authorized validation-proxy semantic result."""

    if deterministic_worker is not True:
        raise PermissionError("validation proxy requires the deterministic worker")
    project = Path(project_dir).resolve()
    path = Path(authority_path)
    if (
        not path.is_absolute() or not path.is_file() or path.is_symlink()
        or _forbidden_route(path)
        or load_json(path) != dict(authority)
    ):
        raise PermissionError("validation proxy authority file differs")
    live, planning, runtime = _measure_live_runtime(authority, project_dir=project)
    authority_sha256 = _authority_digest_only(authority)
    if _static_authority(authority, project_dir=project) != authority_sha256:
        raise PermissionError("validation proxy static authority digest differs")
    inputs = _resolve_inputs(
        authority=authority,
        authority_sha256=authority_sha256,
        planning=planning,
        runtime=runtime,
        live_worker_runtime=live,
    )
    backend = _ProxyBackend(inputs)
    static_validator = lambda value: _static_authority(value, project_dir=project)
    stream_readers = {
        stream: (
            lambda request, assignments, _stream=stream:
            backend.stream_reader(_stream, request, assignments)
        )
        for stream in VALIDATION_PROXY_PATHS
    }
    predictors = {
        stream: (
            lambda request, rows, _stream=stream:
            backend.predictor(_stream, request, rows)
        )
        for stream in VALIDATION_PROXY_PATHS
    }
    result = run_validation_proxy_action(
        authority=authority,
        authority_validator=static_validator,
        selection_reader=backend.selection_reader,
        assignment_reader=backend.assignment_reader,
        stream_readers=stream_readers,
        predictors=predictors,
        model_bindings=inputs.model_bindings,
        input_lineage=inputs.input_lineage,
    )
    semantic_reference = publish_validation_proxy_action_result(
        inputs.semantic_path,
        result=result,
        authority=authority,
        authority_validator=static_validator,
    )
    if Path(semantic_reference["path"]).resolve() != inputs.semantic_path.resolve():
        raise PermissionError("validation proxy publisher changed its semantic route")
    return ValidationProxyProductionResult(
        semantic_result_path=inputs.semantic_path.resolve(),
        semantic_result=result,
        workspace=inputs.workspace.resolve(),
        dependencies={},
        source_task_key=_SOURCE_TASK_KEY,
    )


__all__ = [
    "ValidationProxyProductionResult",
    "execute_validation_proxy_production_action",
]


def _resolve_inputs_impl(
    *, authority: Mapping[str, Any], authority_sha256: str,
    planning: Mapping[str, Any], runtime: Mapping[str, Any],
    live_worker_runtime: Mapping[str, Any],
) -> _Inputs:
    action_inputs_path, action_inputs = _safe_reference(
        authority.get("action_inputs"), name="action inputs",
    )
    action_inputs_sha256 = validate_content_hash(
        action_inputs,
        expected_contract=NONFINAL_ACCEPTANCE_ACTION_INPUTS_CONTRACT,
        expected_schema_version=1,
    )
    if (
        action_inputs_sha256 != authority.get("action_inputs_sha256")
        or action_inputs.get("derivation_kind")
        != "canonical_full_smoke_projection_v1"
        or action_inputs.get("final_role_access_authorized") is not False
        or action_inputs.get("shared_final_authorized") is not False
    ):
        raise PermissionError("validation proxy action-input authority differs")
    action_row = action_inputs.get("actions", {}).get(_ACTION_ID)
    if not isinstance(action_row, Mapping):
        raise ValueError("validation proxy action-input row is absent")
    artifacts = action_row.get("input_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "action_assembly", "bounded_row_selection", "bounded_storage_estimate",
    }:
        raise PermissionError("validation proxy action artifact set differs")
    _, descriptor = _safe_reference(
        artifacts["action_assembly"], name="validation action assembly",
    )
    descriptor_sha256 = validate_content_hash(
        descriptor,
        expected_contract=(
            "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY/v1"
        ),
        expected_schema_version=1,
    )
    _, bounded_selection = _safe_reference(
        artifacts["bounded_row_selection"], name="bounded row selection",
    )
    from .selective_assignment import validate_row_selection

    validate_row_selection(
        bounded_selection,
        split_manifest_sha256=bounded_selection.get("split_manifest_sha256"),
    )
    if (
        bounded_selection.get("seed") != 1337
        or set(bounded_selection.get("roles", {})) != {"train", "validation"}
        or bounded_selection["roles"]["validation"].get("rows") != _ROW_LIMIT
        or bounded_selection.get("access_lock_sha256") != {}
    ):
        raise PermissionError("validation proxy bounded selection differs")

    source_row = resolve_runtime_row(
        runtime, spec=planning, task_key=_SOURCE_TASK_KEY, array_index=None,
    )
    source_row_sha256 = canonical_sha256(dict(source_row))
    if (
        descriptor.get("action_id") != _ACTION_ID
        or descriptor.get("source_task_key") != _SOURCE_TASK_KEY
        or descriptor.get("source_kind") != "parent_import"
        or descriptor.get("source_runtime_row_sha256") != source_row_sha256
        or descriptor.get("bounded_row_selection_sha256")
        != bounded_selection.get("content_hash")
        or descriptor.get("validation_rows") != _ROW_LIMIT
        or descriptor.get("train_rows") != 0
        or descriptor.get("final_rows") != 0
        or descriptor.get("production_bridge_available") is not True
        or descriptor.get("final_role_access_authorized") is not False
        or descriptor.get("shared_final_authorized") is not False
    ):
        raise PermissionError("validation proxy action assembly differs")

    workspace = Path(str(descriptor["workspace"]))
    if not workspace.is_absolute() or workspace.is_symlink():
        raise PermissionError("validation proxy action workspace is unsafe")
    if workspace.parent.name != "workspaces" or workspace.name != _ACTION_ID:
        raise PermissionError("validation proxy action workspace route differs")
    nonfinal_root = workspace.parent.parent.resolve()
    semantic_path = nonfinal_root / "validation_proxy" / "result.json"
    if _forbidden_route(semantic_path):
        raise PermissionError("validation proxy semantic route names final work")
    if semantic_path.exists() or (semantic_path.parent / "access").exists():
        raise FileExistsError(
            "validation proxy semantic/access route already exists"
        )

    registered = _materialize_registered_inputs(source_row, planning=planning)
    parent_campaign = _json_input(registered, "parent_campaign_spec")
    source_manifest = _json_input(registered, "parent_source_manifest")
    split = _json_input(registered, "split_manifest")
    matcher_resources = _json_input(registered, "matcher_resources")
    parent_import = _json_input(registered, "parent_import")
    assignment_path = Path(registered["validation_assignment_manifest"]["path"])
    assignment_manifest = _json_input(registered, "validation_assignment_manifest")
    report_root = Path(registered["parent_reports"]["path"])
    if not report_root.is_dir() or report_root.is_symlink():
        raise PermissionError("validation proxy registered parent-report root is unsafe")

    from .hcwdl_campaign import validate_campaign_spec as validate_parent_campaign
    from .hcwdl_representation_locks import validate_parent_import
    from .splits import validate_split_manifest

    parent_campaign_sha256 = validate_parent_campaign(parent_campaign)
    source_manifest_sha256, inventory = _source_inventory(source_manifest)
    split_sha256 = validate_split_manifest(
        split,
        source_manifest_sha256=source_manifest_sha256,
        expected_inventory=inventory,
    )
    parent_import_sha256 = validate_parent_import(parent_import)
    if any((
        parent_campaign.get("source_manifest_sha256") != source_manifest_sha256,
        parent_campaign.get("split_manifest_sha256") != split_sha256,
        parent_import.get("parents", {}).get("parent_campaign_spec")
        != parent_campaign_sha256,
        parent_import.get("parents", {}).get("source_manifest")
        != source_manifest_sha256,
        parent_import.get("parents", {}).get("split_manifest") != split_sha256,
        bounded_selection.get("split_manifest_sha256") != split_sha256,
    )):
        raise PermissionError("validation proxy parent/split lineage differs")
    data_root = _validate_data_root(parent_campaign.get("data_root"))

    from .highcov_resources import RESOURCE_CONTRACT, resource_validation_report

    matcher_sha256 = validate_content_hash(
        matcher_resources, expected_contract=RESOURCE_CONTRACT,
        expected_schema_version=1,
    )
    if matcher_resources != resource_validation_report():
        raise ValueError("validation proxy matcher resources are not canonical")
    from .highcov_cache import validate_assignment_manifest

    expected_assignment_parents = {
        "split_manifest_sha256": split_sha256,
        "row_selection_sha256": parent_import["parents"]["row_selection"],
        "matcher_resources_sha256": matcher_sha256,
    }
    validate_assignment_manifest(
        assignment_path,
        expected_role="validation",
        expected_mapped_jets=int(parent_campaign["role_counts"]["validation"]),
        expected_parents=expected_assignment_parents,
        require_sub10pct_dustbins=True,
    )
    assignment_sha256 = require_sha256(
        assignment_manifest.get("content_hash"), name="validation assignment manifest",
    )

    models, bindings, reports, model_source_lineage = _parent_models(
        parent_import, report_root=report_root.resolve(), device="cuda",
    )
    registered_hashes = {
        name: reference["sha256"] for name, reference in registered.items()
    }
    input_lineage = build_validation_proxy_input_lineage(
        authority_sha256=authority_sha256,
        action_inputs_sha256=action_inputs_sha256,
        source_runtime_row_sha256=source_row_sha256,
        action_assembly_sha256=descriptor_sha256,
        bounded_row_selection_sha256=bounded_selection["content_hash"],
        parent_campaign_spec_sha256=parent_campaign_sha256,
        source_manifest_sha256=source_manifest_sha256,
        split_manifest_sha256=split_sha256,
        matcher_resources_sha256=matcher_sha256,
        validation_assignment_manifest_sha256=assignment_sha256,
        parent_import_sha256=parent_import_sha256,
        registered_input_bytes_sha256=registered_hashes,
        model_report_sha256=reports,
        model_source_lineage=model_source_lineage,
        live_worker_runtime=live_worker_runtime,
    )
    return _Inputs(
        authority=authority, authority_sha256=authority_sha256,
        action_inputs=action_inputs, source_row=source_row,
        descriptor=descriptor, bounded_selection=bounded_selection,
        planning=planning, runtime=runtime, registered=registered,
        parent_campaign=parent_campaign, source_manifest=source_manifest,
        split=split, matcher_resources=matcher_resources,
        assignment_manifest_path=assignment_path,
        assignment_manifest=assignment_manifest, parent_import=parent_import,
        parent_report_root=report_root.resolve(), data_root=data_root,
        live_worker_runtime=live_worker_runtime, input_lineage=input_lineage,
        model_bindings=bindings, models=models, semantic_path=semantic_path,
        workspace=workspace,
    )
