"""Exploratory final-test comparison for the frozen PMARD model inventory.

This module deliberately does not reuse the confirmatory PMARD finalist
selector.  It records the user's decision to consume the pilot final-test role
as an exploratory comparison set, freezes every candidate before access, and
publishes metrics without durable per-jet predictions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
import re

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, canonical_sha256, identity_key_array, load_json,
    require_sha256, sha256_file, validate_content_hash, with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import validate_source_snapshot_payload
from .campaign import PMARD_PILOT_ROWS, PMARD_SITE
from .engine import validate_pmard_training_report
from .evaluation import classification_metrics
from .inference import assert_hlt_only_signature
from .kd_followup import (
    KD_FOLLOWUP_REPORT_CONTRACT, KD_FOLLOWUP_REPORT_VERSION,
    validate_kd_followup_spec,
)
from .kd_sweep import (
    validate_t100_sweep_report, validate_t100_sweep_spec,
)
from .selective_assignment import (
    RowSelection, build_row_selection, validate_row_selection,
)

EXPLORATORY_TEST_SPEC_CONTRACT = (
    "hlt_classification_pmard_exploratory_test_spec_v2"
)
EXPLORATORY_TEST_SPEC_VERSION = 2
EXPLORATORY_FINALIST_LOCK_CONTRACT = (
    "hlt_classification_pmard_exploratory_finalist_lock_v2"
)
EXPLORATORY_EXECUTION_LOCK_CONTRACT = (
    "hlt_classification_pmard_exploratory_execution_lock_v2"
)
EXPLORATORY_LOCK_VERSION = 2
EXPLORATORY_EVALUATION_CONTRACT = (
    "hlt_classification_pmard_exploratory_test_evaluation_v2"
)
EXPLORATORY_EVALUATION_VERSION = 2
EXPLORATORY_REPORT_CONTRACT = (
    "hlt_classification_pmard_exploratory_test_report_v2"
)
EXPLORATORY_REPORT_VERSION = 2
EXPLORATORY_LEDGER_CONTRACT = (
    "hlt_classification_pmard_exploratory_test_ledger_v2"
)
EXPLORATORY_LEDGER_VERSION = 2

EXPLORATORY_TEST_TASKS = (
    "authorize", "row_selection", "evaluation", "aggregate",
)
EXPLORATORY_TEST_ROWS = PMARD_PILOT_ROWS["final_test"]
EXPECTED_SWEEP_MODELS = 36
MIN_FOLLOWUP_MODELS = 21
MAX_FOLLOWUP_MODELS = 28
TEST_SEMANTICS = "exploratory_all_registered_models_v2"

PRIMARY_METRICS = (
    "cross_entropy", "accuracy", "balanced_accuracy", "always_qcd_accuracy",
    "macro_ovr_auc", "macro_mean_log_qcd_rejection_at_50pct_signal",
    "multiclass_brier", "top_label_ece_15_bin",
)
REQUIRED_VALIDATION_METRICS = (
    "cross_entropy", "accuracy", "macro_ovr_auc",
    "macro_mean_log_qcd_rejection_at_50pct_signal",
    "top_label_ece_15_bin",
)


def _versioned_reference(path: Path) -> dict[str, str]:
    payload = load_json(path)
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"artifact has no versioned identity: {path}")
    digest = validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )
    return {"path": str(path.resolve()), "content_hash": digest}


def _load_reference(
    spec: Mapping[str, object], name: str,
) -> tuple[Path, dict[str, object]]:
    reference = spec.get("artifacts", {}).get(name)
    if not isinstance(reference, Mapping):
        raise ValueError(f"exploratory specification lacks artifact {name!r}")
    path = Path(str(reference.get("path")))
    payload = load_json(path)
    contract = payload.get("contract")
    version = payload.get("schema_version")
    if not isinstance(contract, str) or not isinstance(version, int):
        raise ValueError(f"exploratory artifact {name!r} is unversioned")
    digest = validate_content_hash(
        payload, expected_contract=contract, expected_schema_version=version,
    )
    if digest != reference.get("content_hash"):
        raise ValueError(f"exploratory artifact {name!r} content hash differs")
    return path, payload


def _validate_embedded_references(owner: Mapping[str, object]) -> None:
    """Validate archived references by their frozen contracts and hashes.

    Completed supplemental studies intentionally remain readable after the
    live campaign registry evolves.  Recomputing an archived campaign's
    scientific identity with current campaign constants would be incorrect;
    its immutable bytes and the child study's recorded hash are authoritative.
    """

    references = owner.get("artifacts")
    if not isinstance(references, Mapping) or not references:
        raise ValueError("archived study artifact inventory is absent")
    for name, reference in references.items():
        if not isinstance(reference, Mapping):
            raise ValueError(f"archived reference {name!r} is invalid")
        path = Path(str(reference.get("path")))
        payload = load_json(path)
        contract = payload.get("contract")
        version = payload.get("schema_version")
        if not isinstance(contract, str) or not isinstance(version, int):
            raise ValueError(f"archived reference {name!r} is unversioned")
        digest = validate_content_hash(
            payload, expected_contract=contract, expected_schema_version=version,
        )
        if digest != reference.get("content_hash"):
            raise ValueError(f"archived reference {name!r} content hash differs")


def _validate_archived_studies(
    sweep_spec: Mapping[str, object], sweep_report: Mapping[str, object],
    followup_spec: Mapping[str, object], followup_report: Mapping[str, object],
) -> None:
    validate_t100_sweep_spec(sweep_spec)
    _validate_embedded_references(sweep_spec)
    validate_t100_sweep_report(sweep_spec, sweep_report)
    validate_kd_followup_spec(followup_spec)
    _validate_embedded_references(followup_spec)
    if (
        followup_spec["artifacts"]["parent_sweep_spec"]["content_hash"]
        != sweep_spec["content_hash"]
        or followup_spec["artifacts"]["parent_sweep_report"]["content_hash"]
        != sweep_report["content_hash"]
    ):
        raise ValueError("follow-up is not a child of the supplied T100 sweep")
    _validate_followup_report(followup_spec, followup_report)


def _compact_validation(metrics: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in REQUIRED_VALIDATION_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"training validation metrics are incomplete: {missing}")
    return {name: metrics[name] for name in PRIMARY_METRICS if name in metrics}


def _scientific_axes(candidate: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "index", "experiment_id", "training_report_path",
        "training_report_sha256", "validation", "delta_vs_k1",
        "delta_vs_prior_k2_alpha1", "delta_vs_ce_only",
        "delta_vs_hlt_self_kd",
    }
    return {str(key): value for key, value in candidate.items() if key not in excluded}


def _registry_row(
    *, global_index: int, source_study: str, source_index: int,
    candidate: Mapping[str, object], expected_report_path: Path,
) -> dict[str, object]:
    experiment_id = candidate.get("experiment_id")
    if not isinstance(experiment_id, str) or re.fullmatch(
        r"[A-Za-z0-9_.-]+", experiment_id,
    ) is None:
        raise ValueError("exploratory experiment identity is not path-safe")
    report_path = Path(str(candidate.get("training_report_path"))).resolve()
    if report_path != expected_report_path.resolve():
        raise ValueError("exploratory candidate report path differs from its study")
    report = load_json(report_path)
    report_hash = validate_pmard_training_report(report)
    if (
        report_hash != candidate.get("training_report_sha256")
        or report.get("experiment_id") != experiment_id
    ):
        raise ValueError("exploratory candidate training report differs")
    config = report.get("config", {})
    if (
        not isinstance(config, Mapping)
        or config.get("model_input", "hlt") != "hlt"
        or config.get("representation_arm", "R0") != "R0"
    ):
        raise PermissionError("exploratory inventory contains a non-HLT R0 model")
    checkpoint_name = report.get("selected_checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("exploratory selected checkpoint name is invalid")
    checkpoint_path = report_path.parent / checkpoint_name
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != report.get("selected_checkpoint_sha256"):
        raise ValueError("exploratory selected checkpoint bytes differ")
    return {
        "index": global_index,
        "evaluation_id": f"{source_study}__{experiment_id}",
        "source_study": source_study,
        "source_index": source_index,
        "experiment_id": experiment_id,
        "training_report_path": str(report_path),
        "training_report_sha256": report_hash,
        "selected_checkpoint_path": str(checkpoint_path.resolve()),
        "selected_checkpoint_sha256": checkpoint_hash,
        "validation": _compact_validation(report["validation"]),
        "scientific_axes": _scientific_axes(candidate),
    }


def _build_registry(
    sweep_spec: Mapping[str, object], sweep_report: Mapping[str, object],
    followup_spec: Mapping[str, object], followup_report: Mapping[str, object],
) -> list[dict[str, object]]:
    followup_count = len(followup_spec.get("registry", ()))
    if not MIN_FOLLOWUP_MODELS <= followup_count <= MAX_FOLLOWUP_MODELS:
        raise ValueError("follow-up distinct-model count is outside its registered bounds")
    sources = (
        (
            "t100_sweep", sweep_report.get("candidates"), EXPECTED_SWEEP_MODELS,
            Path(str(sweep_spec["output_root"])),
        ),
        (
            "kd_followup", followup_report.get("candidates"), followup_count,
            Path(str(followup_spec["output_root"])),
        ),
    )
    registry: list[dict[str, object]] = []
    for source_study, candidates, expected, study_root in sources:
        if not isinstance(candidates, list) or len(candidates) != expected:
            raise ValueError(
                f"{source_study} must contribute exactly {expected} models"
            )
        ordered = sorted(candidates, key=lambda row: int(row["index"]))
        if [int(row["index"]) for row in ordered] != list(range(expected)):
            raise ValueError(f"{source_study} candidate indices differ")
        for candidate in ordered:
            registry.append(_registry_row(
                global_index=len(registry), source_study=source_study,
                source_index=int(candidate["index"]), candidate=candidate,
                expected_report_path=(
                    study_root / "training" / str(candidate["experiment_id"])
                    / "training_report.json"
                ),
            ))
    if len(registry) != EXPECTED_SWEEP_MODELS + followup_count:
        raise RuntimeError("exploratory registry does not contain every distinct model")
    if len({row["evaluation_id"] for row in registry}) != len(registry):
        raise ValueError("exploratory evaluation identities collide")
    return registry


def _validate_followup_report(
    followup_spec: Mapping[str, object], report: Mapping[str, object],
) -> str:
    expected_count = len(followup_spec.get("registry", ()))
    if not MIN_FOLLOWUP_MODELS <= expected_count <= MAX_FOLLOWUP_MODELS:
        raise ValueError("follow-up specification model count is invalid")
    digest = validate_content_hash(
        report, expected_contract=KD_FOLLOWUP_REPORT_CONTRACT,
        expected_schema_version=KD_FOLLOWUP_REPORT_VERSION,
    )
    expected_lineage = {
        "followup_spec_sha256": followup_spec["content_hash"],
        "parent_sweep_report_sha256": (
            followup_spec["artifacts"]["parent_sweep_report"]["content_hash"]
        ),
        "teacher_target_manifest_sha256": (
            followup_spec["artifacts"]["teacher_target_manifest"]["content_hash"]
        ),
        "candidate_count": expected_count, "final_test_access": False,
    }
    differences = [
        name for name, value in expected_lineage.items()
        if report.get(name) != value
    ]
    if differences:
        raise ValueError(f"follow-up aggregate lineage differs: {differences}")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError("follow-up aggregate candidate inventory differs")
    registered = {
        str(row["experiment_id"]): row for row in followup_spec["registry"]
    }
    if {row.get("experiment_id") for row in candidates} != set(registered):
        raise ValueError("follow-up aggregate experiment inventory differs")
    for candidate in candidates:
        expected = registered[str(candidate["experiment_id"])]
        if any(candidate.get(name) != value for name, value in expected.items()):
            raise ValueError("follow-up aggregate recipe differs from registry")
        require_sha256(
            candidate.get("training_report_sha256"),
            name="candidate.training_report_sha256",
        )
        if not isinstance(candidate.get("validation"), Mapping):
            raise ValueError("follow-up aggregate candidate lacks validation metrics")
    return digest


def create_exploratory_test_spec(
    *, parent_sweep_root: str | Path, followup_root: str | Path,
    output_root: str | Path, source_snapshot: Mapping[str, object],
    project_dir: str | Path | None = None,
) -> dict[str, object]:
    validate_source_snapshot_payload(source_snapshot)
    if source_snapshot.get("worktree_clean") is not True:
        raise ValueError("exploratory test requires a clean source snapshot")
    sweep_root = Path(parent_sweep_root).resolve()
    follow_root = Path(followup_root).resolve()
    destination = Path(output_root).resolve()
    if len({sweep_root, follow_root, destination}) != 3:
        raise ValueError("exploratory output and parent studies must be distinct")

    sweep_spec_path = sweep_root / "sweep_spec.json"
    sweep_report_path = sweep_root / "aggregate_report.json"
    follow_spec_path = follow_root / "followup_spec.json"
    follow_report_path = follow_root / "aggregate_report.json"
    sweep_spec = load_json(sweep_spec_path)
    follow_spec = load_json(follow_spec_path)
    sweep_report = load_json(sweep_report_path)
    follow_report = load_json(follow_report_path)
    _validate_archived_studies(
        sweep_spec, sweep_report, follow_spec, follow_report,
    )

    registry = _build_registry(
        sweep_spec, sweep_report, follow_spec, follow_report,
    )
    candidate_count = len(registry)
    split_reference = dict(sweep_spec["artifacts"]["split_manifest"])
    artifacts = {
        "parent_sweep_spec": _versioned_reference(sweep_spec_path),
        "parent_sweep_report": _versioned_reference(sweep_report_path),
        "followup_spec": _versioned_reference(follow_spec_path),
        "followup_report": _versioned_reference(follow_report_path),
        "split_manifest": split_reference,
    }
    site = dict(PMARD_SITE)
    if project_dir is not None:
        site["project_dir"] = str(Path(project_dir).resolve())
    identity = canonical_sha256({
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "artifacts": artifacts, "registry": registry, "site": site,
        "candidate_count": candidate_count, "rows": EXPLORATORY_TEST_ROWS,
        "semantics": TEST_SEMANTICS,
    })
    return with_content_hash({
        "contract": EXPLORATORY_TEST_SPEC_CONTRACT,
        "schema_version": EXPLORATORY_TEST_SPEC_VERSION,
        "study_id": f"pmard_exploratory_test_{identity[:16]}",
        "source_snapshot": dict(source_snapshot),
        "parent_sweep_root": str(sweep_root),
        "followup_root": str(follow_root),
        "output_root": str(destination),
        "site": site, "artifacts": artifacts, "registry": registry,
        "candidate_count": candidate_count,
        "tasks": list(EXPLORATORY_TEST_TASKS),
        "role": "final_test", "rows": EXPLORATORY_TEST_ROWS,
        "selection_seed": 1337,
        "test_role_semantics": TEST_SEMANTICS,
        "model_inventory_frozen_before_test_access": True,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
        "posthoc_test_ranking_is_descriptive_only": True,
        "predictions_published": False,
    })


def validate_exploratory_test_spec(spec: Mapping[str, object]) -> str:
    digest = validate_content_hash(
        spec, expected_contract=EXPLORATORY_TEST_SPEC_CONTRACT,
        expected_schema_version=EXPLORATORY_TEST_SPEC_VERSION,
    )
    validate_source_snapshot_payload(spec.get("source_snapshot", {}))
    if spec["source_snapshot"].get("worktree_clean") is not True:
        raise ValueError("exploratory source snapshot is not clean")
    site = spec.get("site")
    if not isinstance(site, Mapping) or set(site) != set(PMARD_SITE):
        raise ValueError("exploratory site inventory differs")
    for name, value in PMARD_SITE.items():
        if name != "project_dir" and site.get(name) != value:
            raise ValueError("exploratory site differs")
    project = str(site.get("project_dir"))
    if not (Path(project).is_absolute() or PurePosixPath(project).is_absolute()):
        raise ValueError("exploratory project directory must be absolute")
    references = spec.get("artifacts")
    expected_references = {
        "parent_sweep_spec", "parent_sweep_report", "followup_spec",
        "followup_report", "split_manifest",
    }
    if not isinstance(references, Mapping) or set(references) != expected_references:
        raise ValueError("exploratory artifact inventory differs")
    for name, reference in references.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ValueError(f"exploratory reference {name!r} is invalid")
        require_sha256(reference.get("content_hash"), name=f"artifacts[{name}]")
    registry = spec.get("registry")
    candidate_count = spec.get("candidate_count")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or not EXPECTED_SWEEP_MODELS + MIN_FOLLOWUP_MODELS
        <= candidate_count
        <= EXPECTED_SWEEP_MODELS + MAX_FOLLOWUP_MODELS
        or not isinstance(registry, list)
        or len(registry) != candidate_count
    ):
        raise ValueError("exploratory exact distinct-model count differs")
    if [row.get("index") for row in registry] != list(range(candidate_count)):
        raise ValueError("exploratory registry indices differ")
    if len({row.get("evaluation_id") for row in registry}) != candidate_count:
        raise ValueError("exploratory registry identities collide")
    followup_count = candidate_count - EXPECTED_SWEEP_MODELS
    expected_source_indices = {
        "t100_sweep": list(range(EXPECTED_SWEEP_MODELS)),
        "kd_followup": list(range(followup_count)),
    }
    observed_source_indices = {name: [] for name in expected_source_indices}
    for row in registry:
        if row.get("source_study") not in {"t100_sweep", "kd_followup"}:
            raise ValueError("exploratory registry source differs")
        source = str(row["source_study"])
        observed_source_indices[source].append(row.get("source_index"))
        if row.get("evaluation_id") != f"{source}__{row.get('experiment_id')}":
            raise ValueError("exploratory evaluation identity differs")
        for name in ("training_report_path", "selected_checkpoint_path"):
            path = str(row.get(name, ""))
            if not (Path(path).is_absolute() or PurePosixPath(path).is_absolute()):
                raise ValueError(f"exploratory registry {name} must be absolute")
        if not isinstance(row.get("validation"), Mapping) or not isinstance(
            row.get("scientific_axes"), Mapping,
        ):
            raise ValueError("exploratory registry metrics or axes differ")
        for name in ("training_report_sha256", "selected_checkpoint_sha256"):
            require_sha256(row.get(name), name=f"registry.{name}")
    if observed_source_indices != expected_source_indices:
        raise ValueError("exploratory source-study inventory differs")
    frozen = {
        "role": "final_test", "rows": EXPLORATORY_TEST_ROWS,
        "selection_seed": 1337, "test_role_semantics": TEST_SEMANTICS,
        "model_inventory_frozen_before_test_access": True,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
        "posthoc_test_ranking_is_descriptive_only": True,
        "predictions_published": False,
        "tasks": list(EXPLORATORY_TEST_TASKS),
    }
    if any(spec.get(name) != value for name, value in frozen.items()):
        raise ValueError("exploratory comparison semantics differ")
    expected_identity = canonical_sha256({
        "source_snapshot_sha256": spec["source_snapshot"]["source_snapshot_sha256"],
        "artifacts": references, "registry": registry, "site": dict(site),
        "candidate_count": candidate_count, "rows": EXPLORATORY_TEST_ROWS,
        "semantics": TEST_SEMANTICS,
    })
    if spec.get("study_id") != f"pmard_exploratory_test_{expected_identity[:16]}":
        raise ValueError("exploratory study identity differs")
    roots = {
        Path(str(spec.get("parent_sweep_root"))).resolve(),
        Path(str(spec.get("followup_root"))).resolve(),
        Path(str(spec.get("output_root"))).resolve(),
    }
    if len(roots) != 3:
        raise ValueError("exploratory study roots must be distinct")
    return digest


def validate_exploratory_test_inputs(spec: Mapping[str, object]) -> dict[str, object]:
    validate_exploratory_test_spec(spec)
    _, sweep_spec = _load_reference(spec, "parent_sweep_spec")
    _, sweep_report = _load_reference(spec, "parent_sweep_report")
    _, follow_spec = _load_reference(spec, "followup_spec")
    _, follow_report = _load_reference(spec, "followup_report")
    split_path, split_manifest = _load_reference(spec, "split_manifest")
    _validate_archived_studies(
        sweep_spec, sweep_report, follow_spec, follow_report,
    )
    expected_registry = _build_registry(
        sweep_spec, sweep_report, follow_spec, follow_report,
    )
    if spec["registry"] != expected_registry:
        raise ValueError("exploratory frozen model inventory differs")
    return {
        "sweep_spec": sweep_spec, "sweep_report": sweep_report,
        "followup_spec": follow_spec, "followup_report": follow_report,
        "split_path": split_path, "split_manifest": split_manifest,
    }


def exploratory_lock_paths(spec: Mapping[str, object]) -> tuple[Path, Path]:
    root = Path(str(spec["output_root"])) / "locks"
    return root / "01_exploratory_finalist.json", root / "02_exploratory_execution.json"


def authorize_exploratory_test(spec: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    validate_exploratory_test_inputs(spec)
    inventory_hash = canonical_sha256(spec["registry"])
    candidate_count = len(spec["registry"])
    finalist = with_content_hash({
        "contract": EXPLORATORY_FINALIST_LOCK_CONTRACT,
        "schema_version": EXPLORATORY_LOCK_VERSION,
        "exploratory_test_spec_sha256": spec["content_hash"],
        "model_inventory_sha256": inventory_hash,
        "candidate_count": candidate_count,
        "authorized_evaluation_ids": [row["evaluation_id"] for row in spec["registry"]],
        "authorization_basis": "explicit_user_request_evaluate_all_distinct_models",
        "test_role_semantics": TEST_SEMANTICS,
        "confirmatory_claim_forbidden": True,
    })
    execution = with_content_hash({
        "contract": EXPLORATORY_EXECUTION_LOCK_CONTRACT,
        "schema_version": EXPLORATORY_LOCK_VERSION,
        "exploratory_test_spec_sha256": spec["content_hash"],
        "exploratory_finalist_lock_sha256": finalist["content_hash"],
        "model_inventory_sha256": inventory_hash,
        "split_manifest_sha256": spec["artifacts"]["split_manifest"]["content_hash"],
        "role": "final_test", "rows": EXPLORATORY_TEST_ROWS,
        "model_input": "hlt", "metrics_only": True,
        "predictions_published": False,
        "test_role_semantics": TEST_SEMANTICS,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
    })
    finalist_path, execution_path = exploratory_lock_paths(spec)
    write_immutable_json(finalist_path, finalist)
    write_immutable_json(execution_path, execution)
    return finalist, execution


def validate_exploratory_locks(
    spec: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    validate_exploratory_test_spec(spec)
    finalist_path, execution_path = exploratory_lock_paths(spec)
    finalist = load_json(finalist_path)
    finalist_hash = validate_content_hash(
        finalist, expected_contract=EXPLORATORY_FINALIST_LOCK_CONTRACT,
        expected_schema_version=EXPLORATORY_LOCK_VERSION,
    )
    execution = load_json(execution_path)
    validate_content_hash(
        execution, expected_contract=EXPLORATORY_EXECUTION_LOCK_CONTRACT,
        expected_schema_version=EXPLORATORY_LOCK_VERSION,
    )
    inventory_hash = canonical_sha256(spec["registry"])
    if (
        finalist.get("exploratory_test_spec_sha256") != spec["content_hash"]
        or finalist.get("model_inventory_sha256") != inventory_hash
        or finalist.get("candidate_count") != len(spec["registry"])
        or finalist.get("authorized_evaluation_ids")
        != [row["evaluation_id"] for row in spec["registry"]]
        or finalist.get("authorization_basis")
        != "explicit_user_request_evaluate_all_distinct_models"
        or finalist.get("test_role_semantics") != TEST_SEMANTICS
        or finalist.get("confirmatory_claim_forbidden") is not True
    ):
        raise PermissionError("exploratory finalist authorization differs")
    expected_execution = {
        "exploratory_test_spec_sha256": spec["content_hash"],
        "exploratory_finalist_lock_sha256": finalist_hash,
        "model_inventory_sha256": inventory_hash,
        "split_manifest_sha256": spec["artifacts"]["split_manifest"]["content_hash"],
        "role": "final_test", "rows": EXPLORATORY_TEST_ROWS,
        "model_input": "hlt", "metrics_only": True,
        "predictions_published": False,
        "test_role_semantics": TEST_SEMANTICS,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
    }
    if any(execution.get(name) != value for name, value in expected_execution.items()):
        raise PermissionError("exploratory execution authorization differs")
    return finalist, execution


def exploratory_row_selection_path(spec: Mapping[str, object]) -> Path:
    return Path(str(spec["output_root"])) / "data/exploratory_test_row_selection.json"


def build_exploratory_row_selection(spec: Mapping[str, object]) -> dict[str, object]:
    inputs = validate_exploratory_test_inputs(spec)
    finalist, execution = validate_exploratory_locks(spec)
    destination = exploratory_row_selection_path(spec)
    if destination.is_file():
        return validate_exploratory_row_selection(spec)
    selection = build_row_selection(
        inputs["split_manifest"], data_root=spec["site"]["data_root"],
        role_budgets={"final_test": EXPLORATORY_TEST_ROWS},
        seed=int(spec["selection_seed"]),
        completed_locks=("finalist", "execution"),
        access_lock_sha256={
            "finalist": finalist["content_hash"],
            "execution": execution["content_hash"],
        },
    )
    validate_exploratory_row_selection(spec, selection=selection)
    write_immutable_json(destination, selection)
    return selection


def validate_exploratory_row_selection(
    spec: Mapping[str, object], *, selection: Mapping[str, object] | None = None,
) -> dict[str, object]:
    finalist, execution = validate_exploratory_locks(spec)
    payload = load_json(exploratory_row_selection_path(spec)) if selection is None else dict(selection)
    validate_row_selection(
        payload, split_manifest_sha256=spec["artifacts"]["split_manifest"]["content_hash"],
    )
    role = payload.get("roles", {}).get("final_test", {})
    if (
        set(payload.get("roles", {})) != {"final_test"}
        or role.get("rows") != EXPLORATORY_TEST_ROWS
        or role.get("all_rows") is not False
        or payload.get("access_lock_sha256") != {
            "execution": execution["content_hash"],
            "finalist": finalist["content_hash"],
        }
    ):
        raise PermissionError("exploratory final-test row selection differs")
    return payload


def metrics_only_inference(
    model, batches: Iterable[Mapping[str, object]], *, device: str = "cuda",
) -> tuple[dict[str, object], str, int]:
    """Evaluate an HLT-only model without publishing row-level predictions."""

    import torch
    assert_hlt_only_signature(model)
    target = torch.device(device)
    model.to(target).eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    identities: list[str] = []
    with torch.inference_mode():
        for batch in batches:
            view = batch["hlt"]
            output = model(
                torch.as_tensor(view.features, device=target),
                torch.as_tensor(view.vectors, device=target),
                torch.as_tensor(view.mask, device=target),
            )
            if output.ndim != 2 or output.shape[1] != 15 or not torch.isfinite(output).all():
                raise FloatingPointError("exploratory inference logits are invalid")
            logits.append(output.float().cpu().numpy())
            labels.append(np.asarray(batch["labels"], np.int64))
            identities.extend(map(str, batch["identity_keys"]))
    if not logits:
        raise ValueError("exploratory inference stream is empty")
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(labels)
    if len(identities) != len(all_logits) or len(identities) != len(set(identities)):
        raise ValueError("exploratory inference identities are invalid")
    identity_hash = array_sha256("identity_keys", identity_key_array(identities))
    return classification_metrics(all_logits, all_labels), identity_hash, len(identities)


def evaluation_report_path(spec: Mapping[str, object], row: Mapping[str, object]) -> Path:
    return (
        Path(str(spec["output_root"])) / "evaluation"
        / str(row["evaluation_id"]) / "evaluation_report.json"
    )


def evaluate_exploratory_model(
    spec: Mapping[str, object], *, index: int, device: str = "cuda",
    batch_size: int = 512,
) -> dict[str, object]:
    from .dataset import iterate_model_batches
    from .loaders import load_pmard_model, scouting_model_factory_for_report

    validate_exploratory_test_spec(spec)
    if index < 0 or index >= len(spec["registry"]):
        raise IndexError("exploratory model index is outside the frozen inventory")
    finalist, execution = validate_exploratory_locks(spec)
    selection_payload = validate_exploratory_row_selection(spec)
    _, split_manifest = _load_reference(spec, "split_manifest")
    row = spec["registry"][index]
    destination = evaluation_report_path(spec, row)
    if destination.is_file():
        existing = load_json(destination)
        validate_exploratory_evaluation(spec, existing)
        return existing
    report_path = Path(str(row["training_report_path"]))
    training_report = load_json(report_path)
    if (
        validate_pmard_training_report(training_report) != row["training_report_sha256"]
        or sha256_file(Path(str(row["selected_checkpoint_path"])))
        != row["selected_checkpoint_sha256"]
    ):
        raise ValueError("exploratory model artifact changed after registration")
    factory = scouting_model_factory_for_report(training_report)
    model, _ = load_pmard_model(report_path, model_factory=factory, device=device)
    selection = RowSelection(
        selection_payload, role="final_test",
        split_manifest_sha256=spec["artifacts"]["split_manifest"]["content_hash"],
    )
    batches = iterate_model_batches(
        split_manifest, data_root=spec["site"]["data_root"], role="final_test",
        input_mode="hlt", completed_locks=("finalist", "execution"),
        shuffle_within_chunk=False, batch_size=batch_size, row_selection=selection,
    )
    metrics, identity_hash, rows = metrics_only_inference(model, batches, device=device)
    if rows != EXPLORATORY_TEST_ROWS or metrics.get("rows") != EXPLORATORY_TEST_ROWS:
        raise ValueError("exploratory evaluation did not scan exactly 100k test jets")
    report = with_content_hash({
        "contract": EXPLORATORY_EVALUATION_CONTRACT,
        "schema_version": EXPLORATORY_EVALUATION_VERSION,
        "exploratory_test_spec_sha256": spec["content_hash"],
        "exploratory_finalist_lock_sha256": finalist["content_hash"],
        "exploratory_execution_lock_sha256": execution["content_hash"],
        "row_selection_sha256": selection_payload["content_hash"],
        "split_manifest_sha256": spec["artifacts"]["split_manifest"]["content_hash"],
        "registry_index": index, "evaluation_id": row["evaluation_id"],
        "source_study": row["source_study"],
        "experiment_id": row["experiment_id"],
        "training_report_sha256": row["training_report_sha256"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
        "role": "final_test_exploratory", "rows": rows,
        "identity_order_sha256": identity_hash, "metrics": metrics,
        "model_input": "hlt", "deployable_hlt_only": True,
        "predictions_published": False,
        "test_role_semantics": TEST_SEMANTICS,
        "confirmatory_claim_forbidden": True,
    })
    write_immutable_json(destination, report)
    return report


def validate_exploratory_evaluation(
    spec: Mapping[str, object], report: Mapping[str, object],
) -> str:
    digest = validate_content_hash(
        report, expected_contract=EXPLORATORY_EVALUATION_CONTRACT,
        expected_schema_version=EXPLORATORY_EVALUATION_VERSION,
    )
    index = report.get("registry_index")
    if not isinstance(index, int) or index < 0 or index >= len(spec["registry"]):
        raise ValueError("exploratory evaluation registry index differs")
    row = spec["registry"][index]
    finalist, execution = validate_exploratory_locks(spec)
    selection = validate_exploratory_row_selection(spec)
    expected = {
        "exploratory_test_spec_sha256": spec["content_hash"],
        "exploratory_finalist_lock_sha256": finalist["content_hash"],
        "exploratory_execution_lock_sha256": execution["content_hash"],
        "row_selection_sha256": selection["content_hash"],
        "split_manifest_sha256": spec["artifacts"]["split_manifest"]["content_hash"],
        "evaluation_id": row["evaluation_id"], "source_study": row["source_study"],
        "experiment_id": row["experiment_id"],
        "training_report_sha256": row["training_report_sha256"],
        "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
        "role": "final_test_exploratory", "rows": EXPLORATORY_TEST_ROWS,
        "model_input": "hlt", "deployable_hlt_only": True,
        "predictions_published": False, "test_role_semantics": TEST_SEMANTICS,
        "confirmatory_claim_forbidden": True,
    }
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError("exploratory evaluation lineage or semantics differ")
    require_sha256(report.get("identity_order_sha256"), name="identity_order_sha256")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("rows") != EXPLORATORY_TEST_ROWS:
        raise ValueError("exploratory evaluation metrics differ")
    _compact_validation(metrics)
    return digest


def aggregate_exploratory_test(spec: Mapping[str, object]) -> dict[str, object]:
    validate_exploratory_test_inputs(spec)
    finalist, execution = validate_exploratory_locks(spec)
    selection = validate_exploratory_row_selection(spec)
    rows = []
    identity_hash: str | None = None
    for registered in spec["registry"]:
        evaluation = load_json(evaluation_report_path(spec, registered))
        validate_exploratory_evaluation(spec, evaluation)
        if identity_hash is None:
            identity_hash = str(evaluation["identity_order_sha256"])
        elif evaluation["identity_order_sha256"] != identity_hash:
            raise ValueError("exploratory models did not evaluate the same ordered jets")
        test_metrics = evaluation["metrics"]
        validation = registered["validation"]
        rows.append({
            **registered,
            "evaluation_report_path": str(evaluation_report_path(spec, registered)),
            "evaluation_report_sha256": evaluation["content_hash"],
            "exploratory_test": test_metrics,
            "test_minus_validation": {
                name: float(test_metrics[name]) - float(validation[name])
                for name in PRIMARY_METRICS
                if (
                    name in test_metrics and name in validation
                    and test_metrics[name] is not None
                    and validation[name] is not None
                )
            },
        })
    report = with_content_hash({
        "contract": EXPLORATORY_REPORT_CONTRACT,
        "schema_version": EXPLORATORY_REPORT_VERSION,
        "exploratory_test_spec_sha256": spec["content_hash"],
        "exploratory_finalist_lock_sha256": finalist["content_hash"],
        "exploratory_execution_lock_sha256": execution["content_hash"],
        "row_selection_sha256": selection["content_hash"],
        "split_manifest_sha256": spec["artifacts"]["split_manifest"]["content_hash"],
        "candidate_count": len(rows), "rows": EXPLORATORY_TEST_ROWS,
        "identity_order_sha256": identity_hash, "candidates": rows,
        "test_role_semantics": TEST_SEMANTICS,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
        "posthoc_test_ranking_is_descriptive_only": True,
        "predictions_published": False,
    })
    write_immutable_json(Path(str(spec["output_root"])) / "aggregate_report.json", report)
    return report


def _sbatch_base(spec: Mapping[str, object], *, name: str, gpu: bool) -> list[str]:
    site = spec["site"]
    command = [
        "sbatch", "--parsable", f"--account={site['account']}",
        f"--partition={site['partition']}", "--cpus-per-task=8",
        "--mem=96G" if gpu else "--mem=64G",
        "--time=04:00:00" if gpu else "--time=08:00:00",
        f"--job-name={spec['study_id']}_{name}",
    ]
    if gpu:
        command.append(f"--gres={site['gpu_gres']}")
    return command


def submit_exploratory_test(
    spec: Mapping[str, object], *, spec_path: str, dry_run: bool,
    runner: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, object]:
    validate_exploratory_test_inputs(spec)
    if not dry_run and runner is None:
        raise ValueError("executing exploratory submission requires a runner")
    worker = str(
        Path(str(spec["site"]["project_dir"]))
        / "sbatch/run_pmard_exploratory_test.sh"
    )
    jobs: dict[str, str] = {}
    commands: list[list[str]] = []
    for ordinal, task in enumerate(EXPLORATORY_TEST_TASKS, start=1):
        command = _sbatch_base(spec, name=task, gpu=task == "evaluation")
        if task != "authorize":
            predecessor = EXPLORATORY_TEST_TASKS[ordinal - 2]
            command.append(f"--dependency=afterok:{jobs[predecessor]}")
        if task == "evaluation":
            command.append(f"--array=0-{len(spec['registry']) - 1}")
        command.extend((
            "--export=ALL,"
            f"PROJECT_DIR={spec['site']['project_dir']},"
            f"PMARD_EXPLORATORY_TEST_SPEC={Path(spec_path).resolve()},"
            f"PMARD_EXPLORATORY_TEST_TASK={task}",
            worker,
        ))
        commands.append(command)
        if dry_run:
            jobs[task] = str(94_000 + ordinal)
        else:
            output = runner(command).strip().split(";")[0]
            if re.fullmatch(r"[1-9][0-9]*", output) is None:
                raise RuntimeError("exploratory sbatch returned an invalid job ID")
            jobs[task] = output
    return with_content_hash({
        "contract": EXPLORATORY_LEDGER_CONTRACT,
        "schema_version": EXPLORATORY_LEDGER_VERSION,
        "exploratory_test_spec_sha256": spec["content_hash"],
        "dry_run": dry_run, "mutated": not dry_run,
        "jobs": jobs, "commands": commands,
    })


__all__ = [
    "EXPLORATORY_TEST_TASKS", "TEST_SEMANTICS",
    "aggregate_exploratory_test", "authorize_exploratory_test",
    "build_exploratory_row_selection", "create_exploratory_test_spec",
    "evaluate_exploratory_model", "metrics_only_inference",
    "submit_exploratory_test", "validate_exploratory_evaluation",
    "validate_exploratory_locks", "validate_exploratory_row_selection",
    "validate_exploratory_test_inputs", "validate_exploratory_test_spec",
]
