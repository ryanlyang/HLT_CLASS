"""Immutable HCWDL-RKD campaign DAG and typed symbolic command planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256, require_sha256, validate_content_hash, with_content_hash,
)

from .hcwdl_representation_contracts import (
    CAMPAIGN_SPEC_CONTRACT,
    COMMAND_PLAN_CONTRACT,
    LOCAL_SMOKE_REPORT_CONTRACT,
    RECOVERY_SUBMISSION_LEDGER_CONTRACT,
    SUBMISSION_EVENT_CONTRACT,
    SUBMISSION_AUTHORIZATION_CONTRACT,
    SUBMISSION_LEDGER_CONTRACT,
    TIGRIS_ACCEPTANCE_CONTRACT,
    TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
)
from .hcwdl_representation_resources import (
    FIXED_SIZE_INVENTORY_CONTRACT,
    MINIATURE_EVIDENCE_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    SCHEDULER_EVIDENCE_CONTRACT,
    STORAGE_ESTIMATE_CONTRACT,
    TIGRIS_ACCOUNT,
    TIGRIS_PARTITION,
    TIGRIS_SITE,
    load_authenticated_json_reference,
    resource_table,
    validate_measured_profile,
    validate_miniature_evidence,
    validate_scheduler_evidence,
    validate_storage_estimate,
)


def _operational_path(value: str | Path, *, name: str) -> str:
    """Freeze scheduler-visible paths without host-dependent separators.

    Tigris paths are POSIX even when the planning CLI runs on Windows.  A
    Windows absolute path used by a nonauthorizing local fixture is encoded as
    an extended ``//?/C:/...`` spelling, which is both a normalized absolute
    POSIX string and reopenable by ``pathlib`` on Windows.
    """

    text = str(value).replace("\\", "/")
    if re.fullmatch(r"[A-Za-z]:/.*", text):
        text = "//?/" + text
    path = PurePosixPath(text)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"representation {name} must be an absolute normalized path")
    return str(path)


RECOVERY_LEDGER_CONTRACT: Final = RECOVERY_SUBMISSION_LEDGER_CONTRACT
LOCAL_SMOKE_CONTRACT: Final = LOCAL_SMOKE_REPORT_CONTRACT
MODES: Final = ("smoke", "pilot", "production")
STRATEGIES: Final = ("RSET", "RREL")
TRACKS: Final = ("c", "w")
CONTROLS: Final = (
    "RSET_M5c_JET_ONLY_REP",
    "RREL_M5c_NO_REL_REP",
    "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP",
    "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP",
)
DETERMINISTIC_KINDS: Final = ("target_build", "prediction_shard")
PARENT_IMPORT_AUTHORITY_ROUTES: Final = {
    "campaign_spec": "${parent_campaign_spec}",
    "source_manifest": "${parent_source_manifest}",
    "split_manifest": "${split_manifest}",
    "row_selection": "${train_row_selection}",
    "matcher_resources": "${matcher_resources}",
    "train_assignment_manifest": "${assignment_manifest:train}",
    "validation_assignment_manifest": "${parent_assignment_manifest:validation}",
    "train_recomputation_audit": "${parent_assignment_recomputation:train}",
    "validation_recomputation_audit": "${parent_assignment_recomputation:validation}",
    "assignment_lock": "${parent_assignment_lock}",
    "recipe": "${parent_recipe}",
    "recipe_lock": "${parent_recipe_lock}",
    "cache_miniature": "${parent_cache_miniature}",
    "diagnostic_authority": "${parent_diagnostic_authority}",
    "qualification_report": "${parent_qualification_report}",
    "endpoint_qualification_lock": "${parent_endpoint_qualification_lock}",
    "screen_aggregate": "${parent_screen_aggregate}",
    "confirmation_registry_lock": "${parent_confirmation_registry_lock}",
    "confirmation_aggregate": "${parent_confirmation_aggregate}",
    "finalist_lock": "${parent_finalist_registry}",
}
PARENT_QUALIFIER_REPORT_ROUTES: Final = {
    name: f"${{parent_qualifier_report:{name}}}"
    for name in ("T0", "TFS", "THC", "TSOFT", "TSHELL", "TOFF")
}
_TASK_KEY_PATTERN = r"[A-Za-z0-9_.-]+"
DEPENDENCY_TOKEN = re.compile(
    rf"^\$\{{afterok:({_TASK_KEY_PATTERN}(?:,{_TASK_KEY_PATTERN})*)\}}$"
)
SLURM_JOB_ID = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")
EXACT_OUTPUT_ROUTE = re.compile(
    rf"^\$\{{task_output:({_TASK_KEY_PATTERN})(?:\[([0-9]+)\])?:([0-9]+)\}}$"
)
AUTHORIZATION_PHRASE: Final = "AUTHORIZE EXACT HCWDL-RKD SPEC FOR TIGRIS"
REQUIRED_TIGRIS_CHECKS: Final = (
    "installed_weaver_parity",
    "ordinary_cache_miniature",
    "toff_cache_miniature",
    "two_update_full_loss",
    "usr1_exact_resume",
    "validation_only_proxy",
    "production_worker_smoke",
)
# Action-specific result contracts are implemented locally.  They still
# require separately authorized Tigris jobs; an empty set here means only that
# the validator surface exists, not that any evidence has been produced.
UNIMPLEMENTED_TIGRIS_ACTION_PROOFS: Final = frozenset()
REQUIRED_ARTIFACT_PATHS: Final = frozenset({
    "source_manifest", "split_manifest", "parent_import",
    "representation_graph", "representation_recipe", "final_disposition",
    "runtime_binding",
})


@dataclass(frozen=True)
class CampaignTask:
    task_key: str
    kind: str
    dependencies: tuple[str, ...]
    resource_class: str
    array: str | None = None
    graph_node: str | None = None
    logical_bank: str | None = None
    target_purpose: str | None = None
    deterministic_worker: bool = False
    array_registry: str | None = None
    registered_inputs: tuple[str, ...] = ()
    registered_outputs: tuple[str, ...] = ()


def primary_node_ids() -> tuple[str, ...]:
    return tuple(
        f"{strategy}_M{rung}{track}"
        for strategy in STRATEGIES
        for track in TRACKS
        for rung in range(1, 7)
    )


def build_submission_authorization(
    *, mode: str, source_commit: str, command_plan_sha256: str,
    executable_candidate_audit_sha256: str,
    resource_profile_sha256: str, storage_estimate_sha256: str,
    tigris_acceptance_sha256: str, parent_import_sha256: str,
    representation_recipe_sha256: str, disposition_sha256: str,
    authorization_phrase: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("representation authorization mode differs")
    if authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("representation submission authorization phrase differs")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("representation authorization source commit differs")
    return with_content_hash({
        "contract": SUBMISSION_AUTHORIZATION_CONTRACT,
        "schema_version": 1,
        "mode": mode,
        "source_commit": source_commit,
        "command_plan_sha256": require_sha256(command_plan_sha256, name="command plan"),
        "executable_candidate_audit_sha256": require_sha256(
            executable_candidate_audit_sha256,
            name="executable candidate audit",
        ),
        "resource_profile_sha256": require_sha256(resource_profile_sha256, name="resource profile"),
        "storage_estimate_sha256": require_sha256(storage_estimate_sha256, name="storage estimate"),
        "tigris_acceptance_sha256": require_sha256(tigris_acceptance_sha256, name="Tigris acceptance"),
        "parent_import_sha256": require_sha256(parent_import_sha256, name="parent import"),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe",
        ),
        "disposition_sha256": require_sha256(disposition_sha256, name="final disposition"),
        "explicit_user_authorization": True,
    })


def validate_submission_authorization(
    value: Mapping[str, Any], *, mode: str, source_commit: str,
    command_plan_sha256: str, resource_profile_sha256: str,
    executable_candidate_audit_sha256: str,
    storage_estimate_sha256: str, tigris_acceptance_sha256: str,
    parent_import_sha256: str, representation_recipe_sha256: str,
    disposition_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=SUBMISSION_AUTHORIZATION_CONTRACT,
        expected_schema_version=1,
    )
    if set(value) != {
        "contract", "schema_version", "mode", "source_commit",
        "command_plan_sha256", "executable_candidate_audit_sha256",
        "resource_profile_sha256", "storage_estimate_sha256",
        "tigris_acceptance_sha256", "parent_import_sha256",
        "representation_recipe_sha256", "disposition_sha256",
        "explicit_user_authorization", "content_hash",
    }:
        raise PermissionError("representation submission authorization fields differ")
    expected = {
        "mode": mode,
        "source_commit": source_commit,
        "command_plan_sha256": command_plan_sha256,
        "executable_candidate_audit_sha256": executable_candidate_audit_sha256,
        "resource_profile_sha256": resource_profile_sha256,
        "storage_estimate_sha256": storage_estimate_sha256,
        "tigris_acceptance_sha256": tigris_acceptance_sha256,
        "parent_import_sha256": parent_import_sha256,
        "representation_recipe_sha256": representation_recipe_sha256,
        "disposition_sha256": disposition_sha256,
        "explicit_user_authorization": True,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PermissionError("representation submission authorization lineage differs")
    return digest


def validate_tigris_acceptance(
    value: Mapping[str, Any], *, source_commit: str,
    representation_recipe_sha256: str, resource_profile_sha256: str,
    storage_estimate_sha256: str, fixed_size_inventory_sha256: str,
) -> str:
    digest = validate_content_hash(
        value, expected_contract=TIGRIS_ACCEPTANCE_CONTRACT, expected_schema_version=1,
    )
    if set(value) != {
        "contract", "schema_version", "source_commit",
        "representation_recipe_sha256", "resource_profile_sha256",
        "storage_estimate_sha256", "fixed_size_inventory_sha256",
        "evidence_bundle",
        "authorizes_pilot_submission", "content_hash",
    }:
        raise PermissionError("representation Tigris acceptance fields differ")
    expected = {
        "source_commit": source_commit,
        "representation_recipe_sha256": representation_recipe_sha256,
        "resource_profile_sha256": resource_profile_sha256,
        "storage_estimate_sha256": storage_estimate_sha256,
        "fixed_size_inventory_sha256": fixed_size_inventory_sha256,
        "authorizes_pilot_submission": True,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PermissionError("representation Tigris acceptance is absent or incompatible")
    bundle, _ = load_authenticated_json_reference(
        value["evidence_bundle"],
        expected_contract=TIGRIS_EVIDENCE_BUNDLE_CONTRACT,
        name="Tigris evidence bundle",
    )
    if set(bundle) != {
        "contract", "schema_version", "source_commit",
        "representation_recipe_sha256", "resource_profile_sha256",
        "storage_estimate_sha256", "fixed_size_inventory_sha256", "site",
        "account", "partition", "resource_profile", "storage_estimate",
        "fixed_size_inventory", "checks", "content_hash",
    }:
        raise PermissionError("representation Tigris evidence bundle fields differ")
    for key, expected_value in (
        ("source_commit", source_commit),
        ("representation_recipe_sha256", representation_recipe_sha256),
        ("resource_profile_sha256", resource_profile_sha256),
        ("storage_estimate_sha256", storage_estimate_sha256),
        ("fixed_size_inventory_sha256", fixed_size_inventory_sha256),
        ("site", TIGRIS_SITE),
        ("account", TIGRIS_ACCOUNT),
        ("partition", TIGRIS_PARTITION),
    ):
        if bundle.get(key) != expected_value:
            raise PermissionError("representation Tigris evidence bundle lineage differs")

    resource_profile, profile_hash = load_authenticated_json_reference(
        bundle["resource_profile"],
        expected_contract=RESOURCE_PROFILE_CONTRACT,
        name="Tigris resource profile",
    )
    if profile_hash != resource_profile_sha256:
        raise PermissionError("representation Tigris resource profile differs")
    validate_measured_profile(
        resource_profile,
        require_genuine_tigris=True,
        expected_source_commit=source_commit,
    )
    storage_estimate, storage_hash = load_authenticated_json_reference(
        bundle["storage_estimate"],
        expected_contract=STORAGE_ESTIMATE_CONTRACT,
        name="Tigris storage estimate",
    )
    if storage_hash != storage_estimate_sha256:
        raise PermissionError("representation Tigris storage estimate differs")
    _, inventory_hash = load_authenticated_json_reference(
        bundle["fixed_size_inventory"],
        expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
        name="Tigris fixed-size inventory",
    )
    if inventory_hash != fixed_size_inventory_sha256:
        raise PermissionError("representation Tigris fixed-size inventory differs")
    validate_storage_estimate(
        storage_estimate,
        require_measured_fixed_sizes=True,
        fixed_size_inventory=bundle["fixed_size_inventory"],
    )

    checks = bundle.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_TIGRIS_CHECKS):
        raise PermissionError("representation Tigris evidence check registry differs")
    environment = resource_profile["measurement_environment"]
    requests = resource_profile["requests"]
    seen_jobs: set[int] = set()
    seen_result_executions: set[str] = set()
    nonfinal_authority_sha256: str | None = None
    for evidence_kind in REQUIRED_TIGRIS_CHECKS:
        row = checks[evidence_kind]
        if evidence_kind in {
            "two_update_full_loss", "usr1_exact_resume", "validation_only_proxy",
        }:
            if not isinstance(row, Mapping) or set(row) != {"composite_proof"}:
                raise PermissionError("non-final composite Tigris check row differs")
            from .hcwdl_representation_contracts import (
                NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
                TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
                USR1_EXACT_RESUME_PROOF_CONTRACT,
            )
            expected_contract = {
                "two_update_full_loss": TWO_UPDATE_ACCEPTANCE_PROOF_CONTRACT,
                "usr1_exact_resume": USR1_EXACT_RESUME_PROOF_CONTRACT,
                "validation_only_proxy": NONFINAL_ACCEPTANCE_ACTION_RESULT_CONTRACT,
            }[evidence_kind]
            proof, proof_hash = load_authenticated_json_reference(
                row["composite_proof"], expected_contract=expected_contract,
                name=f"{evidence_kind} composite proof",
            )
            if evidence_kind == "two_update_full_loss":
                from .hcwdl_representation_nonfinal_acceptance import (
                    validate_two_update_acceptance_proof,
                )

                validate_two_update_acceptance_proof(proof, require_genuine=True)
                job_ids = {
                    int(item["job_id"])
                    for item in proof["scheduler_evidence"].values()
                }
            elif evidence_kind == "usr1_exact_resume":
                from .hcwdl_representation_nonfinal_acceptance import (
                    validate_usr1_exact_resume_proof_v2,
                )

                validate_usr1_exact_resume_proof_v2(proof, require_genuine=True)
                job_ids = {
                    int(item["job_id"])
                    for item in proof["scheduler_evidence"].values()
                }
            else:
                from .hcwdl_representation_nonfinal_acceptance import (
                    validate_nonfinal_acceptance_action_result,
                )

                validate_nonfinal_acceptance_action_result(
                    proof, expected_action_id="validation_proxy",
                    require_genuine=True,
                )
                job_ids = {int(proof["scheduler_job_id"])}
            if (
                proof.get("source_commit") != source_commit
                or proof.get("representation_recipe_sha256")
                != representation_recipe_sha256
            ):
                raise PermissionError("non-final composite proof lineage differs")
            proof_authority = require_sha256(
                proof.get("authority_sha256"),
                name=f"{evidence_kind} non-final authority",
            )
            if nonfinal_authority_sha256 is None:
                nonfinal_authority_sha256 = proof_authority
            elif proof_authority != nonfinal_authority_sha256:
                raise PermissionError(
                    "Tigris composite proofs bind different non-final authorities"
                )
            if seen_jobs & job_ids or proof_hash in seen_result_executions:
                raise PermissionError(
                    "Tigris evidence reuses a non-final job or composite result"
                )
            seen_jobs.update(job_ids)
            seen_result_executions.add(proof_hash)
            continue
        if not isinstance(row, Mapping) or set(row) != {
            "scheduler_evidence", "miniature_evidence", "action_proof",
        }:
            raise PermissionError("representation Tigris evidence check row differs")
        scheduler, _ = load_authenticated_json_reference(
            row["scheduler_evidence"],
            expected_contract=SCHEDULER_EVIDENCE_CONTRACT,
            name=f"{evidence_kind} scheduler evidence",
        )
        resource_class = str(scheduler.get("resource_class"))
        if resource_class not in requests:
            raise PermissionError("Tigris check names an unknown resource class")
        scheduler = validate_scheduler_evidence(
            scheduler,
            resource_class=resource_class,
            request=requests[resource_class],
            expected_source_commit=source_commit,
            expected_recipe_sha256=representation_recipe_sha256,
            expected_workers=environment["production_workers"],
            require_genuine=True,
        )
        job_id = int(scheduler["job_id"])
        if job_id in seen_jobs:
            raise PermissionError("Tigris evidence reuses one job for multiple checks")
        seen_jobs.add(job_id)
        miniature, _ = load_authenticated_json_reference(
            row["miniature_evidence"],
            expected_contract=MINIATURE_EVIDENCE_CONTRACT,
            name=f"{evidence_kind} miniature evidence",
        )
        validate_miniature_evidence(
            miniature,
            expected_kind=evidence_kind,
            expected_source_commit=source_commit,
            expected_recipe_sha256=representation_recipe_sha256,
            scheduler_evidence=scheduler,
            require_genuine=True,
        )
        from .hcwdl_representation_acceptance_evidence import (
            validate_tigris_action_proof,
        )
        from .hcwdl_representation_contracts import TIGRIS_ACTION_PROOF_CONTRACT

        action, _ = load_authenticated_json_reference(
            row["action_proof"],
            expected_contract=TIGRIS_ACTION_PROOF_CONTRACT,
            name=f"{evidence_kind} action proof",
        )
        validate_tigris_action_proof(
            action, resource_request=requests[resource_class],
            expected_workers=environment["production_workers"],
            require_genuine=True,
        )
        if (
            action.get("evidence_kind") != evidence_kind
            or action.get("source_commit") != source_commit
            or action.get("representation_recipe_sha256")
            != representation_recipe_sha256
            or action.get("scheduler_evidence") != row["scheduler_evidence"]
            or action.get("miniature_evidence") != row["miniature_evidence"]
            or action.get("scheduler_evidence_sha256")
            != scheduler["content_hash"]
            or action.get("miniature_evidence_sha256")
            != miniature["content_hash"]
            or action.get("result_artifact") != miniature.get("result_artifact")
            or action.get("result_sha256") != miniature.get("result_sha256")
            or action.get("result_execution_sha256")
            != miniature.get("result_execution_sha256")
        ):
            raise PermissionError("Tigris action proof lineage differs")
        result_execution = str(action["result_execution_sha256"])
        if result_execution in seen_result_executions:
            raise PermissionError(
                "Tigris evidence reuses one action result for multiple checks"
            )
        seen_result_executions.add(result_execution)
    if nonfinal_authority_sha256 is None:
        raise PermissionError("Tigris acceptance lacks its non-final authority")
    return digest


def validate_source_checkout(repository: str | Path, *, expected_commit: str) -> None:
    """Require the exact clean commit and a locally known pushed remote ref."""

    root = Path(repository).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", expected_commit], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    if head != expected_commit or dirty or "origin/" not in remote:
        raise PermissionError(
            "representation source is not the exact clean locally known pushed commit"
        )


def _add_bank_wave(
    tasks: list[CampaignTask], *, bank: str, prior: str,
    nodes: Sequence[str], controls: Sequence[str] = (), purpose: str = "screen",
) -> str:
    build = f"target_{bank}_{purpose}"
    tasks.append(
        CampaignTask(
            build, "target_build", (prior,), "gpu_target",
            logical_bank=bank, target_purpose=purpose, deterministic_worker=True,
        )
    )
    consumers = []
    for node in nodes:
        key = f"train_{node}" if purpose == "screen" else f"confirm_{node}"
        array = None if purpose == "screen" else "0-4"
        rung_match = re.fullmatch(r"(RSET|RREL)_M([1-6])([cw])", node)
        if rung_match is None:
            raise ValueError(f"invalid registered representation node {node!r}")
        strategy, rung_text, track = rung_match.groups()
        rung = int(rung_text)
        direct_parent = (
            () if rung == 1
            else (f"train_{strategy}_M{rung - 1}{track}",)
        )
        if purpose == "confirmation":
            direct_parent = (f"train_{strategy}_M5{track}",)
        dependencies = tuple(dict.fromkeys((build, *direct_parent)))
        tasks.append(
            CampaignTask(
                key, "train_node" if purpose == "screen" else "confirmation",
                dependencies, "gpu_representation", array=array, graph_node=node,
                logical_bank=bank, target_purpose=purpose,
                array_registry=(
                    None if purpose == "screen"
                    else f"confirmation/registry.json#node_id={node}"
                ),
            )
        )
        consumers.append(key)
    for control in controls:
        key = f"control_{control}"
        tasks.append(
            CampaignTask(
                key, "train_control",
                (build, f"train_{control.split('_M5c_', 1)[0]}_M4c"),
                "gpu_representation", graph_node=control, logical_bank=bank,
                target_purpose=purpose,
            )
        )
        consumers.append(key)
    cleanup = f"cleanup_{bank}_{purpose}"
    tasks.append(
        CampaignTask(
            cleanup, "target_cleanup", tuple(consumers), "cpu_io",
            logical_bank=bank, target_purpose=purpose,
        )
    )
    return cleanup


def build_task_registry(
    *, disposition: str, final_source_partitions: int,
    combined_finalist_count: int,
) -> tuple[CampaignTask, ...]:
    if disposition not in {"combined_confirmatory", "validation_only_parent_claim_consumed"}:
        raise ValueError("representation campaign disposition differs")
    if final_source_partitions <= 0 or combined_finalist_count <= 0:
        raise ValueError("final source/finalist counts must be positive")
    tasks: list[CampaignTask] = [
        CampaignTask("tap_schema", "tap_schema", (), "cpu_small"),
        CampaignTask("surface_parity", "surface_parity", ("tap_schema",), "cpu_small"),
        CampaignTask("architecture_attestation", "architecture_attestation", ("surface_parity",), "cpu_small"),
        CampaignTask("parent_loss_attestation", "parent_loss_attestation", (), "cpu_small"),
        CampaignTask(
            "parent_import", "parent_import",
            ("architecture_attestation", "parent_loss_attestation"), "cpu_small",
        ),
        CampaignTask("control_registry", "control_registry", ("parent_import",), "cpu_small"),
        CampaignTask("kernel_resources", "kernel_resources", ("control_registry",), "cpu_small"),
        CampaignTask("representation_recipe", "representation_recipe", ("kernel_resources",), "cpu_small"),
        CampaignTask("numerical_acceptance", "numerical_acceptance", ("representation_recipe",), "cpu_small"),
        CampaignTask(
            "miniature_D100_build", "target_build", ("numerical_acceptance",), "gpu_target",
            logical_bank="D100", target_purpose="miniature",
            deterministic_worker=True,
        ),
        CampaignTask(
            "miniature_D100_verify_cleanup", "cache_miniature_bank",
            ("miniature_D100_build",), "cpu_io", logical_bank="D100",
            target_purpose="miniature",
        ),
        CampaignTask(
            "miniature_TOFF_build", "target_build",
            ("miniature_D100_verify_cleanup",), "gpu_target",
            logical_bank="TOFF", target_purpose="miniature",
            deterministic_worker=True,
        ),
        CampaignTask(
            "miniature_TOFF_verify_cleanup", "cache_miniature_bank",
            ("miniature_TOFF_build",), "cpu_io", logical_bank="TOFF",
            target_purpose="miniature",
        ),
        CampaignTask(
            "cache_miniature", "cache_miniature",
            ("miniature_D100_verify_cleanup", "miniature_TOFF_verify_cleanup"),
            "cpu_small",
        ),
        CampaignTask("smoke_probe", "smoke_probe", ("cache_miniature",), "gpu_representation"),
        CampaignTask(
            "zero_coefficient_acceptance", "zero_coefficient_acceptance",
            ("smoke_probe",), "gpu_representation",
        ),
        CampaignTask(
            "pretraining_reservation", "reservation",
            ("zero_coefficient_acceptance",), "cpu_small",
        ),
    ]
    prior = "pretraining_reservation"
    for rung, source in enumerate(("D0", "D25", "D50", "D75"), start=1):
        cold_nodes = tuple(f"{strategy}_M{rung}c" for strategy in STRATEGIES)
        prior = _add_bank_wave(tasks, bank=f"{source}c", prior=prior, nodes=cold_nodes)
        warm_nodes = tuple(f"{strategy}_M{rung}w" for strategy in STRATEGIES)
        prior = _add_bank_wave(tasks, bank=f"{source}w", prior=prior, nodes=warm_nodes)
    tasks.append(CampaignTask("shuffle_map", "shuffle_map", (prior,), "cpu_io"))
    m5_nodes = tuple(f"{strategy}_M5{track}" for strategy in STRATEGIES for track in TRACKS)
    prior = _add_bank_wave(
        tasks, bank="D100", prior="shuffle_map", nodes=m5_nodes, controls=CONTROLS,
    )
    m6_nodes = tuple(f"{strategy}_M6{track}" for strategy in STRATEGIES for track in TRACKS)
    prior = _add_bank_wave(tasks, bank="TOFF", prior=prior, nodes=m6_nodes)
    tasks.extend(
        (
            CampaignTask("screen_aggregate", "screen_aggregate", (prior,), "cpu_small"),
            CampaignTask("confirmation_registry", "confirmation_registry", ("screen_aggregate",), "cpu_small"),
        )
    )
    confirmation_cleanup = _add_bank_wave(
        tasks,
        bank="TOFF",
        prior="confirmation_registry",
        nodes=m6_nodes,
        purpose="confirmation",
    )
    tasks.append(
        CampaignTask("confirmation_aggregate", "confirmation_aggregate", (confirmation_cleanup,), "cpu_small")
    )
    if disposition == "combined_confirmatory":
        assignment_array = f"0-{final_source_partitions - 1}"
        prediction_rows = combined_finalist_count * final_source_partitions
        tasks.extend(
            (
                CampaignTask("finalist_lock", "finalist_lock", ("confirmation_aggregate",), "cpu_small"),
                CampaignTask("shared_claim_gate", "shared_final_claim", ("finalist_lock",), "cpu_small"),
                CampaignTask("final_selection", "final_selection", ("shared_claim_gate",), "cpu_io"),
                CampaignTask(
                    "final_assignment_shards", "assignment_shard", ("final_selection",),
                    "cpu_io", array=assignment_array,
                    array_registry="final/task_registry.json#purpose=assignment_shard",
                ),
                CampaignTask("final_assignment_manifest", "assignment_finalize", ("final_assignment_shards",), "cpu_small"),
                CampaignTask("final_data_attestation", "data_attestation", ("final_assignment_manifest",), "cpu_small"),
                CampaignTask("representation_execution_lock", "execution_lock", ("final_data_attestation",), "cpu_small"),
                CampaignTask(
                    "final_prediction_shards", "prediction_shard", ("representation_execution_lock",),
                    "gpu_final_prediction", array=f"0-{prediction_rows - 1}",
                    deterministic_worker=True,
                    array_registry="final/task_registry.json#purpose=prediction_shard",
                ),
                CampaignTask(
                    "final_prediction_manifests", "prediction_finalize", ("final_prediction_shards",),
                    "cpu_small", array=f"0-{combined_finalist_count - 1}",
                    array_registry="final/task_registry.json#purpose=prediction_manifest",
                ),
                CampaignTask("locked_metric_join", "metric_join", ("final_prediction_manifests",), "cpu_small"),
                CampaignTask("final_aggregate", "final_aggregate", ("locked_metric_join",), "cpu_small"),
            )
        )
    else:
        tasks.append(
            CampaignTask("final_aggregate", "validation_only_aggregate", ("confirmation_aggregate",), "cpu_small")
        )
    tasks = [
        replace(row, registered_outputs=_registered_outputs(row))
        for row in tasks
    ]
    task_by_key = {row.task_key: row for row in tasks}
    tasks = [
        replace(
            row,
            registered_inputs=_registered_inputs(
                row,
                task_by_key=task_by_key,
                final_source_partitions=final_source_partitions,
                combined_finalist_count=combined_finalist_count,
            ),
        )
        for row in tasks
    ]
    validate_task_registry(tasks, disposition=disposition)
    return tuple(tasks)


def _registered_outputs(task: CampaignTask) -> tuple[str, ...]:
    if task.kind == "validation_only_aggregate":
        return ("reports/validation_only_aggregate.json",)
    fixed = {
        "tap_schema": ("architecture/tap.json",),
        "surface_parity": ("architecture/surface_parity.json",),
        "architecture_attestation": ("import/architecture_attestation.json",),
        "parent_loss_attestation": ("import/parent_loss_attestation.json",),
        "parent_import": ("import/parent_import.json",),
        "control_registry": ("controls/registry.json",),
        "kernel_resources": ("recipes/kernel_resources/committed/${envelope_id}",),
        "representation_recipe": ("recipes/representation_recipe.json",),
        "numerical_acceptance": ("acceptance/numerical.json",),
        "cache_miniature": ("acceptance/cache_miniature.json",),
        "miniature_D100_verify_cleanup": (
            "acceptance/cache_miniature_D100.json",
            "cleanup/D100/${generation_id}/authorization.json",
            "cleanup/D100/${generation_id}/completion.json",
        ),
        "miniature_TOFF_verify_cleanup": (
            "acceptance/cache_miniature_TOFF.json",
            "cleanup/TOFF/${generation_id}/authorization.json",
            "cleanup/TOFF/${generation_id}/completion.json",
        ),
        # The smoke probe is not a Tigris-acceptance or pilot-authorization lock.
        "smoke_probe": ("acceptance/smoke_probe.json",),
        "zero_coefficient_acceptance": ("controls/zero_coefficient/acceptance.json",),
        "pretraining_reservation": (
            "${population_namespace}/reservation.json",
            "final/assignment/specification.json",
            "final/pretraining_finalist_policy.json",
        ),
        "screen_aggregate": ("reports/screen_aggregate.json",),
        "shuffle_map": ("controls/shuffled_representation/committed/${envelope_id}",),
        "confirmation_registry": ("confirmation/registry.json",),
        "confirmation_aggregate": ("confirmation/aggregate.json",),
        "finalist_lock": ("locks/05_finalists.json",),
        "shared_claim_gate": (
            "${population_namespace}/execution_claim.json", "final/task_registry.json",
        ),
        "final_selection": (
            "final/selection/row_selection.json",
            "final/selection/branch_access.json",
            "final/selection/label_escrow/committed/${envelope_id}",
            "final/capabilities/selection.json",
        ),
        "final_assignment_shards": (
            "final/assignment/shards/${source_partition}/committed/${envelope_id}",
            "final/capabilities/${task_id}.json",
        ),
        "final_assignment_manifest": ("final/assignment/manifest.json", "final/assignment/audit.json"),
        "final_data_attestation": ("locks/06_final_data_attestation.json",),
        "representation_execution_lock": (
            "locks/07_execution.json", "final/prediction_spec.json",
        ),
        "final_prediction_shards": (
            "final/predictions/${finalist_id}/shards/${source_partition}/committed/${envelope_id}",
            "final/capabilities/${task_id}.json",
        ),
        "final_prediction_manifests": ("final/predictions/${finalist_id}/manifest.json",),
        "locked_metric_join": (
            "final/metric_join.json", "final/evaluations",
            "reports/paired_bootstrap",
            "final/capabilities/${task_id}.json",
        ),
        "final_aggregate": ("reports/final_aggregate.json",),
    }
    if task.task_key in fixed:
        return fixed[task.task_key]
    if task.kind == "target_build":
        return (f"targets/{task.logical_bank}/generations/${{generation_id}}",)
    if task.kind == "target_cleanup":
        return (
            f"cleanup/{task.logical_bank}/${{generation_id}}/authorization.json",
            f"cleanup/{task.logical_bank}/${{generation_id}}/completion.json",
        )
    if task.kind in {"train_node", "confirmation"}:
        strategy = str(task.graph_node).split("_", 1)[0]
        prefix = f"training/{strategy}/{task.graph_node}/${{execution_id}}"
        outputs = (
            f"{prefix}/training_report.json",
            f"{prefix}/checkpoint_selection.json",
            f"{prefix}/deployable_extraction.json",
            f"{prefix}",
        )
        if task.kind == "confirmation":
            outputs = (*outputs, "confirmation/runs/${execution_id}.json")
        return outputs
    if task.kind == "train_control":
        prefix = f"training/controls/{task.graph_node}/${{execution_id}}"
        return (
            f"{prefix}/training_report.json",
            f"{prefix}/checkpoint_selection.json",
            f"{prefix}/deployable_extraction.json",
            f"{prefix}",
        )
    raise ValueError(f"no registered output layout for task {task.task_key!r}")


def _producer_output_route(
    task_key: str, output_index: int, *, array_index: int | None = None,
) -> str:
    suffix = "" if array_index is None else f"[{array_index}]"
    return f"${{task_output:{task_key}{suffix}:{output_index}}}"


def _training_model_source_routes(task: CampaignTask) -> tuple[str, ...]:
    node = str(task.graph_node)
    if task.kind == "train_control":
        strategy = node.split("_M5c_", 1)[0]
        return (_producer_output_route(f"train_{strategy}_M4c", 3),)
    match = re.fullmatch(r"(RSET|RREL)_M([1-6])([cw])", node)
    if match is None:
        return ()
    strategy, rung_text, track = match.groups()
    rung = int(rung_text)
    if rung == 1:
        # Warm M1 initializes from the authenticated parent D0 checkpoint,
        # not from the report bundle used only for scalar/report evidence.
        return ("${parent_model_sources}",) if track == "w" else ()
    parent_rung = 5 if task.kind == "confirmation" else rung - 1
    return (_producer_output_route(f"train_{strategy}_M{parent_rung}{track}", 3),)


def adapter_registered_input_requirements(
    task: CampaignTask, *, final_source_partitions: int,
    combined_finalist_count: int,
) -> tuple[str, ...]:
    """Return every artifact route consumed by the fixed production adapter.

    Scalar scientific settings remain in the immutable runtime parameters.
    Every file, directory, JSON artifact, checkpoint, and array-produced
    envelope instead appears here and must be resolved through a registered
    input tag.  Producer routes use the exact output ordinal; array routes
    additionally freeze the producer row index.
    """

    rows: list[str] = []
    kind = task.kind
    if kind == "architecture_attestation":
        rows.extend((
            _producer_output_route("tap_schema", 0),
            _producer_output_route("surface_parity", 0),
            "${parent_reports}", "${parent_model_sources}",
        ))
    elif kind == "parent_loss_attestation":
        rows.extend((
            "${parent_campaign_spec}", "${parent_recipe}", "${parent_reports}",
            "${parent_runtime_sources}",
        ))
    elif kind == "parent_import":
        rows.extend((
            "${prebuilt_parent_import}",
            _producer_output_route("architecture_attestation", 0),
            _producer_output_route("parent_loss_attestation", 0),
            "${parent_reports}", "${parent_model_sources}",
            "${parent_confirmation_reports}",
            *PARENT_IMPORT_AUTHORITY_ROUTES.values(),
            *PARENT_QUALIFIER_REPORT_ROUTES.values(),
        ))
    elif kind == "representation_recipe":
        rows.append("${prebuilt_representation_recipe}")
    elif kind == "target_build":
        assert task.logical_bank is not None and task.target_purpose is not None
        bank = task.logical_bank
        purpose = task.target_purpose
        rows.extend((
            f"${{logical_bank:{bank}}}",
            f"${{target_consumer_registry:{bank}:{purpose}}}",
            f"${{target_forward_spec:{bank}:{purpose}}}",
            "${split_manifest}", "${train_row_selection}",
            f"${{teacher_report:{bank}}}",
            _producer_output_route("architecture_attestation", 0),
            _producer_output_route("kernel_resources", 0),
            "${storage_estimate}", "${resource_profile}",
        ))
        if bank.startswith(("D25", "D50", "D75", "D100")):
            rows.append("${assignment_manifest:train}")
    elif kind == "cache_miniature_bank":
        rows.append(_producer_output_route(
            f"miniature_{task.logical_bank}_build", 0,
        ))
    elif kind == "reservation":
        rows.extend((
            "${shared_final_population}", "${final_disposition}",
            "${parent_final_state}", "${matcher_resources}",
            "${parent_finalist_registry}",
        ))
    elif kind in {"train_node", "train_control", "confirmation"}:
        assert task.logical_bank is not None and task.target_purpose is not None
        rows.extend((
            "${parent_recipe}", "${split_manifest}",
            "${train_validation_row_selection}",
            _producer_output_route(
                f"target_{task.logical_bank}_{task.target_purpose}", 0,
            ),
            _producer_output_route("kernel_resources", 0),
            "${producer_runtime_signature}",
            _producer_output_route("architecture_attestation", 0),
            *_training_model_source_routes(task),
        ))
        if task.kind == "confirmation":
            rows.append(_producer_output_route("confirmation_registry", 0))
        if task.kind == "train_control" and "SHUFFLED" in str(task.graph_node):
            rows.append(_producer_output_route("shuffle_map", 0))
    elif kind == "target_cleanup":
        assert task.logical_bank is not None and task.target_purpose is not None
        rows.append(_producer_output_route(
            f"target_{task.logical_bank}_{task.target_purpose}", 0,
        ))
        for consumer in task.dependencies:
            if task.target_purpose == "confirmation":
                rows.extend(
                    _producer_output_route(consumer, 0, array_index=seed_index)
                    for seed_index in range(5)
                )
            else:
                rows.append(_producer_output_route(consumer, 0))
    elif kind == "screen_aggregate":
        rows.extend((
            "${parent_reports}",
            _producer_output_route("architecture_attestation", 0),
        ))
        rows.extend(
            _producer_output_route(f"train_{node}", 0)
            for node in primary_node_ids()
        )
        rows.extend(
            _producer_output_route(f"control_{control}", 0)
            for control in CONTROLS
        )
    elif kind == "confirmation_registry":
        rows.extend((
            _producer_output_route("screen_aggregate", 0),
            "${logical_bank:TOFF}",
        ))
    elif kind == "confirmation_aggregate":
        rows.append(_producer_output_route("confirmation_registry", 0))
        for node in (
            "RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w",
        ):
            rows.extend(
                _producer_output_route(f"confirm_{node}", 4, array_index=seed_index)
                for seed_index in range(5)
            )
    elif kind == "validation_only_aggregate":
        rows.extend((
            _producer_output_route("screen_aggregate", 0),
            _producer_output_route("confirmation_aggregate", 0),
            "${final_disposition}",
        ))
    elif kind == "finalist_lock":
        rows.extend((
            "${parent_finalist_registry}", "${parent_recipe}",
            "${parent_reports}", "${legacy_cancellation}",
            _producer_output_route("pretraining_reservation", 0),
            _producer_output_route("architecture_attestation", 0),
            _producer_output_route("parent_loss_attestation", 0),
            _producer_output_route("screen_aggregate", 0),
            _producer_output_route("confirmation_registry", 0),
            _producer_output_route("confirmation_aggregate", 0),
        ))
        for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
            # The lock authenticates the scientific report together with the
            # selected checkpoint and deployable extraction it freezes.
            rows.extend(
                _producer_output_route(f"train_{node}", output_index)
                for output_index in (0, 1, 2)
            )
    elif kind == "shared_final_claim":
        rows.extend((
            _producer_output_route("pretraining_reservation", 0),
            _producer_output_route("finalist_lock", 0),
            "${submission_ledger}",
        ))
        for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
            rows.append(_producer_output_route(f"train_{node}", 2))
    elif kind == "final_selection":
        rows.extend((
            "${split_manifest}", "${shared_final_population}",
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
        ))
    elif kind == "assignment_shard":
        rows.extend((
            "${split_manifest}", "${matcher_resources}",
            _producer_output_route("final_selection", 0),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
        ))
    elif kind == "assignment_finalize":
        rows.extend((
            "${split_manifest}", "${matcher_resources}",
            _producer_output_route("final_selection", 0),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
        ))
        rows.extend(
            _producer_output_route(
                "final_assignment_shards", 0, array_index=partition,
            )
            for partition in range(final_source_partitions)
        )
    elif kind == "data_attestation":
        rows.extend((
            "${matcher_resources}",
            _producer_output_route("pretraining_reservation", 1),
            _producer_output_route("final_selection", 0),
            _producer_output_route("final_selection", 2),
            _producer_output_route("final_assignment_manifest", 0),
            _producer_output_route("final_assignment_manifest", 1),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
        ))
    elif kind == "execution_lock":
        rows.extend((
            "${prediction_runtime_signature}",
            _producer_output_route("finalist_lock", 0),
            _producer_output_route("final_data_attestation", 0),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
            _producer_output_route("final_selection", 0),
        ))
    elif kind == "prediction_shard":
        rows.extend((
            "${split_manifest}", "${finalist_models}",
            "${prediction_runtime_signature}",
            _producer_output_route("final_selection", 0),
            _producer_output_route("representation_execution_lock", 1),
            _producer_output_route("finalist_lock", 0),
            _producer_output_route("representation_execution_lock", 0),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
            _producer_output_route("final_assignment_manifest", 0),
        ))
        for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
            rows.append(_producer_output_route(f"train_{node}", 3))
    elif kind == "prediction_finalize":
        rows.extend((
            _producer_output_route("final_selection", 0),
            _producer_output_route("representation_execution_lock", 1),
            _producer_output_route("representation_execution_lock", 0),
        ))
        for node in ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"):
            rows.extend(
                _producer_output_route(f"train_{node}", output_index)
                for output_index in (0, 1, 2)
            )
        rows.extend(
            _producer_output_route(
                "final_prediction_shards", 0, array_index=array_index,
            )
            for array_index in range(
                combined_finalist_count * final_source_partitions
            )
        )
    elif kind == "metric_join":
        rows.extend((
            _producer_output_route("final_selection", 0),
            _producer_output_route("final_selection", 2),
            _producer_output_route("representation_execution_lock", 1),
            _producer_output_route("representation_execution_lock", 0),
            _producer_output_route("finalist_lock", 0),
            _producer_output_route("final_data_attestation", 0),
            _producer_output_route("shared_claim_gate", 0),
            _producer_output_route("shared_claim_gate", 1),
        ))
        rows.extend(
            _producer_output_route(
                "final_prediction_manifests", 0, array_index=finalist_index,
            )
            for finalist_index in range(combined_finalist_count)
        )
        rows.extend(
            _producer_output_route(
                "final_prediction_shards", 0, array_index=array_index,
            )
            for array_index in range(
                combined_finalist_count * final_source_partitions
            )
        )
    elif kind == "final_aggregate":
        rows.extend((
            _producer_output_route("locked_metric_join", 0),
            _producer_output_route("locked_metric_join", 1),
            _producer_output_route("locked_metric_join", 2),
            _producer_output_route("finalist_lock", 0),
            _producer_output_route("representation_execution_lock", 0),
            _producer_output_route("confirmation_aggregate", 0),
        ))
    return tuple(dict.fromkeys(rows))


def _registered_inputs(
    task: CampaignTask, *, task_by_key: Mapping[str, CampaignTask],
    final_source_partitions: int, combined_finalist_count: int,
) -> tuple[str, ...]:
    """Freeze every task's immutable logical inputs in addition to Slurm edges."""

    rows: list[str] = ["${campaign_spec}", "${representation_graph}"]
    pre_parent_import = {
        "tap_schema", "surface_parity", "architecture_attestation",
        "parent_loss_attestation", "parent_import",
    }
    pre_representation_recipe = pre_parent_import | {
        "control_registry", "kernel_resources", "representation_recipe",
    }
    if task.kind not in pre_parent_import:
        rows.extend((
            "${parent_import}", _producer_output_route("parent_import", 0),
        ))
    if task.kind not in pre_representation_recipe:
        rows.extend((
            "${representation_recipe}",
            _producer_output_route("representation_recipe", 0),
        ))
    if task.kind not in {
        "tap_schema", "surface_parity", "architecture_attestation",
        "parent_loss_attestation", "parent_import", "control_registry",
    }:
        rows.extend((
            "${control_registry}", _producer_output_route("control_registry", 0),
        ))
    if task.kind in {
        "reservation", "target_build", "train_node", "train_control",
        "target_cleanup", "screen_aggregate", "confirmation_registry",
        "confirmation", "confirmation_aggregate", "finalist_lock",
        "shared_final_claim", "final_selection", "assignment_shard",
        "assignment_finalize", "data_attestation", "execution_lock",
        "prediction_shard", "prediction_finalize", "metric_join",
        "final_aggregate", "validation_only_aggregate",
    } and task.target_purpose != "miniature":
        rows.append("${zero_coefficient_acceptance}")
    if task.kind == "shuffle_map":
        rows.extend(("${split_manifest}", "${train_row_selection}"))
    if task.kind == "zero_coefficient_acceptance":
        # The adapter currently consumes these compatibility names directly;
        # each resolves to one exact scalar producer output.
        rows.extend((
            "${task_output:architecture_attestation}",
            "${task_output:parent_loss_attestation}",
            _producer_output_route("architecture_attestation", 0),
            _producer_output_route("parent_loss_attestation", 0),
            "${parent_recipe}",
        ))
    if task.kind == "kernel_resources":
        rows.append("${prebuilt_representation_recipe}")
    if task.kind == "cache_miniature":
        for bank in ("D100", "TOFF"):
            rows.extend((
                f"${{cache_miniature:{bank}:evidence}}",
                f"${{cache_miniature:{bank}:cleanup_authorization}}",
                f"${{cache_miniature:{bank}:cleanup_completion}}",
            ))
    # Retain the legacy dependency alias only when it has exactly one scalar
    # output.  Multi-output and array producers are always named by the exact
    # ordinal route emitted above.
    for dependency in task.dependencies:
        producer = task_by_key[dependency]
        if producer.array is None and len(producer.registered_outputs) == 1:
            rows.append(f"${{task_output:{dependency}}}")
    if task.kind in {"target_build", "train_node", "train_control", "confirmation"}:
        rows.append("${kernel_resources}")
    if task.logical_bank is not None:
        rows.append(f"${{logical_bank:{task.logical_bank}}}")
    rows.extend(adapter_registered_input_requirements(
        task,
        final_source_partitions=final_source_partitions,
        combined_finalist_count=combined_finalist_count,
    ))
    if task.array_registry is not None:
        rows.append("${array_registry:" + task.array_registry + "}")
    return tuple(dict.fromkeys(rows))


def validate_task_registry(tasks: Sequence[CampaignTask], *, disposition: str) -> None:
    by_key = {row.task_key: row for row in tasks}
    if len(by_key) != len(tasks):
        raise ValueError("representation campaign task keys repeat")
    if {row.graph_node for row in tasks if row.kind == "train_node"} != set(primary_node_ids()):
        raise ValueError("representation campaign does not contain all 24 primary nodes")
    if {row.graph_node for row in tasks if row.kind == "train_control"} != set(CONTROLS):
        raise ValueError("representation campaign does not contain the four controls")
    required_singletons = {
        "control_registry", "cache_miniature", "zero_coefficient_acceptance",
        "shuffle_map",
    }
    if any(sum(row.kind == kind for row in tasks) != 1 for kind in required_singletons):
        raise ValueError("representation campaign gate-task inventory differs")
    for row in tasks:
        if row.resource_class not in resource_table(mode="smoke"):
            raise ValueError("representation campaign resource class differs")
        if any(parent not in by_key for parent in row.dependencies):
            raise ValueError(f"unknown dependency of task {row.task_key}")
        if row.array is not None and "%" in row.array:
            raise ValueError("representation arrays are uncapped until a measured profile says otherwise")
        if row.deterministic_worker != (row.kind in DETERMINISTIC_KINDS):
            raise ValueError("deterministic worker assignment differs")
        if (row.array is None) != (row.array_registry is None):
            raise ValueError("array task must bind one immutable array registry")
        if row.logical_bank is None:
            if row.target_purpose is not None:
                raise ValueError("non-target task declares a target purpose")
        elif row.target_purpose not in {"screen", "confirmation", "miniature"}:
            raise ValueError("target-bank task purpose differs")
        if not row.registered_inputs or not row.registered_outputs:
            raise ValueError("campaign task lacks exact registered inputs or outputs")
    visited: set[str] = set()
    active: set[str] = set()
    def visit(key: str) -> None:
        if key in active:
            raise ValueError("representation campaign graph contains a cycle")
        if key in visited:
            return
        active.add(key)
        for parent in by_key[key].dependencies:
            visit(parent)
        active.remove(key); visited.add(key)
    for key in by_key:
        visit(key)

    def ancestors(key: str) -> set[str]:
        found: set[str] = set()
        pending = list(by_key[key].dependencies)
        while pending:
            parent = pending.pop()
            if parent in found:
                continue
            found.add(parent)
            pending.extend(by_key[parent].dependencies)
        return found

    for consumer in tasks:
        ancestry = ancestors(consumer.task_key)
        for logical in consumer.registered_inputs:
            match = EXACT_OUTPUT_ROUTE.fullmatch(logical)
            if match is None:
                continue
            producer_key, array_text, output_text = match.groups()
            producer = by_key.get(producer_key)
            if producer is None or producer_key not in ancestry:
                raise ValueError(
                    "registered producer output is not a transitive dependency"
                )
            output_index = int(output_text)
            if output_index >= len(producer.registered_outputs):
                raise ValueError("registered producer output ordinal is absent")
            if producer.array is None:
                if array_text is not None:
                    raise ValueError("scalar producer route unexpectedly has an array row")
            else:
                if array_text is None:
                    raise ValueError("array producer route lacks its exact array row")
                lower, upper = map(int, str(producer.array).split("-"))
                if int(array_text) not in range(lower, upper + 1):
                    raise ValueError("registered producer array row is absent")

    if disposition == "combined_confirmatory":
        assignment_array = by_key["final_assignment_shards"].array
        finalist_array = by_key["final_prediction_manifests"].array
        assert assignment_array is not None and finalist_array is not None
        final_source_partitions = int(assignment_array.split("-")[1]) + 1
        combined_finalist_count = int(finalist_array.split("-")[1]) + 1
    else:
        # Final-only adapter branches are absent; these values are not consumed
        # by any remaining requirement row.
        final_source_partitions = combined_finalist_count = 1
    for row in tasks:
        required_inputs = set(adapter_registered_input_requirements(
            row,
            final_source_partitions=final_source_partitions,
            combined_finalist_count=combined_finalist_count,
        ))
        missing = required_inputs - set(row.registered_inputs)
        if missing:
            raise ValueError(
                f"campaign task lacks production-adapter artifact routes: "
                f"{row.task_key}: {sorted(missing)}"
            )

    miniature = (
        ("miniature_D100_build", "target_build", "D100", ("numerical_acceptance",)),
        (
            "miniature_D100_verify_cleanup", "cache_miniature_bank", "D100",
            ("miniature_D100_build",),
        ),
        (
            "miniature_TOFF_build", "target_build", "TOFF",
            ("miniature_D100_verify_cleanup",),
        ),
        (
            "miniature_TOFF_verify_cleanup", "cache_miniature_bank", "TOFF",
            ("miniature_TOFF_build",),
        ),
    )
    for key, kind, bank, dependencies in miniature:
        row = by_key.get(key)
        if (
            row is None or row.kind != kind or row.logical_bank != bank
            or row.target_purpose != "miniature"
            or row.dependencies != dependencies
        ):
            raise ValueError("ordinary/TOFF cache-miniature lifecycle differs")
    if by_key["cache_miniature"].dependencies != (
        "miniature_D100_verify_cleanup", "miniature_TOFF_verify_cleanup",
    ):
        raise ValueError("cache-miniature acceptance lacks both lifecycle reports")
    if by_key["zero_coefficient_acceptance"].dependencies != ("smoke_probe",):
        raise ValueError("zero-coefficient acceptance is not post-smoke")
    d100_builds = [
        row for row in tasks
        if row.kind == "target_build" and row.logical_bank == "D100"
        and row.target_purpose == "screen"
    ]
    if len(d100_builds) != 1 or "shuffle_map" not in d100_builds[0].dependencies:
        raise ValueError("D100 controls can run before the frozen shuffle map")
    final_kinds = {"final_selection", "assignment_shard", "assignment_finalize", "data_attestation", "execution_lock", "prediction_shard", "prediction_finalize", "metric_join"}
    if disposition != "combined_confirmatory" and any(row.kind in final_kinds for row in tasks):
        raise PermissionError("validation-only campaign registered final-role tasks")
    if disposition == "combined_confirmatory":
        if (
            by_key["final_prediction_shards"].dependencies
            != ("representation_execution_lock",)
            or by_key["final_aggregate"].dependencies != ("locked_metric_join",)
        ):
            raise ValueError("final prediction/bootstrap boundary differs")
    # Peak-liveness proof: every target build after the first has the immediately
    # prior cleanup in its ancestry; therefore committed banks cannot overlap.
    builds = [
        row for row in tasks
        if row.kind == "target_build" and row.target_purpose != "miniature"
    ]
    for previous, current in zip(builds, builds[1:], strict=False):
        candidates = [
            row.task_key for row in tasks
            if row.kind == "target_cleanup"
            and row.logical_bank == previous.logical_bank
            and row.target_purpose == previous.target_purpose
        ]
        if len(candidates) != 1:
            raise ValueError("target generation lacks one exact cleanup task")
        previous_cleanup = candidates[0]
        stack = list(current.dependencies); ancestors: set[str] = set()
        while stack:
            key = stack.pop()
            if key not in ancestors:
                ancestors.add(key); stack.extend(by_key[key].dependencies)
        if previous_cleanup not in ancestors:
            raise ValueError("target-bank lifecycle can overlap committed generations")
    # Every scientific training row binds its immediate deployable predecessor
    # directly.  Bank serialization alone is not accepted as model lineage.
    for row in tasks:
        match = re.fullmatch(r"(RSET|RREL)_M([1-6])([cw])", str(row.graph_node))
        if row.kind == "train_node" and match is not None and int(match.group(2)) > 1:
            expected_parent = f"train_{match.group(1)}_M{int(match.group(2)) - 1}{match.group(3)}"
            if expected_parent not in row.dependencies:
                raise ValueError("representation node lacks direct predecessor dependency")
        if row.kind == "confirmation" and match is not None:
            expected_parent = f"train_{match.group(1)}_M5{match.group(3)}"
            if expected_parent not in row.dependencies:
                raise ValueError("confirmation row lacks its selected M5 parent dependency")


def create_campaign_spec(
    *, mode: str, campaign_root: str | Path, checkpoint_namespace: str | Path,
    project_dir: str | Path, source_commit: str, source_manifest_sha256: str,
    split_manifest_sha256: str, parent_import_sha256: str,
    representation_recipe_sha256: str, graph_sha256: str,
    disposition_sha256: str, disposition: str,
    role_counts: Mapping[str, int], final_source_partitions: int,
    combined_finalist_count: int, planning_only: bool = True,
    executable_candidate_audit_sha256: str | None = None,
    executable_candidate_audit: Mapping[str, Any] | None = None,
    submission_authorization_sha256: str | None = None,
    submission_authorization: Mapping[str, Any] | None = None,
    resource_profile: Mapping[str, Any] | None = None,
    storage_estimate: Mapping[str, Any] | None = None,
    fixed_size_inventory: Mapping[str, Any] | None = None,
    tigris_acceptance: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    runtime_binding_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("unknown representation campaign mode")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("representation source commit must be a full lowercase Git SHA")
    if set(role_counts) != {"train", "validation", "final_test"}:
        raise ValueError("representation role counts differ")
    expected = {"smoke": None, "pilot": (300_000, 100_000, 100_000), "production": None}[mode]
    counts = tuple(int(role_counts[name]) for name in ("train", "validation", "final_test"))
    if any(value <= 0 for value in counts):
        raise ValueError("representation role counts must be positive")
    if expected is not None and counts != expected:
        raise ValueError("pilot must contain exactly 300k/100k/100k rows")
    root_text = _operational_path(campaign_root, name="campaign root")
    root = PurePosixPath(root_text)
    if artifact_paths is None:
        artifact_paths = {
            "source_manifest": root / "inputs" / "source_manifest.json",
            "split_manifest": root / "inputs" / "split_manifest.json",
            "parent_import": root / "import" / "parent_import.json",
            "representation_graph": root / "graph" / "ascent_graph.json",
            "representation_recipe": root / "recipes" / "representation_recipe.json",
            "final_disposition": root / "import" / "final_disposition.json",
            "runtime_binding": root / "runtime" / "runtime_binding.json",
        }
    if set(artifact_paths) != REQUIRED_ARTIFACT_PATHS:
        raise ValueError("representation campaign artifact-path registry differs")
    normalized_paths = {
        name: _operational_path(value, name=f"artifact path {name}")
        for name, value in sorted(artifact_paths.items())
    }
    if any(not value for value in normalized_paths.values()):
        raise ValueError("representation campaign artifact path is empty")
    runtime_hash = (
        None if runtime_binding_sha256 is None
        else require_sha256(runtime_binding_sha256, name="runtime binding")
    )
    profile_hash = None
    array_concurrency_limits: dict[str, int] = {}
    if resource_profile is not None:
        profile_hash = validate_measured_profile(
            resource_profile, require_genuine_tigris=not planning_only,
            expected_source_commit=source_commit,
        )
        resources = dict(resource_profile["requests"])
        array_concurrency_limits = {
            str(key): int(value)
            for key, value in resource_profile.get(
                "array_concurrency_limits", {}
            ).items()
        }
    else:
        resources = resource_table(mode=mode)
    fixed_size_inventory_hash = None
    if fixed_size_inventory is not None:
        _, fixed_size_inventory_hash = load_authenticated_json_reference(
            fixed_size_inventory,
            expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
            name="campaign fixed-size inventory",
        )
    storage_hash = None
    if storage_estimate is not None:
        storage_hash = validate_storage_estimate(
            storage_estimate, require_measured_fixed_sizes=not planning_only,
            fixed_size_inventory=fixed_size_inventory,
        )
        if storage_estimate.get("parent_import_sha256") != parent_import_sha256:
            raise ValueError("representation storage estimate parent differs")
        expected_storage_counts = dict(zip(
            ("train", "validation", "final"), counts, strict=True,
        ))
        if storage_estimate.get("row_counts") != expected_storage_counts:
            raise ValueError("representation storage estimate role populations differ")
        expected_prediction_finalists = (
            int(combined_finalist_count)
            if disposition == "combined_confirmatory"
            else 0
        )
        if storage_estimate.get("prediction_finalists") != expected_prediction_finalists:
            raise ValueError("representation storage estimate finalist count differs")
    acceptance_hash = None
    if tigris_acceptance is not None:
        if any(value is None for value in (
            profile_hash, storage_hash, fixed_size_inventory_hash,
        )):
            raise ValueError(
                "Tigris acceptance requires its measured resources, storage, and inventory"
            )
        acceptance_hash = validate_tigris_acceptance(
            tigris_acceptance, source_commit=source_commit,
            representation_recipe_sha256=representation_recipe_sha256,
            resource_profile_sha256=profile_hash,
            storage_estimate_sha256=storage_hash,
            fixed_size_inventory_sha256=fixed_size_inventory_hash,
        )
    if not planning_only and any(value is None for value in (
        executable_candidate_audit, submission_authorization,
        profile_hash, storage_hash, acceptance_hash, runtime_hash,
        fixed_size_inventory_hash,
    )):
        raise PermissionError(
            "executable representation campaign requires runtime, measured resources, "
            "storage, genuine Tigris acceptance, and explicit authorization"
        )
    if planning_only and any(value is not None for value in (
        executable_candidate_audit, executable_candidate_audit_sha256,
        submission_authorization, submission_authorization_sha256,
    )):
        raise ValueError(
            "planning-only campaign cannot embed candidate or submission authority"
        )
    tasks = build_task_registry(
        disposition=disposition,
        final_source_partitions=int(final_source_partitions),
        combined_finalist_count=int(combined_finalist_count),
    )
    if not set(array_concurrency_limits) <= {
        task.task_key for task in tasks if task.array is not None
    }:
        raise ValueError("measured concurrency limit names a non-array task")
    payload = {
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": 1,
        "mode": mode,
        "planning_only": bool(planning_only),
        "live_submission_authorized": not planning_only,
        "campaign_root": root_text,
        "checkpoint_namespace": _operational_path(
            checkpoint_namespace, name="checkpoint namespace",
        ),
        "project_dir": _operational_path(project_dir, name="project directory"),
        "source_commit": source_commit,
        "source_manifest_sha256": require_sha256(source_manifest_sha256, name="source manifest"),
        "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split manifest"),
        "parent_import_sha256": require_sha256(parent_import_sha256, name="parent import"),
        "representation_recipe_sha256": require_sha256(
            representation_recipe_sha256, name="representation recipe"
        ),
        "graph_sha256": require_sha256(graph_sha256, name="graph"),
        "disposition_sha256": require_sha256(disposition_sha256, name="disposition"),
        "disposition": disposition,
        "role_counts": dict(zip(("train", "validation", "final_test"), counts, strict=True)),
        "final_source_partitions": int(final_source_partitions),
        "combined_finalist_count": int(combined_finalist_count),
        "artifact_paths": normalized_paths,
        "runtime_binding_sha256": runtime_hash,
        "runtime_status": (
            "immutable" if runtime_hash is not None else "unresolved_planning_input"
        ),
        "resource_profile": None if resource_profile is None else dict(resource_profile),
        "resource_profile_sha256": profile_hash,
        "storage_estimate": None if storage_estimate is None else dict(storage_estimate),
        "storage_estimate_sha256": storage_hash,
        "fixed_size_inventory": (
            None if fixed_size_inventory is None else dict(fixed_size_inventory)
        ),
        "fixed_size_inventory_sha256": fixed_size_inventory_hash,
        "tigris_acceptance": None if tigris_acceptance is None else dict(tigris_acceptance),
        "tigris_acceptance_sha256": acceptance_hash,
        "resources": resources,
        "array_concurrency_limits": array_concurrency_limits,
        "resource_request_sha256": canonical_sha256(resources),
        "tasks": [asdict(row) for row in tasks],
        "executable_candidate_audit_sha256": None,
        "executable_candidate_audit": None,
        "submission_authorization_sha256": None,
        "submission_authorization": None,
    }
    provisional = with_content_hash(payload)
    plan_hash = build_command_plan(provisional)["content_hash"]
    payload["command_plan_sha256"] = plan_hash
    candidate_hash = None
    if executable_candidate_audit is not None:
        from .hcwdl_representation_candidate import (
            validate_executable_candidate_audit,
        )

        # Candidate identity intentionally excludes both the review artifact
        # and the later human authorization.  Validate it against the exact
        # otherwise-complete live specification before granting either field
        # authority in the final immutable payload.
        candidate_spec = with_content_hash(payload)
        candidate_hash = validate_executable_candidate_audit(
            executable_candidate_audit,
            campaign_spec=candidate_spec,
        )
        if (
            executable_candidate_audit_sha256 is not None
            and executable_candidate_audit_sha256 != candidate_hash
        ):
            raise ValueError("representation supplied candidate-audit hash differs")
        payload["executable_candidate_audit"] = dict(executable_candidate_audit)
        payload["executable_candidate_audit_sha256"] = candidate_hash
    elif executable_candidate_audit_sha256 is not None:
        raise ValueError("candidate-audit hash without its artifact is path-only authority")
    if submission_authorization is not None:
        assert (
            profile_hash is not None
            and storage_hash is not None
            and acceptance_hash is not None
            and candidate_hash is not None
        )
        authorization_hash = validate_submission_authorization(
            submission_authorization, mode=mode, source_commit=source_commit,
            command_plan_sha256=plan_hash, resource_profile_sha256=profile_hash,
            executable_candidate_audit_sha256=candidate_hash,
            storage_estimate_sha256=storage_hash,
            tigris_acceptance_sha256=acceptance_hash,
            parent_import_sha256=parent_import_sha256,
            representation_recipe_sha256=representation_recipe_sha256,
            disposition_sha256=disposition_sha256,
        )
        if (
            submission_authorization_sha256 is not None
            and submission_authorization_sha256 != authorization_hash
        ):
            raise ValueError("representation supplied authorization hash differs")
        payload["submission_authorization"] = dict(submission_authorization)
        payload["submission_authorization_sha256"] = authorization_hash
    elif submission_authorization_sha256 is not None:
        raise ValueError("authorization hash without its artifact is path-only authority")
    return with_content_hash(payload)


def validate_campaign_spec(spec: Mapping[str, Any], *, executable: bool = False) -> str:
    digest = validate_content_hash(
        spec, expected_contract=CAMPAIGN_SPEC_CONTRACT, expected_schema_version=1,
    )
    if spec.get("mode") not in MODES:
        raise ValueError("representation campaign mode differs")
    role_counts = spec.get("role_counts")
    if not isinstance(role_counts, Mapping) or set(role_counts) != {
        "train", "validation", "final_test",
    }:
        raise ValueError("representation campaign role counts differ")
    normalized_role_counts = tuple(
        int(role_counts[name]) for name in ("train", "validation", "final_test")
    )
    if any(value <= 0 for value in normalized_role_counts):
        raise ValueError("representation campaign role counts must be positive")
    if spec.get("mode") == "pilot" and normalized_role_counts != (
        300_000, 100_000, 100_000,
    ):
        raise ValueError("pilot must contain exactly 300k/100k/100k rows")
    tasks = tuple(
        CampaignTask(
            **{
                **row,
                "dependencies": tuple(row["dependencies"]),
                "registered_inputs": tuple(row["registered_inputs"]),
                "registered_outputs": tuple(row["registered_outputs"]),
            }
        )
        for row in spec["tasks"]
    )
    validate_task_registry(tasks, disposition=str(spec["disposition"]))
    expected = build_task_registry(
        disposition=str(spec["disposition"]),
        final_source_partitions=int(spec["final_source_partitions"]),
        combined_finalist_count=int(spec["combined_finalist_count"]),
    )
    if tasks != expected:
        raise ValueError("representation campaign registry differs")
    profile = spec.get("resource_profile")
    if profile is None:
        expected_resources = resource_table(mode=str(spec["mode"]))
        if spec.get("resource_profile_sha256") is not None:
            raise ValueError("representation campaign has a path-only resource profile")
    else:
        profile_hash = validate_measured_profile(
            profile, require_genuine_tigris=executable,
            expected_source_commit=str(spec["source_commit"]),
        )
        if spec.get("resource_profile_sha256") != profile_hash:
            raise ValueError("representation resource-profile lineage differs")
        expected_resources = dict(profile["requests"])
    if spec.get("resources") != expected_resources:
        raise ValueError("representation campaign resource requests differ")
    expected_limits = (
        {} if profile is None else {
            str(key): int(value)
            for key, value in profile.get(
                "array_concurrency_limits", {}
            ).items()
        }
    )
    if spec.get("array_concurrency_limits") != expected_limits:
        raise ValueError("representation measured array concurrency differs")
    if not set(expected_limits) <= {
        task.task_key for task in tasks if task.array is not None
    }:
        raise ValueError("representation concurrency limit names a non-array task")
    if spec.get("resource_request_sha256") != canonical_sha256(expected_resources):
        raise ValueError("representation resource-request hash differs")
    if set(spec.get("artifact_paths", {})) != REQUIRED_ARTIFACT_PATHS:
        raise ValueError("representation campaign artifact path registry differs")
    runtime_hash = spec.get("runtime_binding_sha256")
    if runtime_hash is not None:
        require_sha256(runtime_hash, name="runtime binding")
        if spec.get("runtime_status") != "immutable":
            raise ValueError("representation runtime status differs")
    elif spec.get("runtime_status") != "unresolved_planning_input":
        raise ValueError("representation unresolved runtime status differs")
    fixed_size_inventory = spec.get("fixed_size_inventory")
    if fixed_size_inventory is None:
        fixed_size_inventory_hash = None
        if spec.get("fixed_size_inventory_sha256") is not None:
            raise ValueError("representation campaign has a path-only fixed-size inventory")
    else:
        _, fixed_size_inventory_hash = load_authenticated_json_reference(
            fixed_size_inventory,
            expected_contract=FIXED_SIZE_INVENTORY_CONTRACT,
            name="campaign fixed-size inventory",
        )
        if spec.get("fixed_size_inventory_sha256") != fixed_size_inventory_hash:
            raise ValueError("representation fixed-size inventory lineage differs")
    storage = spec.get("storage_estimate")
    if storage is not None:
        storage_hash = validate_storage_estimate(
            storage, require_measured_fixed_sizes=executable,
            fixed_size_inventory=fixed_size_inventory,
        )
        if spec.get("storage_estimate_sha256") != storage_hash:
            raise ValueError("representation storage-estimate lineage differs")
        if storage.get("parent_import_sha256") != spec.get("parent_import_sha256"):
            raise ValueError("representation storage estimate parent differs")
        expected_storage_counts = {
            "train": int(role_counts["train"]),
            "validation": int(role_counts["validation"]),
            "final": int(role_counts["final_test"]),
        }
        if storage.get("row_counts") != expected_storage_counts:
            raise ValueError("representation storage estimate role populations differ")
        expected_prediction_finalists = (
            int(spec["combined_finalist_count"])
            if spec.get("disposition") == "combined_confirmatory"
            else 0
        )
        if storage.get("prediction_finalists") != expected_prediction_finalists:
            raise ValueError("representation storage estimate finalist count differs")
    elif spec.get("storage_estimate_sha256") is not None:
        raise ValueError("representation campaign has a path-only storage estimate")
    acceptance = spec.get("tigris_acceptance")
    if acceptance is not None:
        if profile is None or storage is None or fixed_size_inventory is None:
            raise ValueError(
                "representation acceptance lacks measured resource/storage evidence"
            )
        acceptance_hash = validate_tigris_acceptance(
            acceptance, source_commit=str(spec["source_commit"]),
            representation_recipe_sha256=str(spec["representation_recipe_sha256"]),
            resource_profile_sha256=str(spec["resource_profile_sha256"]),
            storage_estimate_sha256=str(spec["storage_estimate_sha256"]),
            fixed_size_inventory_sha256=str(
                spec["fixed_size_inventory_sha256"]
            ),
        )
        if spec.get("tigris_acceptance_sha256") != acceptance_hash:
            raise ValueError("representation Tigris-acceptance lineage differs")
    elif spec.get("tigris_acceptance_sha256") is not None:
        raise ValueError("representation campaign has path-only Tigris acceptance")
    if spec.get("command_plan_sha256") != build_command_plan(spec)["content_hash"]:
        raise ValueError("representation command plan lineage differs")
    if executable:
        if (
            spec.get("planning_only") is not False
            or spec.get("live_submission_authorized") is not True
            or runtime_hash is None
            or profile is None
            or storage is None
            or fixed_size_inventory is None
            or acceptance is None
            or not isinstance(spec.get("executable_candidate_audit"), Mapping)
        ):
            raise PermissionError("representation campaign is not authorized for live submission")
        # The executable gate validates the concrete, path-only runtime
        # command-plan subtype itself.  A matching hash string or path alone
        # is never authority to dispatch production work.
        from hlt_classification.data.cache_contracts import load_json
        from .hcwdl_representation_runtime_binding import validate_runtime_binding

        runtime_binding = load_json(spec["artifact_paths"]["runtime_binding"])
        if validate_runtime_binding(runtime_binding, spec=spec) != runtime_hash:
            raise ValueError("representation runtime-binding artifact/hash differs")
        from .hcwdl_representation_candidate import (
            validate_executable_candidate_audit,
        )

        candidate_hash = validate_executable_candidate_audit(
            spec["executable_candidate_audit"], campaign_spec=spec,
        )
        if spec.get("executable_candidate_audit_sha256") != candidate_hash:
            raise ValueError("representation executable-candidate audit hash differs")
        authorization = spec.get("submission_authorization")
        if not isinstance(authorization, Mapping):
            raise PermissionError("representation campaign lacks explicit authorization artifact")
        authorization_hash = validate_submission_authorization(
            authorization, mode=str(spec["mode"]),
            source_commit=str(spec["source_commit"]),
            command_plan_sha256=str(spec["command_plan_sha256"]),
            executable_candidate_audit_sha256=candidate_hash,
            resource_profile_sha256=str(spec["resource_profile_sha256"]),
            storage_estimate_sha256=str(spec["storage_estimate_sha256"]),
            tigris_acceptance_sha256=str(spec["tigris_acceptance_sha256"]),
            parent_import_sha256=str(spec["parent_import_sha256"]),
            representation_recipe_sha256=str(spec["representation_recipe_sha256"]),
            disposition_sha256=str(spec["disposition_sha256"]),
        )
        if spec.get("submission_authorization_sha256") != authorization_hash:
            raise ValueError("representation authorization hash differs")
    elif any(spec.get(key) is not None for key in (
        "executable_candidate_audit",
        "executable_candidate_audit_sha256",
        "submission_authorization",
        "submission_authorization_sha256",
    )):
        raise PermissionError(
            "planning campaign unexpectedly embeds candidate or live authorization"
        )
    elif (
        spec.get("planning_only") is not True
        or spec.get("live_submission_authorized") is not False
    ):
        raise PermissionError("nonauthorizing campaign planning flags differ")
    return digest


def _campaign_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: spec.get(key)
        for key in (
            "mode", "campaign_root", "checkpoint_namespace", "project_dir",
            "source_commit", "source_manifest_sha256", "split_manifest_sha256",
            "parent_import_sha256", "representation_recipe_sha256", "graph_sha256",
            "disposition_sha256", "disposition", "role_counts", "artifact_paths",
            "resource_profile_sha256", "storage_estimate_sha256",
            "fixed_size_inventory_sha256",
            "tigris_acceptance_sha256",
            "resource_request_sha256", "array_concurrency_limits",
        )
    }


def _commands(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    project = str(spec["project_dir"])
    root = str(spec["campaign_root"])
    identity_sha256 = canonical_sha256(_campaign_identity(spec))
    rows = []
    for task in spec["tasks"]:
        resource = spec["resources"][task["resource_class"]]
        reconciliation_token = canonical_sha256({
            "campaign_identity_sha256": identity_sha256,
            "task_key": task["task_key"],
        })
        command = [
            "sbatch", "--parsable", "--account=reu-aisocial", "--partition=tigris",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['memory']}",
            f"--time={resource['walltime']}", f"--job-name=hcwdl_rkd_{task['task_key']}",
            f"--comment=hcwdl-rkd-{reconciliation_token}",
        ]
        if resource["gpu"] is not None:
            command.append(f"--gres={resource['gpu']}")
        if task["kind"] in {"train_node", "train_control", "confirmation"}:
            command.append("--signal=B:USR1@120")
        operational_limit = spec.get("array_concurrency_limits", {}).get(
            task["task_key"]
        )
        if task["array"] is not None:
            array_argument = str(task["array"])
            if operational_limit is not None:
                array_argument += f"%{int(operational_limit)}"
            command.append(f"--array={array_argument}")
        if task["dependencies"]:
            # Slurm consumes one dependency-list argument.  A repeated
            # ``--dependency`` is not used because option parsers may retain
            # only the final occurrence.  The only late-bound material is the
            # ordered task-key group inside this typed token.
            command.extend((
                "--dependency",
                "${afterok:" + ",".join(task["dependencies"]) + "}",
            ))
        command.extend(
            (
                "--export",
                "ALL," + f"PROJECT_DIR={project},HCWDL_REPRESENTATION_SPEC={root}/campaign_spec.json," +
                f"HCWDL_REPRESENTATION_TASK={task['task_key']}," +
                "HCWDL_REPRESENTATION_RUNTIME_BINDING=" +
                str(spec["artifact_paths"]["runtime_binding"]),
                f"{project}/sbatch/" + (
                    "run_hcwdl_representation_deterministic_task.sh"
                    if task["deterministic_worker"]
                    else "run_hcwdl_representation_task.sh"
                ),
            )
        )
        rows.append(
            {
                "task_key": task["task_key"],
                "kind": task["kind"],
                "dependencies": list(task["dependencies"]),
                "resource_class": task["resource_class"],
                "resource_request": dict(resource),
                "array": task["array"],
                "operational_array_concurrency": operational_limit,
                "array_registry": task["array_registry"],
                "deterministic_worker": bool(task["deterministic_worker"]),
                "source_commit": spec["source_commit"],
                "source_checkout_proof": {
                    "clean_required": True,
                    "pushed_remote_ref_required": True,
                },
                "campaign_identity_sha256": identity_sha256,
                "scheduler_reconciliation_token": reconciliation_token,
                "runtime_binding_path": spec["artifact_paths"]["runtime_binding"],
                "canonical_campaign_spec_path": str(
                    PurePosixPath(root) / "campaign_spec.json"
                ),
                "command": command,
                "registered_inputs": list(task["registered_inputs"]),
                "registered_outputs": list(task["registered_outputs"]),
            }
        )
    return rows


def build_command_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    campaign_identity = _campaign_identity(spec)
    return with_content_hash(
        {
            "contract": COMMAND_PLAN_CONTRACT,
            "schema_version": 1,
            "campaign_identity": campaign_identity,
            "campaign_identity_sha256": canonical_sha256(campaign_identity),
            "campaign_root": spec["campaign_root"],
            "canonical_campaign_spec_path": str(
                PurePosixPath(str(spec["campaign_root"])) / "campaign_spec.json"
            ),
            "source_checkout_proof": {
                "expected_commit": spec["source_commit"],
                "clean_required": True,
                "pushed_remote_ref_required": True,
                "verified_only_at_submission": True,
            },
            "commands": _commands(spec),
        }
    )


def materialize_command(
    row: Mapping[str, Any], *, job_ids: Mapping[str, str],
) -> list[str]:
    command = []
    expected_dependencies = tuple(row["dependencies"])
    observed = []
    index = 0
    tokens = list(row["command"])
    while index < len(tokens):
        token = str(tokens[index])
        if token == "--dependency":
            if index + 1 >= len(tokens):
                raise ValueError("dependency option lacks its typed token")
            match = DEPENDENCY_TOKEN.match(str(tokens[index + 1]))
            if match is None:
                raise ValueError("command plan dependency is not a typed token")
            keys = match.group(1).split(",")
            observed.extend(keys)
            missing = [key for key in keys if key not in job_ids]
            if missing:
                raise KeyError(f"upstream scheduler IDs are unavailable for {missing!r}")
            command.extend((
                "--dependency",
                "afterok:" + ":".join(job_ids[key] for key in keys),
            ))
            index += 2
            continue
        if DEPENDENCY_TOKEN.search(token):
            raise ValueError("dependency token may only occupy its own argv element")
        command.append(token); index += 1
    if tuple(observed) != expected_dependencies:
        raise ValueError("materialized dependency order differs")
    return command


def materialize_recovery_command(
    row: Mapping[str, Any], *, job_ids: Mapping[str, str],
    array_indices: Sequence[int] | None,
) -> list[str]:
    """Materialize one reviewed row, optionally narrowing its registered array."""

    if array_indices is None:
        return materialize_command(row, job_ids=job_ids)
    specification = row.get("array")
    match = re.fullmatch(r"(\d+)-(\d+)", str(specification))
    if match is None:
        raise ValueError("array-row recovery requested for a non-array task")
    lower, upper = map(int, match.groups())
    subset = tuple(sorted({int(index) for index in array_indices}))
    if not subset or any(index < lower or index > upper for index in subset):
        raise ValueError("recovery array indices are not a nonempty registered subset")
    replacement = "--array=" + ",".join(str(index) for index in subset)
    copied = dict(row)
    tokens = list(row["command"])
    positions = [index for index, token in enumerate(tokens) if str(token).startswith("--array=")]
    operational_limit = row.get("operational_array_concurrency")
    original_array = f"--array={specification}"
    if operational_limit is not None:
        original_array += f"%{int(operational_limit)}"
        replacement += f"%{int(operational_limit)}"
    if len(positions) != 1 or tokens[positions[0]] != original_array:
        raise ValueError("reviewed command array token differs from its registry")
    tokens[positions[0]] = replacement
    copied["command"] = tokens
    return materialize_command(copied, job_ids=job_ids)


def validate_command_plan(
    command_plan: Mapping[str, Any], *, spec: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        command_plan, expected_contract=COMMAND_PLAN_CONTRACT,
        expected_schema_version=1,
    )
    expected = build_command_plan(spec)
    if dict(command_plan) != expected:
        raise ValueError("representation command plan differs from immutable campaign inputs")
    return digest


def validate_submission_ledger(
    ledger: Mapping[str, Any], *, spec: Mapping[str, Any],
    command_plan: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        ledger, expected_contract=SUBMISSION_LEDGER_CONTRACT,
        expected_schema_version=1,
    )
    validate_command_plan(command_plan, spec=spec)
    if (
        ledger.get("campaign_spec_sha256") != spec["content_hash"]
        or ledger.get("command_plan_sha256") != command_plan["content_hash"]
    ):
        raise ValueError("representation submission ledger parent lineage differs")
    jobs = ledger.get("jobs")
    rows = ledger.get("materialized_commands")
    if not isinstance(jobs, Mapping) or not isinstance(rows, list):
        raise ValueError("representation submission ledger topology differs")
    if list(jobs) != [row["task_key"] for row in command_plan["commands"]]:
        raise ValueError("representation submission ledger task order differs")
    if len(rows) != len(command_plan["commands"]):
        raise ValueError("representation submission ledger row count differs")
    prior: dict[str, str] = {}
    for template, materialized in zip(command_plan["commands"], rows, strict=True):
        task_key = template["task_key"]
        job_id = str(jobs[task_key])
        if not SLURM_JOB_ID.fullmatch(job_id):
            raise ValueError("representation submission ledger contains invalid exact job ID")
        expected_command = materialize_command(template, job_ids=prior)
        if (
            materialized.get("task_key") != task_key
            or materialized.get("job_id") != job_id
            or materialized.get("command") != expected_command
        ):
            raise ValueError("representation submitted argv differs from reviewed template")
        prior[task_key] = job_id
    return digest


def _submission_event(
    *, spec: Mapping[str, Any], command_plan: Mapping[str, Any],
    sequence: int, task_sequence: int, phase: str, task_key: str,
    command: Sequence[str], previous_event_sha256: str | None,
    intent_sha256: str | None = None, job_id: str | None = None,
) -> dict[str, Any]:
    if (
        isinstance(sequence, bool) or not isinstance(sequence, int)
        or sequence < 0 or isinstance(task_sequence, bool)
        or not isinstance(task_sequence, int) or task_sequence < 0
        or phase not in {"intent", "submitted"} or not task_key
        or not isinstance(command, Sequence) or isinstance(command, (str, bytes))
        or not command or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ValueError("representation submission event identity differs")
    previous = (
        None if previous_event_sha256 is None else require_sha256(
            previous_event_sha256, name="previous submission event",
        )
    )
    if phase == "intent":
        if intent_sha256 is not None or job_id is not None:
            raise ValueError("submission intent cannot claim a scheduler result")
        normalized_job = None
        normalized_intent = None
    else:
        normalized_intent = require_sha256(intent_sha256, name="submission intent")
        normalized_job = str(job_id)
        if not SLURM_JOB_ID.fullmatch(normalized_job):
            raise ValueError("scheduler returned an invalid exact job ID")
    return with_content_hash({
        "contract": SUBMISSION_EVENT_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "command_plan_sha256": command_plan["content_hash"],
        "sequence": sequence, "task_sequence": task_sequence,
        "phase": phase, "task_key": task_key, "command": list(command),
        "previous_event_sha256": previous,
        "intent_sha256": normalized_intent, "job_id": normalized_job,
    })


def _submission_prefix_state(
    events: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
    command_plan: Mapping[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    """Authenticate an append-only initial-submission prefix."""

    validate_command_plan(command_plan, spec=spec)
    ordered = sorted(
        (dict(row) for row in events), key=lambda row: int(row.get("sequence", -1)),
    )
    if [row.get("sequence") for row in ordered] != list(range(len(ordered))):
        raise ValueError("representation submission event sequence differs")
    previous: str | None = None
    for row in ordered:
        validate_content_hash(
            row, expected_contract=SUBMISSION_EVENT_CONTRACT,
            expected_schema_version=1,
        )
        if set(row) != {
            "contract", "schema_version", "campaign_spec_sha256",
            "command_plan_sha256", "sequence", "task_sequence", "phase",
            "task_key", "command", "previous_event_sha256", "intent_sha256",
            "job_id", "content_hash",
        }:
            raise ValueError("representation submission event schema differs")
        if (
            row["campaign_spec_sha256"] != spec["content_hash"]
            or row["command_plan_sha256"] != command_plan["content_hash"]
            or row["previous_event_sha256"] != previous
        ):
            raise ValueError("representation submission event chain differs")
        previous = row["content_hash"]

    jobs: dict[str, str] = {}
    materialized: list[dict[str, Any]] = []
    cursor = 0
    for task_sequence, template in enumerate(command_plan["commands"]):
        if cursor == len(ordered):
            break
        command = materialize_command(template, job_ids=jobs)
        intent = ordered[cursor]
        expected_intent = _submission_event(
            spec=spec, command_plan=command_plan, sequence=cursor,
            task_sequence=task_sequence, phase="intent",
            task_key=template["task_key"], command=command,
            previous_event_sha256=(
                None if cursor == 0 else ordered[cursor - 1]["content_hash"]
            ),
        )
        if intent != expected_intent:
            raise ValueError("representation submission intent differs from reviewed argv")
        cursor += 1
        if cursor == len(ordered):
            return jobs, materialized, intent, ordered
        submitted = ordered[cursor]
        expected_submitted = _submission_event(
            spec=spec, command_plan=command_plan, sequence=cursor,
            task_sequence=task_sequence, phase="submitted",
            task_key=template["task_key"], command=command,
            previous_event_sha256=intent["content_hash"],
            intent_sha256=intent["content_hash"], job_id=submitted.get("job_id"),
        )
        if submitted != expected_submitted:
            raise ValueError("representation submitted event differs from its intent")
        job_id = str(submitted["job_id"])
        if job_id in jobs.values():
            raise ValueError("representation submission journal reuses a scheduler job ID")
        jobs[str(template["task_key"])] = job_id
        materialized.append({
            "task_key": template["task_key"], "job_id": job_id,
            "command": command,
        })
        cursor += 1
    if cursor != len(ordered):
        raise ValueError("representation submission journal extends beyond command plan")
    return jobs, materialized, None, ordered


def validate_submission_event_chain(
    events: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
    command_plan: Mapping[str, Any], require_complete: bool = False,
) -> str:
    jobs, _, pending, ordered = _submission_prefix_state(
        events, spec=spec, command_plan=command_plan,
    )
    if require_complete and (
        pending is not None or len(jobs) != len(command_plan["commands"])
    ):
        raise ValueError("representation submission event chain is incomplete")
    return canonical_sha256([row["content_hash"] for row in ordered])


def assemble_submission_ledger_from_events(
    events: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
    command_plan: Mapping[str, Any],
) -> dict[str, Any]:
    jobs, rows, pending, _ = _submission_prefix_state(
        events, spec=spec, command_plan=command_plan,
    )
    if pending is not None or len(jobs) != len(command_plan["commands"]):
        raise ValueError("representation submission event chain is incomplete")
    ledger = with_content_hash({
        "contract": SUBMISSION_LEDGER_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"],
        "command_plan_sha256": command_plan["content_hash"],
        "jobs": jobs, "materialized_commands": rows,
    })
    validate_submission_ledger(ledger, spec=spec, command_plan=command_plan)
    return ledger


def submit_command_plan(
    *, spec: Mapping[str, Any], command_plan: Mapping[str, Any],
    scheduler: Callable[[Sequence[str]], str], execute: bool,
    campaign_spec_path: str | Path | None = None,
    checkout_validator: Callable[[str | Path], None] | None = None,
    event_writer: Callable[[Mapping[str, Any]], None] | None = None,
    prior_events: Sequence[Mapping[str, Any]] = (),
    reconciled_job_ids: Mapping[str, str] | None = None,
    reconciliation_validator: Callable[[Mapping[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    validate_campaign_spec(spec, executable=execute)
    validate_command_plan(command_plan, spec=spec)
    if command_plan["content_hash"] != spec["command_plan_sha256"]:
        raise ValueError("submission command plan differs from campaign spec")
    if not execute:
        raise PermissionError("dry run never invokes the scheduler")
    expected_spec_path = (Path(str(spec["campaign_root"])) / "campaign_spec.json").resolve()
    supplied_spec_path = (
        expected_spec_path if campaign_spec_path is None else Path(campaign_spec_path).resolve()
    )
    if supplied_spec_path != expected_spec_path:
        raise PermissionError("submission requires the canonical campaign-spec path")
    if checkout_validator is None:
        validate_source_checkout(spec["project_dir"], expected_commit=str(spec["source_commit"]))
    else:
        checkout_validator(spec["project_dir"])
    if event_writer is None:
        raise ValueError("live submission requires an immutable event writer")
    job_ids, _, pending, events = _submission_prefix_state(
        prior_events, spec=spec, command_plan=command_plan,
    )
    reconciled = {
        require_sha256(key, name="reconciled submission intent"): str(value)
        for key, value in (reconciled_job_ids or {}).items()
    }
    if pending is not None:
        intent_hash = pending["content_hash"]
        if set(reconciled) != {intent_hash}:
            raise PermissionError(
                "submission journal ends at an unresolved intent; exact scheduler "
                "reconciliation (and no unrelated reconciliation) is required before resume"
            )
        job_id = reconciled.pop(intent_hash)
        if not SLURM_JOB_ID.fullmatch(job_id):
            raise ValueError("reconciled scheduler job ID differs")
        if reconciliation_validator is None:
            raise PermissionError(
                "unresolved submission intent requires live scheduler reconciliation"
            )
        reconciliation_validator(pending, job_id)
        submitted = _submission_event(
            spec=spec, command_plan=command_plan, sequence=len(events),
            task_sequence=int(pending["task_sequence"]), phase="submitted",
            task_key=str(pending["task_key"]), command=pending["command"],
            previous_event_sha256=intent_hash, intent_sha256=intent_hash,
            job_id=job_id,
        )
        event_writer(submitted)
        events.append(submitted)
        job_ids[str(pending["task_key"])] = job_id
    if reconciled:
        raise ValueError("reconciliation names no unresolved submission intent")

    start = len(job_ids)
    for task_sequence, row in enumerate(command_plan["commands"][start:], start=start):
        command = materialize_command(row, job_ids=job_ids)
        intent = _submission_event(
            spec=spec, command_plan=command_plan, sequence=len(events),
            task_sequence=task_sequence, phase="intent", task_key=row["task_key"],
            command=command,
            previous_event_sha256=(None if not events else events[-1]["content_hash"]),
        )
        event_writer(intent)
        events.append(intent)
        job_id = str(scheduler(command)).strip().split(";")[0]
        if not SLURM_JOB_ID.fullmatch(job_id):
            raise ValueError("scheduler returned an invalid exact job ID")
        submitted = _submission_event(
            spec=spec, command_plan=command_plan, sequence=len(events),
            task_sequence=task_sequence, phase="submitted", task_key=row["task_key"],
            command=command, previous_event_sha256=intent["content_hash"],
            intent_sha256=intent["content_hash"], job_id=job_id,
        )
        event_writer(submitted)
        events.append(submitted)
        job_ids[row["task_key"]] = job_id
    return assemble_submission_ledger_from_events(
        events, spec=spec, command_plan=command_plan,
    )


__all__ = [
    "AUTHORIZATION_PHRASE", "CAMPAIGN_SPEC_CONTRACT", "COMMAND_PLAN_CONTRACT",
    "CONTROLS", "CampaignTask",
    "DETERMINISTIC_KINDS", "LOCAL_SMOKE_CONTRACT", "MODES", "RECOVERY_LEDGER_CONTRACT",
    "REQUIRED_TIGRIS_CHECKS",
    "STRATEGIES", "SUBMISSION_AUTHORIZATION_CONTRACT", "SUBMISSION_EVENT_CONTRACT",
    "SUBMISSION_LEDGER_CONTRACT",
    "TIGRIS_ACCEPTANCE_CONTRACT", "TIGRIS_EVIDENCE_BUNDLE_CONTRACT", "TRACKS",
    "adapter_registered_input_requirements", "assemble_submission_ledger_from_events",
    "build_command_plan",
    "build_submission_authorization",
    "build_task_registry", "create_campaign_spec",
    "materialize_command", "primary_node_ids", "submit_command_plan",
    "validate_campaign_spec", "validate_submission_authorization",
    "validate_submission_event_chain",
    "validate_task_registry", "validate_tigris_acceptance",
]
