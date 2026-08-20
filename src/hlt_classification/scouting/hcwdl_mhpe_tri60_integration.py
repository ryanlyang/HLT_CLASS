"""Authenticate the merged TRI60 source and immutable all-mapped foundation."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any, Final, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
)

from .engine import validate_pmard_training_report
from .hcwdl_mhpe_tri60_contracts import (
    ENDPOINT_RESOURCE_LOCK_CONTRACT, FOUNDATION_LOCK_CONTRACT,
    INTEGRATION_LOCK_CONTRACT, artifact, hashes, validate_artifact,
)
from .hcwdl_mhpe_tri60_graph import REPRESENTATION_SOURCE_COMMIT
from .hcwdl_representation_kernels import generate_spectral_resource_bundle
from .hcwdl_unified_balanced_full_campaign import validate_foundation_campaign
from .hcwdl_unified_balanced_full_contracts import validate_foundation_lock


DONOR_BLOBS: Final = {
    "src/hlt_classification/models/hcwdl_representation.py": "f7e3f076040f3fe38aba8f176628ab3e90527b80",
    "src/hlt_classification/models/hcwdl_surfaces.py": "41b234bb63ce99879dce4999b405073729118d26",
    "src/hlt_classification/scouting/hcwdl_representation_artifacts.py": "b39f1adfb9524abbf43f5a574156c22c9869629d",
    "src/hlt_classification/scouting/hcwdl_representation_calibration.py": "c8ce3388b2a7aa1a20376872ddec8eb05ed8f55c",
    "src/hlt_classification/scouting/hcwdl_representation_contracts.py": "cc017840df5fc92d8ee25c8bc0d8f9dfd283182b",
    "src/hlt_classification/scouting/hcwdl_representation_data.py": "be43d3b47ac11c154a0961f47c62acd616c21d1b",
    "src/hlt_classification/scouting/hcwdl_representation_kernels.py": "bce5ebdda53c327be46ca84b184156f051598575",
    "src/hlt_classification/scouting/hcwdl_representation_losses.py": "515d2a55d368237377c4b6f0ebd60a4fb32ed06f",
    "src/hlt_classification/scouting/hcwdl_representation_recipe.py": "024b5ef4da31f38d78d973aaa787264aef5d02c4",
    "src/hlt_classification/scouting/hcwdl_representation_target_runtime.py": "885da5487cfceae793f926816420f8e3ff75e44f",
    "src/hlt_classification/scouting/hcwdl_representation_targets.py": "174827d3bb3450e73758c08fdc21bf27a4e8889f",
    "src/hlt_classification/scouting/hcwdl_representation_training.py": "0f47ef63eb7fae16750fc8936dc6c8b07e2194c0",
}

SEMANTIC_SOURCE_FILES: Final = tuple(DONOR_BLOBS) + (
    "src/hlt_classification/models/scouting_particle_transformer.py",
    "src/hlt_classification/scouting/dataset.py",
    "src/hlt_classification/scouting/hcwdl_homotopy.py",
    "src/hlt_classification/scouting/hcwdl_homotopy_stream.py",
    "src/hlt_classification/scouting/hcwdl_training.py",
    "src/hlt_classification/scouting/pmard_stream.py",
    "src/hlt_classification/scouting/view_cache.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_contracts.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_graph.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_recipe.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_probability.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_ephemeral.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_training.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_acceptance.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_integration.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_runner.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_campaign.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_workflow.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_recovery.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_operations.py",
    "src/hlt_classification/scouting/hcwdl_mhpe_tri60_reporting.py",
    "scripts/create_hcwdl_mhpe_tri60_campaign.py",
    "scripts/submit_hcwdl_mhpe_tri60_campaign.py",
    "scripts/run_hcwdl_mhpe_tri60_task.py",
    "scripts/create_hcwdl_mhpe_tri60_recovery.py",
    "scripts/run_hcwdl_mhpe_tri60_recovery_task.py",
    "scripts/submit_hcwdl_mhpe_tri60_recovery.py",
    "scripts/monitor_hcwdl_mhpe_tri60.py",
    "scripts/cancel_hcwdl_mhpe_tri60.py",
    "scripts/run_hcwdl_mhpe_tri60_acceptance.py",
    "scripts/validate_hcwdl_mhpe_tri60_source.py",
    "sbatch/run_hcwdl_mhpe_tri60_task.sh",
    "sbatch/run_hcwdl_mhpe_tri60_recovery_task.sh",
    "sbatch/run_hcwdl_mhpe_tri60_acceptance.sh",
)


def semantic_source_hashes(project_dir: str | Path) -> dict[str, str]:
    root = Path(project_dir).resolve()
    missing = [name for name in SEMANTIC_SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"TRI60 semantic source files are absent: {missing}")
    return {name: sha256_file(root / name) for name in SEMANTIC_SOURCE_FILES}


def _commit_blobs(project_dir: Path, commit: str) -> dict[str, str]:
    command = ["git", "-C", str(project_dir), "ls-tree", "-r", commit, "--", *DONOR_BLOBS]
    rows = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    result = {}
    for row in rows:
        left, name = row.split("\t", 1)
        mode, kind, digest = left.split()
        if mode != "100644" or kind != "blob":
            raise ValueError("TRI60 representation source entry is not a regular blob")
        result[name] = digest
    if set(result) != set(DONOR_BLOBS):
        raise ValueError("TRI60 representation source blob registry differs")
    return result


def build_integration_lock(
    *, project_dir: str | Path, source_commit: str,
    test_evidence_sha256: str, installed_weaver_parity_sha256: str,
) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 source commit must be a full lowercase Git commit")
    current = _commit_blobs(project, source_commit)
    if current != DONOR_BLOBS:
        changed = sorted(name for name in DONOR_BLOBS if current.get(name) != DONOR_BLOBS[name])
        raise ValueError(f"TRI60 exact representation donor blobs differ: {changed}")
    return artifact({
        "parents": hashes({
            "test_evidence": test_evidence_sha256,
            "installed_weaver_parity": installed_weaver_parity_sha256,
        }),
        "source_commit": source_commit,
        "representation_source_commit": REPRESENTATION_SOURCE_COMMIT,
        "exact_donor_blobs": dict(DONOR_BLOBS),
        "semantic_source_sha256": semantic_source_hashes(project),
        "runtime_sibling_worktree_imports": False,
        "exact_v5_representation_math": True,
        "legacy_logit_probability_math_reused": True,
        "final_test_accessed": False,
    }, contract=INTEGRATION_LOCK_CONTRACT)


def authenticate_foundation(foundation_lock_path: str | Path) -> dict[str, Any]:
    lock_path = Path(foundation_lock_path).resolve()
    lock = load_json(lock_path)
    lock_hash = validate_foundation_lock(lock)
    foundation_root = lock_path.parent.parent
    spec_path = foundation_root / "foundation_spec.json"
    spec = load_json(spec_path)
    spec_hash = validate_foundation_campaign(
        spec, executable=False, verify_source_tree=False,
    )
    if lock.get("foundation_spec_sha256") != spec_hash:
        raise ValueError("TRI60 foundation lock/spec differs")
    role_counts = {name: int(spec["role_counts"][name]) for name in (
        "train", "validation", "final_test",
    )}
    if role_counts["train"] < 2_000_000 or role_counts["validation"] < 900_000:
        raise ValueError("TRI60 foundation is not the all-mapped population")
    if spec.get("final_test_accessed") is not False:
        raise PermissionError("TRI60 foundation reports final-test access")
    m0_path = foundation_root / "training/M0paired/training_report.json"
    m0 = load_json(m0_path)
    m0_hash = validate_pmard_training_report(m0)
    m0_checkpoint = m0_path.parent / str(m0["selected_checkpoint"])
    if (
        not m0_checkpoint.is_file()
        or sha256_file(m0_checkpoint) != m0["selected_checkpoint_sha256"]
        or lock.get("m0paired_report_sha256") != m0_hash
        or lock.get("m0paired_checkpoint_sha256") != m0["selected_checkpoint_sha256"]
    ):
        raise ValueError("TRI60 contextual M0paired lineage differs")
    parent_hashes = {
        "foundation_lock": lock_hash,
        "foundation_spec": spec_hash,
        "split_manifest": spec["parents"]["split_manifest_sha256"],
        "selection_manifest": spec["parents"]["selection_manifest_sha256"],
        "assignment_lock": lock["parents"]["assignment_lock_sha256"],
        "coupling_lock": lock["parents"]["coupling_lock_sha256"],
        "endpoint_lock": lock["parents"]["endpoint_lock_sha256"],
        "train_balanced_manifest": lock["parents"]["train_balanced_manifest_sha256"],
        "validation_balanced_manifest": lock["parents"]["validation_balanced_manifest_sha256"],
        "m0paired_report": m0_hash,
        "m0paired_checkpoint": m0["selected_checkpoint_sha256"],
    }
    return artifact({
        "parents": hashes(parent_hashes),
        "foundation_spec_path": str(spec_path.resolve()),
        "foundation_lock_path": str(lock_path),
        "role_counts": role_counts,
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "contextual_m0paired_report_path": str(m0_path.resolve()),
        "contextual_m0paired_pass_count": int(m0.get("completed_natural_population_passes", 0)),
        "final_test_accessed": False,
    }, contract=FOUNDATION_LOCK_CONTRACT)


def validate_tri60_foundation_lock(value: Mapping[str, Any]) -> str:
    digest = validate_artifact(value, contract=FOUNDATION_LOCK_CONTRACT)
    if (
        value.get("ordinary_access_roles") != ["train", "validation"]
        or value.get("ordinary_final_test_capability") is not False
        or value.get("population_policy") != "all_authenticated_mapped_rows_v1"
        or value.get("final_test_accessed") is not False
    ):
        raise PermissionError("TRI60 foundation capability differs")
    hashes(value.get("parents", {}))
    return digest


def build_endpoint_resource_lock(*, parents: Mapping[str, str]) -> dict[str, Any]:
    bundle = generate_spectral_resource_bundle()
    # Resource generation is deterministic and tiny.  Training jobs regenerate
    # these arrays in RAM and require this exact logical identity.
    from hlt_classification.models.scouting_particle_transformer import (
        build_scouting_particle_transformer,
    )
    import torch

    model = build_scouting_particle_transformer().eval()
    features = torch.zeros(2, 21, 3)
    vectors = torch.ones(2, 4, 3)
    mask = torch.ones(2, 1, 3, dtype=torch.bool)
    visible = torch.arange(3).expand(2, 3)
    family = torch.zeros(2, 3, dtype=torch.int8)
    with torch.inference_mode():
        logits = model(features, vectors, mask)
        surfaces = model.forward_hcwdl_surfaces(features, vectors, mask, visible, family)
    if not torch.equal(logits, surfaces.logits):
        raise RuntimeError("TRI60 ordinary one-forward surface logits differ")
    return artifact({
        "parents": hashes(parents),
        "spectral_resource_sha256": bundle.content_hash,
        "token_resource_sha256": bundle.token.content_hash,
        "relation_resource_sha256": bundle.relation.content_hash,
        "ordinary_surface_logit_byte_parity": True,
        "canonical_input_channels": 21,
        "canonical_token_limit": 200,
        "endpoint_policy": "unified_balanced_exact_endpoints_v1",
        "kernel_arrays_persisted": False,
        "final_test_accessed": False,
    }, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT)


__all__ = [
    "DONOR_BLOBS", "SEMANTIC_SOURCE_FILES", "authenticate_foundation",
    "build_endpoint_resource_lock", "build_integration_lock",
    "semantic_source_hashes", "validate_tri60_foundation_lock",
]
