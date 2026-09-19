"""An opt-in evidence import is not a fresh GPU acceptance or a gate bypass."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hlt_classification.cms_salience_learned import (
    campaign, contracts, preflight_reuse as reuse, production,
)
from hlt_classification.cms_salience_learned.storage import fingerprint, load_receipt, publish_receipt
from hlt_classification.data.cache_contracts import canonical_sha256, load_json, sha256_file, write_immutable_json
from test_cms_salience_coarse import coarse_pair  # noqa: F401
from test_cms_salience_learned import create_from, rehash, tiny_campaign  # noqa: F401


@pytest.fixture
def reused_pair(coarse_pair, monkeypatch):
    source, v4 = coarse_pair
    monkeypatch.setattr(reuse, "execution_code", lambda *args: {"fixture": "identical"})
    spec = campaign.create_coarse_from_dense(source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
        campaign_root=Path(source["campaign_root"]).parent / "reuse_v5",
        project_dir=source["project_dir"], source_commit=source["source_commit"], reuse_dense_preflight=True)
    return source, v4, spec


def test_opt_in_import_is_cpu_only_preserves_real_evidence_and_all_science(reused_pair, monkeypatch):
    source, v4, spec = reused_pair
    before = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    assert spec["schema_version"] == 5
    assert spec["graph"] == v4["graph"] and spec["resources"] == v4["resources"]
    assert spec["acceptance_policy"] == v4["acceptance_policy"]
    assert spec["acceptance_import"]["source_slurm_job_id"] == "21720795"
    plan = campaign.command_plan(spec, "gate")
    assert not any("--gres" in arg for arg in plan["commands"][0]["command"])
    assert campaign.tasks(spec)["gate"][0]["kind"] == "import_preflight"
    assert len(campaign.tasks(spec)["science"]) == 31
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(spec)
    production.run_task(spec, "foundation", device="cpu")
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(spec)
    def forbidden(*args, **kwargs):
        pytest.fail("Evidence import must not run GPU preflight or request GPU allocation")
    monkeypatch.setattr(production, "preflight", forbidden)
    monkeypatch.setattr(production, "validate_gpu_allocation", forbidden)
    production.run_task(spec, "preflight", device="cpu")
    accepted = campaign.gate_check(spec)
    assert accepted["contract"].endswith("ACCEPTANCE_REUSE/v1")
    assert accepted["fresh_gpu_measurement"] is False
    assert accepted["source_measurements"]["peak_cuda_bytes"] == 1
    assert not (Path(spec["campaign_root"]) / "execution_acceptance.json").exists()
    assert accepted["acceptance_import"]["source_campaign_sha256"] == source["content_hash"]
    assert before == {p: sha256_file(p) for p in before}
    assert load_receipt(spec, "preflight")["outputs"] == [fingerprint(Path(spec["campaign_root"]) / "execution_acceptance_import.json")]
    # Existing v4 does NOT acquire this privilege from a later implementation.
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(v4)


def test_old_specs_cannot_gain_an_unversioned_skip(reused_pair):
    _, v4, spec = reused_pair
    with pytest.raises(ValueError, match="Legacy"):
        campaign.validate_campaign(rehash(v4, acceptance_import=spec["acceptance_import"]))
    with pytest.raises(ValueError, match="requires explicit"):
        campaign.validate_campaign(rehash(spec, acceptance_import=None))
    with pytest.raises(ValueError, match="requires explicit"):
        campaign.validate_campaign(rehash(spec, shared_source=None))


@pytest.mark.parametrize("field,value", [
    ("source_slurm_job_id", "999"), ("source_campaign_sha256", "f" * 64),
    ("fresh_gpu_measurement", True), ("measured_envelope", "unbounded"),
])
def test_changed_import_claims_rejected(reused_pair, field, value):
    _, _, spec = reused_pair
    altered = rehash(spec["acceptance_import"], **{field: value})
    with pytest.raises(ValueError, match="provenance"):
        campaign.validate_campaign(rehash(spec, acceptance_import=altered))


def test_incomplete_or_corrupt_original_gate_cannot_be_reused(reused_pair):
    source, _, spec = reused_pair
    path = Path(source["campaign_root"]) / "execution_acceptance.json"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="payload changed"):
        reuse.validate_acceptance_import(spec)
    with pytest.raises(ValueError):
        campaign.gate_check(spec)


def test_changed_runtime_still_requires_fresh_gate(reused_pair, monkeypatch):
    _, _, spec = reused_pair
    # Fixture gives both commits the same hash. Use a distinct new commit to
    # exercise the donor/consumer comparison instead of the descriptor hash.
    monkeypatch.setattr(reuse, "execution_code", lambda project, commit: {"fixture": commit})
    with pytest.raises(ValueError, match="fresh preflight"):
        reuse.build_acceptance_import(dict(spec, source_commit="c" * 40))


def test_reuse_needs_exact_consumer_receipt_and_does_not_copy_acceptance(reused_pair):
    _, _, spec = reused_pair
    production.run_task(spec, "foundation", device="cpu")
    reuse.publish_acceptance_reuse(spec)
    # Path existence without a task receipt cannot authorize science.
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(spec)
    publish_receipt(spec, "preflight", [])
    with pytest.raises(ValueError, match="receipt differs"):
        campaign.gate_check(spec)


def test_accepted_import_enables_existing_submission_gate_not_implicit_submission(reused_pair, monkeypatch):
    _, _, spec = reused_pair
    production.run_task(spec, "foundation", device="cpu")
    production.run_task(spec, "preflight", device="cpu")
    from hlt_classification.cms_salience_learned import coarse_submission
    called = []
    monkeypatch.setattr(coarse_submission, "submit_shared_dag", lambda *a, **kw: called.append(kw) or {"ok": True})
    with pytest.raises(PermissionError):
        campaign.submit(spec, "science", execute=True)
    assert not called
    assert campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.COARSE_AUTHORIZATION) == {"ok": True}
    assert called == [{"execute": True}]


def test_unchanged_real_git_runtime_and_closed_preflight_equivalence():
    root = Path(__file__).resolve().parents[1]
    old = reuse.execution_code(root, "7bb171382b7206013bc5d9308a4c22b2929bc7f4")
    coarse = reuse.execution_code(root, "48ab8609ee87ba72ab9868dfc951a36a1c9d851e")
    assert old == coarse
    node = next(n for n in ast.parse(Path(production.__file__).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "preflight")
    assert canonical_sha256(ast.dump(node, include_attributes=False)) == reuse._PREFLIGHT_CURRENT
    assert all(f"production.{name}:ast" in coarse for name in reuse._PROBE_FUNCTIONS)


def test_direct_dense_create_rejects_preflight_reuse(tiny_campaign):
    with pytest.raises(ValueError, match="coarse shared-source"):
        create_from(tiny_campaign, "invalid_skip", reuse_dense_preflight=True)
