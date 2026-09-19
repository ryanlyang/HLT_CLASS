"""Parallel one-arrow Strategy B: no retrained controls or source cancellation."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.cms_salience_learned import (
    campaign, coarse_submission as scheduler, contracts, preparation_import as preparation,
    preflight_reuse as reuse, production, shared_import as shared,
)
from hlt_classification.cms_salience_learned.storage import load_receipt
from hlt_classification.data.cache_contracts import canonical_sha256, load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_cms_salience_coarse import gate_fixture
from test_cms_salience_learned import create_from, rehash, tiny_campaign  # noqa: F401
from test_hcwdl_offline_hlt_fusion import fake_weaver  # noqa: F401


@pytest.fixture
def direct_pair(tiny_campaign, monkeypatch):
    source = create_from(tiny_campaign, "debug_source", partition="debug")
    for task in campaign.tasks(source)["prepare"]:
        production.run_task(source, task["task_id"], device="cpu")
    gate_fixture(source)  # Synthetic validator evidence, NOT a real GPU claim.
    rows = campaign.command_plan(source, "science")["commands"]
    jobs = {r["task_id"]: str(21721546 + i) for i, r in enumerate(rows)}
    ledger = build_submission_ledger(campaign_spec_sha256=source["content_hash"], jobs=jobs,
        commands={r["task_id"]: _resolved(r, jobs) for r in rows}, dry_run=False)
    write_immutable_json(Path(source["campaign_root"]) / "science_submission_ledger.json", ledger)
    monkeypatch.setattr(preparation, "preparation_code", lambda *a: {"fixture": "same"})
    monkeypatch.setattr(shared, "reference_code", lambda *a: {"fixture": "same"})
    monkeypatch.setattr(reuse, "execution_code", lambda *a: {"fixture": "same"})
    study = campaign.create_direct_fusion_from_dense(
        source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
        campaign_root=Path(source["campaign_root"]).parent / "direct_fusion",
        project_dir=source["project_dir"], source_commit=source["source_commit"])
    return source, study


def test_direct_graph_is_matched_one_transition_without_changing_old_hashes():
    assert contracts.graph()["content_hash"] == "cf594cd57048ed01d90d6ef1c7a0f7088cb7bed91fbf9d9673c455e41ba4adda"
    assert contracts.graph("coarse")["content_hash"] == "3480e7ddf162756d57f55fe27ee629b3135d314ec95170b589e0a64cbef1fa03"
    graph = contracts.graph("direct_fusion")
    assert graph["schema_version"] == 3 and graph["rung_order"] == ["U000", "D000"]
    assert (graph["fit_count"], graph["extraction_count"], graph["reducer_count"]) == (6, 1, 2)
    assert len(graph["tasks"]) == 11
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    acq, wd, direct = (nodes[n] for n in ("ACQUIRE_D000", "WITHDRAW_D000", "DIRECT_D000"))
    assert acq["primary_coordinate"] == "D000" and acq["context_coordinate"] == "U000"
    assert acq["teacher_distribution"] == "U000" and acq["initialization_parent"] is None
    assert wd["teacher_distribution"] == wd["initialization_parent"] == "ACQUIRE_D000"
    assert wd["selection_route"] == "alpha_zero"
    for key in ("seed_alias", "initialization_seed", "sampler_seed"):
        assert acq[key] == wd[key] == direct[key]
    for n in contracts.graph()["nodes"][:4]:
        assert nodes[n["node_id"]] == n
    seen = set()
    for t in graph["tasks"]:
        assert set(t["dependencies"]) <= seen
        assert t["task_id"] not in seen
        seen.add(t["task_id"])


def test_direct_spec_is_isolated_debug_two_fits_and_cpu_imports(direct_pair):
    source, spec = direct_pair
    assert spec["schema_version"] == 6
    assert spec["resources"] == source["resources"]
    assert spec["view_config_sha256"] == source["view_config_sha256"]
    assert spec["split_manifest"] == source["split_manifest"]
    assert spec["preparation_import"]["schema_version"] == 3
    assert spec["shared_source"]["schema_version"] == 2
    assert spec["acceptance_import"]["schema_version"] == 2
    assert campaign.validate_campaign(spec)
    rows = campaign.tasks(spec)["science"]
    assert [t["task_id"] for t in rows if t["kind"] == "train"] == ["train_ACQUIRE_D000", "train_WITHDRAW_D000"]
    assert [t["task_id"] for t in rows if t["kind"] == "import_shared"] == list(contracts.SHARED_TASKS)
    kinds = {t["task_id"]: t["kind"] for v in campaign.tasks(spec).values() for t in v}
    for stage in ("prepare", "gate", "science"):
        for row in campaign.command_plan(spec, stage)["commands"]:
            cmd = row["command"]
            assert "--partition=debug" in cmd
            if kinds[row["task_id"]] == "train":
                assert "--time=1440" in cmd
            assert any(a.startswith("--job-name=cmsdf_") for a in cmd)
            assert not any("scancel" in a or "retire-dense" in a for a in cmd)
            if kinds[row["task_id"]].startswith("import_"):
                assert not any(a.startswith("--gres=") for a in cmd)
                assert any(a.startswith("--job-name=cmsdf_import_") for a in cmd)
        ledger = load_json(Path(spec["campaign_root"]) / f"{stage}_dry_run_submission_ledger.json")
        assert ledger["dry_run"] is True
    assert not (Path(spec["campaign_root"]) / "science_submission_ledger.json").exists()


@pytest.mark.parametrize("field,value", [
    ("shared_source", None), ("acceptance_import", None), ("preparation_import", None),
    ("ladder", "coarse"), ("site", contracts.site_for_partition("tier3")),
])
def test_direct_cannot_drop_required_imports_or_change_site(direct_pair, field, value):
    _, spec = direct_pair
    with pytest.raises(ValueError, match="v6"):
        campaign.validate_campaign(rehash(spec, **{field: value}))


@pytest.mark.parametrize("field,version", [("preparation_import", 2), ("shared_source", 1), ("acceptance_import", 1)])
def test_direct_rejects_coarse_import_descriptors(direct_pair, field, version):
    _, spec = direct_pair
    changed = dict(spec[field])
    changed["contract"] = changed["contract"].rsplit("/v", 1)[0] + f"/v{version}"
    changed["schema_version"] = version
    with pytest.raises(ValueError):
        campaign.validate_campaign(rehash(spec, **{field: rehash(changed)}))


def test_creator_rejects_wrong_donor_existing_root_and_missing_reuse(direct_pair):
    source, spec = direct_pair
    common = dict(project_dir=spec["project_dir"], source_commit=spec["source_commit"])
    with pytest.raises(FileExistsError):
        campaign.create_direct_fusion_from_dense(source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
            campaign_root=spec["campaign_root"], **common)
    with pytest.raises(ValueError, match="original accepted"):
        campaign.create_direct_fusion_from_dense(source_spec=Path(spec["campaign_root"]) / "campaign_spec.json",
            campaign_root=Path(spec["campaign_root"]).parent / "bad_donor", **common)
    with pytest.raises(ValueError, match="all accepted"):
        create_from(source, "bad_reuse", partition="debug", ladder="direct_fusion")


def test_direct_reuses_gate_but_cannot_cancel_or_submit_without_authorization(direct_pair, monkeypatch):
    source, spec = direct_pair
    before = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("No source job mutation, new GPU preflight or premature submission")
    monkeypatch.setattr(scheduler, "queue_rows", forbidden)
    monkeypatch.setattr(production, "preflight", forbidden)
    monkeypatch.setattr(production, "validate_gpu_allocation", forbidden)
    with pytest.raises(PermissionError, match="parallel studies"):
        scheduler.retire_dense(spec, execute=True, authorization_phrase=scheduler.RETIRE_AUTHORIZATION)
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(spec)
    production.run_task(spec, "foundation", device="cpu")
    production.run_task(spec, "preflight", device="cpu")
    accepted = campaign.gate_check(spec)
    assert accepted["schema_version"] == 2 and accepted["fresh_gpu_measurement"] is False
    assert accepted["acceptance_import"]["source_slurm_job_id"] == "21720795"
    assert not (Path(spec["campaign_root"]) / "execution_acceptance.json").exists()
    assert before == {p: sha256_file(p) for p in before}
    called = []
    monkeypatch.setattr(scheduler, "submit_shared_dag", lambda *a, **kw: called.append(kw) or {})
    for phrase in (None, contracts.AUTHORIZATION, contracts.COARSE_AUTHORIZATION):
        with pytest.raises(PermissionError):
            campaign.submit(spec, "science", execute=True, authorization_phrase=phrase)
    assert not called
    campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.DIRECT_FUSION_AUTHORIZATION)
    assert called == [{"execute": True}]


def test_direct_production_chain_shared_outputs_extraction_and_reporting(direct_pair, fake_weaver, monkeypatch, capsys):
    source, spec = direct_pair
    torch.set_num_threads(1)
    kernel = production.train
    monkeypatch.setattr(production, "train", lambda *a, **kw: kernel(*a, **kw, acceptance_passes=1))
    monkeypatch.setattr(production, "validate_gpu_allocation", lambda *a: None)
    for task in contracts.SHARED_TASKS:
        production.run_task(source, task, device="cpu")
    before = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    production.run_task(spec, "foundation", device="cpu")
    production.run_task(spec, "preflight", device="cpu")
    for task in campaign.tasks(spec)["science"]:
        production.run_task(spec, task["task_id"], device="cpu")
        load_receipt(spec, task["task_id"])
    base = Path(spec["campaign_root"])
    acq = load_json(base / "training/ACQUIRE_D000/report.json")
    withdrawal = load_json(base / "training/WITHDRAW_D000/report.json")
    assert withdrawal["parents"]["initialization"] == acq["content_hash"]
    bank = load_json(base / "probabilities/ACQUIRE_D000/bank.json")
    assert withdrawal["parents"]["teacher_bank"] == bank["content_hash"]
    carrier, report = production.load_model(spec, "CARRIER_D000", "cpu")
    assert report["exact_extraction"] and report["context_parameters_removed"]
    assert not any("context" in name for name in carrier.state_dict())
    cache = production.build_cache(spec, "validation", "D000")
    fused, _ = production.load_model(spec, "WITHDRAW_D000", "cpu")
    np.testing.assert_array_equal(
        production.predict(fused, cache, node=withdrawal["node"], device="cpu"),
        production.predict(carrier, cache, node=report["node"], device="cpu"))
    assert before == {p: sha256_file(p) for p in before}
    aggregate = load_json(base / "aggregate.json")
    assert [r["model"] for r in aggregate["rows"]] == ["M0HLT", "OFFLINE", "U000", "DIRECT_D000", "CARRIER_D000"]
    assert aggregate["final_test_accessed"] is False
    with pytest.raises(PermissionError):
        production.build_cache(spec, "final_test", "D000")
    # Completion artifacts, not aged-out Slurm IDs, satisfy source imports.
    commands = []
    def submit(command, **kwargs):
        assert command[0] == "sbatch"
        commands.append(command)
        return SimpleNamespace(stdout=str(300000 + len(commands)))
    monkeypatch.setattr(scheduler.subprocess, "run", submit)
    monkeypatch.setattr(scheduler, "queue_rows", lambda *a: pytest.fail("Completed parents need no Slurm lookup"))
    ledger = campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.DIRECT_FUSION_AUTHORIZATION)
    assert len(commands) == 11
    for command in commands:
        assert not any("217215" in arg or "${" in arg for arg in command)
    assert campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.DIRECT_FUSION_AUTHORIZATION) == ledger
    assert len(commands) == 11
    import importlib.util
    import sys
    cli_path = Path(__file__).resolve().parents[1] / "scripts/cms_salience_learned.py"
    loader = importlib.util.spec_from_file_location("cms_direct_test_cli", cli_path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(cli_path), "results", "--spec", str(base / "campaign_spec.json")])
    assert module.main() == 0
    printed = capsys.readouterr().out
    assert "Fusion minus ordinary KD:" in printed and "not compute matched" in printed


def test_pinned_science_kernels_still_match_accepted_runtime_in_working_files():
    root = Path(__file__).resolve().parents[1]
    accepted = reuse.execution_code(root, "7bb171382b7206013bc5d9308a4c22b2929bc7f4")
    current = reuse.execution_code(root, "7cf690301d66b1d02d98b450d20aababf296429a")
    assert accepted == current
    for key, digest in current.items():
        if not key.endswith(":ast"):
            continue
        module, name = key.removesuffix(":ast").split(".")
        tree = ast.parse((root / f"src/hlt_classification/cms_salience_learned/{module}.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        working = canonical_sha256(ast.dump(fn, include_attributes=False))
        assert working == digest


def test_side_study_helper_has_no_retirement_or_scheduler_mutation():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/queue_cms_direct_fusion.sh").read_text()
    assert "retire-dense" not in text and "scancel" not in text and "scontrol" not in text
    assert "switch_cms_salience_coarse.sh" not in text
    prepare = text.split("  prepare)", 1)[1].split("    ;;", 1)[0]
    assert "--execute" not in prepare
