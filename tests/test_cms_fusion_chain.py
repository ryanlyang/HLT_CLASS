"""Isolated fusion-to-fusion CMS chain and both single-HLT endings."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.cms_salience_learned import (
    campaign, coarse_submission as scheduler, contracts, fusion_chain as chain,
    production, training,
)
from hlt_classification.cms_salience_learned.data import Cache
from hlt_classification.cms_salience_learned.model import build_model
from hlt_classification.cms_salience_learned.storage import load_receipt
from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_cms_direct_fusion import direct_pair  # noqa: F401
from test_cms_salience_learned import make_cache, rehash, tiny_campaign  # noqa: F401
from test_hcwdl_offline_hlt_fusion import fake_weaver  # noqa: F401


@pytest.fixture
def chain_family(direct_pair, fake_weaver, monkeypatch):
    source, _ = direct_pair
    torch.set_num_threads(1)
    kernel = production.train

    def tiny_fit(*args, **kwargs):
        report, state = kernel(*args, **kwargs, acceptance_passes=1)
        # Test-only scientific artifact stand-in, never remote acceptance.
        return rehash(report, scientific_fit=True), state

    monkeypatch.setattr(production, "train", tiny_fit)
    monkeypatch.setattr(production, "validate_gpu_allocation", lambda *a: None)
    for task in contracts.SHARED_TASKS:
        production.run_task(source, task, device="cpu")
    coarse = campaign.create_coarse_from_dense(
        source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
        campaign_root=Path(source["campaign_root"]).parent / "coarse",
        project_dir=source["project_dir"], source_commit=source["source_commit"], reuse_dense_preflight=True)
    production.run_task(coarse, "foundation", device="cpu")
    production.run_task(coarse, "preflight", device="cpu")
    rows = campaign.command_plan(coarse, "science")["commands"]
    jobs = {r["task_id"]: str(21722565 + i) for i, r in enumerate(rows)}
    write_immutable_json(Path(coarse["campaign_root"]) / "science_submission_ledger.json",
        build_submission_ledger(campaign_spec_sha256=coarse["content_hash"], jobs=jobs,
            commands={r["task_id"]: scheduler._command(r, jobs, None) for r in rows}, dry_run=False))
    for task in (*contracts.SHARED_TASKS, *chain.ACQUISITION_TASKS):
        production.run_task(coarse, task, device="cpu")
    spec = campaign.create_fusion_chain_from_coarse(
        source_spec=Path(coarse["campaign_root"]) / "campaign_spec.json",
        campaign_root=Path(source["campaign_root"]).parent / "fusion_chain",
        project_dir=source["project_dir"], source_commit=source["source_commit"])
    return source, coarse, spec


def test_exact_chain_graph_seeds_endpoints_and_old_hashes():
    assert contracts.graph()["content_hash"] == "cf594cd57048ed01d90d6ef1c7a0f7088cb7bed91fbf9d9673c455e41ba4adda"
    assert contracts.graph("coarse")["content_hash"] == "3480e7ddf162756d57f55fe27ee629b3135d314ec95170b589e0a64cbef1fa03"
    g = chain.chain_graph()
    assert (g["schema_version"], g["fit_count"], g["reducer_count"], len(g["tasks"])) == (4, 12, 7, 21)
    assert g["extraction_count"] == 0 and g["fresh_fit_count"] == 7
    nodes = {n["node_id"]: n for n in g["nodes"]}
    teacher = "ACQUIRE_U050"
    for high, low in zip(contracts.COARSE_RUNG_ORDER[1:], contracts.COARSE_RUNG_ORDER[2:]):
        n = nodes["FUSION_" + low]
        assert (n["primary_coordinate"], n["context_coordinate"], n["teacher_distribution"]) == (low, high, teacher)
        assert n["role"] == "fusion_pair_kd" and n["selection_route"] == "ordinary"
        teacher = n["node_id"]
    direct, bridge = (nodes[x] for x in chain.ENDPOINTS)
    for key in ("initialization_seed", "sampler_seed", "seed_alias", "context_architecture_seed"):
        assert direct[key] == bridge[key] == nodes["DIRECT_D000"][key]
    assert direct["teacher_distribution"] == nodes["FUSION_D000_D000"]["teacher_distribution"] == "FUSION_D000"
    assert bridge["teacher_distribution"] == "FUSION_D000_D000"
    assert direct["context_coordinate"] is bridge["context_coordinate"] is None
    assert all(n["initialization_parent"] is None for n in nodes.values())
    assert not any("WITHDRAW" in n or "CARRIER" in n for n in nodes)
    seen = set()
    for task in g["tasks"]:
        assert set(task["dependencies"]) <= seen
        assert task["task_id"] not in seen
        seen.add(task["task_id"])
    pairs = {t["task_id"]: t["dependencies"] for t in g["tasks"]}
    assert pairs["train_FINAL_D000_DIRECT"] == pairs["train_FUSION_D000_D000"] == ["reduce_FUSION_D000"]
    assert pairs["train_FINAL_D000_BRIDGE"] == ["reduce_FUSION_D000_D000"]


def test_same_view_inputs_separate_branches_and_exact_ordinary_kd(fake_weaver):
    torch.set_num_threads(1)
    reference = make_cache()
    arrays = reference.views["D080"]
    cache = Cache({"D000": arrays}, reference.labels, reference.identities, "train", "f" * 64, "D000", "D000")
    n = next(n for n in chain.chain_graph()["nodes"] if n["node_id"] == "FUSION_D000_D000")
    model = build_model(n)
    ix = np.arange(7)
    batch = cache.batch(ix)
    for key in ("features", "vectors", "mask"):
        np.testing.assert_array_equal(batch["primary"][key], batch["context"][key])
    assert {id(p) for p in model.hlt_mod.parameters()}.isdisjoint(id(p) for p in model.context_mod.parameters())
    target = torch.full((len(ix), 15), 1 / 15)
    # Evaluation disables stochastic layers to compare two exact objective calls.
    model.eval()
    logits = model(*training.tensors(batch["primary"], "cpu"), *training.tensors(batch["context"], "cpu"))
    expected = training.kd_loss(logits, torch.from_numpy(batch["labels"]), target)
    actual, terms = training.batch_loss(model, batch, node=n, device="cpu", teacher=target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(terms) == {"total"}
    # The zero-initialized injection initially blocks context gradients; after
    # one optimizer update the context branch must also learn through fusion.
    optimizer = training.optimizer_for(model)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training.batch_loss(model, batch, node=n, device="cpu", teacher=target)
        loss.backward()
        optimizer.step()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.hlt_mod.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.context_mod.parameters())
    p = training.predict(model, cache, node=n, device="cpu")
    assert p.shape == (len(cache), 15) and np.isfinite(p).all()


def test_paired_final_models_start_identically(fake_weaver):
    nodes = {n["node_id"]: n for n in chain.chain_graph()["nodes"]}
    a, b = (build_model(nodes[name]) for name in chain.ENDPOINTS)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])


def test_chain_spec_debug_imports_unchanged_inputs_and_exact_dry_plan(chain_family):
    source, coarse, spec = chain_family
    assert campaign.validate_campaign(spec)
    assert spec["schema_version"] == 7 and spec["ladder"] == "fusion_chain"
    for key in ("resources", "budgets", "view_config_sha256", "split_manifest", "split_roles", "data_root"):
        assert spec[key] == source[key] == coarse[key]
    assert spec["preparation_import"]["schema_version"] == 4
    assert spec["shared_source"]["schema_version"] == spec["acceptance_import"]["schema_version"] == 3
    assert spec["acquisition_source"]["completed_only"] is True
    rows = campaign.tasks(spec)["science"]
    assert len([r for r in rows if r["kind"] == "train"]) == 7
    assert len([r for r in rows if r["kind"] == "reduce"]) == 5
    assert len([r for r in rows if r["kind"].startswith("import_")]) == 7
    assert len(rows) == 21
    kinds = {r["task_id"]: r["kind"] for stage in campaign.tasks(spec).values() for r in stage}
    for stage in ("prepare", "gate", "science"):
        for row in campaign.command_plan(spec, stage)["commands"]:
            cmd = row["command"]
            assert "--partition=debug" in cmd
            assert any(arg.startswith("--job-name=cmsfc_") for arg in cmd)
            assert "scancel" not in cmd and "scontrol" not in cmd
            if kinds[row["task_id"]].startswith("import_"):
                assert not any(arg.startswith("--gres=") for arg in cmd)
            if row["task_id"] in chain.ACQUISITION_TASKS:
                assert not any("SOURCE_JOB" in arg for arg in cmd)
    assert not (Path(spec["campaign_root"]) / "science_submission_ledger.json").exists()


@pytest.mark.parametrize("field,value", [
    ("acquisition_source", None), ("acceptance_import", None), ("shared_source", None),
    ("ladder", "direct_fusion"), ("site", contracts.site_for_partition("tier3")),
])
def test_chain_cannot_drop_sources_or_change_execution(chain_family, field, value):
    _, _, spec = chain_family
    with pytest.raises(ValueError, match="v7"):
        campaign.validate_campaign(rehash(spec, **{field: value}))


def test_wrong_schema_import_or_donor_cannot_be_silently_reused(chain_family):
    source, coarse, spec = chain_family
    for key in ("preparation_import", "shared_source", "acceptance_import"):
        value = dict(spec[key])
        version = value["schema_version"] - 1
        value = rehash(value, schema_version=version, contract=value["contract"].rsplit("/v", 1)[0] + f"/v{version}")
        with pytest.raises(ValueError):
            campaign.validate_campaign(rehash(spec, **{key: value}))
    common = dict(project_dir=spec["project_dir"], source_commit=spec["source_commit"])
    with pytest.raises(ValueError, match="coarse-v5"):
        campaign.create_fusion_chain_from_coarse(source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
            campaign_root=Path(spec["campaign_root"]).parent / "wrong_donor", **common)
    with pytest.raises(FileExistsError):
        campaign.create_fusion_chain_from_coarse(source_spec=Path(coarse["campaign_root"]) / "campaign_spec.json",
            campaign_root=spec["campaign_root"], **common)
    with pytest.raises(ValueError, match="v6"):
        campaign.validate_campaign(rehash(spec, schema_version=6,
            contract=spec["contract"].rsplit("/v", 1)[0] + "/v6"))


def test_mutated_or_missing_acquisition_payloads_rejected(chain_family, monkeypatch):
    _, coarse, spec = chain_family
    original = chain.checked_file
    def corrupt(row):
        if row["path"].endswith("probabilities/ACQUIRE_U050/train.npz") or row["path"].endswith("probabilities\\ACQUIRE_U050\\train.npz"):
            raise ValueError("Artifact payload changed")
        return original(row)
    monkeypatch.setattr(chain, "checked_file", corrupt)
    with pytest.raises(ValueError, match="payload changed"):
        campaign.validate_campaign(spec)
    monkeypatch.setattr(chain, "checked_file", original)
    real_receipt = chain.load_receipt
    def missing(s, task):
        if s["content_hash"] == coarse["content_hash"] and task == "reduce_ACQUIRE_U050":
            raise FileNotFoundError("missing acquisition receipt")
        return real_receipt(s, task)
    monkeypatch.setattr(chain, "load_receipt", missing)
    with pytest.raises(FileNotFoundError, match="acquisition receipt"):
        campaign.validate_campaign(spec)


def test_gate_import_and_authorization_never_mutate_sources(chain_family, monkeypatch):
    source, coarse, spec = chain_family
    before = {p: sha256_file(p) for s in (source, coarse) for p in Path(s["campaign_root"]).rglob("*") if p.is_file()}
    monkeypatch.setattr(production, "preflight", lambda *a: pytest.fail("No duplicate GPU gate"))
    monkeypatch.setattr(scheduler, "queue_rows", lambda *a: pytest.fail("No source queue mutation"))
    with pytest.raises(PermissionError):
        scheduler.retire_dense(spec, execute=True, authorization_phrase=scheduler.RETIRE_AUTHORIZATION)
    with pytest.raises(FileNotFoundError):
        campaign.gate_check(spec)
    for task in ("foundation", "preflight"):
        production.run_task(spec, task, device="cpu")
    evidence = campaign.gate_check(spec)
    assert evidence["fresh_gpu_measurement"] is False and evidence["schema_version"] == 3
    assert evidence["acceptance_import"]["source_slurm_job_id"] == "21720795"
    for phrase in (None, contracts.AUTHORIZATION, contracts.COARSE_AUTHORIZATION, contracts.DIRECT_FUSION_AUTHORIZATION):
        with pytest.raises(PermissionError):
            campaign.submit(spec, "science", execute=True, authorization_phrase=phrase)
    assert before == {p: sha256_file(p) for p in before}


def test_rehashed_donor_ledger_and_non_scientific_fit_rejected(chain_family, monkeypatch):
    _, coarse, spec = chain_family
    base = Path(coarse["campaign_root"])
    original_load = chain.load_json
    ledger_path = base / "science_submission_ledger.json"
    ledger = load_json(ledger_path)
    commands = {k: list(v) for k, v in ledger["commands"].items()}
    commands["train_ACQUIRE_U050"].append("--unregistered-option")
    bad = build_submission_ledger(campaign_spec_sha256=coarse["content_hash"],
        jobs=ledger["jobs"], commands=commands, dry_run=False)
    monkeypatch.setattr(chain, "load_json", lambda p: bad if Path(p) == ledger_path else original_load(p))
    with pytest.raises(ValueError, match="submitted command"):
        chain._source(spec, base / "campaign_spec.json")
    report_path = base / "training/ACQUIRE_U050/report.json"
    report = load_json(report_path)
    wrong = rehash(report, training=rehash(report["training"], scientific_fit=False))
    monkeypatch.setattr(chain, "load_json", lambda p: wrong if Path(p) == report_path else original_load(p))
    with pytest.raises(ValueError, match="scientific-fit"):
        chain.acquisition_outputs(coarse, "train_ACQUIRE_U050")


def test_complete_production_chain_banks_both_endpoints_and_reporting(chain_family, capsys):
    source, coarse, spec = chain_family
    before = {p: sha256_file(p) for s in (source, coarse) for p in Path(s["campaign_root"]).rglob("*") if p.is_file()}
    for task in ("foundation", "preflight"):
        production.run_task(spec, task, device="cpu")
    base = Path(spec["campaign_root"])
    for task in campaign.tasks(spec)["science"]:
        production.run_task(spec, task["task_id"], device="cpu")
        load_receipt(spec, task["task_id"])
    for n in spec["graph"]["nodes"][5:]:
        report = load_json(base / "training" / n["node_id"] / "report.json")
        bank = load_json(base / "probabilities" / n["teacher_distribution"] / "bank.json")
        assert report["parents"] == {"teacher_bank": bank["content_hash"]}
        assert report["node"]["initialization_parent"] is None
        assert report["training"]["validation_history"][0]["alpha_end"] == 1.
        assert set(report["training"]["validation_history"][0]["loss_terms"]) == {"total"}
        assert report["fusion_diagnostics"] == {}
    imported = load_json(base / "training/ACQUIRE_U050/report.json")
    original = load_json(Path(coarse["campaign_root"]) / "training/ACQUIRE_U050/report.json")
    assert imported["imported_from"]["artifact_sha256"] == original["content_hash"]
    assert imported["checkpoint"]["sha256"] == original["checkpoint"]["sha256"]
    bank = load_json(base / "probabilities/ACQUIRE_U050/bank.json")
    assert bank["model_report_sha256"] == imported["content_hash"]
    metrics = load_json(base / "aggregate.json")
    rows = {r["model"]: r for r in metrics["rows"]}
    assert len(rows) == 12 and len(metrics["contrasts"]) == 3
    assert rows["FUSION_D000"]["input_role"] == "privileged"
    assert rows["FUSION_D000_D000"]["input_role"] == "hlt_only"
    assert rows["FUSION_D000_D000"]["branches"] == 2
    for name in chain.ENDPOINTS:
        assert rows[name]["input_role"] == "hlt_only" and rows[name]["branches"] == 1
        model, _ = production.load_model(spec, name, "cpu")
        assert not hasattr(model, "context_mod")
    assert before == {p: sha256_file(p) for p in before}
    with pytest.raises(PermissionError):
        production.build_cache(spec, "final_test", "D000")
    chain.print_results(spec)
    printed = capsys.readouterr().out
    assert "Final test untouched" in printed and "REPORT subset" in printed
    assert "FINAL_D000_BRIDGE minus FINAL_D000_DIRECT" in printed
    assert "PER-CLASS QCD REJECTION" in printed and "dR50=" in printed


def test_science_partial_submission_resume_has_no_external_acquisition_dependency(chain_family, monkeypatch):
    _, _, spec = chain_family
    for task in ("foundation", "preflight"):
        production.run_task(spec, task, device="cpu")
    commands = []
    fail = [True]
    def submit(command, **kwargs):
        assert command[0] == "sbatch"
        if len(commands) == 3 and fail[0]:
            fail[0] = False
            raise scheduler.subprocess.CalledProcessError(1, command, stderr="temporary scheduler refusal")
        commands.append(command)
        return SimpleNamespace(stdout=str(310000 + len(commands)))
    monkeypatch.setattr(scheduler.subprocess, "run", submit)
    monkeypatch.setattr(scheduler, "queue_rows", lambda *a: pytest.fail("Completed source outputs need no scheduler lookup"))
    with pytest.raises(scheduler.subprocess.CalledProcessError):
        campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.FUSION_CHAIN_AUTHORIZATION)
    assert len(commands) == 3
    ledger = campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.FUSION_CHAIN_AUTHORIZATION)
    assert len(commands) == len(ledger["jobs"]) == 21
    assert not any("${" in a or "217225" in a or "217215" in a for c in commands for a in c)
    again = campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.FUSION_CHAIN_AUTHORIZATION)
    assert again == ledger and len(commands) == 21
    direct = ledger["commands"]["train_FINAL_D000_DIRECT"]
    bridge = ledger["commands"]["train_FUSION_D000_D000"]
    dependency = "--dependency=afterok:" + ledger["jobs"]["reduce_FUSION_D000"]
    assert dependency in direct and dependency in bridge


def test_queue_helper_only_prepares_and_submits_its_own_campaign():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/queue_cms_fusion_chain.sh").read_text()
    assert all(word not in text for word in ("scancel", "scontrol", "retire-dense", "switch_cms"))
    prepare = text.split("  prepare)", 1)[1].split("    ;;", 1)[0]
    assert "--execute" not in prepare
    assert "AUTHORIZE CMS FUSION CHAIN 500K EXACT SPEC" in text


def test_installed_weaver_same_view_chain_native_contract():
    pytest.importorskip("weaver")
    torch.set_num_threads(1)
    ref = make_cache("validation")
    cache = Cache({"D000": ref.views["D080"]}, ref.labels, ref.identities, ref.role, ref.foundation_sha256, "D000", "D000")
    n = next(n for n in chain.chain_graph()["nodes"] if n["node_id"] == "FUSION_D000_D000")
    model = build_model(n).eval()
    p = training.predict(model, cache, node=n, device="cpu")
    assert np.isfinite(p).all() and p.shape == (len(cache), 15)
    loss, terms = training.batch_loss(model, cache.batch(np.arange(4)), node=n, device="cpu",
        teacher=torch.full((4, 15), 1 / 15))
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"total"}
