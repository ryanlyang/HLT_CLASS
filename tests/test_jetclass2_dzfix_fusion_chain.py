"""Queue isolation, source boundary, native kernels and both HLT-only endings."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes import (
    dzfix_fusion_chain as chain, dzfix_fusion_data as data,
    dzfix_fusion_runtime as runtime, dzfix_fusion_source as source,
    dzfix_fusion_submit as scheduler,
)
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.contracts import artifact as parent_artifact
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_hcwdl_offline_hlt_fusion import _FakeWeaver


def rehash(value, **updates):
    return with_content_hash({**{k: v for k, v in value.items() if k != "content_hash"}, **updates})


def spec_at(tmp_path):
    return chain.artifact("CAMPAIGN_SPEC", **chain.registration(),
        source_commit="a"*40, campaign_root=str(tmp_path / "campaign"), project_dir=str(tmp_path / "project"),
        data_root=str(tmp_path / "data"), foundation={"content_hash": "f"*64},
        source_import={"content_hash": "b"*64}, launch_sha256="l"*64)


def make_cache(role, coordinate="D000"):
    count, length = 88, 4
    ids = np.asarray([np.frombuffer(bytes.fromhex(f"{i+(1000 if role=='train' else 0):064x}"), np.uint8)
                      for i in range(count)])
    rng = np.random.default_rng(17 if coordinate == "D000" else 18)
    features = rng.normal(size=(count*length, 17)).astype(np.float32)
    vectors = np.tile(np.asarray([1., .1, .2, 2.], np.float32), (count*length, 1))
    block = RamBlock(0, np.arange(count+1)*length, features, vectors, ids, np.arange(count) % 11)
    return RamCache([block], role=role, foundation_sha256="f"*64, coordinate_name=coordinate)


@pytest.fixture
def fake_native(monkeypatch):
    from hlt_classification.jetclass2_delphes import model, salience_learned_model
    class Fake17(_FakeWeaver):
        def __init__(self, **config):
            super().__init__(**config)
            self.embed.projection = nn.Conv1d(17, 128, 1)
    monkeypatch.setattr(model, "load_weaver_particle_transformer_class", lambda: Fake17)
    monkeypatch.setattr(salience_learned_model, "load_weaver_particle_transformer_class", lambda: Fake17)
    torch.set_num_threads(1)


def test_graph_is_requested_coarse_fusion_chain_not_withdrawal_or_cms_import():
    nodes = chain.nodes()
    assert len(nodes) == 12
    fused = [n for n in nodes if n["context_coordinate"] is not None]
    assert [(n["context_coordinate"], n["primary_coordinate"]) for n in fused] == [
        ("U000", "U050"), ("U050", "U100"), ("U100", "D066"),
        ("D066", "D033"), ("D033", "D000"), ("D000", "D000")]
    assert all(n["selection_route"] == "alpha_one" for n in fused)
    assert all(n["initialization_parent"] is None for n in nodes)
    assert not any("withdraw" in n["role"] for n in nodes)
    ends = [n for n in nodes if n["node_id"].startswith("FINAL_")]
    assert {n["teacher_distribution"] for n in ends} == {"FUSION_D000", "FUSION_D000_D000"}
    for k in ("initialization_seed", "sampler_seed"):
        assert ends[0][k] == ends[1][k]
    assert all(n["deployable"] for n in ends)
    assert not next(n for n in nodes if n["node_id"] == "FUSION_D000")["deployable"]
    assert next(n for n in nodes if n["node_id"] == "FUSION_D000_D000")["deployable"]
    assert chain.registration()["role_counts"] == {"train": 500000, "validation": 1000000, "final_test": 1000000}


def test_task_counts_debug_resources_and_exact_afterok_edges(tmp_path):
    spec = spec_at(tmp_path)
    plan = scheduler.plan(spec, "full")
    assert len(plan["commands"]) == 25
    assert len(scheduler.plan(spec, "science")["commands"]) == 21
    assert sum(r["kind"] == "train" for r in spec["tasks"]) == 12
    assert sum(r["kind"] == "reduce" for r in spec["tasks"]) == 7
    seen = set()
    for row in plan["commands"]:
        assert set(row["dependencies"]) <= seen
        seen.add(row["task_id"])
        c = row["command"]
        assert "--partition=debug" in c and "--qos=qos_tier3" in c and "--no-requeue" in c
        assert int(next(x.split("=")[1] for x in c if x.startswith("--time="))) <= 1440
        assert not any("afterany" in x or "tier3" == x.split("=")[-1] for x in c)
    rows = {r["task_id"]: r for r in spec["tasks"]}
    assert rows["train_FUSION_U050"]["dependencies"] == ["reduce_U000"]
    assert rows["train_FINAL_DIRECT_D000"]["dependencies"] == ["reduce_FUSION_D000"]
    assert rows["train_FINAL_BRIDGE_D000"]["dependencies"] == ["reduce_FUSION_D000_D000"]


def test_partition_is_stratified_50_25_25_and_tamper_closed(tmp_path):
    spec = spec_at(tmp_path)
    cache = make_cache("validation")
    codes = data.partition_codes(cache.identities, cache.labels)
    assert np.bincount(codes).tolist() == [44, 22, 22]
    for c in range(11):
        assert np.bincount(codes[cache.labels == c]).tolist() == [4, 2, 2]
    permutation = np.arange(len(cache))[::-1]
    assert np.array_equal(codes[permutation], data.partition_codes(cache.identities[permutation], cache.labels[permutation]))
    data.publish_partition(spec, cache)
    assert len(data.partition_indices(spec, cache)["checkpoint"]) == 44
    cache.identities[0, 0] ^= 1
    with pytest.raises(ValueError, match="population"):
        data.partition_indices(spec, cache)


def test_native_offline_adapter_does_not_call_matching_and_d000_does_not_load_maps(monkeypatch, tmp_path):
    from test_jetclass2_delphes import particle_values
    from hlt_classification.jetclass2_delphes.reader import Jet, Particles
    from hlt_classification.jetclass2_delphes import cache, salience_learned_cache as learned
    jet = Jet("1"*64, 1, Particles(particle_values(4)), Particles(particle_values(2)))
    calls = []
    def reader(*a, **kw):
        calls.append(kw["include_offline"])
        return iter([jet if kw["include_offline"] else Jet(jet.identity, jet.label, jet.hlt, None)])
    monkeypatch.setattr(cache, "DatasetReader", reader)
    monkeypatch.setattr(learned, "DatasetReader", reader)
    monkeypatch.setattr(cache, "load_assignments", lambda *a, **k: pytest.fail("native OFFLINE touched matching"))
    monkeypatch.setattr(learned, "load_assignments", lambda *a, **k: pytest.fail("D000 touched offline assignment"))
    tiny = {"inventory": {}, "splits": {}, "inputs": {"capacity": 16}, "candidate": "SALIENCE_PT_LINEAR"}
    row = {"file_index": 0, "path": "f.root", "role": "train", "rows": 1}
    off = cache._prepare_file((tiny, str(tmp_path), "", row, "U000"))
    hlt = learned._prepare_file((tiny, str(tmp_path), "", row, "D000"))
    assert off.offsets.tolist() == [0, 2]
    assert hlt.offsets.tolist() == [0, 4]
    assert calls == [True, False]


def test_no_final_test_cache_capability(tmp_path):
    with pytest.raises(PermissionError, match="sealed"):
        data.prepare(spec_at(tmp_path), "final_test", "D000")


def test_local_full_dag_dispatch_banks_and_both_final_endings(fake_native, tmp_path, monkeypatch):
    spec = spec_at(tmp_path)
    root = Path(spec["campaign_root"])
    root.mkdir()
    monkeypatch.setattr(runtime, "validate_campaign", lambda s: s["content_hash"])
    monkeypatch.setattr(runtime, "validate_import", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "authenticate_job", lambda *a, **k: "123")
    monkeypatch.setattr(runtime, "execution_gate", lambda *a, **k: "123")
    monkeypatch.setattr(runtime, "cache_bounds", lambda s: {"train": 1, "validation": 1})
    monkeypatch.setattr(data, "prepare", lambda s, role, name: make_cache(role, name))
    monkeypatch.setattr(runtime, "prepare", data.prepare)
    from hlt_classification.jetclass2_delphes import inventory
    monkeypatch.setattr(inventory, "verify_snapshot", lambda *a: None)
    spec["foundation"]["inventory"] = {}
    original = runtime.train_kernel
    def miniature(*a, **kw):
        report, state = original(*a, **kw, acceptance_passes=1)
        # A test-only stand-in, never an installed-Weaver/real GPU attestation.
        return rehash(report, scientific_fit=True, acceptance_only=False), state
    monkeypatch.setattr(runtime, "train_kernel", miniature)
    monkeypatch.setattr(runtime, "preflight", lambda *a: chain.artifact("ACCEPTANCE", test_only=True))
    for row in spec["tasks"]:
        result = runtime.run_task(spec, row["task_id"], device="cpu")
        assert result["final_test_accessed"] is False
        assert runtime.completed(spec, row["task_id"]) == result
    rows = runtime.result_rows(spec)
    assert len(rows) == 12 and all(r["validation"] is not None for r in rows)
    assert rows[0]["recovery"]["macro_ovr_auc"] in {None, 0.}
    for node in spec["nodes"]:
        done = runtime.completed(spec, "train_" + node["node_id"])
        report = load_json(root / done["result"]["training_report"])
        assert report["kernel_report"]["validation_history"][0]["alpha_end"] == 1.
        assert report["report_validation"]["rows"] == 22
        assert report["checkpoint_validation"]["rows"] == 44
        assert report["node"]["initialization_parent"] is None
    assert runtime.run_task(spec, "complete", device="cpu") == runtime.completed(spec, "complete")
    final = runtime.completed(spec, "train_FINAL_BRIDGE_D000")
    (root / final["result"]["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes"):
        runtime.completed(spec, "train_FINAL_BRIDGE_D000")


def test_compact_mask_forward_backward_matches_original(fake_native):
    from hlt_classification.jetclass2_delphes.salience_learned_model import DelphesAdjacentFusionParticleTransformer
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import DzfixFusionParticleTransformer
    torch.manual_seed(17)
    old = DelphesAdjacentFusionParticleTransformer(context_initialization_seed=33).eval()
    new = DzfixFusionParticleTransformer(context_initialization_seed=33).eval()
    for injection in old.injections:
        nn.init.normal_(injection.residual_projection.weight, std=.01)
    new.load_state_dict(old.state_dict())
    batch = make_cache("train").batch(np.arange(3))
    inputs = tuple(torch.from_numpy(batch[k]) for k in ("features", "vectors", "mask"))
    values = []
    for model in (old, new):
        out = model.forward_fused(*inputs, *inputs, alpha=1.).logits
        out.square().sum().backward()
        values.append(out.detach())
    torch.testing.assert_close(*values, rtol=1e-5, atol=1e-6)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)


def test_full_live_submission_and_science_without_gate_are_forbidden(monkeypatch, tmp_path):
    spec = spec_at(tmp_path)
    monkeypatch.setattr(scheduler, "validate_campaign", lambda s: s["content_hash"])
    full = scheduler.submit(spec, stage="full")
    assert full["dry_run"] and len(full["jobs"]) == 25
    with pytest.raises(PermissionError, match="Full-DAG"):
        scheduler.submit(spec, stage="full", execute=True, authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError, match="four fresh"):
        scheduler.submit(spec, stage="science", execute=True, authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError, match="authorization"):
        scheduler.submit(spec, stage="gate", execute=True)


def test_exact_submission_resolves_dependencies_and_is_idempotent(monkeypatch, tmp_path):
    spec = spec_at(tmp_path)
    monkeypatch.setattr(scheduler, "validate_campaign", lambda s: s["content_hash"])
    scheduler.submit(spec, stage="full")
    calls = []
    def sbatch(command, **kw):
        calls.append(command)
        return SimpleNamespace(stdout=str(1000+len(calls))+"\n")
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    first = scheduler.submit(spec, stage="gate", execute=True, authorization=chain.AUTHORIZE)
    second = scheduler.submit(spec, stage="gate", execute=True, authorization=chain.AUTHORIZE)
    assert first == second and len(calls) == 4
    assert "--dependency=afterok:1001" in calls[1]
    assert "--dependency=afterok:1002:1003" in calls[3]
    assert not any("${JOB_" in token for c in calls for token in c)


def test_ambiguous_sbatch_acknowledgement_cannot_duplicate(monkeypatch, tmp_path):
    spec = spec_at(tmp_path)
    monkeypatch.setattr(scheduler, "validate_campaign", lambda s: s["content_hash"])
    scheduler.submit(spec, stage="full")
    def failed(*a, **k):
        raise RuntimeError("lost ack")
    monkeypatch.setattr(scheduler.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="lost ack"):
        scheduler.submit(spec, stage="gate", execute=True, authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError, match="Ambiguous"):
        scheduler.submit(spec, stage="gate", execute=True, authorization=chain.AUTHORIZE)


def test_deferred_launcher_uses_exact_postscreen_boundary_and_no_expired_id(monkeypatch, tmp_path):
    spec = dict(content_hash="a"*64, project_dir=str(tmp_path / "project"), launch_root=str(tmp_path / "launch"),
                campaign_root=str(tmp_path / "campaign"), parent_job_id="21741416", registration=chain.registration())
    parent = {"continuation_root": str(tmp_path / "parent")}
    monkeypatch.setattr(scheduler, "validate_launch", lambda s: s["content_hash"])
    monkeypatch.setattr(scheduler, "_parent", lambda s: (parent, {}))
    pending = scheduler.launcher_plan(spec, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:21741416" in pending and "--partition=debug" in pending
    calls = []
    def sbatch(c, **kw):
        calls.append(c)
        return SimpleNamespace(stdout="12345")
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    first = scheduler.schedule(spec, execute=True, authorization=chain.AUTHORIZE)
    write_immutable_json(tmp_path / "parent/continuation_complete.json", {})
    validated = []
    monkeypatch.setattr(scheduler, "build_import", lambda s: validated.append(True))
    done = scheduler.launcher_plan(spec, "after_matching")["commands"][0]["command"]
    assert validated == [True] and not any(x.startswith("--dependency") for x in done)
    assert scheduler.schedule(spec, execute=True, authorization=chain.AUTHORIZE) == first
    assert len(calls) == 1
    def corrupt(s):
        raise ValueError("corrupt completion")
    monkeypatch.setattr(scheduler, "build_import", corrupt)
    with pytest.raises(ValueError, match="corrupt"):
        scheduler.launcher_plan(spec, "after_matching")


def test_source_parent_does_not_accept_individual_fit_or_dry_run(monkeypatch, tmp_path):
    parent = dict(content_hash="a"*64, continuation_root=str(tmp_path))
    monkeypatch.setattr(source, "validate_continuation", lambda *a, **k: "a"*64)
    write_immutable_json(tmp_path / "continuation_spec.json", parent)
    live = build_submission_ledger(campaign_spec_sha256="a"*64,
        jobs={"fit_LINEAR": "21741411"}, commands={"fit_LINEAR": ["sbatch", "a"]}, dry_run=False)
    write_immutable_json(tmp_path / "launchers/after_screen/submission_ledger.json", live)
    with pytest.raises(ValueError, match="after_screen"):
        source._parent({"continuation_spec_path": str(tmp_path / "continuation_spec.json")})


def test_preflight_cannot_be_claimed_from_local_cpu(tmp_path):
    with pytest.raises(PermissionError, match="Real installed-Weaver"):
        runtime.preflight(spec_at(tmp_path), tmp_path, "cpu")


def test_installed_weaver_compact_parity():
    pytest.importorskip("weaver")
    # Supplemental only. The remote gate repeats native execution on A100.
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import DzfixFusionParticleTransformer
    torch.set_num_threads(1)
    model = DzfixFusionParticleTransformer(context_initialization_seed=17).eval()
    raw = make_cache("validation").batch(np.arange(3))
    x = tuple(torch.from_numpy(raw[k]) for k in ("features", "vectors", "mask"))
    with torch.no_grad():
        fused = model.forward_fused(*x, *x, alpha=1.).logits
        single = model.extract_primary().eval()(*x)
    assert fused.shape == (3, 11) and torch.isfinite(fused).all()
    torch.testing.assert_close(fused, single, rtol=2e-5, atol=2e-6)


@pytest.fixture
def imported_source(tmp_path, monkeypatch):
    """Synthetic producer protocol fixtures; actual ROOT decoding tested separately."""
    root = tmp_path / "producer"
    foundation_root = root / "foundation"
    inventory = parent_artifact("INVENTORY", test_fixture=True)
    foundation = parent_artifact("SALIENCE_FOUNDATION_SPEC", inventory=inventory,
        splits={"profile": "TRAIN_500K", "role_counts": chain.COUNTS}, inputs={"capacity": 240},
        candidate="SALIENCE_PT_LINEAR")
    lock = parent_artifact("SALIENCE_FOUNDATION_LOCK", foundation_sha256=foundation["content_hash"])
    parent = parent_artifact("DZFIX_SALIENCE_CONTINUATION_SPEC", continuation_root=str(root),
        data_root=str(tmp_path / "jetclass2_20260918_dzfix"), source_commit="b"*40, project_dir=str(tmp_path / "producer_project"))
    screen = parent_artifact("SALIENCE_SCREEN_SPEC", screen_root=str(root / "screen"),
        source_commit="b"*40, data_root=parent["data_root"])
    selection = parent_artifact("SALIENCE_SELECTION_LOCK", winner=foundation["candidate"],
        winner_foundation_root=str(foundation_root), winner_foundation_sha256=foundation["content_hash"],
        screen_sha256=screen["content_hash"], final_test_accessed=False)
    profile = parent_artifact("SALIENCE_RUNTIME_PROFILE", screen_sha256=screen["content_hash"], final_test_accessed=False)
    done = parent_artifact("SALIENCE_SCREEN_COMPLETE", screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], final_test_accessed=False)
    campaign = parent_artifact("SALIENCE_CAMPAIGN_SPEC", foundation=foundation,
        source_commit=parent["source_commit"], data_root=parent["data_root"])
    dry = build_submission_ledger(campaign_spec_sha256=campaign["content_hash"], dry_run=True,
        jobs={f"task_{i}": "1" for i in range(30)}, commands={f"task_{i}": ["sbatch", str(i)] for i in range(30)})
    receipt = parent_artifact("DZFIX_SALIENCE_CONTINUATION_RECEIPT", continuation_sha256=parent["content_hash"],
        source_commit=parent["source_commit"], phase="after_screen", campaign_sha256=campaign["content_hash"],
        production_dry_ledger_sha256=dry["content_hash"], screen_complete_sha256=done["content_hash"],
        live_production=False, final_test_accessed=False)
    complete = parent_artifact("DZFIX_SALIENCE_CONTINUATION_COMPLETE", continuation_sha256=parent["content_hash"],
        after_screen_receipt_sha256=receipt["content_hash"], campaign_sha256=campaign["content_hash"],
        live_production=False, production_dry_run=True, final_test_accessed=False)
    ledger = build_submission_ledger(campaign_spec_sha256=parent["content_hash"], dry_run=False,
        jobs={"after_screen": "21741416"}, commands={"after_screen": ["sbatch", "continuation"]})
    for name, value in {
        "continuation_spec.json": parent, "continuation_complete.json": complete,
        "after_screen_receipt.json": receipt, "production/campaign_spec.json": campaign,
        "production/dry_run_submission_ledger.json": dry,
        "launchers/after_screen/submission_ledger.json": ledger,
        "screen/screen_spec.json": screen, "screen/selection_lock.json": selection,
        "screen/screen_complete.json": done, "screen/runtime_profile.json": profile,
        "foundation/foundation_spec.json": foundation, "foundation/foundation_lock.json": lock,
        "inventory.json": inventory,
    }.items():
        write_immutable_json(root / name, value)
    monkeypatch.setattr(source, "_source", lambda *a: None)
    monkeypatch.setattr(source, "validate_continuation", lambda *a, **k: parent["content_hash"])
    monkeypatch.setattr(source, "validate_inventory", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "validate_foundation_spec", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "authenticate_preparation", lambda *a: lock)
    monkeypatch.setattr(source, "_screen_artifacts", lambda *a, **k: (screen, selection, profile))
    def screen_report(_screen, name):
        value = {"result": selection if name == "select" else profile if name == "preflight" else {}}
        write_immutable_json(root / "screen/tasks" / (name + ".json"), value)
        return value
    monkeypatch.setattr(source, "screen_task_report", screen_report)
    launch = source.create_launch(continuation_spec=root / "continuation_spec.json",
        inventory_path=root / "inventory.json", launch_root=tmp_path / "launch",
        campaign_root=tmp_path / "campaign", project=tmp_path / "new_project", source_commit="a"*40)
    return launch, foundation, root


def test_completed_source_import_reuses_only_matching_and_authenticates_all_bytes(imported_source):
    launch, foundation, root = imported_source
    record, actual = source.build_import(launch)
    assert actual == foundation and record["models_imported"] == []
    assert record["consumer_commit"] != record["producer_commit"]
    assert record["selected_candidate"] == "SALIENCE_PT_LINEAR"
    assert source.validate_import(record, deep=True) == record["content_hash"]
    (root / "screen/selection_lock.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        source.validate_import(record)


def test_old_inventory_cannot_be_relabelled_as_dzfix(imported_source):
    launch, _, root = imported_source
    inventory = parent_artifact("INVENTORY", old_snapshot=True)
    write_immutable_json(root / "old_inventory.json", inventory)
    wrong = rehash(launch, inventory_path=str(root / "old_inventory.json"), inventory_sha256=inventory["content_hash"])
    with pytest.raises(ValueError, match="snapshot"):
        source.build_import(wrong)


def test_partial_completion_is_not_a_ready_dependency(imported_source):
    launch, _, root = imported_source
    (root / "continuation_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        source.build_import(launch)


def test_registration_change_cannot_reuse_acceptance(imported_source, monkeypatch):
    launch, _, _ = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    spec = chain.create(launch=launch)
    assert chain.validate_campaign(spec) == spec["content_hash"]
    assert chain.create(launch=launch) == spec
    changed = deepcopy(spec)
    changed["resources"]["train"]["cpus"] = 16
    with pytest.raises(ValueError, match="registration"):
        chain.validate_campaign(rehash(changed))


def test_launcher_automates_gate_then_science_without_resubmitting_parent(imported_source, monkeypatch):
    launch, _, _ = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    monkeypatch.setattr(scheduler, "authenticate_job", lambda *a, **k: "300")
    original_source = source.assignment_source()
    calls = []
    def sbatch(c, **kw):
        assert c[0] == "sbatch"  # Never scancel/scontrol update or another queue family.
        calls.append(c)
        return SimpleNamespace(stdout=str(1000+len(calls)))
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    # Provenance hashing uses git; keep its subprocess unaffected by fake sbatch.
    monkeypatch.setattr(source, "assignment_source", lambda: original_source)
    result = scheduler.run_launcher(launch, "after_matching")
    assert len(calls) == 5  # Four gates + the success-dependent science launcher.
    assert "--dependency=afterok:1004" in calls[-1]
    monkeypatch.setattr(runtime, "science_gate", lambda s: {"passed": True})
    result = scheduler.run_launcher(launch, "after_gate")
    assert result["result"]["science_tasks"] == 21 and len(calls) == 26
    assert all("--partition=debug" in c for c in calls)


def test_native_parity_helper_executes_real_forward_backward_adapter(fake_native):
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import native_mask_parity
    assert native_mask_parity(make_cache("train").batch(np.arange(4)), device="cpu")


def test_longest_population_stress_finds_rare_jets_not_just_padding():
    cache = make_cache("train")
    lengths = np.full(len(cache), 4, np.int64)
    lengths[7], lengths[81] = 240, 239
    cache.blocks[0].offsets = np.r_[0, np.cumsum(lengths)]
    assert runtime.longest_indices(cache, 2).tolist() == [81, 7]


def test_science_gate_rechecks_memory_and_real_execution_fields(monkeypatch, tmp_path):
    spec = spec_at(tmp_path)
    root = Path(spec["campaign_root"])
    evidence = dict(campaign_sha256=spec["content_hash"], passed=True, final_test_accessed=False,
        site=spec["execution_site"], resource=spec["resources"]["preflight"], acceptance_only=True,
        full_population_rows=spec["role_counts"], batch_size=256, checkpoint_round_trip=True,
        bank_round_trip=True, compact_mask_native_parity=True, installed_weaver_fp32_parity=True,
        worst_population_batch_stress=True, peak_cuda_bytes=800, gpu={"total_memory_bytes": 1000},
        peak_rss_bytes=1000, projected_max_fit_seconds=100,
        native_execution=[{"kernel_report": {"acceptance_only": True, "scientific_fit": False}}]*4)
    for name, changes in (("good", {}), ("memory", {"peak_cuda_bytes": 901}),
                          ("batch", {"batch_size": 128}), ("test", {"final_test_accessed": True}),
                          ("time", {"projected_max_fit_seconds": 24*3600})):
        value = chain.artifact("ACCEPTANCE", **{**evidence, **changes})
        write_immutable_json(root / (name + ".json"), value)
        monkeypatch.setattr(runtime, "completed", lambda *a: {"result": {"acceptance": name + ".json"}})
        if not changes:
            assert runtime.science_gate(spec) == value
        else:
            with pytest.raises(ValueError, match="acceptance"):
                runtime.science_gate(spec)
