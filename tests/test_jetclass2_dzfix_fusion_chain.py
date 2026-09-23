"""Queue isolation, source boundary, native kernels and both HLT-only endings."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes import (
    dzfix_fusion_chain as chain, dzfix_fusion_data as data,
    dzfix_fusion_runtime as runtime, dzfix_fusion_source as source,
    dzfix_fusion_submit as scheduler,
)
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.contracts import artifact as parent_artifact
from hlt_classification.jetclass2_delphes.inputs import input_contract
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_hcwdl_offline_hlt_fusion import _FakeWeaver


def rehash(value, **updates):
    return with_content_hash({**{k: v for k, v in value.items() if k != "content_hash"}, **updates})


def spec_at(tmp_path):
    return chain.artifact("CAMPAIGN_SPEC", **chain.registration(),
        source_commit="a"*40, campaign_root=str(tmp_path / "campaign"), project_dir=str(tmp_path / "project"),
        data_root=str(tmp_path / "data"), foundation={"content_hash": "f"*64, "inputs": {"capacity": 320}},
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


@pytest.mark.parametrize("nh,no,capacity", [(4, 2, 16), (319, 310, 320)])
def test_native_offline_adapter_does_not_call_matching_and_d000_does_not_load_maps(monkeypatch, tmp_path, nh, no, capacity):
    from test_jetclass2_delphes import particle_values
    from hlt_classification.jetclass2_delphes.reader import Jet, Particles
    from hlt_classification.jetclass2_delphes import cache, salience_learned_cache as learned
    jet = Jet("1"*64, 1, Particles(particle_values(nh)), Particles(particle_values(no)))
    calls = []
    def reader(*a, **kw):
        calls.append(kw["include_offline"])
        return iter([jet if kw["include_offline"] else Jet(jet.identity, jet.label, jet.hlt, None)])
    monkeypatch.setattr(cache, "DatasetReader", reader)
    monkeypatch.setattr(learned, "DatasetReader", reader)
    monkeypatch.setattr(cache, "load_assignments", lambda *a, **k: pytest.fail("native OFFLINE touched matching"))
    monkeypatch.setattr(learned, "load_assignments", lambda *a, **k: pytest.fail("D000 touched offline assignment"))
    tiny = {"inventory": {}, "splits": {}, "inputs": {"capacity": capacity}, "candidate": "SALIENCE_PT_LINEAR"}
    row = {"file_index": 0, "path": "f.root", "role": "train", "rows": 1}
    off = cache._prepare_file((tiny, str(tmp_path), "", row, "U000"))
    hlt = learned._prepare_file((tiny, str(tmp_path), "", row, "D000"))
    assert off.offsets.tolist() == [0, no]
    assert hlt.offsets.tolist() == [0, nh]
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
                campaign_root=str(tmp_path / "campaign"), parent_job_id="21748725", registration=chain.registration())
    parent = {"screen_root": str(tmp_path / "parent")}
    monkeypatch.setattr(scheduler, "validate_launch", lambda s: s["content_hash"])
    monkeypatch.setattr(scheduler, "_parent", lambda s: (parent, {}))
    pending = scheduler.launcher_plan(spec, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:21748725" in pending and "--partition=debug" in pending
    calls = []
    def sbatch(c, **kw):
        calls.append(c)
        return SimpleNamespace(stdout="12345")
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    first = scheduler.schedule(spec, execute=True, authorization=chain.AUTHORIZE)
    write_immutable_json(tmp_path / "parent/screen_complete.json", {})
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


@pytest.mark.parametrize("fault", ["individual_fit", "dry", "wrong_screen", "duplicate_job", "wrong_dependency"])
def test_source_parent_rejects_inexact_screen_ledgers(imported_source, fault):
    launch, _, root = imported_source
    path = root / "submission_ledger.json"
    value = load_json(path)
    if fault == "individual_fit":
        value["jobs"] = {"fit_LINEAR": "21748721"}
        value["commands"] = {"fit_LINEAR": ["sbatch", "a"]}
    elif fault == "dry":
        value["dry_run"] = True
    elif fault == "wrong_screen":
        value["campaign_spec_sha256"] = "f" * 64
    elif fault == "duplicate_job":
        value["jobs"]["complete"] = value["jobs"]["select"]
    else:
        value["commands"]["complete"] = [
            x.replace("afterok:21748724", "afterok:21741414")
            for x in value["commands"]["complete"]]
    path.unlink()
    write_immutable_json(path, rehash(value))
    with pytest.raises(ValueError, match="ledger"):
        source._parent(launch)


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
    """Exercise actual v2 screen/ledger/receipt validators with synthetic payloads."""
    from hlt_classification.jetclass2_delphes import salience_screen as screen_module
    root = tmp_path / "debug_screen"
    inventory = parent_artifact("INVENTORY", test_fixture=True, files=[
        {"max_selected_particles": {"hlt": 311, "offline": 319}},
        {"max_selected_particles": {"hlt": 200, "offline": 220}},
    ])
    splits = {"profile": "TRAIN_500K", "role_counts": chain.COUNTS}
    candidates, foundations, locks = [], {}, {}
    for name in source.REGISTRY:
        foundation_root = tmp_path / "shared_maps" / name
        value = parent_artifact("SALIENCE_FOUNDATION_SPEC", inventory=inventory,
            splits=splits, inputs=input_contract(capacity=320), candidate=name)
        lock = parent_artifact("SALIENCE_FOUNDATION_LOCK", foundation_sha256=value["content_hash"])
        write_immutable_json(foundation_root / "foundation_spec.json", value)
        write_immutable_json(foundation_root / "foundation_lock.json", lock)
        candidates.append(dict(candidate=name, foundation_root=str(foundation_root),
                               foundation_sha256=value["content_hash"]))
        foundations[name] = (value, foundation_root)
        locks[str(foundation_root)] = lock
    foundation, foundation_root = foundations[source.REGISTRY[0]]
    bottleneck_root = tmp_path / "bottleneck"
    bottleneck = parent_artifact("FOUNDATION_SPEC", inventory=inventory,
                                splits=splits, inputs=input_contract(capacity=320))
    write_immutable_json(bottleneck_root / "foundation_spec.json", bottleneck)
    foundations[screen_module.CONTEXT] = (bottleneck, bottleneck_root)
    template = parent_artifact("RUNTIME_PROFILE", execution_site=source.execution_site("sporc_a100"),
                              cpus=8, memory_mb=73728, workers=8, train_minutes=808)
    screen = parent_artifact("SALIENCE_SCREEN_SPEC", version=2, screen_root=str(root),
        source_commit="b"*40, project_dir=str(tmp_path / "producer_project"),
        data_root=str(tmp_path / "jetclass2_20260918_dzfix"),
        screen_execution_site=source.execution_site("sporc_a100_debug"),
        production_execution_site=source.execution_site("sporc_a100"),
        execution_policy=screen_module.DEBUG_SCREEN_POLICY, scientific_configuration_unchanged=True,
        execution_changes=["partition_tier3_to_debug", "bounded_debug_walltime"],
        debug_walltime_minutes=480, candidate_registry=source.REGISTRY,
        contextual_control=screen_module.CONTEXT, scientific_fit_count=4,
        coordinate="U100", final_test_accessed=False, existing_campaign_mutations=False,
        split_profile="TRAIN_500K", role_counts=chain.COUNTS, candidates=candidates,
        bottleneck_root=str(bottleneck_root), bottleneck_sha256=bottleneck["content_hash"],
        resource_template_path=str(tmp_path / "resource_template.json"),
        resource_template_sha256=template["content_hash"])
    selection = parent_artifact("SALIENCE_SELECTION_LOCK", winner=foundation["candidate"],
        winner_foundation_root=str(foundation_root), winner_foundation_sha256=foundation["content_hash"],
        screen_sha256=screen["content_hash"], final_test_accessed=False)
    profile = parent_artifact("SALIENCE_RUNTIME_PROFILE", version=2,
        screen_sha256=screen["content_hash"], source_commit=screen["source_commit"],
        execution_site=screen["production_execution_site"], screen_execution_site=screen["screen_execution_site"],
        execution_policy=screen["execution_policy"], passed=True, final_test_accessed=False)
    done = parent_artifact("SALIENCE_SCREEN_COMPLETE", screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], scientific_fit_count=4, final_test_accessed=False)
    for name, value in {
        "screen_spec.json": screen, "selection_lock.json": selection,
        "screen_complete.json": done, "runtime_profile.json": profile,
        "screen_split.json": {"test_fixture": True}, "inventory.json": inventory,
    }.items():
        write_immutable_json(root / name, value)
    monkeypatch.setattr(source, "_source", lambda *a: None)
    monkeypatch.setattr(screen_module, "_source", lambda *a: None)
    monkeypatch.setattr(screen_module, "_load_foundations", lambda *a, **k: foundations)
    monkeypatch.setattr(screen_module, "_template", lambda *a, **k: template)
    monkeypatch.setattr(source, "validate_inventory", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "validate_foundation_spec", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "authenticate_preparation", lambda f, p: locks[str(p)])
    plan = screen_module.command_plan(screen)
    jobs = {r["task_id"]: str(21748718+i) for i, r in enumerate(source.screen_task_graph())}
    commands = {}
    for row in plan["commands"]:
        command = list(row["command"])
        for name, job in jobs.items():
            command = [x.replace("${JOB_" + name + "}", job) for x in command]
        commands[row["task_id"]] = command
    ledger = build_submission_ledger(campaign_spec_sha256=screen["content_hash"], dry_run=False,
                                     jobs=jobs, commands=commands)
    write_immutable_json(root / "submission_ledger.json", ledger)
    for row in source.screen_task_graph():
        name = row["task_id"]
        if name == "select":
            value, paths = selection, ["selection_lock.json"]
        elif name == "preflight":
            value, paths = profile, ["runtime_profile.json", "screen_split.json"]
        elif name == "complete":
            value, paths = done, ["screen_complete.json"]
        else:
            path = name + "/payload.json"
            write_immutable_json(root / path, {"test_fixture": name})
            value, paths = {"checkpoint": path}, [path]
        report = parent_artifact("SALIENCE_SCREEN_TASK", task_id=name,
            screen_sha256=screen["content_hash"], source_commit=screen["source_commit"],
            result=value, outputs=[dict(path=p, sha256=sha256_file(root / p)) for p in paths],
            final_test_accessed=False)
        write_immutable_json(root / "tasks" / (name + ".json"), report)
    launch = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=tmp_path / "launch",
        campaign_root=tmp_path / "campaign", project=tmp_path / "new_project", source_commit="a"*40)
    return launch, foundation, root

def test_completed_source_import_reuses_only_matching_and_authenticates_all_bytes(imported_source):
    launch, foundation, root = imported_source
    record, actual = source.build_import(launch)
    assert actual == foundation and record["models_imported"] == []
    assert actual["inputs"]["capacity"] == 320
    assert record["consumer_commit"] != record["producer_commit"]
    assert record["selected_candidate"] == "SALIENCE_PT_LINEAR"
    assert source.validate_import(record, deep=True) == record["content_hash"]
    assert record["parent_job_id"] == "21748725"
    assert Path(record["foundation_root"]).parent == root.parent / "shared_maps"
    assert not (root / "production").exists()
    assert not any("continuation" in name for name in record)
    (root / "selection_lock.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        source.validate_import(record)


def test_old_inventory_cannot_be_relabelled_as_dzfix(imported_source):
    launch, _, root = imported_source
    inventory = parent_artifact("INVENTORY", old_snapshot=True)
    write_immutable_json(root / "old_inventory.json", inventory)
    wrong = rehash(launch, inventory_path=str(root / "old_inventory.json"), inventory_sha256=inventory["content_hash"])
    with pytest.raises(ValueError, match="snapshot"):
        source.build_import(wrong)


@pytest.mark.parametrize("maximum,capacity", [(3, 16), (225, 240), (311, 320), (320, 320), (321, 336)])
def test_population_capacity_uses_inventory_round_up_not_a_fixed_snapshot(imported_source, maximum, capacity):
    _, _, root = imported_source
    screen = load_json(root / "screen_spec.json")
    inventory = rehash(load_json(root / "inventory.json"), files=[
        {"max_selected_particles": {"hlt": maximum, "offline": maximum - 1}},
    ])
    rows = screen["candidates"] + [{"foundation_root": screen["bottleneck_root"]}]
    for row in rows:
        path = Path(row["foundation_root"]) / "foundation_spec.json"
        foundation = rehash(load_json(path), inventory=inventory, inputs=input_contract(capacity=capacity))
        path.unlink()
        write_immutable_json(path, foundation)
        row["foundation_sha256"] = foundation["content_hash"]
    screen["bottleneck_sha256"] = rows[-1]["foundation_sha256"]
    source._population(screen, inventory)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("change", [
    {"capacity": 240}, {"capacity": 304}, {"capacity": 336},
    {"truncation": "allowed"}, {"class_count": 15},
])
def test_all_foundations_reject_wrong_capacity_or_input_semantics(imported_source, index, change):
    _, _, root = imported_source
    screen = load_json(root / "screen_spec.json")
    inventory = load_json(root / "inventory.json")
    rows = screen["candidates"] + [{"foundation_root": screen["bottleneck_root"]}]
    path = Path(rows[index]["foundation_root"]) / "foundation_spec.json"
    foundation = load_json(path)
    foundation = rehash(foundation, inputs=rehash(foundation["inputs"], **change))
    path.unlink()
    write_immutable_json(path, foundation)
    if index < 3:
        rows[index]["foundation_sha256"] = foundation["content_hash"]
    else:
        screen["bottleneck_sha256"] = foundation["content_hash"]
    with pytest.raises(ValueError, match="inventory-derived capacity=320") as error:
        source._population(screen, inventory)
    assert str(path) in str(error.value)


def test_partial_completion_is_not_a_ready_dependency(imported_source):
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        source.build_import(launch)


def test_pending_screen_can_be_queued_without_any_completion_or_production_preview(imported_source):
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    (root / "tasks/complete.json").unlink()
    pending = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=root.parent / "pending_launch",
        campaign_root=root.parent / "pending_campaign", project=Path(launch["project_dir"]),
        source_commit=launch["source_commit"])
    assert pending["parent_task_id"] == "complete" and pending["parent_job_id"] == "21748725"
    command = scheduler.launcher_plan(pending, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:21748725" in command
    assert not (root / "production").exists()
    assert not Path(pending["campaign_root"]).exists()
    assert not (root.parent / "pending_launch/submissions_after_matching").exists()


def test_parent_job_is_read_from_ledger_not_hardcoded(imported_source):
    launch, _, root = imported_source
    path = root / "submission_ledger.json"
    ledger = load_json(path)
    ledger["jobs"]["complete"] = "98765432"
    path.unlink()
    write_immutable_json(path, rehash(ledger))
    second = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=root.parent / "another_launch",
        campaign_root=root.parent / "another_campaign", project=Path(launch["project_dir"]),
        source_commit=launch["source_commit"])
    (root / "screen_complete.json").unlink()
    command = scheduler.launcher_plan(second, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:98765432" in command
    with pytest.raises(ValueError, match="lineage"):
        source.validate_launch(launch)  # An existing launch may not silently retarget.


def test_cli_and_helper_expose_direct_screen_boundary(imported_source, monkeypatch, capsys):
    import runpy
    import sys
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    repo = Path(__file__).resolve().parents[1]
    cli = repo / "scripts/jetclass2_dzfix_fusion_chain.py"
    monkeypatch.setattr(sys, "argv", [str(cli), "schedule", "--spec",
                                     str(Path(launch["launch_root"]) / "launch_spec.json")])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(cli), run_name="__main__")
    assert result.value.code == 0
    printed = capsys.readouterr().out
    assert "--dependency=afterok:21748725" in printed
    assert "New campaign partition: debug" in printed
    assert str(root / "screen_spec.json") in printed
    assert not (Path(launch["launch_root"]) / "submissions_after_matching/submission_ledger.json").exists()
    helper = (repo / "scripts/queue_jetclass2_dzfix_fusion_chain.sh").read_text()
    assert "jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json" in helper
    assert '--screen-spec "${SCREEN_SPEC}"' in helper
    assert "CONT_SPEC" not in helper and "21741416" not in helper


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_launch_schema_is_not_reinterpreted(imported_source, version):
    launch, _, _ = imported_source
    old = rehash(launch, schema_version=version,
                 contract=f"JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_LAUNCH_SPEC/v{version}")
    with pytest.raises(ValueError, match="contract"):
        source.validate_launch(old)
    assert chain.artifact("SOURCE_IMPORT")["schema_version"] == 3
    assert chain.artifact("CAMPAIGN_SPEC")["schema_version"] == 5
    assert chain.artifact("ACCEPTANCE")["schema_version"] == 3
    assert chain.artifact("TRAINING_REPORT")["schema_version"] == 1


def test_old_tier3_screen_is_not_accepted_as_replacement(imported_source):
    launch, _, root = imported_source
    path = root / "screen_spec.json"
    value = rehash(load_json(path), schema_version=1,
                   contract="JETCLASS2_DELPHES_SALIENCE_SCREEN_SPEC/v1")
    path.unlink()
    write_immutable_json(path, value)
    with pytest.raises(ValueError, match="debug screen v2"):
        source._parent(launch)


def republish_screen_result(root, task, filename, value):
    """Make internally hashed synthetic evidence, to test semantic checks too."""
    path = root / filename
    path.unlink()
    write_immutable_json(path, value)
    report_path = root / "tasks" / (task + ".json")
    report = load_json(report_path)
    report["result"] = value
    for output in report["outputs"]:
        if output["path"] == filename:
            output["sha256"] = sha256_file(path)
    report_path.unlink()
    write_immutable_json(report_path, rehash(report))


@pytest.mark.parametrize("fault", ["unregistered_root", "context_control", "wrong_hash"])
def test_selection_must_name_registered_candidate_foundation(imported_source, fault):
    launch, _, root = imported_source
    selection = load_json(root / "selection_lock.json")
    if fault == "unregistered_root":
        selection["winner_foundation_root"] = str(root / "guessed_continuation/foundation")
    elif fault == "context_control":
        selection["winner"] = "BOTTLENECK_CONTEXT"
    else:
        selection["winner_foundation_sha256"] = "f" * 64
    selection = rehash(selection)
    republish_screen_result(root, "select", "selection_lock.json", selection)
    complete = rehash(load_json(root / "screen_complete.json"), selection_lock_sha256=selection["content_hash"])
    republish_screen_result(root, "complete", "screen_complete.json", complete)
    with pytest.raises(ValueError, match="registered"):
        source.build_import(launch)


def test_completion_file_without_task_attestation_is_not_ready(imported_source):
    launch, _, root = imported_source
    (root / "tasks/complete.json").unlink()
    with pytest.raises(ValueError, match="receipts"):
        source.build_import(launch)


def test_wrong_screen_completion_is_not_reusable(imported_source):
    launch, _, root = imported_source
    complete = rehash(load_json(root / "screen_complete.json"), screen_sha256="f" * 64)
    republish_screen_result(root, "complete", "screen_complete.json", complete)
    with pytest.raises(ValueError, match="completion"):
        source.build_import(launch)


def test_debug_screen_profile_cannot_redirect_consumer_or_change_producer_site(imported_source, monkeypatch):
    launch, _, root = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    spec = chain.create(launch=launch)
    assert load_json(root / "runtime_profile.json")["execution_site"]["partition"] == "tier3"
    assert spec["execution_site"]["partition"] == "debug"
    assert all("--partition=debug" in r["command"] for r in scheduler.plan(spec, "full")["commands"])
    profile = rehash(load_json(root / "runtime_profile.json"),
                     execution_site=source.execution_site("sporc_a100_debug"))
    republish_screen_result(root, "preflight", "runtime_profile.json", profile)
    with pytest.raises(ValueError, match="runtime/production site"):
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
    lengths[7], lengths[81] = 320, 319
    cache.blocks[0].offsets = np.r_[0, np.cumsum(lengths)]
    assert runtime.longest_indices(cache, 2).tolist() == [81, 7]


@pytest.mark.parametrize("paired", [False, True])
def test_preflight_stress_preserves_inputs_and_pads_to_foundation_capacity(monkeypatch, paired):
    model = nn.Linear(1, 1)
    raw = make_cache("train").batch(np.arange(2))
    expected = {k: raw[k].copy() for k in ("features", "vectors", "mask")}
    node = next(n for n in chain.nodes() if n["node_id"] == ("FUSION_U050" if paired else "U000"))
    payload = {**raw, "primary": raw, "context": raw} if paired else raw
    calls = []

    def batch_step(model, batch, **kwargs):
        views = [batch["primary"], batch["context"]] if paired else [batch]
        for view in views:
            for key, old in expected.items():
                assert view[key].shape[-1] == 320
                np.testing.assert_array_equal(view[key][..., :old.shape[-1]], old)
                assert not np.any(view[key][..., old.shape[-1]:])
        calls.append(True)
        return model.weight.sum(), {}

    monkeypatch.setattr(runtime, "_train_batch", batch_step)
    runtime._stress(model, payload, node, "cpu", 320)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="exceeds registered capacity"):
        runtime._stress(model, payload, node, "cpu", 3)


def test_cache_bounds_use_foundation_capacity_without_changing_memory_request(tmp_path):
    spec = spec_at(tmp_path)
    spec["foundation"]["assignment_tasks"] = [
        {"role": role, "rows": count // 40}
        for role, count in chain.COUNTS.items() if role != "final_test"
        for _ in range(40)
    ]
    spec["foundation"]["inputs"] = input_contract(capacity=240)
    previous = data.cache_bounds(spec)
    spec["foundation"]["inputs"] = input_contract(capacity=320)
    actual = data.cache_bounds(spec)
    assert all(actual[role] > previous[role] for role in actual)
    assert spec["resources"]["train"]["memory_mb"] == 320000
    spec["resources"]["train"]["memory_mb"] = 1
    with pytest.raises(MemoryError, match="preparation bound"):
        data.cache_bounds(spec)


def test_science_gate_rechecks_memory_and_real_execution_fields(monkeypatch, tmp_path):
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import PARITY_CHECKS, PARITY_BACKEND, PAIR_OFFLOAD_POLICY
    from hlt_classification.jetclass2_delphes.dzfix_fusion_parity import EARLY_PAIRS
    spec = spec_at(tmp_path)
    root = Path(spec["campaign_root"])
    stats = {name: dict(calls=3, saved_cuda_tensors=9, saved_cuda_bytes=4096,
                       restored_cuda_tensors=9) for name in ("context", "primary", "cross")}
    parity = [dict(passed=True, device_type="cuda", precision=precision, steps=3,
        checks=list(PARITY_CHECKS), parity_backend=PARITY_BACKEND, saved_tensor_storage=PAIR_OFFLOAD_POLICY,
        tolerance=dict(rtol=.01, atol=5e-4) if precision == "bf16" else dict(rtol=2e-5, atol=2e-6),
        offload_stats=stats) for precision in ("fp32", "bf16")]
    stress = dict(steps=3, batch_size=256, capacity=320, measurements=[{"seconds": 1.}]*3, offload_stats=stats)
    early = [chain.artifact("EARLY_PARITY", campaign_sha256=spec["content_hash"],
        primary=p, context=c, sample=dict(role="train", rows=4, file_indices=[0]*4,
            identities=[f"{i:064x}" for i in range(4)], final_test_accessed=False),
        compact_mask_native_parity=True, storage_parity=row, acceptance_only=True, final_test_accessed=False)
        for p, c in EARLY_PAIRS for row in parity]
    evidence = dict(campaign_sha256=spec["content_hash"], passed=True, final_test_accessed=False,
        site=spec["execution_site"], resource=spec["resources"]["preflight"], acceptance_only=True,
        full_population_rows=spec["role_counts"], batch_size=256, checkpoint_round_trip=True,
        bank_round_trip=True, compact_mask_native_parity=True, installed_weaver_fp32_parity=True,
        worst_population_batch_stress=True, peak_cuda_bytes=800, gpu={"total_memory_bytes": 1000},
        peak_rss_bytes=1000, projected_max_fit_seconds=100,
        saved_tensor_storage=spec["fusion"]["saved_tensor_storage"], saved_tensor_training_parity=parity,
        early_parity_reports=early,
        native_execution=[{"node": {"context_coordinate": "U000"}, "stress": stress,
                           "kernel_report": {"acceptance_only": True, "scientific_fit": False}}]*4)
    for name, changes in (("good", {}), ("memory", {"peak_cuda_bytes": 901}),
                          ("batch", {"batch_size": 128}), ("test", {"final_test_accessed": True}),
                          ("time", {"projected_max_fit_seconds": 24*3600}),
                          ("offload", {"saved_tensor_storage": {}}),
                          ("parity", {"saved_tensor_training_parity": []}),
                          ("early", {"early_parity_reports": []}),
                          ("backend", {"saved_tensor_training_parity": [{**r, "parity_backend": {}} for r in parity]})):
        value = chain.artifact("ACCEPTANCE", **{**evidence, **changes})
        write_immutable_json(root / (name + ".json"), value)
        monkeypatch.setattr(runtime, "completed", lambda *a: {"result": {"acceptance": name + ".json"}})
        if not changes:
            assert runtime.science_gate(spec) == value
        else:
            with pytest.raises(ValueError, match="acceptance"):
                runtime.science_gate(spec)


def test_noop_offload_cannot_be_claimed_as_new_acceptance():
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import validate_offload_stats
    good = {name: dict(calls=3, saved_cuda_tensors=12, saved_cuda_bytes=100,
                      restored_cuda_tensors=12) for name in ("context", "primary", "cross")}
    validate_offload_stats(good, calls=3)
    for key in ("calls", "saved_cuda_tensors", "saved_cuda_bytes", "restored_cuda_tensors"):
        wrong = deepcopy(good)
        wrong["cross"][key] = 0
        with pytest.raises(ValueError, match="not exercised"):
            validate_offload_stats(wrong, calls=3)
    with pytest.raises(ValueError, match="sites"):
        validate_offload_stats(None, calls=3)


def test_new_storage_policy_is_locked_without_science_resource_changes():
    from hlt_classification.jetclass2_delphes.dzfix_fusion_model import PAIR_OFFLOAD_POLICY
    spec = chain.registration()
    assert spec["fusion"]["saved_tensor_storage"] == PAIR_OFFLOAD_POLICY
    assert spec["training"]["batch_size"] == 256
    assert spec["gpu_peak_fraction_limit"] == .90
    assert spec["fusion"]["pair_population"] == "full_combined_weaver"
    assert spec["resources"]["train"]["memory_mb"] == 320000
    assert spec["execution_site"]["partition"] == "debug"
    spec["fusion"]["saved_tensor_storage"]["scope"].clear()
    assert PAIR_OFFLOAD_POLICY["scope"] == ["context", "primary", "cross"]
