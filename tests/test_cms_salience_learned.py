from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from hlt_classification.cms_salience_learned import campaign, contracts, data, production, training
from hlt_classification.cms_salience_learned.model import CMSFusion, build_model
from hlt_classification.cms_salience_learned.storage import arrays_from, load_receipt, publish_npz, publish_receipt
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from test_hcwdl_offline_hlt_fusion import fake_weaver  # noqa: F401
from test_hcwdl_homotopy import _raw_arrays, _repeated_raw_arrays, _scale


def test_graph_is_only_dense_with_shared_references_and_direct_control():
    graph = contracts.graph()
    assert graph["fit_count"] == 20
    assert graph["extraction_count"] == 8
    assert graph["reducer_count"] == 16
    assert len(graph["tasks"]) == 46
    seen = set()
    for task in graph["tasks"]:
        assert set(task["dependencies"]) <= seen
        assert task["task_id"] not in seen
        seen.add(task["task_id"])
    for higher, lower in zip(contracts.RUNG_ORDER, contracts.RUNG_ORDER[1:]):
        acq = next(n for n in graph["nodes"] if n["node_id"] == "ACQUIRE_" + lower)
        withdrawal = next(n for n in graph["nodes"] if n["node_id"] == "WITHDRAW_" + lower)
        assert acq["context_coordinate"] == higher
        assert acq["initialization_parent"] is None
        assert acq["teacher_distribution"] == ("U000" if higher == "U000" else "CARRIER_" + higher)
        assert withdrawal["initialization_parent"] == acq["node_id"]
        assert withdrawal["teacher_distribution"] == acq["node_id"]
        assert withdrawal["selection_route"] == "alpha_zero"
        assert withdrawal["initialization_seed"] == acq["initialization_seed"]
    assert contracts.coordinate("U033").s == 1 / 3
    assert contracts.coordinate("U066").s == 2 / 3
    assert contracts.coordinate("D080").f == .2


def test_schedule_matches_registered_floor_tail_and_zero_gate():
    assert contracts.learning_rate(3) == 3e-4
    assert contracts.learning_rate(45) == 3e-4
    assert contracts.learning_rate(60) == 1.5e-5
    assert contracts.learning_rate(100) == 1.5e-5
    assert contracts.alpha_for_pass(10) == 1
    assert contracts.alpha_for_pass(35) == pytest.approx(.5)
    assert contracts.alpha_for_pass(60) == 0
    assert contracts.alpha_for_pass(61) == 0
    with pytest.raises(ValueError):
        contracts.coordinate("U050")


def make_cache(role="train", paired=True):
    rng = np.random.default_rng(5)
    count = 30
    views = {name: (rng.normal(size=(count, 21, 4)).astype(np.float32),
                   rng.normal(size=(count, 4, 4)).astype(np.float32),
                   np.ones((count, 1, 4), bool)) for name in ("D080", "U100")}
    ids = np.arange(count * 32, dtype=np.uint8).reshape(count, 32)
    return data.Cache(views, np.arange(count) % 15, ids, role, "f" * 64, "D080", "U100" if paired else None)


@pytest.mark.parametrize("alpha", [1., .5, 0.])
def test_real_cms_wrapper_forward_backward_extract_and_context_skip(fake_weaver, alpha):
    torch.set_num_threads(1)
    n = contracts.node("test", "fusion_withdrawal", "D080", "U100", "teacher")
    model = build_model(n)
    cache = make_cache()
    raw = cache.batch_primary(np.arange(4)) if alpha == 0 else cache.batch(np.arange(4))
    target = torch.full((4, 15), 1 / 15)
    loss, terms = training.batch_loss(model, raw, node=n, device="cpu", teacher=target, alpha=alpha)
    loss.backward()
    assert torch.isfinite(loss)
    assert len(terms) == 8
    assert any(p.grad is not None for p in model.hlt_mod.parameters())
    if alpha == 0:
        assert all(p.grad is None for p in model.context_mod.parameters())
        model.context_mod = torch.nn.Module()  # must remain unreachable
    model.eval()
    zero = training.predict(model, cache, node=n, device="cpu")
    cn = dict(n, role="extracted", context_coordinate=None, selection_route="ordinary")
    single = training.predict(model.extract_primary().eval(), cache, node=cn, device="cpu") if cache.context is None else training.predict(
        model.extract_primary().eval(), data.Cache(cache.views, cache.labels, cache.identities, cache.role,
        cache.foundation_sha256, cache.primary), node=cn, device="cpu")
    np.testing.assert_array_equal(zero, single)


@pytest.mark.parametrize("role", ["reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal"])
def test_full_kernel_selects_restores_and_publishes_cms_report(fake_weaver, role):
    torch.set_num_threads(1)
    paired = role.startswith("fusion")
    n = contracts.node(role, role, "D080", "U100" if paired else None, None if role == "reference_ce" else "teacher")
    tr, va = make_cache(paired=paired), make_cache("validation", paired=paired)
    model = build_model(n)
    p = None if role == "reference_ce" else np.full((len(tr), 15), 1 / 15, np.float32)
    report, state = training.train(model, tr, va, node=n, device="cpu", teacher_probabilities=p,
        teacher_identities=None if p is None else tr.identities, acceptance_passes=1)
    contracts.validate(report, "TRAINING_REPORT")
    assert report["selected_pass"] == 1
    assert not report["scientific_fit"]
    assert report["selected_weights_restored"]
    assert len(report["validation"]["per_class"]) == 15
    assert all(torch.equal(model.state_dict()[key], val) for key, val in state.items())
    if p is not None:
        with pytest.raises(ValueError, match="identity"):
            training.train(model, tr, va, node=n, device="cpu", teacher_probabilities=p,
                           teacher_identities=tr.identities[::-1], acceptance_passes=1)


def test_validation_partition_is_disjoint_stratified_and_identity_stable():
    labels = np.tile(np.arange(15), 8)
    ids = np.frombuffer(b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(len(labels))), np.uint8).reshape(-1, 32)
    parts = data.role_partition(labels, ids)
    assert np.bincount(parts).tolist() == [60, 30, 30]
    order = np.arange(len(labels))[::-1]
    assert np.array_equal(data.role_partition(labels[order], ids[order]), parts[order])
    for p in range(3):
        assert set(labels[parts == p]) == set(range(15))


def test_exact_native_endpoints_and_no_metadata_features():
    raw = _raw_arrays()
    maps, coverage = data._match_chunk((raw,))
    assert coverage.tolist() == [[2, 2, 2]]
    keys = ["a.root::tree::0"]
    coupling = data._couple_chunk((raw, maps, keys, _scale(), "a" * 64))
    views = data._view_chunk((raw, maps, coupling, keys, ("U000", "U033", "U066", "U100", "D100", "D000", "OFFLINE"), "a" * 64))
    from hlt_classification.scouting.inputs import build_hlt_inputs
    hlt = build_hlt_inputs(raw)
    for left, right in zip(views["U100"], views["D100"]):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(views["D000"], (hlt.features, hlt.vectors, hlt.mask)):
        np.testing.assert_array_equal(left, right)
    assert views["U000"][0].shape[1] == 21
    assert all(a[2].sum() == 2 for a in views.values())


@pytest.mark.parametrize("hlt_count,offline_count", [(2, 5), (5, 2)])
def test_persistent_support_only_removes_offline_tail(hlt_count, offline_count):
    from test_hcwdl_homotopy import _resized_hlt_arrays, _resized_offline_arrays
    raw = _resized_offline_arrays(offline_count - 1, 1)
    hlt = _resized_hlt_arrays(hlt_count)
    raw.update({key: value for key, value in hlt.items() if key.startswith("scoutpfcand_") or key == "n_scoutpfcands"})
    maps, coverage = data._match_chunk((raw,))
    assert coverage[0, 2] == min(hlt_count, offline_count)
    keys = ["a.root::tree::0"]
    edits = data._couple_chunk((raw, maps, keys, _scale(), "a" * 64))
    names = ("U000", "U033", "U066", "U100", "D080", "D000")
    views = data._view_chunk((raw, maps, edits, keys, names, "a" * 64))
    sizes = [int(views[name][2].sum()) for name in names]
    assert sizes[0] == max(hlt_count, offline_count)
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-3:] == [hlt_count] * 3
    for name in ("U000", "U033", "U066"):
        np.testing.assert_array_equal(views[name][0][:, :, :hlt_count], views["U100"][0][:, :, :hlt_count])


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    import awkward as ak
    import uproot
    from hlt_classification.scouting.splits import SourceFileRecord, build_split_manifest
    monkeypatch.setattr(campaign, "validate_source_checkout", lambda *a, **kw: None)
    for role, amount in dict(train=90, validation=60, final_test=60).items():
        monkeypatch.setitem(contracts.BUDGETS, role, amount)
    raw = _repeated_raw_arrays(60)
    raw.update(jet_tightId=np.ones(60, np.int32), jet_no=np.zeros(60, np.int32),
               scoutfj_pt=np.full(60, 300, np.float32), scoutfj_sdmass=np.full(60, 100, np.float32),
               fj_label=np.tile([309] + [0] * 14, 4),
               scoutfj_label=np.tile([0, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 21, 23, 20], 4),
               scoutfj_gen_pid=np.tile([0] * 4 + [25] + [0] * 9 + [37], 4))
    # Decoder requires the full matching projection in a real ROOT file.
    for branch in data.matching_required_branches() - set(raw):
        prefix = branch.split("_")[0]
        length = 2 if prefix == "scoutpfcand" else 1
        raw[branch] = [np.zeros(length, np.float32) for _ in range(60)]
    raw = {k: ak.Array(v) if isinstance(v, list) else np.asarray(v) for k, v in raw.items()}
    native = tmp_path / "raw"; native.mkdir()
    records = []
    for i in range(5):
        path = native / f"{i}.root"
        with uproot.recreate(path) as output:
            output["tree"] = raw
        records.append(SourceFileRecord(path.name, "sample", 60, sha256_file(path), 60, (4,) * 15))
    split = build_split_manifest(records, source_manifest_sha256="a" * 64)
    source = tmp_path / "old"; source.mkdir()
    split_path = source / "split.json"; write_immutable_json(split_path, split)
    spec = campaign.create(split_manifest=split_path, data_root=native, campaign_root=tmp_path / "new",
        project_dir=tmp_path / "checkout", source_commit="b" * 40, cpus=2, workers=1)
    return spec


def test_native_preparation_real_root_reads_and_all_cache_endpoints(tiny_campaign):
    spec = tiny_campaign
    source_role_paths = {r["path"] for r in spec["split_roles"]["final_test"]}
    # The source list omits final-test files entirely.
    assert not source_role_paths.intersection(r["path"] for r in data.source_rows(spec))
    for task in campaign.tasks(spec)["prepare"]:
        production.run_task(spec, task["task_id"], device="cpu")
    lock = load_json(Path(spec["campaign_root"]) / "foundation/foundation_lock.json")
    assert lock["final_test_materialized"] is False
    for role in ("train", "validation"):
        cache = data.build_cache(spec, role, "D000", "U000")
        assert len(cache) == contracts.BUDGETS[role]
        assert cache.batch(np.arange(3))["primary"]["features"].shape[1] == 21
        assert set(cache.batch(np.arange(3))["primary"]) == {"features", "vectors", "mask", "labels"}
    assert production.endpoint_audit(spec) == 8
    with pytest.raises(PermissionError):
        data.build_cache(spec, "final_test", "D000")


def test_staged_submission_is_dry_and_science_requires_own_gate(tiny_campaign):
    spec = tiny_campaign
    dry = campaign.submit(spec, "science")
    assert dry["dry_run"] and len(dry["jobs"]) == 46
    assert set(dry["jobs"].values()) == {f"DRY_RUN_{i:04d}" for i in range(46)}
    for row in campaign.command_plan(spec, "science")["commands"]:
        command = row["command"]
        assert "--partition=tier3" in command and "--nodes=1" in command
        assert not any("tigris" in c or "jetclass2" in c for c in command)
        if row["task_id"].startswith(("train_", "reduce_", "extract_")):
            assert "--gres=gpu:a100:1" in command
    with pytest.raises(PermissionError):
        campaign.submit(spec, "science", execute=True)
    with pytest.raises(FileNotFoundError):
        campaign.submit(spec, "science", execute=True, authorization_phrase=contracts.AUTHORIZATION)
    changed = dict(spec, graph=dict(spec["graph"], fit_count=54))
    changed.pop("content_hash")
    with pytest.raises(ValueError):
        campaign.validate_campaign(with_content_hash(changed))


def test_receipt_detects_corruption_and_cross_campaign(tmp_path):
    spec = dict(campaign_root=str(tmp_path), content_hash="a" * 64)
    path = tmp_path / "array.npz"
    fingerprint = publish_npz(path, value=np.arange(4))
    assert np.array_equal(arrays_from(fingerprint)["value"], np.arange(4))
    publish_receipt(spec, "test", [path])
    load_receipt(spec, "test")
    with pytest.raises(ValueError):
        load_receipt(dict(spec, content_hash="b" * 64), "test")
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError):
        load_receipt(spec, "test")


def test_linear_recovery_can_exceed_unity_and_undefined_denominators():
    assert production.recovery(5, 1, 3) == 2
    assert production.recovery(2, 1, 1) is None
    assert production.recovery(None, 1, 3) is None


def test_spawned_preprocessing_matches_serial():
    arguments = [(_raw_arrays(),), (_raw_arrays(),)]
    serial = list(data.bounded_map(data._match_chunk, arguments, workers=1))
    parallel = list(data.bounded_map(data._match_chunk, arguments, workers=2))
    for left, right in zip(serial, parallel):
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)


def test_selected_teacher_bank_fit_withdraw_extract_chain(tiny_campaign, fake_weaver, monkeypatch):
    """Exercise real production publication/load boundaries, not only graph mocks."""
    torch.set_num_threads(1)
    spec = tiny_campaign
    for task in campaign.tasks(spec)["prepare"]:
        production.run_task(spec, task["task_id"], device="cpu")
    kernel = production.train
    def miniature(*args, **kwargs):
        # Local fixtures cannot stand in for scientific acceptance. This test
        # shortens the loop only; all production IO/model/teacher joins run.
        return kernel(*args, **kwargs, acceptance_passes=1)
    monkeypatch.setattr(production, "train", miniature)
    monkeypatch.setattr(production, "gate_check", lambda s: None)
    monkeypatch.setattr(production, "validate_gpu_allocation", lambda s, d: None)
    for name in ("train_U000", "reduce_U000", "train_DIRECT_D000", "train_ACQUIRE_U033",
                 "reduce_ACQUIRE_U033", "train_WITHDRAW_U033", "extract_CARRIER_U033", "reduce_CARRIER_U033"):
        production.run_task(spec, name, device="cpu")
        load_receipt(spec, name)
    cache = data.build_cache(spec, "train", "U033")
    bank, arrays = production.load_bank(spec, "CARRIER_U033", cache)
    assert bank["train_temperature"] == 2.
    assert arrays["probabilities"].shape == (90, 15)
    report = load_json(Path(spec["campaign_root"]) / "training/CARRIER_U033/report.json")
    assert report["exact_extraction"] is True
    assert report["context_parameters_removed"] is True
    assert report["validation_rows_checked"] == 60
    model, _ = production.load_model(spec, "CARRIER_U033", "cpu")
    assert not any("context" in name for name in model.state_dict())


def test_gpu_worker_rejects_local_execution_before_reading_data(tiny_campaign):
    with pytest.raises(PermissionError, match="A100"):
        production.run_task(tiny_campaign, "train_U000", device="cpu")


def test_installed_weaver_native_wrapper_contract():
    pytest.importorskip("weaver")
    # Supplemental local parity when Weaver is installed; not a SPORC gate.
    torch.set_num_threads(1)
    n = contracts.node("parity", "fusion_withdrawal", "D080", "U100", "teacher")
    model = build_model(n).eval()
    cache = make_cache("validation")
    zero = training.predict(model, cache, node=n, device="cpu")
    single_cache = data.Cache(cache.views, cache.labels, cache.identities, cache.role,
                             cache.foundation_sha256, cache.primary)
    n = dict(n, context_coordinate=None, selection_route="ordinary")
    expected = training.predict(model.extract_primary().eval(), single_cache, node=n, device="cpu")
    np.testing.assert_array_equal(zero, expected)
