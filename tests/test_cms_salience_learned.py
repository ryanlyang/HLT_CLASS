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
    for _, vectors, _ in views.values():
        # Weaver consumes (px, py, pz, E), not four independent features.
        # Give every synthetic particle positive energy and unit mass, as in
        # validate_scouting_weaver_fp32_parity. Random E can make rapidity NaN;
        # the fake Weaver pair module cannot expose that fixture error.
        vectors[:, 3] = np.sqrt(np.square(vectors[:, :3]).sum(axis=1) + 1.)
    ids = np.arange(count * 32, dtype=np.uint8).reshape(count, 32)
    return data.Cache(views, np.arange(count) % 15, ids, role, "f" * 64, "D080", "U100" if paired else None)


def test_synthetic_cache_has_physical_four_vectors_without_weaver():
    cache = make_cache("validation")
    for features, vectors, mask in cache.views.values():
        active = mask[:, 0]
        momentum = vectors[:, :3].astype(np.float64)
        energy = vectors[:, 3].astype(np.float64)
        mass_squared = energy**2 - np.square(momentum).sum(axis=1)
        assert np.isfinite(features).all() and np.isfinite(vectors).all()
        assert np.all(energy[active] > 0)
        assert np.all(energy[active] > np.abs(momentum[:, 2][active]))
        np.testing.assert_allclose(mass_squared[active], 1., atol=5e-6, rtol=0)
        rapidity = .5 * np.log((energy + momentum[:, 2]) / (energy - momentum[:, 2]))
        assert np.isfinite(rapidity[active]).all()


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


def create_from(spec, name, **kwargs):
    return campaign.create(split_manifest=spec["split_manifest"]["path"], data_root=spec["data_root"],
        campaign_root=Path(spec["campaign_root"]).parent / name, project_dir=spec["project_dir"],
        source_commit=spec["source_commit"], cpus=spec["resources"]["cpus"],
        workers=spec["resources"]["workers"], memory_mb=spec["resources"]["memory_mb"], **kwargs)


def rehash(value, **changes):
    return with_content_hash({k: v for k, v in dict(value, **changes).items() if k != "content_hash"})


def test_debug_is_pinned_across_all_stages_without_changing_science(tiny_campaign):
    old = tiny_campaign
    debug = create_from(old, "debug", partition="debug")
    assert debug["schema_version"] == 2
    assert debug["graph"] == old["graph"]
    assert debug["budgets"] == old["budgets"]
    assert debug["view_config_sha256"] == old["view_config_sha256"]
    assert debug["site"] == dict(contracts.SITE, partition="debug")
    for stage in ("prepare", "gate", "science"):
        for row in campaign.command_plan(debug, stage)["commands"]:
            command = row["command"]
            assert "--partition=debug" in command
            assert "--partition=tier3" not in command
            assert "--account=reu-aisocial" in command and "--qos=qos_tier3" in command
            assert "--no-requeue" in command
            assert int(next(c.split("=", 1)[1] for c in command if c.startswith("--time="))) <= 1440
        assert campaign.submit(debug, stage)["dry_run"] is True
    assert contracts.allocation_site(debug)["name"] == "sporc_a100_debug"
    assert contracts.allocation_site(old)["name"] == "sporc_a100"
    # A mutation cannot reuse the dry plan even if someone rehashes the spec.
    with pytest.raises(ValueError, match="command plan"):
        campaign.submit(rehash(old, site=debug["site"]), "science")
    for field, bad in (("partition", "tigris"), ("qos", "other"), ("account", "other"), ("gres", "gpu:h100:1")):
        with pytest.raises(ValueError):
            campaign.validate_campaign(rehash(debug, site=dict(debug["site"], **{field: bad})))


def test_legacy_v1_remains_readable_but_tier3_only(tiny_campaign):
    fields = {k: v for k, v in tiny_campaign.items() if k not in {"content_hash", "preparation_import"}}
    legacy = rehash(fields, contract=f"{contracts.FAMILY}_CAMPAIGN_SPEC/v1", schema_version=1)
    campaign.validate_campaign(legacy)
    assert "--partition=tier3" in campaign.command_plan(legacy, "gate")["commands"][0]["command"]
    with pytest.raises(ValueError, match="Legacy"):
        campaign.validate_campaign(rehash(legacy, site=contracts.site_for_partition("debug")))
    with pytest.raises(ValueError, match="Legacy"):
        campaign.validate_campaign(rehash(legacy, preparation_import=None))
    with pytest.raises(ValueError, match="version"):
        campaign.validate_campaign(rehash(legacy, schema_version=3))


def test_worker_authenticates_selected_partition_and_exact_resources(tiny_campaign, monkeypatch):
    from hlt_classification.jetclass2_delphes import execution
    spec = create_from(tiny_campaign, "debug_worker", partition="debug")
    calls = []
    def allocation(site):
        calls.append(site)
        return "123", spec["resources"]["cpus"], spec["resources"]["memory_mb"]
    monkeypatch.setattr(execution, "allocation", allocation)
    production.validate_gpu_allocation(spec, "cuda")
    assert calls[0] == execution.execution_site("sporc_a100_debug")
    monkeypatch.setattr(execution, "allocation", lambda site: ("123", 1, 192000))
    with pytest.raises(PermissionError, match="resources"):
        production.validate_gpu_allocation(spec, "cuda")


@pytest.mark.parametrize("measured_partition", ["tier3", "debug"])
def test_gate_requires_own_partition_acceptance(tiny_campaign, measured_partition):
    spec = create_from(tiny_campaign, "debug_gate", partition="debug")
    path = Path(spec["campaign_root"]) / "execution_acceptance.json"
    proofs = [contracts.artifact("TRAINING_REPORT", node={"role": r}, scientific_fit=False, passes=1)
              for r in ("reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal")]
    acceptance = contracts.artifact("EXECUTION_ACCEPTANCE", campaign_spec_sha256=spec["content_hash"],
        site=contracts.site_for_partition(measured_partition), source_commit=spec["source_commit"], genuine_allocation=True,
        installed_weaver_forward_backward=True, exact_extraction=True, endpoint_parity=True,
        full_population_cache_rows={r: contracts.BUDGETS[r] for r in ("train", "validation")},
        final_test_accessed=False, peak_rss_bytes=1, peak_cuda_bytes=1, total_cuda_bytes=100,
        miniature_reports=proofs)
    write_immutable_json(path, acceptance)
    publish_receipt(spec, "foundation", [])
    publish_receipt(spec, "preflight", [path])
    if measured_partition == "debug":
        assert campaign.gate_check(spec) == acceptance
    else:
        with pytest.raises(ValueError, match="acceptance"):
            campaign.gate_check(spec)


def test_preparation_import_reuses_exact_views_read_only_and_requires_new_gate(tiny_campaign, monkeypatch):
    from hlt_classification.cms_salience_learned import preparation_import as reuse
    source = tiny_campaign
    for task in campaign.tasks(source)["prepare"]:
        production.run_task(source, task["task_id"], device="cpu")
    original = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    # The synthetic fixture has no source checkout; real Git compatibility is
    # tested separately below. Production never skips this proof.
    monkeypatch.setattr(reuse, "preparation_code", lambda project, commit: {"fixture": "same"})
    consumer = create_from(source, "imported_debug", partition="debug",
        reuse_preparation_spec=Path(source["campaign_root"]) / "campaign_spec.json")
    assert campaign.tasks(consumer)["prepare"] == [dict(task_id="foundation", kind="import_foundation", dependencies=[])]
    assert len(campaign.command_plan(consumer, "science")["commands"]) == 46
    with pytest.raises(FileNotFoundError):
        production.build_cache(consumer, "train", "D000")
    production.run_task(consumer, "foundation", device="cpu")
    load_receipt(consumer, "foundation")
    for role in ("train", "validation"):
        before = data.build_cache(source, role, "D000", "U000")
        after = production.build_cache(consumer, role, "D000", "U000")
        assert after.foundation_sha256 == before.foundation_sha256
        np.testing.assert_array_equal(after.identities, before.identities)
        np.testing.assert_array_equal(after.labels, before.labels)
        for name in before.views:
            for a, b in zip(after.views[name], before.views[name]):
                np.testing.assert_array_equal(a, b)
        if role == "validation":
            np.testing.assert_array_equal(production.partitions(consumer, after), production.partitions(source, before))
    assert production.endpoint_audit(consumer) == 8
    with pytest.raises(PermissionError):
        production.build_cache(consumer, "final_test", "D000")
    with pytest.raises(FileNotFoundError):
        campaign.submit(consumer, "science", execute=True, authorization_phrase=contracts.AUTHORIZATION)
    assert original == {p: sha256_file(p) for p in original}
    assert not (Path(consumer["campaign_root"]) / "foundation/assignment_0000.npz").exists()
    with pytest.raises(ValueError, match="Chained"):
        create_from(source, "chained", partition="debug",
            reuse_preparation_spec=Path(consumer["campaign_root"]) / "campaign_spec.json")
    with pytest.raises(ValueError, match="identity"):
        reuse.validate_import(dict(consumer, view_config_sha256="f" * 64))
    with pytest.raises(ValueError, match="coverage"):
        reuse.validate_import(dict(consumer, preparation_import=rehash(consumer["preparation_import"], receipts={})))
    payload = Path(source["campaign_root"]) / "foundation/coupling_0000.npz"
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="payload changed"):
        reuse.validate_import(consumer, deep=True)
    with pytest.raises(ValueError, match="payload changed"):
        production.build_cache(consumer, "train", "D000")


def test_import_rejects_incomplete_and_changed_code(tiny_campaign, monkeypatch):
    from hlt_classification.cms_salience_learned import preparation_import as reuse
    monkeypatch.setattr(reuse, "preparation_code", lambda project, commit: {"fixture": commit})
    source = tiny_campaign
    path = Path(source["campaign_root"]) / "campaign_spec.json"
    with pytest.raises(FileNotFoundError):
        create_from(source, "incomplete", reuse_preparation_spec=path)
    with pytest.raises(ValueError, match="code changed"):
        reuse.build_import(dict(source, source_commit="c" * 40,
                                campaign_root=str(Path(source["campaign_root"]).parent / "changed")), path)


def test_completed_legacy_preparation_import_and_consumer_worker_count(tiny_campaign, monkeypatch):
    from hlt_classification.cms_salience_learned import preparation_import as reuse
    original_artifact = campaign.artifact
    def legacy_artifact(kind, **fields):
        value = original_artifact(kind, **fields)
        if kind == "CAMPAIGN_SPEC":
            value.pop("preparation_import")
            value = rehash(value, schema_version=1, contract=f"{contracts.FAMILY}_CAMPAIGN_SPEC/v1")
        return value
    monkeypatch.setattr(campaign, "artifact", legacy_artifact)
    source = create_from(tiny_campaign, "legacy_producer")
    monkeypatch.setattr(campaign, "artifact", original_artifact)
    assert source["schema_version"] == 1
    for task in campaign.tasks(source)["prepare"]:
        production.run_task(source, task["task_id"], device="cpu")
    monkeypatch.setattr(reuse, "preparation_code", lambda project, commit: {"fixture": "same"})
    consumer = create_from(source, "legacy_consumer", partition="debug",
        reuse_preparation_spec=Path(source["campaign_root"]) / "campaign_spec.json")
    production.run_task(consumer, "foundation", device="cpu")
    assert reuse.preparation_spec(consumer) == source
    # The adapter must not launch the producer's larger process pool if the
    # consumer requests fewer CPUs. No on-disk producer spec is rewritten.
    seen = []
    def cache(spec, *args, **kwargs):
        seen.append(spec)
        return "cache"
    monkeypatch.setattr(production, "build_native_cache", cache)
    assert production.build_cache(dict(consumer, resources=dict(consumer["resources"], workers=2)), "train", "D000") == "cache"
    assert seen[0]["resources"]["workers"] == 2
    assert source["resources"]["workers"] == 1
    assert load_json(Path(source["campaign_root"]) / "campaign_spec.json") == source


def test_preparation_code_fingerprints_real_git_and_coordinate():
    import subprocess
    from hlt_classification.cms_salience_learned.preparation_import import preparation_code, PREPARATION_CODE
    project = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], check=True,
                            text=True, capture_output=True).stdout.strip()
    value = preparation_code(project, commit)
    assert set(value) == {*PREPARATION_CODE, "coordinate_ast_sha256"}
    assert len(value["coordinate_ast_sha256"]) == 64
    assert all(len(value[path]) == 40 for path in PREPARATION_CODE)
    with pytest.raises(ValueError, match="exact"):
        preparation_code(project, "HEAD")


def test_installed_weaver_native_wrapper_contract():
    pytest.importorskip("weaver")
    # Supplemental local parity when Weaver is installed; not a SPORC gate.
    torch.set_num_threads(1)
    n = contracts.node("parity", "fusion_withdrawal", "D080", "U100", "teacher")
    model = build_model(n).eval()
    cache = make_cache("validation")
    # Exercise real pair geometry on the privileged paths as well as the
    # exactly extractable alpha-zero path. Do not relax finiteness or parity.
    for alpha in (1., .5):
        probabilities = training.predict(model, cache, node=n, device="cpu", alpha=alpha)
        assert probabilities.shape == (len(cache), 15)
        assert np.isfinite(probabilities).all()
    zero = training.predict(model, cache, node=n, device="cpu")
    single_cache = data.Cache(cache.views, cache.labels, cache.identities, cache.role,
                             cache.foundation_sha256, cache.primary)
    n = dict(n, context_coordinate=None, selection_route="ordinary")
    expected = training.predict(model.extract_primary().eval(), single_cache, node=n, device="cpu")
    np.testing.assert_array_equal(zero, expected)
