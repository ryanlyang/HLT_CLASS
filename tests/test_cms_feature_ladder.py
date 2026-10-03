"""Local fixtures are not remote acceptance evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.cms_feature_ladder import contracts as k, inputs as i, campaign as c, data as d, training as t, runtime as r, submission as s
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.inputs import build_inputs
from hlt_classification.cms_proxy_ladder.views import match_particles, build_view
from test_hcwdl_homotopy import _raw_arrays, _repeated_raw_arrays, _resized_offline_arrays, _resized_hlt_arrays


def test_registry_default_and_opt_in_counts_seeds_teachers():
    assert k.registry(["SHARED17"])["budgets"] == {"train": 200000, "validation": 50000}
    for arms in (["SHARED17"], ["CMS21"], list(k.ARMS)):
        graph = k.science_tasks(arms)
        assert len(graph) == 16 * len(arms)
        assert sum(x["kind"] == "train" for x in graph) == 9 * len(arms)
        assert sum(x["kind"] == "reduce" for x in graph) == 5 * len(arms)
        seen = set()
        for task in graph:
            assert set(task["dependencies"]) <= seen
            assert task["task_id"] not in seen
            seen.add(task["task_id"])
    for arm in k.ARMS:
        nodes = {n["name"]: n for n in k.nodes(arm)}
        endpoints = [nodes[n] for n in ("M0HLT", "DIRECT_D000", "COARSE_D000")]
        assert len({n["initialization_seed"] for n in endpoints}) == 1
        assert len({n["sampler_seed"] for n in endpoints}) == 1
        assert nodes["COARSE_D000"]["teacher"] == "COARSE_D033"
        assert nodes["DIRECT_D000"]["teacher"] == "U000"
        assert nodes["OFFLINE"]["teacher"] is None
    assert t.model_config("SHARED17")["trim"] is False
    assert t.model_config("CMS21") == dict(t.model_config("SHARED17"), input_dim=21)
    with pytest.raises(ValueError):
        k.registry(["SHARED17", "DENSE"])


def test_units_uncertainties_missingness_and_flags():
    raw = _raw_arrays()
    h, o = next(i.endpoints(raw))
    assert h.particles.tracking[0, 0] == pytest.approx(-.2)
    assert h.particles.tracking[0, 2] == pytest.approx(.1)
    assert h.particles.tracking[0, 1] == pytest.approx(.1)
    assert h.particles.tracking[0, 3] == pytest.approx(.1 / 1.5)
    assert not h.particles.valid[1].any()
    fx, vx = i.features(h, "SHARED17")
    np.testing.assert_array_equal(fx, build_inputs(h.particles).features)
    np.testing.assert_array_equal(vx, h.particles.p4.astype(np.float32))
    raw["scoutpfcand_dxy"][0][0] = 0
    raw["scoutpfcand_dxysig"][0][0] = 0
    raw["scoutpfcand_isMu"][0][0] = 1  # multi-hot: do not force argmax
    raw["scoutpfcand_dzsig"][0][0] = np.nan
    h, _ = next(i.endpoints(raw))
    assert h.particles.valid[0].tolist() == [True, True, False, False]
    assert h.particles.category[0] == 5
    fx, _ = i.features(h, "SHARED17")
    assert fx[0, 6:11].sum() == 2
    assert fx[0, 12] == fx[0, 14] == 0


def test_shared_input_does_not_expose_extra_cms_fields():
    h, _ = next(i.endpoints(_raw_arrays()))
    raw = h.raw.copy()
    raw[:, [0, 11, 15, 16, 17, 20]] += 999
    changed = i.endpoint(h.particles.p4, raw, h.valid, prefix="hlt")
    np.testing.assert_array_equal(i.features(h, "SHARED17")[0], i.features(changed, "SHARED17")[0])
    assert not np.array_equal(i.features(h, "CMS21")[0], i.features(changed, "CMS21")[0])


def test_missing_identity_flag_cannot_be_hidden_by_sanitization():
    h, _ = next(i.endpoints(_raw_arrays()))
    valid = h.valid.copy()
    valid[0, i.PID_CHANNELS[0]] = False
    with pytest.raises(ValueError, match="identity flags/charge"):
        i.endpoint(h.particles.p4, h.raw, valid, prefix="offline")


@pytest.mark.parametrize("nh,no,lost", [(2, 8, 2), (8, 2, 0), (2, 2, 0)])
def test_persistent_support_exact_hlt_and_offline_matched_slots(nh, no, lost):
    raw = _resized_offline_arrays(no - 1, 1, lost=lost)
    for key, value in _resized_hlt_arrays(nh).items():
        if key.startswith("scoutpfcand_") or key == "n_scoutpfcands":
            raw[key] = value
    h, o = next(i.endpoints(raw))
    assert len(o.particles) == no  # lost tracks stay in both arms
    mapping = match_particles(h.particles, o.particles)
    values = {coord: i.view(identity="a" * 64, hlt=h, offline=o, mapping=mapping, coordinate=coord)
              for coord in ("OFFLINE", "U000", "U050", "U100", "D066", "D033", "D000")}
    assert len(values["U000"].particles) == max(nh, no)
    assert len(values["U100"].particles) == nh
    assert len(values["U000"].particles) >= len(values["U050"].particles) >= nh
    assert i.view(identity="unused", hlt=h, offline=None, coordinate="D000") is h
    assert values["OFFLINE"] is o
    for slot in np.flatnonzero(mapping >= 0):
        np.testing.assert_array_equal(values["U100"].particles.p4[slot], o.particles.p4[mapping[slot]])
        np.testing.assert_array_equal(values["U100"].raw[slot], o.raw[mapping[slot]])
    for coord, value in values.items():
        for arm in k.ARMS:
            f, v = i.features(value, arm)
            assert f.shape == (len(value.particles), 17 if arm == "SHARED17" else 21)
            assert np.isfinite(f).all()
            np.testing.assert_array_equal(v, value.particles.p4.astype(np.float32))
        expected = build_view(identity="a" * 64, proxy=h.particles, offline=o.particles, coordinate=coord, mapping=mapping)
        for field in ("p4", "charge", "category", "tracking", "valid"):
            np.testing.assert_array_equal(getattr(value.particles, field), getattr(expected, field))
    np.testing.assert_allclose(values["D066"].particles.p4[mapping >= 0],
        h.particles.p4[mapping >= 0] / 3 + 2 * o.particles.p4[mapping[mapping >= 0]] / 3)


def test_nonphysical_overflow_and_unknown_coordinate_fail():
    raw = _raw_arrays()
    raw["scoutpfcand_energy"][0][0] = -1
    with pytest.raises(ValueError):
        next(i.endpoints(raw))
    raw = _resized_offline_arrays(513, 1)
    h, o = next(i.endpoints(raw))
    with pytest.raises(ValueError, match="512"):
        i.features(o, "SHARED17")
    with pytest.raises(ValueError, match="Unregistered"):
        i.view(identity="a" * 64, hlt=h, offline=o, coordinate="D020")


def test_matching_process_replay():
    raw = _raw_arrays()
    serial = list(d.bounded_map(d._match, [raw], 1))[0]
    parallel = list(d.bounded_map(d._match, [raw], 2))[0]
    for left, right in zip(serial[0], parallel[0]):
        np.testing.assert_array_equal(left, right)
    assert serial[1:] == parallel[1:]
    args = (raw, serial[0], ["a" * 64], "D033", "SHARED17")
    left = list(d.bounded_map(d._view_chunk, [args], 1))[0]
    right = list(d.bounded_map(d._view_chunk, [args], 2))[0]
    for x, y in zip(left[0], right[0]):
        np.testing.assert_array_equal(x, y)


@pytest.fixture
def spec(tmp_path, monkeypatch):
    import awkward as ak
    import uproot
    from hlt_classification.scouting.splits import SourceFileRecord, build_split_manifest
    monkeypatch.setattr(c, "validate_source_checkout", lambda *a, **kw: None)
    monkeypatch.setitem(k.BUDGETS, "train", 90)
    monkeypatch.setitem(k.BUDGETS, "validation", 60)
    # Production contract stays 16 workers; locally invoke the exact bounded-map
    # function serially so each test doesn't spawn 16 imports.
    real_map = d.bounded_map
    monkeypatch.setattr(d, "bounded_map", lambda fn, args, workers: real_map(fn, args, 1))
    raw = _repeated_raw_arrays(60)
    raw.update(jet_tightId=np.ones(60, np.int32), jet_no=np.zeros(60, np.int32),
        scoutfj_pt=np.full(60, 300, np.float32), scoutfj_sdmass=np.full(60, 100, np.float32),
        fj_label=np.tile([309] + [0] * 14, 4),
        scoutfj_label=np.tile([0, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 21, 23, 20], 4),
        scoutfj_gen_pid=np.tile([0] * 4 + [25] + [0] * 9 + [37], 4))
    from hlt_classification.scouting.schema import matching_required_branches
    for branch in matching_required_branches() - set(raw):
        count = 2 if branch.startswith("scoutpfcand_") else 1
        raw[branch] = [np.zeros(count, np.float32) for _ in range(60)]
    raw = {key: ak.Array(val) if isinstance(val, list) else np.asarray(val) for key, val in raw.items()}
    native = tmp_path / "native"; native.mkdir()
    files = []
    for index in range(5):
        target = native / f"{index}.root"
        with uproot.recreate(target) as handle:
            handle["tree"] = raw
        files.append(SourceFileRecord(target.name, "sample", 60, sha256_file(target), 60, (4,) * 15))
    split = build_split_manifest(files, source_manifest_sha256="a" * 64)
    split_path = tmp_path / "split.json"
    write_immutable_json(split_path, split)
    return c.create(project=tmp_path / "checkout", commit="b" * 40, data_root=native,
                    split_path=split_path, root=tmp_path / "campaign", arms=list(k.ARMS))


def prepare(spec):
    c.publish(spec, "select", d.select(spec))
    for index in range(len(spec["sources"])):
        c.publish(spec, f"match_{index:04d}", d.match_source(spec, index))
    c.publish(spec, "foundation", d.foundation(spec))


def test_real_root_cache_pairing_no_test_access(spec, monkeypatch):
    import uproot
    split = c.validate_spec(spec)
    forbidden = {row["path"] for row in split["roles"]["final_test"]["files"]}
    assert not forbidden.intersection(row["path"] for row in spec["sources"])
    opener = uproot.open
    def guarded(path, *args, **kwargs):
        assert Path(str(path)).name not in forbidden
        return opener(path, *args, **kwargs)
    monkeypatch.setattr(uproot, "open", guarded)
    prepare(spec)
    for coordinate in ("D000", "OFFLINE", "U000", "U050", "U100", "D066", "D033"):
        left = d.build_cache(spec, "train", coordinate, "SHARED17")
        right = d.build_cache(spec, "train", coordinate, "CMS21")
        np.testing.assert_array_equal(left.identities, right.identities)
        np.testing.assert_array_equal(left.labels, right.labels)
        np.testing.assert_array_equal(left.views[coordinate][1], right.views[coordinate][1])
        np.testing.assert_array_equal(left.views[coordinate][2], right.views[coordinate][2])
        raw = left.batch_primary(np.arange(3))
        assert set(raw) == {"features", "vectors", "mask", "labels"}
        assert raw["features"].shape == (3, 17, 16)
    with pytest.raises(PermissionError):
        d.build_cache(spec, "final_test", "D000", "SHARED17")


def test_bad_source_assignment_and_receipts_fail(spec):
    prepare(spec)
    record, arrays = d.assignment(spec, 0)
    target = Path(record["payload"]["path"])
    with target.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError):
        c.require(spec, "match_0000")
    with pytest.raises(ValueError):
        d.build_cache(spec, "train", "D000", "SHARED17")


def test_spec_rehash_cannot_change_science_and_gate_requires_acceptance(spec):
    altered = deepcopy(spec)
    altered["scientific"]["training"]["ce_weight"] = .2
    with pytest.raises(ValueError):
        c.validate_spec(with_content_hash({k: v for k, v in altered.items() if k != "content_hash"}))
    with pytest.raises(PermissionError):
        s.plan(spec, "science")
    with pytest.raises(PermissionError):
        r.run(spec, "select")  # no real Slurm allocation


class TinyWeaver(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.fc = nn.Linear(config["input_dim"], config["num_classes"])
        self.cls_token = nn.Parameter(torch.zeros(1))

    def forward(self, features, v=None, mask=None):
        del v
        pooled = (features * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return self.fc(pooled) + self.cls_token


@pytest.fixture
def tiny_model(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(t, "load_weaver_particle_transformer_class", lambda: TinyWeaver)


def test_kernel_parity_ce_kd_restore_and_identity_rejection(spec, tiny_model):
    prepare(spec)
    tr, va = (d.build_cache(spec, role, "D000", "SHARED17") for role in ("train", "validation"))
    assert t.parity(tr, "SHARED17", "cpu")["passed"]
    n = next(n for n in k.nodes("SHARED17") if n["name"] == "M0HLT")
    model = t.fresh(n)
    report, state = t.train(model, tr, va, node=n, device="cpu", acceptance_passes=1)
    assert not report["scientific_fit"]
    assert report["selected_weights_restored"] and report["selected_pass"] == 1
    assert all(torch.equal(v, model.state_dict()[key]) for key, v in state.items())
    probabilities = t.predict(model, tr, device="cpu", temperature=2.)
    n = next(n for n in k.nodes("SHARED17") if n["name"] == "DIRECT_D000")
    kd, _ = t.train(t.fresh(n), tr, va, node=n, device="cpu", teacher=probabilities, teacher_ids=tr.identities, acceptance_passes=1)
    assert len(kd["validation"]["per_class"]) == 15
    with pytest.raises(ValueError, match="identity"):
        t.train(t.fresh(n), tr, va, node=n, device="cpu", teacher=probabilities, teacher_ids=tr.identities[::-1], acceptance_passes=1)


def test_full_chain_publications_join_correct_arm(spec, tiny_model, monkeypatch):
    prepare(spec)
    monkeypatch.setattr(r, "profile", lambda *args, **kwargs: {})
    original_recipe = t.recipe
    def short_recipe():
        value = original_recipe()
        value.update(maximum_passes=1, minimum_passes=1)
        # Keep the real recipe identity solely inside this explicitly mocked
        # local execution test. No such override exists in the CLI.
        return value
    monkeypatch.setattr(t, "recipe", short_recipe)
    for arm in k.ARMS:
        for node in k.nodes(arm):
            c.publish(spec, "train_" + node["node_id"], r.fit(spec, node, device="cpu"))
            if node["name"] in ("U000", "COARSE_U050", "COARSE_U100", "COARSE_D066", "COARSE_D033"):
                c.publish(spec, "reduce_" + node["node_id"], r.reduce(spec, node, device="cpu"))
        rows = r.results(spec, arm)
        assert len(rows) == 9 and all(x["state"] == "COMPLETE" for x in rows)
        final = r.training_report(spec, arm + "_COARSE_D000")
        bank = load_json(r.output(spec, "banks", arm + "_COARSE_D033.json"))
        assert final["teacher_bank_sha256"] == bank["content_hash"]


def test_metrics_censor_zero_background_and_poor_predictions_continue():
    labels = np.tile(np.arange(15), 3)
    p = np.full((len(labels), 15), .001 / 14)
    p[np.arange(len(labels)), labels] = .999
    value = t.metrics(labels, p)
    assert value["per_class"]["Xbb"]["rejection_at_50pct"] is None
    assert value["macro_mean_log_qcd_rejection_at_50pct_signal"] is None
    poor = t.metrics(labels, np.full_like(p, 1 / 15))
    assert poor["accuracy"] == pytest.approx(1 / 15)
    assert poor["per_class"]["Xbb"]["rejection_at_50pct"] == 1


def test_queue_dry_run_shape_authorization_and_no_mutation(spec, monkeypatch):
    import shlex
    calls = []
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **kw: calls.append(a) or pytest.fail("dry run invoked scheduler"))
    result = s.submit(spec, "gate")
    assert result["dry_run"] and not calls
    for row in result["plan"]["commands"]:
        command = row["command"]
        assert "--partition=debug" in command and "--export=NONE" in command
        assert "--account=reu-aisocial" in command
        assert ("--gres=gpu:a100:1" in command) == row["task_id"].startswith("preflight_")
        assert spec["project_dir"] in command[-1]
        assert "source " in command[-1] and "jetclass2_delphes_common.sh" in command[-1]
        wrapped = shlex.split(command[-1].removeprefix("--wrap="))
        assert wrapped[:3] == ["exec", "bash", "-c"] and len(wrapped) == 4
        assert wrapped[3].startswith("set -euo pipefail\n")
    with pytest.raises(PermissionError):
        s.submit(spec, "gate", execute=True, reviewed_hash="f" * 64, authorization=k.AUTHORIZATION)
    assert not calls


def test_submission_claim_and_exact_ledger_idempotence(spec, monkeypatch):
    dry = s.submit(spec, "gate")
    calls = []
    def scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "scontrol":
            return SimpleNamespace(stdout="ClusterName = sporc\n")
        assert not any(key.startswith(("SLURM_", "SBATCH_")) for key in kwargs["env"])
        return SimpleNamespace(stdout=str(9000 + len(calls)) + "\n")
    monkeypatch.setattr(s.subprocess, "run", scheduler)
    monkeypatch.setenv("SBATCH_GRES", "gpu:h100:4")
    args = dict(execute=True, reviewed_hash=dry["plan"]["content_hash"], authorization=k.AUTHORIZATION)
    live = s.submit(spec, "gate", **args)
    count = len([x for x in calls if x[0] == "sbatch" and "--test-only" not in x])
    assert count == len(spec["gate_tasks"])
    assert s.submit(spec, "gate", **args) == live
    assert len([x for x in calls if x[0] == "sbatch" and "--test-only" not in x]) == count
    for command in calls:
        assert all("${JOB_" not in token for token in command)


def test_unresolved_live_claim_blocks_retry(spec, monkeypatch):
    dry = s.submit(spec, "gate")
    directory = Path(spec["campaign_root"]) / "submission_gate"
    (directory / "live.claim").mkdir()
    monkeypatch.setattr(s.subprocess, "run", lambda command, **kw: SimpleNamespace(stdout="ClusterName = sporc\n"))
    with pytest.raises(FileExistsError):
        s.submit(spec, "gate", execute=True, reviewed_hash=dry["plan"]["content_hash"], authorization=k.AUTHORIZATION)


def acceptance_fixture(spec, arm):
    """Deliberately mocked local evidence to exercise fail-closed validators."""
    lock = load_json(d.path(spec, "foundation.json"))
    miniature = {}
    for key, name in (("miniature_ce", "U000"), ("miniature_kd", "COARSE_D033")):
        miniature[key] = k.artifact("TRAINING", parents={"foundation": lock["content_hash"],
            "recipe": spec["scientific"]["training"]["content_hash"]},
            node=next(n for n in k.nodes(arm) if n["name"] == name), scientific_fit=False,
            validation={"rows": spec["scientific"]["budgets"]["validation"]})
    return k.artifact("PREFLIGHT", parents={"spec": spec["content_hash"], "foundation": lock["content_hash"]},
        arm=arm, source_commit=spec["source_commit"], budgets=spec["scientific"]["budgets"],
        passed=True, model=t.model_config(arm), parity=dict(passed=True, model=t.model_config(arm),
            precision="fp32", forward_features_parameters=True), job_id="99999",
        probe_batch=256, probe_steps=3, train_minutes=120, reduce_minutes=30,
        measurement=dict(sampled_peak_tree_rss_bytes=2**30, samples=30),
        gpu_peak_bytes=2**30, gpu=dict(total_memory_bytes=40 * 2**30),
        required_free_bytes=2**30, environment={"fixture_not_remote_evidence": True}, **miniature)


@pytest.mark.parametrize("change", ["valid", "batch", "cpu", "gpu", "parity", "scientific", "walltime"])
def test_acceptance_authentication_never_uses_performance_threshold(spec, change):
    prepare(spec)
    value = acceptance_fixture(spec, "SHARED17")
    if change == "batch": value["probe_batch"] = 32
    if change == "cpu": value["measurement"]["sampled_peak_tree_rss_bytes"] = 200000 * 2**20
    if change == "gpu": value["gpu_peak_bytes"] = value["gpu"]["total_memory_bytes"]
    if change == "parity": value["parity"]["passed"] = False
    if change == "scientific": value["miniature_ce"]["scientific_fit"] = True
    if change == "walltime": value["train_minutes"] = 1441
    value = with_content_hash({key: val for key, val in value.items() if key != "content_hash"})
    path = r.output(spec, "evidence", "SHARED17.json")
    write_immutable_json(path, value)
    c.publish(spec, "preflight_SHARED17", [path])
    if change == "valid":
        assert r.profile(spec, "SHARED17")["passed"]
    else:
        with pytest.raises(ValueError):
            r.profile(spec, "SHARED17")


def test_full_science_plan_requires_every_requested_arm(spec):
    prepare(spec)
    for arm in k.ARMS:
        if arm == "CMS21":
            with pytest.raises(PermissionError):
                s.plan(spec, "science")
        value = acceptance_fixture(spec, arm)
        path = r.output(spec, "evidence", arm + ".json")
        write_immutable_json(path, value)
        c.publish(spec, "preflight_" + arm, [path])
    plan = s.plan(spec, "science")
    assert len(plan["commands"]) == 32
    for row in plan["commands"]:
        cmd = row["command"]
        gpu = row["task_id"].startswith(("train_", "reduce_"))
        assert ("--gres=gpu:a100:1" in cmd) == gpu
        assert "--partition=debug" in cmd
        assert all("DENSE" not in token for token in cmd)
    endpoint = next(row for row in plan["commands"] if row["task_id"] == "train_SHARED17_COARSE_D000")
    assert endpoint["dependencies"] == ["reduce_SHARED17_COARSE_D033"]


def test_source_drift_blocks_even_completed_receipt(spec, monkeypatch):
    prepare(spec)
    def denied(*args, **kwargs):
        raise PermissionError("dirty or different source")
    monkeypatch.setattr(c, "validate_source_checkout", denied)
    with pytest.raises(PermissionError, match="source"):
        r.run(spec, "select")
    with pytest.raises(PermissionError, match="source"):
        s.submit(spec, "gate")
