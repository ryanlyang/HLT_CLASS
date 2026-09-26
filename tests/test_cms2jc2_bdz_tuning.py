"""Frozen control, real ROOT workflow, calibration support and sealed test roles."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    bdz_campaign as campaign, bdz_worker as worker, bdz_maps as maps, bdz_metrics as metrics,
    bounded_campaign as bc, bounded_worker as bw, bounded_metrics as bm,
    dev_campaign as dev, dev_data, dev_submission as submission)
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms2jc2_response.contracts import with_content_hash, sha256_file
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk
from test_cms2jc2_response_science import particles


def mapping():
    samples = {}
    supports = {}
    for j in range(4):
        for side in ("real", "proxy"):
            a = np.linspace(-3, 3, 1001) if j < 2 else np.linspace(-6, -1, 1001)
            samples[f"{side}/all/{j}"] = [a if side == "proxy" else a*1.2]
            supports[f"{side}/all/{j}"] = 1001
    return maps.fit(samples, supports, model_hash="a"*64, samples_hash="b"*64)


def test_monotonic_maps_fallback_zero_validity_and_nontracking_identity():
    m = mapping()
    p = particles()
    for n in maps.CANDIDATES:
        out, counts = maps.apply(p, m, n)
        maps.check_invariants(p, out)
        assert np.all(out.tracking[~out.valid] == 0)
        assert np.all(out.tracking[:, 2:][out.valid[:, 2:]] > 0)
        if n == "B_DZ":
            assert out is p and not counts
        else:
            assert sum(r["pooled_fallback"] for r in counts.values()) == p.valid.sum()
    tr = p.tracking.copy(); tr[p.valid] = 0
    tr[:, 2:][p.valid[:, 2:]] = .1
    zero = Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)
    out, _ = maps.apply(zero, m, "TRACK_FULL")
    np.testing.assert_array_equal(out.tracking[:, :2], 0)
    cell = maps.fit_cell(np.arange(1000.), np.repeat([1., 2.], 500), 1000, 1000)
    assert len(cell["x"]) >= 2 and np.all(np.diff(cell["y"]) >= 0)
    assert maps.fit_cell(np.arange(1000.), np.ones(1000), 1000, 1000)["status"] == "degenerate_source"
    assert maps.fit_cell(np.arange(1000.), np.arange(1000.), 999, 1000)["status"] == "insufficient_unique_jets"
    bad = deepcopy(m); bad["cells"]["all/0"]["x"][1] = bad["cells"]["all/0"]["x"][0]
    with pytest.raises(ValueError, match="monotone"):
        maps.validate_map(with_content_hash(bad))


def test_tail_diagnostics_denominators_missingness_and_guards():
    p = particles()
    row = metrics.tracking_row(p)
    assert row["tails"]["d0"]["count"] == p.valid[:, 0].sum()
    doubled = metrics.merge(deepcopy(row), row)
    assert doubled["correlation"]["count"] == 2*row["correlation"]["count"]
    empty = metrics.tracking_row(p.take([]))
    assert metrics.correlation(empty["correlation"]) == [[None]*6 for _ in range(6)]
    base = dict(score=.2, blocks=dict(core_tv=.2, tail_balanced=.2, joint=.2),
                variable_tv={n: .2 for n in metrics.NAMES}, variable_tail={n: .2 for n in metrics.NAMES})
    scores = {n: deepcopy(base) for n in maps.CANDIDATES}
    assert metrics.choose(scores)[0] == "B_DZ"
    scores["TRACK_FULL"]["score"] = .1
    assert metrics.choose(scores)[0] == "TRACK_FULL"
    scores["TRACK_FULL"]["variable_tail"]["dz"] = .23
    assert metrics.choose(scores)[0] == "B_DZ"


def test_calibration_support_counts_unique_jets_not_replicas():
    p = particles()
    pairs = [SimpleNamespace(identity=f"j{i}", source_group="source", offline=p, hlt=p) for i in range(7)]
    generator = lambda *a, **kw: (p, dict(flags={}))
    row = worker.chunk(pairs, {}, None, None, "calibrate", gen=generator)
    for key in maps.observations(p):
        assert row["supports"]["proxy/"+key] == row["supports"]["real/"+key] == 7
        np.testing.assert_array_equal(row["arrays"]["proxy/"+key], np.tile(maps.observations(p)[key], 21))
    m = maps.fit({k: [v] for k, v in row["arrays"].items()}, row["supports"], model_hash="a"*64, samples_hash="b"*64)
    assert all(c["status"] == "insufficient_unique_jets" for c in m["cells"].values())


def test_endpoint_exposure_and_malformed_map_rejected():
    m, p = mapping(), particles()
    tr = p.tracking.copy(); tr[:, 0][p.valid[:, 0]] = 10000
    out, counts = maps.apply(Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys), m, "TRACK_FULL")
    assert sum(r["outside"] for k, r in counts.items() if k.endswith("/d0")) == p.valid[:, 0].sum()
    assert np.isfinite(out.tracking).all()
    bad = deepcopy(m); bad["cells"]["all/0"]["y"][1] = -100
    with pytest.raises(ValueError, match="monotone"): maps.validate_map(with_content_hash(bad))


@pytest.fixture
def completed_bdz(bounded_root, monkeypatch):
    spec, ctx, put, _, original = bounded_root
    monkeypatch.setattr(bw, "plot_pages", lambda *a, **kw: [])
    for module in (campaign, worker): monkeypatch.setattr(module, "COUNTS", dev_data.COUNTS)
    monkeypatch.setattr(worker, "Measurement", LocalMeasurement)
    put(spec, "bc_acceptance", dict(result=bw.acceptance(ctx, spec)))
    put(spec, "bc_calibrate", dict(result=bw.calibrate(ctx, spec, dict(cpus=1))))
    parent = bc.advance(dev.stage_dir(spec)/"stage_spec.json")
    for t in parent["tasks"][:4]: put(parent, t["task_id"], dict(result=bw.evaluate(ctx, parent, dict(t, cpus=1))))
    report = bw.report(ctx, parent)
    # This tiny ROOT fixture is not intended to reproduce the actual physical
    # winner. Supply a synthetic, internally consistent B_DZ selection so the
    # reuse path (including all real histogram receipts) can be exercised.
    common = deepcopy(report["scores"]["B_DZ"])
    report["scores"] = {n: deepcopy(common) for n in ("B", "B_DZ", "BC")}
    report["scores"]["B"]["score"] += .1
    report["scores"]["BC"]["blocks"]["particle_kinematics"]["score"] += .1
    report["scores"]["BC"]["blocks"]["jet_shape"]["score"] += .1
    winner, reasons = bm.choose(report["scores"])
    assert winner == "B_DZ"
    report.update(selected=winner, guard_failures=reasons, selected_model=bc.registry(parent)["models"][winner])
    put(parent, "bc_select", dict(result=with_content_hash(report)))
    return parent, ctx, put, original


def test_complete_cpu_stages_real_replay_source_pinning_and_no_confirmation(completed_bdz, tmp_path, monkeypatch):
    parent, ctx, put, original = completed_bdz
    # Exercise order-insensitive, multiplicity-preserving old result reader.
    selected = campaign.read_selection(parent)
    assert selected["selected"] == "B_DZ"
    old_files = {p: sha256_file(p) for p in Path(parent["root"]).rglob("*") if p.is_file()}
    spec = campaign.create(parent_spec=dev.stage_dir(parent)/"stage_spec.json", project_dir=tmp_path/"bdz_project",
                           source_commit="c"*40, root=tmp_path/"bdz")
    assert dev.preparation_stage(spec) == dev.preparation_stage(original)
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="PartitionName=debug State=UP" if argv[0] == "scontrol" else str(9000+len(calls))+"\n", stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(spec)
    assert not calls
    assert len(plan["commands"]) == 2
    for row in plan["commands"]:
        assert "--partition=debug" in row["argv"]
        assert not any("gpu" in arg for arg in row["argv"])
    with pytest.raises(PermissionError): submission.submit(spec, execute=True)
    with pytest.raises(FileNotFoundError): campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES[spec["stage"]], reviewed_plan_hash=plan["content_hash"])
    live = [a for a in calls if a[0] == "sbatch" and "--test-only" not in a]
    assert "--dependency=afterok:"+ledger["jobs"]["bz_acceptance"] in live[1]
    accepted = worker.acceptance(ctx, spec)
    assert accepted["serial_process_parity"] and accepted["nonidentity_maps_exercised"]
    put(spec, "bz_acceptance", dict(result=accepted))
    cal = worker.calibrate(ctx, spec, dict(cpus=1))
    assert "arrays" not in cal and cal["durable_particle_arrays"] is False
    put(spec, "bz_calibrate", dict(result=cal))
    compare = campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    assert len(submission.submit(compare)["commands"]) == 5
    for t in compare["tasks"][:4]:
        row = worker.evaluate(ctx, compare, dict(t, cpus=2 if t["params"]["shard"] == 0 else 1))
        put(compare, t["task_id"], dict(result=row))
    monkeypatch.setattr(worker, "plot_pages", lambda *a, **kw: [])
    report = worker.report(ctx, compare)
    assert report["selected"] == "B_DZ"  # no cells have 1,000 jets in this fixture
    assert not report["confirmation_accessed"] and not report["transfer_authorized"]
    put(compare, "bz_select", dict(result=report))
    assert "Frozen development choice: B_DZ" in campaign.render(compare)
    with pytest.raises(PermissionError, match="stops"): campaign.advance(dev.stage_dir(compare)/"stage_spec.json")
    corrupt = deepcopy(spec); corrupt["tasks"][1]["cpus"] = 64
    with pytest.raises(ValueError, match="registration"): dev.validate_stage(with_content_hash(corrupt))
    study = dev.validate_stage(spec)
    changed = deepcopy(study); changed["source"]["files"]["scientific.py"] = "a"*63+"b"
    with pytest.raises(ValueError, match="scientific donor"): campaign.reuse(parent, changed)
    with pytest.raises(PermissionError):
        with dev_data.sample_stream(ctx, "response_confirm") as stream: next(stream)
    assert old_files == {p: sha256_file(p) for p in Path(parent["root"]).rglob("*") if p.is_file()}


def test_cli_commands_registered():
    text = (Path(__file__).parents[1]/"scripts/cms2jc2_response_dev.py").read_text()
    for name in ("create-bdz-tuning", "advance-bdz-tuning", "bdz-results"): assert name in text
