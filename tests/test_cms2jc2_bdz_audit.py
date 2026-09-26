"""Frozen development replay, ratio denominators, bounded forensic evidence."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    bdz_audit_metrics as m, bdz_audit_campaign as campaign, bdz_audit_worker as worker,
    bdz_campaign as bdz, bdz_worker as old, bdz_maps as maps,
    dev_campaign as dev, dev_submission as submission, dev_data)
from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
from hlt_classification.cms2jc2_response.contracts import with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.c_diagnostic_worker import historical_replay_equal
from test_cms2jc2_bdz_tuning import mapping, completed_bdz
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk


def particles():
    return Particles(p4_from_coordinates([1., 2., 3., 4., 5.], [0.]*5, [0.]*5, [0.]*5),
        np.array([1, 1, 1, 0, 1]), np.array([0, 4, 0, 1, 4]),
        np.array([[2., -6., .5, 2.], [-10., 20., .01, .02], [0., 1., .1, 0.], [0., 0., 0., 0.], [0., 0., 1., 1.]]),
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], bool),
        tuple(f"p{i}" for i in range(5)))


def test_exact_ratios_masks_missingness_and_pid_square_attribution():
    r = m.particles(particles(), jet="secret-jet")
    assert r["all"]["pairs"]["dz"]["validity"] == [1, 0, 1, 3]
    assert r["all"]["pairs"]["d0"]["zero_values"] == 2
    sig = r["all"]["variables"]["d0_significance"]
    assert sig["count"] == 4 and sig["jets"] == 1
    assert sig["sum"] == -996 and sig["sumsq"] == 1000016
    assert sig["exceed"] == [2, 2, 1, 1, 1]
    out = m.diagnostics({"file": {"real": r}})
    assert out["real"]["4"]["variables"]["d0_significance"]["pid_square_fraction"] == 1e6/1000016
    assert out["real"]["3"]["variables"]["d0"]["histogram_tv"] is None
    assert len(r["all"]["pairs"]["d0"]["examples"]) == 4
    assert all(e["value_route"] == "observed" for e in r["all"]["pairs"]["d0"]["examples"])
    assert "secret-jet" not in str(r)
    assert np.asarray(r["all"]["pairs"]["d0"]["joint_bins"]).sum() == 4
    assert sum(v["count"] for v in r["all"]["pairs"]["d0"]["by_error"]) == 4


def test_map_routes_and_pre_post_pairs_are_not_changed():
    p, mapping_row = particles(), mapping()
    before = p.tracking.copy()
    out, _ = maps.apply(p, mapping_row, "TRACK_FULL")
    row = m.particles(out, jet="j", base=p, mapping=mapping_row, candidate="TRACK_FULL")
    example = next(x for x in row["4"]["pairs"]["d0"]["examples"] if x["before_mm"][0] != 0)
    assert example["value_route"].startswith("pooled_") and example["error_route"].startswith("pooled_")
    assert example["before_mm"] == p.tracking[1].tolist()
    assert example["tracking_mm"] == out.tracking[1].tolist()
    assert example["significance"] == out.tracking[1, 0]/out.tracking[1, 2]
    np.testing.assert_array_equal(before, p.tracking)
    assert m.route(mapping_row, 4, 0, 0., True, "TRACK_FULL") == "pooled_zero_preserved"
    assert m.route(mapping_row, 4, 0, 1., False, "TRACK_FULL") == "invalid"


def test_merge_top_is_bounded_deterministic_and_jet_support_not_particles():
    rows = [m.particles(particles(), jet=f"j{i}") for i in range(11)]
    forward, reverse = {}, {}
    for row in rows:
        m.merge(forward, row)
    for row in reversed(rows):
        m.merge(reverse, row)
    assert historical_replay_equal(forward, reverse)
    assert len(forward["all"]["pairs"]["d0"]["examples"]) == m.TOP
    assert forward["all"]["variables"]["d0"]["jets"] == 11
    assert forward["all"]["variables"]["d0"]["count"] == 44
    d = m.diagnostics({"f": {"real": forward, "TRACK_FULL/proxy0": forward}})
    assert d["TRACK_FULL/proxy0"]["all"]["variables"]["d0"]["histogram_tv"] == 0
    assert d["real"]["all"]["pairs"]["d0"]["top_square_fraction"] == 5e6/(11*1000016)


def test_replica_pool_counts_exposures_not_independent_jets():
    row = m.particles(particles(), jet="one")
    payload = {"f": {"real": row, **{f"{name}/proxy{r}": row for name in maps.CANDIDATES for r in range(3)}}}
    pooled = m.pooled_proxy_diagnostics(payload)
    v = pooled["TRACK_FULL"]["all"]["variables"]["d0_significance"]
    assert v["count"] == 12 and v["contributing_jet_replica_exposures"] == 3
    assert "jets" not in v
    assert pooled["real"]["all"]["variables"]["d0_significance"]["jets"] == 1


def test_extreme_ties_use_same_key_order_within_and_across_jets():
    p = particles()
    n = 12
    q = Particles(np.repeat(p.p4[:1], n, axis=0), np.ones(n), np.zeros(n),
                  np.repeat(p.tracking[:1], n, axis=0), np.ones((n, 4), bool), tuple(f"key{i}" for i in range(n)))
    examples = m.particles(q, jet="j")["all"]["pairs"]["d0"]["examples"]
    expected = sorted(hashlib.sha256(k.encode()).hexdigest() for k in q.keys)[:m.TOP]
    assert [e["particle_hash"] for e in examples] == expected


def test_tail_overflow_zeros_and_correlations():
    assert m.indices(np.array([0., 1e-20, 1., 1e8]), m.MAG_EDGES, log=True).tolist() == [0, 0, 8, len(m.MAG_EDGES)]
    r = m.variable(np.array([0., 1e6]), 4)
    assert r["bins"][-1] == 1 and r["exceed_sumsq"][-1] == 1e12
    assert m.pearson(m.correlation(np.array([1., 2., 3.]), np.array([2., 4., 6.]))) == pytest.approx(1.)
    assert m.pearson(m.correlation(np.ones(3), np.ones(3))) is None
    with pytest.raises(ValueError, match="Nonfinite"):
        m.moments(np.array([float("inf")]))
    with pytest.raises(ValueError, match="shape"):
        m.merge(dict(bins=[1]), dict(bins=[1, 2]))


@pytest.mark.parametrize("field", ["jets", "flags", "ordered_identities", "tracking"])
def test_historical_replay_is_fail_closed(field):
    row = dict(jets=4, flags=dict(clamp=1), ordered_identities="abc", tracking=dict(sum=1.))
    corrupt = deepcopy(row)
    corrupt[field] = 9
    with pytest.raises(ValueError, match="replay differs"):
        worker.replay(row, corrupt)


@pytest.mark.parametrize("spec", [{}, dict(contract=campaign.CONTRACT, stage="bdz_audit"),
                                   dict(contract="CMS2JC2_RESPONSE_BDZ_STAGE/v1", stage="bdz_gate")])
def test_only_completed_development_comparison_is_allowed(spec):
    with pytest.raises(PermissionError, match="completed B_DZ"):
        campaign.completed(spec)


def test_queue_surface_is_separate_cpu_only_and_bounded():
    tasks = campaign.tasks()
    assert len(tasks) == 6 and tasks[0]["cpus"] == 2
    assert all(t["depends_on"] == ["ba_acceptance"] and t["cpus"] == 36 for t in tasks[1:5])
    assert tasks[-1]["depends_on"] == [f"ba_eval_{i}" for i in range(4)]
    script = (Path(__file__).parents[1]/"scripts/queue_cms2jc2_bdz_audit.sh").read_text()
    assert 'MODE="${4:-}"' in script and '"--execute"' in script
    assert not any(s in script for s in ("scancel", "scontrol update", "git push"))
    assert campaign.protocol()["selection"] is False


def test_full_replay_from_real_root_preserves_donor(completed_bdz, tmp_path, monkeypatch):
    parent, ctx, put, _ = completed_bdz
    monkeypatch.setattr(campaign, "COUNTS", dev_data.COUNTS)
    monkeypatch.setattr(worker, "COUNTS", dev_data.COUNTS)
    monkeypatch.setattr(worker, "Measurement", LocalMeasurement)
    gate = bdz.create(parent_spec=dev.stage_dir(parent)/"stage_spec.json", project_dir=tmp_path/"bproj",
                      source_commit="c"*40, root=tmp_path/"bz")
    put(gate, "bz_acceptance", dict(result=old.acceptance(ctx, gate)))
    put(gate, "bz_calibrate", dict(result=old.calibrate(ctx, gate, dict(cpus=1))))
    donor = bdz.advance(dev.stage_dir(gate)/"stage_spec.json")
    for task in donor["tasks"][:4]:
        put(donor, task["task_id"], dict(result=old.evaluate(ctx, donor, dict(task, cpus=1))))
    monkeypatch.setattr(old, "plot_pages", lambda *a, **kw: [])
    put(donor, "bz_select", dict(result=old.report(ctx, donor)))
    before = {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    spec = campaign.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir=tmp_path/"aproj",
                           source_commit="d"*40, root=tmp_path/"audit")
    study = campaign.validate_stage(spec)
    assert dev.preparation_stage(spec) == dev.preparation_stage(donor)
    altered = deepcopy(study); altered["source"]["files"]["scientific.py"] = "e"*64
    with pytest.raises(ValueError, match="scientific source"):
        campaign.reuse(donor, altered)
    bad = deepcopy(spec); bad["tasks"][1]["cpus"] = 64
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(bad))
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="PartitionName=tier3 State=UP" if argv[0] == "scontrol" else str(8000+len(calls))+"\n", stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(spec)
    assert not calls and len(plan["commands"]) == 6 and plan["gpus"] == 0
    assert all("--partition=tier3" in r["argv"] for r in plan["commands"])
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True)
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_audit"], reviewed_plan_hash=plan["content_hash"])
    sent = [x for x in calls if x[0] == "sbatch" and "--test-only" not in x]
    assert len(sent) == 6 and "--dependency=afterok:"+ledger["jobs"]["ba_acceptance"] in sent[1]
    n = len(calls)
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["bdz_audit"], reviewed_plan_hash=plan["content_hash"]) == ledger
    assert len(calls) == n
    acceptance = worker.acceptance(ctx, spec)
    assert acceptance["resource_envelope_ok"] and acceptance["serial_process_parity"]
    put(spec, "ba_acceptance", dict(result=acceptance))
    # Frozen fixture maps have low support; also exercise genuinely nonidentity
    # transforms through spawned children without publishing/tuning any map.
    _, model, ranges, reg = worker.inputs(spec)
    with dev_data.sample_stream(ctx, "evaluation", shard=0) as stream:
        pairs = tuple(stream)
    serial = worker.run_pairs(pairs, model, mapping(), ranges, workers=1, count=len(pairs))
    parallel = worker.run_pairs(pairs, model, mapping(), ranges, workers=2, count=len(pairs))
    assert all(historical_replay_equal(a, b) for a, b in zip(serial, parallel))
    real_product = dev.product
    with monkeypatch.context() as patch:
        def rejected(s, owner, key):
            row = real_product(s, owner, key)
            return with_content_hash(dict(row, resource_envelope_ok=False)) if owner == "ba_acceptance" else row
        patch.setattr(dev, "product", rejected)
        with pytest.raises(PermissionError, match="acceptance"):
            worker.accepted(spec, reg)
    for task in spec["tasks"][1:5]:
        row = worker.evaluate(ctx, spec, dict(task, cpus=2 if task["params"]["shard"] == 0 else 1))
        assert row["historical_replay"]
        put(spec, task["task_id"], dict(result=row))
    report = worker.report(ctx, spec)
    assert report["jets"] == dev_data.COUNTS["evaluation"] and len(report["figures"]) == 2
    assert not report["confirmation_accessed"] and not report["production_qualified"]
    assert "scores" not in report and "selected" not in report
    put(spec, "ba_report", dict(result=report))
    assert campaign.read(spec) == report
    assert "no new selection" in m.render(report)
    assert report["pooled_proxy_diagnostics"]["TRACK_FULL"]["all"]["variables"]["d0"]["count"] >= 0
    with pytest.raises(PermissionError):
        with dev_data.sample_stream(ctx, "response_confirm") as stream:
            next(stream)
    assert before == {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    # Tampering is rejected even for a read-only report, via the frozen receipt.
    receipt = dev.verified_outputs(spec, "ba_report")
    path = Path(spec["root"])/receipt["outputs"]["result"]["relative"]
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="Corrupt"):
        campaign.read(spec)
