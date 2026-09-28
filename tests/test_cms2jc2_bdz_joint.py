"""Bounded joint repair: physical invariants, frozen ancestry and CPU workflow."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    bdz_joint_maps as maps, bdz_joint_campaign as campaign, bdz_joint_worker as worker,
    bdz_joint_metrics as metrics, bdz_audit_metrics as audit,
    bdz_audit_campaign as ac, bdz_audit_worker as aw,
    bdz_campaign as bdz, bdz_worker as old, dev_campaign as dev,
    dev_data, dev_submission as submission)
from hlt_classification.cms2jc2_response.contracts import with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms2jc2_response.c_diagnostic_worker import historical_replay_equal
from test_cms2jc2_bdz_audit import particles
from test_cms2jc2_bdz_tuning import mapping, completed_bdz
from test_cms2jc2_bounded import bounded_root, LocalMeasurement
from test_cms2jc2_b_tracking import b_root
from test_cms2jc2_frozen_c import frozen_root, free_disk


def joint_mapping():
    arrays = {}
    # Each unique jet supplies all four error ranks; replicas do not boost support.
    jets = np.repeat(np.arange(1000), 4)
    errors = np.tile([-6., -4., -2., 0.], 1000)
    sig = np.repeat(np.linspace(.1, 3., 1000), 4)
    for j in range(2):
        arrays[f"proxy/0/{j}"] = [np.column_stack((errors, sig, jets))]
        arrays[f"real/0/{j}"] = [np.column_stack((errors+np.log(2), sig, jets))]
    return maps.fit(arrays, model_hash="a"*64, samples_hash="b"*64)


def test_joint_maps_preserve_sig_when_only_error_scale_changes():
    m, p = joint_mapping(), particles().take([0])
    track = p.tracking.copy()
    track[0] = [np.sinh(1)*np.exp(-2), np.sinh(2)*np.exp(-4), np.exp(-2), np.exp(-4)]
    p = Particles(p.p4, p.charge, p.category, track, p.valid, p.keys)
    out, counts = maps.apply(p, mapping(), m, "JOINT")
    np.testing.assert_allclose(out.tracking, 2*p.tracking)
    np.testing.assert_allclose(out.tracking[:, :2]/out.tracking[:, 2:], p.tracking[:, :2]/p.tracking[:, 2:])
    assert counts["0/0"]["mapped"] == 1
    assert maps.route(m, 0, 0, p.tracking[0], p.valid[0]).startswith("joint_bin_")


def test_pid_safe_does_not_borrow_pool_and_controls_unchanged():
    historical, p, m = mapping(), particles(), joint_mapping()
    before = deepcopy(historical)
    safe, counters = maps.apply(p, historical, m, "PID_SAFE")
    np.testing.assert_array_equal(safe.tracking, p.tracking)
    assert all("pooled_fallback" not in r for r in counters.values())
    assert historical == before
    for name in maps.old.CANDIDATES:
        x, a = maps.apply(p, historical, m, name)
        y, b = maps.old.apply(p, historical, name)
        assert a == b
        np.testing.assert_array_equal(x.tracking, y.tracking)


def test_joint_fallback_is_pairwise_and_zero_value_is_preserved():
    p, m = particles(), joint_mapping()
    out, _ = maps.apply(p, mapping(), m, "JOINT")
    np.testing.assert_array_equal(out.tracking[p.category == 4], p.tracking[p.category == 4])
    assert out.tracking[2, 0] == 0 and out.tracking[2, 2] > 0
    assert out.tracking[2, 1] == p.tracking[2, 1]  # missing dz error
    assert maps.route(m, 4, 1, p.tracking[1], p.valid[1]) == "identity_pid_pair"
    assert maps.route(m, 0, 1, p.tracking[2], p.valid[2]) == "identity_incomplete_pair"
    assert maps.route(m, 0, 0, p.tracking[2], p.valid[2]) == "zero_preserved"
    changed = deepcopy(m)
    for b in changed["cells"]["0/0"]["significance"]:
        b.update(real_jets=0, proxy_jets=0, real_particles=0, proxy_particles=0,
                 status="insufficient_unique_jets", x=[], y=[])
    changed = with_content_hash(changed); maps.validate_map(changed)
    out, counts = maps.apply(p, mapping(), changed, "JOINT")
    np.testing.assert_array_equal(out.tracking[0, [0, 2]], p.tracking[0, [0, 2]])
    assert counts["0/0"]["identity"] == 1


def test_support_uses_unique_jets_not_particles_replicas():
    arr = np.column_stack((np.linspace(-3, 3, 2000), np.zeros(2000)))
    cell = maps.fit_cell(arr, np.tile(arr, (3, 1)), 500)
    assert cell["real_jets"] == cell["proxy_jets"] == 1
    assert cell["status"] == "insufficient_unique_jets"
    m = joint_mapping()
    assert all(b["real_jets"] == 1000 for b in m["cells"]["0/0"]["significance"])
    assert maps.fit_cell(np.column_stack((np.ones(1000), np.arange(1000))),
                         np.column_stack((np.ones(1000), np.arange(1000))), 1000)["status"] == "degenerate_source"


@pytest.mark.parametrize("fault", ["edges", "knots", "support", "policy", "bin_count"])
def test_malformed_map_rejected(fault):
    m = joint_mapping(); c = m["cells"]["0/0"]
    if fault == "edges": c["edges"]["proxy"] = [2., 0., 1.]
    if fault == "knots": c["error"]["y"] = list(reversed(c["error"]["y"]))
    if fault == "support": c["significance"][0]["real_jets"] = 99999
    if fault == "policy": m["fallback"] = "pooled"
    if fault == "bin_count": c["significance"].pop()
    with pytest.raises(ValueError): maps.validate_map(with_content_hash(m))


def test_quantile_ties_endpoint_and_nonfinite_handling():
    m, p = joint_mapping(), particles().take([0])
    tr = p.tracking.copy(); tr[0] = [1e4, 1e4, 1e-12, 1e-12]
    p = Particles(p.p4, p.charge, p.category, tr, p.valid, p.keys)
    out, counts = maps.apply(p, mapping(), m, "JOINT")
    assert counts["0/0"]["error_endpoint"] == counts["0/0"]["significance_endpoint"] == 1
    assert np.isfinite(out.tracking).all()
    assert maps.route(m, 0, 0, p.tracking[0], p.valid[0]) == "joint_bin_0_endpoint"
    with pytest.raises(ValueError, match="Nonfinite"):
        maps.fit_cell(np.array([[np.inf, 0]]), np.array([[1., 0]]), 500)


def fake_scores():
    hist = dict(score=.2, blocks=dict(joint=.2), variable_tv={n: .2 for n in old.metrics.NAMES},
                variable_tail={n: .2 for n in old.metrics.NAMES})
    return {n: dict(score=.2, historical=deepcopy(hist), pid_cells={"4/dz/joint": .2}, excluded_support={})
            for n in maps.CANDIDATES}


def test_finite_selection_guards_pid_regression_and_ties():
    scores = fake_scores()
    assert metrics.choose(scores)[0] == "B_DZ"
    scores["JOINT"]["score"] = .1
    assert metrics.choose(scores)[0] == "JOINT"
    scores["JOINT"]["pid_cells"]["4/dz/joint"] = .3
    choice, rejected = metrics.choose(scores)
    assert choice == "B_DZ" and rejected["JOINT"] == ["pid/4/dz/joint"]
    scores["PID_SAFE"]["excluded_support"] = {"4/dz": [0]}
    with pytest.raises(ValueError, match="eligibility"): metrics.choose(scores)


def test_pooled_moments_have_replica_exposure_denominator():
    row = audit.particles(particles(), jet="one")
    payload = {"file": {"real": row, **{f"{n}/proxy{i}": row for n in maps.CANDIDATES for i in range(3)}}}
    pooled = metrics.pooled_diagnostics(payload)
    assert pooled["JOINT"]["all"]["variables"]["dz"]["contributing_jet_replica_exposures"] == 3
    assert pooled["real"]["all"]["variables"]["dz"]["jets"] == 1


@pytest.mark.parametrize("status", [0, 7])
def test_queue_helper_preserves_stdin_exit_status_and_never_cancels(status):
    path = Path(__file__).parents[1]/"scripts/queue_cms2jc2_bdz_joint.sh"
    text = path.read_text()
    assert not any(s in text for s in ("scancel", "scontrol update", "git push"))
    assert 'MODE="${6:-}"' in text and "--execute" in text
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not bash: pytest.skip("Bash unavailable")
    syntax = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    function = "run_phase() {"+text.split("run_phase() {", 1)[1].split('\nif [ ! -e', 1)[0]
    script = 'set -euo pipefail\n'+function+'\nsleep() { command sleep 0.01; }\n'
    script += f'''result=0
run_phase test bash -c 'read -r x; printf "%s\\n" "$x"; exit {status}' <<'INPUT' || result=$?
saved stdin
INPUT
printf 'result=%s\\n' "$result"
'''
    result = subprocess.run([bash, "--noprofile", "--norc", "-s"], input=script, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "saved stdin" in result.stdout and f"result={status}" in result.stdout


def test_exact_cpu_dag_no_automatic_confirmation():
    assert len(campaign.tasks("joint_gate")) == 2
    assert campaign.tasks("joint_gate")[1]["depends_on"] == ["bj_acceptance"]
    tasks = campaign.tasks("joint_compare")
    assert len(tasks) == 5 and all(t["cpus"] == 36 and t["hours"] == 8 for t in tasks[:4])
    assert tasks[-1]["depends_on"] == [f"bj_eval_{i}" for i in range(4)]
    with pytest.raises(ValueError): campaign.tasks("joint_confirm")
    with pytest.raises(PermissionError): campaign.completed({})


def test_new_chunk_diagnostics_and_calibration_keep_exact_jet_support():
    from hlt_classification.cms2jc2_response.dev_diagnostics import fit_ranges
    p = particles()
    pairs = [SimpleNamespace(identity=f"j{i}", source_group="source", offline=p, hlt=p) for i in range(3)]
    gen = lambda *a, **kw: (p, dict(flags={}))
    ranges = fit_ranges(pairs, "c"*64)
    row = worker.chunk(pairs, {}, mapping(), joint_mapping(), ranges, "evaluate", gen=gen)
    assert set(row["by_file"]["source"]) == set(maps.CANDIDATES)
    assert all(row["by_file"]["source"][n]["cells"]["proxy0/all"]["variables"]["jet_multiplicity"]
               == row["by_file"]["source"]["B_DZ"]["cells"]["proxy0/all"]["variables"]["jet_multiplicity"]
               for n in maps.CANDIDATES)
    example = row["audit"]["source"]["JOINT/proxy0"]["4"]["pairs"]["dz"]["examples"][0]
    assert example["value_route"] == example["error_route"] == "identity_pid_pair"
    cal = worker.chunk(pairs, {}, mapping(), None, ranges, "calibrate", gen=gen)
    assert len(cal["calibration"]) == 3
    assert all(len(c["proxy/4/0"]) == 3*len(c["real/4/0"]) for c in cal["calibration"])
    # Empty/rare populations are disclosed rather than receiving a zero score.
    h, t = worker.summaries(row)
    score = metrics.score(h["JOINT"], t["JOINT"], row["audit"], "JOINT")
    assert score["pid_score"] == 1 and len(score["unavailable_blocks"]) == 3


@pytest.mark.parametrize("change", [dict(projected_peak_bytes=1),
    dict(projected_seconds={"calibrate": 1, "evaluate": 1}), dict(serial_process_parity=False)])
def test_acceptance_evidence_cannot_be_overridden(change, monkeypatch):
    from hlt_classification.cms2jc2_response.contracts import artifact
    spec = dict(stage="joint_gate", content_hash="a"*64, reuse=dict(content_hash="b"*64))
    count = min(32, campaign.COUNTS["residual"])
    m = dict(wall_seconds=1., sampled_peak_tree_rss_bytes=1000)
    row = artifact("BDZ_JOINT_ACCEPTANCE", parents={"stage": spec["content_hash"],
        "protocol": campaign.protocol()["content_hash"], "reuse": "b"*64},
        resource_envelope_ok=True, serial_process_parity=True, nonidentity_probe_exercised=True,
        confirmation_accessed=False, scientific_quality_gate=False, probe_support_is_synthetic=True,
        jets=count, measurements=dict(calibrate=m, evaluate=m), projected_seconds={
            "calibrate": 2*campaign.COUNTS["residual"]/count,
            "evaluate": 2*(campaign.COUNTS["evaluation"]//4)/count},
        projected_peak_bytes=1.5*1000*18+2*maps.MAX_BYTES)
    row = with_content_hash(dict(row, **change))
    monkeypatch.setattr(dev, "product", lambda *a: row)
    with pytest.raises((ValueError, PermissionError), match="acceptance"):
        campaign.accepted(spec)


def test_debug_parent_dispatch_uses_completed_audit_not_mutable_assumptions(monkeypatch):
    from hlt_classification.cms2jc2_response import bdz_audit_debug as debug
    parent = dict(contract="CMS2JC2_RESPONSE_BDZ_AUDIT_DEBUG/v1", study={}, parent_spec={})
    seen = []
    monkeypatch.setattr(debug, "read", lambda spec: seen.append(spec) or dict(historical_choice="TRACK_FULL"))
    monkeypatch.setattr(campaign, "checked_file", lambda ref: ref)
    monkeypatch.setattr(campaign, "load_json", lambda ref: ref)
    monkeypatch.setattr(ac, "completed", lambda spec: ({}, {}, dict(selected="TRACK_FULL")))
    assert campaign.completed(parent)[3]["historical_choice"] == "TRACK_FULL"
    assert seen == [parent]


@pytest.mark.parametrize("fault", ["duplicate", "missing", "budget"])
def test_calibration_never_drops_jets_to_fit_budget(fault, monkeypatch):
    from hlt_classification.cms2jc2_response.dev_diagnostics import fit_ranges
    p = particles()
    pairs = [SimpleNamespace(identity=f"j{i}", source_group="source", offline=p, hlt=p) for i in range(2)]
    ranges = fit_ranges(pairs, "c"*64)
    monkeypatch.setattr(worker, "Generator", lambda *a: lambda *a, **kw: (p, dict(flags={})))
    if fault == "duplicate":
        pairs[1].identity = pairs[0].identity
    elif fault == "missing":
        pairs.pop()
    else:
        monkeypatch.setattr(maps, "MAX_BYTES", 1)
    with pytest.raises(MemoryError if fault == "budget" else ValueError):
        worker.run_pairs(pairs, dict(runtime_response={}), mapping(), None, ranges,
                         mode="calibrate", workers=1, count=2)


def test_full_joint_repair_real_root(completed_bdz, tmp_path, monkeypatch):
    parent, ctx, put, _ = completed_bdz
    for module in (campaign, worker, ac, aw):
        monkeypatch.setattr(module, "COUNTS", dev_data.COUNTS)
    for module in (old, aw, worker):
        monkeypatch.setattr(module, "Measurement", LocalMeasurement)
    monkeypatch.setattr(old, "plot_pages", lambda *a, **kw: [])
    # Fully authenticated, genuine synthetic-ROOT donor chain; no fixture files
    # are presented as real SPORC evidence.
    gate = bdz.create(parent_spec=dev.stage_dir(parent)/"stage_spec.json", project_dir=tmp_path/"bp",
                      source_commit="c"*40, root=tmp_path/"bz")
    put(gate, "bz_acceptance", dict(result=old.acceptance(ctx, gate)))
    put(gate, "bz_calibrate", dict(result=old.calibrate(ctx, gate, dict(cpus=1))))
    comparison = bdz.advance(dev.stage_dir(gate)/"stage_spec.json")
    for t in comparison["tasks"][:4]:
        put(comparison, t["task_id"], dict(result=old.evaluate(ctx, comparison, dict(t, cpus=1))))
    put(comparison, "bz_select", dict(result=old.report(ctx, comparison)))
    donor = ac.create(parent_spec=dev.stage_dir(comparison)/"stage_spec.json", project_dir=tmp_path/"ap",
                      source_commit="d"*40, root=tmp_path/"audit")
    put(donor, "ba_acceptance", dict(result=aw.acceptance(ctx, donor)))
    for t in donor["tasks"][1:5]:
        put(donor, t["task_id"], dict(result=aw.evaluate(ctx, donor, dict(t, cpus=1))))
    put(donor, "ba_report", dict(result=aw.report(ctx, donor)))
    frozen = {p: sha256_file(p) for root in (parent["root"], gate["root"], donor["root"])
              for p in Path(root).rglob("*") if p.is_file()}
    spec = campaign.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir=tmp_path/"jp",
        source_commit="e"*40, root=tmp_path/"joint", partition="debug")
    with pytest.raises(FileNotFoundError): campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    calls = []
    def scheduler(argv):
        calls.append(argv)
        if argv[0] == "sacct":
            return SimpleNamespace(returncode=0, stdout="\n".join(
                f"{j}|COMPLETED|0:0|" for j in argv[argv.index("-j")+1].split(",")), stderr="")
        return SimpleNamespace(returncode=0, stdout="PartitionName=debug State=UP" if argv[0] == "scontrol"
                               else str(9000+len(calls))+"\n", stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(spec)
    assert not calls and plan["gpus"] == 0
    with pytest.raises(PermissionError): submission.submit(spec, execute=True)
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES[spec["stage"]], reviewed_plan_hash=plan["content_hash"])
    live = [a for a in calls if a[0] == "sbatch" and "--test-only" not in a]
    assert len(live) == 2 and "--dependency=afterok:"+ledger["jobs"]["bj_acceptance"] in live[1]
    length = len(calls)
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES[spec["stage"]], reviewed_plan_hash=plan["content_hash"]) == ledger
    assert len(calls) == length
    accepted = worker.acceptance(ctx, spec)
    assert accepted["resource_envelope_ok"] and accepted["nonidentity_probe_exercised"]
    put(spec, "bj_acceptance", dict(result=accepted))
    put(spec, "bj_calibrate", dict(result=worker.calibrate(ctx, spec, dict(cpus=2))))
    compare = campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    comparison_plan = submission.submit(compare)
    assert len(comparison_plan["commands"]) == 5
    comparison_ledger = submission.submit(compare, execute=True,
        authorization_phrase=dev.PHRASES[compare["stage"]], reviewed_plan_hash=comparison_plan["content_hash"])
    live = [a for a in calls if a[0] == "sbatch" and "--test-only" not in a][-5:]
    assert all(not any(a.startswith("--dependency=") for a in argv) for argv in live[:4])
    assert "--dependency=afterok:"+":".join(comparison_ledger["jobs"][f"bj_eval_{i}"] for i in range(4)) in live[-1]
    for t in compare["tasks"][:4]:
        row = worker.evaluate(ctx, compare, dict(t, cpus=2 if t["params"]["shard"] == 0 else 1))
        assert row["historical_replay"]
        put(compare, t["task_id"], dict(result=row))
    monkeypatch.setattr(worker, "plot_pages", lambda *a, **kw: [])
    result = worker.report(ctx, compare)
    put(compare, "bj_select", dict(result=result))
    assert result["selected"] == "B_DZ" and not result["confirmation_accessed"]
    assert "New frozen development choice: B_DZ" in campaign.render(compare, statistics=True)
    with pytest.raises(PermissionError, match="Stop after"): campaign.advance(dev.stage_dir(compare)/"stage_spec.json")
    bad = deepcopy(spec); bad["tasks"][1]["cpus"] = 64
    with pytest.raises(ValueError, match="registration"): dev.validate_stage(with_content_hash(bad))
    changed = deepcopy(dev.validate_stage(spec)); changed["source"]["files"]["scientific.py"] = "f"*64
    with pytest.raises(ValueError, match="scientific source"): campaign.reuse(donor, changed)
    with pytest.raises(PermissionError):
        with dev_data.sample_stream(ctx, "response_confirm") as stream: next(stream)
    assert frozen == {p: sha256_file(p) for p in frozen}
