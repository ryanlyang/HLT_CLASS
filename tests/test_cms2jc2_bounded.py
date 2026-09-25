"""Finite comparison and sealed confirmation; poor physics is a completed result."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import (
    bounded_campaign as campaign, bounded_worker as worker, bounded_models as models,
    bounded_metrics as metrics, bounded_data as data, dev_campaign as dev,
    dev_data, dev_submission as submission)
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.response import Generator
from hlt_classification.cms2jc2_response.dev_diagnostics import Histograms, fit_ranges, conditions
from test_cms2jc2_b_tracking import b_root, fitted_b
from test_cms2jc2_frozen_c import fitted, frozen_root, free_disk
from test_cms2jc2_response_science import particles, pairs


def test_donor_semantics_and_unit_scale(fitted_b, fitted):
    snapshot = deepcopy(fitted_b)
    for name in models.CANDIDATES:
        model = models.build(fitted_b, fitted, name)
        models.validate_model(model, fitted_b, fitted)
        base, bi = Generator(fitted_b)(particles(), jet="j", trace=True)
        output, info = Generator(model["runtime_response"])(particles(), jet="j", trace=True)
        models.check_state(base, bi, output, info, tracking_only=name != "BC")
        if name != "BC":
            np.testing.assert_array_equal(base.tracking, output.tracking)
        else:
            for key in fitted_b["modules"]:
                expected = fitted["modules"][key] if key.endswith("_value") else fitted_b["modules"][key]
                assert model["runtime_response"]["modules"][key] == expected
    assert fitted_b == snapshot
    changed = with_content_hash(dict(fitted, parents=dict(fitted["parents"], residual="f"*64)))
    with pytest.raises(ValueError, match="interface"):
        models.build(fitted_b, changed, "BC")
    with pytest.raises(ValueError, match="recipe"):
        models.build(fitted_b, fitted, "B_DZ", dz_scale=.6)


@pytest.mark.parametrize("scale", models.SCALES)
def test_dz_repair_only_changes_registered_quantiles(fitted_b, fitted, scale):
    model = models.build(fitted_b, fitted, "B_DZ", dz_scale=scale)
    runtime = model["runtime_response"]
    for key, module in fitted_b["modules"].items():
        new = runtime["modules"][key]
        if not key.endswith("_value"):
            assert new == module
            continue
        assert new["mean"] == module["mean"] and new["scale"] == module["scale"]
        for old, altered in zip(module["backends"], new["backends"]):
            for a, b in zip(old["backend"]["cells"], altered["backend"]["cells"]):
                q, r = np.asarray(a["quantiles"]), np.asarray(b["quantiles"])
                mask = np.arange(q.shape[1]) % 8 == 5
                np.testing.assert_allclose(r[:, mask], scale*q[:, mask], rtol=0, atol=0)
                np.testing.assert_array_equal(r[:, ~mask], q[:, ~mask])
                assert a["shared_factor"] == b["shared_factor"]
    base = Generator(fitted_b)
    gen = Generator(runtime)
    for p in (particles(), particles().take([1, 0]), particles().take([])):
        for replica in range(3):
            b, bi = base(p, jet="held", replica=replica, trace=True)
            out, info = gen(p, jet="held", replica=replica, trace=True)
            models.check_state(b, bi, out, info, tracking_only=True)
    corrupt = deepcopy(model); corrupt["runtime_response"]["ordering"] = "wrong"
    with pytest.raises(ValueError, match="recipe"):
        models.validate_model(with_content_hash(corrupt), fitted_b, fitted)


def test_metrics_absence_guards_ties_and_finite_stop():
    assert metrics.tv(None, None) == (0., "absent_both")
    row = dict(bins=[1, 0], count=1)
    assert metrics.tv(row, None) == (1., "absent_one")
    assert metrics.tv(row, dict(bins=[0, 1], count=1))[0] == 1.
    base = dict(score=.2, blocks={n: dict(score=.2) for n in (*metrics.BLOCKS, "joint_jet")},
                mean_bias={n: .03 for n in metrics.BIAS_NAMES})
    scores = {n: deepcopy(base) for n in models.CANDIDATES}
    assert metrics.choose(scores)[0] == "B"
    scores["BC"]["score"] = .1
    assert metrics.choose(scores)[0] == "BC"
    scores["BC"]["blocks"]["tracking"]["score"] = .4
    assert metrics.choose(scores)[0] == "B"
    result = dict(base, score=.1)
    ci = dict(sufficient_groups=True, improvement_95pct=[.01, .02])
    assert metrics.confirmation_status(base, result, ci, "BC") == "bounded_confirmation_improvement"
    assert metrics.confirmation_status(base, result, dict(ci, sufficient_groups=False), "BC") == "no_clear_improvement"
    assert metrics.confirmation_status(base, result, ci, "B") == "control_retained"


def test_result_columns_do_not_depend_on_sorted_json_key_order(tmp_path, monkeypatch):
    (tmp_path/"receipts").mkdir()
    (tmp_path/"receipts/bc_select.json").write_text("{}")
    row = dict(jets=10, selected="B", scores={"B": dict(score=.3, blocks={
        "counts_state": dict(score=.1), "jet_shape": dict(score=.4),
        "joint_jet": dict(score=.5), "particle_kinematics": dict(score=.2),
        "tracking": dict(score=.3)})})
    monkeypatch.setattr(campaign, "validate_stage", lambda *a, **kw: None)
    monkeypatch.setattr(dev, "stage_dir", lambda *a: tmp_path)
    monkeypatch.setattr(dev, "product", lambda *a: row)
    text = campaign.render(dict(stage="bounded_compare"))
    line = next(line for line in text.splitlines() if line.startswith("B "))
    assert list(map(float, line.split()[1:])) == [.3, .1, .2, .3, .4, .5]


class LocalMeasurement:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def report(self):
        return dict(wall_seconds=.1, sampled_peak_tree_rss_bytes=1024, synthetic=True)


@pytest.fixture
def bounded_root(b_root, tmp_path, monkeypatch):
    _, ctx, put, before, original = b_root
    for module in (campaign, worker):
        monkeypatch.setattr(module, "COUNTS", dev_data.COUNTS)
    monkeypatch.setattr(campaign, "CONFIRM_JETS", 4)
    monkeypatch.setattr(worker, "Measurement", LocalMeasurement)
    spec = campaign.create(parent_spec=dev.stage_dir(original)/"stage_spec.json", project_dir=tmp_path/"bounded_project",
                           source_commit="f"*40, root=tmp_path/"bounded")
    return spec, ctx, put, before, original


def test_three_stages_real_root_replay_selection_and_sealed_confirmation(bounded_root, monkeypatch):
    spec, ctx, put, before, original = bounded_root
    calls = []
    def scheduler(argv):
        calls.append(argv)
        out = "PartitionName=debug State=UP" if argv[0] == "scontrol" else str(6000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(spec)
    assert not calls and len(plan["commands"]) == 2
    assert all("--partition=debug" in r["argv"] and not any("gpu" in a for a in r["argv"]) for r in plan["commands"])
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True)
    with pytest.raises(FileNotFoundError):
        campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES[spec["stage"]], reviewed_plan_hash=plan["content_hash"])
    live = [r for r in calls if r[0] == "sbatch" and "--test-only" not in r]
    assert len(live) == 2
    assert "--dependency=afterok:"+ledger["jobs"]["bc_acceptance"] in live[1]
    accepted = worker.acceptance(ctx, spec)
    assert accepted["exact_unit_scale_replay"] and accepted["serial_process_parity"]
    put(spec, "bc_acceptance", dict(result=accepted))
    cal = worker.calibrate(ctx, spec, dict(spec["tasks"][1], cpus=1))
    assert cal["dz_scale"] in models.SCALES
    put(spec, "bc_calibrate", dict(result=cal))
    compare = campaign.advance(dev.stage_dir(spec)/"stage_spec.json")
    assert len(submission.submit(compare)["commands"]) == 5
    with pytest.raises(FileNotFoundError):
        campaign.advance(dev.stage_dir(compare)/"stage_spec.json")
    for task in compare["tasks"][:4]:
        result = worker.evaluate(ctx, compare, dict(task, cpus=2 if task["params"]["shard"] == 0 else 1))
        put(compare, task["task_id"], dict(result=result))
    # Exercise one real SVG while all numerical cohorts stay registered.
    from hlt_classification.cms2jc2_response import dev_diagnostics as diag
    def one_page(*args, cohort, **kw):
        if cohort != "all": return
        pages = diag.plot_pages(*args, cohort=cohort, **kw)
        try: yield next(pages)
        finally: pages.close()
    monkeypatch.setattr(worker, "plot_pages", one_page)
    selected = worker.report(ctx, compare)
    assert selected["figures"] and set(selected["scores"]) == set(models.CANDIDATES)
    assert not selected["production_qualified"] and not selected["confirmation_accessed"]
    put(compare, "bc_select", dict(result=selected))
    confirm = campaign.advance(dev.stage_dir(compare)/"stage_spec.json")
    assert not (dev.stage_dir(confirm)/"confirmation_access.json").exists()
    study = dev.validate_stage(confirm)
    # Older tiny ROOT fixture omits the operational raw-root string; supply it
    # only to the direct reader under test. Production create_study records it.
    reader_study = dict(study, imported=dict(study["imported"], cms_root=str(ctx["cms_root"])))
    claim = data.claim_value(confirm, reader_study)
    with pytest.raises(FileNotFoundError):
        list(data.iter_confirmation(confirm, reader_study, claim, shard=0))
    data.acquire_claim(confirm, reader_study)
    with data.stream(confirm, reader_study, claim, shard=0) as stream:
        rows = list(stream)
    assert len(rows) == 1 and rows[0].diagnostic_class is None
    assert rows[0].source_group not in {r["sha256"] for r in ctx["roles"]["files"] if r["response_role"] == "response_fit"}
    original_stream = data.stream
    monkeypatch.setattr(data, "stream", lambda s, st, cl, **kw: original_stream(s, reader_study, cl, **kw))
    for task in confirm["tasks"][:4]:
        row = worker.evaluate(ctx, confirm, dict(task, cpus=1))
        assert row["confirmation_accessed"] and row["candidates"] == list(dict.fromkeys(("B", selected["selected"])))
        put(confirm, task["task_id"], dict(result=row))
    finished = worker.report(ctx, confirm)
    assert finished["status"] in ("control_retained", "no_clear_improvement", "bounded_confirmation_improvement")
    assert finished["tuning_finished"] and not finished["automatic_followup"] and not finished["transfer_authorized"]
    put(confirm, "bf_report", dict(result=finished))
    with pytest.raises(PermissionError, match="ends"):
        campaign.advance(dev.stage_dir(confirm)/"stage_spec.json")
    assert "Frozen choice" in campaign.render(confirm)
    assert before == {p: sha256_file(p) for p in Path(original["root"]).rglob("*") if p.is_file()}


def test_source_protocol_and_access_mutation_fail_closed(bounded_root):
    spec, ctx, put, _, original = bounded_root
    changed = deepcopy(spec); changed["protocol"]["dz_scales"] = [.9]
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(changed))
    changed = deepcopy(spec); changed["tasks"][1]["cpus"] = 64
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(changed))
    study = dev.validate_stage(spec)
    with pytest.raises(PermissionError, match="capability"):
        data.claim_value(spec, study)
    membership = data.build_membership(study)
    bad = with_content_hash(dict(membership, outer_role="response_fit"))
    with pytest.raises(PermissionError, match="population"):
        data.validate_membership(bad, study)
    # Old API has not gained confirmation access.
    with pytest.raises(PermissionError):
        with dev_data.sample_stream(ctx, "response_confirm") as stream:
            next(stream)
    donor_study = deepcopy(study); donor_study["source"]["files"]["scientific.py"] = "d"*64
    with pytest.raises(ValueError, match="scientific source"):
        campaign.reuse(original, donor_study)
