"""Frozen-C diagnostics are interventions, not newly qualified response models."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import c_diagnostic as campaign
from hlt_classification.cms2jc2_response import c_diagnostic_worker as worker
from hlt_classification.cms2jc2_response import dev_campaign as dev, dev_worker as old_worker
from hlt_classification.cms2jc2_response import dev_data as data, dev_submission as submission, storage, splits
from hlt_classification.cms2jc2_response.c_diagnostic_generation import (
    VARIANTS, DiagnosticGenerator, check_replay, residual_draw, independent_factor)
from hlt_classification.cms2jc2_response.c_diagnostic_metrics import Counters, combine
from hlt_classification.cms2jc2_response.contracts import artifact, publish, with_content_hash, sha256_file, QUANTILES
from hlt_classification.cms2jc2_response.response import collect, fit_response, Generator
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from hlt_classification.cms2jc2_response.residuals import sample
from test_cms2jc2_response_science import particles, pairs


@pytest.fixture(autouse=True)
def free_disk(monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))


@pytest.fixture(scope="module")
def fitted():
    rules = dev.POLICIES["BASE"]
    loc, lr = collect(pairs("location"), rules, cap=7200)
    res, rr = collect(pairs("residual"), rules, cap=7200)
    return fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="C_L",
        review=provisional_compatibility(inventory_hash="a"*64), rules=rules, budget="TEST", source_hash="b"*64)


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_replay_and_topology_all_variants(fitted, variant):
    old, new = Generator(fitted), DiagnosticGenerator(fitted, variant)
    for p in (particles(), particles().take([1, 0]), particles().take([])):
        for replica in range(3):
            expected, old_info = old(p, jet="held", replica=replica, trace=True)
            actual, info = new(p, jet="held", replica=replica)
            check_replay(expected, old_info, actual, info, full=variant == "FULL")
            again, _ = new(p, jet="held", replica=replica)
            np.testing.assert_array_equal(actual.p4, again.p4)
            counter = Counters(); counter.add(info)
            assert counter.value["jet_replicas"] == 1
            assert counter.value["input_particles"] == len(p)
            assert counter.value["emissions"] == len(info["events"])


def backend_fixture(dim=16):
    # Deliberately strong shared variance, so dropping it would be detectable.
    a, b = np.eye(dim)*np.sqrt(.8), np.eye(dim)*np.sqrt(.2)
    cell = dict(key=[0], supported=False, shared_factor=a.tolist(), independent_factor=b.tolist(),
                quantiles=np.repeat(np.linspace(-2., 2., len(QUANTILES))[:, None], dim, axis=1).tolist())
    return artifact("RESIDUAL_BACKEND", cells=[cell], edges=dict(pt=[], eta=[], crowding=[]),
        with_crowding=True, coordinates=[str(i) for i in range(dim)], probabilities=list(QUANTILES))


def test_historical_moments_tolerate_roundoff_but_bins_and_populations_are_exact():
    reference = dict(sum=1., bins=[4, 2], missing=None)
    assert worker.historical_replay_equal(reference, dict(reference, sum=1.+1e-13))
    assert not worker.historical_replay_equal(reference, dict(reference, sum=1.001))
    assert not worker.historical_replay_equal(reference, dict(reference, bins=[3, 3]))


def test_masks_apply_to_every_split_output_and_independent_preserves_covariance():
    backend = backend_fixture()
    kwargs = dict(jet="held", replica=1, object_key="particle", module="emission2")
    base, expected_info = sample(backend, [0, 0, 0, 0], **kwargs)
    for variant in VARIANTS:
        residual, info, detail = residual_draw(backend, [0, 0, 0, 0], variant=variant, factors={}, **kwargs)
        mask = np.zeros(16, bool)
        if variant == "CENTRAL": mask[:] = True
        if variant == "NO_ANGLE": mask[1::8] = True; mask[2::8] = True
        if variant == "NO_PT": mask[0::8] = True
        if variant != "INDEPENDENT":
            np.testing.assert_array_equal(residual[~mask], base[~mask])
            np.testing.assert_array_equal(residual[mask], 0)
            assert info == expected_info
        assert detail == dict(missing_category=False, coarse_bin_backoff=True, low_statistics=True)
    cell = backend["cells"][0]
    factor = independent_factor(cell)
    np.testing.assert_allclose(factor @ factor.T, np.eye(16), atol=1e-14)
    # Non-diagonal within-emission covariance is preserved as well.
    cell["shared_factor"][1][0] = .2
    factor = independent_factor(cell)
    a, b = np.asarray(cell["shared_factor"]), np.asarray(cell["independent_factor"])
    np.testing.assert_allclose(factor @ factor.T, a @ a.T + b @ b.T, atol=1e-14)
    r, info, detail = residual_draw(backend, [5, 0, 0, 0], variant="FULL", factors={}, **kwargs)
    assert r is None and detail["missing_category"] and not detail["low_statistics"]


def test_missing_module_and_backend_are_not_coarse_bin_backoff(fitted):
    missing = deepcopy(fitted)
    del missing["modules"]["emission1_value"]
    gen = DiagnosticGenerator(with_content_hash(missing), "FULL")
    _, info = gen(particles(), jet="held")
    counters = Counters(); counters.add(info)
    assert all(e["missing_value_module"] for e in info["events"])
    assert sum(v["coarse_bin_backoff"] for k, v in counters.value["emission_groups"].items() if k.endswith("/all")) == 0
    missing = deepcopy(fitted)
    missing["modules"]["emission1_value"]["backends"] = []
    _, info = DiagnosticGenerator(with_content_hash(missing), "CENTRAL")(particles(), jet="held")
    assert all(e["missing_state_backend"] for e in info["events"])


@pytest.mark.parametrize("mechanism", ["merged", "split", "additional", "loss"])
def test_traversal_parity_for_all_emission_mechanisms(fitted, mechanism):
    from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
    response = deepcopy(fitted)
    p = particles()
    p = Particles(p4_from_coordinates([2., 10.], [0., .01], [0., .01], [.1, .2]),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    # Fixed categorical modules exercise the traversal with actual inherited
    # conditional predictors. For split, missing value is an explicit identity
    # fallback; the dimension-16 residual mask is separately tested above.
    def categorical_module(state):
        return dict(states=[state], model=None)
    response["modules"]["merge"] = categorical_module([int(mechanism == "merged")])
    response["modules"]["singleton"] = categorical_module([2 if mechanism == "split" else 0 if mechanism == "loss" else 1])
    response["modules"]["additional_count"] = categorical_module([int(mechanism == "additional")])
    if mechanism == "additional":
        response["modules"]["additional_state"] = deepcopy(response["modules"]["emission1_state"])
        response["modules"]["additional_value"] = deepcopy(response["modules"]["emission1_value"])
    if mechanism == "split":
        response["modules"]["emission2_state"] = categorical_module([0, 1, 15, 1, 1, 0, 0, 1])
    response = with_content_hash(response)
    old = Generator(response)
    expected, old_info = old(p, jet="held", trace=True)
    for variant in VARIANTS:
        actual, info = DiagnosticGenerator(response, variant)(p, jet="held")
        check_replay(expected, old_info, actual, info, full=variant == "FULL")
        counters = Counters(); counters.add(info)
        if mechanism == "loss":
            assert counters.value["topology_operations"]["loss"] == 2
        else:
            assert any(e["mechanism"] == mechanism for e in info["events"])


def test_counters_denominators_clipping_and_merge(fitted):
    _, info = DiagnosticGenerator(fitted, "FULL")(particles(), jet="held")
    event = info["events"][0]
    event["response"] = (np.arange(8, dtype=float), np.zeros(8))
    info["events"] = [event]
    a, b = Counters(), Counters()
    a.add(info); b.add(info)
    total = combine(a.value, b.value)
    assert total["jet_replicas"] == 2 and total["emissions"] == 2
    key = next(k for k in total["coordinates"] if k.endswith("/response/delta_eta"))
    assert total["coordinates"][key]["exposures"] == 2
    assert total["coordinates"][key]["clipped"] == 2
    assert total["coordinates"][key]["absolute_correction"]["sum"] == 2
    assert total["coordinates"][key]["before"]["maximum"] == 1
    # No jet/particle identity or raw array may leak into compact accounting.
    import json
    text = json.dumps(total, allow_nan=False)
    assert "held" not in text and "input_keys" not in text


@pytest.fixture
def frozen_root(tmp_path, monkeypatch):
    """Actual tiny ROOT -> fitted C -> original evaluation receipts -> fresh study."""
    from test_cms2jc2_response_end_to_end import cms_fixture
    from hlt_classification.cms2jc2_response import dev_diagnostics as diag
    raw = tmp_path/"raw"; raw.mkdir()
    inv, _, _ = cms_fixture(raw, monkeypatch)
    rows = [dict(r, source="fixture") for r in inv["files"]]
    inv = with_content_hash(dict(inv, files=rows))
    role_rows = [dict(r, response_role="response_fit" if i < 6 else "response_select" if i == 6 else "response_confirm",
                     fit_role="fit_location" if i < 4 else "fit_residual" if i < 6 else None) for i, r in enumerate(rows)]
    roles = artifact("ROLES", parents={"inventory": inv["content_hash"]}, files=role_rows,
        seed=splits.SEED, minima=[1, 1, 1], counts=dict(response_fit=36, response_select=6, response_confirm=6))
    counts = dict(location=12, residual=12, evaluation=12, pilot=4)
    for module in (data, old_worker, worker, campaign):
        monkeypatch.setattr(module, "COUNTS", counts)
    samples = data.build_samples(inv, roles)
    ctx = dict(cms_root=raw, inventory=inv, roles=roles, review=provisional_compatibility(inventory_hash=inv["content_hash"]), samples=samples)
    with data.sample_stream(ctx, "location") as stream:
        ranges = diag.fit_ranges(stream, samples["content_hash"])
    loc, lr = old_worker.collected(ctx, "location", dev.POLICIES["BASE"], 1)
    res, rr = old_worker.collected(ctx, "residual", dev.POLICIES["BASE"], 1)
    response = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="C_L",
        review=ctx["review"], rules=dev.POLICIES["BASE"], budget="SYNTHETIC_DEV", source_hash="a"*64)
    root = tmp_path/"old"; root.mkdir()
    publish(root/"inv.json", inv, "CMS_INVENTORY"); publish(root/"roles.json", roles, "ROLES")
    imported = dict(files={"cms_inventory.json": data.file_ref(root/"inv.json"),
                           "response_roles.json": data.file_ref(root/"roles.json")},
                    preparation_spec=data.file_ref(root/"inv.json"))
    source = artifact("SOURCE", files={"scientific.py": "a"*64}, commit="b"*40)
    response = with_content_hash(dict(response, parents=dict(response["parents"], source=source["content_hash"])))
    study = artifact("DEV_STUDY", root=str(root), project_dir=str(tmp_path/"project_old"), imported=imported,
        review=ctx["review"], numerical_environment={}, source=source,
        site=dict(partition="debug", account="reu-aisocial", qos="qos_tier3"))
    publish(root/"study_spec.json", study, "DEV_STUDY")
    monkeypatch.setattr(dev, "validate_study", lambda *a, **kw: None)
    pilot = dev.create_stage(root/"study_spec.json", stage="pilot", name="pilot_r1")
    def put(spec, owner, objects):
        outputs = {}
        for key, value in objects.items():
            kind = value["contract"].removeprefix("CMS2JC2_RESPONSE_").split("/")[0]
            outputs[key] = old_worker.publish_result(spec, owner+"_"+key, value, kind)
        return old_worker.publish_outputs(spec, owner, outputs)
    put(pilot, "prepare", dict(samples=samples, ranges=ranges))
    assoc = artifact("DEV_ASSOCIATION", jets=20000, policy="BASE", resolved=0)
    put(pilot, "assoc_BASE", dict(result=assoc))
    confirm = dev.create_stage(root/"study_spec.json", stage="confirm", name="confirm",
        parent_spec=dev.stage_dir(pilot)/"stage_spec.json", policy_id="BASE")
    put(confirm, "association_confirm", dict(result=assoc))
    compare = dev.create_stage(root/"study_spec.json", stage="compare", name="compare",
        parent_spec=dev.stage_dir(confirm)/"stage_spec.json", b_threads=1)
    fit = artifact("DEV_FIT", parents={"response": response["content_hash"], "samples": samples["content_hash"]}, candidate="C_L")
    put(compare, "candidate_C_L", dict(response=response, fit=fit))
    for i in range(4):
        t = next(t for t in compare["tasks"] if t["task_id"] == f"evaluate_C_{i}")
        put(compare, t["task_id"], dict(result=old_worker.evaluate_task(ctx, compare, t)))
    # Plot generator itself is exercised below; avoid ~60 redundant old pages.
    monkeypatch.setattr(old_worker, "plot_pages", lambda *a, **kw: [])
    put(compare, "report_C", dict(result=old_worker.report_task(ctx, compare, dict(params=dict(candidate="C_L")))))
    def create_study(**kw):
        destination = Path(kw["root"]); destination.mkdir(exist_ok=False)
        value = with_content_hash(dict(study, root=str(destination), project_dir=str(kw["project_dir"]),
                                      site=dict(study["site"], partition=kw["partition"])))
        publish(destination/"study_spec.json", value, "DEV_STUDY")
        return value
    monkeypatch.setattr(dev, "create_study", create_study)
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    spec = campaign.create(parent_spec=dev.stage_dir(compare)/"stage_spec.json", project_dir=tmp_path/"new_project",
                           source_commit="c"*40, root=tmp_path/"new")
    return spec, ctx, put, before, compare


def test_complete_frozen_reuse_dry_submission_and_corruption(frozen_root, monkeypatch):
    spec, ctx, put, before, donor = frozen_root
    study = dev.validate_stage(spec)
    assert study["site"]["partition"] == "tier3"
    plan = submission.submit(spec)
    assert len(plan["commands"]) == 27 and plan["gpus"] == 0
    assert all("--cpus-per-task=1" in row["argv"] and "--partition=tier3" in row["argv"] for row in plan["commands"])
    assert spec["tasks"][-1]["depends_on"] == ["c_report_"+v for v in VARIANTS]
    calls = []
    def scheduler(argv):
        calls.append(argv)
        out = "PartitionName=tier3 State=UP" if argv[0] == "scontrol" else str(2000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True, reviewed_plan_hash=plan["content_hash"])
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["cdiag"], reviewed_plan_hash=plan["content_hash"])
    assert len(ledger["jobs"]) == 27
    live = [a for a in calls if a[0] == "sbatch" and "--test-only" not in a]
    assert len(live) == 27
    assert "--dependency=afterok:"+ledger["jobs"]["c_acceptance"] in live[1]
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["cdiag"], reviewed_plan_hash=plan["content_hash"]) == ledger
    assert before == {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    bad = deepcopy(spec); bad["variants"] = ["FULL"]
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(bad))
    changed_study = deepcopy(study); changed_study["source"]["files"]["scientific.py"] = "d"*64
    with pytest.raises(ValueError, match="scientific source"):
        campaign.reuse(donor, changed_study)
    path = Path(donor["root"])/dev.verified_outputs(donor, "candidate_C_L")["outputs"]["response"]["relative"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        dev.validate_stage(spec)


def test_tiny_root_all_variants_through_reports(frozen_root, monkeypatch):
    from hlt_classification.cms2jc2_response import dev_diagnostics as diagnostics
    spec, ctx, put, before, donor = frozen_root
    result = worker.acceptance(ctx, spec)
    assert result["full_exact_replay"] and result["jets"] == 3
    put(spec, "c_acceptance", dict(result=result))
    # Render a real SVG page per variant; all conditional numerical metrics stay.
    def one_page(*args, cohort, **kwargs):
        if cohort != "all": return
        gen = diagnostics.plot_pages(*args, cohort=cohort, **kwargs)
        try: yield next(gen)
        finally: gen.close()
    monkeypatch.setattr(worker, "plot_pages", one_page)
    for t in spec["tasks"][1:]:
        result = worker.dispatch(ctx, spec, t)
        put(spec, t["task_id"], dict(result=result))
        assert result["final_test_accessed"] is False
        if t["action"] == "c_report":
            assert result["instrumentation"]["jet_replicas"] == 36
            assert result["figures"] and result["jets"] == 12
            assert result["comparisons"]["proxy0/all"]["jet_mass"]["status"] == "available"
    assert len(result["variants"]) == 5 and result["no_automatic_winner"]
    from hlt_classification.cms2jc2_response.c_diagnostic_results import render
    text = render(spec)
    assert "INDEPENDENT" in text and "EMISSION-LEVEL" in text and "Final test accessed: False" in text
    assert before == {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
