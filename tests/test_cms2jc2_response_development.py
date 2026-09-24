"""CPU development contracts; synthetic Slurm is never remote acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import dev_data as data, dev_campaign as campaign
from hlt_classification.cms2jc2_response import dev_worker as worker, dev_submission as submission
from hlt_classification.cms2jc2_response import dev_diagnostics as diagnostics, splits, storage
from hlt_classification.cms2jc2_response.contracts import artifact, publish, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from test_cms2jc2_response_science import particles, pairs


@pytest.fixture(autouse=True)
def generous_test_disk(monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))


@pytest.fixture
def population(monkeypatch):
    monkeypatch.setattr(splits, "MINIMA", (1, 1, 1))
    counts = dict(location=80, residual=40, evaluation=40, pilot=20)
    monkeypatch.setattr(data, "COUNTS", counts)
    monkeypatch.setattr(worker, "COUNTS", counts)
    rows, assignments = [], []
    for source in range(2):
        for i in range(5):
            path = f"source{source}/{i}.root"
            rows.append(dict(path=path, source=f"source{source}", sha256=f"{source*5+i+1:064x}",
                             tree_key="tree;1", raw_entries=100, selected_entries=100,
                             entry_mask=splits.pack_entries(range(100), 100), original_role="train"))
            assignments.append(("response_fit", "fit_location") if i < 2 else
                               ("response_fit", "fit_residual") if i == 2 else
                               ("response_select", None) if i == 3 else ("response_confirm", None))
    inv = artifact("CMS_INVENTORY", files=rows, selected_entries=1000)
    roles = artifact("ROLES", parents={"inventory": inv["content_hash"]}, minima=[1, 1, 1], seed=splits.SEED,
        files=[dict(r, response_role=a, fit_role=b) for r, (a, b) in zip(rows, assignments)],
        counts=dict(response_fit=600, response_select=200, response_confirm=200))
    return inv, roles


def test_frozen_population_disjoint_nested_mixture_and_replay(population):
    inv, roles = population
    samples = data.build_samples(inv, roles)
    data.validate_samples(samples, inv, roles, canonical=True)
    assert samples == data.build_samples(inv, roles)
    assert [sum(r["selected_entries"] for r in shard) for shard in samples["evaluation_shards"]] == [10]*4
    assert len(samples["members"]["location"]) == 2  # actual signal-like two-file boundary supported
    assert len(samples["members"]["evaluation"]) == 2
    assert not samples["selection_accessed"] and not samples["confirmation_accessed"]
    bad = deepcopy(samples)
    bad["members"]["evaluation"][0]["path"] = roles["files"][3]["path"]
    with pytest.raises(PermissionError, match="escapes response_fit"):
        data.validate_samples(with_content_hash(bad), inv, roles)
    bad = deepcopy(samples)
    bad["evaluation_shards"][1] = bad["evaluation_shards"][0]
    with pytest.raises(ValueError, match="overlaps"):
        data.validate_samples(with_content_hash(bad), inv, roles)


def test_masks_cannot_change_mixture_or_flags(population):
    inv, roles = population
    samples = data.build_samples(inv, roles)
    for key in ("selection_accessed", "confirmation_accessed"):
        bad = with_content_hash(dict(samples, **{key: True}))
        with pytest.raises(ValueError):
            data.validate_samples(bad, inv, roles)
    bad = deepcopy(samples)
    bad["source_capacities"]["source0"] += 1
    with pytest.raises(ValueError, match="mixture"):
        data.validate_samples(with_content_hash(bad), inv, roles)


def test_imports_only_authenticated_cms_not_old_jc2(population, tmp_path):
    inv, roles = population
    project = tmp_path/"project"
    root = tmp_path/"donor"
    root.mkdir()
    for name in data.SEMANTIC_IMPORTS:
        path = project/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic semantic source\n")
    source = artifact("SOURCE", commit="a"*40, executable=True, dirty=False,
                      files={n: sha256_file(project/n) for n in data.SEMANTIC_IMPORTS})
    spec = artifact("PREPARATION_SPEC", parents={"source": source["content_hash"]}, source=source,
                    campaign_root=str(root), cms_root=str(tmp_path/"cms"), jc2_root="DO_NOT_ACCESS")
    publish(root/"preparation_spec.json", spec, "PREPARATION_SPEC")
    publish(root/"cms_inventory.json", inv, "CMS_INVENTORY")
    publish(root/"response_roles.json", roles, "ROLES")
    receipt = artifact("PREPARATION_REPORT", parents={"spec": spec["content_hash"]},
        outputs=[data.file_ref(root/n) for n in data.IMPORT_FILES], remote_cpu_metadata_job_completed=True)
    publish(root/"preparation_report.json", receipt, "PREPARATION_REPORT")
    imported = data.import_preparation(root/"preparation_spec.json", project)
    data.validate_import(imported, project)
    assert not imported["old_jc2_inputs_imported"] and not imported["acceptance_imported"]
    (project/data.SEMANTIC_IMPORTS[0]).write_text("# changed\n")
    with pytest.raises(ValueError, match="semantics changed"):
        data.validate_import(imported, project)


def fixture_stage(tmp_path, monkeypatch, stage="pilot"):
    study = artifact("DEV_STUDY", project_dir=str(tmp_path/"project"), root=str(tmp_path),
        site=dict(partition="debug", account="reu-aisocial", qos="qos_tier3"))
    spec = artifact("DEV_STAGE", root=str(tmp_path), name="test", stage=stage,
                    policy=None if stage == "pilot" else "BASE", b_threads=1,
                    tasks=campaign.tasks(stage, None if stage == "pilot" else "BASE", 1))
    campaign.stage_dir(spec).mkdir(parents=True)
    publish(campaign.stage_dir(spec)/"command_plan.json", campaign.command_plan(spec, study), "DEV_PLAN")
    monkeypatch.setattr(submission, "validate_stage", lambda s, **k: study)
    return spec, study


@pytest.mark.parametrize("stage,count,cpus", [("pilot", 7, 64), ("confirm", 1, 16), ("compare", 17, 47)])
def test_explicit_cpu_dag_no_gpu_no_cross_family_barrier(tmp_path, monkeypatch, stage, count, cpus):
    spec, study = fixture_stage(tmp_path, monkeypatch, stage)
    plan = submission.submit(spec)
    assert plan["gpus"] == 0 and plan["cpu_upper_bound"] == cpus and len(plan["commands"]) == count
    for row in plan["commands"]:
        assert not any("--gpu" in s or "--gres" in s for s in row["argv"])
        assert "--export=NONE" in row["argv"] and "--no-requeue" in row["argv"]
        assert "--partition=debug" in row["argv"]
    if stage == "compare":
        by = {t["task_id"]: t for t in spec["tasks"]}
        assert by["evaluate_A_0"]["depends_on"] == ["fit_AC"]
        assert by["evaluate_A_0"]["dependency_mode"] == "afterany"
        assert by["report_A"]["depends_on"] == [f"evaluate_A_{i}" for i in range(4)]


def test_live_submission_prechecks_exact_dependencies_and_no_duplicates(tmp_path, monkeypatch):
    spec, study = fixture_stage(tmp_path, monkeypatch, "compare")
    calls = []
    def scheduler(argv):
        calls.append(argv)
        out = "PartitionName=debug State=UP" if argv[0] == "scontrol" else str(1000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    plan = submission.submit(spec)
    assert not calls
    with pytest.raises(PermissionError, match="reviewed"):
        submission.submit(spec, execute=True)
    kw = dict(execute=True, authorization_phrase=campaign.PHRASES["compare"], reviewed_plan_hash=plan["content_hash"])
    ledger = submission.submit(spec, **kw)
    assert len(ledger["jobs"]) == 17
    live = [c for c in calls if c[0] == "sbatch" and "--test-only" not in c]
    eval_a = next(c for c in live if c[-1] == "evaluate_A_0")
    assert "--dependency=afterany:"+ledger["jobs"]["fit_AC"] in eval_a
    assert any("--test-only" in c for c in calls[:calls.index(live[0])])
    n = len(calls)
    assert submission.submit(spec, **kw) == ledger and len(calls) == n


def test_ambiguous_submission_and_site_rejection_never_retry(tmp_path, monkeypatch):
    spec, study = fixture_stage(tmp_path, monkeypatch)
    plan = campaign.command_plan(spec, study)
    kw = dict(execute=True, authorization_phrase=campaign.PHRASES["pilot"], reviewed_plan_hash=plan["content_hash"])
    calls = []
    def scheduler(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=0 if argv[0] == "scontrol" or "--test-only" in argv else 1,
                               stdout="PartitionName=debug State=UP" if argv[0] == "scontrol" else "", stderr="lost")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        submission.submit(spec, **kw)
    n = len(calls)
    with pytest.raises(RuntimeError, match="Unacknowledged"):
        submission.submit(spec, **kw)
    assert len(calls) == n and not (campaign.stage_dir(spec)/"submission_ledger.json").exists()


def test_scheduler_scrubs_hidden_resource_options(monkeypatch):
    monkeypatch.setenv("SBATCH_GRES", "gpu:4")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    seen = {}
    monkeypatch.setattr(submission.subprocess, "run", lambda argv, **kw: seen.update(kw))
    submission.scheduler(["sbatch", "--test-only"])
    assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in seen["env"])


def test_site_resource_rejection_happens_before_any_live_job(tmp_path, monkeypatch):
    spec, study = fixture_stage(tmp_path, monkeypatch)
    plan = campaign.command_plan(spec, study)
    calls = []
    def rejected(argv):
        calls.append(argv)
        return SimpleNamespace(returncode=int(argv[0] == "sbatch"), stdout="PartitionName=debug State=UP",
                               stderr="requested memory exceeds partition")
    monkeypatch.setattr(submission, "scheduler", rejected)
    with pytest.raises(PermissionError, match="resource shape"):
        submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["pilot"],
                          reviewed_plan_hash=plan["content_hash"])
    assert all(c[0] != "sbatch" or "--test-only" in c for c in calls)
    assert not (campaign.stage_dir(spec)/"submission").exists()


def test_stages_require_selected_completed_association_without_quality_cut(tmp_path, monkeypatch):
    study = artifact("DEV_STUDY", root=str(tmp_path), project_dir=str(tmp_path/"project"),
                     site=dict(partition="debug", account="reu-aisocial", qos="qos_tier3"))
    path = tmp_path/"study_spec.json"
    publish(path, study, "DEV_STUDY")
    monkeypatch.setattr(campaign, "validate_study", lambda *a, **kw: None)
    pilot = campaign.create_stage(path, stage="pilot", name="pilot")
    parent = campaign.stage_dir(pilot)/"stage_spec.json"
    with pytest.raises(FileNotFoundError):
        campaign.create_stage(path, stage="confirm", name="confirm", parent_spec=parent, policy_id="BASE")
    report = worker.publish_result(pilot, "dummy", artifact("TEST"), "TEST")
    worker.publish_outputs(pilot, "prepare", dict(result=report))
    worker.publish_outputs(pilot, "assoc_BASE", dict(result=report))
    confirm = campaign.create_stage(path, stage="confirm", name="confirm", parent_spec=parent, policy_id="BASE")
    # Zero resolution is a result, never a reason to suppress subsequent fits.
    association = artifact("DEV_ASSOCIATION", jets=20000, policy="BASE", resolved=0)
    report = worker.publish_result(confirm, "association_confirm", association, "DEV_ASSOCIATION")
    worker.publish_outputs(confirm, "association_confirm", dict(result=report))
    compare = campaign.create_stage(path, stage="compare", name="compare",
        parent_spec=campaign.stage_dir(confirm)/"stage_spec.json", b_threads=1)
    campaign.validate_stage(compare)
    assert compare["policy"] == "BASE" and len(compare["tasks"]) == 17
    with pytest.raises(ValueError, match="confirmed policy"):
        campaign.create_stage(path, stage="compare", name="bad",
            parent_spec=campaign.stage_dir(confirm)/"stage_spec.json", policy_id="BOTH")


def test_unestimable_model_reports_scientific_failure_without_job_failure(tmp_path, monkeypatch):
    spec, _ = fixture_stage(tmp_path, monkeypatch, "compare")
    response = artifact("FITTED_RESPONSE", estimable=False)
    products = {("candidate_A_L", "response"): response, ("prepare", "ranges"): {}}
    monkeypatch.setattr(worker, "preparation_stage", lambda s: s)
    monkeypatch.setattr(worker, "product", lambda s, owner, key: products[(owner, key)])
    for i in range(4):
        t = next(t for t in spec["tasks"] if t["task_id"] == f"evaluate_A_{i}")
        products[(t["task_id"], "result")] = worker.evaluate_task({}, spec, t)
    report = worker.report_task({}, spec, next(t for t in spec["tasks"] if t["task_id"] == "report_A"))
    assert report["evaluated_jets"] == 0 and not report["production_qualified"]


def test_worker_failure_and_source_drift_do_not_certify_or_restart(tmp_path, monkeypatch):
    spec, study = fixture_stage(tmp_path, monkeypatch)
    env = artifact("NUMERICAL_ENVIRONMENT", synthetic=True)
    study["numerical_environment"] = env
    monkeypatch.setattr(worker, "validate_stage", lambda s: study)
    monkeypatch.setattr(worker, "allocation", lambda *a: dict(job_id="123"))
    monkeypatch.setattr(submission, "scheduler_identity", lambda *a: dict(test_only=True))
    monkeypatch.setattr(worker, "numerical_environment", lambda: env)
    monkeypatch.setattr(worker, "context", lambda *a: {})
    monkeypatch.setattr(worker, "verified_outputs", lambda *a: {})
    class Measure:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def report(self): return dict(test_only=True)
    monkeypatch.setattr(worker, "Measurement", Measure)
    def failed(*args):
        raise ValueError("injected task failure")
    monkeypatch.setattr(worker, "b_diagnostic", failed)
    with pytest.raises(ValueError, match="injected"):
        worker.run(spec, "B_threads_1")
    with pytest.raises(FileExistsError):
        worker.run(spec, "B_threads_1")
    assert not (campaign.stage_dir(spec)/"receipts/B_threads_1.json").exists()
    def drift(*args):
        monkeypatch.setattr(worker, "numerical_environment", lambda: {})
        return artifact("DEV_B_DIAGNOSTIC", status="fit_completed")
    monkeypatch.setattr(worker, "b_diagnostic", drift)
    with pytest.raises(ValueError, match="environment changed during"):
        worker.run(spec, "B_threads_16")
    assert not (campaign.stage_dir(spec)/"receipts/B_threads_16.json").exists()


def test_allocation_rejects_gpu_local_or_wrong_cpu_count(monkeypatch):
    site = dict(partition="debug", account="reu-aisocial", conda_prefix="/test/env")
    t = dict(cpus=16)
    monkeypatch.setattr(worker.platform, "system", lambda: "Linux")
    for key, value in dict(SLURM_JOB_ID="123", SLURM_JOB_PARTITION="debug", SLURM_JOB_ACCOUNT="reu-aisocial",
        SLURM_CPUS_PER_TASK="16", SLURM_JOB_NUM_NODES="1", CONDA_PREFIX="/test/env", PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", LD_LIBRARY_PATH="/test/env/lib", SLURM_GPUS="", SLURM_JOB_GPUS="",
        SLURM_GPUS_ON_NODE="").items():
        monkeypatch.setenv(key, value)
    assert worker.allocation(dict(site=site), t)["cpus"] == 16
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")  # GPU ID zero is a real GPU, not a zero count!
    with pytest.raises(PermissionError, match="GPUs"):
        worker.allocation(dict(site=site), t)


def test_diagnostics_include_empty_jets_overflow_missing_and_merge():
    registry = diagnostics.fit_ranges(pairs("fit", 4), "a"*64)
    sample = particles()
    empty = sample.take([])
    hist = diagnostics.Histograms(registry)
    parts = []
    for p in (sample, empty, particles(1000)):
        small = diagnostics.Histograms(registry)
        for side in ("offline", "real", "proxy0", "proxy1", "proxy2"):
            hist.add(side, ["all"], sample, p)
            small.add(side, ["all"], sample, p)
        parts.append(small.payload())
    payload = hist.payload()
    merged = diagnostics.merge_payloads(parts)
    assert payload["cells"] == merged["cells"]
    real = payload["cells"]["real/all"]
    assert real["jets"] == 3 and real["empty_jets"] == 1
    assert real["variables"]["jet_multiplicity"]["count"] == 3
    assert real["variables"]["particle_pt"]["bins"][-1] > 0
    rows = diagnostics.comparisons(payload)["proxy0/all"]
    assert rows["particle_pt"]["histogram_tv"] == 0
    assert rows["electron_pt"]["status"] == "unavailable"
    figures = diagnostics.plot_pages(payload, registry, "A_L")
    name, blob = next(figures)
    figures.close()
    assert name.endswith(".svg") and b"<svg" in blob


def test_a_model_survives_later_c_failure_without_certifying_batch(tmp_path, monkeypatch):
    spec, _ = fixture_stage(tmp_path, monkeypatch, "compare")
    env = artifact("NUMERICAL_ENVIRONMENT", synthetic=True)
    study = dict(source=dict(content_hash="a"*64), numerical_environment=env)
    monkeypatch.setattr(worker, "validate_stage", lambda s: study)
    monkeypatch.setattr(worker, "numerical_environment", lambda: env)
    ctx = dict(samples=dict(content_hash="b"*64), review={})
    prepared = []
    def collected(ctx, role, rules, cpus):
        prepared.append(role)
        return None, {"counts": {"jets": 5}}
    monkeypatch.setattr(worker, "collected", collected)
    def fit(*args, **kw):
        if kw["candidate_id"] == "C_L":
            raise RuntimeError("injected C crash")
        return artifact("FITTED_RESPONSE", candidate_id="A_L")
    monkeypatch.setattr(worker, "fit_response", fit)
    with pytest.raises(RuntimeError, match="injected"):
        worker.fitting(ctx, study, spec, spec["tasks"][0])
    assert prepared == ["location", "residual"]
    assert campaign.product(spec, "candidate_A_L", "response")["candidate_id"] == "A_L"
    assert not (campaign.stage_dir(spec)/"receipts/fit_AC.json").exists()
    response_path = tmp_path/"models/test/A_L_response.json"
    response_path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        campaign.verified_outputs(spec, "candidate_A_L")


@pytest.mark.parametrize("cpu64", [False, True])
def test_tiny_real_root_through_fit_evaluation_and_report(tmp_path, monkeypatch, cpu64):
    from test_cms2jc2_response_end_to_end import cms_fixture
    root = tmp_path/"raw"
    root.mkdir()
    inv, _, _ = cms_fixture(root, monkeypatch)
    # One source, two files in each original outer/inner role, sufficient for
    # file-disjoint development. Only miniature capacities are substituted.
    rows = [dict(r, source="fixture") for r in inv["files"]]
    inv = with_content_hash(dict(inv, files=rows))
    role_rows = [dict(r, response_role="response_fit" if i < 6 else "response_select" if i == 6 else "response_confirm",
                     fit_role="fit_location" if i < 4 else "fit_residual" if i < 6 else None) for i, r in enumerate(rows)]
    roles = artifact("ROLES", parents={"inventory": inv["content_hash"]}, files=role_rows,
                     seed=splits.SEED, minima=[1, 1, 1], counts=dict(response_fit=36, response_select=6, response_confirm=6))
    counts = dict(location=12, residual=12, evaluation=12, pilot=4)
    monkeypatch.setattr(data, "COUNTS", counts)
    monkeypatch.setattr(worker, "COUNTS", counts)
    samples = data.build_samples(inv, roles)
    data.validate_samples(samples, inv, roles, canonical=True)
    ctx = dict(cms_root=root, inventory=inv, roles=roles, review=provisional_compatibility(inventory_hash=inv["content_hash"]), samples=samples)
    with data.sample_stream(ctx, "location") as stream:
        ranges = diagnostics.fit_ranges(stream, samples["content_hash"])
    if cpu64:
        from hlt_classification.cms2jc2_response.dev_parallel import prepare_records
        outputs = prepare_records(ctx, ["location", "residual"], campaign.POLICIES["BASE"], workers=1)
        loc, lr = worker.combine_records(outputs["location"], "location", campaign.POLICIES["BASE"])
        res, rr = worker.combine_records(outputs["residual"], "residual", campaign.POLICIES["BASE"])
    else:
        loc, lr = worker.collected(ctx, "location", campaign.POLICIES["BASE"], 1)
        res, rr = worker.collected(ctx, "residual", campaign.POLICIES["BASE"], 1)
    from hlt_classification.cms2jc2_response.response import fit_response
    response = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="A_L",
        review=ctx["review"], rules=campaign.POLICIES["BASE"], budget="SYNTHETIC_DEV", source_hash="a"*64)
    spec, _ = fixture_stage(tmp_path/"output", monkeypatch, "compare")
    monkeypatch.setattr(worker, "preparation_stage", lambda s: s)
    products = {("candidate_A_L", "response"): response, ("prepare", "ranges"): ranges}
    monkeypatch.setattr(worker, "product", lambda s, owner, key: products[(owner, key)])
    for i in range(4):
        t = next(t for t in spec["tasks"] if t["task_id"] == f"evaluate_A_{i}")
        products[(t["task_id"], "result")] = worker.evaluate_task(ctx, spec, t)
    # Production plotting code runs, bounded here to one page to keep tests fast.
    original = worker.plot_pages
    def first_page(*args, **kw):
        pages = original(*args, **kw)
        try:
            if kw.get("cohort") == "all":
                yield next(pages)
        finally:
            pages.close()
    monkeypatch.setattr(worker, "plot_pages", first_page)
    result = worker.report_task(ctx, spec, next(t for t in spec["tasks"] if t["task_id"] == "report_A"))
    assert result["jets"] == 12 and result["all_registered_jets_included"]
    assert result["payload"]["cells"]["real/all"]["jets"] == 12
    assert result["flag_denominator"] == 36
    assert len(result["figures"]) == 1 and not result["six_block_score_computed"]
    assert not result["final_test_accessed"]
    with pytest.raises(PermissionError, match="Forbidden development read"):
        list(data.iter_sample(root, inv, roles, ctx["review"], samples, "response_confirm"))
