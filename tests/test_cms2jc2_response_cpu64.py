"""CPU36/64 change execution only; these are not RC speed measurements."""
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import dev_parallel as p, dev_restart as restart
from hlt_classification.cms2jc2_response import dev_campaign as campaign, dev_worker as worker, dev_data as data
from hlt_classification.cms2jc2_response import storage
from hlt_classification.cms2jc2_response.contracts import artifact, publish, with_content_hash
from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response.records import Reservoir
from hlt_classification.cms2jc2_response.response import collect
from hlt_classification.cms2jc2_response.topology import Record
from hlt_classification.cms2jc2_response.features import features
from test_cms2jc2_response_science import pairs, particles
from test_cms2jc2_response_development import population  # noqa: F401


@pytest.fixture(autouse=True)
def generous_disk(monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))


def assert_records(a, b):
    assert a.modules() == b.modules()
    for module in a.modules():
        aa, bb = a.arrays(module), b.arrays(module)
        for x, y in zip(aa, bb):
            np.testing.assert_array_equal(x, y)


def test_bottom_hash_merge_preserves_pruned_records_and_exact_weights():
    records = [(f"jet{i}", Record("singleton", "p", features(particles())[i % 2], (i % 3,))) for i in range(301)]
    expected = Reservoir(144)
    for jet, record in records:
        expected.add(jet, record)
    chunks = []
    for start in range(0, len(records), 7):
        part = Reservoir(144)
        for jet, record in records[start:start+7]:
            part.add(jet, record)
        chunks.append(part)
    for order in (chunks, list(reversed(chunks))):
        actual = Reservoir(144)
        for part in order:
            p.merge_reservoir(actual, part)
        assert actual.report() == expected.report()
        assert_records(actual, expected)
    with pytest.raises(ValueError, match="quota"):
        p.merge_reservoir(Reservoir(144), Reservoir(72))


@pytest.mark.parametrize("workers,chunk", [(1, 1), (1, 7), (2, 3)])
def test_chunked_records_exactly_match_legacy_including_unresolved(workers, chunk, capsys):
    rules = policy(search_nodes=1)  # deterministic unresolved results also preserved
    files = [("location", "a", list(pairs("a", 25)), 144),
             ("residual", "b", list(pairs("b", 13)), 144)]
    expected = [collect(rows, rules, cap=cap, progress_every=9999) for _, _, rows, cap in files]
    actual = p.collect_loaded(deepcopy(files), rules, workers=workers, chunk_jets=chunk)
    for (ar, ap), (er, ep) in zip(actual, expected):
        assert ap == ep
        assert_records(ar, er)
    assert "completed_jets=38/38" in capsys.readouterr().out


def test_spawn_resolved_records_and_combined_reports_match_legacy(monkeypatch):
    rules = policy()
    files = [("location", "a", list(pairs("a", 27)), 144),
             ("location", "b", list(pairs("b", 21)), 144),
             ("residual", "c", list(pairs("c", 17)), 288)]
    expected = [collect(rows, rules, cap=cap, progress_every=9999) for _, _, rows, cap in files]
    actual = p.collect_loaded(deepcopy(files), rules, workers=2, chunk_jets=4)
    monkeypatch.setattr(worker, "COUNTS", dict(location=48, residual=17))
    for indexes, role in [([0, 1], "location"), ([2], "residual")]:
        ar, ap = worker.combine_records([actual[i] for i in indexes], role, rules)
        er, ep = worker.combine_records([expected[i] for i in indexes], role, rules)
        assert ap == ep
        assert_records(ar, er)


def test_bounded_scheduler_refills_out_of_order_and_heartbeats(monkeypatch):
    submitted, finished, heartbeats = [], [], []
    class Pool:
        def submit(self, fn, arg):
            f = Future()
            f.set_result(fn(arg))
            submitted.append((arg, f))
            assert len(submitted)-len(finished) <= 4
            return f
    first = True
    def wait(futures, **kw):
        nonlocal first
        if first:
            first = False
            return set(), set(futures)
        chosen = next(f for _, f in reversed(submitted) if f in futures)
        return {chosen}, set(futures)-{chosen}
    monkeypatch.setattr(p, "wait", wait)
    for key, value in p.bounded_results(Pool(), lambda n: n*n, ((i, i) for i in range(15)),
                                      window=4, heartbeat_seconds=.01, on_wait=heartbeats.append):
        assert value == key*key
        finished.append(key)
    assert sorted(finished) == list(range(15)) and finished[0] == 3 and heartbeats == [4]


def test_scheduler_propagates_failure():
    class Pool:
        def submit(self, fn, arg):
            f = Future()
            f.set_exception(ValueError("bad particle"))
            return f
    with pytest.raises(ValueError, match="bad particle"):
        list(p.bounded_results(Pool(), None, [(0, None)], window=2, heartbeat_seconds=.1, on_wait=lambda n: None))


@pytest.mark.parametrize("cpus", [36, 64])
def test_process_capacity_not_capped_by_number_of_files(monkeypatch, cpus, capsys):
    capacities = []
    class Pool:
        def __init__(self, *, max_workers, **kw):
            capacities.append(max_workers)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, arg):
            f = Future()
            f.set_result(fn(arg))
            return f
    monkeypatch.setattr(p, "ProcessPoolExecutor", Pool)
    # Two files can expose >64 independent work items. No actual 64-process
    # benchmark is claimed by this scheduler test.
    p.collect_loaded([("location", "a", list(pairs("a", 35)), 144),
                      ("residual", "b", list(pairs("b", 35)), 144)], policy(), workers=cpus, chunk_jets=1)
    assert capacities == [cpus]
    assert f"phase=records_cpu{cpus}" in capsys.readouterr().out


@pytest.mark.parametrize("cpus", [1, 36])
def test_load_once_per_file_and_combined_role_scheduling(monkeypatch, capsys, cpus):
    files = {("location", "a"): list(pairs("a", 6)), ("residual", "b"): list(pairs("b", 5))}
    ctx = dict(samples=dict(members={r: [dict(path=f, selected_entries=len(rows))]
                                    for (r, f), rows in files.items()}))
    calls = []
    def load(args):
        _, role, path = args
        calls.append((role, path))
        return deepcopy(files[role, path])
    monkeypatch.setattr(p, "_load_file", load)
    capacities = []
    class Pool:
        def __init__(self, *, max_workers, **kw):
            capacities.append(max_workers)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, arg):
            future = Future()
            future.set_result(fn(arg))
            return future
    monkeypatch.setattr(p, "ProcessPoolExecutor", Pool)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    result = p.prepare_records(ctx, ["location", "residual"], policy(), workers=cpus)
    assert calls == [("location", "a"), ("residual", "b")]
    assert result["location"][0][1] == collect(files["location", "a"], policy())[1]
    log = capsys.readouterr().out
    tag = "cpu36" if cpus == 36 else "cpu64"
    assert f"phase=load_{tag}" in log and f"phase=records_{tag}" in log and "completed_jets=11/11" in log
    assert capacities == ([2, 36] if cpus == 36 else [])
    profile = p.execution_profile(36 if cpus == 36 else 64)
    monkeypatch.setattr(p, "execution_profile", lambda *args: dict(profile, loaded_pair_payload_limit_bytes=1))
    with pytest.raises(MemoryError, match="payload"):
        p.prepare_records(ctx, ["location", "residual"], policy(), workers=cpus)


@pytest.mark.parametrize("cpus,partition,upper", [(36, "tier3", 87), (64, "debug", 143)])
def test_parallel_plan_retains_legacy_and_b_thread_limit(tmp_path, cpus, partition, upper):
    study = dict(project_dir=str(tmp_path), site=dict(partition=partition, account="reu-aisocial", qos="qos_tier3"))
    spec = artifact(f"DEV_STAGE{cpus}", stage="compare", root=str(tmp_path), name=f"compare{cpus}_r1",
                    tasks=campaign.tasks("compare", "SEARCH", 1, preprocessing_cpus=cpus))
    plan = campaign.command_plan(spec, study)
    assert len(plan["commands"]) == 17 and plan["cpu_upper_bound"] == upper and plan["gpus"] == 0
    assert plan["allocated_cpu_hour_upper_bound"] == 2*cpus*24+12*12+3*2
    for task, command in zip(spec["tasks"], plan["commands"]):
        assert f"--partition={partition}" in command["argv"]
        assert "--nodes=1" in command["argv"] and "--ntasks=1" in command["argv"]
        assert not any("gpu" in arg for arg in command["argv"])
        if task["action"] == "fit":
            assert f"--cpus-per-task={cpus}" in command["argv"] and "--mem=128G" in command["argv"]
            assert "--time=24:00:00" in command["argv"]
        else:
            assert "--cpus-per-task=1" in command["argv"]
    assert spec["tasks"][1]["params"]["threads"] == 1
    assert spec["tasks"][0]["params"]["threads"] == 16
    assert campaign.tasks("compare", "SEARCH", 1)[0]["cpus"] == 16
    with pytest.raises(ValueError, match="single-thread"):
        campaign.tasks("compare", "SEARCH", 16, preprocessing_cpus=cpus)


def test_reuse_completed_confirmation_not_models_and_reject_scientific_changes(population, tmp_path, monkeypatch):
    inv, roles = population
    for name, value in [("inv", inv), ("roles", roles)]:
        publish(tmp_path/(name+".json"), value, "CMS_INVENTORY" if name == "inv" else "ROLES")
    imported = dict(files={"cms_inventory.json": data.file_ref(tmp_path/"inv.json"),
                           "response_roles.json": data.file_ref(tmp_path/"roles.json")})
    donor = artifact("DEV_STUDY", root=str(tmp_path/"old"), imported=imported, review={}, numerical_environment={},
        source=dict(files={"matcher.py": "a"*64, "src/hlt_classification/cms2jc2_response/dev_worker.py": "b"*64}))
    study = with_content_hash(dict(donor, root=str(tmp_path/"new"), source=dict(files={
        "matcher.py": "a"*64, "src/hlt_classification/cms2jc2_response/dev_worker.py": "c"*64})))
    parent = artifact("DEV_STAGE", stage="confirm", policy="SEARCH")
    samples = data.build_samples(inv, roles)
    report = artifact("DEV_ASSOCIATION", parents={"samples": samples["content_hash"]},
                      jets=120, resolved=0, policy="SEARCH", rules=campaign.POLICIES["SEARCH"])
    ranges = artifact("DEV_RANGES", parents={"samples": samples["content_hash"]})
    monkeypatch.setattr(campaign, "validate_stage", lambda *a, **kw: donor)
    monkeypatch.setattr(campaign, "preparation_stage", lambda s: s)
    products = {("association_confirm", "result"): report, ("prepare", "samples"): samples, ("prepare", "ranges"): ranges}
    monkeypatch.setattr(campaign, "product", lambda s, owner, key: products[(owner, key)])
    monkeypatch.setattr(campaign, "verified_outputs", lambda *a: dict(content_hash="d"*64))
    reuse = restart.reuse_evidence(parent, study)
    assert not reuse["confirmation_rerun"] and not reuse["old_models_imported"] and not reuse["old_jobs_modified"]
    for field in ("review", "numerical_environment", "imported"):
        bad = deepcopy(study)
        bad[field] = {"wrong": True}
        with pytest.raises(ValueError, match="changed CMS"):
            restart.reuse_evidence(parent, bad)
    bad = deepcopy(study)
    bad["source"]["files"]["matcher.py"] = "f"*64
    with pytest.raises(ValueError, match="scientific source"):
        restart.reuse_evidence(parent, bad)
    bad = dict(parent, stage="compare")
    with pytest.raises(ValueError, match="confirmation"):
        restart.reuse_evidence(bad, study)


@pytest.mark.parametrize("cpus", [36, 64])
def test_parallel_fitting_dispatches_both_roles_once(monkeypatch, tmp_path, cpus):
    spec = artifact(f"DEV_STAGE{cpus}", stage="compare", policy="SEARCH", root=str(tmp_path), name=f"compare{cpus}_r1",
                    tasks=campaign.tasks("compare", "SEARCH", 1, preprocessing_cpus=cpus))
    env = artifact("NUMERICAL_ENVIRONMENT", synthetic=True)
    study = dict(source=dict(content_hash="a"*64), numerical_environment=env)
    calls = []
    def prepare(ctx, roles, rules, *, workers):
        calls.append((roles, workers))
        return {r: [] for r in roles}
    monkeypatch.setattr(p, "prepare_records", prepare)
    monkeypatch.setattr(worker, "combine_records", lambda *args: (None, dict(counts=dict(jets=1))))
    monkeypatch.setattr(worker, "collected", lambda *args: pytest.fail("legacy preparation called"))
    monkeypatch.setattr(worker, "validate_stage", lambda *a: study)
    monkeypatch.setattr(worker, "numerical_environment", lambda: env)
    monkeypatch.setattr(worker, "fit_response", lambda *a, **kw: artifact("FITTED_RESPONSE", candidate_id=kw["candidate_id"]))
    worker.fitting(dict(review={}, samples=dict(content_hash="c"*64)), study, spec, spec["tasks"][0])
    assert calls == [(["location", "residual"], cpus)]
    assert campaign.product(spec, "candidate_A_L", "response")["candidate_id"] == "A_L"
    assert campaign.product(spec, "candidate_C_L", "response")["candidate_id"] == "C_L"


@pytest.mark.parametrize("cpus,partition", [(36, "tier3"), (64, "debug")])
def test_full_reuse_publication_validation_dry_and_fake_live_plan(population, tmp_path, monkeypatch, cpus, partition):
    from hlt_classification.cms2jc2_response import dev_submission as submission
    from hlt_classification.cms2jc2_response.contracts import sha256_file
    inv, roles = population
    old = tmp_path/"old"
    old.mkdir()
    for name, value, kind in [("inv", inv, "CMS_INVENTORY"), ("roles", roles, "ROLES")]:
        publish(old/(name+".json"), value, kind)
    imported = dict(files={"cms_inventory.json": data.file_ref(old/"inv.json"),
                           "response_roles.json": data.file_ref(old/"roles.json")},
                    preparation_spec=data.file_ref(old/"inv.json"))
    site = dict(partition="debug", account="reu-aisocial", qos="qos_tier3")
    study = artifact("DEV_STUDY", root=str(old), project_dir=str(tmp_path/"project_old"),
        imported=imported, review={}, numerical_environment={}, site=site,
        source=dict(files={"scientific.py": "a"*64}, commit="b"*40))
    publish(old/"study_spec.json", study, "DEV_STUDY")
    monkeypatch.setattr(campaign, "validate_study", lambda *a, **kw: None)
    pilot = campaign.create_stage(old/"study_spec.json", stage="pilot", name="pilot_r1")
    samples = data.build_samples(inv, roles)
    ranges = artifact("DEV_RANGES", parents={"samples": samples["content_hash"]})
    sample_path = worker.publish_result(pilot, "samples", samples, "DEV_SAMPLES")
    range_path = worker.publish_result(pilot, "ranges", ranges, "DEV_RANGES")
    worker.publish_outputs(pilot, "prepare", dict(samples=sample_path, ranges=range_path))
    result = artifact("DEV_ASSOCIATION", parents={"samples": samples["content_hash"]},
                      jets=120, resolved=0, policy="SEARCH", rules=campaign.POLICIES["SEARCH"])
    report_path = worker.publish_result(pilot, "assoc_SEARCH", result, "DEV_ASSOCIATION")
    worker.publish_outputs(pilot, "assoc_SEARCH", dict(result=report_path))
    confirm = campaign.create_stage(old/"study_spec.json", stage="confirm", name="confirm_search_r1",
        parent_spec=campaign.stage_dir(pilot)/"stage_spec.json", policy_id="SEARCH")
    report_path = worker.publish_result(confirm, "association_confirm", result, "DEV_ASSOCIATION")
    worker.publish_outputs(confirm, "association_confirm", dict(result=report_path))
    before = {p: sha256_file(p) for p in old.rglob("*") if p.is_file()}
    def create_study(**kw):
        root = Path(kw["root"])
        root.mkdir(exist_ok=False)
        value = with_content_hash(dict(study, root=str(root), project_dir=str(kw["project_dir"]),
                                      site=dict(site, partition=kw["partition"])))
        publish(root/"study_spec.json", value, "DEV_STUDY")
        return value
    monkeypatch.setattr(campaign, "create_study", create_study)
    create = getattr(restart, f"create_compare{cpus}")
    spec = create(parent_spec=campaign.stage_dir(confirm)/"stage_spec.json",
        project_dir=tmp_path/"project_new", source_commit="c"*40, root=tmp_path/"new")
    assert spec["contract"] == f"CMS2JC2_RESPONSE_DEV_STAGE{cpus}/v1"
    assert spec["execution"] == p.execution_profile(cpus)
    assert spec["reuse"]["contract"] == f"CMS2JC2_RESPONSE_DEV_CPU{cpus}_REUSE/v1"
    assert not spec["reuse"]["old_models_imported"] and not spec["reuse"]["old_jobs_modified"]
    campaign.validate_stage(spec)
    plan = submission.submit(spec)
    assert all(f"--partition={partition}" in row["argv"] for row in plan["commands"])
    # All resource admission checks precede any live sbatch calls.
    rejected = []
    def reject_shape(argv):
        rejected.append(argv)
        if argv[0] == "scontrol":
            return SimpleNamespace(returncode=0, stdout=f"PartitionName={partition} State=UP", stderr="")
        assert "--test-only" in argv
        return SimpleNamespace(returncode=1, stdout="", stderr="Requested node configuration is not available")
    monkeypatch.setattr(submission, "scheduler", reject_shape)
    with pytest.raises(PermissionError, match="Site rejected"):
        submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["compare"],
                          reviewed_plan_hash=plan["content_hash"])
    assert len(rejected) == 2
    assert not (campaign.stage_dir(spec)/"submission_ledger.json").exists()
    calls = []
    def scheduler(argv):
        calls.append(argv)
        output = f"PartitionName={partition} State=UP" if argv[0] == "scontrol" else str(2000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    ledger = submission.submit(spec, execute=True, authorization_phrase=campaign.PHRASES["compare"],
                               reviewed_plan_hash=plan["content_hash"])
    assert len(ledger["jobs"]) == 17
    assert all(c[0] in {"sbatch", "scontrol"} for c in calls)
    assert all("update" not in c for c in calls)
    assert before == {p: sha256_file(p) for p in old.rglob("*") if p.is_file()}
    if cpus == 36:
        donor_study = artifact("DEV_STUDY", site=dict(partition="debug"))
        publish(tmp_path/"wrong_study.json", donor_study, "DEV_STUDY")
        wrong_partition = with_content_hash(dict(spec, study=data.file_ref(tmp_path/"wrong_study.json")))
        with pytest.raises(ValueError, match="tier3 only"):
            campaign.validate_stage(wrong_partition)
    bad = deepcopy(spec)
    bad["tasks"][0]["cpus"] = 32
    with pytest.raises(ValueError, match="registration"):
        campaign.validate_stage(with_content_hash(bad))
    report_path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        campaign.validate_stage(spec)


def test_profiles_and_legacy_cpu64_registration_are_unchanged():
    old = artifact("DEV_CPU64_EXECUTION", preprocessing_cpus=64, chunk_jets=8,
        in_flight_per_worker=2, heartbeat_seconds=15, loaded_pair_payload_limit_bytes=8*1024**3,
        reader="one_authenticated_read_per_file_v1", scheduler="bounded_completion_order_ram_chunks_v1",
        sampling="unchanged_per_file_bottom_hash_and_weights_v1", preparation_roles_together=["location", "residual"],
        fit_threads_ac=16, fit_threads_b=1, memory_gib=128, durable_pairs=False, durable_records=False)
    assert p.execution_profile() == old
    new = p.execution_profile(36)
    assert new == with_content_hash(dict(old, contract="CMS2JC2_RESPONSE_DEV_CPU36_EXECUTION/v1", preprocessing_cpus=36))
    assert campaign.tasks("compare", "SEARCH", 1, cpu64=True) == campaign.tasks("compare", "SEARCH", 1, preprocessing_cpus=64)
    for cpus in (0, 32, 72, True, 36.0):
        with pytest.raises(ValueError):
            p.execution_profile(cpus)
        with pytest.raises(ValueError):
            campaign.tasks("compare", "SEARCH", 1, preprocessing_cpus=cpus)
    with pytest.raises(ValueError, match="one preprocessing"):
        campaign.tasks("compare", "SEARCH", 1, cpu64=True, preprocessing_cpus=36)
    with pytest.raises(ValueError, match="comparison-only"):
        campaign.tasks("pilot", "SEARCH", 1, preprocessing_cpus=36)


def test_cpu36_rejects_wrong_partition_before_creating_anything(tmp_path):
    with pytest.raises(ValueError, match="tier3 only"):
        restart.create_compare36(parent_spec=tmp_path/"absent.json", project_dir=tmp_path,
                                 source_commit="c"*40, root=tmp_path/"new", partition="debug")
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("cpus,partition", [(36, "tier3"), (64, "debug")])
def test_cli_profile_defaults(monkeypatch, cpus, partition):
    import runpy
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []
    monkeypatch.setattr(restart, f"create_compare{cpus}", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(sys, "argv", ["cms2jc2_response_dev.py", f"create-compare{cpus}",
        "--parent-spec", "parent.json", "--project-dir", "project", "--source-commit", "a"*40, "--root", "new"])
    main = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/cms2jc2_response_dev.py"))["main"]
    assert main() == 0
    assert calls[0]["partition"] == partition


def test_cpu36_worker_rejects_wrong_allocation_and_partition(monkeypatch):
    site = dict(partition="tier3", account="reu-aisocial", conda_prefix="/test/env")
    task = campaign.tasks("compare", "SEARCH", 1, preprocessing_cpus=36)[0]
    monkeypatch.setattr(worker.platform, "system", lambda: "Linux")
    for key, value in dict(SLURM_JOB_ID="123", SLURM_JOB_PARTITION="tier3", SLURM_JOB_ACCOUNT="reu-aisocial",
        SLURM_CPUS_PER_TASK="36", SLURM_JOB_NUM_NODES="1", CONDA_PREFIX="/test/env", PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", LD_LIBRARY_PATH="/test/env/lib", SLURM_GPUS="", SLURM_JOB_GPUS="",
        SLURM_STEP_GPUS="", SLURM_GPUS_ON_NODE="").items():
        monkeypatch.setenv(key, value)
    assert worker.allocation(dict(site=site), task)["cpus"] == 36
    for field, value in [("SLURM_CPUS_PER_TASK", "32"), ("SLURM_CPUS_PER_TASK", "64"),
                         ("SLURM_JOB_PARTITION", "debug"), ("SLURM_JOB_GPUS", "0")]:
        with monkeypatch.context() as m:
            m.setenv(field, value)
            with pytest.raises(PermissionError):
                worker.allocation(dict(site=site), task)
