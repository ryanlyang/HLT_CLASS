"""Synthetic-only CONTEXT_V1 -> classifier integration; no RC submissions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_literature_context_consumer import parent, parent_v2, final_pilot, study, relocated, consumer
from hlt_classification.literature_context_production import campaign as producer, engine, output
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.cms_proxy_ladder import context as l, release as r, data, cache
from hlt_classification.cms_proxy_ladder import gate as g, submission as s, production as p
from hlt_classification.cms_proxy_ladder.campaign import task_graph
from hlt_classification.cms_proxy_ladder.contracts import artifact, write_json
from hlt_classification.cms_proxy_ladder.views import build_view, view_contract
from hlt_classification.jetclass2_delphes.model import model_contract
from hlt_classification.jetclass2_delphes.contracts import artifact as jc2_artifact
from hlt_classification.jetclass2_delphes.campaign import recipe

MEASURE_PREFLIGHT = g._measure_preflight


@pytest.fixture
def dataset(relocated, monkeypatch):
    root, digest, _ = relocated
    monkeypatch.setattr(l, "COUNTS", dict(train=22, validation=11))
    monkeypatch.setattr(l, "MANIFEST_SHA256", digest)
    return dict(root=root/consumer.PROXY_DIRECTORY,
                data_root=root/consumer.SNAPSHOT_DIRECTORY/"jetclass2",
                provenance_root=root/"oscar_provenance_v1")


def paths(dataset):
    return dict(study_root=dataset["root"], offline_root=dataset["data_root"],
                provenance_root=dataset["provenance_root"])


@pytest.fixture
def released(dataset, tmp_path):
    request = l.release_request(**paths(dataset))
    root = tmp_path / "release"
    return r.build_release(request, output_root=root), root


def test_population_default_and_old_contract_unchanged():
    assert l.COUNTS == dict(train=100000, validation=50000)
    assert r.COUNTS == dict(train=200000, validation=100000)
    assert l.DOMAIN != r.SELECTION_DOMAIN
    assert view_contract()["endpoints"]["D000"] == "exact_cms_calibrated_proxy_hlt"
    assert view_contract(literature=True)["endpoints"]["D000"] == "exact_frozen_literature_noise_v3_proxy"
    assert view_contract(literature=True)["schema_version"] == 2


def test_selection_reads_no_labels_or_test_and_is_deterministic(dataset, tmp_path, monkeypatch):
    import uproot
    monkeypatch.setattr(uproot, "open", lambda *a, **k: pytest.fail("Selection must not open ROOT"))
    request = l.release_request(**paths(dataset))
    first = r.build_release(request, output_root=tmp_path / "first")
    second = r.build_release(request, output_root=tmp_path / "second")
    a = r.load_bank(first, root=tmp_path / "first")
    b = r.load_bank(second, root=tmp_path / "second")
    assert first["schema_version"] == 3 and first["counts"] == l.COUNTS
    assert first["study_contract"] == "JC2_CONTEXT_PRODUCTION_STUDY/v1"
    assert first["labels_read"] is False and first["final_test_accessed"] is False
    assert first["identity_sha256"] == second["identity_sha256"]
    for field in a:
        np.testing.assert_array_equal(a[field], b[field])
    for role, code in (("train", 0), ("validation", 1)):
        assert (a["role"] == code).sum() == l.COUNTS[role]


def test_paired_raw_offline_units_identity_and_no_native_hlt(released, monkeypatch):
    import uproot
    release, root = released
    monkeypatch.setattr(data, "from_jc2", lambda *a, **k: pytest.fail("No CMS mapping conversion"))
    old = uproot.behaviors.TBranch.HasBranches.arrays
    requests = []
    def tracked(self, expressions, **kwargs):
        requests.append(expressions)
        assert not any(str(name).startswith("hlt_part_") for name in expressions)
        return old(self, expressions, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", tracked)
    bank = r.load_bank(release, root=root)
    for role, code in (("train", 0), ("validation", 1)):
        rows = list(data.iter_paired(release, release_root=root, role=role))
        assert len(rows) == l.COUNTS[role]
        assert [x.ordinal for x in rows] == list(range(len(rows)))
        assert [x.identity for x in rows] == [bytes(i).hex() for i in bank["identity"][bank["role"] == code]]
        for row in rows:
            # Synthetic native file d0=.3 mm, not .03 cm or sign-flipped.
            np.testing.assert_allclose(row.offline.tracking[:, 0], .3)
            np.testing.assert_allclose(row.offline.tracking[:, 1], -.2)
            assert 0 <= row.label < 11
    assert requests
    with pytest.raises(PermissionError):
        list(data.iter_paired(release, release_root=root, role="final_test"))


def test_foundation_and_all_ladder_caches_use_same_rows_and_exact_endpoint(released, tmp_path):
    release, root = released
    froot = tmp_path / "foundation"
    f = data.build_foundation(release, release_root=root, output_root=froot, capacity=512, workers=1)
    assert f["schema_version"] == 3 and f["views"] == view_contract(context=True)
    assert data.validate_foundation(f, root=froot) == f["content_hash"]
    plan = l.scientific_plan(f, foundation_root=froot)
    assert plan["schema_version"] == 4 and plan["recipe"] == recipe()
    assert plan["fresh_fit_count"] == 9 and plan["probability_publication_count"] == 5
    assert list(plan["branches"]) == ["DIRECT", "COARSE"]
    assert len(task_graph(plan)) == 16
    assert not any("DENSE" in t["task_id"] for t in task_graph(plan))
    d000 = [n for n in plan["nodes"] if n["coordinate"] == "D000"]
    assert len({(n["initialization_seed"], n["sampler_seed"]) for n in d000}) == 1
    for role in l.COUNTS:
        physical = list(data.iter_paired(release, release_root=root, role=role))
        for coordinate in ("U000", "U050", "U100", "D066", "D033", "D000", "OFFLINE"):
            c = cache.prepare_cache(f, foundation_root=froot, role=role, coordinate=coordinate,
                                    workers=1, max_ram_bytes=2**30)
            assert len(c) == len(physical)
            assert [bytes(i).hex() for i in c.identities] == [row.identity for row in physical]
            if coordinate in ("D000", "OFFLINE"):
                from hlt_classification.cms_proxy_ladder.context_inputs import build_inputs
                expected = np.concatenate([build_inputs(
                    row.proxy if coordinate == "D000" else row.offline).features for row in physical])
                actual = np.concatenate([block.features for block in c.blocks])
                np.testing.assert_array_equal(actual, expected)
                assert f["inputs"]["schema_version"] == 2
                assert f["inputs"]["encoding"]["tracking_clipping"] is False
        ids, offsets, mappings = data.load_assignments(f, root=froot, role=role)
        for row, lo, hi in zip(physical, offsets[:-1], offsets[1:]):
            mapping = mappings[lo:hi]
            assert (mapping >= 0).sum() == min(len(row.proxy), len(row.offline))
            assert build_view(identity=row.identity, proxy=row.proxy, offline=None, coordinate="D000") is row.proxy


def test_reject_wrong_generator_counts_and_corrupt_source(released):
    release, root = released
    for field, value in (("study_contract", "CMS2JC2_PROXY_STUDY/v1"),
                         ("counts", dict(train=200000, validation=100000)),
                         ("schema_version", 1)):
        bad = with_content_hash(dict(release, **{field: value}))
        with pytest.raises((ValueError, KeyError)):
            r.validate_release(bad, root=root)
    block = Path(release["study_root"]) / release["proxy_blocks"][0]["path"]
    block.write_bytes(b"synthetic corruption")
    with pytest.raises(ValueError, match="bytes differ"):
        list(data.iter_paired(release, release_root=root, role="train"))


def fake_source(project, commit, **kwargs):
    return artifact("SOURCE", commit=commit, files={"synthetic": "e"*64})


def profile_for(gate, foundation):
    budget = cache.cache_budgets(foundation, 90000, 6)
    parity = jc2_artifact("WEAVER_PARITY", model=model_contract(), device="cuda", passed=True,
        forward_and_feature_and_parameter_gradients=True, final_test_accessed=False)
    report = jc2_artifact("KERNEL_TRAINING_REPORT", foundation_sha256=foundation["content_hash"],
        node=next(n for n in l.scientific_plan(foundation)["nodes"] if n["node_id"] == "U000"),
        recipe_sha256=recipe()["content_hash"],
        runtime_seconds=10., passes=1, scientific_fit=False, acceptance_only=True, final_test_accessed=False)
    return artifact("RUNTIME_PROFILE", version=8, parents={"gate": gate["content_hash"],
        "foundation": foundation["content_hash"]}, source_commit=gate["source_commit"],
        foundation_sha256=foundation["content_hash"], execution_site=gate["execution_site"],
        model=model_contract(), passed=True, measured_full_population=True, measured_role_counts=dict(l.COUNTS),
        cpus=6, workers=6, memory_mb=90000, train_minutes=60, reduce_minutes=30,
        installed_weaver_parity=parity, installed_environment=jc2_artifact("INSTALLED_ENVIRONMENT", version=2,
            architecture="x86_64"), acceptance_training_report=report,
        acceptance_kd_training_report=with_content_hash(dict(report,
            node=dict(report["node"], node_id="PREFLIGHT_CONTEXT_KD", teacher="PREFLIGHT_U000_CE"))),
        ram_only_views=True, rolling_resume=False, one_pass_seconds=10., inference_seconds=2.,
        cache_seconds_by_coordinate={"U000": 1., "D050": 1.}, cache_budgets=budget,
        gpu={"name": "L40S", "total_memory_bytes": 40*2**30}, gpu_peak_bytes=2**30)


@pytest.fixture
def gated(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(g, "source_lock", fake_source)
    project = tmp_path / "classifier"
    project.mkdir()
    spec = l.create_gate(**paths(dataset),
        gate_root=tmp_path / "gate", project_dir=project, source_commit="e"*40)
    g.run_gate_task(spec, "authenticate_release")
    # Keep registered remote workers=16; unit-test fixture computes serially.
    old = g.build_foundation
    monkeypatch.setattr(g, "build_foundation", lambda *a, **k: old(*a, **dict(k, workers=1)))
    g.run_gate_task(spec, "build_foundation")
    monkeypatch.setattr(g, "_measure_preflight", lambda spec, foundation, **k: profile_for(spec, foundation))
    g.run_gate_task(spec, "preflight")
    return spec


def test_full_gate_science_dry_dags_oscar_only(gated, tmp_path):
    assert g.validate_gate(gated, check_source=True)
    gate_plan = s.gate_plan(gated)
    assert len(gate_plan["commands"]) == 3
    campaign = l.create_campaign(gate_root=gated["gate_root"], campaign_root=tmp_path / "science")
    assert p.validate_campaign(campaign, check_source=True)
    science = s.science_plan(campaign)
    assert len(science["commands"]) == 16
    for plan in (gate_plan, science):
        for row in plan["commands"]:
            command = row["command"]
            assert "--partition=gpu" in command and "--account=default" in command
            assert "--qos=norm-gpu" in command and not any("tier3" == a.split("=")[-1] for a in command if a.startswith("--partition="))
            assert not any("DENSE" in a for a in command)
            assert any("oscar_l40s" in a and "jetclass2_delphes_common.sh" in a for a in command)
            if row["task_id"].startswith(("train_", "reduce_")) or row["task_id"] == "preflight":
                assert "--gres=gpu:l40s:1" in command and "--cpus-per-task=6" in command
    s.submit(subject=gated, mode="gate", execute=False, authorization_phrase=None)
    s.submit(subject=campaign, mode="science", execute=False, authorization_phrase=None)
    assert (tmp_path / "science/dry_run_submission_ledger.json").is_file()
    assert not (tmp_path / "science/submission_ledger.json").exists()
    with pytest.raises(ValueError, match="gates"):
        p.create_campaign(gate_root=gated["gate_root"], campaign_root=tmp_path / "wrongcreator")
    assert not (tmp_path / "wrongcreator").exists()
    with pytest.raises(PermissionError):
        s.submit(subject=campaign, mode="science", execute=True, authorization_phrase="wrong")
    for key, value in (("fresh_fit_count", 17), ("selected_branches", ["DIRECT", "COARSE", "DENSE"]),
                       ("dataset_kind", "cms_calibrated")):
        bad = with_content_hash(dict(campaign, **{key: value}))
        with pytest.raises(ValueError):
            p.validate_campaign(bad)


def test_runtime_rejects_over_limit_transfer_or_faked_parity(gated):
    f = load_json(Path(gated["gate_root"]) / "foundation/foundation.json")
    profile = profile_for(gated, f)
    assert g.validate_profile(profile, foundation=f, spec=gated)
    for key, value in (("train_minutes", 2881), ("workers", 36), ("measured_full_population", False),
                       ("measured_role_counts", dict(train=100000, validation=50000))):
        with pytest.raises(ValueError):
            g.validate_profile(with_content_hash(dict(profile, **{key: value})), foundation=f, spec=gated)
    bad = copy.deepcopy(profile)
    bad["installed_weaver_parity"] = with_content_hash(dict(bad["installed_weaver_parity"], passed=False))
    with pytest.raises(ValueError):
        g.validate_profile(with_content_hash(bad), foundation=f, spec=gated)
    bad = copy.deepcopy(profile)
    bad["acceptance_training_report"] = with_content_hash(dict(bad["acceptance_training_report"], recipe_sha256="0"*64))
    with pytest.raises(ValueError):
        g.validate_profile(with_content_hash(bad), foundation=f, spec=gated)


def test_preflight_exercises_installed_parity_ce_kd_reducer_and_oscar(gated, monkeypatch):
    import itertools
    from hlt_classification.jetclass2_delphes import acceptance
    foundation = load_json(Path(gated["gate_root"]) / "foundation/foundation.json")
    expected = profile_for(gated, foundation)
    calls = []

    class FakeCache:
        nbytes = 1024
        def __init__(self, role):
            self.role = role
            self.identities = np.arange(l.COUNTS[role])
        def __len__(self):
            return l.COUNTS[self.role]

    def prepare(foundation, **kwargs):
        calls.append(("cache", kwargs["role"], kwargs["coordinate"]))
        assert kwargs["workers"] == 6 and kwargs["population_selection"] is None
        return FakeCache(kwargs["role"])

    def parity(cache, *, device):
        calls.append(("parity", cache.role, device))
        return expected["installed_weaver_parity"]

    def train(model, train, validation, **kwargs):
        calls.append(("train", kwargs["node"]["node_id"], kwargs["acceptance_passes"]))
        assert len(train) == l.COUNTS["train"] and len(validation) == l.COUNTS["validation"]
        assert kwargs["device"] == "cuda"
        if kwargs["node"]["teacher"] is not None:
            assert kwargs["teacher_probabilities"].shape == (l.COUNTS["train"], 11)
            np.testing.assert_array_equal(kwargs["teacher_identities"], train.identities)
            return expected["acceptance_kd_training_report"], {}
        return expected["acceptance_training_report"], {}

    def predict(model, cache, **kwargs):
        calls.append(("reduce", cache.role, kwargs["temperature"]))
        return np.zeros((len(cache), 11), np.float32)

    monkeypatch.setattr(g, "allocation", lambda site: ("123", 6, 90000))
    monkeypatch.setattr(g, "installed_environment", lambda: expected["installed_environment"])
    monkeypatch.setattr(g, "gpu_identity", lambda: expected["gpu"])
    monkeypatch.setattr(g, "prepare_cache", prepare)
    monkeypatch.setattr(acceptance, "installed_parity", parity)
    monkeypatch.setattr(g, "DelphesParticleTransformer", lambda: object())
    monkeypatch.setattr(g, "train_kernel", train)
    monkeypatch.setattr(g, "predict", predict)
    monkeypatch.setattr(g, "production_site", lambda *a: pytest.fail("No tier3 transfer"))
    monkeypatch.setattr(g, "build_scientific_plan", lambda *a, **k: pytest.fail("No dense plan"))
    monkeypatch.setattr(g.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(g.torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(g.torch.cuda, "empty_cache", lambda: None)
    # Fake inference can finish within one Windows monotonic clock tick;
    # retain the production requirement for positive measured work durations.
    ticks = itertools.count(100.)
    monkeypatch.setattr(g.time, "monotonic", lambda: next(ticks))
    result = MEASURE_PREFLIGHT(gated, foundation, output_root=Path(gated["gate_root"]) / "evidence")
    assert result["execution_site"]["partition"] == "gpu"
    assert result["measured_role_counts"] == l.COUNTS
    assert calls == [("cache", "train", "U000"), ("cache", "validation", "U000"),
        ("parity", "train", "cuda"), ("train", "U000", 1),
        ("reduce", "train", 2.), ("reduce", "validation", 1.),
        ("train", "PREFLIGHT_CONTEXT_KD", 1),
        ("cache", "train", "D050"), ("cache", "validation", "D050")]


def test_no_science_before_gate_and_no_source_dataset_overlap(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(g, "source_lock", fake_source)
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(PermissionError):
        l.create_gate(**paths(dataset),
            gate_root=Path(dataset["root"]) / "forbidden", project_dir=project, source_commit="e"*40)
    gate = l.create_gate(**paths(dataset),
        gate_root=tmp_path / "freshgate", project_dir=project, source_commit="e"*40)
    with pytest.raises(FileNotFoundError):
        l.create_campaign(gate_root=gate["gate_root"], campaign_root=tmp_path / "premature")
    assert not (tmp_path / "premature").exists()


def test_site_probes_are_read_only_and_clear_inherited_resource_requests(monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setenv("SBATCH_GRES", "gpu:gh200:8")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    def run(argv, **kwargs):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ClusterName = slurmctld\n" if argv[0] == "scontrol" else "accepted", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    plan = {"commands": [dict(command=["sbatch", "--partition=gpu", "--cpus-per-task=6",
        "--mem=90000M", "--time=08:00:00", "--gres=gpu:l40s:1", "--dependency=afterok:${JOB_parent}", "--wrap=script"])]}
    l.check_submission_site(plan)
    assert "--test-only" in calls[1] and "--wrap=true" in calls[1]
    assert not any(a.startswith("--dependency=") for a in calls[1])


@pytest.mark.parametrize("interrupt", [False, True])
def test_live_claim_cleans_environment_and_prevents_duplicate_or_ambiguous_retry(tmp_path, monkeypatch, interrupt):
    import subprocess
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
    subject = artifact("SYNTHETIC_TEST")
    plan = artifact("COMMAND_PLAN", commands=[
        dict(task_id="first", dependencies=[], command=["sbatch", "--wrap=true"]),
        dict(task_id="second", dependencies=["first"], command=["sbatch", "--dependency=afterok:${JOB_first}", "--wrap=true"]),
    ])
    dry = build_submission_ledger(campaign_spec_sha256=subject["content_hash"],
        jobs={r["task_id"]: "1" for r in plan["commands"]},
        commands={r["task_id"]: r["command"] for r in plan["commands"]}, dry_run=True)
    write_json(tmp_path / "dry_run_submission_ledger.json", dry)
    monkeypatch.setenv("SBATCH_GRES", "gpu:gh200:8")
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kwargs["env"])
        if interrupt and len(calls) == 2:
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout=str(700 + len(calls)))
    monkeypatch.setattr(subprocess, "run", run)
    if interrupt:
        with pytest.raises(subprocess.CalledProcessError):
            l.submit_claimed(subject, plan, tmp_path)
        with pytest.raises(PermissionError, match="active or interrupted"):
            l.submit_claimed(subject, plan, tmp_path)
        assert not (tmp_path / "submission_ledger.json").exists()
    else:
        result = l.submit_claimed(subject, plan, tmp_path)
        assert result["jobs"] == {"first": "701", "second": "702"}
        assert "--dependency=afterok:701" in calls[1]
        assert l.submit_claimed(subject, plan, tmp_path) == result
    assert len(calls) == 2 and (tmp_path / "live_submission_claim.json").is_file()


def test_wrapper_exposes_gate_and_science_without_automatic_submission():
    project = Path(__file__).resolve().parents[1]
    text = (project / "scripts/queue_jetclass2_context_ladder.sh").read_text()
    assert 'JC2_SITE=oscar_l40s' in text
    assert '"${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"' in text
    assert '--plan-hash "$6"' in text
    assert '"${MODE}" == gate || "${MODE}" == science' in text
    assert 'Dry review only; no jobs submitted' in text
    assert "scancel" not in text
