"""Synthetic SPORC contracts only; never claims an actual A100 acceptance."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from test_jetclass2_delphes import snapshot
from test_jetclass2_delphes_production import campaign, fake_slurm
from test_jetclass2_delphes_training import TinyModel, toy_cache
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.jetclass2_delphes import execution, readiness, production, submission
from hlt_classification.jetclass2_delphes.cache import cache_budgets, preparation_bound


def test_site_profiles_and_wrong_site_rejection():
    site = execution.execution_site("sporc_a100")
    options = execution.slurm_options(site)
    assert "--partition=tier3" in options and "--qos=qos_tier3" in options
    assert site["gres"] == "gpu:a100:1" and site["architecture"] == "x86_64"
    assert site["conda_base"] == "/home/ryreu/miniconda3"
    assert "--no-requeue" in options
    assert "--partition=tigris" in execution.slurm_options(execution.execution_site("tigris_gh200"))
    with pytest.raises(ValueError, match="Unknown"):
        execution.execution_site("auto")
    with pytest.raises(ValueError, match="registered"):
        execution.validate_site(with_content_hash({**site, "qos": "qos_interactive"}))
    for cpus, memory, workers in ((72, 98304, 8), (8, 98304, 9), (8, 0, 8), (8, 400000, 8)):
        with pytest.raises(ValueError, match="envelope"):
            execution.validate_resources(site, cpus, memory, workers)


def _allocation_fixture(monkeypatch):
    site = execution.execution_site("sporc_a100")
    prefix = "/home/ryreu/miniconda3/envs/atlas_kd_sporc"
    for name, value in dict(SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="8", SLURM_MEM_PER_NODE="98304",
                            SLURM_CLUSTER_NAME="sporc", CONDA_PREFIX=prefix, PYTHONNOUSERSITE="1").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(execution.sys, "prefix", prefix)
    monkeypatch.setattr(execution.platform, "machine", lambda: "x86_64")
    fields = dict(Account="reu-aisocial", Partition="tier3", QOS="qos_tier3", NumNodes="1", NumTasks="1", NumCPUs="8")
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=" ".join(f"{k}={v}" for k, v in fields.items())))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    return site, fields


@pytest.mark.parametrize("field,value", [("Partition", "tigris"), ("Account", "other"),
                                       ("QOS", "qos_interactive"), ("NumNodes", "2"),
                                       ("NumTasks", "4"), ("NumCPUs", "36")])
def test_allocation_exact_scheduler_binding(monkeypatch, field, value):
    site, fields = _allocation_fixture(monkeypatch)
    assert execution.allocation(site) == ("123", 8, 98304)
    fields[field] = value
    with pytest.raises(PermissionError, match="scheduler"):
        execution.allocation(site)


def test_allocation_wrong_environment_gpu_and_login_node(monkeypatch):
    site, _ = _allocation_fixture(monkeypatch)
    monkeypatch.setenv("CONDA_PREFIX", "/home/ryreu/miniconda3/envs/albert")
    with pytest.raises(PermissionError, match="Conda"):
        execution.allocation(site)
    monkeypatch.setenv("CONDA_PREFIX", execution.sys.prefix)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA H100")
    with pytest.raises(PermissionError, match="GPU"):
        execution.allocation(site)
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(PermissionError, match="Slurm"):
        execution.allocation(site)


def test_large_fixed_validation_memory_is_not_starved():
    spec = dict(inputs=dict(capacity=240), assignment_tasks=[
        dict(role=role, rows=rows) for role, rows in
        [("train", 5000)] * 100 + [("validation", 20000)] * 50])
    budgets = cache_budgets(spec, 98304, 8)
    assert budgets["validation"] > 2 * budgets["train"]
    assert sum(budgets.values()) == 98304 * 1024**2 * 3 // 4
    for role in budgets:
        assert budgets[role] >= preparation_bound(spec, role, 8)
    with pytest.raises(MemoryError, match="reduce workers"):
        cache_budgets(spec, 8192, 8)
    with pytest.raises(PermissionError, match="sealed"):
        preparation_bound(spec, "final_test", 8)


def test_readiness_dry_live_exact_arrays_and_science_is_never_submitted(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_source", lambda *args: None)
    calls, _ = fake_slurm(monkeypatch)
    spec = readiness.create_readiness(inventory=campaign["foundation"]["inventory"],
                                     split_profile=campaign["foundation"]["splits"],
                                     data_root=Path(campaign["data_root"]), output_root=tmp_path / "ready",
                                     project=tmp_path / "pinned", source_commit="d" * 40)
    root = Path(spec["readiness_root"])
    assert calls == []
    plan = load_json(root / "command_plan.json")
    assert [r["task_id"] for r in plan["commands"]] == ["sample", "assign", "lock", "profile"]
    assert plan["scientific_fits"] == 0 and not plan["automatically_launches_science"]
    count = len(spec["foundation"]["assignment_tasks"])
    assert f"--array=0-{count-1}%16" in plan["commands"][1]["command"]
    assert "--dependency=afterok:${JOB_assign}" in plan["commands"][2]["command"]
    assert "--time=240" in plan["commands"][3]["command"]
    assert "--mem=81920M" in plan["commands"][3]["command"]
    assert "--gres=gpu:a100:1" in plan["commands"][3]["command"]
    assert all(not any("--gres" in arg for arg in r["command"]) for r in plan["commands"][:3])
    assert all(t["role"] in {"train", "validation"} for t in spec["foundation"]["assignment_tasks"])
    with pytest.raises(PermissionError, match="readiness-only"):
        readiness.submit_readiness(spec, execute=True, authorization_phrase=production.AUTHORIZE)
    live = readiness.submit_readiness(spec, execute=True, authorization_phrase=readiness.AUTHORIZE_READINESS)
    assert len(calls) == 4 and len(live["jobs"]) == 4
    assert all(c[0] == "sbatch" for c in calls)
    assert not any("${JOB_" in arg for c in calls for arg in c)
    assert readiness.submit_readiness(spec, execute=True, authorization_phrase=readiness.AUTHORIZE_READINESS) == live
    assert len(calls) == 4
    assert not list(root.rglob("*.pt")) and not list(root.rglob("*.npy"))
    with pytest.raises(ValueError, match="scope"):
        readiness.validate_readiness(with_content_hash({**spec, "science_submission_authorized": True}))


def test_readiness_rejects_raw_path_and_bad_memory_before_publication(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_source", lambda *args: None)
    kwargs = dict(inventory=campaign["foundation"]["inventory"], split_profile=campaign["foundation"]["splits"],
                  data_root=Path(campaign["data_root"]), project=tmp_path / "project", source_commit="d"*40)
    with pytest.raises(ValueError, match="fresh root"):
        readiness.create_readiness(output_root=Path(campaign["data_root"]) / "bad", **kwargs)
    with pytest.raises(ValueError, match="envelope"):
        readiness.create_readiness(output_root=tmp_path / "bad", cpus=72, **kwargs)
    assert not (tmp_path / "bad").exists()


def test_runtime_profile_cannot_cross_gpu_or_cache_policy(campaign, monkeypatch):
    p = campaign["runtime_profile"]
    monkeypatch.setattr(production, "allocation", lambda site: ("123", p["cpus"], p["memory_mb"]))
    monkeypatch.setattr(production, "installed_environment", lambda: p["installed_environment"])
    monkeypatch.setattr(production, "gpu_identity", lambda: {**p["gpu"], "total_memory_bytes": 80*2**30})
    with pytest.raises(ValueError, match="GPU"):
        production.execution_gate(p, "cuda")
    old_budgets = dict(train=int(p["memory_mb"] * 1024**2 * .53), validation=int(p["memory_mb"] * 1024**2 * .22))
    bad = with_content_hash({**p, "cache_budgets": old_budgets})
    with pytest.raises(ValueError, match="runtime evidence"):
        production.validate_profile(bad, campaign["foundation"], campaign["source_commit"])
    other = with_content_hash({**p, "execution_site": execution.execution_site("tigris_gh200")})
    with pytest.raises(ValueError, match="runtime evidence"):
        production.validate_profile(other, campaign["foundation"], campaign["source_commit"])


def test_workers_use_explicit_isolated_conda_and_no_global_default_edits():
    root = Path(__file__).resolve().parents[1]
    helper = (root / "sbatch/jetclass2_delphes_common.sh").read_text()
    assert "CONDA_BASE=/home/ryreu/miniconda3" in helper
    assert "CONDA_ENV=atlas_kd_sporc" in helper
    assert "unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH" in helper
    assert "PYTHONNOUSERSITE=1" in helper and "NUMEXPR_NUM_THREADS=1" in helper
    for worker in ("run_jetclass2_delphes.sh", "run_jetclass2_delphes_preparation.sh"):
        text = (root / "sbatch" / worker).read_text()
        assert '${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh' in text
        assert "JC2_SITE=" in text
    assert "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}" in (root / "sbatch/common.sh").read_text()


def test_profile_pipeline_records_all_cache_paths_and_fails_closed_on_walltime(campaign, tmp_path, monkeypatch):
    """Mock CUDA/volume only; exercise actual profile assembly and validation."""
    profile = campaign["runtime_profile"]
    seen = []
    def prepare(*args, role, coordinate_name, max_ram_bytes, **kwargs):
        seen.append((role, coordinate_name, max_ram_bytes))
        cache = toy_cache(role)
        cache.foundation_sha256 = campaign["foundation"]["content_hash"]
        cache.coordinate_name = coordinate_name
        return cache
    monkeypatch.setattr(production, "prepare_cache", prepare)
    monkeypatch.setattr(production, "allocation", lambda site: ("123", 2, 4096))
    monkeypatch.setattr(production, "gpu_identity", lambda: profile["gpu"])
    monkeypatch.setattr(production, "installed_environment", lambda: profile["installed_environment"])
    monkeypatch.setattr(production, "run_acceptance", lambda *a, **kw: profile["miniature"])
    monkeypatch.setattr(production, "DelphesParticleTransformer", TinyModel)
    # Rehash the synthetic resource report because its runtime is an added field.
    monkeypatch.setattr(production, "train_kernel", lambda *a, **kw:
                        (with_content_hash({**profile["resource_training_report"], "runtime_seconds": 1.}), {}))
    monkeypatch.setattr(production, "predict", lambda model, cache, **kwargs:
                        np.full((len(cache), 11), 1/11, np.float32))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index:
                        SimpleNamespace(total_memory=40*2**30))
    kwargs = dict(foundation_root=Path(campaign["foundation_root"]), data_root=Path(campaign["data_root"]),
                  project=Path(campaign["project_dir"]), source_commit=campaign["source_commit"],
                  site=profile["execution_site"], workers=1)
    result = production.measure_runtime(campaign["foundation"], output_root=tmp_path / "profile", **kwargs)
    assert result["train_minutes"] == 60 and result["reduce_minutes"] == 30
    assert set(result["cache_seconds_by_coordinate"]) == {"U000", "U050", "D050"}
    assert [(r, c) for r, c, _ in seen] == [(r, c) for c in ("U000", "U050", "D050") for r in ("train", "validation")]
    assert all(b == result["cache_budgets"][r] for r, _, b in seen)
    assert not list((tmp_path / "profile").rglob("*.pt"))
    monkeypatch.setattr(production, "train_kernel", lambda *a, **kw:
                        (with_content_hash({**profile["resource_training_report"], "runtime_seconds": 3600.}), {}))
    with pytest.raises(ValueError, match="planning envelope"):
        production.measure_runtime(campaign["foundation"], output_root=tmp_path / "over_budget", **kwargs)
    assert (tmp_path / "over_budget/resource_measurements.json").is_file()
    assert not (tmp_path / "over_budget/runtime_profile.json").exists()
