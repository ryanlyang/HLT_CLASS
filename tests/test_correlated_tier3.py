"""Same measured science, explicit tier3 execution, no live scheduler calls."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.cms_proxy_ladder import correlated_tier3 as t, production as p, gate as g
from hlt_classification.cms_proxy_ladder.contracts import artifact, file_ref, write_json
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash
from hlt_classification.jetclass2_delphes.execution import execution_site


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    before = 'def validate_campaign(spec, *, check_source=False):\n' + t.NEEDLE + '    return version\n'
    old, new = tmp_path / "old", tmp_path / "new"
    contents = {t.PRODUCTION: before, "src/hlt_classification/jetclass2_delphes/runner.py": "# unchanged kernel\n"}
    for root in (old, new):
        for name, text in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    (new / t.PRODUCTION).write_text(before.replace(t.NEEDLE, t.NEEDLE + t.INSERTION))
    (new / t.ADDED).write_text("# registered new adapter\n")
    base = dict(project_dir=str(old), source_commit=t.BASE_COMMIT,
                source=artifact("SOURCE", commit=t.BASE_COMMIT, files={k: sha256_file(old / k) for k in contents}))
    def lock(project, commit, **kw):
        return artifact("SOURCE", commit=commit, files={
            str(path.relative_to(project)).replace("\\", "/"): sha256_file(path)
            for path in project.rglob("*.py")})
    monkeypatch.setattr(g, "source_lock", lock)
    return SimpleNamespace(base=base, old=old, new=new)


def test_exact_dispatch_only_source_transfer(source_case):
    c = source_case
    source, transfer = t.source_transfer(c.base, c.new, "f" * 40)
    assert transfer["parents"]["measurement_source"] == c.base["source"]["content_hash"]
    assert transfer["parents"]["executor_source"] == source["content_hash"]
    (c.new / t.PRODUCTION).write_text((c.new / t.PRODUCTION).read_text() + "\n# extra change\n")
    with pytest.raises(ValueError, match="dispatch"):
        t.source_transfer(c.base, c.new, "f" * 40)


@pytest.mark.parametrize("change", ["kernel", "extra_file", "remove", "pin"])
def test_reject_unmeasured_source_changes(source_case, change):
    c = source_case
    path = c.new / "src/hlt_classification/jetclass2_delphes/runner.py"
    if change == "kernel":
        path.write_text("# scientific change\n")
    elif change == "extra_file":
        (c.new / "extra.py").write_text("# not registered\n")
    elif change == "remove":
        path.unlink()
    else:
        c.base["source_commit"] = "a" * 40
    with pytest.raises(ValueError):
        t.source_transfer(c.base, c.new, "f" * 40)


def test_historical_executor_diff_is_only_the_registered_dispatch():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    before = subprocess.run(["git", "show", f"{t.BASE_COMMIT}:{t.PRODUCTION}"], cwd=root,
                            check=True, capture_output=True, text=True).stdout
    # This amendment authorizes the historical executor, not all future main
    # revisions. New experiments require fresh measured gates of their own.
    executor = '779382740cdd44a5120449f02fde4a7719fed298'
    after = subprocess.run(['git', 'show', f'{executor}:{t.PRODUCTION}'], cwd=root,
                           check=True, capture_output=True, text=True).stdout
    assert after == before.replace(t.NEEDLE, t.NEEDLE + t.INSERTION, 1)


def test_new_experiment_source_cannot_reuse_historical_transfer(source_case):
    root = Path(__file__).resolve().parents[1]
    (source_case.new / t.PRODUCTION).write_text((root / t.PRODUCTION).read_text())
    with pytest.raises(ValueError, match='dispatch'):
        t.source_transfer(source_case.base, source_case.new, 'f'*40)


@pytest.mark.parametrize("name", ["docs/plans/JETCLASS2_CORRELATED_TRACKING_PRODUCTION_PLAN.md",
                                 "docs/contracts/JETCLASS2_CORRELATED_TRACKING_LADDER.md"])
def test_old_source_hashed_plan_and_contract_unchanged(name):
    import subprocess
    root = Path(__file__).resolve().parents[1]
    before = subprocess.run(["git", "show", f"{t.BASE_COMMIT}:{name}"], cwd=root,
                            check=True, capture_output=True, text=True).stdout
    assert (root / name).read_text() == before


@pytest.fixture
def transferred(tmp_path, monkeypatch):
    # Fast operational fixture. A real synthetic measured gate is exercised below.
    measured = artifact("RUNTIME_PROFILE", version=9, execution_site=execution_site("sporc_a100_debug"),
        source_commit=t.BASE_COMMIT, cpus=6, memory_mb=90000, workers=6,
        train_minutes=120, reduce_minutes=30, gpu={"name": "A100", "total_memory_bytes": 40*2**30},
        installed_environment={"python": "same"})
    tasks = [dict(task_id="train_" + name, kind="train", dependencies=[], node_id=name)
             for name in ("M0HLT", "OFFLINE", "DIRECT", "D066", "D033", "D000")]
    tasks += [dict(task_id="reduce_" + name, kind="reduce", dependencies=["train_" + name], node_id=name)
              for name in ("OFFLINE", "D066", "D033")]
    tasks += [dict(task_id="aggregate", kind="aggregate", dependencies=[r["task_id"] for r in tasks]),
              dict(task_id="complete", kind="complete", dependencies=["aggregate"])]
    original = artifact("CAMPAIGN_SPEC", version=5, parents=dict(gate="a"*64, profile=measured["content_hash"]),
        gate_root=str(tmp_path / "gate"), project_dir=str(tmp_path / "old"), source_commit=t.BASE_COMMIT,
        source=artifact("SOURCE", commit=t.BASE_COMMIT, files={}), runtime_profile=measured,
        campaign_root=str(tmp_path / "science"), foundation_root=str(tmp_path / "foundation"),
        tasks=tasks, fresh_fit_count=6, reducer_count=3, selected_branches=["DIRECT", "COARSE"])
    old_path = tmp_path / "science/campaign_spec.json"
    old_path.parent.mkdir()
    write_json(old_path, original)
    (tmp_path / "gate").mkdir()
    write_json(tmp_path / "gate/gate_spec.json", artifact("TEST", request=dict(
        study_root=str(tmp_path / "dataset"), offline_root=str(tmp_path / "raw"))))
    monkeypatch.setattr(t.original, "validate_campaign", lambda *a, **k: None)
    monkeypatch.setattr(t, "source_transfer", lambda base, project, commit: (
        artifact("SOURCE", commit=commit, files={}), artifact("TIER3_SOURCE_TRANSFER", unchanged=True)))
    spec = t.create_campaign(measurement_spec=old_path, project_dir=tmp_path / "new",
                             source_commit="f"*40, campaign_root=tmp_path / "science_tier3")
    return SimpleNamespace(base=original, spec=spec, root=tmp_path / "science_tier3")


def test_tier3_plan_keeps_measurements_original_and_exact_resources(transferred):
    c = transferred
    assert p.validate_campaign(c.spec, check_source=True)
    profile = c.spec["runtime_profile"]
    assert c.base["runtime_profile"]["execution_site"]["partition"] == "debug"
    assert profile["measurement_site"]["partition"] == "debug"
    assert profile["execution_site"]["partition"] == "tier3"
    assert profile["source_commit"] == t.BASE_COMMIT != c.spec["source_commit"]
    plan = t.submit(c.spec)
    assert len(plan["commands"]) == 11
    for row in plan["commands"]:
        cmd = row["command"]
        assert "--partition=tier3" in cmd and "--partition=debug" not in cmd
        assert f"--job-name=jc2crt_{row['task_id']}" in cmd
        assert any("JC2_SITE=sporc_a100\n" in a for a in cmd)
        if row["task_id"].startswith(("train_", "reduce_")):
            assert "--cpus-per-task=6" in cmd and "--mem=90000M" in cmd and "--gres=gpu:a100:1" in cmd
    assert (c.root / "dry_run_submission_ledger.json").is_file()
    assert not (c.root / "submission_ledger.json").exists()


@pytest.mark.parametrize("field,value", [("fresh_fit_count", 9), ("selected_branches", ["DENSE"]),
    ("runtime_profile", {}), ("source_commit", "0"*40), ("final_test_accessed", True)])
def test_transfer_rejects_rehashed_mutation(transferred, field, value):
    bad = with_content_hash(dict(transferred.spec, **{field: value}))
    with pytest.raises((ValueError, PermissionError, KeyError)):
        t.validate_campaign(bad)


def test_worker_enforces_tier3_actual_gpu_and_environment(transferred, monkeypatch):
    calls = []
    profile = transferred.spec["runtime_profile"]
    monkeypatch.setattr(p, "allocation", lambda site: (calls.append(site) or ("123", 6, 90000)))
    monkeypatch.setattr(p, "gpu_identity", lambda: profile["gpu"])
    monkeypatch.setattr(p, "installed_environment", lambda: profile["installed_environment"])
    p._execution_gate(transferred.spec, "cuda")
    assert calls[0]["partition"] == "tier3"
    monkeypatch.setattr(p, "gpu_identity", lambda: {"name": "different A100"})
    with pytest.raises(ValueError, match="GPU/environment"):
        p._execution_gate(transferred.spec, "cuda")


def test_submit_auth_failures_and_idempotent_success(transferred, monkeypatch):
    import subprocess
    c, calls = transferred, []
    plan = t.submit(c.spec)
    for kwargs in ({"plan_hash": "0"*64, "authorization_phrase": t.AUTHORIZATION},
                   {"plan_hash": plan["content_hash"], "authorization_phrase": "wrong"}):
        with pytest.raises(PermissionError):
            t.submit(c.spec, execute=True, **kwargs)
    monkeypatch.setattr(t.original, "check_submission_site", lambda plan: None)
    def launch(argv, **kwargs):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kwargs["env"])
        return SimpleNamespace(stdout=str(1000+len(calls)))
    monkeypatch.setattr(subprocess, "run", launch)
    kwargs = dict(execute=True, plan_hash=plan["content_hash"], authorization_phrase=t.AUTHORIZATION)
    first = t.submit(c.spec, **kwargs)
    assert len(first["jobs"]) == 11 and len(calls) == 11
    assert t.submit(c.spec, **kwargs) == first and len(calls) == 11
    assert (c.root / "live_submission_claim.json").is_file()


def test_ambiguous_submission_never_automatically_retries(transferred, monkeypatch):
    import subprocess
    c, calls = transferred, []
    plan = t.submit(c.spec)
    monkeypatch.setattr(t.original, "check_submission_site", lambda plan: None)
    def launch(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout="1001")
    monkeypatch.setattr(subprocess, "run", launch)
    kwargs = dict(execute=True, plan_hash=plan["content_hash"], authorization_phrase=t.AUTHORIZATION)
    with pytest.raises(subprocess.CalledProcessError):
        t.submit(c.spec, **kwargs)
    with pytest.raises(PermissionError, match="active or interrupted"):
        t.submit(c.spec, **kwargs)
    assert len(calls) == 2


# Import original synthetic ROOT/pilot/gate fixtures to prove full gate reuse.
from test_correlated_tracking_ladder import (  # noqa: E402,F401
    parent, final_pilot, study, dataset, gated,
)


def test_real_synthetic_gate_transfers_without_rebuilding_foundation(gated, tmp_path, monkeypatch):
    from hlt_classification.cms_proxy_ladder import correlated as original
    base = original.create_campaign(gate_root=gated["gate_root"], campaign_root=tmp_path / "science")
    monkeypatch.setattr(t, "source_transfer", lambda b, project, commit: (
        artifact("SOURCE", commit=commit, files={}), artifact("TIER3_SOURCE_TRANSFER", fixture=True)))
    before = (Path(gated["gate_root"]) / "gate_complete.json").read_bytes()
    spec = t.create_campaign(measurement_spec=tmp_path / "science/campaign_spec.json",
        project_dir=tmp_path / "executor", source_commit="f"*40, campaign_root=tmp_path / "science_tier3")
    assert p.validate_campaign(spec, check_source=True)
    assert spec["foundation"] == base["foundation"]
    assert spec["scientific_plan"] == base["scientific_plan"] and spec["tasks"] == base["tasks"]
    assert len(t.submit(spec)["commands"]) == 11
    assert (Path(gated["gate_root"]) / "gate_complete.json").read_bytes() == before
