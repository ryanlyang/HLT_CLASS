"""Worker failure injection: mocked Slurm is explicitly not production evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.cms2jc2_response import campaign, orchestration, submission, tasks, storage
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, publish, with_content_hash
from test_cms2jc2_response_campaign import fixture_spec


@pytest.fixture
def worker(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(orchestration, "validate_spec", lambda s: {})
    monkeypatch.setattr(orchestration, "allocation", lambda **k: {"test_only": True})
    monkeypatch.setattr(orchestration, "validate_source", lambda *a, **k: None)
    monkeypatch.setattr(orchestration, "numerical_environment", lambda: artifact("ENVIRONMENT", test_only=True))
    monkeypatch.setattr(submission, "submitted_jobs", lambda s: {"miniature_A_L": "1234", "miniature_B_L": "1234"})
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))
    class Measure:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def report(self): return {"test_only": True}
    monkeypatch.setattr(orchestration, "Measurement", Measure)
    return spec


def test_worker_receipts_are_last_and_completed_tasks_are_idempotent(worker, monkeypatch):
    calls = []
    def dispatch(spec, task, ctx):
        calls.append(task["task_id"])
        return {"result": ("TEST", artifact("TEST", qualified=False))}
    monkeypatch.setattr(tasks, "dispatch", dispatch)
    a = orchestration.run(worker, "miniature_A_L")
    assert a == orchestration.run(worker, "miniature_A_L")
    assert calls == ["miniature_A_L"]
    assert not orchestration.product(worker, "miniature_A_L")["qualified"]
    assert orchestration.product(worker, "miniature_A_L", "measurement")["production_worker"]


def test_worker_failure_never_certifies_outputs_or_restarts_same_claim(worker, monkeypatch):
    def dispatch(*args): raise ValueError("injected invalid input")
    monkeypatch.setattr(tasks, "dispatch", dispatch)
    with pytest.raises(ValueError, match="injected"):
        orchestration.run(worker, "miniature_A_L")
    assert not orchestration.receipt_path(worker, "miniature_A_L").exists()
    with pytest.raises(FileExistsError):
        orchestration.run(worker, "miniature_A_L")


def test_worker_refuses_changed_source_before_publication(worker, monkeypatch):
    monkeypatch.setattr(tasks, "dispatch", lambda *a: {"result": ("TEST", artifact("TEST"))})
    def changed(*args, **kwargs): raise ValueError("source changed")
    monkeypatch.setattr(orchestration, "validate_source", changed)
    with pytest.raises(ValueError, match="source changed"):
        orchestration.run(worker, "miniature_A_L")
    assert not orchestration.receipt_path(worker, "miniature_A_L").exists()


def test_execution_lock_uses_measurements_and_refuses_excess_resources(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    rows = [{"response_role": r, "fit_role": f, "selected_entries": n} for r, f, n in (
        ("response_fit", "fit_location", 1_800_000), ("response_fit", "fit_residual", 450_000),
        ("response_select", None, 280_000), ("response_confirm", None, 280_000))]
    from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
    ctx = {"roles": {"files": rows}, "compatibility": provisional_compatibility(inventory_hash="a"*64)}
    measurements = dict(wall_seconds=10., sampled_peak_tree_rss_bytes=10_000_000)
    def product(spec, task, name="result"):
        if name == "measurement":
            return artifact("TASK_MEASUREMENT", parents={"source": spec["source"]["content_hash"]},
                numerical_environment=artifact("ENVIRONMENT", test_only=True), production_worker=True, measurement=measurements)
        if task == "miniature_verify":
            return artifact("CPU_MINIATURE", passed=True, real_probe_jets=20_000)
        if name == "response":
            return {"calibration": {s: {"sampling": {"cap_per_role_module": 2_000_000,
                    "per_file": [{"strata": [{"retained": 20_000}]}]}} for s in ("location", "residual")}}
        gate = int(task[-2:])/100
        return {"jets": 100_000, "policy": tasks.policy(gate=gate)}
    payload = tmp_path/"model.json"
    publish(payload, artifact("TEST"), "TEST")
    monkeypatch.setattr(tasks, "product", product)
    monkeypatch.setattr(tasks, "verified_product", lambda *a: payload)
    ready = tasks.execution_lock(spec, ctx)
    assert ready["ready"] and not ready["resource_blockers"]
    assert len(campaign.science_tasks(ready)) == 71
    measurements["wall_seconds"] = 100_000.
    blocked = tasks.execution_lock(spec, ctx)
    assert not blocked["ready"] and blocked["resource_blockers"]
    with pytest.raises(PermissionError, match="Measured"):
        campaign.science_tasks(blocked)


def test_source_parent_and_transfer_role_fail_before_particle_access(tmp_path):
    spec = fixture_spec(tmp_path)
    changed = with_content_hash(dict(spec, parents={"source": "f"*64}))
    with pytest.raises(ValueError, match="source parent"):
        campaign.validate_spec(changed, source=False)
    from hlt_classification.cms2jc2_response.transfer import evaluate_transfer
    selected = artifact("SELECTION", response_hash="a"*64)
    claim = artifact("TRANSFER_CLAIM", parents={"selection": selected["content_hash"], "response": "a"*64, "profile": "b"*64},
                     role="validation", separately_authorized=True)
    def forbidden():
        raise AssertionError("Reader must not be opened")
        yield
    with pytest.raises(PermissionError, match="claim"):
        evaluate_transfer(forbidden(), {"content_hash": "a"*64}, selection=selected, claim=claim, profile_hash="b"*64, role="train")


def test_reverse_tie_diagnostic_changes_only_exact_physical_ties():
    from hlt_classification.cms2jc2_response.association import groups
    from test_cms2jc2_response import particles
    p = particles(pt=(1., 1., 1.), eta=(0., 0., 0.), phi=(0., 0., 0.))
    a = groups(p, .05, max_size=2)
    b = groups(p, .05, max_size=2, reverse_ties=True)
    assert a == tuple(reversed(b)) and set(a) == set(b)
    q = particles(pt=(1., 2., 4.), eta=(0., .01, .04), phi=(0., 0., 0.))
    assert groups(q, .05) == groups(q, .05, reverse_ties=True)
