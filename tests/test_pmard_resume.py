from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _resume_module():
    path = REPOSITORY / "scripts/resume_pmard_campaign.py"
    spec = importlib.util.spec_from_file_location("resume_pmard_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_omits_historical_reusable_dependencies_and_keeps_new_ones():
    module = _resume_module()
    jobs = {"training_lock": "43732", "teachers": "90001"}

    assert module._active_dependency_ids(
        {"dependencies": ["training_lock"]}, jobs, ["teachers"],
    ) == []
    assert module._active_dependency_ids(
        {"dependencies": ["training_lock", "teachers"]}, jobs, ["teachers"],
    ) == ["90001"]


def test_resume_scheduler_failure_includes_stderr(monkeypatch):
    module = _resume_module()

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "sbatch: error: Invalid job dependency specification\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match="Invalid job dependency specification"):
        module._run(["sbatch"])
