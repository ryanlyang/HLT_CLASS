from pathlib import Path

from hlt_classification.scouting.hcwdl_representation_task_runtime import (
    RepresentationPreemptionMonitor,
)


ROOT = Path(__file__).resolve().parents[1]


def test_signal_monitor_exposes_safe_callback_and_restores_handlers():
    monitor = RepresentationPreemptionMonitor()
    monitor.install()
    try:
        assert monitor.is_requested() is False
        monitor._request(None, None)
        assert monitor.is_requested() is True
    finally:
        monitor.restore()
    assert monitor.previous == {}


def test_both_workers_exec_python_and_only_deterministic_worker_sets_cublas():
    ordinary = (ROOT / "sbatch" / "run_hcwdl_representation_task.sh").read_text()
    deterministic = (
        ROOT / "sbatch" / "run_hcwdl_representation_deterministic_task.sh"
    ).read_text()
    assert "exec python -s" in ordinary
    assert "exec python -s" in deterministic
    assert "CUBLAS_WORKSPACE_CONFIG" not in ordinary
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in deterministic

