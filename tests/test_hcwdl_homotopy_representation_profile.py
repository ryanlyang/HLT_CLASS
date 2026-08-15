from pathlib import Path

import pytest

from hlt_classification.scouting.hcwdl_homotopy_representation_profile import (
    PROFILE_CONTRACT, PROFILE_NODE_IDS, _summaries,
)


def test_profile_registry_and_exclusive_timing_summary():
    assert PROFILE_CONTRACT == "HCWDL_U_RKD_PERFORMANCE_PROFILE/v1"
    assert PROFILE_NODE_IDS == ("F_RSET_U020", "F_RREL_U020")
    value = _summaries({
        "step_total": [2.0, 4.0],
        "model_forward": [0.5, 1.0],
        "set_representation_loss": [1.0, 2.0],
        "step_other": [0.5, 1.0],
    })
    assert value["step_total"]["mean_seconds"] == pytest.approx(3.0)
    assert value["set_representation_loss"]["p90_seconds"] == pytest.approx(2.0)
    assert value["set_representation_loss"]["percent_of_step_wall"] == pytest.approx(50.0)
    assert value["model_forward"]["percent_of_step_wall"] == pytest.approx(25.0)


def test_profile_cli_and_worker_are_diagnostic_only():
    repository = Path(__file__).resolve().parents[1]
    cli = (
        repository / "scripts/profile_hcwdl_homotopy_representation_node.py"
    ).read_text(encoding="utf-8")
    worker = (
        repository / "sbatch/run_hcwdl_homotopy_representation_profile.sh"
    ).read_text(encoding="utf-8")
    assert "--campaign-spec" in cli
    assert "--batches" in cli
    assert "output must remain outside campaign root" in cli
    assert "train_hcwdl_representation_node" not in cli
    assert ': "${HCWDL_U_RKD_PROFILE_OUTPUT:?HCWDL_U_RKD_PROFILE_OUTPUT is required}"' in worker
    assert 'exec python -s \\\n  "${PROJECT_DIR}/scripts/profile_hcwdl_homotopy_representation_node.py"' in worker
    assert "HCWDL_U_RKD_TASK" not in worker
