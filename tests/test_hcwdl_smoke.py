from pathlib import Path

from hlt_classification.scouting.hcwdl_ladder import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_smoke import run_local_smoke


def test_complete_local_fixture_smoke_runs_all_nodes(tmp_path: Path):
    report = run_local_smoke(tmp_path / "smoke")
    assert report["nodes_completed"] == list(NODE_REGISTRY)
    assert report["updates_per_node"] == 2
    assert report["final_test_accessed"] is False
    assert report["repair_endpoint_exact"] is True
    assert set(report["assignment_manifests"]) == {"train", "validation"}
    assert (tmp_path / "smoke/smoke_report.json").is_file()
