"""Focused contracts for the standalone native-offline CE control."""
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.cache import RamBlock
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes.execution import execution_site


def source_campaign(tmp_path):
    foundation = artifact(
        "SALIENCE_FOUNDATION_SPEC", candidate="SALIENCE_PT_LINEAR",
        inventory={}, splits={"role_counts": {
            "train": 500_000, "validation": 1_000_000, "final_test": 1_000_000,
        }}, inputs={}, assignment_tasks=[], final_test_accessed=False,
    )
    return artifact(
        "SALIENCE_CAMPAIGN_SPEC", source_commit="b" * 40,
        project_dir=str(tmp_path / "old_project"),
        campaign_root=str(tmp_path / "source"), data_root=str(tmp_path / "data"),
        foundation=foundation, foundation_root=str(tmp_path / "foundation"),
        foundation_lock_sha256="c" * 64,
        runtime_profile=dict(
            execution_site=execution_site("sporc_a100"), cpus=8,
            workers=8, memory_mb=73728, train_minutes=808,
            cache_budgets={"train": 10**9, "validation": 2 * 10**9},
            gpu={}, installed_environment={},
        ),
        scientific_plan=dict(
            recipe=recipe(), split_profile="TRAIN_500K",
            role_counts={"train": 500_000, "validation": 1_000_000,
                         "final_test": 1_000_000},
            role_membership_sha256={"train": "d" * 64,
                                    "validation": "e" * 64},
            nodes=[dict(node_id="U000", teacher=None,
                        initialization_seed=1234, sampler_seed=5678)],
        ),
        final_test_accessed=False,
    )


def test_control_is_one_paired_native_offline_fit_and_isolated_tier3(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes import pure_offline_control as control

    source = source_campaign(tmp_path)
    monkeypatch.setattr(control, "_source", lambda *args: None)
    monkeypatch.setattr(control, "_load_source", lambda path: source)
    spec = control.create_control(
        source_campaign_spec=tmp_path / "source/campaign_spec.json",
        output_root=tmp_path / "offline_control", project=tmp_path,
        source_commit="a" * 40,
    )
    assert spec["fresh_fit_count"] == 1
    assert spec["reducer_count"] == 0
    assert spec["pure_offline"] is True
    assert spec["persistent_hlt"] is False
    assert spec["matching_assignments_used"] is False
    assert spec["scheduler_dependency_on_source_campaign"] is False
    assert spec["node"]["coordinate"] == "OFFLINE"
    assert spec["node"]["initialization_seed"] == 1234
    assert spec["node"]["sampler_seed"] == 5678
    assert spec["training_recipe"] == recipe()

    plan = load_json(tmp_path / "offline_control/command_plan.json")
    assert [row["task_id"] for row in plan["commands"]] == [
        "authenticate", "train_OFFLINE", "control_complete",
    ]
    assert all("--partition=tier3" in row["command"] for row in plan["commands"])
    assert plan["commands"][0]["dependencies"] == []
    assert plan["commands"][1]["dependencies"] == ["authenticate"]
    assert plan["commands"][2]["dependencies"] == ["train_OFFLINE"]
    assert plan["commands"][1]["command"].count("--gres=gpu:a100:1") == 1
    assert not any(token.startswith("--gres=") for token in plan["commands"][0]["command"])
    assert plan["source_campaign_dependencies"] == []
    with pytest.raises(PermissionError, match="authorization"):
        control.submit_control(
            spec, bookkeeping_root=tmp_path / "offline_control",
            execute=True, authorization_phrase="wrong",
        )


def test_native_offline_cache_uses_assignment_free_original_u000_adapter(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes import native_offline

    foundation = {
        "assignment_tasks": [
            {"file_index": 4, "path": "train.root", "role": "train", "rows": 2},
        ],
        "splits": {"role_counts": {"train": 2, "validation": 3}},
    }
    monkeypatch.setattr(native_offline, "validate_foundation_spec", lambda value: "f" * 64)
    seen = []

    def prepare(arguments):
        seen.append(arguments)
        return RamBlock(
            file_index=4, offsets=np.asarray([0, 2, 3], np.int64),
            features=np.zeros((3, 17), np.float32),
            vectors=np.zeros((3, 4), np.float32),
            identities=np.zeros((2, 32), np.uint8),
            labels=np.asarray([0, 1], np.int64),
        )

    monkeypatch.setattr(native_offline, "_prepare_file", prepare)
    cache = native_offline.prepare_native_offline_cache(
        foundation, data_root=tmp_path, role="train", workers=1,
        max_ram_bytes=10_000_000,
    )
    assert len(cache) == 2
    assert cache.coordinate_name == "OFFLINE"
    assert cache.foundation_sha256 == "f" * 64
    assert seen[0][2] == ""
    assert seen[0][4] == "U000"
    with pytest.raises(PermissionError, match="sealed"):
        native_offline.prepare_native_offline_cache(
            foundation, data_root=tmp_path, role="final_test", workers=1,
            max_ram_bytes=10_000_000,
        )
