from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from hlt_classification.scouting.hcwdl_representation_graph import ASCENT_GRAPH_SHA256
from hlt_classification.scouting.hcwdl_representation_training import (
    _prune_checkpoint_candidates_for_resume,
)
from hlt_classification.scouting.hcwdl_representation_resume import (
    REQUIRED_STATE_NAMESPACES,
    ResumePublicationInterrupted,
    build_state_inventory,
    load_highest_valid_resume,
    publish_resume_generation,
    scan_resume_generations,
    validate_resume_generation,
)


def _lineage(marker: str = "a") -> dict[str, str]:
    return {
        "ascent_graph": ASCENT_GRAPH_SHA256,
        "execution": marker * 64,
        "producer_runtime_signature": "b" * 64,
        "representation_recipe": "c" * 64,
        "target_generation": "d" * 64,
        "target_logical": "e" * 64,
    }


def _state(
    *,
    sequence: int,
    lineage: dict[str, str] | None = None,
    projections: tuple[str, ...] = ("jet", "token"),
    calibration_hashes: dict[str, str] | None = None,
):
    bound = _lineage() if lineage is None else lineage
    calibration = {} if calibration_hashes is None else calibration_hashes
    return {
        "deployable_model": {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + sequence,
        },
        "representation_heads": {
            name: {"weight": torch.eye(2, dtype=torch.float32) + sequence}
            for name in projections
        },
        "optimizer": {
            "state": {0: {"step": torch.tensor(sequence), "exp_avg": torch.ones(2)}},
            "param_groups": [{"params": [0], "lr": 3e-4}],
        },
        "scheduler": {"last_epoch": sequence},
        "sampler": {"epoch": sequence, "batch_offset": 0},
        "trimmer": {"counter": sequence},
        "rng": {
            "torch": torch.arange(8, dtype=torch.uint8) + sequence,
            "python": (3, (1, 2, 3), None),
        },
        "rng_streams": {
            "contract": "HCWDL_REP_PAIRED_RNG_STREAMS/v1",
            "content_hash": "9" * 64,
        },
        "model_runtime": {"training": True, "buffers": {}},
        "cursor": {
            "completed_pass": sequence,
            "completed_update": sequence * 10,
            "next_canonical_batch": sequence + 1,
        },
        "interval_aggregates": {
            "start_update": sequence * 10 + 1,
            "updates": 0,
            "sums": {"base": torch.tensor(0.0)},
        },
        "validation_history": [
            {"pass": sequence, "macro_ovr_auc": 0.9, "cross_entropy": 0.5},
        ],
        "selection_state": {
            "best_checkpoint_identity": f"checkpoint-{sequence}",
            "best_update": sequence * 10,
        },
        "calibration": {
            "artifact_hashes": dict(sorted(calibration.items())),
            "component_status": {},
        },
        "target_bindings": {
            "generation_sha256": bound["target_generation"],
            "logical_target_sha256": bound["target_logical"],
        },
        "producer_runtime_signature": {
            "content_hash": bound["producer_runtime_signature"],
            "python": "fixture",
        },
    }


def _publish(root: Path, sequence: int, **kwargs):
    lineage = kwargs.pop("lineage", _lineage())
    projections = kwargs.pop("projections", ("jet", "token"))
    calibration_hashes = kwargs.pop("calibration_hashes", {})
    state = kwargs.pop(
        "state",
        _state(
            sequence=sequence,
            lineage=lineage,
            projections=projections,
            calibration_hashes=calibration_hashes,
        ),
    )
    return publish_resume_generation(
        root,
        sequence=sequence,
        state=state,
        lineage=lineage,
        completed_pass=sequence,
        completed_update=sequence * 10,
        next_canonical_batch=sequence + 1,
        active_projections=projections,
        calibration_artifact_hashes=calibration_hashes,
        **kwargs,
    )


def test_inventory_is_exact_and_supports_realistic_optimizer_integer_keys():
    state = _state(sequence=1)
    inventory, digest = build_state_inventory(state)
    assert set(inventory) == REQUIRED_STATE_NAMESPACES
    assert len(digest) == 64
    assert inventory["deployable_model"]["tensor_and_array_leaves"][0] == {
        "path": "deployable_model[str:weight]",
        "kind": "torch_tensor",
        "dtype": "torch.float32",
        "shape": [2, 3],
        "requires_grad": False,
        "logical_sha256": inventory["deployable_model"]["tensor_and_array_leaves"][0][
            "logical_sha256"
        ],
    }

    missing = dict(state)
    missing.pop("rng")
    with pytest.raises(ValueError, match="namespaces differ"):
        build_state_inventory(missing)
    extra = dict(state)
    extra["teacher_model"] = {}
    with pytest.raises(ValueError, match="namespaces differ"):
        build_state_inventory(extra)
    nonfinite = deepcopy(state)
    nonfinite["selection_state"]["score"] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        build_state_inventory(nonfinite)


def test_publish_load_and_two_generation_retention(tmp_path: Path):
    for sequence in range(4):
        published = _publish(tmp_path, sequence)
        assert published.sequence == sequence
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "commit_2.json", "commit_3.json",
        "state_2.json", "state_2.pt", "state_3.json", "state_3.pt",
    ]
    loaded, scan = load_highest_valid_resume(tmp_path, expected_lineage=_lineage())
    assert loaded is not None and loaded.sequence == 3
    assert [row.sequence for row in scan.valid_generations] == [2, 3]
    assert scan.invalid_commits == ()
    assert scan.orphan_files == ()
    assert torch.equal(
        loaded.state["deployable_model"]["weight"],
        _state(sequence=3)["deployable_model"]["weight"],
    )


@pytest.mark.parametrize(
    ("crash_point", "valid", "orphan_count"),
    [
        ("before_state", False, 0),
        ("after_state", False, 1),
        ("after_sidecar", False, 2),
        ("before_commit", False, 2),
        ("after_commit", True, 0),
        ("after_validation_before_cleanup", True, 0),
    ],
)
def test_fault_injection_never_exposes_an_uncommitted_pair(
    tmp_path: Path, crash_point: str, valid: bool, orphan_count: int,
):
    root = tmp_path / crash_point
    with pytest.raises(ResumePublicationInterrupted, match=crash_point):
        _publish(root, 0, crash_after=crash_point)
    loaded, scan = load_highest_valid_resume(root, expected_lineage=_lineage())
    assert (loaded is not None) is valid
    assert len(scan.orphan_files) == orphan_count
    if valid:
        assert loaded is not None and loaded.sequence == 0


def test_same_generation_retry_completes_orphan_pair_and_reuses_exact_bytes(tmp_path: Path):
    with pytest.raises(ResumePublicationInterrupted):
        _publish(tmp_path, 0, crash_after="after_sidecar")
    generation = _publish(tmp_path, 0)
    assert generation.sequence == 0
    assert validate_resume_generation(
        tmp_path, sequence=0, expected_lineage=_lineage(),
    ).commit["content_hash"] == generation.commit["content_hash"]


@pytest.mark.parametrize(
    ("crash_point", "orphan_names"),
    [
        (
            "during_pruning_after_commit_delete",
            {"state_0.pt", "state_0.json"},
        ),
        (
            "during_pruning_after_sidecar_delete",
            {"state_0.pt"},
        ),
    ],
)
def test_crash_during_old_generation_pruning_retains_two_newest_commits(
    tmp_path: Path, crash_point: str, orphan_names: set[str],
):
    _publish(tmp_path, 0)
    _publish(tmp_path, 1)
    with pytest.raises(ResumePublicationInterrupted, match=crash_point):
        _publish(tmp_path, 2, crash_after=crash_point)
    loaded, scan = load_highest_valid_resume(tmp_path, expected_lineage=_lineage())
    assert loaded is not None and loaded.sequence == 2
    assert [row.sequence for row in scan.valid_generations] == [1, 2]
    assert set(scan.orphan_files) == orphan_names


def test_corrupt_newest_commit_falls_back_to_highest_fully_valid_generation(tmp_path: Path):
    _publish(tmp_path, 0)
    _publish(tmp_path, 1)
    (tmp_path / "state_1.pt").write_bytes(b"corrupt")
    loaded, scan = load_highest_valid_resume(tmp_path, expected_lineage=_lineage())
    assert loaded is not None and loaded.sequence == 0
    assert [row.sequence for row in scan.invalid_commits] == [1]
    assert "state bytes differ" in scan.invalid_commits[0].reason


def test_best_checkpoint_change_retains_candidate_for_two_generation_fallback(
    tmp_path: Path,
):
    resume = tmp_path / "resume"
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    first = candidates / "first.pt"
    first.write_bytes(b"first selected checkpoint")

    state_zero = _state(sequence=0)
    state_zero["selection_state"] = {"checkpoint_path": str(first.resolve())}
    _publish(resume, 0, state=state_zero)
    _prune_checkpoint_candidates_for_resume(
        candidates, resume_root=resume, lineage=_lineage(),
    )

    second = candidates / "second.pt"
    second.write_bytes(b"second selected checkpoint")
    state_one = _state(sequence=1)
    state_one["selection_state"] = {"checkpoint_path": str(second.resolve())}
    _publish(resume, 1, state=state_one)
    _prune_checkpoint_candidates_for_resume(
        candidates, resume_root=resume, lineage=_lineage(),
    )
    assert first.is_file() and second.is_file()

    (resume / "state_1.pt").write_bytes(b"corrupt newest state")
    loaded, scan = load_highest_valid_resume(resume, expected_lineage=_lineage())
    assert loaded is not None and loaded.sequence == 0
    assert loaded.state["selection_state"]["checkpoint_path"] == str(first.resolve())
    assert first.is_file()
    assert [row.sequence for row in scan.invalid_commits] == [1]


def test_resume_rejects_cross_target_runtime_or_namespace_lineage(tmp_path: Path):
    _publish(tmp_path, 0)
    changed = _lineage()
    changed["target_logical"] = "f" * 64
    loaded, scan = load_highest_valid_resume(tmp_path, expected_lineage=changed)
    assert loaded is None
    assert [row.sequence for row in scan.invalid_commits] == [0]
    assert "lineage differs" in scan.invalid_commits[0].reason

    bad_state = _state(sequence=1)
    bad_state["target_bindings"] = {
        "generation_sha256": _lineage()["target_generation"],
        "logical_target_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="target bindings"):
        _publish(tmp_path / "bad", 1, state=bad_state)


def test_report_or_uncommitted_files_are_never_resume_contracts(tmp_path: Path):
    torch.save(_state(sequence=0), tmp_path / "training_report.pt")
    (tmp_path / "state_8.pt").write_bytes(b"orphan")
    loaded, scan = load_highest_valid_resume(tmp_path)
    assert loaded is None
    assert scan.valid_generations == ()
    assert set(scan.orphan_files) == {"state_8.pt", "training_report.pt"}
