from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, sha256_file,
)
from hlt_classification.scouting.hcwdl_mhpe_roc import _probability_metrics
from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble import (
    CANDIDATE_ORDER, ENSEMBLE_POLICY, MAX_ENSEMBLE_SIZE, OBJECTIVE_ORDER,
    SHARDS, _validate_foundation_boundary, greedy_paths, shard_paths,
    validate_prediction_shard,
)
from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble_campaign import (
    RESOURCES, SCHEDULER_NICE, _command_plan, campaign_tasks,
)
from hlt_classification.scouting.hcwdl_tri60_m1_greedy_ensemble_contracts import (
    SHARD_REPORT_CONTRACT, SOURCE_LOCK_CONTRACT, artifact, validate_artifact,
)


def _probabilities(labels: np.ndarray, strength: float) -> np.ndarray:
    rng = np.random.default_rng(round(strength * 10_000))
    logits = rng.normal(0.0, 1.0, size=(len(labels), 15))
    logits[np.arange(len(labels)), labels] = strength
    logits[np.arange(len(labels)), labels] += rng.normal(0.0, 0.35, len(labels))
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    values /= values.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(values, dtype=np.float32)


def test_registry_is_exact_twenty_candidates_in_five_four_model_shards():
    assert len(CANDIDATE_ORDER) == 20
    assert len(SHARDS) == 5
    assert all(len(shard) == 4 for shard in SHARDS)
    assert tuple(item for shard in SHARDS for item in shard) == CANDIDATE_ORDER
    assert ENSEMBLE_POLICY.endswith("fp64_accumulation_to_fp32_v1")


def test_foundation_boundary_distinguishes_tri60_lock_from_foundation_spec(
    tmp_path: Path,
):
    spec_path = (tmp_path / "foundation_spec.json").resolve()
    spec_hash = "1" * 64
    tri60_lock_hash = "2" * 64
    source_hash = "3" * 64
    screen = {
        "parents": {
            "foundation": tri60_lock_hash,
            "source_campaign": source_hash,
        },
        "role_counts": {"validation": 957541},
    }
    source = {"content_hash": source_hash}
    tri60_lock = {
        "parents": {"foundation_spec": spec_hash},
        "foundation_spec_path": str(spec_path),
    }

    _validate_foundation_boundary(
        screen=screen, source_campaign=source,
        foundation_path=spec_path, foundation_hash=spec_hash,
        tri60_foundation=tri60_lock,
        tri60_foundation_hash=tri60_lock_hash,
    )

    with pytest.raises(ValueError, match="foundation boundary differs"):
        _validate_foundation_boundary(
            screen=screen, source_campaign=source,
            foundation_path=spec_path, foundation_hash=spec_hash,
            tri60_foundation=tri60_lock,
            # The production bug compared this spec hash to the lock parent.
            tri60_foundation_hash=spec_hash,
        )


def test_greedy_paths_are_deterministic_equal_weight_and_stop_at_five():
    labels = np.concatenate((
        np.zeros(1200, dtype=np.int64),
        np.repeat(np.arange(1, 15, dtype=np.int64), 80),
    ))
    values = {
        candidate: _probabilities(labels, 1.5 + 0.01 * index)
        for index, candidate in enumerate(CANDIDATE_ORDER)
    }
    expected = {
        candidate: _probability_metrics(probability, labels)
        for candidate, probability in values.items()
    }
    teacher = _probability_metrics(_probabilities(labels, 2.0), labels)
    paths, evaluated = greedy_paths(
        probabilities=values, labels=labels, expected_metrics=expected,
        teacher_metrics=teacher, workers=1,
    )
    assert tuple(paths) == OBJECTIVE_ORDER
    assert evaluated >= len(CANDIDATE_ORDER)
    for rows in paths.values():
        assert [row["ensemble_size"] for row in rows] == list(
            range(1, MAX_ENSEMBLE_SIZE + 1)
        )
        for size, row in enumerate(rows, 1):
            assert len(row["members"]) == len(set(row["members"])) == size
            assert row["uniform_member_weight"] == 1 / size
            assert row["added_member"] in row["members"]
            assert np.isfinite(row["objective_score"])


def test_prediction_shard_roundtrip_is_content_and_lineage_bound(tmp_path: Path):
    rows = 30
    identities = np.arange(rows * 32, dtype=np.uint8).reshape(rows, 32)
    labels = np.arange(rows, dtype=np.int64) % 15
    candidates = SHARDS[0]
    arrays = {"identity_digests": identities, "labels": labels}
    lineage = {}
    for index, candidate in enumerate(candidates):
        name = f"probabilities__{candidate}"
        arrays[name] = _probabilities(labels, 1.4 + index / 10)
        lineage[candidate] = {
            "report_sha256": f"{10 + index:064x}",
            "checkpoint_sha256": f"{20 + index:064x}",
            "probabilities_sha256": array_sha256(name, arrays[name]),
        }
    root = tmp_path / "campaign"
    data_path, _ = shard_paths(root, 0)
    atomic_publish_bytes(data_path, deterministic_npz_bytes(arrays))
    spec = {
        "content_hash": "1" * 64,
        "parents": {"source_lock": "2" * 64},
        "campaign_root": str(root),
        "artifact_paths": {"source_lock": str(root / "source.json")},
        "role_counts": {"validation": rows},
        "maximum_shard_bytes": 320 * 1024**2,
    }
    source = artifact({
        "parents": {}, "artifact_paths": {},
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_reports": {
            candidate: {
                "report_sha256": lineage[candidate]["report_sha256"],
                "checkpoint_sha256": lineage[candidate]["checkpoint_sha256"],
            }
            for candidate in candidates
        },
        "teacher": {}, "replicate_seed": 1,
        "role_counts": {"validation": rows},
        "screen_source_outputs_read_only": True,
        "source_scheduler_dependency": False,
        "ordinary_access_roles": ["validation"], "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    # The shard validator binds to the spec's source-lock hash, so the fixture
    # uses that exact immutable source artifact.
    spec["parents"]["source_lock"] = source["content_hash"]
    Path(spec["artifact_paths"]["source_lock"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["artifact_paths"]["source_lock"]).write_text(json.dumps(source))
    report = artifact({
        "parents": {
            "campaign_spec": spec["content_hash"],
            "source_lock": spec["parents"]["source_lock"],
            "screen_campaign": "3" * 64, "foundation": "4" * 64,
        },
        "shard_index": 0, "candidate_ids": list(candidates),
        "candidate_lineage": lineage,
        "prediction_path": str(data_path.resolve()),
        "prediction_file_sha256": sha256_file(data_path),
        "prediction_file_bytes": data_path.stat().st_size,
        "identity_order_sha256": array_sha256("identity_digests", identities),
        "labels_sha256": array_sha256("labels", labels),
        "validation_rows": rows, "probability_dtype": "float32",
        "persistent_particle_views": False, "persistent_logits": False,
        "ordinary_access_roles": ["validation"], "runtime_seconds": 1.0,
        "final_test_accessed": False,
    }, contract=SHARD_REPORT_CONTRACT)
    observed_ids, observed_labels, observed = validate_prediction_shard(
        report, spec=spec, shard_index=0,
    )
    assert np.array_equal(observed_ids, identities)
    assert np.array_equal(observed_labels, labels)
    assert tuple(observed) == candidates
    assert np.array_equal(observed[candidates[2]], arrays[f"probabilities__{candidates[2]}"])


def test_campaign_dag_is_isolated_five_gpu_shards_then_one_reducer(tmp_path: Path):
    tasks = campaign_tasks()
    assert len(tasks) == 8
    shards = [row for row in tasks if row["kind"] == "inference_shard"]
    assert len(shards) == 5
    assert all(row["dependencies"] == ["authenticate"] for row in shards)
    reducer = next(row for row in tasks if row["kind"] == "greedy_reduce")
    assert reducer["dependencies"] == [row["task_id"] for row in shards]
    assert RESOURCES["gpu_inference"].gpu == "gpu:gh200:1"
    spec = {
        "content_hash": "5" * 64,
        "project_dir": str(tmp_path / "project"),
        "campaign_root": str(tmp_path / "campaign"),
        "spec_path": str(tmp_path / "campaign/campaign_spec.json"),
        "tasks": tasks,
        "resources": {
            name: {
                "cpus": row.cpus, "memory": row.memory,
                "walltime": row.walltime, "gpu": row.gpu,
            }
            for name, row in RESOURCES.items()
        },
    }
    plan = _command_plan(spec)
    assert len(plan["commands"]) == 8
    assert plan["source_scheduler_dependencies"] == []
    assert all(
        f"--nice={SCHEDULER_NICE}" in row["command"]
        for row in plan["commands"]
    )
    assert not any("final_test" in json.dumps(row) for row in plan["commands"])


def test_campaign_publication_has_exact_isolated_shape(tmp_path: Path, monkeypatch):
    from hlt_classification.scouting import (
        hcwdl_tri60_m1_greedy_ensemble_campaign as campaign,
    )

    source = artifact({
        "parents": {
            "screen_campaign": "1" * 64, "screen_aggregate": "2" * 64,
            "screen_complete": "3" * 64, "screen_source_lock": "4" * 64,
            "source_campaign": "5" * 64, "foundation": "6" * 64,
            "foundation_spec": "7" * 64, "recipe": "8" * 64,
            "screen_graph": "9" * 64, "teacher_stage": "a" * 64,
        },
        "artifact_paths": {"screen_campaign_spec": str(tmp_path / "screen.json")},
        "candidate_order": list(CANDIDATE_ORDER), "candidate_reports": {},
        "teacher": {"row_id": "LOGIT_D000E", "metrics": {}},
        "replicate_seed": 11,
        "role_counts": {"train": 2777855, "validation": 957541, "final_test": 899779},
        "screen_source_outputs_read_only": True,
        "source_scheduler_dependency": False,
        "ordinary_access_roles": ["validation"], "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(
        campaign, "validate_source_lock",
        lambda value: validate_artifact(value, contract=SOURCE_LOCK_CONTRACT),
    )
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        screen_campaign_spec=tmp_path / "screen.json", campaign_root=root,
        project_dir=tmp_path / "project", source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 8
    assert spec["candidate_count"] == 20
    assert spec["inference_shard_count"] == 5
    assert spec["source_campaign_scheduler_dependency"] is False
    assert spec["screen_campaign_scheduler_dependency"] is False
    assert spec["fresh_fit_count"] == 0
    assert spec["ordinary_access_roles"] == ["validation"]


def test_worker_is_absolute_source_pinned_and_memory_only_for_views():
    worker = Path(
        "sbatch/run_hcwdl_tri60_m1_greedy_ensemble_task.sh"
    ).read_text()
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in worker
    assert '"${PROJECT_DIR}/scripts/run_hcwdl_tri60_m1_greedy_ensemble_task.py"' in worker
    assert "HCWDL_UB_VIEW_SOURCE_BACKEND=process" in worker
    assert "HCWDL_M1_GREEDY_REDUCER_WORKERS=32" in worker
    assert "final_test" not in worker
