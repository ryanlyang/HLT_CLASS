from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT
from hlt_classification.scouting.campaign import PMARD_SITE
from hlt_classification.scouting.exploratory_test import (
    TEST_SEMANTICS, authorize_exploratory_test,
    metrics_only_inference, submit_exploratory_test,
    validate_exploratory_locks, validate_exploratory_test_spec,
)
from hlt_classification.scouting.inputs import ParticleInputs


def _source_snapshot():
    tracked = "1" * 64
    identity = canonical_sha256({
        "git_commit": "a" * 40, "git_tree": "b" * 40,
        "tracked_files_sha256": tracked,
    })
    return with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT, "schema_version": 1,
        "git_commit": "a" * 40, "git_tree": "b" * 40,
        "tracked_files_sha256": tracked, "tracked_file_count": 1,
        "worktree_clean": True, "source_snapshot_sha256": identity,
    })


def _spec(tmp_path: Path, *, followup_models: int = 27):
    artifacts = {
        name: {"path": str((tmp_path / f"{name}.json").resolve()), "content_hash": "2" * 64}
        for name in (
            "parent_sweep_spec", "parent_sweep_report", "followup_spec",
            "followup_report", "split_manifest",
        )
    }
    registry = []
    model_count = 36 + followup_models
    for index in range(model_count):
        source = "t100_sweep" if index < 36 else "kd_followup"
        experiment = f"model_{index:02d}"
        registry.append({
            "index": index, "evaluation_id": f"{source}__{experiment}",
            "source_study": source,
            "source_index": index if index < 36 else index - 36,
            "experiment_id": experiment,
            "training_report_path": str((tmp_path / f"report_{index}.json").resolve()),
            "training_report_sha256": f"{index + 1:064x}",
            "selected_checkpoint_path": str((tmp_path / f"model_{index}.pt").resolve()),
            "selected_checkpoint_sha256": f"{index + 101:064x}",
            "validation": {}, "scientific_axes": {},
        })
    source = _source_snapshot()
    site = dict(PMARD_SITE); site["project_dir"] = str(tmp_path.resolve())
    identity = canonical_sha256({
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "artifacts": artifacts, "registry": registry, "site": site,
        "candidate_count": model_count, "rows": 100_000,
        "semantics": TEST_SEMANTICS,
    })
    return with_content_hash({
        "contract": "hlt_classification_pmard_exploratory_test_spec_v2",
        "schema_version": 2,
        "study_id": f"pmard_exploratory_test_{identity[:16]}",
        "source_snapshot": source,
        "parent_sweep_root": str((tmp_path / "parent").resolve()),
        "followup_root": str((tmp_path / "followup").resolve()),
        "output_root": str((tmp_path / "output").resolve()),
        "site": site, "artifacts": artifacts, "registry": registry,
        "candidate_count": model_count,
        "tasks": ["authorize", "row_selection", "evaluation", "aggregate"],
        "role": "final_test", "rows": 100_000, "selection_seed": 1337,
        "test_role_semantics": TEST_SEMANTICS,
        "model_inventory_frozen_before_test_access": True,
        "holdout_consumed_for_model_comparison": True,
        "confirmatory_claim_forbidden": True,
        "posthoc_test_ranking_is_descriptive_only": True,
        "predictions_published": False,
    })


def test_exploratory_spec_requires_exact_frozen_distinct_inventory(tmp_path: Path):
    spec = _spec(tmp_path)
    validate_exploratory_test_spec(spec)
    broken = dict(spec); broken["registry"] = broken["registry"][:-1]
    broken = with_content_hash({key: value for key, value in broken.items() if key != "content_hash"})
    with pytest.raises(ValueError, match="distinct-model count"):
        validate_exploratory_test_spec(broken)

    # The implementation must also accept the maximum 36+28 inventory when
    # no follow-up recipe was deduplicated.
    validate_exploratory_test_spec(_spec(tmp_path / "maximum", followup_models=28))


def test_exploratory_authorization_records_consumed_holdout(tmp_path: Path, monkeypatch):
    import hlt_classification.scouting.exploratory_test as exploratory
    spec = _spec(tmp_path)
    monkeypatch.setattr(exploratory, "validate_exploratory_test_inputs", lambda value: {})
    finalist, execution = authorize_exploratory_test(spec)
    assert finalist["candidate_count"] == 63
    assert len(finalist["authorized_evaluation_ids"]) == 63
    assert execution["holdout_consumed_for_model_comparison"] is True
    assert execution["confirmatory_claim_forbidden"] is True
    assert execution["metrics_only"] is True
    assert execution["predictions_published"] is False
    assert validate_exploratory_locks(spec) == (finalist, execution)


class _TinyModel(torch.nn.Module):
    def forward(self, features, vectors, mask):
        del vectors
        pooled = (features * mask).sum((1, 2))
        return pooled[:, None].repeat(1, 15)


def test_metrics_only_inference_is_hlt_only_and_keeps_no_prediction_artifact(tmp_path: Path):
    rows = 30
    view = ParticleInputs(
        np.ones((rows, 21, 2), np.float32),
        np.zeros((rows, 4, 2), np.float32),
        np.ones((rows, 1, 2), np.bool_), np.full(rows, 2, np.int32),
    )
    batches = [{
        "hlt": view, "labels": np.tile(np.arange(15), 2),
        "identity_keys": np.asarray([f"jet::{index}" for index in range(rows)]),
    }]
    metrics, identity_hash, observed = metrics_only_inference(
        _TinyModel(), batches, device="cpu",
    )
    assert observed == rows and metrics["rows"] == rows
    assert len(identity_hash) == 64
    assert list(tmp_path.iterdir()) == []


def test_exploratory_slurm_is_uncapped_exact_model_array(tmp_path: Path, monkeypatch):
    import hlt_classification.scouting.exploratory_test as exploratory
    spec = _spec(tmp_path)
    monkeypatch.setattr(exploratory, "validate_exploratory_test_inputs", lambda value: {})
    ledger = submit_exploratory_test(
        spec, spec_path="/tmp/exploratory.json", dry_run=True,
    )
    assert ledger["jobs"] == {
        "authorize": "94001", "row_selection": "94002",
        "evaluation": "94003", "aggregate": "94004",
    }
    evaluation = ledger["commands"][2]
    assert "--array=0-62" in evaluation
    assert all("%" not in value for value in evaluation if value.startswith("--array="))
    assert "--dependency=afterok:94002" in evaluation
    assert "--dependency=afterok:94003" in ledger["commands"][3]


def test_exploratory_worker_execs_python():
    worker = Path("sbatch/run_pmard_exploratory_test.sh").read_text()
    assert 'exec python -s "${PROJECT_DIR}/scripts/run_pmard_exploratory_test_task.py"' in worker
