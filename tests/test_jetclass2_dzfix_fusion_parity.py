"""Early fusion diagnostics cannot read test, rerun matching or authorize fits."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes import (
    dzfix_fusion_parity as parity, dzfix_fusion_runtime as runtime,
    dzfix_fusion_chain as campaign, salience_foundation as foundation,
    salience_learned_cache as cache,
)
from test_jetclass2_delphes import snapshot
from test_jetclass2_delphes_salience import registered
from test_jetclass2_dzfix_fusion_chain import spec_at, make_cache


@pytest.fixture
def sample_spec(registered, tmp_path):
    data, inventory, splits = registered
    f = foundation.build_foundation_spec(inventory, splits, "SALIENCE_PT_LINEAR")
    root = tmp_path / "foundation"
    for task in f["assignment_tasks"]:
        if task["role"] == "train":
            foundation.build_assignment_shard(f, data_root=data, output_root=root, file_index=task["file_index"])
    return dict(foundation=f, source_import=dict(foundation_root=str(root)), data_root=str(data))


def test_early_sample_exact_production_views_and_no_rematching(sample_spec, monkeypatch):
    native_reader = parity.DatasetReader
    def reader(*args, **kwargs):
        assert kwargs["role"] == "train" and kwargs["step_size"] == 64
        assert len(kwargs["file_paths"]) == 1
        return native_reader(*args, **kwargs)
    monkeypatch.setattr(parity, "DatasetReader", reader)
    def forbidden(*args, **kwargs):
        pytest.fail("Must consume authenticated assignments, not rematch")
    from hlt_classification.jetclass2_delphes import salience_views
    monkeypatch.setattr(salience_views, "match_particles", forbidden)
    small, sample = parity.parity_sample(sample_spec)
    assert sample["rows"] == 4 and sample["role"] == "train" and not sample["final_test_accessed"]
    for coordinate, subset in small.items():
        full = cache.prepare_cache(sample_spec["foundation"], data_root=Path(sample_spec["data_root"]),
            foundation_root=Path(sample_spec["source_import"]["foundation_root"]),
            role="train", coordinate_name=coordinate, workers=1, max_ram_bytes=100_000_000)
        positions = np.asarray([np.flatnonzero(np.all(full.identities == i, axis=1)).item() for i in subset.identities])
        for key, value in subset.batch(np.arange(4)).items():
            np.testing.assert_array_equal(value, full.batch(positions)[key])


def test_early_sample_rejects_wrong_identity(sample_spec, monkeypatch):
    original = parity.load_assignments
    def wrong(*args, **kwargs):
        report, arrays = original(*args, **kwargs)
        arrays["identities"][0] = 0
        return report, arrays
    monkeypatch.setattr(parity, "load_assignments", wrong)
    with pytest.raises(ValueError, match="identity join"):
        parity.parity_sample(sample_spec)


def test_early_sample_never_falls_back_to_validation_or_test():
    with pytest.raises(ValueError, match="four distinct"):
        parity.parity_sample(dict(foundation=dict(assignment_tasks=[dict(role="final_test", rows=10)])))


@pytest.mark.parametrize("failure", [None, "bf16"])
def test_early_parity_durable_diagnostics_and_fail_before_cache(tmp_path, monkeypatch, failure):
    spec = spec_at(tmp_path)
    raw_cache = make_cache("train")
    sample = dict(role="train", rows=4, file_indices=[0]*4,
                  identities=[f"{i:064x}" for i in range(4)], final_test_accessed=False)
    monkeypatch.setattr(parity, "parity_sample", lambda s: ({k:raw_cache for k in ("U000","U050","D000")}, sample))
    monkeypatch.setattr(parity, "native_mask_parity", lambda *a, **k: True)
    seen = []
    def storage(*args, **kwargs):
        precision = "bf16" if kwargs["bf16"] else "fp32"
        seen.append(precision)
        if precision == failure:
            raise AssertionError("injected gradient mismatch")
        return dict(precision=precision, passed=True, test_only=True)
    monkeypatch.setattr(parity, "native_offload_parity", storage)
    if failure:
        monkeypatch.setitem(sys.modules, "resource", SimpleNamespace())
        monkeypatch.setattr(runtime, "execution_gate", lambda *a, **k: "123")
        monkeypatch.setattr(runtime, "installed_environment", lambda: {})
        def forbidden(*a, **k):
            pytest.fail("Full cache must not start before early parity passes")
        monkeypatch.setattr(runtime, "prepare", forbidden)
        with pytest.raises(AssertionError, match="injected gradient mismatch"):
            runtime.preflight(spec, tmp_path, "cuda")
        assert seen == ["fp32", "bf16"]
        assert len(list(tmp_path.glob("early_parity_*.json"))) == 1
        assert not (tmp_path / "acceptance.json").exists()
    else:
        reports = parity.early_parity(spec, tmp_path, "cuda")
        assert seen == ["fp32", "bf16", "fp32", "bf16"]
        for path in tmp_path.glob("early_parity_*.json"):
            report = load_json(path)
            campaign.validate(report, "EARLY_PARITY")
            assert report in reports and report["acceptance_only"]


def test_worker_configures_cublas_for_preflight_only():
    worker = (Path(__file__).resolve().parents[1] / "sbatch/run_jetclass2_dzfix_fusion_chain.sh").read_text()
    assert '"${MODE}" == run && "${TASK}" == preflight' in worker
    assert 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' in worker
    assert worker.index('export CUBLAS_WORKSPACE_CONFIG') < worker.index('exec python')
