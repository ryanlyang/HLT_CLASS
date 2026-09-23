"""Early K2 parity uses real registered rows and fails before full RAM caches."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_data as data, concat_k2_parity as parity,
    concat_k2_runtime as runtime, concat_k2_campaign as campaign,
)
from test_jetclass2_concat_k2 import spec_at
from test_jetclass2_delphes import snapshot
from test_jetclass2_concat_k2_reuse import completed_donor
from test_jetclass2_concat_k2_memory import memory_evidence
from test_jetclass2_dzfix_fusion_chain import make_cache


def test_bounded_authenticated_train_sample_matches_production_views(completed_donor, monkeypatch):
    spec = completed_donor
    actual_reader = parity.DatasetReader
    calls = []

    def reader(*args, **kwargs):
        assert kwargs["role"] == "train" and kwargs["step_size"] == 64
        assert len(kwargs["file_paths"]) == 1
        calls.append(kwargs)
        return actual_reader(*args, **kwargs)

    def forbidden(*a, **k):
        raise AssertionError("Early parity must not rerun matching")

    monkeypatch.setattr(parity, "DatasetReader", reader)
    monkeypatch.setattr(data, "assignment", forbidden)
    caches, sample = parity.parity_sample(spec)
    assert calls and len(caches["D100"]) == len(caches["D000"]) == sample["rows"] == 4
    assert sample["role"] == "train" and sample["final_test_accessed"] is False
    for coordinate, small in caches.items():
        full = data.prepare(spec,"train",coordinate)
        positions = [np.flatnonzero(np.all(full.identities == identity,axis=1)).item()
                     for identity in small.identities]
        for key, value in small.batch(np.arange(4)).items():
            np.testing.assert_array_equal(value,full.batch(np.asarray(positions))[key])


def test_early_sample_rejects_bad_identity_join(completed_donor, monkeypatch):
    original = parity.load_assignment

    def wrong(*args):
        report, values = original(*args)
        values = dict(values, identities=values["identities"].copy())
        values["identities"][0] = 0
        return report, values

    monkeypatch.setattr(parity,"load_assignment",wrong)
    with pytest.raises(ValueError,match="identity join"):
        parity.parity_sample(completed_donor)


def test_early_sample_rejects_missing_train_rows(tmp_path):
    spec = spec_at(tmp_path)
    spec["foundation"]["assignment_tasks"] = [dict(role="validation",rows=4)]
    with pytest.raises(ValueError,match="four distinct"):
        parity.parity_sample(spec)


@pytest.mark.parametrize("failure", [None, "bf16"])
def test_early_reports_are_durable_and_failure_precedes_full_cache(tmp_path, monkeypatch, failure):
    spec = spec_at(tmp_path)
    fake = memory_evidence(spec)
    small = make_cache("train")
    seen = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    monkeypatch.setattr(parity,"parity_sample",lambda s: (
        {"D100":small,"D000":small},fake["early_parity_reports"][0]["sample"]))
    monkeypatch.setattr(parity,"installed_parity",lambda *a,**k: fake["early_parity_reports"][0]["native_parity"])

    def storage(*args,**kwargs):
        precision = "bf16" if kwargs["bf16"] else "fp32"
        seen.append(precision)
        if precision == failure:
            raise AssertionError("injected gradient mismatch")
        return fake["early_parity_reports"][int(kwargs["bf16"])]["storage_parity"]

    monkeypatch.setattr(parity,"storage_parity",storage)
    if failure:
        monkeypatch.setitem(sys.modules,"resource",SimpleNamespace())
        monkeypatch.setattr(runtime,"execution_gate",lambda *a,**k:"123")
        monkeypatch.setattr(runtime,"installed_environment",lambda:{})
        monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
        def forbidden(*a,**k):
            raise AssertionError("Full cache was built before early parity passed")
        monkeypatch.setattr(runtime,"prepare",forbidden)
        with pytest.raises(AssertionError,match="injected gradient mismatch"):
            runtime.preflight(spec,tmp_path,"cuda")
        assert seen == ["fp32","bf16"]
        assert len(list(tmp_path.glob("early_parity_*.json"))) == 1
        assert not (tmp_path/"acceptance.json").exists()
    else:
        records = parity.early_parity(spec,tmp_path,"cuda")
        assert seen == ["fp32","bf16","fp32","bf16"]
        assert len(records) == 4
        for path in tmp_path.glob("early_parity_*.json"):
            row = load_json(path)
            campaign.validate(row,"EARLY_PARITY")
            assert row in records
