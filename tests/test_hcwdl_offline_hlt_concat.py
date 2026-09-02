from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_homotopy import prepare_offline_endpoints
from hlt_classification.scouting.hcwdl_offline_hlt_concat_data import (
    HLT_CONTENT, OFFLINE_CONTENT, build_tagged_concat_inputs,
)
from hlt_classification.scouting.hcwdl_representation_training import normalize_hlt_batch
from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.repair import transform_endpoint_features
from hlt_classification.scouting.schema import HLT_FEATURE_SPECS


def _raw_arrays():
    arrays = {spec.branch: [np.zeros(2, np.float32)] for spec in HLT_FEATURE_SPECS}
    hlt = {
        "px": [10, 0], "py": [0, 5], "pz": [0, 0], "energy": [10.1, 5.1],
        "quality": [1, 2], "charge": [1, 0], "isEl": [0, 0], "isMu": [0, 0],
        "isChargedHad": [1, 0], "isGamma": [0, 1], "isNeutralHad": [0, 0],
        "phirel": [.1, -.2], "etarel": [.1, -.2], "abseta": [.1, .2],
        "pt_log": [1.0, .5], "normchi2": [2, 0], "dz": [.01, 0],
        "dxy": [-.02, 0], "dxysig": [-2, 0], "btagEtaRel": [.4, 0],
        "btagPtRatio": [.6, 0], "btagPParRatio": [.7, 0], "dzsig": [1.5, 0],
        "e_log": [1.1, .6], "lostInnerHits": [1, 0],
    }
    for suffix, values in hlt.items():
        arrays[f"scoutpfcand_{suffix}"] = [np.asarray(values, np.float32)]
    charged = {
        "px": 20, "py": 0, "pz": 1, "energy": 20.1, "quality": 5,
        "charge": -1, "isEl": 1, "isMu": 0, "isChargedHad": 0,
        "phirel": -.3, "etarel": .4, "abseta": .8, "pt_log_nopuppi": 2.3,
        "normchi2": 4, "dz": .03, "dxy": -.04, "dxysig": -3,
        "btagEtaRel": .9, "btagPtRatio": .8, "btagPParRatio": .6,
        "dzsig": 2.5, "e_log_nopuppi": 2.5, "lostInnerHits": 2,
    }
    neutral = {
        "px": 0, "py": 12, "pz": -1, "energy": 12.1, "isGamma": 0,
        "isNeutralHad": 1, "phirel": .3, "etarel": -.3, "abseta": .7,
        "pt_log_nopuppi": 1.7, "e_log_nopuppi": 1.8,
    }
    for suffix, value in charged.items():
        arrays[f"cpfcandlt_{suffix}"] = [np.asarray([value], np.float32)]
    for suffix, value in neutral.items():
        arrays[f"npfcand_{suffix}"] = [np.asarray([value], np.float32)]
    arrays["n_scoutpfcands"] = np.asarray([2], np.int32)
    arrays["n_cpfcands"] = np.asarray([1], np.int32)
    arrays["n_lts"] = np.asarray([0], np.int32)
    arrays["n_npfcands"] = np.asarray([1], np.int32)
    return arrays


def test_tagged_concat_is_exact_offline_then_hlt_without_deduplication():
    arrays = _raw_arrays()
    view = build_tagged_concat_inputs(arrays)
    offline = prepare_offline_endpoints(arrays)
    projected = transform_endpoint_features(offline.raw_features[0], offline.validity[0])
    hlt = build_hlt_inputs(arrays)
    assert view.raw_lengths.tolist() == [4]
    assert view.mask[0, 0, :4].all() and not view.mask[0, 0, 4:].any()
    np.testing.assert_array_equal(view.features[0, :, :2], projected.T)
    np.testing.assert_array_equal(view.features[0, :, 2:4], hlt.features[0, :, :2])
    np.testing.assert_array_equal(view.vectors[0, :, :2], np.asarray(offline.p4[0], np.float32).T)
    np.testing.assert_array_equal(view.vectors[0, :, 2:4], hlt.vectors[0, :, :2])
    assert view.content_source_codes[0, :4].tolist() == [
        int(OFFLINE_CONTENT), int(OFFLINE_CONTENT), int(HLT_CONTENT), int(HLT_CONTENT),
    ]
    assert np.all(view.content_source_codes[0, 4:] == -1)
    assert view.visible_indices[0, :4].tolist() == [0, 1, 2, 3]


def test_tagged_concat_fails_instead_of_truncating():
    arrays = _raw_arrays()
    for name, rows in list(arrays.items()):
        if name.startswith("scoutpfcand_"):
            arrays[name] = [np.resize(np.asarray(rows[0]), 200)]
        elif name.startswith("cpfcandlt_"):
            arrays[name] = [np.resize(np.asarray(rows[0]), 297)]
    arrays["n_scoutpfcands"] = np.asarray([200], np.int32)
    arrays["n_cpfcands"] = np.asarray([297], np.int32)
    with pytest.raises(ValueError, match="hidden truncation"):
        build_tagged_concat_inputs(arrays)


def test_tagged_concat_retains_raw_hlt_particles_beyond_deployable_cap():
    arrays = _raw_arrays()
    for name, rows in list(arrays.items()):
        if name.startswith("scoutpfcand_"):
            arrays[name] = [np.resize(np.asarray(rows[0]), 214)]
        elif name.startswith("cpfcandlt_"):
            arrays[name] = [np.resize(np.asarray(rows[0]), 278)]
    arrays["n_scoutpfcands"] = np.asarray([214], np.int32)
    arrays["n_cpfcands"] = np.asarray([278], np.int32)
    view = build_tagged_concat_inputs(arrays)
    assert view.features.shape == (1, 21, 496)
    assert view.raw_lengths.tolist() == [493]
    assert view.mask[0, 0, :493].all()
    assert not view.mask[0, 0, 493:].any()
    assert np.all(view.content_source_codes[0, :279] == OFFLINE_CONTENT)
    assert np.all(view.content_source_codes[0, 279:493] == HLT_CONTENT)


def test_normalized_batch_keeps_source_codes_outside_physics_features():
    view = build_tagged_concat_inputs(_raw_arrays())
    batch = normalize_hlt_batch({
        "hlt": view, "labels": np.asarray([2]),
        "identity_digests": np.arange(32, dtype=np.uint8)[None],
    })
    assert batch.features.shape[1] == 21
    assert batch.content_source_codes is view.content_source_codes
    bad = view.content_source_codes.copy(); bad[0, 0] = 2
    from dataclasses import replace
    with pytest.raises(ValueError, match="source-code values"):
        normalize_hlt_batch({
            "hlt": replace(view, content_source_codes=bad),
            "labels": np.asarray([2]),
            "identity_digests": np.arange(32, dtype=np.uint8)[None],
        })


class _Trimmer:
    def __call__(self, features, vectors, mask, extra):
        return features, vectors, mask, extra


def test_tagged_model_embedding_receives_gradient_and_padding_is_masked(monkeypatch):
    import torch
    from torch import nn
    from hlt_classification.models import hcwdl_tagged_concat_transformer as module

    class Embed(nn.Module):
        def __init__(self):
            super().__init__(); self.projection = nn.Linear(21, 128)
        def forward(self, value): return self.projection(value.transpose(1, 2))
    class Pair(nn.Module):
        def forward(self, vectors, uu=None, mask=None):
            return torch.zeros((len(vectors), 8, vectors.shape[2], vectors.shape[2]), device=vectors.device)
    class Block(nn.Module):
        def forward(self, value, **_kwargs): return value
    class Weaver(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__(); self.trimmer=_Trimmer();self.embed=Embed();self.pair_embed=Pair();self.blocks=nn.ModuleList([Block() for _ in range(8)]);self.block_ids_with_attn_mask=();self.cls_token=nn.Parameter(torch.zeros(1,1,128));self.fc=nn.Linear(128,15)
        def _forward_aggregator(self, hidden, padding_mask):
            active=(~padding_mask)[...,None];return (hidden*active).sum(1)/active.sum(1).clamp_min(1)
    monkeypatch.setattr(module, "_weaver_class", lambda: Weaver)
    model = module.TaggedConcatParticleTransformer()
    features = torch.randn(2, 21, 5)
    vectors = torch.randn(2, 4, 5)
    mask = torch.tensor([[[1,1,1,0,0]], [[1,1,1,1,0]]], dtype=torch.bool)
    sources = torch.tensor([[0,0,1,-1,-1],[0,1,1,1,-1]], dtype=torch.int8)
    output = model(features, vectors, mask, sources)
    output.sum().backward()
    assert model.content_source_embedding.weight.grad is not None
    assert model.content_source_embedding.weight.grad.abs().sum() > 0
    changed = features.clone();changed[~mask.expand_as(features)] = 1.0e6
    with torch.no_grad():
        np.testing.assert_allclose(
            model(features, vectors, mask, sources).numpy(),
            model(changed, vectors, mask, sources).numpy(), rtol=0, atol=0,
        )


def test_tri60_engine_tagged_protocol_is_opt_in_and_persisted(tmp_path):
    import torch
    from hlt_classification.scouting.dataset import _take_batch
    from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
        Tri60TrainingRuntime, train_tri60_node,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_concat_runner import training_authority

    rows, tokens = 30, 4
    view = build_tagged_concat_inputs(_raw_arrays())
    # Repeat the real builder output so training exercises slicing/shuffling
    # of the extra metadata field, not a hand-constructed shortcut.
    from hlt_classification.scouting.dataset import _concat_particle_views
    view = _concat_particle_views([view for _ in range(rows)])
    identities = np.zeros((rows, 32), np.uint8)
    identities[:, :2] = np.asarray(
        [(index // 256, index % 256) for index in range(rows)], np.uint8,
    )

    class Cache:
        header={"rows":rows,"array_bytes":view.features.nbytes}
        def iterate_batches(self, *, epoch, sampler_seed, batch_size):
            del epoch,sampler_seed
            batch={"labels":np.arange(rows)%15,"identity_keys":np.asarray([f"r-{i}" for i in range(rows)]),"identity_digests":identities,"privileged":view}
            for start in range(0,rows,batch_size):
                yield _take_batch(batch,np.arange(start,min(rows,start+batch_size)))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.physics=torch.nn.Linear(21,15);self.source=torch.nn.Embedding(2,15)
        def forward(self,features,vectors,mask,codes):
            del vectors
            active=mask.float();pooled=(features*active).sum(-1)/active.sum(-1).clamp_min(1)
            source=(self.source(codes.clamp_min(0).long())*mask[:,0,:,None]).sum(1)/mask[:,0].sum(1,keepdim=True).clamp_min(1)
            return self.physics(pooled)+source
        def no_weight_decay(self):return set()

    report=train_tri60_node(
        node_id="CONCAT_TAGGED",train_cache=Cache(),validation_cache=Cache(),
        input_key="privileged",output_dir=tmp_path,
        parents={"source":"a"*64},campaign_spec_sha256="b"*64,
        recipe_sha256="c"*64,replicate_seed=1337,device="cpu",
        runtime=Tri60TrainingRuntime(passes=2,batch_size=30),
        execution_mode="synthetic_test",model_factory=Tiny,
        authority=training_authority(),
        model_input_protocol="tagged_offline_hlt_concat_v2",
    )
    assert report["model_input_protocol"] == "tagged_offline_hlt_concat_v2"
    checkpoint=torch.load(tmp_path/"selected_model.pt",map_location="cpu",weights_only=False)
    assert checkpoint["model_input_protocol"] == "tagged_offline_hlt_concat_v2"


def test_campaign_is_one_isolated_fit_with_required_gates(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_offline_hlt_concat_campaign as campaign
    from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import TRAINING_REPORT_CONTRACT as CE
    from hlt_classification.scouting.hcwdl_offline_hlt_concat_contracts import SOURCE_LOCK_CONTRACT, artifact
    from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_contracts import TRAINING_REPORT_CONTRACT as SP

    u000_path = tmp_path / "u000.json"
    u000 = with_content_hash({"contract":"TEST_U000/v1","schema_version":1,"validation":{}})
    write_immutable_json(u000_path, u000)
    source = artifact({
        "parents":{"foundation":"f"*64,"foundation_spec":"a"*64,"pure_offline_u000_report":u000["content_hash"]},
        "foundation_spec_path":str(tmp_path/"foundation.json"),
        "pure_offline_u000":{"report_path":str(u000_path)},
        "replicate_seed":1337,"role_counts":{"train":10,"validation":5},
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda _path: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    baseline=with_content_hash({"contract":CE,"schema_version":1,"node_id":"M0CE60","validation":{},"final_test_accessed":False})
    persistent=with_content_hash({"contract":SP,"schema_version":1,"node_id":"SP4P_U000","validation":{},"final_test_accessed":False})
    baseline_path=tmp_path/"baseline.json";persistent_path=tmp_path/"persistent.json"
    write_immutable_json(baseline_path,baseline);write_immutable_json(persistent_path,persistent)
    spec=campaign.create_campaign(foundation_spec=tmp_path/"foundation.json",m0ce60_report=baseline_path,persistent_anchor_report=persistent_path,campaign_root=tmp_path/"campaign",project_dir=tmp_path,source_commit="b"*40,authorize_live_submission=True,authorization_phrase=campaign.CREATION_PHRASE)
    assert spec["fresh_fit_count"] == 1
    assert [row["task_id"] for row in spec["tasks"]] == ["authenticate","capacity_audit","preflight","train_CONCAT_TAGGED","aggregate","campaign_complete"]
    task_ids = [row["task_id"] for row in spec["tasks"]]
    assert spec["existing_campaign_dependencies"] == []
    assert spec["ram_only_particle_views"] is True
    assert spec["capacity"] == 496
    assert spec["all_raw_hlt_particles_retained"] is True
    plan=__import__("json").loads((tmp_path/"campaign/command_plan.json").read_text())
    assert len(plan["commands"]) == 6
    gate_plan=__import__("json").loads((tmp_path/"campaign/gate_command_plan.json").read_text())
    science_plan=__import__("json").loads((tmp_path/"campaign/science_command_plan.json").read_text())
    assert [row["task_id"] for row in gate_plan["commands"]] == task_ids[:3]
    assert [row["task_id"] for row in science_plan["commands"]] == task_ids[3:]
    assert science_plan["satisfied_completed_tasks"] == ["preflight"]
    assert science_plan["commands"][0]["dependencies"] == []
    train=next(row for row in plan["commands"] if row["task_id"]=="train_CONCAT_TAGGED")
    assert "--gres=gpu:gh200:1" in train["command"]
    assert "--mem=384G" in train["command"]

    from hlt_classification.scouting.hcwdl_offline_hlt_concat_recovery import (
        build_monitor, create_recovery, validate_recovery,
    )
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
    jobs = {task_id: str(80000 + index) for index, task_id in enumerate(task_ids)}
    commands = {
        row["task_id"]: row["command"] for row in plan["commands"]
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = build_monitor(
        spec=spec, ledger=ledger,
        states_by_job_id={job_id: "FAILED" for job_id in jobs.values()},
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    recovery = create_recovery(
        subject_spec=tmp_path / "campaign/campaign_spec.json",
        subject_ledger=ledger_path, monitor_report=monitor_path,
        recovery_root=tmp_path / "recovery", project_dir=tmp_path,
        source_commit="c" * 40,
    )
    assert recovery["retry_tasks"] == task_ids
    assert recovery["restart_from_zero"] is True
    assert validate_recovery(recovery) == recovery["content_hash"]

    gate_jobs = {task_id: str(81000 + index) for index, task_id in enumerate(task_ids[:3])}
    gate_ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=gate_jobs,
        commands={row["task_id"]: row["command"] for row in gate_plan["commands"]},
        dry_run=False,
    )
    gate_ledger_path = tmp_path / "gate-ledger.json"
    write_immutable_json(gate_ledger_path, gate_ledger)
    gate_monitor = build_monitor(
        spec=spec, ledger=gate_ledger,
        states_by_job_id={job_id: "FAILED" for job_id in gate_jobs.values()},
    )
    gate_monitor_path = tmp_path / "gate-monitor.json"
    write_immutable_json(gate_monitor_path, gate_monitor)
    gate_recovery = create_recovery(
        subject_spec=tmp_path / "campaign/campaign_spec.json",
        subject_ledger=gate_ledger_path, monitor_report=gate_monitor_path,
        recovery_root=tmp_path / "gate-recovery", project_dir=tmp_path,
        source_commit="d" * 40,
    )
    assert gate_recovery["scope_tasks"] == task_ids[:3]
    assert validate_recovery(gate_recovery) == gate_recovery["content_hash"]
