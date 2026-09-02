from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json


def test_graph_freezes_all_requested_fits_and_fixed_teacher():
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_graph import (
        FIT_ORDER, GRAPH_SHA256, NODE_REGISTRY, ORACLE_NODES, STUDY_C_NODES,
        TEACHER_NODE, validate_graph,
    )

    assert len(FIT_ORDER) == 11
    assert len(ORACLE_NODES) == 8
    assert len(STUDY_C_NODES) == 3
    assert TEACHER_NODE == "ANCHORED_FUSION_OH"
    assert validate_graph() == GRAPH_SHA256
    assert not NODE_REGISTRY["ANCHORED_FUSION_OH"].deployable
    assert NODE_REGISTRY["FUSION_WITHDRAW_COS"].deployable


def test_alpha_schedules_have_exact_endpoint():
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_graph import (
        COSINE_ALPHA, STEP_ALPHA,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import (
        alpha_for_effective_pass,
    )

    assert alpha_for_effective_pass(COSINE_ALPHA, effective_pass=10) == 1
    assert alpha_for_effective_pass(COSINE_ALPHA, effective_pass=60) == 0
    assert 0 < alpha_for_effective_pass(COSINE_ALPHA, effective_pass=35) < 1
    assert alpha_for_effective_pass(STEP_ALPHA, effective_pass=60) == 1
    assert alpha_for_effective_pass(STEP_ALPHA, effective_pass=60.01) == 0


def test_withdrawal_loss_collapses_routes_at_zero():
    from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
        AnchoredFusionOutput, WithdrawalOutput,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import (
        withdrawal_loss,
    )

    logits = torch.randn(4, 15, requires_grad=True)
    route = AnchoredFusionOutput(logits, (), torch.ones(4, 3, dtype=torch.bool))
    teacher = torch.softmax(torch.randn(4, 15) / 2, dim=1)
    losses = withdrawal_loss(
        WithdrawalOutput(route, route, 0.0), torch.arange(4), teacher,
    )
    assert losses["logit_consistency"].item() == 0
    assert losses["representation_consistency"].item() == 0
    losses["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


class _Trimmer(nn.Module):
    def forward(self, features, vectors, mask, extra):
        return features, vectors, mask, extra


class _Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(21, 128, 1)

    def forward(self, features):
        return self.projection(features).transpose(1, 2)


class _Pair(nn.Module):
    def forward(self, vectors, uu=None, mask=None):
        del uu
        rows, _, tokens = vectors.shape
        return vectors.new_zeros(rows, 8, tokens, tokens)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, hidden, x_cls=None, padding_mask=None, attn_mask=None):
        del x_cls, attn_mask
        value = hidden + .01 * torch.tanh(self.linear(hidden))
        return value.masked_fill(padding_mask[..., None], 0)


class _FakeWeaver(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.trimmer = _Trimmer()
        self.embed = _Embed()
        self.pair_embed = _Pair()
        self.blocks = nn.ModuleList(
            _Block() for _ in range(int(config.get("num_layers", 8)))
        )
        self.block_ids_with_attn_mask = tuple(range(len(self.blocks)))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.fc = nn.Linear(128, int(config.get("num_classes", 15)))

    def _forward_aggregator(self, hidden, padding_mask):
        active = (~padding_mask)[..., None]
        return (hidden * active).sum(1) / active.sum(1).clamp_min(1)

    def forward(self, features, v, mask):
        hidden = self.embed(features)
        pair = self.pair_embed(v, mask=mask)
        for block in self.blocks:
            hidden = block(hidden, padding_mask=~mask[:, 0], attn_mask=pair)
        return self.fc(self._forward_aggregator(hidden, ~mask[:, 0]))


@pytest.fixture
def fake_weaver(monkeypatch):
    from hlt_classification.models import hcwdl_offline_hlt_fusion_transformer as fusion
    from hlt_classification.models import scouting_particle_transformer as scouting

    monkeypatch.setattr(fusion, "_weaver_class", lambda: _FakeWeaver)
    monkeypatch.setattr(scouting, "_weaver_class", lambda: _FakeWeaver)


def _tagged_batch(rows=3, offline=4, hlt=5):
    tokens = offline + hlt
    features = torch.randn(rows, 21, tokens)
    vectors = torch.randn(rows, 4, tokens)
    mask = torch.ones(rows, 1, tokens, dtype=torch.bool)
    source = torch.tensor([[0] * offline + [1] * hlt] * rows, dtype=torch.int8)
    return features, vectors, mask, source


def test_symmetric_controls_are_parameter_matched(fake_weaver):
    from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
        SymmetricFusionParticleTransformer,
    )

    states = []
    for arm in ("OO", "HH", "OH"):
        torch.manual_seed(17)
        model = SymmetricFusionParticleTransformer(arm)
        states.append(model.state_dict())
        assert model(*_tagged_batch()).shape == (3, 15)
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])
        assert torch.equal(states[0][name], states[2][name])


def test_anchored_zero_is_exact_extracted_hlt_and_skips_context(fake_weaver):
    from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
        AnchoredFusionParticleTransformer, _content_view,
    )

    model = AnchoredFusionParticleTransformer("O").eval()
    model.context_mod = nn.Module()  # would fail if the zero route touched it
    batch = _tagged_batch()
    with torch.inference_mode():
        zero = model.forward_zero(*batch).logits
        offline = batch[3] == 0
        changed = model.forward_zero(
            batch[0].masked_fill(offline[:, None], float("nan")),
            batch[1].masked_fill(offline[:, None], float("nan")),
            batch[2], batch[3],
        ).logits
        hlt = _content_view(*batch, code=1, capacity=200)
        extracted = model.extract_hlt().eval()(*hlt)
    assert torch.allclose(zero, extracted, rtol=1e-6, atol=1e-7)
    assert torch.equal(zero, changed)


def test_anchored_privileged_and_withdrawal_backward(fake_weaver):
    from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
        AnchoredFusionParticleTransformer,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import withdrawal_loss

    model = AnchoredFusionParticleTransformer("O").train()
    output = model.forward_withdrawal(*_tagged_batch(), alpha=.5)
    teacher = torch.softmax(torch.randn(3, 15) / 2, dim=1)
    losses = withdrawal_loss(output, torch.arange(3), teacher)
    losses["total"].backward()
    gradients = [
        injection.residual_projection.weight.grad
        for injection in model.injections
    ]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert output.zero.logits.shape == output.privileged.logits.shape == (3, 15)


def test_tri60_engine_runs_complete_withdrawal_protocol(tmp_path: Path):
    """Exercise the production trainer dispatch, selection route, and report."""

    from hlt_classification.scouting.dataset import _take_batch
    from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
        Tri60TrainingRuntime, train_tri60_node,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_graph import (
        COSINE_ALPHA,
    )
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_runner import (
        training_authority,
    )
    from hlt_classification.scouting.hcwdl_representation_data import (
        HCWDLTaggedParticleInputs,
    )

    rows, offline, hlt = 30, 2, 3
    tokens = offline + hlt
    identities = np.zeros((rows, 32), np.uint8)
    identities[:, :2] = np.asarray(
        [(index // 256, index % 256) for index in range(rows)], np.uint8,
    )
    features = np.random.default_rng(7).normal(
        size=(rows, 21, tokens),
    ).astype(np.float32)
    vectors = np.ones((rows, 4, tokens), np.float32)
    mask = np.ones((rows, 1, tokens), np.bool_)
    visible = np.tile(np.arange(tokens, dtype=np.int64), (rows, 1))
    source = np.tile(
        np.asarray([0] * offline + [1] * hlt, np.int8), (rows, 1),
    )
    view = HCWDLTaggedParticleInputs(
        features, vectors, mask, np.full(rows, tokens, np.int32), visible,
        np.zeros((rows, tokens), np.int8),
        np.zeros((rows, tokens), np.int8), source,
    )
    batch = {
        "labels": np.arange(rows, dtype=np.int64) % 15,
        "identity_keys": np.asarray([f"fusion-row-{i}" for i in range(rows)]),
        "identity_digests": identities,
        "privileged": view,
    }

    class Cache:
        header = {"rows": rows, "array_bytes": features.nbytes}

        def iterate_batches(self, *, epoch, sampler_seed, batch_size):
            del epoch, sampler_seed
            for start in range(0, rows, batch_size):
                yield _take_batch(
                    batch, np.arange(start, min(rows, start + batch_size)),
                )

    class Targets:
        temperature = 2.0

        def join(self, requested):
            assert requested.shape[1] == 32
            values = np.full((len(requested), 15), 1 / 15, np.float32)
            return values

    class TinyWithdrawal(nn.Module):
        def __init__(self):
            super().__init__()
            self.physics = nn.Linear(21, 15)
            self.context = nn.Linear(21, 15)
            self.hidden = nn.Linear(21, 8)

        @staticmethod
        def _pool(features, mask):
            active = mask.float()
            return (
                (features * active).sum(-1)
                / active.sum(-1).clamp_min(1)
            )

        def _routes(self, features, mask, source, alpha):
            hlt_mask = mask[:, 0] & (source == 1)
            off_mask = mask[:, 0] & (source == 0)
            hlt_pool = self._pool(features, hlt_mask[:, None])
            off_pool = self._pool(features, off_mask[:, None])
            zero_logits = self.physics(hlt_pool)
            privileged_logits = zero_logits + float(alpha) * self.context(off_pool)
            zero_state = self.hidden(hlt_pool)[:, None]
            privileged_state = zero_state + float(alpha) * self.hidden(off_pool)[:, None]
            from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import (
                AnchoredFusionOutput, WithdrawalOutput,
            )
            active = torch.ones(
                len(features), 1, dtype=torch.bool, device=features.device,
            )
            zero = AnchoredFusionOutput(zero_logits, (zero_state,) * 4, active)
            privileged = AnchoredFusionOutput(
                privileged_logits, (privileged_state,) * 4, active,
            )
            return WithdrawalOutput(zero, privileged, float(alpha))

        def forward_zero(self, features, vectors, mask, source):
            del vectors
            return self._routes(features, mask, source, 0).zero

        def forward_withdrawal(self, features, vectors, mask, source, *, alpha):
            del vectors
            return self._routes(features, mask, source, alpha)

        def no_weight_decay(self):
            return set()

    report = train_tri60_node(
        node_id="FUSION_WITHDRAW_COS",
        train_cache=Cache(), validation_cache=Cache(),
        input_key="privileged", probability_targets=Targets(),
        output_dir=tmp_path, parents={"source": "a" * 64},
        campaign_spec_sha256="b" * 64, recipe_sha256="c" * 64,
        replicate_seed=1337, device="cpu",
        runtime=Tri60TrainingRuntime(passes=100, batch_size=30),
        execution_mode="synthetic_test", model_factory=TinyWithdrawal,
        authority=training_authority("FUSION_WITHDRAW_COS"),
        model_input_protocol="anchored_withdrawal_v1",
        withdrawal_schedule=COSINE_ALPHA,
        learning_rate_schedule={
            "kind": "warmup_hold_cosine_floor_tail_v1",
            "warmup_passes": 3, "hold_through_pass": 45,
            "decay_through_pass": 60, "minimum_lr_fraction": .05,
        },
        early_stopping={
            "kind": "macro_auc_patience_v1", "minimum_passes": 60,
            "patience_passes": 15, "minimum_auc_delta": 1e-5,
        },
        initialization_lineage={
            "source_report": "d" * 64,
            "source_checkpoint": "e" * 64,
        },
    )
    assert report["model_input_protocol"] == "anchored_withdrawal_v1"
    assert report["withdrawal_schedule"] == COSINE_ALPHA
    assert report["checkpoint_selection_route"] == "alpha_zero_macro_auc_v1"
    assert report["validation_route"] == "exact_alpha_zero_v1"
    assert report["offline_context_skipped_during_validation"] is True
    assert report["passes"] >= 60
    assert all(
        set(row["mean_losses"]) == {
            "alpha", "ce_privileged", "ce_zero", "kd_privileged",
            "kd_zero", "logit_consistency", "representation_consistency",
            "total",
        }
        for row in report["training_history"]
    )
    assert report["training_history"][0]["mean_losses"]["alpha"] == 1
    assert report["training_history"][60]["mean_losses"]["alpha"] == 0
    assert not list(tmp_path.rglob("*resume*"))


def test_campaign_has_staged_exact_dag(tmp_path: Path, monkeypatch):
    from hlt_classification.scouting import hcwdl_offline_hlt_fusion_campaign as campaign

    source = with_content_hash({
        "contract": "TEST_SOURCE/v1", "schema_version": 1,
        "parents": {"foundation_lock": "1" * 64, "foundation_spec": "2" * 64},
        "u000": {"report_sha256": "3" * 64, "report_path": str(tmp_path / "u000.json")},
        "replicate_seed": 1337,
        "role_counts": {"train": 2_777_855, "validation": 957_541, "final_test": 899_779},
        "final_test_accessed": False,
    })
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    baseline = with_content_hash({
        "contract": "HCWDL_MHPE_TRI60_CE60_CONTROL_TRAINING_REPORT/v1",
        "schema_version": 1, "node_id": "M0CE60",
        "validation": {"accuracy": .8, "macro_ovr_auc": .94},
        "final_test_accessed": False,
    })
    baseline_path = tmp_path / "m0.json"
    write_immutable_json(baseline_path, baseline)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        foundation_spec=tmp_path / "foundation.json",
        m0ce60_report=baseline_path, campaign_root=root,
        project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 19
    assert len(json.loads((root / "gate_command_plan.json").read_text())["commands"]) == 3
    science = json.loads((root / "science_command_plan.json").read_text())
    assert len(science["commands"]) == 16
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    assert tasks["train_FUSION_WITHDRAW_COS"]["dependencies"] == [
        "reduce_ANCHORED_FUSION_OH_T2"
    ]
    assert tasks["extract_FUSION_WITHDRAW_COS"]["dependencies"] == [
        "train_FUSION_WITHDRAW_COS"
    ]
    assert spec["existing_campaign_dependencies"] == []
    assert spec["run_study_c_regardless_of_oracle_metrics"] is True


def test_teacher_probability_bank_roundtrip(tmp_path: Path):
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_probability import (
        FusionProbabilityTargets, load_role, publish_role,
    )

    identities = np.zeros((5, 32), np.uint8)
    identities[:, 0] = np.arange(5)
    manifest = publish_role(
        tmp_path, role="train", identity_digests=identities,
        logits=np.arange(75, dtype=np.float32).reshape(5, 15),
        teacher_report_sha256="1" * 64,
        teacher_checkpoint_sha256="2" * 64,
        campaign_spec_sha256="3" * 64, producer_commit="a" * 40,
    )
    loaded, keys, probabilities = load_role(tmp_path / "train_manifest.json", role="train")
    assert loaded == manifest
    assert np.array_equal(keys, identities)
    assert np.allclose(probabilities.sum(1), 1)
    targets = FusionProbabilityTargets.load(tmp_path / "train_manifest.json")
    assert np.array_equal(targets.join(identities[::-1]), probabilities[::-1])


def test_monitor_and_restart_zero_recovery_cover_full_scope(
    tmp_path: Path, monkeypatch,
):
    from hlt_classification.scouting import hcwdl_offline_hlt_fusion_recovery as recovery
    from hlt_classification.scouting.hcwdl_offline_hlt_fusion_campaign import tasks
    from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger

    root = tmp_path / "subject"
    root.mkdir()
    subject = with_content_hash({
        "contract": "TEST/v1", "schema_version": 1,
        "campaign_root": str(root), "tasks": tasks(),
        "resources": {
            name: {"cpus": 1, "memory": "1G", "walltime": "01:00:00", "gpu": None}
            for name in {row["resource"] for row in tasks()}
        },
        "final_test_accessed": False,
    })
    subject_path = root / "campaign_spec.json"
    write_immutable_json(subject_path, subject)
    monkeypatch.setattr(recovery, "validate_campaign", lambda value: value["content_hash"])
    commands = {row["task_id"]: ["sbatch", row["task_id"]] for row in tasks()}
    jobs = {name: str(40_000 + index) for index, name in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=subject["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = recovery.build_monitor(
        spec=subject, ledger=ledger,
        states_by_job_id={job: "FAILED" for job in jobs.values()},
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    value = recovery.create_recovery(
        subject_spec=subject_path, subject_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=tmp_path / "recovery",
        project_dir=tmp_path, source_commit="b" * 40,
    )
    assert recovery.validate_recovery(value) == value["content_hash"]
    assert value["retry_tasks"] == [row["task_id"] for row in tasks()]
    assert value["restart_from_zero"] is True


def test_worker_is_source_pinned_and_storage_safe():
    text = Path("sbatch/run_hcwdl_offline_hlt_fusion_task.sh").read_text()
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
    assert "PYTHONNOUSERSITE=1" in text
    assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
    assert 'exec python -s "${PROJECT_DIR}/scripts/' in text
