from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger


def test_attention_recipe_and_full_four_spine_graph_are_frozen():
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        DEFAULT_ATTENTION_RECIPE, attention_stage, normalize_attention_recipe,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_attention_graph import (
        BRANCH_NODES, BRANCH_ORDER, DOWNSTREAM_FIT_ORDER, FIT_ORDER,
        REDUCER_ORDER, RELATIONAL_CARRIERS, validate_graph,
    )

    assert normalize_attention_recipe(
        DEFAULT_ATTENTION_RECIPE.payload()
    ) == DEFAULT_ATTENTION_RECIPE
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 0) == "stage0"
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 59) == "stage0"
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 60) == "stage_a"
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 74) == "stage_a"
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 75) == "stage_b"
    assert attention_stage(DEFAULT_ATTENTION_RECIPE, 99) == "stage_b"
    assert len(FIT_ORDER) == 30
    assert len(DOWNSTREAM_FIT_ORDER) == 29
    assert len(REDUCER_ORDER) == 26
    assert tuple(len(BRANCH_NODES[name]) for name in BRANCH_ORDER) == (1, 5, 8, 15)
    assert set(RELATIONAL_CARRIERS) == set(DOWNSTREAM_FIT_ORDER)
    assert validate_graph()


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.mlp = torch.nn.Linear(8, 8)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = torch.nn.Module()
        self.mod.pair_embed = torch.nn.Linear(4, 4)
        self.mod.blocks = torch.nn.ModuleList([_Block() for _ in range(8)])
        self.mod.embed = torch.nn.Linear(8, 8)
        self.mod.fc = torch.nn.Linear(8, 3)


def test_parameter_registry_is_structural_and_stage_a_is_exact():
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        compile_attention_parameter_registry, configure_attention_stage,
        validate_attention_parameter_registry,
    )

    model = _Model()
    registry = compile_attention_parameter_registry(model)
    assert validate_attention_parameter_registry(registry)
    assert registry["substring_parameter_selection"] is False
    allowed = {row["name"] for row in registry["parameter_rows"]}
    configure_attention_stage(model, registry, "stage_a")
    actual = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert actual == allowed
    assert all(".mlp." not in name for name in actual)
    assert all(".embed." not in name and ".fc." not in name for name in actual)
    configure_attention_stage(model, registry, "stage_b")
    assert all(parameter.requires_grad for parameter in model.parameters())


def _surfaces(deltas, ids, families, mask):
    return SimpleNamespace(
        block_residual_deltas=tuple(deltas.clone() for _ in range(8)),
        visible_indices=ids,
        family_codes=families,
        particle_mask=mask,
    )


def test_relational_loss_aligns_intersection_and_ignores_channel_rotation():
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        support_aligned_block_delta_gram_loss,
    )

    teacher_delta = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    student_delta = teacher_delta[:, [2, 0]]
    teacher = _surfaces(
        teacher_delta, torch.tensor([[4, 7, 9]]),
        torch.tensor([[1, 1, 2]], dtype=torch.int8),
        torch.tensor([[True, True, True]]),
    )
    student = _surfaces(
        student_delta, torch.tensor([[9, 4]]),
        torch.tensor([[2, 1]], dtype=torch.int8),
        torch.tensor([[True, True]]),
    )
    loss, diagnostics = support_aligned_block_delta_gram_loss(student, teacher)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["common_tokens"].item() == 2
    assert diagnostics["common_ordered_pairs"].item() == 2

    # A common orthogonal channel rotation preserves token Gram geometry.
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotated = _surfaces(
        student_delta @ rotation, torch.tensor([[9, 4]]),
        torch.tensor([[2, 1]], dtype=torch.int8),
        torch.tensor([[True, True]]),
    )
    rotated_loss, _ = support_aligned_block_delta_gram_loss(rotated, teacher)
    assert rotated_loss.item() == pytest.approx(0.0, abs=1e-7)


def test_relational_loss_excludes_padding_and_rejects_duplicate_identity():
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        support_aligned_block_delta_gram_loss,
    )

    teacher_delta = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [99.0, -99.0]]])
    student_delta = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-500.0, 500.0]]])
    mask = torch.tensor([[True, True, False]])
    ids = torch.tensor([[4, 7, -1]])
    family = torch.tensor([[1, 1, -1]], dtype=torch.int8)
    loss, diagnostics = support_aligned_block_delta_gram_loss(
        _surfaces(student_delta, ids, family, mask),
        _surfaces(teacher_delta, ids, family, mask),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["common_tokens"].item() == 2

    duplicate_ids = torch.tensor([[4, 4, -1]])
    duplicate = _surfaces(student_delta, duplicate_ids, family, mask)
    with pytest.raises(ValueError, match="one-to-one"):
        support_aligned_block_delta_gram_loss(
            duplicate, _surfaces(teacher_delta, ids, family, mask),
        )

    disjoint = _surfaces(
        student_delta, torch.tensor([[14, 17, -1]]), family, mask,
    )
    with pytest.raises(ValueError, match="no common token pairs"):
        support_aligned_block_delta_gram_loss(
            disjoint, _surfaces(teacher_delta, ids, family, mask),
        )


def test_attention_teacher_freeze_guard():
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        assert_frozen_attention_teacher, freeze_attention_teacher,
    )

    teacher = freeze_attention_teacher(_Model())
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert_frozen_attention_teacher(teacher)
    next(teacher.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="not frozen"):
        assert_frozen_attention_teacher(teacher)


class _AttentionCache:
    def __init__(self, rows: int = 30, tokens: int = 3):
        from hlt_classification.scouting.hcwdl_homotopy import (
            HCWDLParticleInputs,
        )

        labels = np.arange(rows, dtype=np.int64) % 15
        identities = np.zeros((rows, 32), dtype=np.uint8)
        identities[:, :2] = np.asarray(
            [(index // 256, index % 256) for index in range(rows)],
            dtype=np.uint8,
        )
        features = np.zeros((rows, 21, tokens), dtype=np.float32)
        features[:, 0] = np.arange(rows, dtype=np.float32)[:, None] / rows
        vectors = np.ones((rows, 4, tokens), dtype=np.float32)
        mask = np.ones((rows, 1, tokens), dtype=np.bool_)
        visible = np.tile(np.arange(tokens, dtype=np.int64), (rows, 1))
        family = np.zeros((rows, tokens), dtype=np.int8)
        reasons = np.zeros((rows, tokens), dtype=np.int8)
        view = HCWDLParticleInputs(
            features, vectors, mask, np.full(rows, tokens, np.int32),
            visible, family, reasons,
        )
        self._batch = {
            "labels": labels,
            "identity_keys": np.asarray([f"row-{index}" for index in range(rows)]),
            "identity_digests": identities,
            "hlt": view,
        }
        self.identity_digests = identities
        self.header = {
            "rows": rows,
            "array_bytes": sum(
                value.nbytes
                for value in (labels, identities, features, vectors, mask)
            ),
        }

    def iterate_batches(self, *, epoch, sampler_seed, batch_size):
        del epoch, sampler_seed
        from hlt_classification.scouting.dataset import _take_batch

        for start in range(0, self.header["rows"], batch_size):
            indexes = np.arange(
                start, min(start + batch_size, self.header["rows"]),
            )
            yield _take_batch(self._batch, indexes)


def _tiny_attention_model():
    from hlt_classification.models.scouting_particle_transformer import (
        HCWDLAttentionReoptimizationSurfaces,
    )

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Linear(8, 8)
            self.mlp = torch.nn.Linear(8, 8)

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = torch.nn.Module()
            self.mod.embed = torch.nn.Linear(21, 8)
            self.mod.pair_embed = torch.nn.Linear(4, 8)
            self.mod.blocks = torch.nn.ModuleList([Block() for _ in range(8)])
            self.classifier = torch.nn.Linear(8, 15)

        def _surfaces(self, features, vectors, mask, visible, family):
            hidden = (
                self.mod.embed(features.transpose(1, 2))
                + self.mod.pair_embed(vectors.transpose(1, 2))
            )
            deltas = []
            for block in self.mod.blocks:
                delta = torch.tanh(block.attn(hidden))
                hidden = hidden + delta + 0.01 * block.mlp(hidden)
                deltas.append(delta)
            particle_mask = mask.squeeze(1)
            weights = particle_mask[..., None].float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            return HCWDLAttentionReoptimizationSurfaces(
                logits=self.classifier(pooled),
                block_residual_deltas=tuple(deltas),
                particle_mask=particle_mask,
                visible_indices=visible,
                family_codes=family,
            )

        def forward(self, features, vectors, mask):
            rows, _, tokens = features.shape
            visible = torch.arange(tokens, device=features.device)[None].expand(
                rows, -1,
            )
            family = torch.zeros_like(visible, dtype=torch.int8)
            return self._surfaces(
                features, vectors, mask, visible, family,
            ).logits

        def forward_attention_reoptimization_surfaces(
            self, features, vectors, mask, visible_indices, family_codes,
        ):
            return self._surfaces(
                features, vectors, mask, visible_indices, family_codes,
            )

        def no_weight_decay(self):
            return set()

    return TinyAttention()


def test_real_trainer_executes_exact_stage_transitions_and_ram_teacher(tmp_path):
    from hlt_classification.scouting.hcwdl_attention_reoptimization import (
        DEFAULT_ATTENTION_RECIPE, compile_attention_parameter_registry,
        freeze_attention_teacher,
    )
    from hlt_classification.scouting.hcwdl_mhpe_tri60_probability import (
        Tri60ProbabilityTargets,
    )
    from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
        Tri60TrainingRuntime, train_tri60_node,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_attention_graph import (
        BRANCH_NODES, LR_SCHEDULE,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_attention_runner import (
        training_authority,
    )

    cache = _AttentionCache()
    probabilities = np.full((cache.header["rows"], 15), 1 / 15, np.float32)
    targets = Tri60ProbabilityTargets(
        identities=cache.identity_digests,
        probabilities=probabilities,
        manifest=MappingProxyType({"temperature": 2.0}),
        _lookup={
            bytes(row): index
            for index, row in enumerate(cache.identity_digests)
        },
    )
    registry = compile_attention_parameter_registry(_tiny_attention_model())
    node_id = BRANCH_NODES["DIRECT"][0]
    report = train_tri60_node(
        node_id=node_id,
        train_cache=cache,
        validation_cache=cache,
        input_key="hlt",
        probability_targets=targets,
        output_dir=tmp_path,
        parents={"foundation": "a" * 64},
        campaign_spec_sha256="b" * 64,
        recipe_sha256="c" * 64,
        execution_source_commit="d" * 40,
        replicate_seed=1337,
        device="cpu",
        runtime=Tri60TrainingRuntime(passes=100, batch_size=256),
        execution_mode="synthetic_test",
        model_factory=_tiny_attention_model,
        authority=training_authority(node_id),
        learning_rate_schedule=dict(LR_SCHEDULE),
        attention_reoptimization=DEFAULT_ATTENTION_RECIPE.payload(),
        attention_parameter_registry=registry,
        relational_teacher_model=freeze_attention_teacher(
            _tiny_attention_model(),
        ),
        relational_train_cache=cache,
        relational_input_key="hlt",
    )
    assert [row["stage"] for row in report["attention_stage_history"]] == [
        "stage0", "stage_a", "stage_b",
    ]
    assert [row["attention_stage"] for row in report["validation_history"]] == (
        ["stage0"] * 60 + ["stage_a"] * 15 + ["stage_b"] * 25
    )
    assert report["dense_attention_target_durable_bytes"] == 0
    assert report["relational_target_generation"] == (
        "same_job_per_batch_eval_no_grad_v1"
    )
    assert report["complete"] is True
    assert not list(tmp_path.rglob("*attention_target*"))
    assert not list(tmp_path.rglob("*block_delta*"))
    assert not list(tmp_path.rglob("*resume*"))


def test_campaign_task_shape_and_resources():
    from hlt_classification.scouting.hcwdl_tri100_spine4_attention_campaign import (
        RESOURCES, campaign_tasks,
    )

    tasks = campaign_tasks()
    assert len(tasks) == 61
    assert len({row["task_id"] for row in tasks}) == 61
    assert sum(row["kind"] == "train" for row in tasks) == 30
    assert sum(row["kind"] == "reducer" for row in tasks) == 26
    assert RESOURCES["gpu_fit"].memory == "500G"
    assert RESOURCES["gpu_fit"].walltime == "5-00:00:00"
    assert tasks[-2]["task_id"] == "aggregate"
    assert tasks[-1]["task_id"] == "campaign_complete"


def test_cache_roles_are_explicitly_validated(monkeypatch):
    from hlt_classification.scouting import hcwdl_unified_balanced_runner as runner

    with pytest.raises(ValueError, match="role registry"):
        runner._cache_student_views(
            foundation_spec={}, split={}, selections={}, assignments={},
            balanced={}, behavior="hlt", coordinate=object(), batch_size=1,
            sampler_seed=1, repair_seed=2, memory_gib=1.0, roles=(),
        )


def _fake_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hlt_classification.scouting import (
        hcwdl_tri100_spine4_attention_campaign as campaign,
    )
    from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import (
        TRAINING_REPORT_CONTRACT as CE_CONTRACT,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_contracts import (
        SOURCE_LOCK_CONTRACT, artifact as persistent_artifact,
    )
    from hlt_classification.scouting.hcwdl_homotopy import (
        PERSISTENT_HLT_SUPPORT_POLICY,
    )

    source = persistent_artifact({
        "parents": {
            "source_campaign": "1" * 64,
            "foundation_lock": "2" * 64,
            "foundation_spec": "3" * 64,
            "assignment_lock": "4" * 64,
            "matcher_spec": "5" * 64,
        },
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "replicate_seed": 1337,
        "role_counts": {
            "train": 2_777_855,
            "validation": 957_541,
            "final_test": 899_779,
        },
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(
        campaign, "validate_source_lock", lambda value: value["content_hash"],
    )

    persistent_root = tmp_path / "persistent"
    persistent_root.mkdir()
    persistent = with_content_hash({
        "campaign_root": str(persistent_root),
        "parents": {"foundation": "2" * 64},
        "support_policy": PERSISTENT_HLT_SUPPORT_POLICY,
        "role_counts": dict(source["role_counts"]),
        "replicate_seed": 1337,
        "final_test_accessed": False,
    })
    persistent_path = tmp_path / "persistent.json"
    write_immutable_json(persistent_path, persistent)
    monkeypatch.setattr(
        campaign, "validate_persistent_campaign",
        lambda value: value["content_hash"],
    )

    baseline = with_content_hash({
        "contract": CE_CONTRACT,
        "schema_version": 1,
        "node_id": "M0CE60",
        "validation": {"accuracy": .8, "macro_ovr_auc": .94},
        "final_test_accessed": False,
    })
    baseline_path = tmp_path / "m0.json"
    write_immutable_json(baseline_path, baseline)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        foundation_spec=tmp_path / "foundation.json",
        persistent_campaign_spec=persistent_path,
        m0ce60_report=baseline_path,
        campaign_root=root,
        project_dir=tmp_path,
        source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    return campaign, spec, root


def test_campaign_publishes_an_isolated_exact_61_task_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting.hcwdl_tri100_spine4_attention_graph import (
        ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER, SOURCE_DISTRIBUTION,
    )

    campaign, spec, root = _fake_campaign(tmp_path, monkeypatch)
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 61
    assert spec["fresh_fit_count"] == 30
    assert spec["reducer_count"] == 26
    assert spec["source_fit_reuse_count"] == 0
    assert spec["persistent_campaign_completion_required"] is False
    assert spec["existing_campaign_dependencies"] == []
    assert spec["existing_campaign_outputs_mutated"] is False
    assert spec["existing_campaign_jobs_cancelled_held_or_reprioritized"] is False
    assert spec["dense_attention_target_artifacts"] is False
    assert spec["attention_targets"] == "same_job_batch_local_ram_or_device_only_v1"
    assert spec["rolling_resume"] is False
    assert spec["ordinary_final_test_capability"] is False

    tasks = {row["task_id"]: row for row in spec["tasks"]}
    assert tasks["support_audit"]["dependencies"] == ["authenticate"]
    assert tasks["preflight"]["dependencies"] == ["support_audit"]
    anchor_reducer = f"reduce_{SOURCE_DISTRIBUTION}"
    assert tasks[f"train_{ANCHOR_NODE_ID}"]["dependencies"] == ["preflight"]
    assert tasks[anchor_reducer]["dependencies"] == [f"train_{ANCHOR_NODE_ID}"]
    for branch in BRANCH_ORDER:
        first = BRANCH_NODES[branch][0]
        assert tasks[f"train_{first}"]["dependencies"] == [anchor_reducer]

    plan = load_json(root / "command_plan.json")
    gate = load_json(root / "gate_command_plan.json")
    science = load_json(root / "science_command_plan.json")
    assert len(plan["commands"]) == 61
    assert [row["task_id"] for row in gate["commands"]] == [
        "authenticate", "support_audit", "preflight",
    ]
    assert len(science["commands"]) == 58
    assert science["commands"][0]["task_id"] == f"train_{ANCHOR_NODE_ID}"
    assert science["commands"][0]["dependencies"] == []
    assert science["commands"][0]["completed_gate_dependencies"] == ["preflight"]
    assert not any(
        item.startswith("--dependency=")
        for item in science["commands"][0]["command"]
    )
    assert all(not row["external_dependencies"] for row in plan["commands"])
    fits = [
        row["command"] for row in plan["commands"]
        if row["task_id"].startswith("train_")
    ]
    assert len(fits) == 30
    assert all("--cpus-per-task=72" in row for row in fits)
    assert all("--mem=500G" in row for row in fits)
    assert all("--time=5-00:00:00" in row for row in fits)
    assert all("--gres=gpu:gh200:1" in row for row in fits)


def test_restart_zero_recovery_covers_the_exact_failed_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    campaign, spec, root = _fake_campaign(tmp_path, monkeypatch)
    from hlt_classification.scouting import (
        hcwdl_tri100_spine4_attention_recovery as recovery,
    )

    plan = load_json(root / "command_plan.json")
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    jobs = {name: str(20_000 + index) for index, name in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs=jobs,
        commands=commands,
        dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = recovery.build_monitor(
        spec=spec,
        ledger=ledger,
        states_by_job_id={job: "FAILED" for job in jobs.values()},
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    value = recovery.create_recovery(
        subject_spec=root / "campaign_spec.json",
        subject_ledger=ledger_path,
        monitor_report=monitor_path,
        recovery_root=tmp_path / "recovery",
        project_dir=tmp_path,
        source_commit="b" * 40,
    )
    assert recovery.validate_recovery(value) == value["content_hash"]
    assert value["restart_from_zero"] is True
    assert value["rolling_resume"] is False
    assert len(value["retry_tasks"]) == 61
    assert len(load_json(tmp_path / "recovery/command_plan.json")["commands"]) == 61


def test_gate_and_science_dry_runs_materialize_without_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting.hcwdl_exact_dag_submission import (
        submit_exact_dag,
    )

    _, spec, root = _fake_campaign(tmp_path, monkeypatch)
    for phase, expected in (("gate", 3), ("science", 58)):
        plan = load_json(root / f"{phase}_command_plan.json")
        destination = root / f"dry_run_{phase}_submission_ledger.json"
        ledger = submit_exact_dag(
            identity=spec["content_hash"],
            plan=plan,
            output=destination,
            canonical_dry_run=destination,
            execute=False,
        )
        assert ledger["dry_run"] is True
        assert len(ledger["jobs"]) == expected
        assert load_json(destination) == ledger


def test_attention_workers_are_absolute_source_pinned_shells():
    for path in (
        Path("sbatch/run_hcwdl_tri100_spine4_attention_task.sh"),
        Path("sbatch/run_hcwdl_tri100_spine4_attention_recovery_task.sh"),
    ):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "PYTHONNOUSERSITE=1" in text
        assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert 'exec python -s "${PROJECT_DIR}/scripts/' in text
