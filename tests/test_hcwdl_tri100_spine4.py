from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_data import HCWDLParticleInputs
from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime, train_tri60_node, tri60_early_stopping, tri60_learning_rate,
    tri60_learning_rate_schedule,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_graph import (
    BRANCH_NODES, BRANCH_ORDER, BRANCH_PATHS, EARLY_STOPPING, ENDPOINT_NODES,
    EXECUTION, FIT_ORDER, LR_SCHEDULE, NODE_REGISTRY, PROBABILITY_COMPONENTS,
    REDUCER_ORDER, distribution_consumers, recipe_payload, validate_graph,
)


SHA = "a" * 64


def test_graph_is_exact_four_single_spines_with_matched_endpoint_seeds():
    assert validate_graph()
    assert BRANCH_ORDER == ("DIRECT", "COARSE", "DENSE", "ULTRADENSE")
    assert tuple(len(BRANCH_NODES[name]) for name in BRANCH_ORDER) == (1, 5, 8, 15)
    assert len(FIT_ORDER) == 29
    assert len(REDUCER_ORDER) == 25
    assert len(PROBABILITY_COMPONENTS) == 25
    assert all(len(value) == 1 for value in PROBABILITY_COMPONENTS.values())
    assert BRANCH_PATHS["COARSE"] == (
        "U050", "U100", "D066", "D033", "D000",
    )
    assert BRANCH_PATHS["DENSE"] == (
        "U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000",
    )
    assert len({NODE_REGISTRY[name].seed_alias for name in ENDPOINT_NODES}) == 1
    u100 = [
        node.seed_alias for node in NODE_REGISTRY.values()
        if node.coordinate_name == "U100"
    ]
    assert len(u100) == 3 and len(set(u100)) == 1
    for branch in BRANCH_ORDER:
        previous = None
        for node_id in BRANCH_NODES[branch]:
            node = NODE_REGISTRY[node_id]
            assert node.parent_node_id == previous
            assert node.initialization == "fresh"
            assert node.ce_weight == .25 and node.kd_weight == .75
            assert node.temperature == 2.0
            assert node.auxiliary == "none"
            if node.output_distribution_id is not None:
                assert PROBABILITY_COMPONENTS[node.output_distribution_id] == (node_id,)
                assert len(distribution_consumers(node.output_distribution_id)) == 1
            previous = node_id


def test_recipe_and_floor_tail_learning_rate_are_exact():
    recipe = recipe_payload()
    assert recipe["training"]["learning_rate_schedule"] == dict(LR_SCHEDULE)
    assert recipe["training"]["early_stopping"] == dict(EARLY_STOPPING)
    assert recipe["loss"] == {
        "kind": "constant_ce_kd_v1", "ce_weight": .25,
        "kd_weight": .75, "temperature": 2.0,
    }
    runtime = Tri60TrainingRuntime(passes=100, batch_size=256)
    schedule = tri60_learning_rate_schedule(runtime, dict(LR_SCHEDULE))
    total = 1000
    rates = {
        update: tri60_learning_rate(
            runtime, update=update, total_updates=total,
            updates_per_pass=10, schedule=schedule,
        )
        for update in (0, 29, 30, 449, 450, 599, 600, 999)
    }
    assert rates[0] == pytest.approx(1.0e-5)
    assert rates[29] == pytest.approx(3.0e-4)
    assert rates[30] == pytest.approx(3.0e-4)
    assert rates[449] == pytest.approx(3.0e-4)
    assert rates[450] == pytest.approx(3.0e-4)
    assert rates[599] == pytest.approx(1.5e-5)
    assert rates[600] == pytest.approx(1.5e-5)
    assert rates[999] == pytest.approx(1.5e-5)
    normalized = tri60_early_stopping(runtime, dict(EARLY_STOPPING))
    assert normalized["minimum_passes"] == 60
    assert normalized["patience_passes"] == 15
    assert normalized["patience_accumulates_before_minimum"] is True


def test_production_single_gpu_acceptance_is_fully_validated():
    from hlt_classification.scouting.hcwdl_tri100_spine4_contracts import (
        EXECUTION_ACCEPTANCE_CONTRACT, artifact,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_execution import (
        validate_execution_acceptance,
    )

    value = artifact({
        "parents": {"campaign_spec": "b" * 64, "recipe": "c" * 64},
        "source_commit": "a" * 40, "execution": dict(EXECUTION),
        "hostname": "gh-a-001", "pid": 1000,
        "slurm_job_id": "12345", "slurm_nodes": "1", "slurm_tasks": "1",
        "visible_cuda_devices": 1, "device_name": "NVIDIA GH200",
        "backward_output_expected": 4.0, "backward_output_observed": 4.0,
        "backward_gradient_expected": 8.0, "backward_gradient_observed": 8.0,
        "genuine_tigris_single_gh200_worker": True, "passed": True,
        "final_test_accessed": False,
    }, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    assert validate_execution_acceptance(
        value, campaign_spec_sha256="b" * 64, recipe_sha256="c" * 64,
    ) == value["content_hash"]
    bad_payload = {
        key: item for key, item in value.items() if key != "content_hash"
    }
    bad_payload["visible_cuda_devices"] = 2
    bad = with_content_hash(bad_payload)
    with pytest.raises(ValueError, match="execution acceptance differs"):
        validate_execution_acceptance(
            bad, campaign_spec_sha256="b" * 64, recipe_sha256="c" * 64,
        )


class _SyntheticCache:
    def __init__(self, rows: int = 30, tokens: int = 3):
        labels = np.arange(rows, dtype=np.int64) % 15
        identities = np.zeros((rows, 32), dtype=np.uint8)
        identities[:, :2] = np.asarray(
            [(index // 256, index % 256) for index in range(rows)],
            dtype=np.uint8,
        )
        features = np.zeros((rows, 21, tokens), dtype=np.float32)
        vectors = np.ones((rows, 4, tokens), dtype=np.float32)
        mask = np.ones((rows, 1, tokens), dtype=np.bool_)
        visible = np.tile(np.arange(tokens, dtype=np.int64), (rows, 1))
        family = np.zeros((rows, tokens), dtype=np.int8)
        reasons = np.zeros((rows, tokens), dtype=np.int8)
        self._batch = {
            "labels": labels,
            "identity_keys": np.asarray([f"row-{i}" for i in range(rows)]),
            "identity_digests": identities,
            "hlt": HCWDLParticleInputs(
                features, vectors, mask, np.full(rows, tokens, np.int32),
                visible, family, reasons,
            ),
        }
        self.identities = tuple(self._batch["identity_keys"])
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
            yield _take_batch(
                self._batch,
                np.arange(start, min(start + batch_size, self.header["rows"])),
            )


def _tiny_model_factory():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(21, 15)

        def forward(self, features, vectors, mask):
            del vectors
            weight = mask.float()
            pooled = (features * weight).sum(-1) / weight.sum(-1).clamp_min(1)
            return self.linear(pooled)

        def no_weight_decay(self):
            return set()

    return Tiny()


def test_early_stop_completes_integral_budget_and_restores_best(tmp_path: Path):
    cache = _SyntheticCache()
    report = train_tri60_node(
        node_id="U000", train_cache=cache, validation_cache=cache,
        input_key="hlt", output_dir=tmp_path,
        parents={"foundation": SHA}, campaign_spec_sha256="b" * 64,
        recipe_sha256="c" * 64, replicate_seed=1337, device="cpu",
        runtime=Tri60TrainingRuntime(passes=3, batch_size=30),
        execution_mode="synthetic_test", model_factory=_tiny_model_factory,
        early_stopping={
            "kind": "macro_auc_patience_v1", "minimum_passes": 2,
            "patience_passes": 1, "minimum_auc_delta": 1.0,
        },
    )
    assert report["passes"] == report["validations"] == 2
    assert report["maximum_passes"] == 3 and report["minimum_passes"] == 2
    assert report["stopped_early"] is True
    assert report["performance_early_termination"] is True
    assert report["stop_reason"] == "macro_auc_patience_exhausted"
    assert report["last_meaningful_improvement_pass"] == 1
    assert report["updates"] == 2
    assert (tmp_path / "selected_model.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    assert not list(tmp_path.rglob("*resume*"))


def test_patience_exhaustion_at_maximum_pass_is_normal_completion(tmp_path: Path):
    cache = _SyntheticCache()
    report = train_tri60_node(
        node_id="U000", train_cache=cache, validation_cache=cache,
        input_key="hlt", output_dir=tmp_path,
        parents={"foundation": SHA}, campaign_spec_sha256="b" * 64,
        recipe_sha256="c" * 64, replicate_seed=1337, device="cpu",
        runtime=Tri60TrainingRuntime(passes=2, batch_size=30),
        execution_mode="synthetic_test", model_factory=_tiny_model_factory,
        early_stopping={
            "kind": "macro_auc_patience_v1", "minimum_passes": 2,
            "patience_passes": 1, "minimum_auc_delta": 1.0,
        },
    )
    assert report["passes"] == report["maximum_passes"] == 2
    assert report["stopped_early"] is False
    assert report["performance_early_termination"] is False
    assert report["stop_reason"] == "maximum_passes_reached"


def test_single_component_probability_bank_round_trip(tmp_path: Path):
    from hlt_classification.scouting.hcwdl_tri100_spine4_probability import (
        SpineProbabilityTargets, publish_probability_lock,
        publish_probability_role, validate_probability_lock,
    )

    distribution = REDUCER_ORDER[0]
    component = PROBABILITY_COMPONENTS[distribution][0]
    identities = np.zeros((4, 32), dtype=np.uint8)
    identities[:, 0] = np.arange(4, dtype=np.uint8)
    logits = np.arange(60, dtype=np.float32).reshape(4, 15) / 50
    lineage = {
        component: {
            "report_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "logits_sha256": "3" * 64,
        }
    }
    parents = {"campaign": "4" * 64}
    train = publish_probability_role(
        tmp_path, distribution_id=distribution, role="train",
        identity_digests=identities, component_logits={component: logits},
        component_lineage=lineage, parents=parents, producer_commit="a" * 40,
    )
    validation = publish_probability_role(
        tmp_path, distribution_id=distribution, role="validation",
        identity_digests=identities, component_logits={component: logits},
        component_lineage=lineage, parents=parents, producer_commit="a" * 40,
    )
    lock = publish_probability_lock(
        tmp_path / "lock.json", distribution_id=distribution,
        train_manifest=train, validation_manifest=validation, parents=parents,
    )
    validated, manifests = validate_probability_lock(
        tmp_path / "lock.json", distribution_id=distribution,
    )
    assert validated == lock
    assert manifests["train"]["temperature"] == 2.0
    assert manifests["validation"]["temperature"] == 1.0
    target = SpineProbabilityTargets.load(
        tmp_path / "train_manifest.json", distribution_id=distribution,
    )
    assert target.temperature == 2.0
    assert np.array_equal(target.join(identities[[3, 1]]), target.probabilities[[3, 1]])


def test_campaign_dag_has_four_parallel_heads_and_exact_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_tri100_spine4_campaign as campaign
    from hlt_classification.scouting.hcwdl_tri100_spine4_campaign import (
        CREATION_PHRASE, campaign_tasks, create_campaign, validate_campaign,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_contracts import (
        SOURCE_LOCK_CONTRACT, artifact,
    )

    source = artifact({
        "parents": {
            "source_campaign": "1" * 64, "source_graph": "2" * 64,
            "source_recipe": "3" * 64, "foundation": "4" * 64,
        },
        "source_campaign_spec_path": str(tmp_path / "source.json"),
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "replicate_seed": 1337,
        "role_counts": {
            "train": 2_777_855, "validation": 957_541,
            "final_test": 899_779,
        },
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(
        campaign, "validate_source_lock", lambda value: value["content_hash"],
    )
    root = tmp_path / "campaign"
    spec = create_campaign(
        source_campaign_spec=tmp_path / "source.json", campaign_root=root,
        project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True, authorization_phrase=CREATION_PHRASE,
    )
    assert validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 58
    tasks = {row["task_id"]: row for row in campaign_tasks()}
    heads = [
        f"train_{BRANCH_NODES[name][0]}" for name in BRANCH_ORDER
    ]
    assert all(tasks[name]["dependencies"] == ["preflight"] for name in heads)
    assert all(not row["external_dependencies"] for row in tasks.values())
    assert spec["source_completion_required"] is False
    assert spec["source_campaign_outputs_mutated"] is False
    assert spec["ensembles"] is False and spec["weight_continuation"] is False
    assert spec["execution"] == dict(EXECUTION)
    assert spec["resources"]["gpu_fit"]["nodes"] == 1
    assert spec["resources"]["gpu_fit"]["execution_world_size"] == 1
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 58
    fit = next(
        row["command"] for row in plan["commands"]
        if row["task_id"] == heads[0]
    )
    assert "--cpus-per-task=72" in fit
    assert "--mem=320G" in fit
    assert "--time=3-00:00:00" in fit
    assert "--nodes=1" in fit
    assert "--ntasks=1" in fit
    assert "--ntasks-per-node=1" in fit
    assert any("HCWDL_SPINE4_EXECUTION_WORLD_SIZE=1" in item for item in fit)
    assert all("hcwtri60" not in item for item in fit)
    reducer = next(
        row["command"] for row in plan["commands"]
        if row["task_id"] == f"reduce_{REDUCER_ORDER[0]}"
    )
    assert "--nodes=1" in reducer
    assert "--ntasks=1" in reducer
    assert any("HCWDL_SPINE4_EXECUTION_WORLD_SIZE=1" in item for item in reducer)


def test_source_lock_authenticates_only_completed_u000_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_tri100_spine4_source as source
    from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import (
        GRAPH_SHA256 as SOURCE_GRAPH_SHA256,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_source import (
        build_source_lock, source_consumers, validate_source_lock,
    )

    root = tmp_path / "source_campaign"
    training = root / "training/U000"
    probability = root / "probabilities/U000"
    training.mkdir(parents=True)
    probability.mkdir(parents=True)
    (training / "selected_model.pt").write_bytes(b"selected")
    (training / "final_model.pt").write_bytes(b"final")
    spec = with_content_hash({
        "parents": {
            "graph": SOURCE_GRAPH_SHA256, "recipe": "1" * 64,
            "foundation": "2" * 64,
        },
        "campaign_root": str(root),
        "artifact_paths": {"foundation_spec": str(tmp_path / "foundation.json")},
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
        "replicate_seed": 1337,
        "role_counts": {
            "train": 2_777_855, "validation": 957_541,
            "final_test": 899_779,
        },
        "population_policy": "all_authenticated_mapped_rows_v1",
    })
    spec_path = tmp_path / "source_spec.json"
    write_immutable_json(spec_path, spec)
    report = with_content_hash({
        "node_id": "U000", "campaign_spec_sha256": spec["content_hash"],
        "graph_sha256": SOURCE_GRAPH_SHA256, "recipe_sha256": "1" * 64,
        "passes": 60, "validations": 60, "complete": True,
        "rolling_resume_published": False, "partial_checkpoint_reuse": False,
        "final_test_accessed": False,
        "selected_checkpoint": "selected_model.pt",
        "selected_checkpoint_sha256": sha256_file(training / "selected_model.pt"),
        "final_checkpoint": "final_model.pt",
        "final_checkpoint_sha256": sha256_file(training / "final_model.pt"),
    })
    write_immutable_json(training / "training_report.json", report)
    probability_lock = with_content_hash({"distribution_id": "U000"})
    manifests = {
        "train": with_content_hash({"role": "train"}),
        "validation": with_content_hash({"role": "validation"}),
    }
    monkeypatch.setattr(
        source, "validate_source_campaign",
        lambda value, executable, verify_source_tree: value["content_hash"],
    )
    monkeypatch.setattr(
        source, "validate_source_artifact",
        lambda value, contract: value["content_hash"],
    )
    monkeypatch.setattr(
        source, "validate_probability_lock",
        lambda path, distribution_id: (probability_lock, manifests),
    )
    lock = build_source_lock(spec_path)
    assert validate_source_lock(lock) == lock["content_hash"]
    assert lock["u000"]["report_sha256"] == report["content_hash"]
    assert lock["u000_probability"]["lock_sha256"] == probability_lock["content_hash"]
    assert lock["authorized_probability_consumers"] == list(source_consumers())
    assert len(lock["authorized_probability_consumers"]) == 4
    assert lock["source_completion_not_required"] is True
    assert lock["read_only_import"] is True


def test_recovery_inherits_completed_parent_and_retries_from_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_tri100_spine4_recovery as recovery
    from hlt_classification.scouting.hcwdl_recovery import (
        build_submission_ledger, build_task_attestation,
        task_attestation_path,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_recovery import (
        build_monitor, create_recovery, validate_recovery,
    )

    campaign_root = tmp_path / "campaign"
    source_output = campaign_root / "source.json"
    acceptance_output = campaign_root / "acceptance.json"
    write_immutable_json(source_output, with_content_hash({"value": "source"}))
    write_immutable_json(
        acceptance_output, with_content_hash({"value": "acceptance"}),
    )
    tasks = [
        {
            "task_id": "authenticate", "kind": "authenticate",
            "dependencies": [], "external_dependencies": [],
            "resource": "cpu_lock", "node_id": None,
            "distribution_id": None,
        },
        {
            "task_id": "preflight", "kind": "preflight",
            "dependencies": ["authenticate"], "external_dependencies": [],
            "resource": "gpu_acceptance", "node_id": None,
            "distribution_id": None,
        },
        {
            "task_id": "train_example", "kind": "train",
            "dependencies": ["preflight"], "external_dependencies": [],
            "resource": "gpu_fit", "node_id": "example",
            "distribution_id": None,
        },
    ]
    subject = with_content_hash({
        "campaign_root": str(campaign_root), "tasks": tasks,
        "resources": {
            "cpu_lock": {
                "cpus": 4, "memory": "32G", "walltime": "02:00:00",
                "gpu": None, "nodes": 1, "tasks_per_node": 1,
                "execution_world_size": 1,
            },
            "gpu_acceptance": {
                "cpus": 4, "memory": "32G", "walltime": "00:30:00",
                "gpu": "gpu:gh200:1", "nodes": 1, "tasks_per_node": 1,
                "execution_world_size": 1,
            },
            "gpu_fit": {
                "cpus": 72, "memory": "320G", "walltime": "3-00:00:00",
                "gpu": "gpu:gh200:1", "nodes": 1, "tasks_per_node": 1,
                "execution_world_size": 1,
            },
        },
    })
    subject_path = tmp_path / "subject.json"
    write_immutable_json(subject_path, subject)
    monkeypatch.setattr(
        recovery, "validate_campaign", lambda value: value["content_hash"],
    )
    commands = {
        "authenticate": ["cmd-a"], "preflight": ["cmd-b"],
        "train_example": ["cmd-c"],
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=subject["content_hash"],
        jobs={
            "authenticate": "101", "preflight": "102",
            "train_example": "103",
        },
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    attestation = build_task_attestation(
        campaign_spec_sha256=subject["content_hash"],
        task_id="authenticate", array_index=None, outputs=[source_output],
    )
    write_immutable_json(
        task_attestation_path(campaign_root, "authenticate", None),
        attestation,
    )
    acceptance_attestation = build_task_attestation(
        campaign_spec_sha256=subject["content_hash"],
        task_id="preflight", array_index=None, outputs=[acceptance_output],
    )
    write_immutable_json(
        task_attestation_path(campaign_root, "preflight", None),
        acceptance_attestation,
    )
    monitor = build_monitor(
        spec=subject, ledger=ledger,
        states_by_job_id={
            "101": "COMPLETED", "102": "COMPLETED", "103": "FAILED",
        },
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    value = create_recovery(
        subject_spec=subject_path, subject_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=tmp_path / "recovery",
        project_dir=tmp_path, source_commit="a" * 40,
    )
    assert validate_recovery(value) == value["content_hash"]
    assert value["completed_tasks"] == ["authenticate", "preflight"]
    assert value["retry_tasks"] == ["train_example"]
    assert value["restart_from_zero"] is True
    plan = load_json(tmp_path / "recovery/command_plan.json")
    assert len(plan["commands"]) == 1
    assert plan["commands"][0]["task_id"] == "train_example"
    assert plan["commands"][0]["dependencies"] == []
    command = plan["commands"][0]["command"]
    assert "--nodes=1" in command
    assert "--ntasks=1" in command
    assert "--ntasks-per-node=1" in command
    assert any("HCWDL_SPINE4_EXECUTION_WORLD_SIZE=1" in item for item in command)
