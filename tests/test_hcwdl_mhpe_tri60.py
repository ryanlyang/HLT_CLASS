from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
    CONTRACTS, GRAPH_CONTRACT, RECIPE_CONTRACT, validate_artifact,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_graph import (
    COORDINATES, ENSEMBLE_COMPONENTS, FIT_ORDER, GRAPH_SHA256,
    NODE_REGISTRY, REDUCER_ORDER, REPRESENTATION_CARRIERS, graph_payload,
    validate_graph,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_campaign import (
    RESOURCES, campaign_tasks,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_operations import (
    build_cancellation, build_monitor, validate_monitor,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_recovery import (
    create_recovery, failed_downstream_closure, validate_recovery,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_recipe import (
    recipe_payload, validate_recipe,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_ephemeral import (
    EphemeralRepresentationTargetBank,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_probability import (
    Tri60ProbabilityTargets, load_probability_role,
    publish_probability_lock, publish_probability_role,
)
from hlt_classification.scouting.hcwdl_mhpe_tri60_training import (
    Tri60TrainingInterrupted, Tri60TrainingRuntime, load_tri60_model,
    train_tri60_node,
    tri60_base_loss,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_submission_event, build_submission_ledger,
)
from hlt_classification.data.cache_contracts import (
    sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_representation_data import HCWDLParticleInputs
from hlt_classification.scouting.hcwdl_representation_target_runtime import (
    PreparedTargetGeneration, PreparedTargetPartition,
)
from hlt_classification.scouting.hcwdl_representation_kernels import (
    generate_spectral_resource_bundle,
)
from hlt_classification.scouting.hcwdl_representation_targets import (
    ORDINARY_BANK, identity_order_sha256, identity_set_sha256,
    target_array_schema,
)


SHA = "a" * 64


def test_foundation_authentication_hashes_the_selection_artifact_not_a_missing_parent_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_integration as integration

    root = tmp_path / "foundation"
    (root / "locks").mkdir(parents=True)
    (root / "source").mkdir()
    (root / "training/M0paired").mkdir(parents=True)

    split = with_content_hash({
        "contract": "TEST_SPLIT/v1", "schema_version": 1,
    })
    split_path = root / "source/split.json"
    write_immutable_json(split_path, split)
    selection = with_content_hash({
        "contract": "TEST_SELECTION/v1", "schema_version": 1,
        "split_manifest_sha256": split["content_hash"],
    })
    selection_path = root / "source/selection.json"
    write_immutable_json(selection_path, selection)
    spec = with_content_hash({
        "contract": "TEST_FOUNDATION/v1", "schema_version": 1,
        "campaign_root": str(root.resolve()),
        "artifact_paths": {
            "split_manifest": str(split_path.resolve()),
            "selection_manifest": str(selection_path.resolve()),
        },
        "parents": {"split_manifest_sha256": split["content_hash"]},
        "role_counts": {
            "train": 2_600_000, "validation": 1_000_000,
            "final_test": 1_000_000,
        },
        "final_test_accessed": False,
    })
    write_immutable_json(root / "foundation_spec.json", spec)

    checkpoint = root / "training/M0paired/selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    m0 = with_content_hash({
        "contract": "TEST_PMARD_REPORT/v1", "schema_version": 1,
        "selected_checkpoint": checkpoint.name,
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "completed_natural_population_passes": 20,
    })
    write_immutable_json(root / "training/M0paired/training_report.json", m0)
    lock = with_content_hash({
        "contract": "TEST_FOUNDATION_LOCK/v1", "schema_version": 1,
        "foundation_spec_sha256": spec["content_hash"],
        "m0paired_report_sha256": m0["content_hash"],
        "m0paired_checkpoint_sha256": m0["selected_checkpoint_sha256"],
        "parents": {
            "assignment_lock_sha256": "1" * 64,
            "coupling_lock_sha256": "2" * 64,
            "endpoint_lock_sha256": "3" * 64,
            "train_balanced_manifest_sha256": "4" * 64,
            "validation_balanced_manifest_sha256": "5" * 64,
        },
    })
    lock_path = root / "locks/foundation.json"
    write_immutable_json(lock_path, lock)

    monkeypatch.setattr(
        integration, "validate_foundation_lock",
        lambda value: value["content_hash"],
    )
    monkeypatch.setattr(
        integration, "validate_foundation_campaign",
        lambda value, **kwargs: value["content_hash"],
    )
    monkeypatch.setattr(
        integration, "validate_pmard_training_report",
        lambda value: value["content_hash"],
    )

    authenticated = integration.authenticate_foundation(lock_path)
    assert authenticated["parents"]["split_manifest"] == split["content_hash"]
    assert authenticated["parents"]["selection_manifest"] == selection["content_hash"]


def test_graph_has_exact_fit_and_reducer_registries():
    assert validate_graph() == GRAPH_SHA256
    assert len(NODE_REGISTRY) == len(FIT_ORDER) == 32
    assert len(ENSEMBLE_COMPONENTS) == len(REDUCER_ORDER) == 12
    assert FIT_ORDER[0] == "U000"
    assert FIT_ORDER[-1] == "M2"
    assert REDUCER_ORDER[-1] == "M1E"
    assert ENSEMBLE_COMPONENTS["LOGIT_U050E"] == (
        "LOGIT_U050_from_U000",
    )
    assert ENSEMBLE_COMPONENTS["M1E"] == (
        "M1_LOGIT", "M1_RSET", "M1_RREL",
    )
    payload = graph_payload()
    assert validate_artifact(payload, contract=GRAPH_CONTRACT) == GRAPH_SHA256


def test_fit_order_is_the_predeclared_32_fit_registry():
    assert FIT_ORDER == (
        "U000",
        "LOGIT_U050_from_U000",
        "LOGIT_U100_from_U000", "LOGIT_U100_from_U050E",
        "LOGIT_D066_from_U000", "LOGIT_D066_from_U050E",
        "LOGIT_D066_from_U100E",
        "LOGIT_D033_from_U000", "LOGIT_D033_from_U050E",
        "LOGIT_D033_from_U100E", "LOGIT_D033_from_D066E",
        "LOGIT_D000_from_U000", "LOGIT_D000_from_U050E",
        "LOGIT_D000_from_U100E", "LOGIT_D000_from_D066E",
        "LOGIT_D000_from_D033E", "M1_LOGIT",
        "RSET_U100_from_U000",
        "RSET_D050_from_U000", "RSET_D050_from_U100E",
        "RSET_D000_from_U000", "RSET_D000_from_U100E",
        "RSET_D000_from_D050E", "M1_RSET",
        "RREL_U100_from_U000",
        "RREL_D050_from_U000", "RREL_D050_from_U100E",
        "RREL_D000_from_U000", "RREL_D000_from_U100E",
        "RREL_D000_from_D050E", "M1_RREL", "M2",
    )


def test_campaign_dag_is_exact_parallel_and_resource_locked():
    tasks = campaign_tasks()
    assert len(tasks) == 50
    assert sum(row["kind"] == "train" for row in tasks) == 32
    assert sum(row["kind"] == "reducer" for row in tasks) == 13
    by_id = {row["task_id"]: row for row in tasks}
    assert by_id["train_M1_LOGIT"]["dependencies"] == ["reduce_LOGIT_D000E"]
    assert by_id["train_M1_RSET"]["dependencies"] == ["reduce_RSET_D000E"]
    assert by_id["train_M1_RREL"]["dependencies"] == ["reduce_RREL_D000E"]
    assert by_id["reduce_M1E"]["dependencies"] == [
        "train_M1_LOGIT", "train_M1_RSET", "train_M1_RREL",
    ]
    assert by_id["train_M2"]["dependencies"] == ["reduce_M1E"]
    assert RESOURCES["gpu_logit"].memory == "256G"
    assert RESOURCES["gpu_rset"].memory == RESOURCES["gpu_rrel"].memory == "384G"
    assert RESOURCES["gpu_rset"].walltime == RESOURCES["gpu_rrel"].walltime == "6-00:00:00"
    assert all(row["resource_class"] != "gpu_reducer" or row["kind"] == "reducer" for row in tasks)


def test_graph_coordinates_are_exact_rationals():
    assert COORDINATES["U000"].payload()["structural"] == [0, 1]
    assert COORDINATES["U050"].payload()["structural"] == [1, 2]
    assert COORDINATES["U100"].payload()["structural"] == [1, 1]
    assert COORDINATES["D066"].payload()["feature"] == [1, 3]
    assert COORDINATES["D050"].payload()["feature"] == [1, 2]
    assert COORDINATES["D033"].payload()["feature"] == [2, 3]
    assert COORDINATES["D000"].payload()["feature"] == [1, 1]


def test_every_fit_is_fresh_60_pass_and_loss_routing_is_exact():
    for node in NODE_REGISTRY.values():
        assert node.initialization == "fresh"
        assert node.training_passes == 60
        assert node.batch_size == 256
        if node.node_id == "U000":
            assert (node.ce_weight, node.kd_weight) == (1.0, 0.0)
        elif node.node_id.startswith("M1_") or node.node_id == "M2":
            assert (node.ce_weight, node.kd_weight, node.temperature) == (
                .10, .90, 1.0,
            )
        else:
            assert (node.ce_weight, node.kd_weight, node.temperature) == (
                .25, .75, 2.0,
            )
        if node.track in {"RSET", "RREL"}:
            assert node.auxiliary == node.track.lower()
            assert node.representation_carrier_id is not None
        else:
            assert node.auxiliary == "none"
            assert node.representation_carrier_id is None


def test_representation_carriers_are_fixed_and_not_metric_selected():
    assert REPRESENTATION_CARRIERS == {
        "U000": "U000",
        "RSET_U100E": "RSET_U100_from_U000",
        "RSET_D050E": "RSET_D050_from_U100E",
        "RSET_D000E": "RSET_D000_from_D050E",
        "RREL_U100E": "RREL_U100_from_U000",
        "RREL_D050E": "RREL_D050_from_U100E",
        "RREL_D000E": "RREL_D000_from_D050E",
    }
    assert NODE_REGISTRY["M1_RSET"].representation_carrier_id == (
        "RSET_D000_from_D050E"
    )
    assert NODE_REGISTRY["M1_RREL"].representation_carrier_id == (
        "RREL_D000_from_D050E"
    )


def test_same_target_nodes_share_backbone_data_seed_but_rep_heads_do_not():
    u100 = [node for node in NODE_REGISTRY.values()
            if node.coordinate_name == "U100"]
    assert len({node.seed_alias for node in u100}) == 1
    assert {
        node.representation_seed_alias for node in u100
        if node.representation_seed_alias is not None
    } == {
        "HCWDL-MHPE-THREE-TRACK-60E-FULL/v1/rset/U100/representation",
        "HCWDL-MHPE-THREE-TRACK-60E-FULL/v1/rrel/U100/representation",
    }
    assert NODE_REGISTRY["M1_LOGIT"].seed_alias == NODE_REGISTRY["M1_RSET"].seed_alias
    assert NODE_REGISTRY["M1_RSET"].seed_alias == NODE_REGISTRY["M1_RREL"].seed_alias


def test_recipe_locks_ram_only_no_resume_semantics():
    recipe = recipe_payload(
        base_recipe_sha256=SHA,
        representation_recipe_sha256="b" * 64,
        unified_balanced_recipe_sha256="c" * 64,
    )
    assert validate_recipe(recipe) == recipe["content_hash"]
    assert recipe["contract"] == RECIPE_CONTRACT
    assert recipe["training"]["passes"] == 60
    assert recipe["training"]["effective_batch_size"] == 256
    assert recipe["persistence"]["representation_targets"] == (
        "ram_only_never_persist_v1"
    )
    assert recipe["persistence"]["rolling_resume"] is False
    assert recipe["persistence"]["partial_checkpoint_reuse"] is False
    assert recipe["loss"]["rrel"]["relation_state"] == (
        "raw_block2_fp32_normalized_v1"
    )


def test_runtime_adapter_maps_recipe_floor_to_runtime_field(tmp_path: Path):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_runner as runner

    recipe = recipe_payload(
        base_recipe_sha256=SHA,
        representation_recipe_sha256="b" * 64,
        unified_balanced_recipe_sha256="c" * 64,
    )
    recipe_path = tmp_path / "recipe.json"
    write_immutable_json(recipe_path, recipe)

    runtime = runner._runtime(
        {"artifact_paths": {"recipe": str(recipe_path)}},
        "U000",
    )

    assert runtime.minimum_lr_fraction == recipe["training"][
        "learning_rate_floor_fraction"
    ]
    runtime.validate(execution_mode="scientific")


def test_u000_production_cache_requests_strict_identity_metadata(monkeypatch):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_runner as runner

    observed = {}
    foundation = {"content_hash": SHA}
    split = {"contract": "synthetic"}
    selections = {"train": object(), "validation": object()}
    assignments = {"train": object(), "validation": object()}
    balanced = {"train": object(), "validation": object()}

    monkeypatch.setattr(runner, "_foundation", lambda _spec: foundation)
    monkeypatch.setattr(
        runner,
        "_load_common",
        lambda _foundation: (
            split, "b" * 64, "c" * 64, selections, assignments, balanced,
        ),
    )

    def fake_cache_student_views(**kwargs):
        observed.update(kwargs)
        return {"train": object(), "validation": object()}, "privileged"

    monkeypatch.setattr(runner, "_cache_student_views", fake_cache_student_views)

    result = runner._student_caches({"replicate_seed": 1337}, node_id="U000")

    assert result[-1] == "privileged"
    assert observed["behavior"] == "p0"
    assert observed["include_hcwdl_metadata"] is True


def test_carrier_partitions_resolve_all_rows_and_skip_empty_sources():
    from hlt_classification.scouting import hcwdl_mhpe_tri60_runner as runner

    records = (
        SimpleNamespace(path="empty.root", mapped_entries=0),
        SimpleNamespace(path="explicit.root", mapped_entries=9),
        SimpleNamespace(path="all.root", mapped_entries=4),
    )

    class Selection:
        rows = 7

        @staticmethod
        def source_rows(path):
            return {"empty.root": 0, "explicit.root": 3, "all.root": -1}[path]

    partitions = runner._carrier_source_partitions(
        records, selection=Selection(),
    )

    assert partitions == (
        {
            "partition": "source_0001", "source_index": 1,
            "source_file_id": 1, "source_path": "explicit.root", "rows": 3,
        },
        {
            "partition": "source_0002", "source_index": 2,
            "source_file_id": 2, "source_path": "all.root", "rows": 4,
        },
    )


def test_carrier_partitions_fail_closed_on_population_mismatch():
    from hlt_classification.scouting import hcwdl_mhpe_tri60_runner as runner

    records = (SimpleNamespace(path="all.root", mapped_entries=4),)

    class Selection:
        rows = 5

        @staticmethod
        def source_rows(_path):
            return -1

    with pytest.raises(ValueError, match="partition coverage differs"):
        runner._carrier_source_partitions(records, selection=Selection())


def test_contract_inventory_is_versioned_and_unique():
    assert len(CONTRACTS) == len(set(CONTRACTS))
    assert all(value.startswith("HCWDL_MHPE_THREE_TRACK_60E_") for value in CONTRACTS)
    assert all(value.endswith("/v1") for value in CONTRACTS)


def test_campaign_publication_dry_shape_and_restart_zero_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_campaign as campaign
    from hlt_classification.scouting.hcwdl_mhpe_tri60_campaign import (
        CREATION_PHRASE, create_campaign, validate_campaign,
    )
    from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
        ENDPOINT_RESOURCE_LOCK_CONTRACT, FOUNDATION_LOCK_CONTRACT,
        INTEGRATION_LOCK_CONTRACT, RUNTIME_PROFILE_CONTRACT,
        TEST_EVIDENCE_CONTRACT, artifact,
    )

    source_commit = "d" * 40
    foundation_root = tmp_path / "foundation"
    foundation_root.mkdir()
    base_recipe = with_content_hash({
        "contract": "TEST_TRI60_BASE_RECIPE/v1", "schema_version": 1,
    })
    write_immutable_json(foundation_root / "recipe.json", base_recipe)
    foundation_spec = with_content_hash({
        "contract": "TEST_TRI60_FOUNDATION/v1", "schema_version": 1,
        "parents": {"recipe_overlay_sha256": "e" * 64},
    })
    foundation_spec_path = foundation_root / "foundation_spec.json"
    write_immutable_json(foundation_spec_path, foundation_spec)
    foundation = artifact({
        "parents": {"foundation_spec": foundation_spec["content_hash"]},
        "foundation_spec_path": str(foundation_spec_path.resolve()),
        "role_counts": {
            "train": 2_600_000, "validation": 1_000_000,
            "final_test": 1_000_000,
        },
        "ordinary_access_roles": ["train", "validation"],
        "ordinary_final_test_capability": False,
        "population_policy": "all_authenticated_mapped_rows_v1",
        "contextual_m0paired_report_path": str(tmp_path / "m0.json"),
        "contextual_m0paired_pass_count": 20,
        "final_test_accessed": False,
    }, contract=FOUNDATION_LOCK_CONTRACT)
    integration = artifact({
        "parents": {"tests": "1" * 64, "parity": "2" * 64},
        "source_commit": source_commit,
        "semantic_source_sha256": {"semantic.py": "3" * 64},
        "runtime_sibling_worktree_imports": False,
        "final_test_accessed": False,
    }, contract=INTEGRATION_LOCK_CONTRACT)
    endpoint = artifact({
        "parents": {"integration": integration["content_hash"]},
        "final_test_accessed": False,
    }, contract=ENDPOINT_RESOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "authenticate_foundation", lambda path: foundation)
    monkeypatch.setattr(campaign, "build_integration_lock", lambda **kwargs: integration)
    monkeypatch.setattr(campaign, "build_endpoint_resource_lock", lambda **kwargs: endpoint)
    monkeypatch.setattr(
        campaign, "semantic_source_hashes",
        lambda path: {"semantic.py": "3" * 64},
    )
    monkeypatch.setattr(
        campaign, "validate_representation_recipe",
        lambda value: value["content_hash"],
    )
    representation_recipe = with_content_hash({
        "contract": "HCWDL_REPRESENTATION_RECIPE/v5", "schema_version": 5,
    })
    representation_recipe_path = tmp_path / "representation_recipe.json"
    write_immutable_json(representation_recipe_path, representation_recipe)
    tests = artifact({
        "source_commit": source_commit, "passed": True,
        "final_test_accessed": False,
    }, contract=TEST_EVIDENCE_CONTRACT)
    tests_path = tmp_path / "tests.json"
    write_immutable_json(tests_path, tests)
    # The established Weaver artifact has nested factory results and no
    # invented top-level ``passed`` field.
    parity = with_content_hash({
        "contract": "HCWDL_STRUCTURAL_FEATURE_WEAVER_PARITY/v1",
        "schema_version": 1, "source_commit": source_commit,
        "device": "cuda", "unified_factory": {"passed": True},
        "native_teacher_factory": {"passed": True},
        "final_test_accessed": False,
    })
    parity_path = tmp_path / "parity.json"
    write_immutable_json(parity_path, parity)
    profile = artifact({
        "parents": {"foundation": foundation["content_hash"]},
        "source_commit": source_commit, "passed": True,
        "genuine_tigris_production_worker": True,
        "ram_only_targets_proved": True, "no_resume_proved": True,
        "peak_request_fraction": .5, "temporary_artifacts_deleted": True,
        "temporary_artifact_bytes_after_cleanup": 0,
        "final_test_accessed": False,
    }, contract=RUNTIME_PROFILE_CONTRACT)
    profile_path = tmp_path / "profile.json"
    write_immutable_json(profile_path, profile)

    root = tmp_path / "campaign"
    spec = create_campaign(
        foundation_lock=tmp_path / "foundation-lock.json",
        representation_recipe=representation_recipe_path,
        test_evidence=tests_path, installed_weaver_parity=parity_path,
        runtime_profile=profile_path, campaign_root=root,
        project_dir=tmp_path, source_commit=source_commit,
        authorize_live_submission=True, authorization_phrase=CREATION_PHRASE,
    )
    assert validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 50
    assert not any("final_test" in row["task_id"] for row in spec["tasks"])
    plan = __import__("json").loads((root / "command_plan.json").read_text())
    assert len(plan["commands"]) == 50
    assert "--mem=384G" in next(
        row["command"] for row in plan["commands"]
        if row["task_id"] == "train_RSET_U100_from_U000"
    )

    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    jobs = {
        task_id: str(91000 + index)
        for index, task_id in enumerate(commands)
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = build_monitor(
        subject=spec, ledger=ledger,
        states_by_job_id={job: "FAILED" for job in jobs.values()},
        attestation_root=root,
    )
    monitor_path = tmp_path / "monitor.json"
    write_immutable_json(monitor_path, monitor)
    recovery = create_recovery(
        subject_spec=root / "campaign_spec.json",
        subject_ledger=ledger_path, monitor_report=monitor_path,
        recovery_root=tmp_path / "recovery", project_dir=tmp_path,
        source_commit=source_commit,
    )
    assert validate_recovery(recovery) == recovery["content_hash"]
    assert recovery["recovery_tasks"] == [row["task_id"] for row in spec["tasks"]]
    assert recovery["resume_policy"] == "disabled_restart_from_zero_v1"


def test_live_submission_journal_replays_exact_dependency_commands(tmp_path):
    import importlib.util

    script_path = (
        Path(__file__).parents[1] / "scripts/submit_hcwdl_mhpe_tri60_campaign.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "tri60_submitter_test", script_path,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    campaign = {"content_hash": SHA}
    plan = {"commands": [
        {"task_id": "root", "dependencies": [], "command": ["sbatch", "root.sh"]},
        {
            "task_id": "child", "dependencies": ["root"],
            "command": ["sbatch", "--dependency=afterok:${JOB_root}", "child.sh"],
        },
    ]}
    journal = tmp_path / "journal"
    root_event = build_submission_event(
        campaign_spec_sha256=SHA, task_id="root", job_id="90001",
        command=["sbatch", "root.sh"], sequence=0,
    )
    child_event = build_submission_event(
        campaign_spec_sha256=SHA, task_id="child", job_id="90002",
        command=["sbatch", "--dependency=afterok:90001", "child.sh"], sequence=1,
    )
    write_immutable_json(journal / "0000_root.json", root_event)
    write_immutable_json(journal / "0001_child.json", child_event)
    events, jobs = module._load_journal(
        journal=journal, spec=campaign, plan=plan,
    )
    assert events == [root_event, child_event]
    assert jobs == {"root": "90001", "child": "90002"}
    (journal / "0001_child.json").rename(journal / "0002_child.json")
    with pytest.raises(ValueError, match="journal differs"):
        module._load_journal(journal=journal, spec=campaign, plan=plan)


def _prepared_targets(rows=3):
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in target_array_schema(ORDINARY_BANK, rows).items()
    }
    arrays["source_file_id"][:] = 7
    arrays["source_entry"][:] = np.arange(rows, dtype=np.uint64)
    for index in range(rows):
        arrays["identity_digest"][index, 0] = index + 1
    arrays["label"][:] = np.arange(rows, dtype=np.uint8)
    arrays["token_family_eligibility"][:] = 1
    arrays["token_count"][:] = 2
    arrays["token_scalar_pt_sum"][:] = 3.0
    arrays["family_reason_counts"][:] = 2
    arrays["relation_pair_count"][:] = 1
    arrays["relation_effective_sample"][:] = 1.0
    identities = arrays["identity_digest"]
    partition = PreparedTargetPartition(
        arrays=MappingProxyType(arrays),
        runtime_audit=MappingProxyType({"canonical_batches": [], "rows": rows}),
        teacher_forward_calls=1,
    )
    return PreparedTargetGeneration(
        bank_kind=ORDINARY_BANK,
        partitions=MappingProxyType({"source-0007": partition}),
        partition_specs=MappingProxyType({"source-0007": {"rows": rows, "source_file_id": 7}}),
        class_counts=tuple([1] * rows + [0] * (15 - rows)),
        identity_order_sha256=identity_order_sha256(identities),
        identity_set_sha256=identity_set_sha256(identities),
        population_rows_sha256="d" * 64,
        canonical_batches=(),
        teacher_forward_calls=1,
        construction_seconds=.25,
    )


def test_ephemeral_representation_bank_joins_and_never_exposes_a_path(tmp_path):
    bank = EphemeralRepresentationTargetBank.from_prepared(
        _prepared_targets(), strategy="RREL", carrier_node_id="RREL_U100_from_U000",
        carrier_report_sha256="e" * 64, carrier_checkpoint_sha256="f" * 64,
        campaign_spec_sha256=SHA, graph_sha256="b" * 64,
        recipe_sha256="c" * 64,
    )
    requested = np.ascontiguousarray(bank.arrays["identity_digest"][[2, 0]])
    joined = bank.join(requested)
    assert joined["logits"].shape == (2, 15)
    assert joined["relation_kernel_mean"].shape == (2, 3, 256)
    assert bank.nbytes > 0
    assert not hasattr(bank, "save")
    assert not list(tmp_path.iterdir())
    audit = bank.audit(peak_rss_bytes=123, peak_cuda_bytes=456)
    assert audit["representation_targets_persisted"] is False
    assert audit["durable_payload"] is False
    assert audit["audit_cannot_reconstruct_target_bytes"] is True
    bank.release()
    assert bank.nbytes == 0
    with pytest.raises(RuntimeError, match="released"):
        bank.join(requested)


def test_ephemeral_representation_bank_equals_prepared_reference_arrays():
    prepared = _prepared_targets(rows=4)
    bank = EphemeralRepresentationTargetBank.from_prepared(
        prepared, strategy="RREL", carrier_node_id="RREL_U100_from_U000",
        carrier_report_sha256="e" * 64, carrier_checkpoint_sha256="f" * 64,
        campaign_spec_sha256=SHA, graph_sha256="b" * 64,
        recipe_sha256="c" * 64,
    )
    source = next(iter(prepared.partitions.values())).arrays
    for name, value in bank.arrays.items():
        assert np.array_equal(value, source[name])
    assert bank.header["representation_targets_persisted"] is False


def test_ephemeral_representation_bank_fails_closed_on_missing_or_repeated_join():
    bank = EphemeralRepresentationTargetBank.from_prepared(
        _prepared_targets(), strategy="RSET", carrier_node_id="U000",
        carrier_report_sha256="e" * 64, carrier_checkpoint_sha256="f" * 64,
        campaign_spec_sha256=SHA, graph_sha256="b" * 64,
        recipe_sha256="c" * 64,
    )
    existing = np.ascontiguousarray(bank.arrays["identity_digest"][[0, 0]])
    with pytest.raises(ValueError, match="repeats"):
        bank.join(existing)
    missing = np.zeros((1, 32), dtype=np.uint8)
    missing[0, 0] = 255
    with pytest.raises(KeyError, match="incomplete"):
        bank.join(missing)


def _probability_inputs(components, rows=4):
    identities = np.zeros((rows, 32), dtype=np.uint8)
    identities[:, 0] = np.arange(1, rows + 1, dtype=np.uint8)
    logits = {
        name: np.ascontiguousarray(
            np.arange(rows * 15, dtype=np.float32).reshape(rows, 15)
            * np.float32(index + 1) / np.float32(100),
        )
        for index, name in enumerate(components)
    }
    lineage = {
        name: {
            "report_sha256": "abcdef"[index % 6] * 64,
            "checkpoint_sha256": "abcdef"[(index + 1) % 6] * 64,
            "logits_sha256": "abcdef"[(index + 2) % 6] * 64,
        }
        for index, name in enumerate(components)
    }
    return identities, logits, lineage


def test_probability_bundle_uses_binary_identities_and_real_one_member_bank(tmp_path):
    identities, logits, lineage = _probability_inputs(("U000",))
    parents = {"campaign": SHA, "graph": "b" * 64}
    train = publish_probability_role(
        tmp_path, distribution_id="U000", role="train",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents=parents, producer_commit="c" * 40,
    )
    validation = publish_probability_role(
        tmp_path, distribution_id="U000", role="validation",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents=parents, producer_commit="c" * 40,
    )
    lock = publish_probability_lock(
        tmp_path / "lock.json", distribution_id="U000",
        train_manifest=train, validation_manifest=validation, parents=parents,
    )
    assert lock["authorized"] is True
    loaded, loaded_identities, probabilities = load_probability_role(
        tmp_path / "train_manifest.json", expected_distribution_id="U000",
        expected_role="train",
    )
    assert loaded["temperature"] == 2.0
    assert loaded_identities.dtype == np.uint8
    assert loaded_identities.shape == (4, 32)
    expected = np.exp(logits["U000"] / 2 - (logits["U000"] / 2).max(1, keepdims=True))
    expected /= expected.sum(1, keepdims=True)
    assert np.allclose(probabilities, expected, rtol=1e-6, atol=1e-7)
    target = Tri60ProbabilityTargets.load(
        tmp_path / "train_manifest.json", distribution_id="U000",
    )
    assert np.array_equal(target.join(identities[[3, 1]]), probabilities[[3, 1]])


def test_probability_bundle_fails_closed_when_compact_bytes_change(tmp_path):
    identities, logits, lineage = _probability_inputs(("U000",))
    manifest = publish_probability_role(
        tmp_path, distribution_id="U000", role="train",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents={"campaign": SHA},
        producer_commit="c" * 40,
    )
    data = tmp_path / "train.npz"
    raw = bytearray(data.read_bytes())
    raw[len(raw) // 2] ^= 1
    data.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes differ"):
        load_probability_role(
            tmp_path / "train_manifest.json",
            expected_distribution_id=manifest["distribution_id"],
        )


def test_probability_publication_rejects_non_commit_identity_before_writing(tmp_path):
    identities, logits, lineage = _probability_inputs(("U000",))
    with pytest.raises(ValueError, match="producer commit"):
        publish_probability_role(
            tmp_path, distribution_id="U000", role="train",
            identity_digests=identities, component_logits=logits,
            component_lineage=lineage, parents={"campaign": SHA},
            producer_commit="c" * 64,
        )
    assert not list(tmp_path.iterdir())


def test_probability_reducer_is_uniform_and_d000_train_is_t1(tmp_path):
    components = (
        "LOGIT_D000_from_U000", "LOGIT_D000_from_U050E",
        "LOGIT_D000_from_U100E", "LOGIT_D000_from_D066E",
        "LOGIT_D000_from_D033E",
    )
    identities, logits, lineage = _probability_inputs(components)
    manifest = publish_probability_role(
        tmp_path, distribution_id="LOGIT_D000E", role="train",
        identity_digests=identities, component_logits=logits,
        component_lineage=lineage, parents={"campaign": SHA},
        producer_commit="c" * 40,
    )
    assert manifest["temperature"] == 1.0
    _, _, actual = load_probability_role(tmp_path / "train_manifest.json")
    manual = np.zeros_like(actual, dtype=np.float64)
    for name in sorted(components):
        value = logits[name]
        exponent = np.exp(value - value.max(1, keepdims=True), dtype=np.float32)
        manual += (exponent / exponent.sum(1, keepdims=True)).astype(np.float64)
    manual = np.asarray(manual / len(components), dtype="<f4")
    assert np.array_equal(actual, manual)


def test_tri60_probability_base_loss_matches_manual_forward_kl():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[.1, .2] + [0.0] * 13], dtype=torch.float32)
    labels = torch.tensor([1])
    teacher_logits = torch.tensor([[.3, -.2] + [0.0] * 13])
    probability = torch.softmax(teacher_logits / 2.0, dim=-1)
    result = tri60_base_loss(
        logits, labels, teacher_probabilities=probability,
        ce_weight=.25, kd_weight=.75, temperature=2.0,
    )
    manual_ce = torch.nn.functional.cross_entropy(logits, labels)
    manual_kd = torch.nn.functional.kl_div(
        torch.log_softmax(logits / 2.0, dim=-1), probability,
        reduction="batchmean",
    ) * 4.0
    assert torch.allclose(result["total"], .25 * manual_ce + .75 * manual_kd)


class _SyntheticCache:
    def __init__(self, rows=30, tokens=3):
        labels = np.arange(rows, dtype=np.int64) % 15
        identities = np.zeros((rows, 32), dtype=np.uint8)
        identities[:, :2] = np.asarray(
            [(index // 256, index % 256) for index in range(rows)], dtype=np.uint8,
        )
        features = np.zeros((rows, 21, tokens), dtype=np.float32)
        features[:, 0, :] = np.arange(rows, dtype=np.float32)[:, None] / rows
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
            "labels": labels, "identity_keys": np.asarray([f"row-{i}" for i in range(rows)]),
            "identity_digests": identities, "hlt": view,
        }
        self.identities = tuple(self._batch["identity_keys"])
        self.identity_digests = identities
        self.header = {"rows": rows, "array_bytes": sum(
            value.nbytes for value in (labels, identities, features, vectors, mask)
        )}

    def iterate_batches(self, *, epoch, sampler_seed, batch_size):
        del epoch, sampler_seed
        from hlt_classification.scouting.dataset import _take_batch
        for start in range(0, self.header["rows"], batch_size):
            yield _take_batch(
                self._batch,
                np.arange(start, min(start + batch_size, self.header["rows"])),
            )

    def iterate_identity_digest_batches(self, ordered_identity_sha256s, *, batch_size):
        from hlt_classification.scouting.dataset import _take_batch
        lookup = {
            bytes(row).hex(): index
            for index, row in enumerate(self.identity_digests)
        }
        indexes = np.asarray(
            [lookup[value] for value in ordered_identity_sha256s], dtype=np.int64,
        )
        for start in range(0, len(indexes), batch_size):
            yield _take_batch(self._batch, indexes[start:start + batch_size])


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


def _tiny_scouting_factory():
    torch = pytest.importorskip("torch")
    from hlt_classification.models.scouting_particle_transformer import (
        HCWDLScoutingSurfaces, ScoutingParticleTransformer,
    )

    class TinyScouting(ScoutingParticleTransformer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.mod = torch.nn.Module()
            self.mod.embed = torch.nn.Linear(21, 128)
            self.mod.pair_embed = torch.nn.Linear(4, 128)
            self.mod.blocks = torch.nn.ModuleList([
                torch.nn.Linear(128, 128), torch.nn.Linear(128, 128),
            ])
            self.classifier = torch.nn.Linear(128, 15)

        def _tokens(self, features, vectors):
            value = (
                self.mod.embed(features.transpose(1, 2))
                + self.mod.pair_embed(vectors.transpose(1, 2))
            )
            for block in self.mod.blocks:
                value = torch.nn.functional.gelu(block(value))
            return value

        def forward(self, features, vectors, mask):
            tokens = self._tokens(features, vectors)
            weights = mask.transpose(1, 2).float()
            jet = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
            return self.classifier(jet)

        def forward_hcwdl_surfaces(
            self, features, vectors, mask, visible_indices, family_codes,
        ):
            tokens = self._tokens(features, vectors)
            weights = mask.transpose(1, 2).float()
            jet = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
            return HCWDLScoutingSurfaces(
                logits=self.classifier(jet), particle_block_2=tokens,
                jet_penultimate=jet, particle_mask=mask.squeeze(1),
                vectors=vectors, visible_indices=visible_indices,
                family_codes=family_codes,
            )

        def no_weight_decay(self):
            return set()

    return TinyScouting()


def test_no_resume_synthetic_fit_publishes_only_terminal_checkpoints(tmp_path):
    cache = _SyntheticCache()
    report = train_tri60_node(
        node_id="U000", train_cache=cache, validation_cache=cache,
        input_key="hlt", output_dir=tmp_path,
        parents={"foundation": SHA}, campaign_spec_sha256="b" * 64,
        recipe_sha256="c" * 64, replicate_seed=1337, device="cpu",
        runtime=Tri60TrainingRuntime(passes=2, batch_size=30),
        execution_mode="synthetic_test", model_factory=_tiny_model_factory,
    )
    assert report["passes"] == report["validations"] == 2
    assert report["rolling_resume_published"] is False
    assert (tmp_path / "selected_model.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    assert not list(tmp_path.rglob("*resume*"))
    loaded, loaded_report = load_tri60_model(
        tmp_path / "training_report.json", device="cpu",
        model_factory=_tiny_model_factory,
    )
    assert loaded_report["content_hash"] == report["content_hash"]
    assert loaded.training is False
    assert {parameter.dtype for parameter in loaded.parameters()} == {
        pytest.importorskip("torch").float32,
    }


def test_interrupted_fit_publishes_only_small_restart_from_zero_attestation(
    tmp_path, monkeypatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_training as module

    original = module._SignalMonitor.install

    def requested(self):
        original(self)
        self.requested = True
        self.number = int(__import__("signal").SIGTERM)

    monkeypatch.setattr(module._SignalMonitor, "install", requested)
    with pytest.raises(Tri60TrainingInterrupted, match="restart from zero"):
        train_tri60_node(
            node_id="U000", train_cache=_SyntheticCache(),
            validation_cache=_SyntheticCache(), input_key="hlt",
            output_dir=tmp_path, parents={"foundation": SHA},
            campaign_spec_sha256="b" * 64, recipe_sha256="c" * 64,
            execution_source_commit="d" * 40, replicate_seed=1337,
            device="cpu", runtime=Tri60TrainingRuntime(passes=2, batch_size=30),
            execution_mode="synthetic_test", model_factory=_tiny_model_factory,
        )
    assert not (tmp_path / "selected_model.pt").exists()
    assert not (tmp_path / "final_model.pt").exists()
    assert not list(tmp_path.rglob("*resume*"))
    interruptions = list((tmp_path / "interruptions").glob("*.json"))
    assert len(interruptions) == 1
    payload = __import__("json").loads(interruptions[0].read_text())
    assert payload["resume_checkpoint_published"] is False
    assert payload["restart_policy"] == "restart_from_update_zero_v1"


def test_ram_only_rset_targets_execute_through_the_real_representation_model(tmp_path):
    cache = _SyntheticCache(rows=30)
    prepared = _prepared_targets(rows=30)
    partition = next(iter(prepared.partitions.values()))
    arrays = dict(partition.arrays)
    arrays["identity_digest"][:] = cache.identity_digests
    arrays["label"][:] = np.arange(30, dtype=np.uint8) % 15
    prepared = PreparedTargetGeneration(
        bank_kind=prepared.bank_kind,
        partitions=MappingProxyType({
            "source-0007": PreparedTargetPartition(
                arrays=MappingProxyType(arrays),
                runtime_audit=partition.runtime_audit,
                teacher_forward_calls=partition.teacher_forward_calls,
            ),
        }),
        partition_specs=prepared.partition_specs,
        class_counts=tuple([2] * 15),
        identity_order_sha256=identity_order_sha256(cache.identity_digests),
        identity_set_sha256=identity_set_sha256(cache.identity_digests),
        population_rows_sha256=prepared.population_rows_sha256,
        canonical_batches=prepared.canonical_batches,
        teacher_forward_calls=prepared.teacher_forward_calls,
        construction_seconds=prepared.construction_seconds,
    )
    bank = EphemeralRepresentationTargetBank.from_prepared(
        prepared, strategy="RSET", carrier_node_id="U000",
        carrier_report_sha256="e" * 64, carrier_checkpoint_sha256="f" * 64,
        campaign_spec_sha256=SHA, graph_sha256="b" * 64,
        recipe_sha256="c" * 64,
    )
    probabilities = np.full((30, 15), 1 / 15, dtype=np.float32)
    targets = Tri60ProbabilityTargets(
        identities=cache.identity_digests, probabilities=probabilities,
        manifest=MappingProxyType({"temperature": 2.0}),
        _lookup={bytes(row): index for index, row in enumerate(cache.identity_digests)},
    )
    bundle = generate_spectral_resource_bundle()
    report = train_tri60_node(
        node_id="RSET_U100_from_U000", train_cache=cache,
        validation_cache=cache, input_key="hlt",
        probability_targets=targets, representation_targets=bank,
        representation_audit_sha256="d" * 64,
        token_resources=bundle.token, relation_resources=bundle.relation,
        output_dir=tmp_path, parents={"foundation": SHA},
        campaign_spec_sha256="b" * 64, recipe_sha256="c" * 64,
        execution_source_commit="d" * 40, replicate_seed=1337,
        device="cpu", runtime=Tri60TrainingRuntime(passes=2, batch_size=30),
        execution_mode="synthetic_test",
        model_factory=_tiny_scouting_factory,
    )
    assert report["complete"] is True
    assert report["ephemeral_representation_target_bytes"] == bank.nbytes
    assert report["rolling_resume_published"] is False
    assert (tmp_path / "calibration/jet_set_after_pass_2.json").is_file()
    assert not list(tmp_path.rglob("*resume*"))


def test_exact_monitor_cancellation_and_failed_downstream_closure(tmp_path):
    tasks = campaign_tasks()
    commands = {row["task_id"]: ["sbatch", row["task_id"]] for row in tasks}
    jobs = {row["task_id"]: str(90000 + index) for index, row in enumerate(tasks)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=SHA, jobs=jobs, commands=commands, dry_run=False,
    )
    states = {job: "PENDING" for job in jobs.values()}
    states[jobs["train_RSET_U100_from_U000"]] = "FAILED"
    subject = {"content_hash": SHA}
    monitor = build_monitor(
        subject=subject, ledger=ledger, states_by_job_id=states,
        attestation_root=tmp_path,
    )
    validate_monitor(
        monitor, subject_sha256=SHA, ledger_sha256=ledger["content_hash"],
    )
    closure = failed_downstream_closure(("train_RSET_U100_from_U000",))
    assert closure == (
        "train_RSET_U100_from_U000", "reduce_RSET_U100E",
        "train_RSET_D050_from_U100E", "reduce_RSET_D050E",
        "train_RSET_D000_from_U100E", "train_RSET_D000_from_D050E",
        "reduce_RSET_D000E", "train_M1_RSET", "reduce_M1E",
        "train_M2", "aggregate", "finalist_lock", "campaign_complete",
    )
    cancellation = build_cancellation(
        ledger=ledger, monitor=monitor, task_ids=closure, executed=False,
    )
    assert cancellation["job_ids"] == [jobs[task] for task in closure]
    assert cancellation["exact_ids_only"] is True
    assert cancellation["rows"][0]["state_category"] == "terminal"


def test_recovery_preserves_active_parent_from_an_independent_track():
    from hlt_classification.scouting import hcwdl_mhpe_tri60_recovery as recovery

    task = {
        "dependencies": [
            "train_M1_LOGIT", "train_M1_RSET", "train_M1_RREL",
        ],
    }
    monitor_rows = {
        "train_M1_LOGIT": {"disposition": "active_or_unknown"},
        "train_M1_RSET": {"disposition": "retryable_failure"},
        "train_M1_RREL": {"disposition": "retryable_failure"},
    }
    dependencies, subject_dependencies, dependency_jobs = (
        recovery._recovery_dependency_plan(
            task=task,
            closure={"train_M1_RSET", "train_M1_RREL"},
            monitor_rows=monitor_rows,
            subject_jobs={
                "train_M1_LOGIT": "90699",
                "train_M1_RSET": "90700",
                "train_M1_RREL": "90701",
            },
        )
    )

    assert dependencies == ["train_M1_RSET", "train_M1_RREL"]
    assert subject_dependencies == [
        {"task_id": "train_M1_LOGIT", "job_id": "90699"},
    ]
    assert dependency_jobs == [
        "90699", "${JOB_train_M1_RSET}", "${JOB_train_M1_RREL}",
    ]

    inherited = recovery._recovery_dependency_plan(
        task=task,
        closure={"train_M1_RSET", "train_M1_RREL"},
        monitor_rows={
            "train_M1_RSET": {"disposition": "retryable_failure"},
            "train_M1_RREL": {"disposition": "retryable_failure"},
        },
        subject_jobs={
            "train_M1_RSET": "90800", "train_M1_RREL": "90801",
        },
        inherited_subject_dependencies={"train_M1_LOGIT": "90699"},
    )
    assert inherited == (
        ["train_M1_RSET", "train_M1_RREL"],
        [{"task_id": "train_M1_LOGIT", "job_id": "90699"}],
        ["90699", "${JOB_train_M1_RSET}", "${JOB_train_M1_RREL}"],
    )

    completed = recovery._recovery_dependency_plan(
        task={"dependencies": ["preflight", "train_U000"]},
        closure={"train_U000"},
        monitor_rows={
            "train_U000": {"disposition": "retryable_failure"},
        },
        subject_jobs={"train_U000": "90900"},
        completed_external_dependencies={"authenticate", "preflight"},
    )
    assert completed == (["train_U000"], [], ["${JOB_train_U000}"])


def test_completed_dependency_tasks_walks_recovery_ancestry(tmp_path: Path):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_recovery as recovery
    from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
        RECOVERY_SPEC_CONTRACT, artifact,
    )

    parent = artifact({
        "completed_task_attestations": [{"task_id": "preflight"}],
        "parent_recovery_spec_path": None,
    }, contract=RECOVERY_SPEC_CONTRACT)
    parent_path = tmp_path / "parent.json"
    write_immutable_json(parent_path, parent)
    child = artifact({
        "completed_task_attestations": [{"task_id": "train_U000"}],
        "parent_recovery_spec_path": str(parent_path),
    }, contract=RECOVERY_SPEC_CONTRACT)

    assert recovery._completed_dependency_tasks(child) == {
        "preflight", "train_U000",
    }


def test_subject_dependency_registry_accepts_legacy_missing_field(
    tmp_path: Path,
):
    from hlt_classification.scouting import hcwdl_mhpe_tri60_recovery as recovery
    from hlt_classification.scouting.hcwdl_mhpe_tri60_contracts import (
        COMMAND_PLAN_CONTRACT, artifact,
    )

    plan = artifact({
        "spec_sha256": SHA,
        "commands": [{
            "task_id": "train_U000", "dependencies": [],
            "command": ["sbatch", "worker.sh"],
        }],
        "mutated": False, "recovery": False,
        "final_test_accessed": False,
    }, contract=COMMAND_PLAN_CONTRACT)
    write_immutable_json(tmp_path / "command_plan.json", plan)

    assert recovery._subject_dependency_rows(
        tmp_path, allowed_tasks=("train_U000",),
    ) == {"train_U000": {}}
