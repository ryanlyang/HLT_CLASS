from __future__ import annotations

from fractions import Fraction
import numpy as np
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256
from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_mhpe_contracts import (
    CAMPAIGN_SPEC_CONTRACT_C10P90,
    CAMPAIGN_SPEC_CONTRACT_C10P90_300K60,
    CAMPAIGN_SPEC_CONTRACT_C25P75_300K60, RECIPE_CONTRACT_C10P90,
    CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
    CAMPAIGN_SPEC_CONTRACT_DENSE_C25P75_300K60,
    campaign_profile, execution_lock_payload, graph_payload, recipe_payload,
    validate_execution_lock, validate_graph, validate_recipe, validate_waiver,
    waiver_payload,
)
from hlt_classification.scouting.hcwdl_mhpe_campaign import (
    campaign_tasks, command_plan, create_campaign, submission_phrase,
    validate_campaign,
)
from hlt_classification.scouting.hcwdl_mhpe_contracts import reuse_lock_payload
from hlt_classification.scouting.hcwdl_mhpe_graph import (
    C10P90_GRAPH_SHA256, C10P90_NODE_REGISTRY, COORDINATES,
    ENSEMBLE_COMPONENTS, GRAPH_SHA256, NODE_REGISTRY, PROFILE_C10P90,
    PROFILE_C10P90_300K60, PROFILE_C25P75_300K60,
    PROFILE_DENSE_ANCHOR50_300K60, DENSE_FINALISTS,
    PROFILE_DENSE_C25P75_300K60,
    ensemble_components, ensemble_weight_rationals, node_registry,
    stage_teachers,
    validate_graph as validate_registry,
)
from hlt_classification.scouting.hcwdl_mhpe_targets import (
    DurableProbabilityTargets, load_probability_shard,
    publish_probability_manifest, publish_probability_shard,
    target_lock_payload, uniform_probability_ensemble,
    validate_probability_bundle, weighted_probability_ensemble,
)
from hlt_classification.scouting.hcwdl_mhpe_recovery import failed_downstream_closure
from hlt_classification.scouting.targets import EphemeralProbabilityTargets
from hlt_classification.scouting.training import (
    GenerationalLossConfiguration, generational_pmard_loss,
)


def test_exact_graph_shape_and_routes():
    assert GRAPH_SHA256 == "3399cdf7f19e3461b9f5cfdcee2e38257a567d5bdb8547b8deb9dbddd856daf9"
    assert validate_registry() == GRAPH_SHA256
    assert len(NODE_REGISTRY) == 16
    assert sum(name.startswith("D000_from_") for name in NODE_REGISTRY) == 5
    assert ENSEMBLE_COMPONENTS["U100E"] == (
        "U100_from_U000", "U100_from_U050",
    )
    assert NODE_REGISTRY["M1"].teacher_kind == "probabilities"
    assert NODE_REGISTRY["M1"].temperature == 1
    assert validate_graph(graph_payload()) == graph_payload()["content_hash"]
    assert COORDINATES["U050"].payload()["structural"] == [1, 2]
    assert COORDINATES["U100"].payload()["structural"] == [1, 1]
    assert COORDINATES["D066"].payload()["feature"] == [1, 3]
    assert COORDINATES["D033"].payload()["feature"] == [2, 3]
    assert COORDINATES["D000"].payload()["feature"] == [1, 1]


def test_c10p90_graph_changes_only_specialist_loss_contract():
    assert validate_registry(PROFILE_C10P90) == C10P90_GRAPH_SHA256
    assert set(C10P90_NODE_REGISTRY) == set(NODE_REGISTRY)
    for node_id, primary in NODE_REGISTRY.items():
        parallel = C10P90_NODE_REGISTRY[node_id]
        assert (
            parallel.coordinate_name, parallel.teacher_id, parallel.seed_alias,
            parallel.temperature, parallel.teacher_kind,
        ) == (
            primary.coordinate_name, primary.teacher_id, primary.seed_alias,
            primary.temperature, primary.teacher_kind,
        )
        if node_id == "M1":
            assert (parallel.ce_weight, parallel.kd_weight) == (.10, .90)
            assert (primary.ce_weight, primary.kd_weight) == (.10, .90)
        else:
            assert (parallel.ce_weight, parallel.kd_weight) == (.10, .90)
            assert (primary.ce_weight, primary.kd_weight) == (.25, .75)
        assert parallel.contract.endswith("/v2")
    payload = graph_payload(PROFILE_C10P90)
    assert payload["recipe_profile"] == PROFILE_C10P90
    assert validate_graph(payload) == payload["content_hash"]


def test_paired_300k60_graphs_change_only_specialist_loss():
    primary = node_registry(PROFILE_C25P75_300K60)
    parallel = node_registry(PROFILE_C10P90_300K60)
    assert set(primary) == set(parallel) == set(NODE_REGISTRY)
    for node_id in primary:
        left, right = primary[node_id], parallel[node_id]
        assert left.training_passes == right.training_passes == 60
        assert left.seed_alias == right.seed_alias
        assert left.coordinate == right.coordinate
        assert left.teacher_id == right.teacher_id
        assert left.teacher_kind == right.teacher_kind
        assert left.temperature == right.temperature
        if node_id == "M1":
            assert (left.ce_weight, left.kd_weight) == (.10, .90)
            assert (right.ce_weight, right.kd_weight) == (.10, .90)
        else:
            assert (left.ce_weight, left.kd_weight) == (.25, .75)
            assert (right.ce_weight, right.kd_weight) == (.10, .90)
    for profile in (PROFILE_C25P75_300K60, PROFILE_C10P90_300K60):
        graph = graph_payload(profile)
        recipe = recipe_payload(foundation_recipe_sha256="a" * 64, profile=profile)
        assert validate_graph(graph) == graph["content_hash"]
        assert validate_recipe(recipe) == recipe["content_hash"]
        assert recipe["training_passes"] == 60
        assert recipe["population_profile"] == "pilot_300k_60pass"


def test_dense_anchor50_graph_exact_shape_routes_and_weights():
    profile = PROFILE_DENSE_ANCHOR50_300K60
    registry = node_registry(profile)
    components = ensemble_components(profile)
    assert len(registry) == 29
    assert tuple(components) == (
        "U066E", "U100E", "D75E", "D50E", "D25E", "D0E",
    )
    assert sum(name.startswith("D0_from_") for name in registry) == 7
    assert registry["M1"].teacher_id == "D0E"
    assert registry["M1"].temperature == 1
    assert all(node.training_passes == 60 for node in registry.values())
    assert all(
        (node.ce_weight, node.kd_weight) == (.10, .90)
        for node in registry.values()
    )
    assert COORDINATES["U033"].payload()["structural"] == [1, 3]
    assert COORDINATES["U066"].payload()["structural"] == [2, 3]
    assert COORDINATES["D75"].payload()["feature"] == [1, 4]
    assert COORDINATES["D50"].payload()["feature"] == [1, 2]
    assert COORDINATES["D25"].payload()["feature"] == [3, 4]
    assert registry["D0_from_D25E"].input_domain == "hlt"
    assert len(DENSE_FINALISTS) == 10
    for ensemble_id, names in components.items():
        weights = ensemble_weight_rationals(profile, ensemble_id)
        local = {
            "U066E": "U066_from_U033", "U100E": "U100_from_U066E",
            "D75E": "D75_from_U100E", "D50E": "D50_from_D75E",
            "D25E": "D25_from_D50E", "D0E": "D0_from_D25E",
        }[ensemble_id]
        assert weights[local] == [1, 2]
        assert sum(Fraction(*weights[name]) for name in names) == 1
    graph = graph_payload(profile)
    recipe = recipe_payload(foundation_recipe_sha256="a" * 64, profile=profile)
    assert validate_graph(graph) == graph["content_hash"]
    assert validate_recipe(recipe) == recipe["content_hash"]
    assert graph["fresh_fit_count"] == 29
    assert recipe["ensemble"]["weights"] == (
        "local_predecessor_half_skip_half_exact_rational"
    )


def test_dense_c25p75_is_a_loss_only_pair_of_dense_c10p90():
    primary = PROFILE_DENSE_ANCHOR50_300K60
    parallel = PROFILE_DENSE_C25P75_300K60
    left = node_registry(primary)
    right = node_registry(parallel)
    assert set(left) == set(right)
    assert ensemble_components(primary) == ensemble_components(parallel)
    for ensemble_id in ensemble_components(primary):
        assert ensemble_weight_rationals(primary, ensemble_id) == (
            ensemble_weight_rationals(parallel, ensemble_id)
        )
    for node_id in left:
        old, new = left[node_id], right[node_id]
        assert (
            old.coordinate_name, old.teacher_id, old.teacher_kind,
            old.seed_alias, old.temperature, old.training_passes,
        ) == (
            new.coordinate_name, new.teacher_id, new.teacher_kind,
            new.seed_alias, new.temperature, new.training_passes,
        )
        if node_id == "M1":
            assert (old.ce_weight, old.kd_weight) == (.10, .90)
            assert (new.ce_weight, new.kd_weight) == (.10, .90)
        else:
            assert (old.ce_weight, old.kd_weight) == (.10, .90)
            assert (new.ce_weight, new.kd_weight) == (.25, .75)
        assert old.contract.endswith("/v5")
        assert new.contract.endswith("/v6")
    graph = graph_payload(parallel)
    recipe = recipe_payload(foundation_recipe_sha256="a" * 64, profile=parallel)
    assert validate_graph(graph) == graph["content_hash"]
    assert validate_recipe(recipe) == recipe["content_hash"]
    assert graph["fresh_fit_count"] == 29
    assert recipe["specialist_loss"] == {
        "ce": .25, "kd": .75, "temperature": 2.0,
    }
    assert recipe["m1_loss"] == {"ce": .10, "kd": .90, "temperature": 1.0}
    assert recipe["paired_study"] == "dense_specialist_ce_kd_weights_only"


def test_foundation_reuse_dispatch_keeps_legacy_and_300k_paths_separate(
    monkeypatch, tmp_path,
):
    from hlt_classification.scouting import hcwdl_mhpe_campaign as campaign

    calls = []
    monkeypatch.setattr(
        campaign, "_reuse_full",
        lambda **kwargs: calls.append(("full", kwargs)) or {"path": "full"},
    )
    monkeypatch.setattr(
        campaign, "_reuse_300k",
        lambda **kwargs: calls.append(("300k", kwargs)) or {"path": "300k"},
    )
    common = {
        "foundation_lock": tmp_path / "lock.json",
        "project": tmp_path,
        "source_commit": "a" * 40,
    }
    assert campaign._reuse(**common, profile="C25P75") == {"path": "full"}
    assert campaign._reuse(
        **common, profile=PROFILE_C25P75_300K60,
    ) == {"path": "300k"}
    assert campaign._reuse(
        **common, profile=PROFILE_C10P90_300K60,
    ) == {"path": "300k"}
    assert campaign._reuse(
        **common, profile=PROFILE_DENSE_ANCHOR50_300K60,
    ) == {"path": "300k"}
    assert campaign._reuse(
        **common, profile=PROFILE_DENSE_C25P75_300K60,
    ) == {"path": "300k"}
    assert [row[0] for row in calls] == [
        "full", "300k", "300k", "300k", "300k",
    ]


def test_population_specific_runtime_seed_and_cache_limits(tmp_path):
    from hlt_classification.scouting.hcwdl_mhpe_runner import (
        _runtime_parameters, _training_recipe,
    )

    assert _runtime_parameters("C25P75") == ("ub_full/repair/v1", 224.0)
    assert _runtime_parameters("C10P90") == ("ub_full/repair/v1", 224.0)
    assert _runtime_parameters(PROFILE_C25P75_300K60) == (
        "ub/repair/v1", 72.0,
    )
    assert _runtime_parameters(PROFILE_C10P90_300K60) == (
        "ub/repair/v1", 72.0,
    )
    assert _runtime_parameters(PROFILE_DENSE_ANCHOR50_300K60) == (
        "ub/repair/v1", 72.0,
    )
    assert _runtime_parameters(PROFILE_DENSE_C25P75_300K60) == (
        "ub/repair/v1", 72.0,
    )

    # The 300k foundation root contains the UB scientific overlay, while its
    # authenticated artifact registry points at the executable HCWDL recipe.
    # Loading the overlay here caused the real jobs to fail on recipe["batching"].
    overlay = with_content_hash({
        "contract": "overlay", "schema_version": 1,
        "training_passes": 60,
    })
    executable = with_content_hash({
        "contract": "executable", "schema_version": 1,
        "batching": {"effective_batch_size": 256},
    })
    write_immutable_json(tmp_path / "recipe.json", overlay)
    write_immutable_json(tmp_path / "executable_recipe.json", executable)
    foundation = {
        "artifact_paths": {"recipe": str(tmp_path / "executable_recipe.json")},
    }
    assert _training_recipe(
        profile=PROFILE_C25P75_300K60,
        foundation_root=tmp_path,
        foundation=foundation,
    ) == executable
    assert _training_recipe(
        profile=PROFILE_C10P90_300K60,
        foundation_root=tmp_path,
        foundation=foundation,
    ) == executable
    assert _training_recipe(
        profile=PROFILE_DENSE_ANCHOR50_300K60,
        foundation_root=tmp_path,
        foundation=foundation,
    ) == executable
    assert _training_recipe(
        profile=PROFILE_DENSE_C25P75_300K60,
        foundation_root=tmp_path,
        foundation=foundation,
    ) == executable
    assert _training_recipe(
        profile="C25P75", foundation_root=tmp_path, foundation=foundation,
    ) == overlay


def test_task_graph_has_stage_parallelism_and_exact_closure():
    tasks = {row["task_id"]: row for row in campaign_tasks()}
    assert len([row for row in tasks.values() if row["kind"] == "train"]) == 16
    assert tasks["train_U050_from_U000"]["dependencies"] == []
    assert tasks["train_U100_from_U000"]["dependencies"] == ["train_U050_from_U000"]
    assert tasks["ensemble_U100E"]["dependencies"] == [
        "train_U100_from_U000", "train_U100_from_U050",
    ]
    assert all(tasks[f"train_D000_from_{teacher}"]["dependencies"] == ["ensemble_D033E"] for teacher in ("U000", "U050", "U100E", "D066E", "D033E"))
    closure = failed_downstream_closure(["train_D066_from_U050"])
    assert "ensemble_D066E" in closure and "train_M1" in closure and "train_U100_from_U000" not in closure


def test_dense_anchor50_task_graph_is_38_jobs_and_exactly_sequential_by_stage():
    profile = PROFILE_DENSE_ANCHOR50_300K60
    tasks = {row["task_id"]: row for row in campaign_tasks(profile)}
    assert len(tasks) == 38
    assert sum(row["kind"] == "train" for row in tasks.values()) == 29
    assert sum(row["kind"] == "ensemble" for row in tasks.values()) == 6
    assert tasks["train_U033_from_U000"]["dependencies"] == []
    assert tasks["train_U066_from_U000"]["dependencies"] == [
        "train_U033_from_U000"
    ]
    assert tasks["ensemble_U066E"]["dependencies"] == [
        "train_U066_from_U000", "train_U066_from_U033",
    ]
    assert tasks["train_U100_from_U000"]["dependencies"] == ["ensemble_U066E"]
    assert tasks["train_M1"]["dependencies"] == ["ensemble_D0E"]
    assert tasks["aggregate"]["dependencies"] == [
        *(f"train_{name}" for name in node_registry(profile)),
        *(f"ensemble_{name}" for name in ensemble_components(profile)),
    ]
    closure = failed_downstream_closure(
        ["train_D50_from_U033"], profile=profile,
    )
    assert "ensemble_D50E" in closure
    assert "train_M1" in closure
    assert "ensemble_D75E" not in closure

    spec = {
        "contract": CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
        "recipe_profile": profile,
        "population_profile": "pilot_300k_60pass",
        "project_dir": "/project", "spec_path": "/campaign/spec.json",
        "content_hash": "a" * 64, "tasks": list(tasks.values()),
        "resources": {
            "gpu_training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
            "gpu_targets": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
            "cpu_report": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": None},
        },
    }
    plan = command_plan(spec)
    assert len(plan["commands"]) == 38
    assert all(
        any(item.startswith("--job-name=hcwmhped_") for item in row["command"])
        for row in plan["commands"]
    )


def test_dense_c25p75_task_graph_and_namespace_are_independent():
    profile = PROFILE_DENSE_C25P75_300K60
    tasks = campaign_tasks(profile)
    assert len(tasks) == 38
    assert sum(row["kind"] == "train" for row in tasks) == 29
    assert sum(row["kind"] == "ensemble" for row in tasks) == 6
    assert tasks == campaign_tasks(PROFILE_DENSE_ANCHOR50_300K60)
    spec = {
        "contract": CAMPAIGN_SPEC_CONTRACT_DENSE_C25P75_300K60,
        "recipe_profile": profile,
        "population_profile": "pilot_300k_60pass",
        "project_dir": "/project", "spec_path": "/campaign/spec.json",
        "content_hash": "a" * 64, "tasks": tasks,
        "resources": {
            "gpu_training": {
                "cpus": 8, "memory": "96G", "walltime": "06:00:00",
                "gpu": "gpu:gh200:1",
            },
            "gpu_targets": {
                "cpus": 8, "memory": "96G", "walltime": "06:00:00",
                "gpu": "gpu:gh200:1",
            },
            "cpu_report": {
                "cpus": 4, "memory": "32G", "walltime": "01:00:00",
                "gpu": None,
            },
        },
    }
    plan = command_plan(spec)
    assert len(plan["commands"]) == 38
    assert all(
        any(item.startswith("--job-name=hcwmhpe25d_") for item in row["command"])
        for row in plan["commands"]
    )
    closure = failed_downstream_closure(
        ["train_D50_from_U033"], profile=profile,
    )
    assert "ensemble_D50E" in closure
    assert "train_M1" in closure
    assert "ensemble_D75E" not in closure


def test_recipe_is_exactly_locked():
    parent = "a" * 64
    recipe = recipe_payload(foundation_recipe_sha256=parent)
    assert validate_recipe(recipe) == recipe["content_hash"]
    assert recipe["specialist_loss"] == {"ce": .25, "kd": .75, "temperature": 2.0}
    assert recipe["m1_loss"] == {"ce": .10, "kd": .90, "temperature": 1.0}
    parallel = recipe_payload(
        foundation_recipe_sha256=parent, profile=PROFILE_C10P90,
    )
    assert parallel["contract"] == RECIPE_CONTRACT_C10P90
    assert parallel["specialist_loss"] == {"ce": .10, "kd": .90, "temperature": 2.0}
    assert parallel["m1_loss"] == recipe["m1_loss"]
    assert parallel["single_changed_variable"] == "specialist_ce_kd_weights_only"
    assert validate_recipe(parallel) == parallel["content_hash"]

    waiver = waiver_payload(
        source_commit="a" * 40, graph_sha256="b" * 64,
        reuse_lock_sha256="c" * 64, recipe_sha256=parallel["content_hash"],
        semantic_source_registry_sha256="d" * 64,
        resource_request_sha256="e" * 64,
        implementation_evidence_sha256={"test": "f" * 64},
        authorization_phrase=(
            "AUTHORIZE HCWDL MHPE C10P90 FULL DIRECT EXECUTION WITHOUT NEW SMOKE"
        ),
        profile=PROFILE_C10P90,
    )
    assert validate_waiver(waiver) == waiver["content_hash"]
    tampered = dict(waiver)
    tampered.pop("content_hash")
    tampered["single_changed_variable"] = "wrong"
    with pytest.raises(ValueError, match="waiver identity"):
        validate_waiver(with_content_hash(tampered))


def test_final_test_requires_exact_human_execution_phrase():
    with pytest.raises(PermissionError):
        execution_lock_payload(
            campaign_spec_sha256="a" * 64, finalist_lock_sha256="b" * 64,
            source_commit="c" * 40, authorization_phrase="wrong",
        )
    lock = execution_lock_payload(
        campaign_spec_sha256="a" * 64, finalist_lock_sha256="b" * 64,
        source_commit="c" * 40,
        authorization_phrase="AUTHORIZE HCWDL MHPE SEALED FINAL TEST",
    )
    assert validate_execution_lock(lock) == lock["content_hash"]


def test_uniform_ensemble_is_probability_not_logit_average():
    logits = {
        "b": np.array([[8.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32),
        "a": np.array([[0.0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32),
    }
    actual = uniform_probability_ensemble(logits, temperature=2)
    probability_mean = sum(
        np.exp((value / 2) - (value / 2).max(axis=1, keepdims=True), dtype=np.float32)
        / np.exp((value / 2) - (value / 2).max(axis=1, keepdims=True), dtype=np.float32).sum(axis=1, keepdims=True)
        for value in logits.values()
    ) / 2
    assert np.allclose(actual, probability_mean, atol=2e-7)
    logit_mean = np.mean(list(logits.values()), axis=0)
    logit_probability = np.exp(logit_mean / 2) / np.exp(logit_mean / 2).sum(axis=1, keepdims=True)
    assert not np.allclose(actual, logit_probability, atol=1e-3)


@pytest.mark.parametrize("components", [2, 3, 4, 5])
def test_uniform_ensemble_is_order_invariant_for_every_stage_width(components):
    values = {
        f"node_{index}": np.asarray(
            [[(index + 1) * (class_index - 7) / 10 for class_index in range(15)]],
            dtype=np.float32,
        )
        for index in range(components)
    }
    forward = uniform_probability_ensemble(values, temperature=2)
    reverse = uniform_probability_ensemble(dict(reversed(list(values.items()))), temperature=2)
    assert np.array_equal(forward, reverse)
    manual = np.zeros((1, 15), np.float64)
    for name in sorted(values):
        scaled = np.asarray(values[name] / np.float32(2), np.float32)
        exponent = np.exp(np.asarray(scaled - scaled.max(1, keepdims=True), np.float32), dtype=np.float32)
        manual += np.asarray(exponent / exponent.sum(1, keepdims=True, dtype=np.float32), np.float32).astype(np.float64)
    assert np.array_equal(forward, np.asarray(manual / components, dtype="<f4"))


def test_probability_target_kl_matches_direct_formula_and_is_not_retempered():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[1.2, -.3] + [0.0] * 13], requires_grad=True)
    target = torch.tensor([[.7, .2] + [.1 / 13] * 13])
    config = GenerationalLossConfiguration(
        arm="HCWDL_UB_MHPE_TEST", ce=0, parent_kd=1,
        grandparent_kd=0, parent_temperature=2, grandparent_temperature=2,
    )
    parts = generational_pmard_loss(
        student, torch.tensor([0]), class_weights=torch.ones(15), configuration=config,
        parent_teacher_probabilities=target,
    )
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student.float() / 2, dim=-1),
        target, reduction="batchmean",
    ) * 4
    assert torch.allclose(parts["total"], expected)
    parts["total"].backward()
    assert torch.isfinite(student.grad).all()


def test_legacy_logit_kd_path_remains_exactly_the_original_formula():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[.8, -.2] + [0.0] * 13], requires_grad=True)
    teacher = torch.tensor([[-.4, .9] + [0.0] * 13])
    config = GenerationalLossConfiguration(
        arm="HCWDL_UB_LEGACY_REGRESSION", ce=0, parent_kd=1,
        grandparent_kd=0, parent_temperature=2, grandparent_temperature=2,
    )
    actual = generational_pmard_loss(
        student, torch.tensor([0]), class_weights=torch.ones(15),
        configuration=config, parent_teacher_logits=teacher,
    )["total"]
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student.float() / 2, dim=-1),
        torch.nn.functional.softmax(teacher.float() / 2, dim=-1),
        reduction="batchmean",
    ) * 4
    assert torch.equal(actual, expected)


def test_ephemeral_probability_identity_and_temperature_contract():
    probabilities = np.full((2, 15), 1 / 15, np.float32)
    table = EphemeralProbabilityTargets.create(
        ["a", "b"], probabilities, target_manifest_sha256=canonical_sha256("target"),
        split_manifest_sha256=canonical_sha256("split"), temperature=1,
    )
    assert table.temperature == 1
    assert np.array_equal(table.join(["b", "a"]), probabilities)
    with pytest.raises(KeyError):
        table.join(["missing"])


@pytest.mark.parametrize("profile", [
    PROFILE_DENSE_ANCHOR50_300K60,
    PROFILE_DENSE_C25P75_300K60,
])
def test_anchor50_weighted_probability_reduction_is_exact_and_not_uniform(profile):
    ensemble_id = "D0E"
    names = ensemble_components(profile)[ensemble_id]
    logits = {
        name: np.full((3, 15), -4.0, np.float32) for name in names
    }
    for index, name in enumerate(names):
        logits[name][:, index] = 4.0
    weights = ensemble_weight_rationals(profile, ensemble_id)
    weighted = weighted_probability_ensemble(
        logits, temperature=2, weights=weights,
    )
    uniform = uniform_probability_ensemble(logits, temperature=2)
    local = names.index("D0_from_D25E")
    assert weighted.dtype.str == "<f4"
    assert np.allclose(weighted.sum(1), 1, rtol=0, atol=2e-6)
    assert np.all(weighted[:, local] > uniform[:, local])
    assert not np.array_equal(weighted, uniform)
    with pytest.raises(ValueError, match="sum to one"):
        weighted_probability_ensemble(
            logits, temperature=2,
            weights={name: [1, len(names) + 1] for name in names},
        )


@pytest.mark.parametrize("profile", [
    PROFILE_DENSE_ANCHOR50_300K60,
    PROFILE_DENSE_C25P75_300K60,
])
def test_dense_anchor50_complete_synthetic_graph_executes_all_routes(profile):
    teachers = {"U000": np.linspace(-1, 1, 60, dtype=np.float32).reshape(4, 15)}
    specialists = {}
    ensembles = {}
    for stage, teacher_ids in stage_teachers(profile).items():
        for teacher_index, teacher_id in enumerate(teacher_ids):
            base = teachers[teacher_id]
            node_id = f"{stage}_from_{teacher_id}"
            specialists[node_id] = np.asarray(
                base + np.float32((teacher_index + 1) / 1000), dtype=np.float32,
            )
        if stage == "U033":
            teachers["U033"] = specialists["U033_from_U000"]
            continue
        ensemble_id = f"{stage}E"
        probability = weighted_probability_ensemble(
            {name: specialists[name] for name in ensemble_components(profile)[ensemble_id]},
            temperature=2,
            weights=ensemble_weight_rationals(profile, ensemble_id),
        )
        ensembles[ensemble_id] = probability
        teachers[ensemble_id] = np.log(np.maximum(probability, 1e-30)).astype(np.float32)
    specialists["M1"] = np.log(np.maximum(ensembles["D0E"], 1e-30)).astype(np.float32)
    assert set(specialists) == set(node_registry(profile))
    assert set(ensembles) == set(ensemble_components(profile))
    assert all(np.allclose(value.sum(1), 1, atol=2e-6) for value in ensembles.values())


def test_probability_artifact_roundtrip_and_lineage(tmp_path):
    identities = ["a", "b"]
    components = ENSEMBLE_COMPONENTS["U100E"]
    logits = {name: np.arange(30, dtype=np.float32).reshape(2, 15) * (index + 1) / 100 for index, name in enumerate(components)}
    lineage = {name: {"report_sha256": canonical_sha256(name + "r"), "checkpoint_sha256": canonical_sha256(name + "c"), "logits_sha256": canonical_sha256(name + "l")} for name in components}
    _, metadata = publish_probability_shard(
        tmp_path / "shard", ensemble_id="U100E", role="train", identities=identities,
        component_logits=logits, component_lineage=lineage, temperature=2,
        source_path="all", parents={"campaign": "a" * 64}, producer_commit="b" * 40,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = publish_probability_manifest(
        manifest_path, ensemble_id="U100E", role="train", shard_paths=[metadata],
        expected_sources=["all"], expected_rows=2, temperature=2,
        consumers=["D066_from_U100E"], parents={"campaign": "a" * 64},
    )
    durable = DurableProbabilityTargets(manifest_path)
    ephemeral = durable.as_ephemeral(split_manifest_sha256="c" * 64)
    assert ephemeral.temperature == 2
    assert ephemeral.header["target_manifest_sha256"] == manifest["content_hash"]
    assert np.array_equal(ephemeral.join(["b", "a"]), durable.probabilities[[1, 0]])


@pytest.mark.parametrize("profile", [
    PROFILE_DENSE_ANCHOR50_300K60,
    PROFILE_DENSE_C25P75_300K60,
])
def test_dense_anchor50_probability_artifact_binds_profile_and_weights(
    tmp_path, profile,
):
    ensemble_id = "U066E"
    components = ensemble_components(profile)[ensemble_id]
    logits = {
        name: np.arange(30, dtype=np.float32).reshape(2, 15) * (index + 1) / 100
        for index, name in enumerate(components)
    }
    lineage = {
        name: {
            "report_sha256": canonical_sha256(name + "r"),
            "checkpoint_sha256": canonical_sha256(name + "c"),
            "logits_sha256": canonical_sha256(name + "l"),
        }
        for name in components
    }
    _, metadata_path = publish_probability_shard(
        tmp_path / "dense", ensemble_id=ensemble_id, role="train",
        identities=["a", "b"], component_logits=logits,
        component_lineage=lineage, temperature=2, source_path="all",
        parents={"campaign": "a" * 64}, producer_commit="b" * 40,
        profile=profile, weights=ensemble_weight_rationals(profile, ensemble_id),
    )
    metadata, arrays = load_probability_shard(metadata_path)
    assert metadata["recipe_profile"] == profile
    assert metadata["ensemble_weights"] == {
        "U066_from_U000": [1, 2], "U066_from_U033": [1, 2],
    }
    expected = weighted_probability_ensemble(
        logits, temperature=2, weights=metadata["ensemble_weights"],
    )
    assert np.array_equal(arrays["probabilities"], expected)


def test_probability_artifacts_reject_duplicate_identity_temperature_and_corruption(tmp_path):
    components = ENSEMBLE_COMPONENTS["U100E"]
    logits = {
        name: np.zeros((2, 15), np.float32) + index
        for index, name in enumerate(components)
    }
    lineage = {
        name: {
            "report_sha256": canonical_sha256(name + "r"),
            "checkpoint_sha256": canonical_sha256(name + "c"),
            "logits_sha256": canonical_sha256(name + "l"),
        }
        for name in components
    }
    with pytest.raises(ValueError, match="identity"):
        publish_probability_shard(
            tmp_path / "duplicate", ensemble_id="U100E", role="train",
            identities=["same", "same"], component_logits=logits,
            component_lineage=lineage, temperature=2, source_path="all",
            parents={"campaign": "a" * 64}, producer_commit="b" * 40,
        )
    _, metadata = publish_probability_shard(
        tmp_path / "temperature", ensemble_id="U100E", role="train",
        identities=["a", "b"], component_logits=logits,
        component_lineage=lineage, temperature=2, source_path="all",
        parents={"campaign": "a" * 64}, producer_commit="b" * 40,
    )
    with pytest.raises(ValueError, match="lineage"):
        publish_probability_manifest(
            tmp_path / "bad_manifest.json", ensemble_id="U100E", role="train",
            shard_paths=[metadata], expected_sources=["all"], expected_rows=2,
            temperature=1, consumers=[], parents={"campaign": "a" * 64},
        )
    npz = tmp_path / "temperature.npz"
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes"):
        load_probability_shard(metadata)


def test_probability_bundle_binds_both_roles_and_exact_consumers(tmp_path):
    components = ENSEMBLE_COMPONENTS["U100E"]
    lineage = {
        name: {
            "report_sha256": canonical_sha256(name + "r"),
            "checkpoint_sha256": canonical_sha256(name + "c"),
            "logits_sha256": canonical_sha256(name + "l"),
        }
        for name in components
    }
    parents = {"campaign": "a" * 64}
    manifests = {}
    for role in ("train", "validation"):
        _, metadata = publish_probability_shard(
            tmp_path / f"{role}_all", ensemble_id="U100E", role=role,
            identities=[f"{role}-a", f"{role}-b"],
            component_logits={name: np.zeros((2, 15), np.float32) for name in components},
            component_lineage=lineage, temperature=2, source_path=f"{role}-all",
            parents=parents, producer_commit="b" * 40,
        )
        manifest = publish_probability_manifest(
            tmp_path / f"{role}_manifest.json", ensemble_id="U100E", role=role,
            shard_paths=[metadata], expected_sources=[f"{role}-all"], expected_rows=2,
            temperature=2, consumers=["D066_from_U100E"], parents=parents,
        )
        manifests[role] = manifest["content_hash"]
    lock = target_lock_payload(
        manifests=manifests, ensemble_id="U100E",
        consumers=["D066_from_U100E"], parents=parents,
    )
    write_immutable_json(tmp_path / "lock.json", lock)
    validate_probability_bundle(
        tmp_path, ensemble_id="U100E", temperature=2,
        consumers=["D066_from_U100E"],
    )
    with pytest.raises(ValueError, match="consumers"):
        validate_probability_bundle(
            tmp_path, ensemble_id="U100E", temperature=2,
            consumers=["D033_from_U100E"],
        )


def test_bounded_synthetic_lattice_exercises_all_specialists_and_ensembles():
    rng = np.random.default_rng(1729)
    labels = rng.integers(0, 15, size=32)
    logits_by_teacher = {"U000": rng.normal(size=(32, 15)).astype(np.float32)}
    specialists = {}
    ensembles = {}
    for stage in ("U050", "U100", "D066", "D033", "D000"):
        for teacher in (node.teacher_id for node in NODE_REGISTRY.values()
                        if node.node_id.startswith(stage + "_from_")):
            source = (np.log(np.maximum(ensembles[teacher], 1e-30))
                      if teacher.endswith("E") else logits_by_teacher[teacher])
            node_id = f"{stage}_from_{teacher}"
            perturbation = (labels[:, None] == np.arange(15)[None, :]).astype(np.float32)
            specialists[node_id] = np.asarray(source + .01 * perturbation, np.float32)
        if stage == "U050":
            logits_by_teacher["U050"] = specialists["U050_from_U000"]
        else:
            ensemble_id = stage + "E"
            ensembles[ensemble_id] = uniform_probability_ensemble(
                {name: specialists[name] for name in ENSEMBLE_COMPONENTS[ensemble_id]},
                temperature=1 if ensemble_id == "D000E" else 2,
            )
    specialists["M1"] = np.log(np.maximum(ensembles["D000E"], 1e-30)).astype(np.float32)
    assert set(specialists) == set(NODE_REGISTRY)
    assert set(ensembles) == set(ENSEMBLE_COMPONENTS)
    assert all(np.isfinite(value).all() for value in specialists.values())
    assert all(np.allclose(value.sum(1), 1, atol=2e-6) for value in ensembles.values())


def test_aggregate_distinguishes_graph_artifact_and_semantic_hashes(tmp_path):
    from hlt_classification.scouting.hcwdl_mhpe_contracts import graph_payload
    from hlt_classification.scouting.hcwdl_mhpe_workflow import (
        _authenticated_semantic_graph_sha256,
    )

    graph = graph_payload(PROFILE_C25P75_300K60)
    assert graph["content_hash"] != graph["graph_sha256"]
    write_immutable_json(tmp_path / "graph.json", graph)
    spec = {"graph_sha256": graph["content_hash"]}
    assert _authenticated_semantic_graph_sha256(spec, tmp_path) == graph["graph_sha256"]

    with pytest.raises(ValueError, match="graph artifact lineage"):
        _authenticated_semantic_graph_sha256(
            {"graph_sha256": "0" * 64}, tmp_path,
        )


def test_campaign_publication_and_full_dry_run_shape(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_mhpe_campaign as campaign
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    recipe = with_content_hash({"contract": "test", "schema_version": 1})
    write_immutable_json(foundation / "recipe.json", recipe)
    reuse = reuse_lock_payload(
        foundation_spec_path=foundation / "foundation_spec.json",
        foundation_spec_sha256="1" * 64, foundation_lock_sha256="2" * 64,
        role_counts={"train": 2_600_000, "validation": 1_000_000, "final_test": 1_000_000},
        u000_report_sha256="3" * 64, u000_checkpoint_sha256="4" * 64,
        u000_target_manifest_sha256="5" * 64, m0paired_report_sha256="6" * 64,
        source_commit="a" * 40, semantic_source_sha256={"source": "7" * 64},
        foundation_parents={"foundation_recipe_sha256": recipe["content_hash"], "parent": "8" * 64},
        foundation_core_compatibility={
            "policy": "byte_exact_except_probability_target_adapter_v1",
            "byte_exact_files": {
                "model": {"foundation_sha256": "9" * 64, "current_sha256": "9" * 64},
            },
            "additive_adapter_files": {
                "src/hlt_classification/scouting/engine.py": {
                    "foundation_sha256": "a" * 64, "current_sha256": "b" * 64,
                },
                "src/hlt_classification/scouting/hcwdl_training.py": {
                    "foundation_sha256": "c" * 64, "current_sha256": "d" * 64,
                },
            },
            "legacy_logit_path_numerically_regressed": True,
            "adapter_scope": "test",
        },
    )
    monkeypatch.setattr(campaign, "_reuse", lambda **kwargs: reuse)
    monkeypatch.setattr(campaign, "semantic_source_hashes", lambda path: {"source": "7" * 64})
    root = tmp_path / "campaign"
    worktree = tmp_path / "worktree"
    for name in campaign.IMPLEMENTATION_EVIDENCE_FILES:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"evidence: {name}\n")
    spec = create_campaign(
        foundation_lock=foundation / "locks/foundation.json", campaign_root=root,
        project_dir=worktree, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase="AUTHORIZE HCWDL MHPE FULL EXACT SPEC",
    )
    assert validate_campaign(spec) == spec["content_hash"]
    plan = command_plan(spec)
    assert len(plan["commands"]) == 23
    by_task = {row["task_id"]: row for row in plan["commands"]}
    assert "--gres=gpu:gh200:1" in by_task["train_D000_from_U000"]["command"]
    assert "--mem=256G" in by_task["ensemble_D000E"]["command"]
    assert "--signal=B:USR1@120" in by_task["ensemble_D000E"]["command"]
    assert "--dependency=afterok:${JOB_ensemble_D033E}" in by_task["train_D000_from_U000"]["command"]
    assert all("final_test" not in row["task_id"] for row in spec["tasks"])
    drifted = dict(spec); drifted["semantic_source_sha256"] = {"source": "9" * 64}
    drifted = with_content_hash({key: value for key, value in drifted.items() if key != "content_hash"})
    with pytest.raises(ValueError, match="reuse/source"):
        validate_campaign(drifted)

    for name in campaign.C10P90_IMPLEMENTATION_EVIDENCE_FILES:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"evidence: {name}\n")
    parallel_root = tmp_path / "parallel"
    parallel = create_campaign(
        foundation_lock=foundation / "locks/foundation.json",
        campaign_root=parallel_root, project_dir=worktree,
        source_commit="a" * 40, recipe_profile=PROFILE_C10P90,
        authorize_live_submission=True,
        authorization_phrase="AUTHORIZE HCWDL MHPE C10P90 FULL EXACT SPEC",
    )
    assert parallel["contract"] == CAMPAIGN_SPEC_CONTRACT_C10P90
    assert campaign_profile(parallel) == PROFILE_C10P90
    assert parallel["recipe_profile"] == PROFILE_C10P90
    assert parallel["single_changed_variable"] == "specialist_ce_kd_weights_only"
    assert validate_campaign(parallel) == parallel["content_hash"]
    assert validate_campaign(parallel, executable=True) == parallel["content_hash"]
    parallel_plan = command_plan(parallel)
    assert len(parallel_plan["commands"]) == 23
    assert all(
        any(item.startswith("--job-name=hcwmhpe90_") for item in row["command"])
        for row in parallel_plan["commands"]
    )
    assert campaign_tasks(PROFILE_C10P90) == campaign_tasks()
    assert submission_phrase(PROFILE_C10P90) == (
        "SUBMIT HCWDL MHPE C10P90 FULL EXACT LEDGER"
    )


def test_paired_300k60_campaigns_publish_independent_queue_plans(
    tmp_path, monkeypatch,
):
    from hlt_classification.scouting import hcwdl_mhpe_campaign as campaign

    foundation = tmp_path / "foundation"
    foundation.mkdir()
    foundation_recipe = with_content_hash({
        "contract": "test", "schema_version": 1, "training_passes": 60,
    })
    write_immutable_json(foundation / "recipe.json", foundation_recipe)
    worktree = tmp_path / "worktree"
    for name in campaign.P300_IMPLEMENTATION_EVIDENCE_FILES:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"evidence: {name}\n")
    for name in campaign.DENSE_IMPLEMENTATION_EVIDENCE_FILES:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"evidence: {name}\n")
    for name in campaign.DENSE_C25P75_IMPLEMENTATION_EVIDENCE_FILES:
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"evidence: {name}\n")
    compatibility = {
        "policy": "authenticated_immutable_300k_products_additive_mhpe_v2",
        "byte_exact_files": {},
        "additive_adapter_files": {
            "src/hlt_classification/scouting/engine.py": {
                "foundation_sha256": "a" * 64, "current_sha256": "b" * 64,
            },
            "src/hlt_classification/scouting/hcwdl_training.py": {
                "foundation_sha256": "c" * 64, "current_sha256": "d" * 64,
            },
        },
        "authenticated_foundation_source_sha256": {"source": "e" * 64},
        "u000_target_lineage_evidence": with_content_hash({
            "contract": "HCWDL_UNIFIED_BALANCED_TARGET_DIGEST_SHADOW_EVIDENCE/v1",
            "schema_version": 1,
            "classification": "direct",
            "actual_target_manifest_sha256": "5" * 64,
        }),
        "legacy_logit_path_numerically_regressed": True,
        "foundation_products_immutable": True,
        "adapter_scope": "test",
    }

    def reuse_for(profile):
        target_evidence_hash = compatibility[
            "u000_target_lineage_evidence"
        ]["content_hash"]
        return reuse_lock_payload(
            foundation_spec_path=foundation / "foundation_spec.json",
            foundation_spec_sha256="1" * 64,
            foundation_lock_sha256="2" * 64,
            role_counts={
                "train": 300_000, "validation": 100_000,
                "final_test": 100_000,
            },
            u000_report_sha256="3" * 64,
            u000_checkpoint_sha256="4" * 64,
            u000_target_manifest_sha256="5" * 64,
            m0paired_report_sha256="6" * 64,
            source_commit="a" * 40,
            semantic_source_sha256={"source": "7" * 64},
            foundation_parents={
                "foundation_recipe_sha256": foundation_recipe["content_hash"],
                "parent": "8" * 64,
                "u000_target_lineage_evidence_sha256": target_evidence_hash,
            },
            foundation_core_compatibility=compatibility,
            profile=profile,
        )

    monkeypatch.setattr(
        campaign, "_reuse", lambda **kwargs: reuse_for(kwargs["profile"]),
    )
    monkeypatch.setattr(
        campaign, "semantic_source_hashes", lambda path: {"source": "7" * 64},
    )
    cases = (
        (
            PROFILE_C25P75_300K60,
            CAMPAIGN_SPEC_CONTRACT_C25P75_300K60,
            "AUTHORIZE HCWDL MHPE C25P75 300K60 EXACT SPEC",
            "hcwmhpe25p_",
        ),
        (
            PROFILE_C10P90_300K60,
            CAMPAIGN_SPEC_CONTRACT_C10P90_300K60,
            "AUTHORIZE HCWDL MHPE C10P90 300K60 EXACT SPEC",
            "hcwmhpe90p_",
        ),
        (
            PROFILE_DENSE_ANCHOR50_300K60,
            CAMPAIGN_SPEC_CONTRACT_DENSE_ANCHOR50_300K60,
            "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 EXACT SPEC",
            "hcwmhped_",
        ),
        (
            PROFILE_DENSE_C25P75_300K60,
            CAMPAIGN_SPEC_CONTRACT_DENSE_C25P75_300K60,
            "AUTHORIZE HCWDL MHPE DENSE C25P75 ANCHOR50 300K60 EXACT SPEC",
            "hcwmhpe25d_",
        ),
    )
    specs = []
    for profile, contract, phrase, prefix in cases:
        spec = create_campaign(
            foundation_lock=foundation / "locks/foundation.json",
            campaign_root=tmp_path / profile,
            project_dir=worktree, source_commit="a" * 40,
            recipe_profile=profile, authorize_live_submission=True,
            authorization_phrase=phrase,
        )
        specs.append(spec)
        assert spec["contract"] == contract
        assert spec["role_counts"] == {
            "train": 300_000, "validation": 100_000,
            "final_test": 100_000,
        }
        assert spec["resources"]["gpu_training"] == {
            "cpus": 8, "memory": "96G", "walltime": "06:00:00",
            "gpu": "gpu:gh200:1",
        }
        assert validate_campaign(spec, executable=True) == spec["content_hash"]
        commands = command_plan(spec)["commands"]
        assert len(commands) == (
            38 if profile in {
                PROFILE_DENSE_ANCHOR50_300K60,
                PROFILE_DENSE_C25P75_300K60,
            } else 23
        )
        assert all(
            any(item.startswith(f"--job-name={prefix}") for item in row["command"])
            for row in commands
        )
        assert all(
            node.training_passes == 60 for node in node_registry(profile).values()
        )
    assert len({item["content_hash"] for item in specs}) == 4
    assert len({item["campaign_root"] for item in specs}) == 4
