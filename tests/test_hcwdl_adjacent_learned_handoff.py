from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import write_immutable_json

from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_campaign import (
    CREATION_PHRASE, RESOURCES, create_campaign, tasks, validate_campaign,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_contracts import (
    AGGREGATE_CONTRACT, CONTROL_LOCK_CONTRACT, POPULATION_LOCK_CONTRACT,
    SEED_LOCK_CONTRACT, SOURCE_LOCK_CONTRACT, STAGE_REPORT_CONTRACT,
    VALIDATION_PARTITION_CONTRACT, artifact, validate_artifact,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_data import (
    PairedViewCache, _tagged_pair, morph_context_for_pass,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_graph import (
    FIT_ORDER, NODE_REGISTRY, RUNG_ORDER, TRAINING, graph_payload,
    recipe_payload, validate_graph,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_graph import (
    EARLY_STOPPING as OUTPUT_EARLY_STOPPING,
    NODE_REGISTRY as OUTPUT_NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_probability import (
    ROLES, load_role, publish_lock, publish_role, validate_lock,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_runner import (
    _CoordinateNode, _cache_owners,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_partition import (
    load_partition as load_learned_partition,
    publish_partition as publish_learned_partition,
)
from hlt_classification.scouting.hcwdl_adjacent_output_handoff_partition import (
    partition_codes as output_partition_codes,
)
from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_workflow import (
    _recoveries, validate_aggregate,
)
from hlt_classification.scouting.hcwdl_representation_data import HCWDLParticleInputs


def test_strategy_b_exact_25_fit_registry_and_controls():
    assert validate_graph()
    assert len(FIT_ORDER) == len(set(FIT_ORDER)) == 25
    assert sum(x.role == "fusion_acquisition" for x in NODE_REGISTRY.values()) == 5
    assert sum(x.role == "fusion_withdrawal" for x in NODE_REGISTRY.values()) == 5
    assert sum(x.role == "direct_kd" for x in NODE_REGISTRY.values()) == 5
    required = {
        "FUSION_LOW_LOW_D080", "LOW_WARM_CONTINUE_D080",
        "LOW_PARAMETER_MATCHED_D080", "FUSION_LOW_LOW_D000",
        "LOW_WARM_CONTINUE_D000", "LOW_PARAMETER_MATCHED_D000",
        "CE_SINGLE_D000", "STATIC_U100_D000",
        "DIRECT_VIEW_MORPH_U100_TO_D000",
        "DIRECT_VIEW_MORPH_WITHDRAW_D000",
    }
    assert required <= set(NODE_REGISTRY)
    assert all(
        NODE_REGISTRY[f"LEARNED_WITHDRAW_{coordinate}"].selection_route == "alpha_zero"
        for coordinate in RUNG_ORDER
    )
    for coordinate in RUNG_ORDER[:-1]:
        assert NODE_REGISTRY[f"LEARNED_DIRECT_{coordinate}"].seed_alias == (
            OUTPUT_NODE_REGISTRY[f"OUTPUT_DIRECT_{coordinate}"].seed_alias
        )
    assert NODE_REGISTRY["LEARNED_DIRECT_D000"].seed_alias == (
        OUTPUT_NODE_REGISTRY["OUTPUT_DIRECT_D000_S1"].seed_alias
    )
    assert NODE_REGISTRY["LEARNED_WITHDRAW_D080"].deployable is False
    assert NODE_REGISTRY["LEARNED_WITHDRAW_D000"].deployable is True
    assert NODE_REGISTRY["FUSION_LOW_LOW_D000"].deployable is False
    assert NODE_REGISTRY["CE_SINGLE_D000"].deployable is True
    assert TRAINING["minimum_auc_delta"] == OUTPUT_EARLY_STOPPING[
        "minimum_auc_delta"
    ]


def test_fusion_primary_uses_matched_seed_but_context_has_separate_seed():
    import torch

    from hlt_classification.models.hcwdl_adjacent_fusion_transformer import (
        AdjacentFusionParticleTransformer,
    )
    from hlt_classification.models.scouting_particle_transformer import (
        ScoutingParticleTransformer,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        direct = ScoutingParticleTransformer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        first = AdjacentFusionParticleTransformer(context_initialization_seed=101)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        second = AdjacentFusionParticleTransformer(context_initialization_seed=103)
    direct_state = direct.mod.state_dict()
    assert all(
        torch.equal(direct_state[name], first.hlt_mod.state_dict()[name])
        for name in direct_state
    )
    assert all(
        torch.equal(first.hlt_mod.state_dict()[name], second.hlt_mod.state_dict()[name])
        for name in direct_state
    )
    assert any(
        not torch.equal(value, second.context_mod.state_dict()[name])
        for name, value in first.context_mod.state_dict().items()
        if value.is_floating_point()
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in first.context_mod.named_parameters()
        if not name.startswith(("embed.", "pair_embed.", "blocks."))
    )


def test_source_reducer_preserves_exact_upstream_u100_node_seed(monkeypatch):
    from hlt_classification.scouting import (
        hcwdl_adjacent_learned_handoff_runner as runner,
    )
    from hlt_classification.scouting.hcwdl_adjacent_output_handoff_source import (
        SOURCE_U100_NODE_ID,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_graph import (
        NODE_REGISTRY as SOURCE_NODES,
    )

    monkeypatch.setattr(
        runner, "_source",
        lambda spec: {"u100_node_id": SOURCE_U100_NODE_ID},
    )
    observed = runner._source_node({})
    expected = SOURCE_NODES[SOURCE_U100_NODE_ID]
    assert observed is expected
    assert observed.seed_alias == expected.seed_alias
    caches = {"train": object(), "validation": object()}
    captured = {}
    monkeypatch.setattr(
        runner, "_load_source", lambda spec, device: (object(), object()),
    )
    monkeypatch.setattr(
        runner, "_standard_caches",
        lambda spec, node: (object(), "split", "selection", caches, "hlt"),
    )
    monkeypatch.setattr(
        runner, "_publish_distribution",
        lambda *args, **kwargs: captured.update(kwargs) or {"complete": True},
    )
    assert runner.run_source_reducer({}, device="cpu") == {"complete": True}
    assert captured["seed_alias"] == expected.seed_alias


def test_exact_direct_view_morph_schedule():
    assert graph_payload()["morph_checkpoint_selection_minimum_pass"] == 51
    assert recipe_payload()["morph_schedule"][
        "checkpoint_selection_minimum_pass"
    ] == 51
    expected = {
        1: "U100", 2: "D098", 3: "D096", 50: "D002",
        51: "D000", 52: "D000", 100: "D000",
    }
    for pass_number, name in expected.items():
        observed, coordinate = morph_context_for_pass(pass_number)
        assert observed == name
        assert coordinate.feature_denominator in {1, 50}
    assert [morph_context_for_pass(p)[0] for p in range(51, 101)] == ["D000"] * 50
    with pytest.raises(ValueError): morph_context_for_pass(0)
    with pytest.raises(ValueError): morph_context_for_pass(101)


def test_intermediate_morph_coordinate_is_explicit_not_fixed_rung_lookup():
    name, coordinate = morph_context_for_pass(2)
    node = _CoordinateNode(name, "morph/test", coordinate)
    assert node.coordinate_name == "D098"
    assert node.coordinate == coordinate
    assert node.coordinate.feature_numerator == 1
    assert node.coordinate.feature_denominator == 50


class _FakeCache:
    def __init__(self, batch):
        self.batch = batch
        rows = len(batch["labels"])
        self.identity_digests = batch["identity_digests"]
        self.header = {
            "rows": rows, "array_bytes": 1000, "identity_order_sha256": "a" * 64,
            "identity_set_sha256": "b" * 64,
        }

    def iterate_batches(self, **kwargs):
        yield self.batch

    def iterate_canonical_batches(self, **kwargs):
        yield self.batch

    def iterate_identity_digest_batches(self, identities, **kwargs):
        yield self.batch


def _view(value: float):
    rows, width = 2, 3
    mask = np.asarray([[[1, 1, 0]], [[1, 0, 0]]], dtype=np.bool_)
    return HCWDLParticleInputs(
        np.full((rows, 21, width), value, np.float32),
        np.full((rows, 4, width), value, np.float32), mask,
        np.asarray([2, 1], np.int32),
        np.asarray([[0, 1, -1], [0, -1, -1]], np.int64),
        np.asarray([[0, 1, -1], [0, -1, -1]], np.int8),
        np.asarray([[0, 1, -1], [0, -1, -1]], np.int8),
    )


def test_paired_cache_tags_context_zero_primary_one_and_never_publishes():
    identities = np.arange(64, dtype=np.uint8).reshape(2, 32)
    common = {"labels": np.asarray([1, 2]), "identity_digests": identities}
    left = _FakeCache({**common, "hlt": _view(1.0)})
    right = _FakeCache({**common, "hlt": _view(2.0)})
    cache = PairedViewCache(left, right, role="train", lineage={"test": "pair"})
    batch = next(cache.iterate_batches(epoch=0, sampler_seed=1, batch_size=2))
    view = batch["hlt"]
    assert view.features.shape == (2, 21, 6)
    assert np.all(view.features[:, :, :3] == 1)
    assert np.all(view.features[:, :, 3:] == 2)
    assert set(view.content_source_codes[view.mask[:, 0]]) == {0, 1}
    assert np.all(view.content_source_codes[~view.mask[:, 0]] == -1)
    assert cache.header["durable_artifact_published"] is False


def test_fixed_pair_and_dynamic_morph_cache_owners_normalize_for_cleanup():
    first, second, manager = object(), object(), object()
    assert _cache_owners((first, second)) == [first, second]
    assert _cache_owners(manager) == [manager]


def test_alpha_validation_curve_reuses_endpoints_and_only_infers_interiors(
    monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import (
        hcwdl_adjacent_learned_handoff_runner as runner,
    )

    identities = np.arange(64, dtype=np.uint8).reshape(2, 32)
    labels = np.asarray([1, 2], dtype=np.int64)
    calls = []

    def infer(*args, alpha, **kwargs):
        calls.append(alpha)
        logits = np.zeros((2, 15), dtype=np.float32)
        logits[:, 0] = alpha
        return identities, logits, labels

    def split(ids, probabilities, observed_labels, lookup):
        assert lookup == {b"registered": 2}
        return {"V_report": (ids, probabilities, observed_labels)}

    def metrics(log_probabilities, observed_labels):
        return {
            "accuracy": float(log_probabilities[0, 0]),
            "macro_ovr_auc": float(log_probabilities[0, 0]),
            "macro_mean_log_qcd_rejection_at_50pct_signal": 1.0,
            "labels": observed_labels.tolist(),
        }

    monkeypatch.setattr(runner, "_infer", infer)
    monkeypatch.setattr(runner, "_split_validation", split)
    monkeypatch.setattr(runner, "classification_metrics", metrics)
    alpha_zero = {"endpoint": "zero"}
    alpha_one = {"endpoint": "one"}
    curve = runner._alpha_validation_curve(
        object(), object(), sampler_seed=7, device="cpu",
        protocol="adjacent_two_view_v1",
        partition_lookup={b"registered": 2}, expected_ids=identities,
        expected_labels=labels, alpha_zero_metrics=alpha_zero,
        alpha_one_metrics=alpha_one,
    )
    assert [row["alpha"] for row in curve] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert calls == [0.25, 0.5, 0.75]
    assert curve[0]["metrics"] is alpha_zero
    assert curve[-1]["metrics"] is alpha_one


def test_strategy_b_partition_has_own_contract_and_exact_shared_assignments(
    tmp_path: Path,
):
    identities = np.arange(45 * 32, dtype=np.int64).reshape(45, 32).astype(np.uint8)
    identities[:, 0] = np.arange(45, dtype=np.uint8)
    labels = np.repeat(np.arange(15, dtype=np.int16), 3)
    report = publish_learned_partition(
        tmp_path / "partition.json", identity_digests=identities,
        labels=labels, parents={"campaign_spec": "1" * 64},
        source_commit="a" * 40,
    )
    loaded, arrays = load_learned_partition(tmp_path / "partition.json")
    assert report == loaded
    assert report["contract"].endswith("VALIDATION_PARTITION/v1")
    assert np.array_equal(
        arrays["partition"], output_partition_codes(identities, labels),
    )
    invalid = artifact({
        **{
            key: value for key, value in report.items()
            if key not in {"content_hash", "contract", "schema_version"}
        },
        "complete_validation_coverage": False,
    }, contract=VALIDATION_PARTITION_CONTRACT)
    bad_path = tmp_path / "invalid_partition.json"
    write_immutable_json(bad_path, invalid)
    with pytest.raises(ValueError, match="partition semantics"):
        load_learned_partition(bad_path)

    data_path = tmp_path / "wrong_dtype_partition.npz"
    wrong_dtype = {
        "identity_digest": arrays["identity_digest"],
        "label": arrays["label"].astype(np.int64),
        "partition": arrays["partition"],
    }
    from hlt_classification.data.cache_contracts import (
        array_sha256, atomic_publish_bytes, deterministic_npz_bytes,
        sha256_file,
    )
    atomic_publish_bytes(data_path, deterministic_npz_bytes(wrong_dtype))
    invalid_dtype_report = artifact({
        **{
            key: value for key, value in report.items()
            if key not in {
                "content_hash", "contract", "schema_version", "data_path",
                "data_sha256", "array_sha256",
            }
        },
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "array_sha256": {
            name: array_sha256(name, value)
            for name, value in wrong_dtype.items()
        },
    }, contract=VALIDATION_PARTITION_CONTRACT)
    wrong_dtype_path = tmp_path / "wrong_dtype_partition.json"
    write_immutable_json(wrong_dtype_path, invalid_dtype_report)
    with pytest.raises(ValueError, match="partition dtypes"):
        load_learned_partition(wrong_dtype_path)


def test_low_low_pair_requires_byte_identical_views():
    identities = np.arange(64, dtype=np.uint8).reshape(2, 32)
    common = {"labels": np.asarray([1, 2]), "identity_digests": identities}
    left = {**common, "hlt": _view(1.0)}
    right = {**common, "hlt": _view(1.0)}
    _tagged_pair(left, right, require_identical_views=True)
    changed = {**common, "hlt": _view(2.0)}
    with pytest.raises(ValueError, match="identity-equal"):
        _tagged_pair(left, changed, require_identical_views=True)


def test_compact_probability_roles_are_identity_and_consumer_bound(tmp_path: Path):
    identities = np.arange(96, dtype=np.uint8).reshape(3, 32)
    probabilities = np.full((3, 15), 1 / 15, dtype=np.float32)
    parents = {"campaign_spec": "1" * 64}
    lineage = {"teacher": {"report": "2" * 64, "checkpoint": "3" * 64}}
    manifests = {}
    for role in ROLES:
        manifests[role] = publish_role(
            tmp_path, distribution_id="TEACHER", role=role,
            identity_digests=identities, probabilities=probabilities,
            component_order=("teacher",), component_lineage=lineage,
            consumers=("student",), parents=parents,
            producer_commit="a" * 40,
            target_temperature=2.0 if role == "train" else 1.0,
        )
    publish_lock(
        tmp_path / "lock.json", distribution_id="TEACHER",
        manifests=manifests, consumers=("student",), parents=parents,
    )
    lock, loaded = validate_lock(tmp_path / "lock.json", distribution_id="TEACHER")
    assert lock["consumers"] == ["student"]
    assert loaded["train"]["consumers"] == ["student"]
    assert all(loaded[role]["consumers"] == [] for role in ROLES[1:])
    _, observed_ids, observed_probabilities = load_role(
        tmp_path / "V_report_manifest.json",
        distribution_id="TEACHER", role="V_report",
    )
    assert np.array_equal(observed_ids, identities)
    assert np.array_equal(observed_probabilities, probabilities)


def test_nonteacher_probability_lock_is_report_only(tmp_path: Path):
    identities = np.arange(96, dtype=np.uint8).reshape(3, 32)
    probabilities = np.full((3, 15), 1 / 15, dtype=np.float32)
    manifest = publish_role(
        tmp_path, distribution_id="CONTROL", role="V_report",
        identity_digests=identities, probabilities=probabilities,
        component_order=("control",),
        component_lineage={
            "control": {"report": "4" * 64, "checkpoint": "5" * 64},
        },
        consumers=(), parents={"campaign_spec": "1" * 64},
        producer_commit="a" * 40, target_temperature=1.0,
    )
    publish_lock(
        tmp_path / "lock.json", distribution_id="CONTROL",
        manifests={"V_report": manifest}, consumers=(),
        parents={"campaign_spec": "1" * 64},
    )
    lock, manifests = validate_lock(
        tmp_path / "lock.json", distribution_id="CONTROL",
    )
    assert lock["roles"] == ["V_report"]
    assert set(manifests) == {"V_report"}


def test_reporting_recovery_includes_every_available_class():
    def metrics(auc, macro_r50, hbb_r50, hcc_r50):
        return {
            "macro_ovr_auc": auc,
            "macro_mean_log_qcd_rejection_at_50pct_signal": np.log(macro_r50),
            "per_class": {
                "Hbb": {"qcd_rejection": {"50pct": {"rejection": hbb_r50}}},
                "Hcc": {"qcd_rejection": {"50pct": {"rejection": hcc_r50}}},
            },
        }

    baseline = metrics(.90, 100, 80, 40)
    oracle = metrics(.95, 200, 180, 90)
    result = _recoveries(metrics(.925, 150, 130, 65), baseline, oracle)
    assert result["auc_pct"] == pytest.approx(50)
    assert result["macro_r50_linear_pct"] == pytest.approx(50)
    assert result["per_class_r50_linear_pct"] == pytest.approx({
        "Hbb": 50, "Hcc": 50,
    })


def test_aggregate_validator_counts_source_separately_from_all_25_fits():
    alpha_curve = [
        {
            "alpha": alpha,
            "metrics": {
                "accuracy": .5, "macro_ovr_auc": .6,
                "macro_mean_log_qcd_rejection_at_50pct_signal": 1.0,
            },
        }
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    fit_rows = []
    for node_id in FIT_ORDER:
        row = {
            "model_id": node_id,
            "training": {
                "rolling_resume_published": False,
                "partial_checkpoint_reuse": False,
                "selected_pass": (
                    51 if node_id == "DIRECT_VIEW_MORPH_U100_TO_D000"
                    else 1
                ),
                "checkpoint_selection_minimum_pass": (
                    51 if node_id == "DIRECT_VIEW_MORPH_U100_TO_D000"
                    else 1
                ),
            },
        }
        if NODE_REGISTRY[node_id].input_protocol != "standard_hlt_v1":
            if NODE_REGISTRY[node_id].role in {
                "fusion_withdrawal", "morph_withdrawal",
            }:
                row["diagnostics"] = {
                    "all_routes_use_identical_validation_identities": True,
                    "all_routes_use_identical_validation_labels": True,
                    "alpha_validation_curve": alpha_curve,
                }
            else:
                row["diagnostics"] = {
                    "all_routes_use_identical_V_report_identities": True,
                    "all_routes_use_identical_V_report_labels": True,
                    "alpha_validation_curve": alpha_curve,
                }
        fit_rows.append(row)
    value = artifact({
        "report_role": "V_report",
        "model_rows": [
            {"model_id": "SOURCE_U100"},
            *fit_rows,
        ],
        "control_rows": [
            {"model_id": "M0CE60"}, {"model_id": "U000"},
        ],
        "required_causal_comparisons": [
            {"left": left, "right": right}
            for left, right in (
                ("CE_FUSION_D000_D000", "CE_SINGLE_D000"),
                (
                    "CE_FUSION_D000_D000",
                    "PARAMETER_MATCHED_SINGLE_D000",
                ),
                (
                    "DIRECT_VIEW_MORPH_U100_TO_D000",
                    "CE_FUSION_D000_D000",
                ),
                (
                    "STATIC_U100_D000",
                    "DIRECT_VIEW_MORPH_U100_TO_D000",
                ),
                (
                    "alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)",
                    "CE_SINGLE_D000",
                ),
                (
                    "DIRECT_VIEW_MORPH_WITHDRAW_D000",
                    "alpha0(DIRECT_VIEW_MORPH_U100_TO_D000)",
                ),
                (
                    "DIRECT_VIEW_MORPH_WITHDRAW_D000",
                    "LEARNED_T_D000",
                ),
                ("LEARNED_T_D000", "DIRECT_KD_D000"),
            )
        ],
        "adjacent_carrier_comparisons": [
            {"left": f"LEARNED_T_{coordinate}", "right": parent}
            for coordinate, parent in zip(
                RUNG_ORDER,
                ("SOURCE_U100", "LEARNED_T_D080", "LEARNED_T_D060",
                 "LEARNED_T_D040", "LEARNED_T_D020"),
                strict=True,
            )
        ],
        "learned_carrier_minus_direct_comparisons": [
            {
                "left": f"LEARNED_T_{coordinate}",
                "right": f"LEARNED_DIRECT_{coordinate}",
            }
            for coordinate in RUNG_ORDER
        ],
        "rung_withdrawal_decomposition": [
            {"coordinate": coordinate} for coordinate in RUNG_ORDER
        ],
        "paired_bootstrap_samples": 2000,
        "paired_bootstrap_seed": 17,
        "self_ensemble_identity_control": {"byte_identical": True},
        "all_25_fits_reported": True,
        "all_fit_histories_reported": True,
        "poor_metrics_do_not_control_completion": True,
        "final_test_accessed": False,
    }, contract=AGGREGATE_CONTRACT)
    assert validate_aggregate(value) == value["content_hash"]
    bad = artifact({
        **{key: item for key, item in value.items() if key not in {
            "content_hash", "contract", "schema_version",
        }},
        "all_25_fits_reported": False,
    }, contract=AGGREGATE_CONTRACT)
    with pytest.raises(ValueError, match="aggregate semantics"):
        validate_aggregate(bad)
    missing_curve = artifact({
        **{key: item for key, item in value.items() if key not in {
            "content_hash", "contract", "schema_version",
        }},
        "model_rows": [
            {
                **row,
                **(
                    {"diagnostics": {
                        key: item
                        for key, item in row["diagnostics"].items()
                        if key != "alpha_validation_curve"
                    }}
                    if row.get("model_id") == "LEARNED_ACQUIRE_D080"
                    else {}
                ),
            }
            for row in value["model_rows"]
        ],
    }, contract=AGGREGATE_CONTRACT)
    with pytest.raises(ValueError, match="aggregate semantics"):
        validate_aggregate(missing_curve)

    wrong_comparison = artifact({
        **{
            key: item for key, item in value.items()
            if key not in {"content_hash", "contract", "schema_version"}
        },
        "required_causal_comparisons": [
            {
                **row,
                **(
                    {"right": "WRONG_CONTROL"}
                    if index == 0 else {}
                ),
            }
            for index, row in enumerate(value["required_causal_comparisons"])
        ],
    }, contract=AGGREGATE_CONTRACT)
    with pytest.raises(ValueError, match="aggregate semantics"):
        validate_aggregate(wrong_comparison)


def test_task_dag_has_25_trains_and_bounded_ram_resources():
    rows = tasks(); ids = [x["task_id"] for x in rows]
    assert len(ids) == len(set(ids)) == 65
    assert sum(x["kind"] == "train" for x in rows) == 25
    assert {x["node_id"] for x in rows if x["kind"] == "train"} == set(FIT_ORDER)
    assert RESOURCES["fit"].memory == "500G"
    assert RESOURCES["fit"].walltime == "7-00:00:00"
    assert not any("final_test" in task_id for task_id in ids)
    positions = {task_id: index for index, task_id in enumerate(ids)}
    for row in rows:
        assert all(positions[parent] < positions[row["task_id"]] for parent in row["dependencies"])


def test_campaign_publication_has_staged_gate_and_science(tmp_path: Path, monkeypatch):
    from hlt_classification.scouting import hcwdl_adjacent_learned_handoff_campaign as campaign

    foundation = tmp_path / "foundation.json"; foundation.write_text("{}")
    source = artifact({
        "parents": {"foundation": "1" * 64}, "foundation_spec_path": str(foundation),
        "replicate_seed": 7, "role_counts": {"train": 10, "validation": 6},
        "source_campaign_spec_path": str(tmp_path / "source.json"),
        "u100_report_path": str(tmp_path / "u100.json"),
        "u100_checkpoint_path": str(tmp_path / "u100.pt"),
        "source_outputs_mutated": False, "final_test_accessed": False,
    }, contract=SOURCE_LOCK_CONTRACT)
    controls = artifact({
        "parents": {"m0": "2" * 64}, "m0ce60_report_path": str(tmp_path / "m0.json"),
        "pure_offline_u000_report_path": str(tmp_path / "u000.json"),
        "reporting_only": True, "final_test_accessed": False,
    }, contract=CONTROL_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda **kwargs: source)
    monkeypatch.setattr(campaign, "build_control_lock", lambda **kwargs: controls)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    monkeypatch.setattr(campaign, "validate_control_lock", lambda value: value["content_hash"])
    root = tmp_path / "campaign"
    spec = create_campaign(
        source_campaign_spec=tmp_path / "source.json",
        u100_training_report=tmp_path / "u100.json",
        u100_selected_checkpoint=tmp_path / "u100.pt",
        m0ce60_training_report=tmp_path / "m0.json",
        pure_offline_u000_training_report=tmp_path / "u000.json",
        campaign_root=root, project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True, authorization_phrase=CREATION_PHRASE,
    )
    assert spec["fresh_fit_count"] == 25
    assert spec["paired_bootstrap_samples"] == 2000
    assert isinstance(spec["paired_bootstrap_seed"], int)
    assert spec["ram_only_particle_and_hidden_state"] is True
    assert spec["rolling_resume"] is False
    assert spec["projected_durable_bytes"] == 16 * 1024**3
    assert spec["probability_retention_policy"]["nonteacher_models"] == ["V_report"]
    assert validate_artifact(
        json.loads((root / "locks/population.json").read_text()),
        contract=POPULATION_LOCK_CONTRACT,
    ) == spec["parents"]["population_lock"]
    seed_lock = json.loads((root / "locks/seeds.json").read_text())
    assert validate_artifact(seed_lock, contract=SEED_LOCK_CONTRACT) == spec[
        "parents"
    ]["seed_lock"]
    assert seed_lock["node_domains"]["LEARNED_DIRECT_D080"]["seed_alias"] == (
        OUTPUT_NODE_REGISTRY["OUTPUT_DIRECT_D080"].seed_alias
    )
    assert validate_campaign(spec, executable=True) == spec["content_hash"]
    gate = json.loads((root / "gate_command_plan.json").read_text())
    science = json.loads((root / "science_command_plan.json").read_text())
    assert [x["task_id"] for x in gate["commands"]] == [
        "authenticate", "partition_validation", "audit_sources_and_storage", "preflight",
    ]
    assert len([x for x in science["commands"] if x["task_id"].startswith("train_")]) == 25
    assert len(science["commands"]) == 61
    assert all("--mem=500G" in x["command"] for x in science["commands"] if x["task_id"].startswith("train_"))
    science_by_id = {row["task_id"]: row for row in science["commands"]}
    for node_id in (
        "FUSION_LOW_LOW_D080", "LOW_PARAMETER_MATCHED_D080",
        "FUSION_LOW_LOW_D000", "LOW_PARAMETER_MATCHED_D000",
        "CE_SINGLE_D000", "STATIC_U100_D000",
        "DIRECT_VIEW_MORPH_U100_TO_D000",
    ):
        assert science_by_id[f"train_{node_id}"]["dependencies"] == []
    assert science_by_id["train_LOW_WARM_CONTINUE_D080"]["dependencies"] == [
        "train_LEARNED_DIRECT_D080"
    ]
    assert science_by_id["train_DIRECT_VIEW_MORPH_WITHDRAW_D000"][
        "dependencies"
    ] == ["reduce_DIRECT_VIEW_MORPH_U100_TO_D000"]

    from hlt_classification.scouting import (
        hcwdl_adjacent_learned_handoff_workflow as workflow,
    )
    from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_workflow import (
        SCIENCE_GATE_TASKS, task_outputs, validate_science_gate,
    )
    from hlt_classification.scouting.hcwdl_recovery import (
        build_task_attestation, task_attestation_path,
    )

    monkeypatch.setattr(
        workflow, "validate_execution_acceptance",
        lambda _spec, value: value["content_hash"],
    )
    for task_id in SCIENCE_GATE_TASKS:
        for output in task_outputs(spec, task_id):
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix == ".json":
                write_immutable_json(
                    output,
                    artifact({
                        "task_id": task_id, "final_test_accessed": False,
                    }, contract=STAGE_REPORT_CONTRACT),
                )
            else:
                output.write_bytes(task_id.encode())
        attestation = build_task_attestation(
            campaign_spec_sha256=spec["content_hash"], task_id=task_id,
            array_index=None, outputs=task_outputs(spec, task_id),
        )
        write_immutable_json(
            task_attestation_path(root, task_id, None), attestation,
        )
    gate_evidence = validate_science_gate(spec)
    assert set(gate_evidence["task_attestations"]) == set(SCIENCE_GATE_TASKS)
    (root / "reports/stages/authenticate.json").unlink()
    with pytest.raises(FileNotFoundError, match="gate output is absent"):
        validate_science_gate(spec)

    from hlt_classification.scouting.hcwdl_adjacent_learned_handoff_recovery import (
        create_recovery,
    )
    from hlt_classification.scouting.hcwdl_recovery import (
        build_monitor_report, build_submission_ledger,
    )

    commands = {row["task_id"]: row["command"] for row in science["commands"]}
    jobs = {
        task_id: str(99000 + index)
        for index, task_id in enumerate(commands)
    }
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "science_ledger.json"
    write_immutable_json(ledger_path, ledger)
    monitor = build_monitor_report(
        ledger, states_by_job_id={job: "FAILED" for job in jobs.values()},
        artifact_validity={task_id: False for task_id in jobs},
    )
    monitor_path = tmp_path / "science_monitor.json"
    write_immutable_json(monitor_path, monitor)
    recovery_root = tmp_path / "recovery"
    recovery = create_recovery(
        campaign_spec=root / "campaign_spec.json",
        submission_ledger=ledger_path, monitor_report=monitor_path,
        recovery_root=recovery_root, project_dir=tmp_path,
        source_commit="b" * 40,
    )
    assert set(recovery["retry_tasks"]) == set(jobs)
    recovery_plan = __import__("json").loads(
        (recovery_root / "command_plan.json").read_text()
    )
    assert recovery_plan["commands"][0]["task_id"] == "reduce_SOURCE_U100"
    assert recovery_plan["commands"][0]["dependencies"] == []
