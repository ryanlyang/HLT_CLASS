from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.scouting.hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY_SHA256,
    NODE_REGISTRY,
    RREL_STRATEGY,
    RSET_STRATEGY,
)
from hlt_classification.scouting.hcwdl_representation_recipe import (
    FROZEN_SCIENTIFIC_VALUES_SHA256,
    KERNEL_RESOURCE_NAMES,
    PARENT_RECIPE_CONTRACT,
    REQUIRED_EVIDENCE_KEYS,
    REQUIRED_PARENT_KEYS,
    build_representation_recipe,
    derive_recipe_producer_source_sha256,
    example_representation_recipe,
    frozen_scientific_values,
    validate_representation_recipe,
)


def _fixture_inputs():
    parents = {name: "1" * 64 for name in REQUIRED_PARENT_KEYS}
    kernels = {name: "2" * 64 for name in KERNEL_RESOURCE_NAMES}
    evidence = {name: "3" * 64 for name in REQUIRED_EVIDENCE_KEYS}
    return parents, kernels, evidence


def _rehash(artifact):
    return with_content_hash({key: value for key, value in artifact.items() if key != "content_hash"})


def test_overlay_binds_graph_controls_resources_evidence_and_parent_lineage():
    parents, kernels, evidence = _fixture_inputs()
    recipe = build_representation_recipe(
        parents=parents, kernel_array_logical_hashes=kernels, evidence=evidence,
    )
    assert validate_representation_recipe(
        recipe, expected_parents=parents,
    ) == recipe["content_hash"]
    payload = recipe["payload"]
    assert payload["ascent_graph_sha256"] == ASCENT_GRAPH_SHA256
    assert payload["control_registry_sha256"] == CONTROL_REGISTRY_SHA256
    assert payload["primary_node_ids"] == sorted(NODE_REGISTRY)
    assert payload["scientific_values_sha256"] == FROZEN_SCIENTIFIC_VALUES_SHA256
    assert payload["kernel_array_logical_hashes"] == dict(sorted(kernels.items()))
    assert payload["acceptance_evidence"] == dict(sorted(evidence.items()))


def test_recipe_producer_source_is_derived_from_clean_checkout(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "producer"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "hcwdl-rkd@example.invalid"],
        cwd=repository, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "HCWDL RKD Test"],
        cwd=repository, check=True,
    )
    source = repository / "producer.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "producer.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "producer fixture"],
        cwd=repository, check=True, capture_output=True,
    )
    expected = capture_source_snapshot(repository)["source_snapshot_sha256"]
    assert derive_recipe_producer_source_sha256(repository) == expected

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        derive_recipe_producer_source_sha256(repository)


def test_every_frozen_representation_value_is_explicit():
    values = frozen_scientific_values()
    assert values["representation_coefficient"] == 0.10
    assert values["orthogonality_coefficient"] == 1e-3
    assert values["strategies"][RSET_STRATEGY]["components_at_full_strength"] == {
        "jet": 0.4, "set": 0.6, "relation": 0.0,
    }
    assert values["strategies"][RREL_STRATEGY]["components_at_full_strength"] == {
        "jet": 0.3, "set": 0.45, "relation": 0.25,
    }
    assert values["ramps"]["jet_set"] == {
        "zero_through_pass": 2.0, "linear_full_at_pass": 6.0,
    }
    assert values["ramps"]["relation"] == {
        "zero_through_pass": 4.0, "linear_full_at_pass": 8.0,
    }
    assert values["token_set"]["kernel"]["total_features"] == 1024
    assert values["relations"]["kernel"]["features_per_stratum"] == 256
    assert values["relations"]["population"]["maximum_tokens"] == 32
    assert [row["upper_exclusive"] for row in values["relations"]["strata"]] == [
        0.05, 0.20, None,
    ]
    assert values["calibration"]["selection_rows"] == 4096
    assert values["calibration"]["minimum_supported_batches"] == 12
    assert values["calibration"]["active_scale_bounds_inclusive"] == [1e-4, 1e4]
    assert values["seeds"] == {
        **values["seeds"],
        "screening": 1337,
        "confirmation": [11, 22, 33, 44, 55],
        "within_class_shuffle": 20260809,
    }
    assert values["training"]["passes"] == 60
    assert values["training"]["validation_every_passes"] == 1
    assert values["training"]["representation_row_weight_source"] == (
        "parent_recipe_exact_15_ones"
    )
    assert values["training"]["class_weighted_representation_row_reduction"] is False
    assert values["training"]["representation_row_reduction"] == "mean(per_jet_loss)"
    assert values["training"]["selection_order"][0] == "highest_macro_ovr_auc"
    assert values["target_forward"]["canonical_rows_per_source_batch"] == 256
    assert values["target_lifecycle"]["logical_banks"] == [
        "D0c", "D0w", "D25c", "D25w", "D50c", "D50w",
        "D75c", "D75w", "D100", "TOFF",
    ]


def test_frozen_values_are_defensive_and_overlay_contains_no_base_recipe_override():
    first = frozen_scientific_values()
    first["representation_coefficient"] = 999
    assert frozen_scientific_values()["representation_coefficient"] == 0.10

    payload = example_representation_recipe()["payload"]
    forbidden = set(payload["scientific_values"]["forbidden_parent_overrides"])
    assert not forbidden & set(payload)
    assert payload["parent_recipe_contract"] == PARENT_RECIPE_CONTRACT


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("scientific_values", "representation_coefficient"), 0.20),
        (("scientific_values", "training", "passes"), 40),
        (("scientific_values", "seeds", "screening"), 1338),
        (("scientific_values", "token_set", "kernel", "total_features"), 2048),
        (("ascent_graph_sha256",), "f" * 64),
    ],
)
def test_any_scientific_or_graph_mutation_fails_even_when_rehashed(path, replacement):
    recipe = deepcopy(example_representation_recipe())
    target = recipe["payload"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    recipe = _rehash(recipe)
    with pytest.raises(ValueError, match="frozen scientific|identity or graph"):
        validate_representation_recipe(recipe)


def test_recipe_rejects_missing_extra_or_cross_lineage_parents_and_hashes():
    parents, kernels, evidence = _fixture_inputs()
    missing = dict(parents)
    missing.pop("teacher_import")
    with pytest.raises(ValueError, match="parent lineage keys"):
        build_representation_recipe(
            parents=missing, kernel_array_logical_hashes=kernels, evidence=evidence,
        )
    extra = dict(kernels)
    extra["dynamic_bandwidth"] = "4" * 64
    with pytest.raises(ValueError, match="kernel logical hash registry keys"):
        build_representation_recipe(
            parents=parents, kernel_array_logical_hashes=extra, evidence=evidence,
        )

    recipe = build_representation_recipe(
        parents=parents, kernel_array_logical_hashes=kernels, evidence=evidence,
    )
    changed = dict(parents)
    changed["parent_recipe"] = "5" * 64
    with pytest.raises(ValueError, match="parent lineage differs"):
        validate_representation_recipe(recipe, expected_parents=changed)
