"""Fresh-reference four-spine graph and registered optimization protocol.

This module emits a scientific plan, not an authorization to submit Slurm jobs.
Production allocation must be measured on the new data before submission exists.
"""
from fractions import Fraction
import hashlib
import math

from .contracts import artifact
from .foundation import validate_foundation_spec
from .splits import is_subset_profile

PATHS = {
    "DIRECT": ("D000",),
    "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    "DENSE": ("U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000"),
    "ULTRADENSE": ("U020", "U040", "U060", "U080", "U100", "D090", "D080", "D070",
                   "D060", "D050", "D040", "D030", "D020", "D010", "D000"),
}


def coordinate(name: str) -> tuple[Fraction, Fraction]:
    registered = {r for path in PATHS.values() for r in path} | {"U000"}
    if name not in registered:
        raise ValueError(f"Unregistered coordinate: {name}")
    fraction = {"033": Fraction(1, 3), "066": Fraction(2, 3)}.get(name[1:], Fraction(int(name[1:]), 100))
    return (fraction, Fraction(0)) if name.startswith("U") else (Fraction(1), 1 - fraction)


def paired_seed(coordinate_name: str, domain: str) -> int:
    coordinate(coordinate_name)
    if domain not in {"initialization", "sampler"}:
        raise ValueError("Unknown RNG domain")
    # Teacher/branch deliberately omitted. All D000s and the CE HLT control pair.
    return int.from_bytes(hashlib.sha256(f"JC2/four-spine/v1/{domain}/{coordinate_name}".encode()).digest()[:4], "big")


def recipe() -> dict:
    return artifact(
        "RECIPE", maximum_passes=100, minimum_passes=60,
        warmup_passes=3, hold_through=45, decay_through=60,
        peak_lr=3e-4, floor_lr=1.5e-5, patience=15, minimum_auc_delta=5e-5,
        patience_clock="starts_at_first_validation_not_pass60",
        optimizer="AdamW", betas=[.9, .999], eps=1e-8, weight_decay=.01,
        batch_size=256, validation_every_pass=1, class_weights=None,
        ce_weight=.25, kd_weight=.75, temperature=2., kd_loss="forward_KL_one_T_squared",
        references_loss="CE_only", accumulation=1, gpus_per_fit=1,
        precision="bf16_forward_fp32_loss", rolling_resume=False,
        checkpoint_policy="selected_weights_only_restore_best",
        selection_order=["maximum_macro_auc", "minimum_cross_entropy", "maximum_macro_log_R50", "earliest_update"],
        source_population="all_selected_train_rows_no_replacement_per_pass",
    )


def learning_rate(pass_position: float) -> float:
    """Continuous completed-pass coordinate, evaluated at each update endpoint."""
    if not math.isfinite(pass_position) or not 0 < pass_position <= 100:
        raise ValueError("Schedule position must be in (0,100]")
    if pass_position <= 3:
        return 3e-4 * pass_position / 3
    if pass_position <= 45:
        return 3e-4
    if pass_position <= 60:
        return 1.5e-5 + (3e-4 - 1.5e-5) * .5 * (1 + math.cos(math.pi * (pass_position - 45) / 15))
    return 1.5e-5


def build_campaign_plan(foundation: dict) -> dict:
    parent = validate_foundation_spec(foundation)
    nodes = [dict(node_id="M0HLT", coordinate="D000", teacher=None, branch="REFERENCE"),
             dict(node_id="U000", coordinate="U000", teacher=None, branch="REFERENCE")]
    publications = ["U000"]
    branches = {}
    for branch, path in PATHS.items():
        teacher, previous, names = "U000", "U000", []
        for i, name in enumerate(path):
            node_id = f"JC2_{branch}_{name}_from_{previous}"
            nodes.append(dict(node_id=node_id, coordinate=name, teacher=teacher, branch=branch))
            if i + 1 < len(path):
                publications.append(node_id)
            names.append(node_id)
            teacher, previous = node_id, name
        branches[branch] = names
    for node in nodes:
        u, f = coordinate(node["coordinate"])
        node.update(u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
                    initialization_seed=paired_seed(node["coordinate"], "initialization"),
                    sampler_seed=paired_seed(node["coordinate"], "sampler"),
                    deployable=node["coordinate"] == "D000")
    population = {}
    if is_subset_profile(foundation["splits"]):
        split = foundation["splits"]
        population = dict(split_profile=split["profile"], split_profile_sha256=split["content_hash"],
                          split_registry_sha256=split["registry_sha256"], role_counts=split["role_counts"],
                          role_membership_sha256={r: m["content_hash"] for r, m in split["memberships"].items()},
                          reference_training_population="same_profile_as_all_students_no_imported_full_data_teacher")
    return artifact(
        "CAMPAIGN_PLAN", version=2 if population else 1,
        foundation_sha256=parent, inputs_sha256=foundation["inputs"]["content_hash"],
        recipe=recipe(), nodes=nodes, branches=branches, probability_publications=publications,
        fresh_fit_count=len(nodes), probability_publication_count=len(publications),
        imported_models=[], final_test_accessed=False, executable=False,
        remaining_gate="new_dataset_installed_Weaver_parity_and_measured_Tigris_miniature",
        **population,
    )
