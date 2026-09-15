"""Three salience spines with exact all-ancestor MT20 logit supervision."""
from __future__ import annotations

from fractions import Fraction

from hlt_classification.data.cache_contracts import with_content_hash

from .campaign import PATHS as FOUR_SPINE_PATHS, coordinate, paired_seed, recipe
from .contracts import artifact
from .salience_foundation import validate_foundation_spec


BRANCH_ORDER = ("DIRECT", "COARSE", "DENSE")
PATHS = {branch: FOUR_SPINE_PATHS[branch] for branch in BRANCH_ORDER}
CE_WEIGHT = Fraction(1, 5)
KD_WEIGHT = Fraction(4, 5)
IMMEDIATE_WEIGHT = Fraction(1, 2)
HISTORICAL_WEIGHT = Fraction(3, 10)


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def teacher_loss_weights(count: int) -> tuple[Fraction, ...]:
    """Nearest-first loss contributions that sum exactly to four fifths."""
    if type(count) is not int or count < 1:
        raise ValueError("MT20 requires at least one teacher")
    if count == 1:
        return (KD_WEIGHT,)
    raw = tuple(Fraction(1, 2**index) for index in range(1, count))
    normalizer = sum(raw, Fraction())
    values = (
        IMMEDIATE_WEIGHT,
        *(HISTORICAL_WEIGHT * value / normalizer for value in raw),
    )
    if sum(values, Fraction()) != KD_WEIGHT or any(value <= 0 for value in values):
        raise RuntimeError("MT20 teacher weight construction differs")
    return values


def mt20_recipe() -> dict:
    """Keep the JetClass2 schedule fixed; change only C25/P75 to C20/P80."""
    value = dict(recipe())
    value.pop("content_hash")
    value.update(ce_weight=.20, kd_weight=.80)
    return with_content_hash(value)


def build_campaign_plan(foundation: dict) -> dict:
    parent = validate_foundation_spec(foundation)
    nodes = [
        dict(node_id="M0HLT", coordinate="D000", teacher=None,
             teachers=[], branch="REFERENCE"),
        dict(node_id="U000", coordinate="U000", teacher=None,
             teachers=[], branch="REFERENCE"),
    ]
    publications = ["U000"]
    branches: dict[str, list[str]] = {}
    for branch, path in PATHS.items():
        ancestry = ["U000"]
        previous, names = "U000", []
        for index, name in enumerate(path):
            node_id = f"JC2SMT20_{branch}_{name}_from_{previous}"
            nearest_first = list(reversed(ancestry))
            weights = teacher_loss_weights(len(nearest_first))
            teachers = [
                dict(node_id=teacher, loss_weight=_rational(weight))
                for teacher, weight in zip(nearest_first, weights, strict=True)
            ]
            nodes.append(dict(
                node_id=node_id, coordinate=name, teacher=nearest_first[0],
                teachers=teachers, branch=branch,
            ))
            if index + 1 < len(path):
                publications.append(node_id)
            names.append(node_id)
            ancestry.append(node_id)
            previous = name
        branches[branch] = names
    for node in nodes:
        u, f = coordinate(node["coordinate"])
        node.update(
            u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(node["coordinate"], "initialization"),
            sampler_seed=paired_seed(node["coordinate"], "sampler"),
            deployable=node["coordinate"] == "D000",
        )
    splits = foundation["splits"]
    result = artifact(
        "SALIENCE_MT20_CAMPAIGN_PLAN",
        foundation_sha256=parent,
        matcher_sha256=foundation["matcher"]["content_hash"],
        candidate=foundation["candidate"],
        inputs_sha256=foundation["inputs"]["content_hash"],
        recipe=mt20_recipe(), nodes=nodes, branches=branches,
        probability_publications=publications,
        fresh_fit_count=len(nodes),
        probability_publication_count=len(publications),
        branch_order=list(BRANCH_ORDER), ultradense_present=False,
        teacher_policy="all_prior_same_spine_geometric_history_v1",
        teacher_order="nearest_first",
        immediate_loss_weight=_rational(IMMEDIATE_WEIGHT),
        historical_total_loss_weight=_rational(HISTORICAL_WEIGHT),
        historical_decay_ratio=_rational(Fraction(1, 2)),
        single_teacher_loss_weight=_rational(KD_WEIGHT),
        cross_spine_teachers=False, ensembles=False, weight_continuation=False,
        persistent_hlt=True, pure_offline_u000=False,
        ram_only_weighted_teacher_mixtures=True,
        durable_mixture_arrays=False,
        split_profile=splits["profile"],
        split_profile_sha256=splits["content_hash"],
        split_registry_sha256=splits["registry_sha256"],
        role_counts=splits["role_counts"],
        role_membership_sha256={
            role: membership["content_hash"]
            for role, membership in splits["memberships"].items()
        },
        imported_models=[], final_test_accessed=False, executable=True,
    )
    if (
        len(nodes) != 16 or len(publications) != 12
        or set(branches) != set(BRANCH_ORDER)
    ):
        raise AssertionError("JetClass2 salience MT20 graph census differs")
    return result


__all__ = [
    "BRANCH_ORDER", "CE_WEIGHT", "HISTORICAL_WEIGHT", "IMMEDIATE_WEIGHT",
    "KD_WEIGHT", "build_campaign_plan", "mt20_recipe", "teacher_loss_weights",
]
