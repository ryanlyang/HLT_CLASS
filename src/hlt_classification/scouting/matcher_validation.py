"""Matching-only synthetic, stress, solver-agreement, and coverage validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from hlt_classification.data.cache_contracts import with_content_hash
from .matcher_training import contextual_scores, contextual_scores_many, likelihood_scores
from .matching import (
    CandidateGraph, ParticleSet, build_candidate_graph, match_variant,
)

MATCHER_VALIDATION_CONTRACT = "hlt_classification_pmard_matcher_validation_v2"


def select_matcher_variant(reports: Iterable[dict[str, object]]) -> dict[str, object]:
    """Apply the frozen matching-only selector across held-out fold reports."""
    rows = tuple(reports)
    if not rows:
        raise ValueError("matcher selection requires held-out reports")
    candidates = []
    for variant in ("M1", "M2", "M3", "M4", "M5"):
        arms = [row["variants"][variant] for row in rows]
        eligible = all(
            bool(arm["passes_initial_99pct_lcb"])
            and float(arm["event_mixing_false_positive_rate"]) <= .001
            and float(arm["independent_perturbation_stability"]) >= .99
            and float(arm["synthetic"]["confidence_brier"]) <= .01
            for arm in arms
        )
        stress_upper = max(
            1.0 if arm["stress_false_match_interval"][1] is None
            else float(arm["stress_false_match_interval"][1])
            for arm in arms
        )
        coverage = float(np.mean([arm["native_coverage"] for arm in arms]))
        stability = float(np.mean([arm["independent_perturbation_stability"] for arm in arms]))
        solver_agreement = float(np.mean([arm["solver_consensus_fraction"] for arm in arms]))
        candidates.append({
            "variant": variant, "eligible": eligible, "stress_false_upper": stress_upper,
            "native_coverage": coverage, "stability": stability,
            "solver_agreement": solver_agreement,
        })
    eligible_rows = [row for row in candidates if row["eligible"]]
    selected = min(
        eligible_rows or [row for row in candidates if row["variant"] == "M5"],
        key=lambda row: (
            row["stress_false_upper"], -row["native_coverage"],
            -row["solver_agreement"], row["variant"],
        ),
    )
    selected_arms = [row["variants"][selected["variant"]] for row in rows]
    category_eligibility = {
        str(category): all(
            arm["synthetic"]["by_category"][str(category)]["precision_interval"][0] is not None
            and float(arm["synthetic"]["by_category"][str(category)]["precision_interval"][0]) >= .99
            for arm in selected_arms
        ) for category in range(5)
    }
    return {
        "selected_variant": selected["variant"],
        "selector_requirements_met": bool(eligible_rows),
        "category_eligibility": category_eligibility,
        "candidates": candidates,
    }


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        return (float("nan"), float("nan"))
    p = successes / trials; denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def synthetic_particle_pair(
    *, seed: int, particles: int = 40, angular_sigma: float = .003,
    response_sigma: float = .04, loss_fraction: float = .1,
    fake_fraction: float = .05, split_fraction: float = .03,
    merge_fraction: float = .03, category_confusion: float = .01,
    dense: bool = True,
) -> tuple[ParticleSet, ParticleSet, np.ndarray]:
    rng = np.random.default_rng(seed)
    categories = rng.choice(5, particles, p=(.05, .05, .52, .23, .15)).astype(np.int8)
    charge = np.where(categories < 3, rng.choice((-1.0, 1.0), particles), 0.0)
    centers = max(2, particles // 8)
    center_eta = rng.uniform(-2, 2, centers); center_phi = rng.uniform(-np.pi, np.pi, centers)
    cluster = rng.integers(0, centers, particles)
    width = .035 if dense else .25
    eta = center_eta[cluster] + rng.normal(0, width, particles)
    phi = (center_phi[cluster] + rng.normal(0, width, particles) + np.pi) % (2 * np.pi) - np.pi
    pt = np.exp(rng.uniform(np.log(.7), np.log(150), particles))
    mass = .14; px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    offline_p4 = np.stack((px, py, pz, energy), axis=1)
    offline_measurements = rng.normal(0, 1, (particles, 7))
    offline_measurements[:, 0] *= .02; offline_measurements[:, 2] *= .05
    offline_measurements[:, 4] = np.abs(offline_measurements[:, 4] * 3)
    offline_measurements[:, 5] = rng.integers(0, 8, particles)
    offline_measurements[:, 6] = rng.integers(0, 3, particles)
    offline_validity = np.repeat((categories < 3)[:, None], 7, axis=1)
    keep = rng.random(particles) >= loss_fraction
    kept = np.flatnonzero(keep)
    local_density = np.asarray([
        np.count_nonzero(
            (categories == categories[index])
            & (np.hypot(eta - eta[index], (phi - phi[index] + np.pi) % (2 * np.pi) - np.pi) < .10)
        ) for index in kept
    ])
    type_scale = np.where(categories[kept] < 3, 1.0, 1.8)
    density_scale = 1 + .08 * np.maximum(0, local_density - 1)
    response_width = response_sigma * type_scale * density_scale * (1 + 1 / np.sqrt(np.maximum(pt[kept], 1)))
    angular_width = angular_sigma * type_scale * density_scale
    hpt = pt[kept] * np.exp(rng.normal(0, response_width))
    heta = eta[kept] + rng.normal(0, angular_width)
    hphi = phi[kept] + rng.normal(0, angular_width)
    hpx, hpy, hpz = hpt * np.cos(hphi), hpt * np.sin(hphi), hpt * np.sinh(heta)
    he = np.sqrt(hpx * hpx + hpy * hpy + hpz * hpz + mass * mass)
    hlt_p4 = np.stack((hpx, hpy, hpz, he), axis=1)
    hlt_categories = categories[kept].copy(); hlt_charge = charge[kept].copy()
    truth = kept.astype(np.int16)
    confused = rng.random(len(kept)) < category_confusion
    hlt_categories[confused] = (hlt_categories[confused] + rng.integers(1, 5, confused.sum())) % 5
    hlt_charge[(hlt_categories >= 3)] = 0
    hlt_measurements = offline_measurements[kept] + rng.normal(0, (.003, .5, .008, .5, .8, .2, .2), (len(kept), 7))
    hlt_validity = offline_validity[kept] & (rng.random((len(kept), 7)) > .03)
    hlt_measurements[~hlt_validity] = 0

    split_indexes = np.flatnonzero(rng.random(len(hlt_p4)) < split_fraction)
    if len(split_indexes):
        fractions = rng.uniform(.25, .75, len(split_indexes))
        fragments = hlt_p4[split_indexes] * fractions[:, None]
        hlt_p4[split_indexes] *= (1 - fractions)[:, None]
        hlt_p4 = np.concatenate((hlt_p4, fragments))
        hlt_categories = np.concatenate((hlt_categories, hlt_categories[split_indexes]))
        hlt_charge = np.concatenate((hlt_charge, hlt_charge[split_indexes]))
        hlt_measurements = np.concatenate((hlt_measurements, hlt_measurements[split_indexes]))
        hlt_validity = np.concatenate((hlt_validity, hlt_validity[split_indexes]))
        truth[split_indexes] = -1; truth = np.concatenate((truth, np.full(len(split_indexes), -1, np.int16)))

    merge_candidates = list(np.flatnonzero(rng.random(len(hlt_p4)) < merge_fraction))
    removed: set[int] = set()
    for left in merge_candidates:
        if left in removed: continue
        partners = [right for right in range(left + 1, len(hlt_p4))
                    if right not in removed and hlt_categories[right] == hlt_categories[left]]
        if not partners: continue
        right = partners[0]; hlt_p4[left] += hlt_p4[right]; truth[left] = -1; removed.add(right)
    if removed:
        retain = np.asarray([index not in removed for index in range(len(hlt_p4))])
        hlt_p4, hlt_categories, hlt_charge = hlt_p4[retain], hlt_categories[retain], hlt_charge[retain]
        hlt_measurements, hlt_validity, truth = hlt_measurements[retain], hlt_validity[retain], truth[retain]

    fake_count = int(round(fake_fraction * particles))
    if fake_count:
        parent = rng.integers(0, particles, fake_count)
        fpt = pt[parent] * np.exp(rng.normal(0, .35, fake_count))
        feta = eta[parent] + rng.normal(0, .04, fake_count)
        fphi = phi[parent] + rng.normal(0, .04, fake_count)
        fpx, fpy, fpz = fpt * np.cos(fphi), fpt * np.sin(fphi), fpt * np.sinh(feta)
        fe = np.sqrt(fpx * fpx + fpy * fpy + fpz * fpz + mass * mass)
        fake_categories = categories[parent]; fake_charge = charge[parent]
        hlt_p4 = np.concatenate((hlt_p4, np.stack((fpx, fpy, fpz, fe), axis=1)))
        hlt_categories = np.concatenate((hlt_categories, fake_categories))
        hlt_charge = np.concatenate((hlt_charge, fake_charge))
        hlt_measurements = np.concatenate((hlt_measurements, np.zeros((fake_count, 7))))
        hlt_validity = np.concatenate((hlt_validity, np.zeros((fake_count, 7), bool)))
        truth = np.concatenate((truth, np.full(fake_count, -1, np.int16)))
    order = np.argsort(-np.hypot(hlt_p4[:, 0], hlt_p4[:, 1]), kind="stable")
    hlt = ParticleSet(
        hlt_p4[order], hlt_categories[order], hlt_charge[order],
        np.zeros(len(order), bool), hlt_measurements[order], hlt_validity[order],
    )
    offline = ParticleSet(
        offline_p4, categories, charge, np.zeros(particles, bool),
        offline_measurements, offline_validity,
    )
    return hlt, offline, truth[order]


def validate_contextual_matcher(
    model, native_graphs: Iterable[CandidateGraph], *, device: str = "cpu",
    threshold: float = .99, synthetic_jets: int = 200, seed: int = 8041,
    parents: dict[str, str], native_sampling: dict[str, object] | None = None,
) -> dict[str, object]:
    if synthetic_jets <= 0 or not 0 <= threshold <= 1:
        raise ValueError("matcher validation budget or threshold is invalid")
    variants = tuple(f"M{index}" for index in range(6))
    solver_agree = solver_total = 0
    counters = {name: {
        "truth": 0, "accepted": 0, "correct": 0, "stress_accepted": 0,
        "stress_correct": 0, "mixed_accepted": 0, "mixed_total": 0,
        "native_hlt": 0, "native_accepted": 0, "stable": 0, "stability_total": 0,
        "confidence_squared_error": 0.0, "confidence_rows": 0,
        "category_accepted": np.zeros(5, int), "category_correct": np.zeros(5, int),
    } for name in variants}

    def results(active_hlt, active_offline):
        graph = build_candidate_graph(active_hlt, active_offline)
        contextual = contextual_scores(model, graph, device=device)
        likelihood = likelihood_scores(model, graph)
        return graph, {
            name: match_variant(
                graph, name, contextual_scores=contextual,
                likelihood_scores=likelihood, threshold=threshold,
                assignment_calibrator=model.assignment_calibration,
            ) for name in variants
        }

    def perturb(particles: ParticleSet, rng: np.random.Generator, scale: float) -> ParticleSet:
        pt = np.hypot(particles.p4[:, 0], particles.p4[:, 1])
        phi = np.arctan2(particles.p4[:, 1], particles.p4[:, 0]) + rng.normal(0, scale, len(pt))
        eta = np.arcsinh(np.divide(
            particles.p4[:, 2], pt, out=np.zeros_like(pt), where=pt > 0,
        )) + rng.normal(0, scale, len(pt))
        shifted_pt = pt * np.exp(rng.normal(0, 2 * scale, len(pt)))
        px, py, pz = shifted_pt * np.cos(phi), shifted_pt * np.sin(phi), shifted_pt * np.sinh(eta)
        mass2 = np.maximum(0, particles.p4[:, 3] ** 2 - np.sum(particles.p4[:, :3] ** 2, axis=1))
        energy = np.sqrt(px * px + py * py + pz * pz + mass2)
        measurements = particles.measurements + rng.normal(0, scale, particles.measurements.shape)
        measurements[~particles.measurement_validity] = 0
        return ParticleSet(
            np.stack((px, py, pz, energy), axis=1), particles.categories,
            particles.charge, particles.lost_track, measurements,
            particles.measurement_validity,
        )

    for jet in range(synthetic_jets):
        hlt, offline, truth = synthetic_particle_pair(seed=seed + jet)
        _, nominal = results(hlt, offline)
        perturbation_rng = np.random.default_rng(seed + 50_000 + jet)
        _, perturbed = results(
            perturb(hlt, perturbation_rng, .0015),
            perturb(offline, perturbation_rng, .0008),
        )
        solver_mask = nominal["M4"].accepted | nominal["M5"].accepted
        solver_agree += int(np.count_nonzero(
            solver_mask
            & (nominal["M4"].hlt_to_offline == nominal["M5"].hlt_to_offline)
        )); solver_total += int(np.count_nonzero(solver_mask))
        for name in variants:
            result = nominal[name]; correct = result.hlt_to_offline == truth; row = counters[name]
            row["truth"] += int(np.count_nonzero(truth >= 0)); row["accepted"] += int(result.accepted.sum())
            row["correct"] += int(np.count_nonzero(result.accepted & correct))
            row["stable"] += int(np.count_nonzero(
                result.accepted
                & (result.hlt_to_offline == perturbed[name].hlt_to_offline)
            ))
            row["stability_total"] += int(result.accepted.sum())
            confidence_target = correct[result.accepted].astype(np.float64)
            row["confidence_squared_error"] += float(np.square(
                result.confidence[result.accepted] - confidence_target
            ).sum())
            row["confidence_rows"] += int(result.accepted.sum())
            for i in np.flatnonzero(result.accepted):
                category = int(hlt.categories[i]); row["category_accepted"][category] += 1
                row["category_correct"][category] += int(correct[i])
        stress_hlt, stress_offline, stress_truth = synthetic_particle_pair(
            seed=seed + 10_000 + jet, angular_sigma=.010, response_sigma=.14,
            loss_fraction=.25, fake_fraction=.15, split_fraction=.12,
            merge_fraction=.10, category_confusion=.04, dense=True,
        )
        _, stress_results = results(stress_hlt, stress_offline)
        for name, result in stress_results.items():
            counters[name]["stress_accepted"] += int(result.accepted.sum())
            counters[name]["stress_correct"] += int(np.count_nonzero(
                result.accepted & (result.hlt_to_offline == stress_truth)
            ))
        source_scale = math.log(max(1.0e-6, float(np.hypot(
            hlt.p4[:, 0], hlt.p4[:, 1],
        ).sum())))
        mixed_pool = [
            synthetic_particle_pair(
                seed=seed + 1_000_000 + 8 * jet + candidate,
                particles=len(offline.p4), dense=True,
            )[1]
            for candidate in range(8)
        ]
        mixed = min(
            mixed_pool,
            key=lambda candidate: abs(math.log(max(1.0e-6, float(np.hypot(
                candidate.p4[:, 0], candidate.p4[:, 1],
            ).sum()))) - source_scale),
        )
        _, mixed_results = results(hlt, mixed)
        for name, result in mixed_results.items():
            counters[name]["mixed_accepted"] += int(result.accepted.sum())
            counters[name]["mixed_total"] += len(hlt.p4)
    native_materialized = list(native_graphs)
    native_contextual = contextual_scores_many(model, native_materialized, device=device)
    for graph, contextual in zip(native_materialized, native_contextual, strict=True):
        likelihood = likelihood_scores(model, graph)
        for name in variants:
            result = match_variant(
                graph, name, contextual_scores=contextual,
                likelihood_scores=likelihood, threshold=threshold,
                assignment_calibrator=model.assignment_calibration,
            )
            counters[name]["native_hlt"] += graph.hlt_count
            counters[name]["native_accepted"] += int(result.accepted.sum())
    solver_fraction = solver_agree / max(1, solver_total)
    reports = {}
    for name, row in counters.items():
        accepted = int(row["accepted"]); correct = int(row["correct"]); truth_total = int(row["truth"])
        lower, upper = wilson_interval(correct, accepted)
        category = {}
        category_pass = True
        for index in range(5):
            category_accepted = int(row["category_accepted"][index]); category_correct = int(row["category_correct"][index])
            interval = wilson_interval(category_correct, category_accepted)
            category_pass &= bool(category_accepted > 0 and interval[0] >= .99)
            category[str(index)] = {
                "accepted": category_accepted,
                "precision": category_correct / category_accepted if category_accepted else 0.0,
                "precision_interval": list(interval) if category_accepted else [None, None],
            }
        stress_accepted = int(row["stress_accepted"]); stress_correct = int(row["stress_correct"])
        stress_false = stress_accepted - stress_correct
        reports[name] = {
            "synthetic": {"truth": truth_total, "accepted": accepted, "correct": correct,
                          "precision": correct / accepted if accepted else 0.0,
                          "precision_interval": [lower, upper],
                          "recall": correct / truth_total if truth_total else 0.0,
                          "by_category": category,
                          "confidence_brier": row["confidence_squared_error"] / max(1, row["confidence_rows"])},
            "stress_false_match_interval": (
                list(wilson_interval(stress_false, stress_accepted))
                if stress_accepted else [None, None]
            ),
            "stress_precision": stress_correct / stress_accepted if stress_accepted else 0.0,
            "event_mixing_false_positive_rate": row["mixed_accepted"] / max(1, row["mixed_total"]),
            "independent_perturbation_stability": row["stable"] / max(1, row["stability_total"]),
            "solver_consensus_fraction": solver_fraction,
            "native_coverage": row["native_accepted"] / max(1, row["native_hlt"]),
            "passes_initial_99pct_lcb": bool(lower >= .99 and category_pass),
        }
    primary = reports["M5"]
    return with_content_hash({
        "contract": MATCHER_VALIDATION_CONTRACT, "schema_version": 2,
        "parents": dict(parents), "threshold": threshold,
        "synthetic_jets": synthetic_jets, "synthetic_seed": seed,
        "native_sampling": dict(native_sampling or {"mode": "caller_supplied_graphs"}),
        "variants": reports, "synthetic": primary["synthetic"],
        "stress_precision": primary["stress_precision"],
        "event_mixing_false_positive_rate": primary["event_mixing_false_positive_rate"],
        "solver_consensus_fraction": solver_fraction,
        "native_coverage": primary["native_coverage"],
        "passes_initial_99pct_lcb": primary["passes_initial_99pct_lcb"],
        "downstream_classifier_or_label_used": False,
    })


__all__ = [
    "select_matcher_variant", "synthetic_particle_pair",
    "validate_contextual_matcher", "wilson_interval",
]
