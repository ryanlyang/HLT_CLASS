"""Matching-only synthetic, stress, solver-agreement, and coverage validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from hlt_classification.data.cache_contracts import with_content_hash
from .matcher_training import contextual_scores, likelihood_scores
from .matching import (
    CandidateGraph, ParticleSet, build_candidate_graph, match_variant,
)

MATCHER_VALIDATION_CONTRACT = "hlt_classification_pmard_matcher_validation_v1"


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
            and float(arm["rotation_stability"]) >= .99
            for arm in arms
        )
        stress_upper = max(
            1.0 if arm["stress_false_match_interval"][1] is None
            else float(arm["stress_false_match_interval"][1])
            for arm in arms
        )
        coverage = float(np.mean([arm["native_coverage"] for arm in arms]))
        stability = float(np.mean([arm["rotation_stability"] for arm in arms]))
        candidates.append({
            "variant": variant, "eligible": eligible, "stress_false_upper": stress_upper,
            "native_coverage": coverage, "stability": stability,
        })
    eligible_rows = [row for row in candidates if row["eligible"]]
    selected = min(
        eligible_rows or [row for row in candidates if row["variant"] == "M5"],
        key=lambda row: (
            row["stress_false_upper"], -row["native_coverage"],
            -row["stability"], row["variant"],
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
) -> tuple[ParticleSet, ParticleSet, np.ndarray]:
    rng = np.random.default_rng(seed)
    categories = np.arange(particles, dtype=np.int8) % 5
    charge = np.where(categories < 3, rng.choice((-1.0, 1.0), particles), 0.0)
    pt = rng.uniform(1, 100, particles); eta = rng.uniform(-2, 2, particles); phi = rng.uniform(-np.pi, np.pi, particles)
    mass = .14; px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    offline_p4 = np.stack((px, py, pz, energy), axis=1)
    keep = rng.random(particles) >= loss_fraction
    kept = np.flatnonzero(keep)
    hpt = pt[kept] * np.exp(rng.normal(0, response_sigma, len(kept)))
    heta = eta[kept] + rng.normal(0, angular_sigma, len(kept))
    hphi = phi[kept] + rng.normal(0, angular_sigma, len(kept))
    hpx, hpy, hpz = hpt * np.cos(hphi), hpt * np.sin(hphi), hpt * np.sinh(heta)
    he = np.sqrt(hpx * hpx + hpy * hpy + hpz * hpz + mass * mass)
    hlt = ParticleSet(np.stack((hpx, hpy, hpz, he), axis=1), categories[kept], charge[kept], np.zeros(len(kept), bool))
    offline = ParticleSet(offline_p4, categories, charge, np.zeros(particles, bool))
    return hlt, offline, kept.astype(np.int16)


def validate_contextual_matcher(
    model, native_graphs: Iterable[CandidateGraph], *, device: str = "cpu",
    threshold: float = .99, synthetic_jets: int = 200, seed: int = 8041,
    parents: dict[str, str],
) -> dict[str, object]:
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
            ) for name in variants
        }

    def rotate(particles: ParticleSet, angle: float) -> ParticleSet:
        cosine, sine = math.cos(angle), math.sin(angle); p4 = particles.p4.copy()
        px, py = p4[:, 0].copy(), p4[:, 1].copy()
        p4[:, 0] = cosine * px - sine * py; p4[:, 1] = sine * px + cosine * py
        return ParticleSet(p4, particles.categories, particles.charge, particles.lost_track)

    for jet in range(synthetic_jets):
        hlt, offline, truth = synthetic_particle_pair(seed=seed + jet)
        _, nominal = results(hlt, offline)
        _, rotated = results(rotate(hlt, .37), rotate(offline, .37))
        solver_agree += int(np.count_nonzero(
            nominal["M4"].hlt_to_offline == nominal["M5"].hlt_to_offline
        )); solver_total += len(truth)
        for name in variants:
            result = nominal[name]; correct = result.hlt_to_offline == truth; row = counters[name]
            row["truth"] += len(truth); row["accepted"] += int(result.accepted.sum())
            row["correct"] += int(np.count_nonzero(result.accepted & correct))
            row["stable"] += int(np.count_nonzero(result.hlt_to_offline == rotated[name].hlt_to_offline))
            row["stability_total"] += len(truth)
            confidence_target = (result.accepted & correct).astype(np.float64)
            row["confidence_squared_error"] += float(np.square(result.confidence - confidence_target).sum())
            row["confidence_rows"] += len(truth)
            for i in np.flatnonzero(result.accepted):
                category = int(hlt.categories[i]); row["category_accepted"][category] += 1
                row["category_correct"][category] += int(correct[i])
        stress_hlt, stress_offline, stress_truth = synthetic_particle_pair(
            seed=seed + 10_000 + jet, angular_sigma=.008, response_sigma=.10, loss_fraction=.2,
        )
        _, stress_results = results(stress_hlt, stress_offline)
        for name, result in stress_results.items():
            counters[name]["stress_accepted"] += int(result.accepted.sum())
            counters[name]["stress_correct"] += int(np.count_nonzero(
                result.accepted & (result.hlt_to_offline == stress_truth)
            ))
        mixed_p4 = offline.p4.copy(); mixed_p4[:, :2] *= -1
        mixed = ParticleSet(mixed_p4, offline.categories, offline.charge, offline.lost_track)
        _, mixed_results = results(hlt, mixed)
        for name, result in mixed_results.items():
            counters[name]["mixed_accepted"] += int(result.accepted.sum())
            counters[name]["mixed_total"] += len(hlt.p4)
    for graph in native_graphs:
        contextual = contextual_scores(model, graph, device=device); likelihood = likelihood_scores(model, graph)
        for name in variants:
            result = match_variant(
                graph, name, contextual_scores=contextual,
                likelihood_scores=likelihood, threshold=threshold,
            )
            counters[name]["native_hlt"] += graph.hlt_count
            counters[name]["native_accepted"] += int(result.accepted.sum())
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
            "rotation_stability": row["stable"] / max(1, row["stability_total"]),
            "native_coverage": row["native_accepted"] / max(1, row["native_hlt"]),
            "passes_initial_99pct_lcb": bool(lower >= .99 and category_pass),
        }
    primary = reports["M5"]
    return with_content_hash({
        "contract": MATCHER_VALIDATION_CONTRACT, "schema_version": 1,
        "parents": dict(parents), "threshold": threshold,
        "variants": reports, "synthetic": primary["synthetic"],
        "stress_precision": primary["stress_precision"],
        "event_mixing_false_positive_rate": primary["event_mixing_false_positive_rate"],
        "solver_consensus_fraction": solver_agree / max(1, solver_total),
        "native_coverage": primary["native_coverage"],
        "passes_initial_99pct_lcb": primary["passes_initial_99pct_lcb"],
        "downstream_classifier_or_label_used": False,
    })


__all__ = [
    "select_matcher_variant", "synthetic_particle_pair",
    "validate_contextual_matcher", "wilson_interval",
]
