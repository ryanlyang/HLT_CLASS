"""Paired source-file uncertainty and immutable, non-classifier selection.

Replica metrics are averaged within each source-block bootstrap. A replica is
never an additional observed CMS jet or an independent source group.
"""
from __future__ import annotations

import numpy as np

from .contracts import artifact, candidates, validate
from .metrics import BLOCKS, CATEGORICAL, Summary

LIMITS = {**{f"block:{b}": .10 for b in BLOCKS}, "conditional_max": .25,
          "identity_max": .05, "correlation_max": .10,
          "bias:multiplicity": .05, "bias:pt": .05, "bias:mass": .05,
          "association_unresolved": .01}


def qualification_status(tests):
    states = {r["state"] for r in tests.values()}
    if states == {"pass"}:
        return "qualified"
    return "unqualified" if "fail" in states else "inconclusive"


def bootstrap_weights(groups: list[str]) -> np.ndarray:
    if groups != sorted(set(groups)) or not groups:
        raise ValueError("Bootstrap source groups must be unique and canonical")
    rng = np.random.default_rng(20260918)
    draws = rng.integers(len(groups), size=(200, len(groups)))
    return np.asarray([np.bincount(row, minlength=len(groups)) for row in draws])


def _average(reports: list[dict], conditional_scores: list[float], calibration_coverage: float):
    mean = lambda values: float(np.mean(values))
    result = {"score": mean([r["primary_score"] for r in reports]),
              **{f"block:{b}": mean([r["blocks"][b] for r in reports]) for b in BLOCKS}}
    result["identity_max"] = max(mean([r["observables"][n]["discrepancy"] for r in reports])
                                   for n in CATEGORICAL)
    result["correlation_max"] = max(mean([r["observables"][n]["discrepancy"] for r in reports])
                                      for n in reports[0]["observables"] if "_corr_" in n)
    result["conditional_max"] = max(conditional_scores, default=0.)
    result["association_unresolved"] = max(1-calibration_coverage,
        *[1-r[k] for r in reports for k in ("reference_association_coverage", "proxy_association_coverage")])
    for name in ("multiplicity", "pt", "mass"):
        values = [r["mean_jet_biases"][name]["relative"] for r in reports]
        result["bias:"+name] = None if any(v is None for v in values) else abs(mean(values))
    return result


def compare_groups(real: dict[str, dict[str, Summary]], proxies: dict[int, dict[str, dict[str, Summary]]],
                   *, calibration_coverage: float, membership_hash: str, fitted_hash: str,
                   candidate_id: str, budget: str, gate: float, role: str) -> dict:
    """Inputs are source -> condition -> summary; 'all' is mandatory.

    Conditional cells below 1000 observed jets explicitly back off to 'all',
    which is reported once, not multiplied by the number of sparse cells.
    """
    if role not in {"response_select", "response_confirm", "development"}:
        raise PermissionError("Unknown comparison role")
    groups = sorted(real)
    if not groups or not 0 <= calibration_coverage <= 1:
        raise ValueError("Evaluation population/calibration coverage differs")
    replicas = sorted(proxies)
    if replicas != ([0, 1, 2, 3, 4] if role == "response_confirm" else [0, 1, 2]):
        raise ValueError("Replica registry differs")
    conditions = sorted(set.union(*(set(real[g]) for g in groups)))
    if "all" not in conditions or any("all" not in real[g] for g in groups):
        raise ValueError("Every file requires unconditional coverage")
    registry = real[groups[0]]["all"].registry
    for replica in replicas:
        if sorted(proxies[replica]) != groups:
            raise ValueError("Proxy source groups differ from real CMS")
        for g in groups:
            if set(proxies[replica][g]) != set(real[g]):
                raise ValueError("Conditional coverage registry differs")
            for condition in real[g]:
                if real[g][condition].jets != proxies[replica][g][condition].jets:
                    raise ValueError("Paired evaluation membership differs")
    populations = {c: sum(real[g][c].jets for g in groups if c in real[g]) for c in conditions}
    eligible = [c for c in conditions if c != "all" and populations[c] >= 1000]
    cache = {}
    conditional_backoff = {}

    def evaluate(weights):
        key = tuple(map(int, weights))
        if key in cache:
            return cache[key]
        reports, conditional = [], []
        uncertainty = {name: 0. for name in LIMITS}
        errors = []
        for c in ["all", *eligible]:
            a = Summary(registry)
            for g, w in zip(groups, key):
                if c in real[g]:
                    a.merge(real[g][c], w)
            if not a.jets:
                # Bootstrap can omit the only file supporting a rare cell.
                continue
            cell = []
            for replica in replicas:
                b = Summary(registry)
                for g, w in zip(groups, key):
                    if c in proxies[replica][g]:
                        b.merge(proxies[replica][g][c], w)
                row = a.compare(b)
                if c != "all":
                    fallback = []
                    for name, value in row["observables"].items():
                        if value["covered_jets"] < 1000:
                            row["observables"][name] = reports[replica]["observables"][name]
                            fallback.append(name)
                    row["blocks"] = {block: float(np.mean([row["observables"][n]["discrepancy"] for n in names]))
                                     for block, names in BLOCKS.items()}
                    row["block_error_bounds"] = {block: float(np.mean([row["observables"][n]["error_bound"] for n in names]))
                                                 for block, names in BLOCKS.items()}
                    row["primary_score"] = float(np.mean(list(row["blocks"].values())))
                    if key == (1,)*len(groups) and replica == 0:
                        conditional_backoff[c] = fallback
                cell.append(row)
            if c == "all":
                reports = cell
                errors = [float(np.mean(list(r["block_error_bounds"].values()))) for r in cell]
                for b in BLOCKS:
                    uncertainty[f"block:{b}"] = float(np.mean([r["block_error_bounds"][b] for r in cell]))
                uncertainty["correlation_max"] = max(float(np.mean([
                    r["observables"][n]["error_bound"] for r in cell])) for n in cell[0]["observables"]
                    if "_corr_" in n)
            else:
                conditional.append(float(np.mean([r["primary_score"] for r in cell])))
                uncertainty["conditional_max"] = max(uncertainty["conditional_max"],
                    float(np.mean([np.mean(list(r["block_error_bounds"].values())) for r in cell])))
        value = _average(reports, conditional, calibration_coverage)
        # Equal conditional-cell weighting comes after equal observable/block
        # weighting. Include 'all' once so rare/unmatched jets retain a voice.
        if conditional:
            value["score"] = float(np.mean([value["score"], *conditional]))
        cache[key] = (value, float(np.mean(errors)), uncertainty)
        return cache[key]

    point, approximation, numerical = evaluate(np.ones(len(groups), int))
    draws = bootstrap_weights(groups)
    bootstrap = [evaluate(w)[0] for w in draws]
    intervals = {}
    for name in point:
        values = [r[name] for r in bootstrap]
        intervals[name] = (None if any(v is None for v in values) else
                           [float(x) for x in np.quantile(values, [.025, .975])])
    standard_error = float(np.std([r["score"] for r in bootstrap], ddof=1))
    tests = {}
    for name, threshold in LIMITS.items():
        value, interval = point[name], intervals[name]
        state = ("undefined" if value is None else "fail" if value > threshold else
                 "inconclusive" if interval is None or interval[1]+numerical[name] > threshold else "pass")
        tests[name] = dict(value=value, maximum=threshold, interval=interval, state=state,
                           summary_error_bound=numerical[name])
    return artifact("EVALUATION", parents={"membership": membership_hash,
                    "response": fitted_hash, "metric_registry": registry["content_hash"]},
                    candidate_id=candidate_id, budget=budget, gate=gate, role=role,
                    point=point, intervals=intervals, standard_error=standard_error,
                    bootstrap_scores=[r["score"] for r in bootstrap], source_groups=groups,
                    independent_source_groups=len(groups), weak_source_group_coverage=len(groups) < 10,
                    real_jets=populations["all"], replicas=replicas, bootstrap_count=200,
                    conditional_populations=populations, eligible_conditions=eligible,
                    conditional_observable_backoff_to_all=conditional_backoff,
                    sparse_conditions={c: "backoff_to_all" for c in conditions if c not in ["all", *eligible]},
                    overall_score_approximation_bound=approximation,
                    qualification=tests, scientific_status=qualification_status(tests),
                    qualified=all(r["state"] == "pass" for r in tests.values()))


def family_finalists(evaluations: list[dict]) -> dict:
    expected = {r["id"] for r in candidates()["candidates"]}
    rows = {}
    for value in evaluations:
        validate(value, "EVALUATION")
        if value["budget"] == "FULL" and value["gate"] == .1:
            if value["candidate_id"] in rows:
                raise ValueError("Duplicate FULL candidate")
            rows[value["candidate_id"]] = value
    if set(rows) != expected:
        raise ValueError("All nine FULL comparisons required, including poor fits")
    if len({(r["parents"]["membership"], r["parents"]["metric_registry"], tuple(r["source_groups"]))
            for r in rows.values()}) != 1 or any(r["role"] != "response_select" for r in rows.values()):
        raise ValueError("Selection comparisons are not paired on the frozen selection role")
    return {family: min((r for key, r in rows.items() if key.startswith(family)),
                        key=lambda r: (r["point"]["score"], r["candidate_id"]))["candidate_id"]
            for family in "ABC"}


def select(evaluations: list[dict], sensitivities: list[dict], *, registry_hash: str) -> dict:
    if registry_hash != candidates()["content_hash"]:
        raise ValueError("Candidate registry differs")
    for r in evaluations:
        validate(r, "EVALUATION")
        if r["role"] != "response_select" or r["gate"] != .1:
            raise ValueError("All primaries require the frozen selection role and primary gate")
    finalists = family_finalists(evaluations)
    if len(evaluations) != 27 or {(r["candidate_id"], r["budget"]) for r in evaluations} != {
        (r["id"], b) for r in candidates()["candidates"] for b in candidates()["budgets"]}:
        raise ValueError("All 27 registered fits must finish before selection")
    expected = {(c, g) for c in finalists.values() for g in (.05, .2)}
    if len(sensitivities) != 6 or {(r["candidate_id"], r["gate"]) for r in sensitivities} != expected:
        raise ValueError("Six family-finalist association refits must be disclosed")
    full = {r["candidate_id"]: r for r in evaluations if r["budget"] == "FULL"}
    for r in sensitivities:
        validate(r, "EVALUATION")
        primary = full[r["candidate_id"]]
        if (r["budget"] != "FULL" or r["role"] != "response_select" or
            any(r["parents"][k] != primary["parents"][k] for k in ("membership", "metric_registry"))):
            raise ValueError("Sensitivity comparison lineage differs")
    best = min(full.values(), key=lambda r: (r["point"]["score"], r["candidate_id"]))
    cutoff = best["point"]["score"]+best["standard_error"]
    eligible = [r["id"] for r in candidates()["candidates"]
                if full[r["id"]]["point"]["score"] <= cutoff and full[r["id"]]["qualified"]]
    chosen = eligible[0] if eligible else best["candidate_id"]
    robust = True
    checks = []
    for row in sensitivities:
        primary = full[row["candidate_id"]]
        base = primary["point"]["score"]
        delta = abs(row["point"]["score"]-base)
        stable = delta <= .2*base and row["qualified"] == primary["qualified"]
        checks.append(dict(candidate=row["candidate_id"], gate=row["gate"], stable=stable,
                           absolute_score_change=delta))
        if row["candidate_id"] == chosen:
            robust &= stable
    # The one-SE simplicity winner need not be the family minimum. In that case
    # six best-per-family sensitivities do not certify its own gate robustness.
    sensitivity_covered = chosen in finalists.values()
    qualified = bool(eligible and robust and sensitivity_covered)
    status = qualification_status(full[chosen]["qualification"])
    if not qualified and status == "qualified":
        status = "unqualified" if sensitivity_covered and not robust else "inconclusive"
    return artifact("SELECTION", parents={"candidate_registry": registry_hash,
                    **{f"primary_{i}": r["content_hash"] for i, r in enumerate(evaluations)},
                    **{f"sensitivity_{i}": r["content_hash"] for i, r in enumerate(sensitivities)}},
                    selected_candidate=chosen, response_hash=full[chosen]["parents"]["response"],
                    best_score_candidate=best["candidate_id"], one_standard_error_cutoff=cutoff,
                    simplicity_eligible=eligible, family_finalists=finalists, sensitivity_checks=checks,
                    selected_sensitivity_covered=sensitivity_covered,
                    association_robust=bool(robust and sensitivity_covered),
                    qualified=qualified, scientific_status=status,
                    classifier_metrics_used=False, confirmation_used=False, jc2_used=False)
