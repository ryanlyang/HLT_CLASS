"""Registered task graph and a deterministic combined CPU admission bound.

This is not a live submitter. A science graph needs separate real CPU evidence
and source-bound execution locks before it may become an executable plan.
"""
from __future__ import annotations

from .contracts import artifact, candidates

RESOURCES = {
    "fit": dict(cpus=16, memory_gib=128, time="24:00:00"),
    "evaluate": dict(cpus=8, memory_gib=32, time="08:00:00"),
    "metrics": dict(cpus=16, memory_gib=128, time="24:00:00"),
    "report": dict(cpus=2, memory_gib=8, time="02:00:00"),
    "visual": dict(cpus=1, memory_gib=32, time="08:00:00"),
}


def science_graph() -> dict:
    registry = candidates()
    rows = []
    fit_lanes = [None]*2
    eval_lanes = [None]*4

    def add(task, kind, depends, **params):
        rows.append(dict(task_id=task, kind=kind, depends_on=sorted(set(depends)), params=params,
                         resources=RESOURCES[kind]))

    add("metric_lock", "metrics", [])
    evaluations = []

    def pair(name, index, parent, **params):
        fit, evaluate = "fit_"+name, "evaluate_"+name
        f, e = index % 2, index % 4
        dependencies = [parent, *([fit_lanes[f]] if fit_lanes[f] else [])]
        add(fit, "fit", dependencies, **params)
        add(evaluate, "evaluate", [fit, *([eval_lanes[e]] if eval_lanes[e] else [])], fit_task=fit)
        fit_lanes[f], eval_lanes[e] = fit, evaluate
        evaluations.append(evaluate)

    index = 0
    for candidate in registry["candidates"]:
        for budget in registry["budgets"]:
            pair(f"{candidate['id']}_{budget}", index, "metric_lock",
                 candidate_id=candidate["id"], budget=budget, gate=.1)
            index += 1
    add("family_finalists", "report", evaluations.copy())
    for family in "ABC":
        for gate in (.05, .2):
            pair(f"{family}_G{int(100*gate):02d}", index, "family_finalists",
                 family_finalist=family, budget="FULL", gate=gate)
            index += 1
    add("selection_lock", "report", evaluations.copy())
    add("visual_select", "visual", ["selection_lock"])
    add("comparison_complete", "report", ["selection_lock", "visual_select"])
    return artifact("SCIENCE_GRAPH", parents={"candidates": registry["content_hash"]}, tasks=rows,
                    primary_fits=27, sensitivity_fits=6, combined_cpu_upper_bound=64,
                    admission_policy="two_16_cpu_fit_lanes_plus_four_8_cpu_evaluation_lanes",
                    maximum_simultaneous_fits=2, maximum_simultaneous_evaluations=4,
                    metrics_and_report_barriers=True, quality_thresholds_are_not_dependencies=True,
                    confirmation_requires_new_claim=True)


def validate_graph(graph: dict):
    if graph != science_graph():
        raise ValueError("Scientific task graph differs from the registered comparison")
    seen = set()
    for row in graph["tasks"]:
        if row["task_id"] in seen or not set(row["depends_on"]) <= seen:
            raise ValueError("Invalid task ordering or dependency")
        seen.add(row["task_id"])
    return graph["content_hash"]
