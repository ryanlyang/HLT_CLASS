"""HCWDL-RKD screen, confirmation, diagnostic and final aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    with_content_hash,
)

from .hcwdl_representation_graph import NODE_REGISTRY
from .hcwdl_paired_bootstrap import BASE_METRICS


TRAINING_REPORT_CONTRACT: Final = "HCWDL_REPRESENTATION_TRAINING_REPORT/v1"
SCREEN_CONTRACT: Final = "HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v1"
CONFIRMATION_REGISTRY_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v1"
CONFIRMATION_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1"
CONFIRMATION_RUN_CONTRACT: Final = "HCWDL_REPRESENTATION_CONFIRMATION_RUN/v1"
FINAL_AGGREGATE_CONTRACT: Final = "HCWDL_REPRESENTATION_FINAL_AGGREGATE/v1"
VALIDATION_ONLY_AGGREGATE_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1"
)
CONFIRMATION_SEEDS: Final = (11, 22, 33, 44, 55)
T_975_DF4: Final = 2.7764451051977987
CONFIRMATION_METRICS: Final = BASE_METRICS
GAP_METRIC_DIRECTIONS: Final = {
    "cross_entropy": False,
    "macro_ovr_auc": True,
    "macro_mean_log_qcd_rejection_at_50pct_signal": True,
}


def derive_representation_execution_id(
    *,
    campaign_sha256: str,
    strategy: str,
    node_id: str,
    purpose: str,
    seed: int,
    initialization_parent: str | None,
    teacher: str,
    logical_target_bank_sha256: str,
    target_purpose: str,
    recipe_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Return the exact Section 24.4 execution identity and audited payload."""

    if strategy not in {"HCWDL_REP_SET/v1", "HCWDL_REP_REL/v1"}:
        raise ValueError("representation execution strategy differs")
    if not node_id or not teacher:
        raise ValueError("representation execution node/teacher differs")
    if purpose not in {"screen", "confirmation"} or target_purpose != purpose:
        raise ValueError("representation execution purpose differs")
    if isinstance(seed, bool) or int(seed) <= 0:
        raise ValueError("representation execution seed differs")
    if initialization_parent is not None and not initialization_parent:
        raise ValueError("representation initialization parent differs")
    payload = {
        "campaign": require_sha256(campaign_sha256, name="campaign"),
        "strategy": strategy,
        "node_id": node_id,
        "purpose": purpose,
        "seed": int(seed),
        "initialization_parent": initialization_parent,
        "teacher": teacher,
        "logical_target_bank": require_sha256(
            logical_target_bank_sha256, name="logical target bank",
        ),
        "target_purpose": target_purpose,
        "recipe": require_sha256(recipe_sha256, name="representation recipe"),
    }
    return canonical_sha256(payload), payload


def checkpoint_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Frozen macro-AUC, CE, log-R50, earliest-update, checkpoint-ID order."""

    metrics = row["validation"]
    auc = float(metrics["macro_ovr_auc"])
    ce = float(metrics["cross_entropy"])
    logr = float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
    update = int(row["update"])
    checkpoint_id = str(row["checkpoint_id"])
    if not all(math.isfinite(value) for value in (auc, ce, logr)):
        raise FloatingPointError("checkpoint selector input is nonfinite")
    if update < 0 or not checkpoint_id:
        raise ValueError("checkpoint selector identity differs")
    return (-auc, ce, -logr, update, checkpoint_id)


def select_checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("checkpoint selector requires validation rows")
    selected = min(rows, key=checkpoint_key)
    return {
        "selected_checkpoint_id": str(selected["checkpoint_id"]),
        "selected_update": int(selected["update"]),
        "ordering": [str(row["checkpoint_id"]) for row in sorted(rows, key=checkpoint_key)],
    }


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("validation")
    if not isinstance(metrics, Mapping):
        raise ValueError("representation report lacks validation metrics")
    for key in BASE_METRICS:
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"representation validation metric {key!r} differs")
    return metrics


def _delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(left[key]) - float(right[key])
        for key in (
            "cross_entropy",
            "accuracy",
            "balanced_accuracy",
            "macro_ovr_auc",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
            "multiclass_brier",
            "top_label_ece_15_bin",
        )
        if key in left and key in right and left[key] is not None and right[key] is not None
    }


def gap_recovery(
    *, student: float, lower: float, upper: float, higher_is_better: bool,
) -> dict[str, Any]:
    """Compute the frozen un-clipped gap fraction or an explicit undefined row."""

    values = (float(student), float(lower), float(upper))
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("gap-recovery input is nonfinite")
    denominator = upper - lower if higher_is_better else lower - upper
    numerator = student - lower if higher_is_better else lower - student
    if denominator <= 0:
        return {
            "value": None,
            "reason": "nonpositive_declared_oracle_denominator",
            "numerator": numerator,
            "denominator": denominator,
            "higher_is_better": bool(higher_is_better),
        }
    return {
        "value": numerator / denominator,
        "reason": None,
        "numerator": numerator,
        "denominator": denominator,
        "higher_is_better": bool(higher_is_better),
    }


def build_screen_aggregate(
    *,
    primary_reports: Sequence[Mapping[str, Any]],
    control_reports: Sequence[Mapping[str, Any]],
    parent_reports: Mapping[str, Mapping[str, Any]],
    graph_sha256: str,
    recipe_sha256: str,
    campaign_spec_sha256: str,
    expected_primary_ids: Sequence[str],
    expected_control_ids: Sequence[str],
) -> dict[str, Any]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for report in (*primary_reports, *control_reports):
        node_id = str(report.get("node_id", ""))
        if not node_id or node_id in by_id:
            raise ValueError("representation screen repeats or omits a node identity")
        if report.get("complete") is not True:
            raise ValueError("finite representation result must be terminally complete")
        if report.get("graph_sha256") != graph_sha256 or report.get("recipe_sha256") != recipe_sha256:
            raise ValueError("representation report lineage differs")
        _metrics(report)
        by_id[node_id] = report
    primary = tuple(expected_primary_ids)
    controls = tuple(expected_control_ids)
    if set(by_id) != set(primary) | set(controls):
        raise ValueError("representation screen registry is incomplete")
    if set(parent_reports) == set():
        raise ValueError("representation screen requires authenticated parent controls")

    comparisons: list[dict[str, Any]] = []

    def add_comparison(left_id: str, right_id: str, *, kind: str) -> None:
        left_report = by_id[left_id]
        right_report = by_id.get(right_id, parent_reports.get(right_id))
        if right_report is None:
            raise ValueError(f"ordered comparison lacks {right_id!r}")
        comparisons.append(
            {
                "comparison": f"{left_id}-minus-{right_id}",
                "kind": kind,
                "left": left_id,
                "right": right_id,
                "delta": _delta(_metrics(left_report), _metrics(right_report)),
            }
        )

    for node_id in primary:
        report = by_id[node_id]
        parent_id = str(report.get("parent_counterpart", ""))
        parent = parent_reports.get(parent_id)
        if parent is None:
            raise ValueError(f"missing parent counterpart {parent_id!r}")
        if NODE_REGISTRY[node_id].parent_counterpart != parent_id:
            raise ValueError("representation report parent counterpart differs")
        add_comparison(node_id, parent_id, kind="representation_minus_logit")

    for rung in range(1, 7):
        for suffix in ("c", "w"):
            add_comparison(
                f"RREL_M{rung}{suffix}", f"RSET_M{rung}{suffix}",
                kind="relation_package_minus_set_package",
            )
    for strategy in ("RSET", "RREL"):
        for rung in range(1, 7):
            add_comparison(
                f"{strategy}_M{rung}w", f"{strategy}_M{rung}c",
                kind="warm_minus_cold",
            )
    for control_id in controls:
        report = by_id[control_id]
        counterpart = str(report.get("control_counterpart", ""))
        if counterpart not in by_id:
            raise ValueError("control counterpart is absent")
        # The frozen estimand is the primary M5c result minus its registered
        # ablation/control.  Keeping the primary on the left makes the sign
        # consistent with every scientific claim in Section 24.3.
        add_comparison(counterpart, control_id, kind="registered_m5_control")
        comparisons[-1]["screening_seed_only"] = True
    for strategy in ("RSET", "RREL"):
        for suffix in ("c", "w"):
            for rung in range(2, 7):
                add_comparison(
                    f"{strategy}_M{rung}{suffix}",
                    f"{strategy}_M{rung - 1}{suffix}",
                    kind="child_minus_same_branch_predecessor",
                )
            add_comparison(
                f"{strategy}_M6{suffix}", f"{strategy}_M1{suffix}",
                kind="terminal_minus_first_rung",
            )
            add_comparison(
                f"{strategy}_M6{suffix}", "M0",
                kind="terminal_minus_hlt_baseline",
            )

    required_parent_ids = {
        NODE_REGISTRY[node_id].parent_counterpart for node_id in primary
    } | {"M0", "D0c", "D0w", "D100", "TOFF"}
    missing_parent = required_parent_ids - set(parent_reports)
    if missing_parent:
        raise ValueError(
            f"representation screen lacks mandatory parent reports {sorted(missing_parent)}"
        )
    parent_rows = [
        {
            "node_id": node_id,
            "validation": dict(_metrics(parent_reports[node_id])),
            "report_sha256": require_sha256(
                parent_reports[node_id]["content_hash"],
                name=f"{node_id} parent report",
            ),
        }
        for node_id in sorted(required_parent_ids)
    ]
    rows = [
        {
            "node_id": node_id,
            "validation": dict(_metrics(by_id[node_id])),
            "report_sha256": require_sha256(by_id[node_id]["content_hash"], name=f"{node_id} report"),
        }
        for node_id in (*primary, *controls)
    ]
    gap_tables: list[dict[str, Any]] = []
    for node_id in primary:
        node = NODE_REGISTRY[node_id]
        student_metrics = _metrics(by_id[node_id])
        upper_id = "TOFF" if node.rung == 6 else "D100"
        lower_ids = (
            node.parent_counterpart,
            "M0",
            "D0c" if node.track == "c" else "D0w",
        )
        for lower_id in lower_ids:
            lower_metrics = _metrics(parent_reports[lower_id])
            upper_metrics = _metrics(parent_reports[upper_id])
            gap_tables.append({
                "node_id": node_id,
                "lower_anchor": lower_id,
                "upper_oracle": upper_id,
                "metrics": {
                    name: gap_recovery(
                        student=float(student_metrics[name]),
                        lower=float(lower_metrics[name]),
                        upper=float(upper_metrics[name]),
                        higher_is_better=higher,
                    )
                    for name, higher in GAP_METRIC_DIRECTIONS.items()
                },
            })
    return with_content_hash(
        {
            "contract": SCREEN_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign spec"),
            "graph_sha256": require_sha256(graph_sha256, name="graph"),
            "recipe_sha256": require_sha256(recipe_sha256, name="recipe"),
            "primary_rows": rows[: len(primary)],
            "control_rows": rows[len(primary) :],
            "parent_rows": parent_rows,
            "ordered_comparisons": comparisons,
            "gap_recovery_tables": gap_tables,
            "gap_recovery_metric_directions": dict(GAP_METRIC_DIRECTIONS),
            "all_registered_nodes_completed": True,
            "finite_poor_results_retained": True,
        }
    )


def build_confirmation_registry(
    *,
    screen_sha256: str,
    campaign_sha256: str,
    recipe_sha256: str,
    target_logical_bank_sha256: str,
    objectives: Sequence[str],
    seeds: Sequence[int] = CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    objective_ids = tuple(str(value) for value in objectives)
    seed_values = tuple(int(value) for value in seeds)
    exact_objectives = ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w")
    if objective_ids != exact_objectives:
        raise ValueError("confirmation requires the ordered four M6 objectives")
    if seed_values != CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed registry differs from the frozen recipe")
    rows = []
    for objective in objective_ids:
        node = NODE_REGISTRY[objective]
        for seed in seed_values:
            execution_id, identity_payload = derive_representation_execution_id(
                campaign_sha256=campaign_sha256,
                strategy=node.strategy,
                node_id=node.node_id,
                purpose="confirmation",
                seed=seed,
                initialization_parent=node.initialization_parent,
                teacher=node.representation_logit_teacher,
                logical_target_bank_sha256=target_logical_bank_sha256,
                target_purpose="confirmation",
                recipe_sha256=recipe_sha256,
            )
            rows.append(
                {
                    "execution_id": execution_id,
                    "execution_identity_payload": identity_payload,
                    "objective_id": objective,
                    "seed": seed,
                    "logical_bank_sha256": require_sha256(
                        target_logical_bank_sha256, name="TOFF logical bank"
                    ),
                    "physical_generation_sha256": None,
                }
            )
    return with_content_hash(
        {
            "contract": CONFIRMATION_REGISTRY_CONTRACT,
            "schema_version": 1,
            "screen_sha256": require_sha256(screen_sha256, name="screen"),
            "campaign_sha256": require_sha256(campaign_sha256, name="campaign"),
            "recipe_sha256": require_sha256(
                recipe_sha256, name="representation recipe",
            ),
            "logical_bank_sha256": require_sha256(
                target_logical_bank_sha256, name="TOFF logical bank"
            ),
            "seeds": list(seed_values),
            "rows": rows,
            "execution_count": 20,
        }
    )


def _five_seed_summary(values: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if len(rows) != 5 or not all(math.isfinite(value) for value in rows):
        raise ValueError("confirmation summary requires five finite raw values")
    mean = sum(rows) / 5.0
    variance = sum((value - mean) ** 2 for value in rows) / 4.0
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(5.0)
    half_width = T_975_DF4 * standard_error
    return {
        "raw": rows,
        "mean": mean,
        "standard_deviation_ddof1": standard_deviation,
        "standard_error": standard_error,
        "student_t_df": 4,
        "t_0.975": T_975_DF4,
        "lower_95": mean - half_width,
        "upper_95": mean + half_width,
    }


def build_confirmation_run(
    *, registry: Mapping[str, Any], training_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the immutable execution-scoped pointer consumed by aggregation."""

    from .hcwdl_representation_training import validate_representation_training_report

    if registry.get("contract") != CONFIRMATION_REGISTRY_CONTRACT:
        raise ValueError("confirmation registry contract differs")
    validate_representation_training_report(training_report)
    registered_execution_id = str(training_report.get("registered_execution_id", ""))
    rows = [
        row for row in registry.get("rows", ())
        if str(row.get("execution_id", "")) == registered_execution_id
    ]
    if len(rows) != 1:
        raise ValueError("confirmation execution is not uniquely registered")
    frozen = rows[0]
    if (
        training_report.get("node_id") != frozen.get("objective_id")
        or int(training_report.get("replicate_seed", -1)) != int(frozen.get("seed", -2))
        or training_report.get("mode") != "scientific"
        or training_report.get("scientific_complete") is not True
    ):
        raise ValueError("confirmation training report differs from its frozen row")
    return with_content_hash({
        "contract": CONFIRMATION_RUN_CONTRACT,
        "schema_version": 1,
        "registry_sha256": require_sha256(
            registry["content_hash"], name="confirmation registry",
        ),
        "execution_id": registered_execution_id,
        "execution_identity_payload": dict(frozen["execution_identity_payload"]),
        "objective_id": frozen["objective_id"],
        "seed": int(frozen["seed"]),
        "logical_bank_sha256": require_sha256(
            frozen["logical_bank_sha256"], name="confirmation logical bank",
        ),
        "training_report_sha256": require_sha256(
            training_report["content_hash"], name="confirmation training report",
        ),
        "validation": dict(_metrics(training_report)),
        "complete": True,
    })


def build_confirmation_aggregate(
    *,
    registry: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str] = CONFIRMATION_METRICS,
) -> dict[str, Any]:
    if registry.get("contract") != CONFIRMATION_REGISTRY_CONTRACT:
        raise ValueError("confirmation registry contract differs")
    expected = {str(row["execution_id"]): row for row in registry["rows"]}
    metric_names = tuple(str(value) for value in metric_names)
    if metric_names != CONFIRMATION_METRICS:
        raise ValueError("confirmation metric registry differs from the frozen full set")
    actual = {str(report.get("execution_id")): report for report in reports}
    if len(actual) != len(reports) or set(actual) != set(expected):
        raise ValueError("confirmation reports differ from the frozen registry")
    for execution_id, report in actual.items():
        frozen = expected[execution_id]
        if (
            report.get("objective_id") != frozen["objective_id"]
            or report.get("seed") != frozen["seed"]
            or report.get("execution_identity_payload")
            != frozen["execution_identity_payload"]
            or report.get("logical_bank_sha256")
            != frozen["logical_bank_sha256"]
        ):
            raise ValueError("confirmation report identity lineage differs")
    by_objective: dict[str, dict[str, Any]] = {}
    for objective in sorted({str(row["objective_id"]) for row in expected.values()}):
        objective_reports = [
            actual[str(row["execution_id"])]
            for row in registry["rows"]
            if row["objective_id"] == objective
        ]
        by_objective[objective] = {
            metric: _five_seed_summary([float(_metrics(report)[metric]) for report in objective_reports])
            for metric in metric_names
        }
    paired: dict[str, Any] = {}
    # Section 24.4 predeclares exactly the two same-track strategy contrasts
    # and two same-strategy warm-minus-cold contrasts.  Extra cross-package
    # comparisons are not silently promoted to registered estimands.
    contrasts = (
        ("RREL_M6c", "RSET_M6c"),
        ("RREL_M6w", "RSET_M6w"),
        ("RSET_M6w", "RSET_M6c"),
        ("RREL_M6w", "RREL_M6c"),
    )
    for left, right in contrasts:
        left_by_seed = {
            int(report["seed"]): report
            for report in reports if report["objective_id"] == left
        }
        right_by_seed = {
            int(report["seed"]): report
            for report in reports if report["objective_id"] == right
        }
        if set(left_by_seed) != set(right_by_seed) or set(left_by_seed) != set(
            CONFIRMATION_SEEDS
        ):
            raise ValueError("confirmation paired seed sets differ")
        paired[f"{left}-minus-{right}"] = {
            metric: _five_seed_summary(
                [
                    float(_metrics(left_by_seed[seed])[metric])
                    - float(_metrics(right_by_seed[seed])[metric])
                    for seed in CONFIRMATION_SEEDS
                ]
            )
            for metric in metric_names
        }
    return with_content_hash(
        {
            "contract": CONFIRMATION_AGGREGATE_CONTRACT,
            "schema_version": 1,
            "registry_sha256": require_sha256(registry["content_hash"], name="confirmation registry"),
            "objectives": by_objective,
            "paired_seed_contrasts": paired,
            "conditional_terminal_m6_only": True,
            "used_for_finalist_selection": False,
        }
    )


def build_validation_only_aggregate(
    *, screen_aggregate: Mapping[str, Any],
    confirmation_aggregate: Mapping[str, Any],
    campaign_spec_sha256: str,
    final_disposition_sha256: str,
) -> dict[str, Any]:
    """Close a consumed-parent-claim campaign without touching final data."""

    if screen_aggregate.get("contract") != SCREEN_CONTRACT:
        raise ValueError("validation-only screen aggregate contract differs")
    if confirmation_aggregate.get("contract") != CONFIRMATION_AGGREGATE_CONTRACT:
        raise ValueError("validation-only confirmation aggregate contract differs")
    for artifact, name in (
        (screen_aggregate, "screen aggregate"),
        (confirmation_aggregate, "confirmation aggregate"),
    ):
        require_sha256(artifact.get("content_hash"), name=name)
    return with_content_hash({
        "contract": VALIDATION_ONLY_AGGREGATE_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": require_sha256(
            campaign_spec_sha256, name="campaign spec",
        ),
        "final_disposition_sha256": require_sha256(
            final_disposition_sha256, name="final disposition",
        ),
        "screen_aggregate_sha256": screen_aggregate["content_hash"],
        "confirmation_aggregate_sha256": confirmation_aggregate["content_hash"],
        "disposition": "validation_only_parent_claim_consumed",
        "final_role_accessed": False,
        "prediction_artifacts_created": False,
        "confirmatory_claim_allowed": False,
        "all_registered_validation_rows_retained": True,
    })


__all__ = [
    "CONFIRMATION_AGGREGATE_CONTRACT",
    "CONFIRMATION_RUN_CONTRACT",
    "CONFIRMATION_REGISTRY_CONTRACT",
    "CONFIRMATION_SEEDS",
    "CONFIRMATION_METRICS",
    "FINAL_AGGREGATE_CONTRACT",
    "SCREEN_CONTRACT",
    "TRAINING_REPORT_CONTRACT",
    "VALIDATION_ONLY_AGGREGATE_CONTRACT",
    "build_confirmation_run",
    "build_confirmation_aggregate",
    "build_confirmation_registry",
    "build_screen_aggregate",
    "build_validation_only_aggregate",
    "checkpoint_key",
    "derive_representation_execution_id",
    "gap_recovery",
    "select_checkpoint",
]
