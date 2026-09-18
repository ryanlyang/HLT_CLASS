"""Closed scientific registry for the native-CMS learned handoff."""
from __future__ import annotations

import math
from fractions import Fraction

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
)
from hlt_classification.scouting.hcwdl_homotopy import HomotopyCoordinate
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import matcher_spec

FAMILY = "CMS_SALIENCE_LEARNED_DENSE"
AUTHORIZATION = "AUTHORIZE CMS SALIENCE LEARNED DENSE 500K EXACT SPEC"
RUNG_ORDER = ("U000", "U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000")
BUDGETS = {"train": 500_000, "validation": 250_000, "final_test": 250_000}
SITE = dict(account="reu-aisocial", partition="tier3", qos="qos_tier3",
            cluster="sporc", gres="gpu:a100:1", conda_env="atlas_kd_sporc")
TRAINING = dict(maximum_passes=100, minimum_passes=60, patience=15,
                patience_clock_start_pass=60, minimum_auc_delta=5e-5,
                batch_size=256, weight_decay=.01, betas=[.9, .999], eps=1e-8,
                peak_lr=3e-4, floor_lr=1.5e-5, warmup_passes=3,
                hold_through_pass=45, decay_through_pass=60,
                precision="bf16_forward_fp32_loss", restore_best=True)


def artifact(artifact_type: str, **fields):
    version = 2 if artifact_type == "CAMPAIGN_SPEC" else 1
    return with_content_hash(dict(fields, contract=f"{FAMILY}_{artifact_type}/v{version}", schema_version=version))


def validate(value, artifact_type):
    version = value.get("schema_version") if artifact_type == "CAMPAIGN_SPEC" else 1
    if type(version) is not int or version not in (1, 2) or (artifact_type != "CAMPAIGN_SPEC" and version != 1):
        raise ValueError("Unsupported CMS contract version")
    return validate_content_hash(value, expected_contract=f"{FAMILY}_{artifact_type}/v{version}",
                                 expected_schema_version=version)


def site_for_partition(partition="tier3"):
    if partition not in ("tier3", "debug"):
        raise ValueError("CMS partition must be tier3 or debug")
    return dict(SITE, partition=partition)


def allocation_site(spec):
    # Reuse scheduler/environment authentication only, not Delphes science or
    # its debug-to-tier3 profiling transfer. CMS science stays on its own site.
    from hlt_classification.jetclass2_delphes.execution import execution_site
    site = spec["site"]
    if site != site_for_partition(site["partition"]):
        raise ValueError("CMS execution site differs")
    if spec["schema_version"] == 1 and site != SITE:
        raise ValueError("Legacy CMS campaigns require tier3")
    return execution_site("sporc_a100_debug" if site["partition"] == "debug" else "sporc_a100")


def coordinate(name):
    if name not in (*RUNG_ORDER, "D100"):
        raise ValueError("Unregistered CMS coordinate")
    if name[0] == "U":
        u = {"U000": Fraction(0), "U033": Fraction(1, 3),
             "U066": Fraction(2, 3), "U100": Fraction(1)}[name]
        f = Fraction(0)
    else:
        u, f = Fraction(1), 1 - Fraction(int(name[1:]), 100)
    return HomotopyCoordinate(u.numerator, u.denominator, f.numerator, f.denominator)


def learning_rate(position):
    if not math.isfinite(position) or not 0 < position <= 100:
        raise ValueError("Learning-rate position differs")
    if position <= 3:
        return 3e-4 * position / 3
    if position <= 45:
        return 3e-4
    if position <= 60:
        return 1.5e-5 + (3e-4 - 1.5e-5) * .5 * (1 + math.cos(math.pi * (position - 45) / 15))
    return 1.5e-5


def alpha_for_pass(position):
    if not math.isfinite(position) or position <= 0:
        raise ValueError("Withdrawal position differs")
    if position <= 10:
        return 1.
    if position >= 60:
        return 0.
    return .5 * (1 + math.cos(math.pi * (position - 10) / 50))


def node(name, role, primary, context=None, teacher=None, parent=None, alias=None):
    alias = alias or name
    seed = lambda domain: int(canonical_sha256([FAMILY, alias, domain])[:8], 16)
    return dict(node_id=name, role=role, primary_coordinate=primary,
                context_coordinate=context, teacher_distribution=teacher,
                initialization_parent=parent,
                selection_route="alpha_zero" if role == "fusion_withdrawal" else "ordinary",
                initialization_seed=seed("initialization"), sampler_seed=seed("sampler"),
                context_architecture_seed=seed("context"), seed_alias=alias)


def graph():
    nodes = [node("M0HLT", "reference_ce", "D000", alias="D000"), node("OFFLINE", "reference_ce", "OFFLINE"),
             node("U000", "reference_ce", "U000"),
             node("DIRECT_D000", "direct_kd", "D000", teacher="U000", alias="D000")]
    tasks = []
    def task(task_id, kind, dependencies, model=None):
        tasks.append(dict(task_id=task_id, kind=kind, dependencies=list(dependencies), model=model))
    for n in nodes[:3]:
        task("train_" + n["node_id"], "train", [], n["node_id"])
    task("reduce_U000", "reduce", ["train_U000"], "U000")
    task("train_DIRECT_D000", "train", ["reduce_U000"], "DIRECT_D000")
    carrier = "U000"
    for higher, lower in zip(RUNG_ORDER, RUNG_ORDER[1:]):
        acq, withdrawal, extracted = f"ACQUIRE_{lower}", f"WITHDRAW_{lower}", f"CARRIER_{lower}"
        nodes += [node(acq, "fusion_acquisition", lower, higher, carrier, alias=lower),
                  node(withdrawal, "fusion_withdrawal", lower, higher, acq, acq, alias=lower)]
        task("train_" + acq, "train", ["reduce_" + carrier], acq)
        task("reduce_" + acq, "reduce", ["train_" + acq], acq)
        task("train_" + withdrawal, "train", ["reduce_" + acq], withdrawal)
        task("extract_" + extracted, "extract", ["train_" + withdrawal], withdrawal)
        if lower != "D000":
            task("reduce_" + extracted, "reduce", ["extract_" + extracted], extracted)
        carrier = extracted
    task("aggregate", "aggregate", [t["task_id"] for t in tasks])
    task("complete", "complete", ["aggregate"])
    assert len(nodes) == 20 and len(tasks) == 46
    return artifact("GRAPH", nodes=nodes, tasks=tasks, rung_order=list(RUNG_ORDER),
                    training=TRAINING, budgets=BUDGETS, matcher=matcher_spec("SALIENCE_PT_LINEAR"),
                    fit_count=20, extraction_count=8, reducer_count=16, final_test_accessed=False)
