"""Closed scientific registry for the native-CMS learned handoff."""
from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
)
from hlt_classification.scouting.hcwdl_homotopy import HomotopyCoordinate
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import matcher_spec

FAMILY = "CMS_SALIENCE_LEARNED_DENSE"
AUTHORIZATION = "AUTHORIZE CMS SALIENCE LEARNED DENSE 500K EXACT SPEC"
COARSE_AUTHORIZATION = "AUTHORIZE CMS SALIENCE LEARNED COARSE 500K EXACT SPEC"
DIRECT_FUSION_AUTHORIZATION = "AUTHORIZE CMS DIRECT FUSION 500K EXACT SPEC"
RUNG_ORDER = ("U000", "U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000")
COARSE_RUNG_ORDER = ("U000", "U050", "U100", "D066", "D033", "D000")
SHARED_TASKS = ("train_M0HLT", "train_OFFLINE", "train_U000", "reduce_U000", "train_DIRECT_D000")
BUDGETS = {"train": 500_000, "validation": 250_000, "final_test": 250_000}
SITE = dict(account="reu-aisocial", partition="tier3", qos="qos_tier3",
            cluster="sporc", gres="gpu:a100:1", conda_env="atlas_kd_sporc")
TRAINING = dict(maximum_passes=100, minimum_passes=60, patience=15,
                patience_clock_start_pass=60, minimum_auc_delta=5e-5,
                batch_size=256, weight_decay=.01, betas=[.9, .999], eps=1e-8,
                peak_lr=3e-4, floor_lr=1.5e-5, warmup_passes=3,
                hold_through_pass=45, decay_through_pass=60,
                precision="bf16_forward_fp32_loss", restore_best=True)
ACCEPTANCE_POLICY = dict(cpu_peak_fraction_limit=.85, cuda_peak_fraction_limit=.90,
    withdrawal_probe_steps_per_alpha=5, withdrawal_probe_alphas=[1., .5, 0.],
    withdrawal_probe_batch_size=256, withdrawal_probe_batch_selection="longest_u000")
CONTRACT_VERSIONS = {"CAMPAIGN_SPEC": (1, 2, 3, 4, 5, 6), "EXECUTION_ACCEPTANCE": (1, 2),
                     "GRAPH": (1, 2, 3), "PREPARATION_IMPORT": (1, 2, 3),
                     "SHARED_SOURCE": (1, 2), "ACCEPTANCE_IMPORT": (1, 2),
                     "ACCEPTANCE_REUSE": (1, 2)}


def artifact(artifact_type: str, *, contract_version=None, **fields):
    supported = CONTRACT_VERSIONS.get(artifact_type, (1,))
    version = supported[-1] if contract_version is None else contract_version
    if type(version) is not int or version not in supported:
        raise ValueError("Unsupported CMS contract version")
    return with_content_hash(dict(fields, contract=f"{FAMILY}_{artifact_type}/v{version}", schema_version=version))


def validate(value, artifact_type):
    version = value.get("schema_version")
    if type(version) is not int or version not in CONTRACT_VERSIONS.get(artifact_type, (1,)):
        raise ValueError("Unsupported CMS contract version")
    return validate_content_hash(value, expected_contract=f"{FAMILY}_{artifact_type}/v{version}",
                                 expected_schema_version=version)


def acceptance_policy(spec):
    """Old specs retain their 85% gate; v3-v6 specs explicitly opt into 90%."""
    version = spec["schema_version"]
    if version in (3, 4, 5, 6):
        if spec.get("acceptance_policy") != ACCEPTANCE_POLICY:
            raise ValueError("CMS v3 acceptance policy differs")
        return deepcopy(ACCEPTANCE_POLICY)
    if version not in (1, 2) or "acceptance_policy" in spec:
        raise ValueError("Legacy CMS acceptance policy cannot be overridden")
    return dict(deepcopy(ACCEPTANCE_POLICY), cuda_peak_fraction_limit=.85,
                withdrawal_probe_steps_per_alpha=1,
                withdrawal_probe_batch_selection="legacy_first")


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
    if name not in (*RUNG_ORDER, *COARSE_RUNG_ORDER, "D100"):
        raise ValueError("Unregistered CMS coordinate")
    if name[0] == "U":
        u = {"U000": Fraction(0), "U033": Fraction(1, 3),
             "U050": Fraction(1, 2), "U066": Fraction(2, 3), "U100": Fraction(1)}[name]
        f = Fraction(0)
    else:
        retained = {"D066": Fraction(2, 3), "D033": Fraction(1, 3)}.get(name, Fraction(int(name[1:]), 100))
        u, f = Fraction(1), 1 - retained
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


def graph(ladder="dense"):
    if ladder not in {"dense", "coarse", "direct_fusion"}:
        raise ValueError("Unregistered CMS ladder")
    rungs = {"dense": RUNG_ORDER, "coarse": COARSE_RUNG_ORDER,
             "direct_fusion": ("U000", "D000")}[ladder]
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
    for higher, lower in zip(rungs, rungs[1:]):
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
    steps = len(rungs) - 1
    assert len(nodes) == 4 + 2 * steps and len(tasks) == 6 + 5 * steps
    return artifact("GRAPH", contract_version={"dense": 1, "coarse": 2, "direct_fusion": 3}[ladder],
                    nodes=nodes, tasks=tasks, rung_order=list(rungs),
                    training=TRAINING, budgets=BUDGETS, matcher=matcher_spec("SALIENCE_PT_LINEAR"),
                    fit_count=len(nodes), extraction_count=steps, reducer_count=2 * steps, final_test_accessed=False)
