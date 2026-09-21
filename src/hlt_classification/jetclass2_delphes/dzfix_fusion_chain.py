"""Immutable, debug-only dz-fix fusion-to-fusion campaign registration."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact as _artifact, validate as _validate
from .execution import execution_site
from .model import model_contract
from .production import _source
from .salience_learned_graph import TRAINING

AUTHORIZE = "AUTHORIZE JETCLASS2 DZFIX FUSION CHAIN DEBUG 500K EXACT SPEC"
PREFIX = "jc2fc"
COUNTS = {"train": 500_000, "validation": 1_000_000, "final_test": 1_000_000}
GATES = ("authenticate", "partition_validation", "audit_storage", "preflight")
RESOURCES = {
    "metadata": {"cpus": 1, "memory_mb": 8192, "minutes": 240, "gpu": False},
    "partition": {"cpus": 8, "memory_mb": 320000, "minutes": 360, "gpu": False},
    "preflight": {"cpus": 8, "memory_mb": 320000, "minutes": 720, "gpu": True},
    "train": {"cpus": 8, "memory_mb": 320000, "minutes": 1440, "gpu": True},
    "reduce": {"cpus": 8, "memory_mb": 320000, "minutes": 360, "gpu": True},
}


def artifact(kind, **fields):
    version = 3 if kind in {"LAUNCH_SPEC", "SOURCE_IMPORT", "CAMPAIGN_SPEC"} else 1
    return _artifact("DZFIX_FUSION_CHAIN_" + kind, version=version, **fields)


def validate(value, kind):
    version = 3 if kind in {"LAUNCH_SPEC", "SOURCE_IMPORT", "CAMPAIGN_SPEC"} else 1
    return _validate(value, "DZFIX_FUSION_CHAIN_" + kind, version=version)


def seed(alias, domain):
    return int.from_bytes(hashlib.sha256(
        f"JC2/DZFIX/FUSION_CHAIN/v1/{alias}/{domain}".encode()).digest()[:4], "big")


def nodes():
    rows = [
        ("M0HLT", "D000", None, None),
        ("OFFLINE", "OFFLINE", None, None),
        ("U000", "U000", None, None),
        ("DIRECT_D000", "D000", None, "U000"),
        ("FUSION_U050", "U050", "U000", "U000"),
        ("FUSION_U100", "U100", "U050", "FUSION_U050"),
        ("FUSION_D066", "D066", "U100", "FUSION_U100"),
        ("FUSION_D033", "D033", "D066", "FUSION_D066"),
        ("FUSION_D000", "D000", "D033", "FUSION_D033"),
        ("FINAL_DIRECT_D000", "D000", None, "FUSION_D000"),
        ("FUSION_D000_D000", "D000", "D000", "FUSION_D000"),
        ("FINAL_BRIDGE_D000", "D000", None, "FUSION_D000_D000"),
    ]
    result = []
    for name, primary, context, teacher in rows:
        alias = "PAIRED_FINAL_D000" if name.startswith("FINAL_") else name
        result.append(dict(
            node_id=name, primary_coordinate=primary, context_coordinate=context,
            teacher_distribution=teacher, role="fusion_pair_kd" if context else (
                "reference_ce" if teacher is None else "direct_kd"),
            input_protocol="single_view" if context is None else "paired_view",
            selection_route="ordinary" if context is None else "alpha_one",
            initialization="fresh", initialization_parent=None,
            initialization_seed=seed(alias, "initialization"),
            sampler_seed=seed(alias, "sampler"),
            context_architecture_seed=None if context is None else seed(alias, "context"),
            deployable=primary == "D000" and context in {None, "D000"},
        ))
    return result


def task_graph():
    rows = [
        dict(task_id="authenticate", kind="authenticate", dependencies=[], resource="metadata"),
        dict(task_id="partition_validation", kind="partition", dependencies=["authenticate"], resource="partition"),
        dict(task_id="audit_storage", kind="storage", dependencies=["authenticate"], resource="metadata"),
        dict(task_id="preflight", kind="preflight", dependencies=["partition_validation", "audit_storage"], resource="preflight"),
    ]
    teachers = {n["teacher_distribution"] for n in nodes()} - {None}
    for node in nodes():
        name, teacher = node["node_id"], node["teacher_distribution"]
        rows.append(dict(task_id="train_" + name, node_id=name, kind="train",
                         dependencies=["reduce_" + teacher] if teacher else ["preflight"], resource="train"))
        if name in teachers:
            rows.append(dict(task_id="reduce_" + name, node_id=name, kind="reduce",
                             dependencies=["train_" + name], resource="reduce"))
    rows.append(dict(task_id="aggregate", kind="aggregate", resource="metadata",
                     dependencies=[r["task_id"] for r in rows if r["task_id"] not in GATES]))
    rows.append(dict(task_id="complete", kind="complete", resource="metadata", dependencies=["aggregate"]))
    return rows


def registration():
    return dict(
        nodes=nodes(), tasks=task_graph(), training=deepcopy(TRAINING),
        loss={"ce": .25, "kd": .75, "temperature": 2., "alpha": 1.},
        execution_site=execution_site("sporc_a100_debug"), resources=deepcopy(RESOURCES),
        model=model_contract(), role_counts=dict(COUNTS), split_profile="TRAIN_500K",
        input_capacity_policy="inventory_max_selected_round_up_16_no_truncation_v1",
        fusion={"injection_blocks": [2, 4, 6, 8], "direction": "context_to_primary",
                "pair_population": "full_combined_weaver", "mask_execution": "compact_rectangle_padding_once_v1"},
        validation_partition={"fractions": [2, 1, 1], "names": ["checkpoint", "diagnostic", "report"],
                              "domain": "JC2/DZFIX/FUSION_CHAIN/v1/validation"},
        gpu_peak_fraction_limit=.90, cpu_peak_fraction_limit=.80,
        cache_fraction_limit=.65, runtime_projection_margin=1.30,
        minimum_free_disk_bytes=32 * 1024**3,
        fresh_fit_count=12, reducer_count=7, science_task_count=21,
        all_scientific_models_fresh=True, matching_selection_used_validation=True,
        ordinary_roles=["train", "validation"], ordinary_final_test_capability=False,
        ram_only_particle_views=True, rolling_resume=False,
        final_test_accessed=False, existing_campaign_mutations=False,
    )


def validate_campaign(spec, *, check_source=True):
    digest = validate(spec, "CAMPAIGN_SPEC")
    if any(spec.get(k) != v for k, v in registration().items()):
        raise ValueError("Fusion-chain scientific/resource registration differs")
    from .dzfix_fusion_source import validate_import
    validate_import(spec["source_import"], deep=False)
    if spec["foundation"] != load_json(spec["source_import"]["foundation_spec_path"]):
        raise ValueError("Fusion-chain foundation differs from the imported winner")
    if spec["data_root"] != spec["source_import"]["data_root"]:
        raise ValueError("Fusion-chain data source differs")
    if spec["source_import"]["consumer_commit"] != spec["source_commit"]:
        raise ValueError("Fusion-chain import names a different consumer commit")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def create(*, launch):
    from .dzfix_fusion_source import build_import, validate_launch
    validate_launch(launch)
    source, foundation = build_import(launch)
    root = Path(launch["campaign_root"])
    spec = artifact("CAMPAIGN_SPEC", **registration(),
        source_commit=launch["source_commit"], project_dir=launch["project_dir"],
        campaign_root=str(root), data_root=source["data_root"],
        launch_sha256=launch["content_hash"], source_import=source, foundation=foundation)
    if root.exists():
        existing = root / "campaign_spec.json"
        if not existing.is_file() or load_json(existing) != spec:
            raise FileExistsError("Campaign root already exists with different or incomplete materialization")
        return spec
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "source_import.json", source)
    write_immutable_json(root / "campaign_spec.json", spec)
    return spec
