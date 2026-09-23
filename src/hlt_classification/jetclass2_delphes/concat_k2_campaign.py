"""Isolated K=2 registration with explicit SPORC tier3/debug portability."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import artifact as base_artifact, validate as base_validate
from .concat_k2_execution import execution_policy, site_for_partition
from .concat_k2_model import PAIR_STORAGE, BATCH_PROBE_POLICY, PARITY_BACKEND, PARITY_TOLERANCES
from .inputs import input_contract
from .model import model_contract
from .production import _source
from .salience_learned_graph import TRAINING as SHARED_TRAINING
from .concat_k2_views import view_contract

AUTHORIZE = "AUTHORIZE JETCLASS2 DZFIX CONCAT K2 PORTABLE 500K EXACT SPEC"
PREFIX = "jc2k2"
COUNTS = dict(train=500_000, validation=1_000_000, final_test=1_000_000)
TRAINING = {**deepcopy(SHARED_TRAINING), "batch_size": 128}
INFERENCE_BATCH_SIZE = 128
RESOURCES = {
    "metadata": dict(cpus=1, memory_mb=8192, minutes=240, gpu=False),
    "assignment": dict(cpus=1, memory_mb=8192, minutes=720, gpu=False),
    "partition": dict(cpus=4, memory_mb=320000, minutes=360, gpu=False),
    "preflight": dict(cpus=4, memory_mb=320000, minutes=1440, gpu=True),
    "train": dict(cpus=4, memory_mb=320000, minutes=1440, gpu=True),
    "reduce": dict(cpus=4, memory_mb=320000, minutes=360, gpu=True),
}


VERSIONS = {"LAUNCH_SPEC": 6, "CAMPAIGN_SPEC": 6, "ACCEPTANCE": 5,
            "TRAINING_REPORT": 2, "BATCH_PROBE": 2}


def artifact(kind, **fields):
    return base_artifact("CONCAT_K2_" + kind, version=VERSIONS.get(kind, 1), **fields)


def validate(value, kind):
    return base_validate(value, "CONCAT_K2_" + kind, version=VERSIONS.get(kind, 1))


def seed(domain):
    return int.from_bytes(hashlib.sha256(("JC2/CONCAT_K2/v1/" + domain).encode()).digest()[:4], "big")


def nodes():
    rows = [("HLT_X1_CE", "HLT_X1", None), ("HLT_X3_CE", "HLT_X3", None),
            ("OFFLINE_CE", "OFFLINE", None), ("CONCAT_K2_D100", "D100", None),
            ("DIRECT_HLT_X3_KD", "HLT_X3", "CONCAT_K2_D100")]
    previous = "CONCAT_K2_D100"
    for view in ("D075", "D050", "D025", "D000"):
        name = "CONCAT_K2_" + view
        rows.append((name, view, previous))
        previous = name
    rows.append(("HLT_X1_COMPRESSED", "HLT_X1", previous))
    return [dict(node_id=name, primary_coordinate=view, context_coordinate=None,
                 teacher_distribution=teacher, role="reference_ce" if teacher is None else "direct_kd",
                 input_protocol="single_view", selection_route="ordinary",
                 initialization="fresh", initialization_parent=None,
                 initialization_seed=seed("initialization"), sampler_seed=seed("sampler"),
                 deployable=view in {"HLT_X1", "HLT_X3", "D000"})
            for name, view, teacher in rows]


def task_graph(foundation, *, reuse=False):
    rows = [dict(task_id="authenticate", kind="authenticate", dependencies=[], resource="metadata"),
            dict(task_id="matcher_acceptance", kind="matcher_acceptance", dependencies=["authenticate"], resource="assignment")]
    for file in foundation["assignment_tasks"]:
        rows.append(dict(task_id=f"assign_{file['file_index']:04d}", kind="assign",
                         file_index=file["file_index"], dependencies=["matcher_acceptance"], resource="assignment"))
    if reuse:
        rows = [rows[0], dict(task_id="import_preparation", kind="import_preparation",
                             dependencies=["authenticate"], resource="metadata")]
    rows.extend([
        dict(task_id="foundation_lock", kind="foundation", dependencies=(["import_preparation"] if reuse else
             [r["task_id"] for r in rows if r["kind"] == "assign"]), resource="metadata"),
        dict(task_id="partition_validation", kind="partition", dependencies=["foundation_lock"], resource="partition"),
        dict(task_id="audit_storage", kind="storage", dependencies=["foundation_lock"], resource="metadata"),
        dict(task_id="preflight", kind="preflight", dependencies=["partition_validation", "audit_storage"], resource="preflight"),
    ])
    teachers = {n["teacher_distribution"] for n in nodes()} - {None}
    for node in nodes():
        name, teacher = node["node_id"], node["teacher_distribution"]
        rows.append(dict(task_id="train_"+name, node_id=name, kind="train", resource="train",
                         dependencies=["reduce_"+teacher] if teacher else ["preflight"]))
        if name in teachers:
            rows.append(dict(task_id="reduce_"+name, node_id=name, kind="reduce", resource="reduce", dependencies=["train_"+name]))
    rows.append(dict(task_id="aggregate", kind="aggregate", resource="metadata",
                     dependencies=[r["task_id"] for r in rows if r["kind"] in {"train", "reduce"}]))
    rows.append(dict(task_id="complete", kind="complete", resource="metadata", dependencies=["aggregate"]))
    return rows


def gates(spec):
    return tuple(r["task_id"] for r in spec["tasks"] if r["kind"] not in {"train", "reduce", "aggregate", "complete"})


def registration(partition="tier3"):
    return dict(nodes=nodes(), training=deepcopy(TRAINING), execution_site=site_for_partition(partition),
        inference_batch_size=INFERENCE_BATCH_SIZE,
        execution_policy=execution_policy(),
        pair_storage=deepcopy(PAIR_STORAGE), batch_probe_policy=deepcopy(BATCH_PROBE_POLICY),
        parity_backend=deepcopy(PARITY_BACKEND), parity_tolerances=deepcopy(PARITY_TOLERANCES),
        resources=deepcopy(RESOURCES), model=model_contract(), role_counts=dict(COUNTS), split_profile="TRAIN_500K",
        k=2, copies=3, loss=dict(ce=.25, kd=.75, temperature=2.),
        salience_source="authenticated_existing_dzfix_screen_winner_formula_only",
        validation_partition=dict(fractions=[2,1,1], names=["checkpoint","diagnostic","report"],
                                  domain="JC2/CONCAT_K2/v1/validation"),
        gpu_peak_fraction_limit=.90, cpu_peak_fraction_limit=.80, cache_fraction_limit=.65,
        runtime_projection_margin=1.30, minimum_free_disk_bytes=32*1024**3,
        fresh_fit_count=10, reducer_count=5, science_task_count=17,
        matching_selection_used_validation=True, ram_only_particle_views=True, rolling_resume=False,
        ordinary_roles=["train","validation"], ordinary_final_test_capability=False,
        final_test_accessed=False, existing_campaign_mutations=False)


def foundation_spec(parent, source_hash, *, reuse=False):
    files = parent["inventory"]["files"]
    native = max(max(r["max_selected_particles"].values()) for r in files)
    expanded = 3*max(r["max_selected_particles"]["hlt"] for r in files)
    return artifact("FOUNDATION_SPEC", inventory=parent["inventory"], splits=parent["splits"],
        assignment_tasks=parent["assignment_tasks"], candidate=parent["candidate"],
        source_import_sha256=source_hash, views=view_contract(parent["candidate"]),
        native_capacity=max(16, ((native+15)//16)*16),
        inputs=input_contract(capacity=max(16, ((max(native, expanded)+15)//16)*16)),
        capacity_policy="max_native_or_three_hlt_inventory_counts_round_up_16_no_truncation",
        assignment_orientation="per_native_hlt_two_native_offline_indices_or_minus_one",
        reused_assignments=reuse, final_test_accessed=False)


def validate_campaign(spec, *, check_source=True):
    digest = validate(spec, "CAMPAIGN_SPEC")
    if any(spec.get(k) != v for k,v in registration(spec["execution_site"]["partition"]).items()):
        raise ValueError("K2 scientific/resource registration differs")
    from .concat_k2_source import validate_import
    validate_import(spec["source_import"])
    from .concat_k2_preparation_import import validate_reuse
    reuse = spec.get("preparation_import")
    if reuse is not None:
        validate_reuse(reuse, source=spec["source_import"], destination=spec["campaign_root"])
    parent = load_json(spec["source_import"]["foundation_spec_path"])
    if (spec["foundation"] != foundation_spec(parent, spec["source_import"]["content_hash"], reuse=reuse is not None)
            or spec["tasks"] != task_graph(spec["foundation"], reuse=reuse is not None)
            or spec["data_root"] != spec["source_import"]["data_root"]
            or spec["source_commit"] != spec["source_import"]["consumer_commit"]):
        raise ValueError("K2 foundation/graph/source differs")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def create(*, launch):
    from .concat_k2_source import build_import, validate_launch
    validate_launch(launch)
    source, parent = build_import(launch)
    from .concat_k2_preparation_import import validate_reuse
    reuse = launch.get("preparation_import")
    if reuse is not None:
        validate_reuse(reuse, source=source, destination=launch["campaign_root"])
    foundation = foundation_spec(parent, source["content_hash"], reuse=reuse is not None)
    root = Path(launch["campaign_root"])
    spec = artifact("CAMPAIGN_SPEC", **launch["registration"], foundation=foundation,
        preparation_import=reuse, tasks=task_graph(foundation, reuse=reuse is not None),
        source_commit=launch["source_commit"], project_dir=launch["project_dir"], campaign_root=str(root),
        data_root=source["data_root"], launch_sha256=launch["content_hash"], source_import=source)
    if root.exists():
        if not (root/"campaign_spec.json").is_file() or load_json(root/"campaign_spec.json") != spec:
            raise FileExistsError("Campaign root contains a different or partial execution")
        return spec
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/"source_import.json", source)
    write_immutable_json(root/"foundation_spec.json", foundation)
    write_immutable_json(root/"campaign_spec.json", spec)
    return spec
