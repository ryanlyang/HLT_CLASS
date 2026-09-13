"""Four explicitly authorized stages; no scientific jobs are auto-launched."""
from __future__ import annotations

from pathlib import Path
import re

from ..execution import execution_site
from ..inputs import input_contract
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout
from .contracts import (artifact, validate, ARMS, LAMBDAS, recipe, seed_register, load_json,
                        write_immutable_json, checked_payload)
from .targets import definition
from .model import model_contract
from .roles import authenticate_profile
from .reporting import metrics_contract

STAGES = ("GATE", "PREPARE", "DISCOVERY", "CONFIRMATION")


def _source(project, commit):
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact source commit required")
    validate_source_checkout(project, expected_commit=commit)


def discovery_nodes():
    nodes = [dict(node_id="DISC_CE", arm="CE", replicate="DISCOVERY", **{"lambda": [0, 1]})]
    nodes += [dict(node_id=f"DISC_{arm}_{name}", arm=arm, replicate="DISCOVERY", **{"lambda": value})
              for arm in ARMS[1:] for name, value in LAMBDAS.items()]
    return nodes


def confirmation_nodes(configuration):
    validate(configuration, "CONFIGURATION_LOCK")
    return [dict(node_id=f"CONFIRM_{i:02d}_{arm}", arm=arm, replicate=f"CONFIRM_{i:02d}",
                 **{"lambda": [0, 1] if arm == "CE" else configuration["selected_lambda"][arm]})
            for i in range(1, 4) for arm in ARMS]


def create_study(*, root, data_root, inventory, profile, project_dir, source_commit):
    root, data_root, project = Path(root).resolve(), Path(data_root).resolve(), Path(project_dir).resolve()
    _source(project, source_commit)
    authenticate_profile(inventory, profile)
    if not data_root.is_dir():
        raise FileNotFoundError("Verified raw-data directory is not accessible")
    if root.exists() or root.is_relative_to(data_root) or data_root.is_relative_to(root):
        raise ValueError("Study needs a fresh output root separate from the raw snapshot")
    spec = artifact("STUDY_SPEC", study="JC2_OFFLINE_AUX_500K", root=str(root), data_root=str(data_root),
                    project_dir=str(project), source_commit=source_commit, inventory=inventory,
                    profile=profile, inputs=input_contract(capacity=240), recipe=recipe(),
                    targets=definition(), model=model_contract(), site=execution_site("sporc_a100"),
                    seed_register=seed_register(),
                    metrics=metrics_contract(),
                    fit_count=22, discovery=discovery_nodes(), stage_order=list(STAGES),
                    initial_resources=dict(cpus=8, workers=8, memory_mb=72*1024, minutes=240),
                    no_matching_foundation=True, deployment="HLT_only", rolling_resume=False)
    write_immutable_json(root / "study_spec.json", spec)
    return spec


def validate_study(spec, *, source=False):
    digest = validate(spec, "STUDY_SPEC")
    authenticate_profile(spec["inventory"], spec["profile"])
    expected = dict(study="JC2_OFFLINE_AUX_500K", fit_count=22, discovery=discovery_nodes(),
                    inputs=input_contract(capacity=240), recipe=recipe(), targets=definition(),
                    model=model_contract(), site=execution_site("sporc_a100"), stage_order=list(STAGES),
                    seed_register=seed_register(),
                    metrics=metrics_contract(),
                    initial_resources=dict(cpus=8, workers=8, memory_mb=72*1024, minutes=240),
                    no_matching_foundation=True, deployment="HLT_only", rolling_resume=False)
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("Study registered semantics differ")
    if "preparation_import" in spec:
        from .preparation_import import validate_import
        validate_import(spec)
    elif "profile_measurement_site" in spec:
        raise ValueError("Debug measurement requires an explicit preparation-import continuation")
    if source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def stage_path(study, name):
    if name not in STAGES:
        raise ValueError("Unknown stage")
    return Path(study["root"]) / "stages" / name


def task_result(stage, task):
    path = Path(stage["root"]) / "receipts" / (task + ".json")
    receipt = load_json(path)
    validate(receipt, "TASK_RECEIPT")
    if receipt["stage_sha256"] != stage["content_hash"] or receipt["task_id"] != task:
        raise ValueError("Task receipt identity differs")
    for output in receipt["outputs"]:
        checked_payload(stage["root"], output)
    result_root = Path(stage["root"]) / receipt["output_directory"]
    if not result_root.resolve().is_relative_to(Path(stage["root"]).resolve()):
        raise ValueError("Task output escaped stage")
    if load_json(result_root / "result.json") != receipt["result"]:
        raise ValueError("Task result differs from sealed receipt")
    return receipt["result"], result_root


def completed(stage, task):
    path = Path(stage["root"]) / "receipts" / (task + ".json")
    return task_result(stage, task) if path.is_file() else None


def prior(study, stage_name, task):
    if "preparation_import" in study:
        from .preparation_import import IMPORTED, imported_stage
        if task in IMPORTED.get(stage_name, ()):
            return task_result(imported_stage(study, stage_name), task)
    stage = load_json(stage_path(study, stage_name) / "stage_spec.json")
    validate(stage, "STAGE_SPEC")
    if stage["study_sha256"] != study["content_hash"]:
        raise ValueError("Prior stage belongs to another study")
    return task_result(stage, task)


def task_graph(name, *, configuration=None):
    def task(task_id, kind, dependencies=(), **fields):
        return dict(task_id=task_id, kind=kind, dependencies=list(dependencies), **fields)
    if name == "GATE":
        return [task("sample", "sample")]
    if name == "PREPARE":
        banks = [task(f"targets_{role}_{i:02d}", "target", role=role, shard=i)
                 for role, count in (("TRAIN", 5), ("VAL_SELECT", 2)) for i in range(count)]
        return banks + [task("normalize", "normalize", [t["task_id"] for t in banks]),
                        task("profile", "profile", ["normalize"])]
    if name == "DISCOVERY":
        fits = [task(n["node_id"], "train", node=n) for n in discovery_nodes()]
        return fits + [task("configuration_lock", "configuration", [t["task_id"] for t in fits])]
    if name == "CONFIRMATION":
        fits = [task(n["node_id"], "train", node=n) for n in confirmation_nodes(configuration)]
        lock = task("reporting_lock", "report_lock", [t["task_id"] for t in fits])
        banks = [task(f"targets_VAL_REPORT_{i:02d}", "target", ["reporting_lock"], role="VAL_REPORT", shard=i)
                 for i in range(8)]
        evaluations = [task("eval_"+t["task_id"], "evaluate", ["reporting_lock"]+[b["task_id"] for b in banks], node=t["node"])
                       for t in fits]
        draws = [task(f"bootstrap_{i:02d}", "bootstrap", [t["task_id"] for t in evaluations], start=i*50, end=(i+1)*50)
                 for i in range(20)]
        return fits + [lock] + banks + evaluations + draws + [
            task("aggregate", "aggregate", [t["task_id"] for t in draws]), task("complete", "complete", ["aggregate"])]
    raise ValueError("Unknown stage")


def study_task_graph(study, name, *, configuration=None):
    if "preparation_import" in study:
        if name == "GATE":
            raise ValueError("GATE is already imported; create PREPARE for the profile only")
        if name == "PREPARE":
            return [dict(task_id="profile", kind="profile", dependencies=[])]
    return task_graph(name, configuration=configuration)


def task_site(study, task):
    if task["kind"] == "profile":
        return study.get("profile_measurement_site", study["site"])
    return study["site"]


def create_stage(study, name):
    validate_study(study, source=True)
    root = stage_path(study, name)
    if root.exists():
        raise FileExistsError("Stage already exists; inspect or use exact restart-zero recovery")
    configuration, profile, parents = None, None, {}
    resources = study["initial_resources"]
    if name != "GATE":
        sample, _ = prior(study, "GATE", "sample")
        validate(sample, "SAMPLE_PROFILE")
        parents["sample"] = sample["content_hash"]
        resources = dict(resources, target_minutes=sample["target_minutes"])
    if name in {"DISCOVERY", "CONFIRMATION"}:
        profile, _ = prior(study, "PREPARE", "profile")
        from .execution import validate_acceptance
        validate_acceptance(profile, study)
        parents["acceptance"] = profile["content_hash"]
        resources = profile["resources"]
    if name == "CONFIRMATION":
        configuration, _ = prior(study, "DISCOVERY", "configuration_lock")
        validate(configuration, "CONFIGURATION_LOCK")
        parents["configuration"] = configuration["content_hash"]
    stage = artifact("STAGE_SPEC", name=name, root=str(root), study_sha256=study["content_hash"],
                     study_spec_path=str(Path(study["root"]) / "study_spec.json"), parents=parents,
                     source_commit=study["source_commit"], resources=resources,
                     tasks=study_task_graph(study, name, configuration=configuration))
    write_immutable_json(root / "stage_spec.json", stage)
    return stage


def validate_stage(stage, study):
    validate(stage, "STAGE_SPEC")
    validate_study(study)
    if (stage["study_sha256"] != study["content_hash"] or stage["source_commit"] != study["source_commit"]
            or Path(stage["root"]).resolve() != stage_path(study, stage["name"]).resolve()):
        raise ValueError("Stage/source/root identity differs")
    configuration = None
    if stage["name"] == "CONFIRMATION":
        configuration, _ = prior(study, "DISCOVERY", "configuration_lock")
        if stage["parents"]["configuration"] != configuration["content_hash"]:
            raise ValueError("Configuration lock changed")
    if stage["tasks"] != study_task_graph(study, stage["name"], configuration=configuration):
        raise ValueError("Stage task graph differs")
    if "preparation_import" in study and stage["name"] == "PREPARE":
        sample, _ = prior(study, "GATE", "sample")
        expected_resources = dict(study["initial_resources"], target_minutes=sample["target_minutes"])
        if (stage["resources"] != expected_resources
                or stage["parents"] != {"sample": sample["content_hash"]}):
            raise ValueError("Debug profile preparation/resources differ from the imported gate")
    if stage["name"] in {"DISCOVERY", "CONFIRMATION"}:
        from .execution import validate_acceptance
        profile, _ = prior(study, "PREPARE", "profile")
        validate_acceptance(profile, study)
        if stage["parents"]["acceptance"] != profile["content_hash"] or stage["resources"] != profile["resources"]:
            raise ValueError("Measured resources differ")


def configuration_lock(study, split, models, *, normalizer, preparation_lock_sha256):
    from .contracts import selection_key
    expected = {n["node_id"]: n for n in discovery_nodes()}
    validate(normalizer, "NORMALIZATION")
    if set(models) != set(expected):
        raise ValueError("All ten discovery reports, including losers, are required")
    for key, m in models.items():
        if m["node"] != expected[key]:
            raise ValueError("Discovery node changed")
    selected = {}
    for arm in ARMS[1:]:
        candidate = max((m for m in models.values() if m["node"]["arm"] == arm), key=lambda m:
            (*selection_key(m["validation"], m["selected_update"]), -m["node"]["lambda"][0]/m["node"]["lambda"][1]))
        selected[arm] = candidate["node"]["lambda"]
    return artifact("CONFIGURATION_LOCK", study_sha256=study["content_hash"], role_split_sha256=split["content_hash"],
                    models=models, selected_lambda=selected, reporting_accessed=False,
                    normalization_sha256=normalizer["content_hash"], hlt_pt_edges=normalizer["hlt_pt_edges"],
                    preparation_lock_sha256=preparation_lock_sha256, metrics=metrics_contract())


def reporting_lock(study, split, configuration, models):
    expected = {n["node_id"]: n for n in confirmation_nodes(configuration)}
    if set(models) != set(expected) or any(m["node"] != expected[k] for k, m in models.items()):
        raise ValueError("All twelve fresh confirmation fits must be frozen before reporting")
    return artifact("REPORTING_LOCK", study_sha256=study["content_hash"], role_split_sha256=split["content_hash"],
                    configuration_sha256=configuration["content_hash"], models=models,
                    normalization_sha256=configuration["normalization_sha256"],
                    hlt_pt_edges=configuration["hlt_pt_edges"], metrics=metrics_contract())
