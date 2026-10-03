"""Immutable campaign, source, DAG and result receipt authentication."""
from pathlib import Path
import re

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout
from hlt_classification.scouting.splits import validate_split_manifest
from hlt_classification.cms_salience_learned.storage import fingerprint, checked_file
from hlt_classification.jetclass2_delphes.execution import execution_site
from .contracts import artifact, validate, registry, science_tasks


def sources(split):
    return [dict(row, role=role) for role in ("train", "validation") for row in split["roles"][role]["files"]]


def gate_tasks(split, arms):
    tasks = [dict(task_id="select", kind="select", dependencies=[])]
    for i in range(len(sources(split))):
        tasks.append(dict(task_id=f"match_{i:04d}", kind="match", index=i, dependencies=["select"]))
    tasks.append(dict(task_id="foundation", kind="foundation", dependencies=[t["task_id"] for t in tasks[1:]]))
    for arm in arms:
        tasks.append(dict(task_id="preflight_" + arm, kind="preflight", arm=arm, dependencies=["foundation"]))
    return tasks


def create(*, project, commit, data_root, split_path, root, arms=("SHARED17",)):
    project, root, data_root = Path(project).resolve(), Path(root).resolve(), Path(data_root).resolve(strict=True)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact pushed 40-character commit required")
    validate_source_checkout(project, expected_commit=commit)
    split_path = Path(split_path).resolve(strict=True)
    split = load_json(split_path)
    validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    config = registry(arms)
    if root.exists() or root.is_relative_to(data_root) or data_root.is_relative_to(root) or split_path.is_relative_to(root):
        raise ValueError("New output root must be disjoint from input data")
    spec = artifact("SPEC", parents={"split": split["content_hash"]},
        project_dir=str(project), source_commit=commit, campaign_root=str(root), data_root=str(data_root),
        split=fingerprint(split_path), scientific=config, sources=sources(split),
        gate_tasks=gate_tasks(split, arms), science_tasks=science_tasks(arms),
        site=execution_site("sporc_a100_debug"), resources=dict(cpus=16, workers=16, memory_mb=160000))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "campaign_spec.json", spec)
    return spec


def validate_spec(spec, *, source=False):
    validate(spec, "SPEC")
    split = load_json(checked_file(spec["split"]))
    validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    if (spec["parents"] != {"split": split["content_hash"]}
        or spec["scientific"] != registry(spec["scientific"]["arms"])
        or spec["sources"] != sources(split)
        or spec["gate_tasks"] != gate_tasks(split, spec["scientific"]["arms"])
        or spec["science_tasks"] != science_tasks(spec["scientific"]["arms"])
        or spec["site"] != execution_site("sporc_a100_debug")
        or spec["resources"] != dict(cpus=16, workers=16, memory_mb=160000)
        or re.fullmatch(r"[0-9a-f]{40}", spec["source_commit"]) is None):
        raise ValueError("CMS campaign registry differs")
    if source:
        validate_source_checkout(spec["project_dir"], expected_commit=spec["source_commit"])
    return split


def task(spec, name):
    return next(t for t in spec["gate_tasks"] + spec["science_tasks"] if t["task_id"] == name)


def receipt(spec, name):
    task(spec, name)
    root = Path(spec["campaign_root"])
    path = root / "receipts" / f"{name}.json"
    if not path.exists():
        return None
    value = load_json(path)
    validate(value, "TASK", parents={"spec": spec["content_hash"]})
    if value["task"] != name or value["source_commit"] != spec["source_commit"] or not value["outputs"]:
        raise ValueError("Task identity/outputs differ")
    for row in value["outputs"]:
        if not Path(row["path"]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Task output escapes campaign root")
        checked_file(row)
    return value


def publish(spec, name, paths):
    value = artifact("TASK", parents={"spec": spec["content_hash"]}, task=name,
        source_commit=spec["source_commit"], outputs=[fingerprint(p) for p in paths])
    write_immutable_json(Path(spec["campaign_root"]) / "receipts" / f"{name}.json", value)
    return value


def require(spec, name):
    row = receipt(spec, name)
    if row is None:
        raise PermissionError(f"Required authenticated task is incomplete: {name}")
    return row
