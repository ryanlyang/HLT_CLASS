"""Production task claims and immutable, authenticated output inventories."""
from __future__ import annotations

import os
from pathlib import Path

from .campaign import stage_dir, validate_spec
from .contracts import artifact, load_json, safe_relative, sha256_file, validate
from .measurement import allocation, Measurement
from .provenance import numerical_environment, validate_source
from .storage import GIB, TOTAL_CAP, publish_json, publish_bytes, usage


def receipt_path(spec, task_id):
    return stage_dir(spec)/"receipts"/(task_id+".json")


def verified_receipt(spec, task_id, *, checked=None):
    checked = {} if checked is None else checked
    key = (spec["content_hash"], task_id)
    if key in checked:
        return checked[key]
    tasks = {r["task_id"]: r for r in spec["tasks"]}
    if task_id not in tasks:
        raise ValueError("Unknown campaign task")
    reusable = spec["reusable_tasks"].get(task_id)
    if reusable:
        from .campaign import resolve_ref
        parent = resolve_ref(reusable["spec"])
        value = verified_receipt(parent, task_id, checked=checked)
        if value["content_hash"] != reusable["receipt_hash"]:
            raise ValueError("Reused task receipt changed")
        checked[key] = value
        return value
    value = load_json(receipt_path(spec, task_id))
    expected = {"spec": spec["content_hash"], "source": spec["source"]["content_hash"]}
    for dep in tasks[task_id]["depends_on"]:
        expected[dep] = verified_receipt(spec, dep, checked=checked)["content_hash"]
    validate(value, "TASK_OUTPUTS", parents=expected)
    if value["task_id"] != task_id or not value["outputs"]:
        raise ValueError("Task output inventory differs")
    root = Path(spec["campaign_root"])
    for row in value["outputs"].values():
        path = safe_relative(root, row["relative"])
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Completed task payload changed")
        if row.get("kind"):
            if validate(load_json(path), row["kind"]) != row["content_hash"]:
                raise ValueError("Completed task artifact identity differs")
    checked[key] = value
    return value


def verified_product(spec, task_id, name):
    receipt = verified_receipt(spec, task_id)
    if name not in receipt["outputs"]:
        raise ValueError("Required completed task product is absent")
    return safe_relative(Path(spec["campaign_root"]), receipt["outputs"][name]["relative"])


def product(spec, task, name="result"):
    return load_json(verified_product(spec, task, name))


def run(spec, task_id):
    context = validate_spec(spec)
    tasks = {r["task_id"]: r for r in spec["tasks"]}
    if task_id not in tasks or task_id in spec["reusable_tasks"]:
        raise PermissionError("Worker is not a fresh task in this execution")
    task = tasks[task_id]
    allocation_record = allocation(cpus=task["resources"]["cpus"])
    directory = stage_dir(spec)
    from .submission import submitted_jobs
    import time
    for retry in range(60):
        jobs = submitted_jobs(spec)
        if jobs.get(task_id) is not None:
            break
        # A fast allocation may start before the submitter persists sbatch's
        # acknowledgement. Never infer its ID from a job name or retry submit.
        time.sleep(.5)
    if jobs.get(task_id) != os.environ["SLURM_JOB_ID"]:
        raise PermissionError("Worker allocation is not the exact receipted campaign job")
    parents = {"spec": spec["content_hash"], "source": spec["source"]["content_hash"]}
    cache = {}
    for dep in task["depends_on"]:
        parents[dep] = verified_receipt(spec, dep, checked=cache)["content_hash"]
    if receipt_path(spec, task_id).exists():
        return verified_receipt(spec, task_id)
    root = Path(spec["campaign_root"])
    remaining = spec["estimated_remaining_writes"]
    import shutil
    if usage(root)[0] >= TOTAL_CAP or shutil.disk_usage(root).free < 2*remaining+5*GIB:
        raise OSError("Response campaign storage/headroom budget unavailable")
    claim_dir = directory/"claims"/task_id
    claim_dir.parent.mkdir(exist_ok=True)
    claim_dir.mkdir(exist_ok=False)
    claim = artifact("TASK_CLAIM", parents=parents, task_id=task_id, allocation=allocation_record)
    publish_json(root, (claim_dir/"claim.json").relative_to(root).as_posix(), claim, "TASK_CLAIM", remaining_bytes=remaining)
    numerical = numerical_environment()
    if spec["stage"] != "acceptance" and numerical != context["execution"]["numerical_environment"]:
        raise PermissionError("Installed numerical-library bytes differ from acceptance")
    from .tasks import dispatch
    with Measurement() as measurement:
        results = dispatch(spec, task, context)
    evidence = artifact("TASK_MEASUREMENT", parents={"claim": claim["content_hash"], "source": spec["source"]["content_hash"]},
                        allocation=allocation_record, numerical_environment=numerical,
                        measurement=measurement.report(), task_id=task_id, production_worker=True,
                        final_scientific_status_is_not_an_exit_code=True)
    results["measurement"] = ("TASK_MEASUREMENT", evidence)
    validate_source(spec["source"], Path(spec["project_dir"]), executable=True)
    if numerical_environment() != numerical:
        raise ValueError("Numerical environment changed during task")
    outputs = {}
    prefix = f"{spec['stage']}/{spec['attempt']}/{task_id}"
    for name, (kind, value) in results.items():
        if name not in {"result", "response", "measurement", "examples"}:
            raise ValueError("Unexpected task output name")
        relative = ("models/" if kind == "FITTED_RESPONSE" else "examples/" if kind == "EXAMPLES" else "reports/")+prefix+f"/{name}.json"
        path = publish_json(root, relative, value, kind, remaining_bytes=remaining)
        outputs[name] = dict(relative=relative, sha256=sha256_file(path), bytes=path.stat().st_size,
                             content_hash=value["content_hash"], kind=kind)
        if kind == "EXAMPLES":
            from .plots import example_svg
            for rule, identities in value["sets"].items():
                for identity in identities:
                    data = example_svg(value["examples"][identity], response_hash=value["parents"]["response"],
                                       role=value["role"], selection_rule=rule)
                    # Identity is a hash from authenticated readers, not a path supplied by data.
                    from .contracts import canonical_sha256
                    key = canonical_sha256([rule, identity])
                    rel = f"figures/{prefix}/{key}.svg"
                    path = publish_bytes(root, rel, data, remaining_bytes=remaining)
                    outputs["figure_"+key] = dict(relative=rel, sha256=sha256_file(path), bytes=len(data), kind=None)
    result = artifact("TASK_OUTPUTS", parents=parents, task_id=task_id, job_id=os.environ["SLURM_JOB_ID"],
                      outputs=outputs, claim_hash=claim["content_hash"])
    publish_json(root, receipt_path(spec, task_id).relative_to(root).as_posix(), result, "TASK_OUTPUTS", remaining_bytes=remaining)
    return result
