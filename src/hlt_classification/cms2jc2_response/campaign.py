"""Isolated staged campaign creation; creation never invokes a scheduler."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import artifact, load_json, safe_relative, sha256_file, validate, validate_compatibility
from .graph import science_graph
from .provenance import source_record, validate_source
from .storage import GIB, TOTAL_CAP, publish_json

STAGES = ("acceptance", "science", "confirmation")
PHRASES = {s: f"AUTHORIZE CMS2JC2 RESPONSE {s.upper()} EXACT PLAN" for s in STAGES}
SITE = dict(partition="debug", account="reu-aisocial", qos="qos_tier3", nodes=1, tasks=1, gpus=0,
            conda_prefix="/home/ryreu/miniconda3/envs/atlas_kd_sporc")
DEFAULT_WRITES = 2*GIB


def file_ref(path: Path):
    path = path.resolve(strict=True)
    value = load_json(path)
    from hlt_classification.data.cache_contracts import validate_content_hash
    digest = validate_content_hash(value, expected_contract=value["contract"], expected_schema_version=value["schema_version"])
    return dict(path=str(path), sha256=sha256_file(path), content_hash=digest)


def resolve_ref(ref):
    from hlt_classification.data.cache_contracts import validate_content_hash
    path = Path(ref["path"]).resolve(strict=True)
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("Referenced artifact bytes changed")
    value = load_json(path)
    if validate_content_hash(value, expected_contract=value["contract"], expected_schema_version=value["schema_version"]) != ref["content_hash"]:
        raise ValueError("Referenced artifact identity differs")
    return value


def _task(name, action, deps=(), *, cpus=2, memory=8, hours=2, **params):
    return dict(task_id=name, action=action, depends_on=list(deps), params=params,
                resources=dict(cpus=cpus, memory_gib=memory, time=f"{hours:02}:00:00"))


def acceptance_graph():
    rows = [_task(f"miniature_{f}_L", "probe_fit", cpus=16, memory=128, hours=24,
                  candidate_id=f"{f}_L", probe_jets=20_000) for f in "ABC"]
    rows.append(_task("miniature_verify", "miniature_verify", [r["task_id"] for r in rows],
                      cpus=16, memory=128, hours=24))
    for f in "ABC":
        rows.append(_task(f"profile_{f}_H", "probe_fit", ["miniature_verify"],
                          cpus=16, memory=128, hours=24, candidate_id=f"{f}_H", probe_jets=100_000))
    rows.append(_task("profile_metrics", "profile_metrics", [f"profile_{f}_H" for f in "ABC"],
                      cpus=1, memory=128, hours=24))
    for f in "ABC":
        rows.append(_task(f"profile_evaluate_{f}_H", "profile_evaluate", ["profile_metrics"],
                          cpus=1, memory=32, hours=8, fit_task=f"profile_{f}_H", probe_jets=100_000))
    for gate in (.05, .1, .2):
        rows.append(_task(f"profile_association_G{int(100*gate):02d}", "profile_association", [f"profile_evaluate_{f}_H" for f in "ABC"],
                          cpus=8, memory=32, hours=6, gate=gate, probe_jets=100_000))
    rows.append(_task("execution_lock", "execution_lock", [r["task_id"] for r in rows]))
    return rows


def science_tasks(execution):
    validate(execution, "EXECUTION_LOCK")
    if execution.get("ready") is not True or execution.get("resource_blockers"):
        raise PermissionError("Measured resources do not permit science submission")
    rows = []
    for task in science_graph()["tasks"]:
        kind, params = task["kind"], task["params"]
        if kind == "fit":
            family = params.get("family_finalist", params.get("candidate_id", "")[0:1])
            resource = execution["resources"][f"fit_{family}"]
        elif kind == "evaluate":
            # Upper bound across the three measured high-complexity families.
            resource = execution["resources"]["evaluate"]
        elif kind == "visual":
            resource = execution["resources"]["visual_select"]
        else:
            resource = execution["resources"]["metrics" if kind == "metrics" else "report"]
        rows.append(dict(task_id=task["task_id"], action=kind if kind not in {"report", "visual"} else task["task_id"],
                         depends_on=task["depends_on"], params=params, resources=deepcopy(resource)))
    return rows


def confirmation_tasks(selection, execution):
    selected = selection["selected_candidate"]
    names = [selected]+[v for k, v in sorted(selection["family_finalists"].items()) if k != selected[0]]
    if len(set(names)) != 3:
        raise ValueError("Confirmation must freeze one response from each family")
    rows = [dict(task_id="confirm_"+c, action="confirm", depends_on=[], params=dict(candidate_id=c),
                 resources=deepcopy(execution["resources"]["confirm"])) for c in names]
    rows.append(_task("confirmation_report", "confirmation_report", [r["task_id"] for r in rows]))
    for role in ("train", "validation"):
        rows.append(dict(task_id="transfer_"+role, action="transfer", depends_on=["confirmation_report"],
                         params=dict(role=role), resources=deepcopy(execution["resources"]["transfer_"+role])))
    rows.append(dict(task_id="visual_confirm", action="visual_confirm", depends_on=["transfer_train", "transfer_validation"],
                     params={}, resources=deepcopy(execution["resources"]["visual_confirm"])))
    rows.append(dict(task_id="visual_jc2", action="visual_jc2", depends_on=["visual_confirm"],
                     params={}, resources=deepcopy(execution["resources"]["visual_jc2"])))
    rows.append(_task("campaign_complete", "campaign_complete", [r["task_id"] for r in rows]))
    return rows


def stage_dir(spec):
    return safe_relative(Path(spec["campaign_root"]), "stages/"+spec["stage"]+"/"+spec["attempt"])


def validate_spec(spec, *, source=True):
    validate(spec, "CAMPAIGN_SPEC")
    if spec["parents"].get("source") != spec["source"]["content_hash"]:
        raise ValueError("Campaign source parent differs")
    if (spec.get("stage") not in STAGES or spec.get("site") != SITE
            or spec.get("storage_cap_bytes") != TOTAL_CAP or spec.get("combined_cpu_cap") != 64
            or spec.get("automatic_followon_submission") is not False):
        raise ValueError("Campaign site/stage/storage contract differs")
    stage_dir(spec)  # Also rejects unsafe attempt names.
    if source:
        validate_source(spec["source"], Path(spec["project_dir"]), executable=True)
    context = {k: resolve_ref(v) for k, v in spec["inputs"].items()}
    from .audit import validate_inventory
    from .splits import validate_roles, validate_memberships
    validate_inventory(context["inventory"])
    validate_roles(context["roles"], context["inventory"])
    validate_memberships(context["membership"], context["roles"])
    validate_compatibility(context["compatibility"], inventory_hash=context["inventory"]["content_hash"])
    from hlt_classification.jetclass2_delphes.inventory import validate_inventory as validate_jc2
    from hlt_classification.jetclass2_delphes.splits import validate_splits
    validate_jc2(context["jc2_inventory"])
    validate_splits(context["jc2_profile"], context["jc2_inventory"])
    if context["jc2_profile"].get("profile") != "TRAIN_500K":
        raise ValueError("Transfer profile differs from registered 500k study")
    validate(context["jc2_audit"], "JC2_SOURCE_AUDIT", parents={
        "inventory": context["jc2_inventory"]["content_hash"], "profile": context["jc2_profile"]["content_hash"]})
    if spec["stage"] == "acceptance":
        expected = acceptance_graph()
    else:
        execution = context["execution"]
        validate(execution, "EXECUTION_LOCK")
        if execution["parents"]["source"] != spec["source"]["content_hash"]:
            raise ValueError("Production evidence is not bound to current source")
        if execution["inputs"] != {k: v["content_hash"] for k, v in spec["inputs"].items()
                                   if k not in {"execution", "selection", "science_spec"}}:
            raise ValueError("Production evidence data/conventions differ")
        expected = science_tasks(execution) if spec["stage"] == "science" else confirmation_tasks(context["selection"], execution)
    if spec["tasks"] != expected:
        raise ValueError("Registered stage graph/resources were changed")
    done = set()
    for task in spec["tasks"]:
        if task["task_id"] in done or not set(task["depends_on"]) <= done:
            raise ValueError("Task graph ordering differs")
        done.add(task["task_id"])
    if not set(spec["reusable_tasks"]) <= done:
        raise ValueError("Reused task escapes stage coverage")
    for task, row in spec["reusable_tasks"].items():
        parent = resolve_ref(row["spec"])
        validate(parent, "CAMPAIGN_SPEC")
        if (parent["content_hash"] == spec["content_hash"] or parent["source"] != spec["source"]
                or parent["tasks"] != spec["tasks"] or parent["inputs"] != spec["inputs"]
                or parent["campaign_root"] != spec["campaign_root"] or parent["stage"] != spec["stage"]):
            raise ValueError("Recovery reuse escapes the same source-bound scientific stage")
    return context


def _publish_spec(spec):
    root = Path(spec["campaign_root"])
    directory = stage_dir(spec)
    directory.mkdir(parents=True, exist_ok=False)
    remaining = spec["estimated_remaining_writes"]
    publish_json(root, str(directory.relative_to(root)/"campaign_spec.json").replace("\\", "/"),
                 spec, "CAMPAIGN_SPEC", remaining_bytes=remaining)
    plan = command_plan(spec)
    publish_json(root, str(directory.relative_to(root)/"command_plan.json").replace("\\", "/"),
                 plan, "COMMAND_PLAN", remaining_bytes=remaining)
    if spec["stage"] == "confirmation":
        ctx = {k: resolve_ref(spec["inputs"][k]) for k in ("roles", "selection", "compatibility")}
        claim = artifact("CONFIRMATION_CLAIM", parents={"roles": ctx["roles"]["content_hash"],
                          "selection": ctx["selection"]["content_hash"], "compatibility": ctx["compatibility"]["content_hash"]},
                          execution_spec=spec["content_hash"], separate_stage_authorization_required=True)
        publish_json(root, (directory/"confirmation_claim.json").relative_to(root).as_posix(),
                     claim, "CONFIRMATION_CLAIM", remaining_bytes=remaining)
    return spec


def create_acceptance(*, preparation_spec: Path, compatibility: Path, campaign_root: Path,
                      project_dir: Path, source_commit: str):
    prep = load_json(preparation_spec)
    validate(prep, "PREPARATION_SPEC")
    prep_root = Path(prep["campaign_root"])
    report = load_json(prep_root/"preparation_report.json")
    validate(report, "PREPARATION_REPORT", parents={"spec": prep["content_hash"]})
    expected = {"cms_inventory.json", "response_roles.json", "response_memberships.json", "jc2_source_audit.json",
                "compatibility_review_required.json", "environment.json"}
    if {Path(r["path"]).name for r in report["outputs"]} != expected:
        raise ValueError("Preparation output coverage differs")
    for record in report["outputs"]:
        if Path(record["path"]).resolve().parent != prep_root.resolve() or sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("Preparation receipt was corrupted")
    output = campaign_root.resolve()
    forbidden = [Path(prep[k]).resolve(strict=True) for k in ("cms_root", "jc2_root")]+[prep_root.resolve()]
    if output.exists() or any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in forbidden):
        raise ValueError("Acceptance needs a fresh root disjoint from sources and preparation")
    source = source_record(project_dir, source_commit, executable=True)
    inputs = {k: file_ref(prep_root/v) for k, v in dict(inventory="cms_inventory.json", roles="response_roles.json",
              membership="response_memberships.json", jc2_audit="jc2_source_audit.json").items()}
    inputs.update(compatibility=file_ref(compatibility), preparation=file_ref(preparation_spec),
                  preparation_report=file_ref(prep_root/"preparation_report.json"))
    for k in ("jc2_inventory", "jc2_profile"):
        ref = prep["inputs"][k]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("JC2 preparation source changed")
        inputs[k] = file_ref(Path(ref["path"]))
    spec = artifact("CAMPAIGN_SPEC", parents={"source": source["content_hash"], "preparation": prep["content_hash"]},
                    source=source, project_dir=str(project_dir.resolve()), campaign_root=str(output),
                    cms_root=prep["cms_root"], jc2_root=prep["jc2_root"], inputs=inputs, site=SITE,
                    stage="acceptance", attempt="r1", tasks=acceptance_graph(), combined_cpu_cap=64,
                    storage_cap_bytes=TOTAL_CAP, estimated_remaining_writes=DEFAULT_WRITES,
                    automatic_followon_submission=False, reusable_tasks={})
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=False)
    return _publish_spec(spec)


def create_followon(previous_spec: Path, *, stage: str, execution_path: Path,
                    selection_path: Path | None = None):
    if stage not in {"science", "confirmation"}:
        raise ValueError("Invalid follow-on stage")
    previous = load_json(previous_spec)
    validate_spec(previous)
    if previous["stage"] != ("acceptance" if stage == "science" else "science"):
        raise PermissionError("Stage order differs")
    from .orchestration import verified_product
    if stage == "science":
        expected_path = verified_product(previous, "execution_lock", "result")
    else:
        expected_path = Path(previous["inputs"]["execution"]["path"])
    if execution_path.resolve() != expected_path.resolve():
        raise ValueError("Execution lock must be an attested acceptance output")
    execution = load_json(execution_path)
    inputs = deepcopy(previous["inputs"])
    inputs["execution"] = file_ref(execution_path)
    if stage == "confirmation":
        selected_path = verified_product(previous, "selection_lock", "result")
        verified_product(previous, "comparison_complete", "result")
        if selection_path is None or selection_path.resolve() != selected_path.resolve():
            raise PermissionError("Confirmation requires the attested locked selection")
        inputs["selection"] = file_ref(selection_path)
        inputs["science_spec"] = file_ref(previous_spec)
        tasks = confirmation_tasks(load_json(selection_path), execution)
    else:
        tasks = science_tasks(execution)
    spec = artifact("CAMPAIGN_SPEC", parents={"source": previous["source"]["content_hash"], "previous": previous["content_hash"]},
                    **{k: previous[k] for k in ("source", "project_dir", "campaign_root", "cms_root", "jc2_root", "site",
                                                "combined_cpu_cap", "storage_cap_bytes", "automatic_followon_submission")},
                    stage=stage, attempt="r1", inputs=inputs, tasks=tasks, reusable_tasks={},
                    estimated_remaining_writes=execution["estimated_remaining_writes"])
    validate_spec(spec)
    return _publish_spec(spec)


def command_plan(spec):
    validate(spec, "CAMPAIGN_SPEC")
    if spec["site"] != SITE:
        raise ValueError("Campaign site differs; create a fresh source-pinned debug specification")
    project, directory = Path(spec["project_dir"]), stage_dir(spec)
    rows = []
    for row in spec["tasks"]:
        if row["task_id"] in spec["reusable_tasks"]:
            continue
        r = row["resources"]
        argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1",
                f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_gib']}G", f"--time={r['time']}",
                f"--partition={SITE['partition']}", f"--account={SITE['account']}", f"--qos={SITE['qos']}",
                f"--comment=c2jr:{spec['content_hash']}:{row['task_id']}",
                f"--job-name=c2jr_{row['task_id']}", f"--chdir={project}",
                f"--output={directory}/slurm-%j.out", "--export=NONE",
                str(project/"sbatch/run_cms2jc2_response_cpu.sh"), str(project),
                str(directory/"campaign_spec.json"), row["task_id"]]
        rows.append(dict(task_id=row["task_id"], argv=argv, depends_on=row["depends_on"], resources=r))
    def hours(value):
        h, m, s = map(int, value.split(":")); return h+m/60+s/3600
    return artifact("COMMAND_PLAN", parents={"spec": spec["content_hash"]}, commands=rows,
                    dry_run=True, dependency_resolution="afterok_exact_receipted_parent_ids_only",
                    allocated_cpu_hour_upper_bound=sum(r["resources"]["cpus"]*hours(r["resources"]["time"]) for r in rows),
                    combined_cpu_upper_bound=64, automatic_followon_submission=False,
                    estimated_remaining_writes=spec["estimated_remaining_writes"])
