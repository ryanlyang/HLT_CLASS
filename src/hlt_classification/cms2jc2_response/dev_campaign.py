"""Isolated, source-pinned CPU development stages; never submits implicitly."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from .association import policy
from .assumptions import provisional_compatibility
from .contracts import artifact, load_json, safe_relative, sha256_file, validate, with_content_hash
from .dev_data import checked_file, file_ref, import_preparation, validate_import
from .provenance import source_record, numerical_environment
from .storage import GIB, publish_json

PLAN = "docs/plans/CMS2JC2_RESPONSE_CPU_DEVELOPMENT_PLAN.md"
EXTRA_SOURCE = (PLAN, "docs/contracts/CMS2JC2_RESPONSE_CPU_DEVELOPMENT.md",
                "scripts/cms2jc2_response_dev.py", "sbatch/run_cms2jc2_response_dev_cpu.sh")
POLICIES = {
    "BASE": policy(),
    "SIZE": policy(max_component_objects=256, max_component_hypotheses=16384),
    "SEARCH": policy(search_nodes=1_000_000),
    "BOTH": policy(max_component_objects=256, max_component_hypotheses=16384, search_nodes=1_000_000),
}
PHRASES = {s: f"AUTHORIZE CMS2JC2 CPU DEVELOPMENT {s.upper()} EXACT PLAN" for s in ("pilot", "confirm", "compare", "cdiag", "ctopo", "btrack")}
PHRASES.update({s: f"AUTHORIZE CMS2JC2 {s.upper()} EXACT PLAN"
                for s in ("bounded_gate", "bounded_compare", "bounded_confirm")})
PHRASES.update({s: f"AUTHORIZE CMS2JC2 {s.upper()} EXACT PLAN" for s in ("bdz_gate", "bdz_compare")})
REMAINING_WRITES = 2*GIB


def source_snapshot(project, commit=None, *, executable=True):
    project = Path(project)
    source = source_record(project, commit, executable=executable)
    extra = EXTRA_SOURCE + ("docs/plans/CMS2JC2_FROZEN_C_DIAGNOSTIC_PLAN.md",
                            "docs/contracts/CMS2JC2_FROZEN_C_DIAGNOSTIC.md",
                            "docs/plans/CMS2JC2_C_OBSERVED_TOPOLOGY_DIAGNOSTIC_PLAN.md",
                            "docs/contracts/CMS2JC2_C_OBSERVED_TOPOLOGY_DIAGNOSTIC.md",
                            "docs/plans/CMS2JC2_FROZEN_B_TRACKING_AUDIT_PLAN.md",
                            "docs/contracts/CMS2JC2_FROZEN_B_TRACKING_AUDIT.md",
                            "docs/plans/CMS2JC2_B_TRACKING_DEBUG_REPLACEMENT_PLAN.md",
                            "docs/contracts/CMS2JC2_B_TRACKING_DEBUG_REPLACEMENT.md",
                            "docs/plans/CMS2JC2_C_TOPOLOGY_DEBUG_REPLACEMENT_PLAN.md",
                            "docs/contracts/CMS2JC2_C_TOPOLOGY_DEBUG_REPLACEMENT.md",
                            "docs/plans/CMS2JC2_BOUNDED_FINAL_COMPARISON_PLAN.md",
                            "docs/contracts/CMS2JC2_BOUNDED_FINAL_COMPARISON.md",
                            "docs/plans/CMS2JC2_BDZ_TRACKING_TUNING_PLAN.md",
                            "docs/contracts/CMS2JC2_BDZ_TRACKING_TUNING.md",
                            "docs/plans/CMS2JC2_BDZ_TIER3_REPLACEMENT_PLAN.md",
                            "docs/contracts/CMS2JC2_BDZ_TIER3_REPLACEMENT.md",
                            "scripts/queue_cms2jc2_bdz_tier3.sh")
    if executable:
        for name in extra:
            subprocess.check_output(["git", "-C", str(project), "ls-files", "--error-unmatch", name])
    return with_content_hash({**source, "files": {**source["files"], **{n: sha256_file(project/n) for n in extra}}})


def write(root, relative, value, kind):
    return publish_json(Path(root), relative, value, kind, remaining_bytes=REMAINING_WRITES)


def create_study(*, project_dir, source_commit, preparation_spec, root, partition="debug"):
    project, root = Path(project_dir).resolve(strict=True), Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", partition):
        raise ValueError("One explicit CPU partition is required")
    donor = import_preparation(Path(preparation_spec), project)
    for other in (Path(donor["cms_root"]).resolve(), Path(donor["donor_root"]), project):
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError("Development root must be disjoint from data, donor and checkout")
    if root.exists():
        raise FileExistsError("Use a fresh isolated development root")
    source = source_snapshot(project, source_commit)
    environment = numerical_environment()
    inventory = load_json(checked_file(donor["files"]["cms_inventory.json"]))
    value = artifact("DEV_STUDY", parents={"source": source["content_hash"], "import": donor["content_hash"]},
        project_dir=str(project), root=str(root), source=source, imported=donor, numerical_environment=environment,
        review=provisional_compatibility(inventory_hash=inventory["content_hash"]),
        site=dict(partition=partition, account="reu-aisocial", qos="qos_tier3", nodes=1, tasks=1, gpus=0,
                  conda_prefix="/home/ryreu/miniconda3/envs/atlas_kd_sporc"),
        intended_jc2_release="jetclass2_10M_20260918_dzfix", jc2_particle_access=False,
        production_acceptance=False, automatic_stage_submission=False,
        producer_update="User-supplied Luka reply confirms CMS cm/Delphes mm; other conventions remain provisional.")
    root.mkdir(parents=True, exist_ok=False)
    write(root, "study_spec.json", value, "DEV_STUDY")
    return value


def validate_study(study, *, source=True):
    validate(study, "DEV_STUDY", parents={"source": study["source"]["content_hash"], "import": study["imported"]["content_hash"]})
    project = Path(study["project_dir"])
    validate_import(study["imported"], project)
    site = study["site"]
    if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", site["partition"]) or site != dict(
            partition=site["partition"], account="reu-aisocial", qos="qos_tier3", nodes=1, tasks=1, gpus=0,
            conda_prefix="/home/ryreu/miniconda3/envs/atlas_kd_sporc")):
        raise ValueError("CPU development site differs")
    inventory = load_json(checked_file(study["imported"]["files"]["cms_inventory.json"]))
    if study["review"] != provisional_compatibility(inventory_hash=inventory["content_hash"]):
        raise ValueError("Development physical interface changed")
    if (study["jc2_particle_access"] is not False or study["production_acceptance"] is not False
            or study["automatic_stage_submission"] is not False
            or study["intended_jc2_release"] != "jetclass2_10M_20260918_dzfix"):
        raise PermissionError("Development access boundary changed")
    if source and source_snapshot(project, study["source"]["commit"]) != study["source"]:
        raise ValueError("Development source changed")


def task(name, action, cpus, memory, hours, deps=(), mode="afterok", **params):
    return dict(task_id=name, action=action, cpus=cpus, memory_gib=memory, hours=hours,
                depends_on=list(deps), dependency_mode=mode, params=params)


def tasks(stage, policy_id, b_threads, *, cpu64=False, preprocessing_cpus=None):
    if cpu64:
        if preprocessing_cpus is not None:
            raise ValueError("Choose one preprocessing execution profile")
        preprocessing_cpus = 64  # Preserve the historical caller interface.
    if preprocessing_cpus is not None:
        if type(preprocessing_cpus) is not int or preprocessing_cpus not in (36, 64):
            raise ValueError("Unregistered preprocessing CPU count")
        if stage != "compare" or b_threads != 1:
            raise ValueError("Chunked preparation is comparison-only with single-thread B")
    if stage == "pilot":
        return [task("prepare", "prepare", 1, 16, 4), *[
            task("assoc_"+p, "association", 8, 32, 12, ["prepare"], policy=p, population="pilot") for p in POLICIES], *[
            task(f"B_threads_{n}", "b_diagnostic", 16, 128, 12, ["prepare"], threads=n) for n in (1, 16)]]
    if stage == "confirm":
        return [task("association_confirm", "association", 16, 128, 24, policy=policy_id, population="fitting")]
    if stage != "compare":
        raise ValueError("Unknown development stage")
    result = [task("fit_AC", "fit", 16, 128, 24, candidates=["A_L", "C_L"], threads=16),
              task("fit_B", "fit", 16, 128, 24, candidates=["B_L"], threads=b_threads)]
    if preprocessing_cpus is not None:
        for row in result:
            row["cpus"] = preprocessing_cpus
    for f in "ABC":
        parent = "fit_B" if f == "B" else "fit_AC"
        names = [f"evaluate_{f}_{i}" for i in range(4)]
        result += [task(name, "evaluate", 1, 32, 12, [parent], mode="afterany", candidate=f+"_L", shard=i)
                   for i, name in enumerate(names)]
        result.append(task("report_"+f, "report", 1, 16, 2, names, candidate=f+"_L"))
    return result


def stage_dir(spec):
    return Path(spec["root"])/"stages"/spec["name"]


def verified_outputs(spec, owner):
    if not re.fullmatch(r"[A-Za-z0-9_]+", owner):
        raise ValueError("Unsafe development owner")
    path = stage_dir(spec)/"receipts"/(owner+".json")
    receipt = load_json(path)
    validate(receipt, "DEV_OUTPUTS", parents={"stage": spec["content_hash"]})
    if receipt["owner"] != owner:
        raise ValueError("Output owner differs")
    for row in receipt["outputs"].values():
        target = safe_relative(Path(spec["root"]), row["relative"])
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            raise ValueError("Corrupt development product")
    return receipt


def product(spec, owner, key):
    row = verified_outputs(spec, owner)["outputs"][key]
    return load_json(safe_relative(Path(spec["root"]), row["relative"]))


def create_stage(study_path, *, stage, name, parent_spec=None, policy_id=None, b_threads=1):
    study_path = Path(study_path).resolve(strict=True)
    study = load_json(study_path)
    validate_study(study)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) or b_threads not in (1, 16):
        raise ValueError("Invalid stage name/thread count")
    parent_ref = None
    if stage == "pilot":
        if parent_spec is not None or policy_id is not None:
            raise ValueError("Pilot may not import a chosen policy")
    else:
        if parent_spec is None:
            raise ValueError("A completed parent stage is required")
        parent_ref = file_ref(parent_spec)
        parent = load_json(checked_file(parent_ref))
        validate_stage(parent)
        if parent["study"] != file_ref(study_path):
            raise PermissionError("Stages cannot cross studies")
        if stage == "confirm":
            if parent["stage"] != "pilot" or policy_id not in POLICIES:
                raise ValueError("Choose a registered pilot policy")
            verified_outputs(parent, "prepare")
            verified_outputs(parent, "assoc_"+policy_id)
        elif stage == "compare":
            if parent["stage"] != "confirm":
                raise ValueError("Comparison needs the 20k association confirmation")
            report = product(parent, "association_confirm", "result")
            if report["jets"] != 20000 or report["policy"] != parent["policy"]:
                raise ValueError("Association confirmation population/policy differs")
            if policy_id is not None and policy_id != parent["policy"]:
                raise ValueError("Cannot change confirmed policy")
            policy_id = parent["policy"]
        else:
            raise ValueError("Unknown stage")
    spec = artifact("DEV_STAGE", parents={"study": study["content_hash"]}, study=file_ref(study_path),
                    root=study["root"], stage=stage, name=name, parent_spec=parent_ref,
                    policy=policy_id, b_threads=b_threads, tasks=tasks(stage, policy_id, b_threads),
                    scientific_qualification=False, resources_are_development_envelopes=True)
    directory = stage_dir(spec)
    directory.mkdir(parents=True, exist_ok=False)
    write(spec["root"], f"stages/{name}/stage_spec.json", spec, "DEV_STAGE")
    plan = command_plan(spec, study)
    write(spec["root"], f"stages/{name}/command_plan.json", plan, "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    if spec.get("contract") == "CMS2JC2_RESPONSE_BDZ_TIER3/v1":
        from .bdz_tier3 import validate_stage as validate_tier3
        return validate_tier3(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_BDZ_STAGE/v1":
        from .bdz_campaign import validate_stage as validate_bdz
        return validate_bdz(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_BOUNDED_STAGE/v1":
        from .bounded_campaign import validate_stage as validate_bounded
        return validate_bounded(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_C_TOPOLOGY_DEBUG/v1":
        from .c_topology_debug import validate_stage as validate_debug
        return validate_debug(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_B_TRACKING_DEBUG/v1":
        from .b_tracking_debug import validate_stage as validate_debug
        return validate_debug(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_B_TRACKING/v1":
        from .b_tracking import validate_stage as validate_bt
        return validate_bt(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_C_TOPOLOGY/v1":
        from .c_topology import validate_stage as validate_ct
        return validate_ct(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_C_DIAGNOSTIC/v1":
        from .c_diagnostic import validate_stage as validate_c
        return validate_c(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_STAGE36/v1":
        from .dev_restart import validate_compare36
        return validate_compare36(spec, source=source)
    if spec.get("contract") == "CMS2JC2_RESPONSE_DEV_STAGE64/v1":
        from .dev_restart import validate_compare64
        return validate_compare64(spec, source=source)
    study = load_json(checked_file(spec["study"]))
    validate_study(study, source=source)
    validate(spec, "DEV_STAGE", parents={"study": study["content_hash"]})
    if (spec["root"] != study["root"] or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", spec["name"])
            or spec["tasks"] != tasks(spec["stage"], spec["policy"], spec["b_threads"])
            or spec["b_threads"] not in (1, 16) or spec["scientific_qualification"] is not False
            or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Development registration differs")
    if spec["stage"] == "pilot":
        if spec["parent_spec"] is not None or spec["policy"] is not None:
            raise ValueError("Invalid pilot ancestry")
    else:
        parent = load_json(checked_file(spec["parent_spec"]))
        validate_stage(parent, source=False)
        if parent["study"] != spec["study"] or spec["policy"] not in POLICIES:
            raise ValueError("Invalid development ancestry")
        expected = "pilot" if spec["stage"] == "confirm" else "confirm"
        if parent["stage"] != expected:
            raise ValueError("Wrong parent stage")
        if spec["stage"] == "compare" and parent["policy"] != spec["policy"]:
            raise ValueError("Confirmed association policy changed")
    return study


def preparation_stage(spec):
    current = spec
    while current["stage"] != "pilot":
        current = load_json(checked_file(current["parent_spec"]))
    return current


def command_plan(spec, study):
    commands = []
    root, project, site = stage_dir(spec), Path(study["project_dir"]), study["site"]
    for t in spec["tasks"]:
        argv = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--export=NONE",
                f"--cpus-per-task={t['cpus']}", f"--mem={t['memory_gib']}G", f"--time={t['hours']:02}:00:00",
                f"--partition={site['partition']}", f"--account={site['account']}", f"--qos={site['qos']}",
                f"--job-name=c2jd_{t['task_id']}", f"--comment=c2jd:{spec['content_hash']}:{t['task_id']}",
                f"--chdir={project}", f"--output={root}/slurm-%j.out", str(project/"sbatch/run_cms2jc2_response_dev_cpu.sh"),
                str(project), str(root/"stage_spec.json"), t["task_id"]]
        commands.append(dict(task_id=t["task_id"], depends_on=t["depends_on"], dependency_mode=t["dependency_mode"], argv=argv))
    return artifact("DEV_PLAN", parents={"stage": spec["content_hash"]}, commands=commands,
                    cpu_upper_bound={"CMS2JC2_RESPONSE_DEV_STAGE64/v1": 143,
                                     "CMS2JC2_RESPONSE_DEV_STAGE36/v1": 87}.get(
                                         spec.get("contract"), {"pilot": 64, "confirm": 16, "compare": 47, "cdiag": 27, "ctopo": 144, "btrack": 144,
                                         "bounded_gate": 36, "bounded_compare": 144, "bounded_confirm": 144,
                                         "bdz_gate": 36, "bdz_compare": 144}[spec["stage"]]),
                    allocated_cpu_hour_upper_bound=sum(t["cpus"]*t["hours"] for t in spec["tasks"]),
                    gpus=0, live_authorization_phrase=PHRASES[spec["stage"]])
