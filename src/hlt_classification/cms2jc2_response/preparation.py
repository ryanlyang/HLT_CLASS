"""Isolated, read-only preparation submission. Never launches response science."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

from .audit import inventory_cms, audit_jc2_sources
from .contracts import (artifact, compatibility_template, load_json, publish, read,
                        sha256_file, validate, write_immutable_json)
from .provenance import environment, source_record, validate_source
from .splits import build_roles, build_memberships

AUTHORIZATION = "AUTHORIZE CMS2JC2 RESPONSE READ ONLY PREPARATION"
SITE = dict(partition="debug", account="reu-aisocial", qos="qos_tier3",
            cpus=4, memory_gib=16, time="02:00:00", gpus=0,
            conda_prefix="/home/ryreu/miniconda3/envs/atlas_kd_sporc")


def _external_file(path: Path) -> dict:
    path = path.resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path))


def _verify_file(record: dict) -> Path:
    path = Path(record["path"]).resolve(strict=True)
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError("Preparation input bytes changed")
    return path


def create(*, project_dir: Path, source_commit: str, campaign_root: Path,
           cms_root: Path, cms_split_manifest: Path,
           jc2_root: Path, jc2_inventory: Path, jc2_profile: Path) -> dict:
    from hlt_classification.scouting.splits import validate_split_manifest
    from hlt_classification.jetclass2_delphes.inventory import validate_inventory
    from hlt_classification.jetclass2_delphes.splits import validate_splits
    project = project_dir.resolve(strict=True)
    output = campaign_root.resolve()
    cms, jc2 = cms_root.resolve(strict=True), jc2_root.resolve(strict=True)
    if output.exists() or any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (cms,jc2)) or output == project:
        raise ValueError("Preparation needs a new root disjoint from raw datasets/project root")
    source = source_record(project,source_commit,executable=True)
    split,inv,profile = load_json(cms_split_manifest),load_json(jc2_inventory),load_json(jc2_profile)
    validate_split_manifest(split,source_manifest_sha256=split["source_manifest_sha256"])
    validate_inventory(inv); validate_splits(profile,inv)
    if profile.get("profile") != "TRAIN_500K":
        raise ValueError("Transfer audit requires registered TRAIN_500K profile")
    spec = artifact("PREPARATION_SPEC", parents={"source":source["content_hash"]},
                    source=source, project_dir=str(project), campaign_root=str(output), site=SITE,
                    cms_root=str(cms),jc2_root=str(jc2),
                    inputs={"cms_split":_external_file(cms_split_manifest),
                            "jc2_inventory":_external_file(jc2_inventory),"jc2_profile":_external_file(jc2_profile)},
                    scientific_fits=0, automatic_science_submission=False,
                    particle_arrays_read=False, read_only_sources=True)
    output.mkdir(parents=True,exist_ok=False)
    publish(output/"preparation_spec.json",spec,"PREPARATION_SPEC")
    plan = command_plan(spec)
    publish(output/"command_plan.json",plan,"COMMAND_PLAN")
    publish(output/"dry_run_submission_ledger.json",artifact("DRY_LEDGER",parents={"spec":spec["content_hash"],
            "plan":plan["content_hash"]},argv=plan["argv"],dry_run=True),"DRY_LEDGER")
    return spec


def validate_spec(spec:dict):
    validate(spec,"PREPARATION_SPEC",parents={"source":spec["source"]["content_hash"]})
    if spec["site"] != SITE or spec["scientific_fits"] != 0 or spec["automatic_science_submission"] is not False:
        raise ValueError("Preparation cannot turn into scientific execution")
    for record in spec["inputs"].values():
        _verify_file(record)
    validate_source(spec["source"],Path(spec["project_dir"]),executable=True)


def command_plan(spec:dict) -> dict:
    validate(spec,"PREPARATION_SPEC")
    if spec["site"] != SITE:
        raise ValueError("Unregistered preparation resource envelope")
    root,project = Path(spec["campaign_root"]),Path(spec["project_dir"])
    argv = ["sbatch","--parsable","--no-requeue","--nodes=1","--ntasks=1",
            "--cpus-per-task=4","--mem=16G","--time=02:00:00",
            f"--partition={SITE['partition']}",f"--account={SITE['account']}",f"--qos={SITE['qos']}",
            "--job-name=cms2jc2_prepare",f"--chdir={project}",f"--output={root}/slurm-%j.out",
            "--export=NONE",str(project/"sbatch/run_cms2jc2_response_cpu.sh"),str(project),
            str(root/"preparation_spec.json")]
    return artifact("COMMAND_PLAN",parents={"spec":spec["content_hash"]},argv=argv,
                    tasks=["read_only_preparation"],global_cpu_bound=4,scientific_fits=0)


def submit(spec:dict,*,execute:bool=False,authorization_phrase:str|None=None) -> dict:
    validate_spec(spec)
    root = Path(spec["campaign_root"]); plan = command_plan(spec)
    if read(root/"command_plan.json","COMMAND_PLAN") != plan:
        raise ValueError("Preparation plan changed")
    dry = read(root/"dry_run_submission_ledger.json","DRY_LEDGER")
    if dry["parents"] != {"spec":spec["content_hash"],"plan":plan["content_hash"]} or dry["argv"] != plan["argv"]:
        raise ValueError("Canonical dry ledger differs")
    if not execute:
        return dry
    if authorization_phrase != AUTHORIZATION:
        raise PermissionError("Exact read-only preparation authorization required")
    # Check the declared CPU partition is usable before writing a submission
    # intent. Account/QoS eligibility is finally enforced by sbatch itself.
    partition = SITE["partition"]
    site = subprocess.run(["scontrol","show","partition",partition,"-o"],
                          text=True,capture_output=True,check=True)
    if f"PartitionName={partition}" not in site.stdout or "State=UP" not in site.stdout:
        raise PermissionError(f"SPORC {partition} partition is not confirmed UP")
    receipt = root/"submission_ledger.json"
    if receipt.exists():
        value = read(receipt,"SUBMISSION_RECEIPT")
        if value["parents"] != {"spec":spec["content_hash"],"plan":plan["content_hash"]}:
            raise ValueError("Existing receipt belongs to another preparation")
        return value
    if shutil.disk_usage(root).free < 7*1024**3:
        raise OSError("Preparation needs at least 7 GiB free shared storage")
    # Durable exclusive intent: a lost sbatch acknowledgement must be reconciled
    # by exact job ID, not retried blindly or overwritten.
    intent = root/"submission_intent"
    intent.mkdir(exist_ok=False)
    publish(intent/"intent.json",artifact("SUBMISSION_INTENT",parents={"spec":spec["content_hash"],
            "plan":plan["content_hash"]},argv=plan["argv"]),"SUBMISSION_INTENT")
    result = subprocess.run(plan["argv"],text=True,capture_output=True,check=False)
    if result.returncode or not re.fullmatch(r"[1-9][0-9]*(;[A-Za-z0-9_-]+)?\n?",result.stdout):
        raise RuntimeError("Submission ambiguous/failed; retain intent and reconcile exact receipt. "
                           +result.stdout+result.stderr)
    job = result.stdout.strip().split(";")[0]
    value = artifact("SUBMISSION_RECEIPT",parents={"spec":spec["content_hash"],"plan":plan["content_hash"]},
                     jobs={"read_only_preparation":job},argv=plan["argv"],dry_run=False)
    publish(receipt,value,"SUBMISSION_RECEIPT")
    return value


def run(spec:dict) -> dict:
    validate_spec(spec)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != SITE["partition"]:
        raise PermissionError(f"Preparation worker requires a real {SITE['partition']} allocation")
    if int(os.environ.get("SLURM_CPUS_PER_TASK","0")) != SITE["cpus"]:
        raise PermissionError("CPU allocation differs from preparation")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or os.environ.get("CONDA_PREFIX") != SITE["conda_prefix"]:
        raise PermissionError("Preparation environment differs")
    root = Path(spec["campaign_root"])
    complete = root/"preparation_report.json"
    if complete.exists():
        value = read(complete,"PREPARATION_REPORT")
        for row in value["outputs"]:
            _verify_file(row)
        if value["parents"] != {"spec":spec["content_hash"]}:
            raise ValueError("Preparation output lineage differs")
        return value
    (root/"worker_claim").mkdir(exist_ok=False)
    publish(root/"worker_claim"/"claim.json",artifact("WORKER_CLAIM",parents={"spec":spec["content_hash"]},
            job_id=os.environ["SLURM_JOB_ID"]),"WORKER_CLAIM")
    inventory = inventory_cms(Path(spec["cms_root"]),load_json(_verify_file(spec["inputs"]["cms_split"])))
    publish(root/"cms_inventory.json",inventory,"CMS_INVENTORY")
    roles = build_roles(inventory); publish(root/"response_roles.json",roles,"ROLES")
    memberships = build_memberships(roles); publish(root/"response_memberships.json",memberships,"MEMBERSHIPS")
    jc2_audit = audit_jc2_sources(Path(spec["jc2_root"]),load_json(_verify_file(spec["inputs"]["jc2_inventory"])),
                                 load_json(_verify_file(spec["inputs"]["jc2_profile"])))
    publish(root/"jc2_source_audit.json",jc2_audit,"JC2_SOURCE_AUDIT")
    review = compatibility_template(inventory_hash=inventory["content_hash"])
    publish(root/"compatibility_review_required.json",review,"COMPATIBILITY_REVIEW")
    env = environment(); publish(root/"environment.json",env,"ENVIRONMENT")
    validate_source(spec["source"],Path(spec["project_dir"]),executable=True)
    value = artifact("PREPARATION_REPORT",parents={"spec":spec["content_hash"]},
                     outputs=[_external_file(root/name) for name in ("cms_inventory.json","response_roles.json",
                              "response_memberships.json","jc2_source_audit.json",
                              "compatibility_review_required.json","environment.json")],
                     role_counts=roles["counts"],scientific_fits=0,remote_cpu_metadata_job_completed=True,
                     production_worker_miniature_passed=False,science_queue_ready=False,
                     blockers=["Human producer-semantic review unresolved",
                               "Response generator/evaluation/science orchestration not yet integrated",
                               "Real CPU miniature and full-size resource profile required"])
    publish(complete,value,"PREPARATION_REPORT")
    return value
