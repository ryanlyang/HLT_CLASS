"""Wait for the existing CORR_MID debug gate; dry-review and submit tier3 once."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from hlt_classification import literature_ladder_followup as common
from hlt_classification.cms_proxy_ladder import correlated_tier3 as tier3
from hlt_classification.cms_proxy_ladder.contracts import artifact, write_json
from hlt_classification.scouting.hcwdl_authorization import validate_source_checkout

POLICY = dict(train=100000, validation=50000, branches=["DIRECT", "COARSE"],
              fits=6, reducers=3, jobs=11, partition="tier3", cpus=6, memory_mb=90000,
              gpu="gpu:a100:1", max_minutes=1440, final_test_accessed=False)
INSPECT = common.INSPECT.replace("import literature as l, submission as s",
                                 "import correlated as l, submission as s")
DATASET_HASH = "ef015d3c86f8188784169f67723a0f9772af96c4dab0a06387b208d9d33b4610"
CONTROLLER_FILES = (
    "src/hlt_classification/correlated_ladder_followup.py",
    "src/hlt_classification/literature_ladder_followup.py",
    "src/hlt_classification/cms_proxy_ladder/correlated_tier3.py",
    "scripts/jetclass2_correlated_tier3.py",
    "scripts/start_jetclass2_correlated_tier3_followup.sh",
)


def controller_source(project, commit):
    project = Path(project).resolve(strict=True)
    validate_source_checkout(project, expected_commit=commit)
    if project != Path(__file__).resolve().parents[2]:
        raise ValueError("Controller must run from the exact executor checkout")
    return dict(commit=commit, files={p: hashlib.sha256((project / p).read_bytes()).hexdigest()
                                    for p in CONTROLLER_FILES})


def bind(spec_path, gate_hash, preflight_job):
    if not re.fullmatch(r"[0-9a-f]{64}", gate_hash) or not re.fullmatch(r"[1-9][0-9]*", preflight_job):
        raise ValueError("Exact gate hash and preflight job required")
    path = Path(spec_path).resolve(strict=True)
    gate = common.read(path)
    if (gate["content_hash"] != gate_hash or gate["source_commit"] != tier3.BASE_COMMIT
            or gate["contract"] != "JETCLASS2_CMS_PROXY_LADDER_GATE_SPEC/v9"
            or path != Path(gate["gate_root"]).resolve() / "gate_spec.json" or path.parent.name != "gate"):
        raise ValueError("Original CORR_MID gate/source identity differs")
    config = common.capture(["scontrol", "show", "config"])
    if not re.search(r"(?m)^\s*ClusterName\s*=\s*sporc\s*$", config):
        raise PermissionError("Arm this follow-up on SPORC")
    print("Authenticating original pinned gate, exact ledger and journal...", flush=True)
    evidence = json.loads(common.capture([sys.executable, "-s", "-c", INSPECT,
                                         gate["project_dir"], str(path)], timeout=3600))
    if evidence["jobs"].get("preflight") != preflight_job:
        raise ValueError("Preflight ID differs from the authenticated gate ledger")
    manifest = common.read(Path(gate["request"]["study_root"]) / "dataset_manifest.json")
    if manifest["content_hash"] != DATASET_HASH:
        raise ValueError("Frozen CORR_MID dataset manifest differs")
    return gate, evidence


def states(text, jobs, user):
    expected, result = {job: task for task, job in jobs.items()}, {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 6 or fields[0] not in expected or fields[0] in result:
            raise ValueError("Unexpected/duplicate accounting row")
        job, name, owner, account, state, code = fields
        if name != "jc2crg_" + expected[job] or owner != user or account != "reu-aisocial":
            raise PermissionError("Gate scheduler identity differs: " + job)
        state = state.split()[0].rstrip("+")
        if state not in common.ACTIVE and not (state == "COMPLETED" and code == "0:0"):
            raise RuntimeError(f"Gate job {job}: {state}, exit={code}; no science submitted")
        result[job] = state
    return result


def wait_gate(jobs, *, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + 14 * 86400
    while monotonic() < deadline:
        raw = common.capture(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()),
                              "-o", "JobIDRaw,JobName%100,User%100,Account%100,State%40,ExitCode"])
        current = states(raw, jobs, os.environ["USER"])
        print("Gate accounting: " + json.dumps(current, sort_keys=True), flush=True)
        if set(current) == set(jobs.values()) and all(s == "COMPLETED" for s in current.values()):
            return
        print("Waiting for all three gate jobs; no science submitted yet.", flush=True)
        sleep(60)
    raise TimeoutError("14-day waiting bound reached; existing jobs unchanged")


def evidence(path, **fields):
    value = artifact("CORRELATED_TIER3_FOLLOWUP", **fields)
    path = Path(path)
    if path.exists():
        if common.read(path) != value:
            raise ValueError("Immutable follow-up evidence differs: " + str(path))
    else:
        write_json(path, value)
    return value


def no_debug_science(root):
    # Do not accidentally start a duplicate comparison in another partition.
    for name in ("submission_ledger.json", "live_submission_claim.json", "submission_ledger_journal"):
        if (Path(root) / "science" / name).exists():
            raise PermissionError("Debug science has a live/ambiguous submission; preserve and inspect")


def original_campaign(gate, root):
    spec = root / "science/campaign_spec.json"
    if not spec.exists():
        command = [sys.executable, "-u", "-s", str(Path(gate["project_dir"]) /
                   "scripts/jetclass2_correlated_tracking_ladder.py"), "create-campaign",
                   "--gate-root", gate["gate_root"], "--campaign-root", str(spec.parent)]
        print("Validating the passed gate and freezing the unchanged original science...", flush=True)
        subprocess.run(command, check=True, env=common.environment())
    # Validation is repeated by the transfer creator/worker, not just existence.
    return spec


def review(plan, spec):
    if (plan["parents"] != {"subject": spec["content_hash"]} or plan["mode"] != "science"
            or len(plan["commands"]) != POLICY["jobs"] or spec["fresh_fit_count"] != 6
            or spec["reducer_count"] != 3 or spec["selected_branches"] != POLICY["branches"]
            or spec["final_test_accessed"] is not False):
        raise ValueError("Tier3 plan exceeds the authorized comparison")
    profile = spec["runtime_profile"]
    if not (60 <= profile["train_minutes"] <= 1440 and 30 <= profile["reduce_minutes"] <= 1440):
        raise ValueError("Measured runtime exceeds authorized bounds")
    for row in plan["commands"]:
        args = row["command"]
        if (args[0] != "sbatch" or [a for a in args if a.startswith("--partition=")] != ["--partition=tier3"]
                or "--account=reu-aisocial" not in args or "--qos=qos_tier3" not in args):
            raise PermissionError("Not the authorized tier3 account/partition")
        if row["task_id"].startswith(("train_", "reduce_")):
            if any(flag not in args for flag in ("--gres=gpu:a100:1", "--cpus-per-task=6", "--mem=90000M")):
                raise ValueError("Tier3 GPU resource shape differs")
    print("FULL TIER3 DRY PLAN:\n" + json.dumps(plan, indent=2, sort_keys=True), flush=True)
    return plan["content_hash"]


def run(spec_path, *, gate_hash, preflight_job, project_dir, executor_commit, execute=False):
    controller = controller_source(project_dir, executor_commit)
    args = (spec_path, gate_hash, preflight_job)
    gate, binding = bind(*args)
    tier3.source_transfer(gate, project_dir, executor_commit)
    root = Path(gate["gate_root"]).parent
    no_debug_science(root)
    print("Bound policy: " + json.dumps(POLICY, sort_keys=True), flush=True)
    if not execute:
        print("Read-only review; --execute authorizes automatic exact-plan tier3 submission.", flush=True)
        return
    with common.lock(root / "followup_tier3"):
        authorization = evidence(root / "followup_tier3/authorization.json", phase="authorization",
            parents={"gate": gate_hash, "gate_ledger": binding["ledger_hash"]},
            controller=controller, gate_spec=str(Path(spec_path).resolve()),
            executor_project=str(Path(project_dir).resolve()), jobs=binding["jobs"], policy=POLICY)
        print("ARMED: after the existing debug gate succeeds, all 11 science jobs will submit on tier3.", flush=True)
        wait_gate(binding["jobs"])
        if bind(*args) != (gate, binding) or controller_source(project_dir, executor_commit) != controller:
            raise ValueError("Source/gate changed while waiting")
        no_debug_science(root)
        measured = original_campaign(gate, root)
        path = root / "science_tier3/campaign_spec.json"
        if not path.exists():
            tier3.create_campaign(measurement_spec=measured, project_dir=project_dir,
                                  source_commit=executor_commit, campaign_root=path.parent)
        spec = common.read(path)
        if (Path(spec["campaign_root"]).resolve() != path.parent.resolve()
                or spec["gate_root"] != gate["gate_root"] or spec["source_commit"] != executor_commit
                or Path(spec["project_dir"]).resolve() != Path(project_dir).resolve()
                or Path(spec["measurement_campaign"]["path"]).resolve() != measured.resolve()):
            raise ValueError("Existing tier3 campaign identity differs")
        plan = tier3.submit(spec)  # Authentic full dry run, after measured profile exists.
        plan_hash = review(plan, spec)
        evidence(root / "followup_tier3/review.json", phase="review",
            parents={"authorization": authorization["content_hash"], "plan": plan_hash}, policy=POLICY)
        no_debug_science(root)
        ledger = tier3.submit(spec, execute=True, plan_hash=plan_hash, authorization_phrase=tier3.AUTHORIZATION)
        evidence(root / "followup_tier3/complete.json", phase="submitted",
            parents={"authorization": authorization["content_hash"], "plan": plan_hash,
                     "ledger": ledger["content_hash"]}, jobs=ledger["jobs"], training_complete=False)
        print("ALL 11 TIER3 SCIENCE JOBS QUEUED: " + json.dumps(ledger["jobs"], sort_keys=True), flush=True)
        print("Controller finished. No existing jobs were cancelled or moved.", flush=True)
