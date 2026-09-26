"""Exact pending-only debug replacement of a frozen significance audit."""
import math
from pathlib import Path

from . import bdz_audit_campaign as original, bdz_audit_worker as worker
from . import dev_campaign as dev, dev_submission as submission
from .contracts import artifact, load_json, validate
from .dev_data import checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES
from .storage import GIB

KIND = "BDZ_AUDIT_DEBUG"
CONTRACT = "CMS2JC2_RESPONSE_BDZ_AUDIT_DEBUG/v1"
NAME = "bdz_audit_debug_r1"
RETIRE_PHRASE = "AUTHORIZE CMS2JC2 BDZ AUDIT DEBUG REPLACEMENT"
TARGETS = tuple([f"ba_eval_{i}" for i in range(4)] + ["ba_report"])


def tasks():
    return [*[dev.task(f"ba_eval_{i}", "ba_evaluate", 36, 128, 8, shard=i) for i in range(4)],
            dev.task("ba_report", "ba_report", 1, 32, 4, TARGETS[:4])]


def acceptance(subject):
    reg = worker.inputs(subject)[3]
    row = worker.accepted(subject, reg)
    m = row["measurement"]
    values = (m["wall_seconds"], m["sampled_peak_tree_rss_bytes"],
              row["projected_seconds"], row["projected_peak_bytes"])
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in values):
        raise ValueError("Original audit acceptance has invalid measurements")
    seconds = 2*m["wall_seconds"]*(original.COUNTS["evaluation"]//4)/row["jets"]
    ram = 1.5*m["sampled_peak_tree_rss_bytes"]*18 + 2*GIB
    if (row["projected_seconds"] != seconds or row["projected_peak_bytes"] != ram
            or seconds > 8*3600 or ram > 128*GIB):
        raise PermissionError("Original audit acceptance resource projections differ")
    return row


def subject_metadata(subject):
    if subject.get("contract") != original.CONTRACT:
        raise ValueError("Debug replacement requires the original tier3 significance audit")
    study = original.validate_stage(subject, source=False)
    for task in TARGETS:
        if (dev.stage_dir(subject)/"claims"/task).exists():
            raise PermissionError("Preserve original audit execution claim: "+task)
        if (dev.stage_dir(subject)/"receipts"/(task+".json")).exists():
            dev.verified_outputs(subject, task)
            raise PermissionError("Preserve completed audit output: "+task)
    accepted = acceptance(subject)
    plan = dev.command_plan(subject, study)
    if load_json(dev.stage_dir(subject)/"command_plan.json") != plan:
        raise ValueError("Original audit command plan differs")
    ledger = load_json(dev.stage_dir(subject)/"submission_ledger.json")
    validate(ledger, "DEV_LEDGER", parents={"stage": subject["content_hash"], "plan": plan["content_hash"]})
    jobs = submission.submitted_jobs(subject, plan)
    if (ledger["dry_run"] is not False or ledger["jobs"] != jobs
            or set(jobs) != {"ba_acceptance", *TARGETS}):
        raise ValueError("Original audit live ledger/journals are incomplete or differ")
    return study, accepted, ledger


def reuse(subject, study):
    donor, accepted, ledger = subject_metadata(subject)
    for field in ("imported", "review", "numerical_environment"):
        if donor[field] != study[field]:
            raise ValueError("Audit debug reuse changed "+field)
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("Audit debug reuse changed scientific source: "+name)
    original.disjoint(study["root"], donor["root"])
    parent = load_json(checked_file(subject["parent_spec"]))
    frozen = original.reuse(parent, study)
    migration = artifact("BDZ_AUDIT_DEBUG_REUSE", parents={
        "study": study["content_hash"], "subject": subject["content_hash"],
        "acceptance": accepted["content_hash"],
        "acceptance_receipt": dev.verified_outputs(subject, "ba_acceptance")["content_hash"],
        "subject_ledger": ledger["content_hash"], "frozen_reuse": frozen["content_hash"]},
        subject_jobs=ledger["jobs"], acceptance_reused=True, acceptance_rerun=False,
        scientific_source_unchanged=True, old_outputs_modified=False, evaluation_reused=False)
    return frozen, migration


def create(*, parent_spec, project_dir, source_commit, root):
    ref = file_ref(parent_spec)
    subject = load_json(checked_file(ref))
    donor, _, _ = subject_metadata(subject)
    root = Path(root).resolve()
    for old in (donor["root"], load_json(checked_file(subject["parent_spec"]))["root"]):
        original.disjoint(root, old)
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in root.parents):
        raise PermissionError("Do not nest the debug audit in an existing study")
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="debug")
    frozen, migration = reuse(subject, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "protocol": original.protocol()["content_hash"],
        "reuse": frozen["content_hash"], "migration": migration["content_hash"]},
        study=file_ref(root/"study_spec.json"), root=study["root"], name=NAME, stage="bdz_audit",
        subject_spec=ref, parent_spec=subject["parent_spec"], policy=subject["policy"],
        protocol=original.protocol(), reuse=frozen, migration=migration, b_threads=1,
        tasks=tasks(), scientific_qualification=False, resources_are_development_envelopes=True)
    dev.write(root, f"stages/{NAME}/stage_spec.json", spec, KIND)
    dev.write(root, f"stages/{NAME}/command_plan.json", dev.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec["study"]))
    dev.validate_study(study, source=source)
    subject = load_json(checked_file(spec["subject_spec"]))
    frozen, migration = reuse(subject, study)
    validate(spec, KIND, parents={"study": study["content_hash"], "protocol": original.protocol()["content_hash"],
        "reuse": frozen["content_hash"], "migration": migration["content_hash"]})
    if (spec["root"] != study["root"] or spec["name"] != NAME or spec["stage"] != "bdz_audit"
            or study["site"]["partition"] != "debug" or spec["parent_spec"] != subject["parent_spec"]
            or spec["protocol"] != original.protocol() or spec["policy"] != subject["policy"]
            or spec["reuse"] != frozen or spec["migration"] != migration or spec["tasks"] != tasks()
            or spec["b_threads"] != 1 or spec["scientific_qualification"] is not False
            or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Audit debug registration differs")
    return study


def old_states(spec):
    jobs = spec["migration"]["subject_jobs"]
    status = submission.states(jobs)
    if status.get(jobs["ba_acceptance"]) != ("COMPLETED", "0:0"):
        raise PermissionError("Original audit acceptance must be COMPLETED/0:0 in accounting")
    rows = {t: dict(job=jobs[t], state=status.get(jobs[t], ("UNKNOWN", None))[0],
                    exit_code=status.get(jobs[t], (None, None))[1]) for t in TARGETS}
    for task, row in rows.items():
        if row["state"] not in submission.TERMINAL | {"PENDING"} or row["state"] == "COMPLETED":
            raise PermissionError(f"Preserve original audit job {row['job']} ({task}): {row['state']}; stop migration")
    return rows


def retirement_path(spec):
    return dev.stage_dir(spec)/"retirement.json"


def verify_retirement(spec, *, live=False):
    value = load_json(retirement_path(spec))
    plan = load_json(dev.stage_dir(spec)/"command_plan.json")
    validate(value, "BDZ_AUDIT_DEBUG_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]})
    rows = value["tasks"]
    if (set(rows) != set(TARGETS) or value["acceptance_state"] != "COMPLETED/0:0"
            or value["cancellation_policy"] != "exact_pending_ids_only"
            or any(r["job"] != spec["migration"]["subject_jobs"][t]
                   or r["state"] not in submission.TERMINAL - {"COMPLETED"} for t, r in rows.items())):
        raise ValueError("Audit debug retirement evidence differs")
    if live and any(r["state"] not in submission.TERMINAL - {"COMPLETED"} for r in old_states(spec).values()):
        raise PermissionError("Old audit jobs can still run; do not submit debug replacements")
    return value


def retire(spec, *, execute=False, authorization_phrase=None, reviewed_plan_hash=None):
    study = validate_stage(spec)
    plan = submission.submit(spec)  # Dry validation, never submission.
    rows = old_states(spec)
    if not execute:
        return dict(read_only=True, tasks=rows, plan_sha256=plan["content_hash"],
                    required_phrase=RETIRE_PHRASE, final_test_accessed=False)
    if authorization_phrase != RETIRE_PHRASE or reviewed_plan_hash != plan["content_hash"]:
        raise PermissionError("Explicit audit debug retirement phrase and reviewed plan hash required")
    if retirement_path(spec).exists():
        return verify_retirement(spec, live=True)
    submission.site_checks(spec, study, plan)
    subject = load_json(checked_file(spec["subject_spec"]))
    old_study, _, _ = subject_metadata(subject)
    # Admission/authentication can be slow. Refresh accounting before cancellation.
    rows = old_states(spec)
    pending = []
    for task, row in rows.items():
        if row["state"] == "PENDING":
            fields = submission.scheduler_identity(subject, old_study, task, row["job"], pending=True)
            if fields.get("JobState") != "PENDING" or fields.get("JobName") != "c2jd_"+task:
                raise PermissionError("Original audit job changed state/identity; nothing cancelled")
            pending.append(row["job"])
    if pending:
        result = submission.scheduler(["scancel", "--ctld", "--state=PENDING", "--partition=tier3",
                                       "--account=reu-aisocial", *pending])
        if result.returncode:
            raise RuntimeError("Audit cancellation returned an error; inspect states before retry: "+result.stderr)
    rows = old_states(spec)
    if any(r["state"] not in submission.TERMINAL - {"COMPLETED"} for r in rows.values()):
        raise RuntimeError("Audit cancellation not yet terminal in accounting; inspect before explicitly retrying")
    subject_metadata(subject)
    value = artifact("BDZ_AUDIT_DEBUG_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]},
        tasks=rows, acceptance_state="COMPLETED/0:0", cancellation_policy="exact_pending_ids_only")
    dev.write(spec["root"], f"stages/{NAME}/retirement.json", value, "BDZ_AUDIT_DEBUG_RETIREMENT")
    return value


def read(spec):
    validate_stage(spec, source=False)
    verify_retirement(spec)
    row = dev.product(spec, "ba_report", "result")
    validate(row, "BDZ_AUDIT_REPORT", parents={"stage": spec["content_hash"],
        "protocol": original.protocol()["content_hash"], "selection": spec["reuse"]["parents"]["selection"],
        "acceptance": spec["migration"]["parents"]["acceptance"]})
    if (row["historical_choice"] != spec["reuse"]["historical_choice"] or row["jets"] != original.COUNTS["evaluation"]
            or row["shard_hashes"] != [dev.product(spec, f"ba_eval_{i}", "result")["content_hash"] for i in range(4)]
            or any(row[k] is not False for k in ("confirmation_accessed", "production_qualified", "transfer_authorized", "automatic_followup"))):
        raise ValueError("Debug audit report differs")
    return row
