"""Exact pending-C retirement and fresh debug execution; no scientific changes."""
import math
from pathlib import Path

from . import c_topology as original
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES

KIND = "DEV_C_TOPOLOGY_DEBUG"
CONTRACT = "CMS2JC2_RESPONSE_DEV_C_TOPOLOGY_DEBUG/v1"
NAME = "observed_topology_debug_r1"
RETIRE_PHRASE = "AUTHORIZE CMS2JC2 C TOPOLOGY DEBUG REPLACEMENT"
TARGETS = tuple([f"ct_eval_{i}" for i in range(4)] + ["ct_report"])


def tasks():
    from .dev_campaign import task
    return [*[task(f"ct_eval_{i}", "ct_evaluate", 36, 128, 8, shard=i)
              for i in range(4)], task("ct_report", "ct_report", 1, 32, 4, TARGETS[:4])]


def subject_metadata(subject):
    from . import dev_campaign as c, dev_submission as s
    if subject.get("contract") != "CMS2JC2_RESPONSE_DEV_C_TOPOLOGY/v1":
        raise ValueError("Debug replacement requires the original tier3 C topology stage")
    study = c.validate_stage(subject, source=False)
    for task in TARGETS:
        if (c.stage_dir(subject)/"receipts"/(task+".json")).exists():
            c.verified_outputs(subject, task)
            raise PermissionError("Preserve completed C output; replacement needs a revised plan: "+task)
    accepted = c.product(subject, "ct_acceptance", "result")
    parents = subject["reuse"]["parents"]
    validate(accepted, "DEV_C_TOPOLOGY_ACCEPTANCE", parents={"stage": subject["content_hash"],
        **{k: parents[k] for k in ("response", "samples", "ranges")}})
    if (accepted["jets"] != min(32, COUNTS["evaluation"]//4)
            or accepted["serial_process_parity"] is not True
            or accepted["exact_free_replay"] is not True
            or not all(type(accepted[k]) is int for k in ("jets", "resolved", "comparable"))
            or not 0 <= accepted["comparable"] <= accepted["resolved"] <= accepted["jets"]
            or accepted["scientific_quality_gate"] is not False):
        raise ValueError("Original C acceptance differs")
    # The unchanged codec worker already checked relative/absolute closure for
    # each particle. Do not invent an absolute cutoff on its aggregate maxima,
    # or turn a low resolved/comparable count into a quality gate.
    closure = accepted["closure"]
    if (set(closure) != {"p4_maximum", "tracking_maximum", "source_energy_projection_maximum"}
            or any(not math.isfinite(v) or v < 0 for v in closure.values())):
        raise ValueError("Original C codec closure evidence differs")
    plan = c.command_plan(subject, study)
    if load_json(c.stage_dir(subject)/"command_plan.json") != plan:
        raise ValueError("Original C command plan differs")
    ledger = load_json(c.stage_dir(subject)/"submission_ledger.json")
    validate(ledger, "DEV_LEDGER", parents={"stage": subject["content_hash"], "plan": plan["content_hash"]})
    jobs = s.submitted_jobs(subject, plan)
    if (ledger["dry_run"] is not False or ledger["jobs"] != jobs
            or set(jobs) != {"ct_acceptance", *TARGETS}):
        raise ValueError("Original C live ledger/journals are incomplete or differ")
    return study, accepted, ledger


def reuse(subject, study):
    from . import dev_campaign as c
    donor, accepted, ledger = subject_metadata(subject)
    for field in ("imported", "review", "numerical_environment"):
        if donor[field] != study[field]:
            raise ValueError("C debug reuse changed "+field)
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("C debug reuse changed scientific source: "+name)
    old_root, root = Path(donor["root"]).resolve(), Path(study["root"]).resolve()
    if old_root == root or old_root.is_relative_to(root) or root.is_relative_to(old_root):
        raise PermissionError("C debug root must be disjoint from original C")
    comparison = load_json(checked_file(subject["parent_spec"]))
    frozen = original.reuse(comparison, study)
    migration = artifact("DEV_C_TOPOLOGY_DEBUG_REUSE", parents={
        "subject": subject["content_hash"], "acceptance": accepted["content_hash"],
        "acceptance_receipt": c.verified_outputs(subject, "ct_acceptance")["content_hash"],
        "subject_ledger": ledger["content_hash"], "frozen_reuse": frozen["content_hash"],
        "study": study["content_hash"]}, acceptance_reused=True, acceptance_rerun=False,
        scientific_source_unchanged=True, subject_jobs=ledger["jobs"],
        old_outputs_modified=False, evaluation_reused=False)
    return frozen, migration


def create(*, parent_spec, project_dir, source_commit, root):
    from . import dev_campaign as c
    subject_ref = file_ref(parent_spec)
    subject = load_json(checked_file(subject_ref))
    donor, _, _ = subject_metadata(subject)
    root = Path(root).resolve()
    for old in (Path(donor["root"]).resolve(), Path(load_json(checked_file(subject["parent_spec"]))["root"]).resolve()):
        if root == old or root.is_relative_to(old) or old.is_relative_to(root):
            raise PermissionError("Use a fresh disjoint C debug root")
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in root.parents):
        raise PermissionError("Do not nest C debug inside another study")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="debug")
    frozen, migration = reuse(subject, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "reuse": frozen["content_hash"],
        "migration": migration["content_hash"]}, study=file_ref(root/"study_spec.json"),
        root=study["root"], name=NAME, stage="ctopo", subject_spec=subject_ref,
        parent_spec=subject["parent_spec"], policy=subject["policy"], b_threads=1,
        reuse=frozen, migration=migration, views=list(original.VIEWS), replicas=[0, 1, 2],
        tasks=tasks(), scientific_qualification=False, resources_are_development_envelopes=True,
        observed_state_privileged=True, evaluation_association_recomputed=True, fitted_model_modified=False)
    c.write(root, f"stages/{NAME}/stage_spec.json", spec, KIND)
    c.write(root, f"stages/{NAME}/command_plan.json", c.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    from . import dev_campaign as c
    study = load_json(checked_file(spec["study"]))
    c.validate_study(study, source=source)
    subject = load_json(checked_file(spec["subject_spec"]))
    frozen, migration = reuse(subject, study)
    validate(spec, KIND, parents={"study": study["content_hash"], "reuse": frozen["content_hash"],
                                 "migration": migration["content_hash"]})
    if (study["site"]["partition"] != "debug" or spec["root"] != study["root"]
            or spec["name"] != NAME or spec["stage"] != "ctopo" or spec["tasks"] != tasks()
            or spec["parent_spec"] != subject["parent_spec"] or spec["policy"] != subject["policy"]
            or spec["b_threads"] != 1 or spec["views"] != list(original.VIEWS)
            or spec["replicas"] != [0, 1, 2] or spec["reuse"] != frozen or spec["migration"] != migration
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True
            or spec["observed_state_privileged"] is not True or spec["evaluation_association_recomputed"] is not True
            or spec["fitted_model_modified"] is not False):
        raise ValueError("C debug registration differs")
    return study


def old_states(spec):
    from . import dev_submission as s
    jobs = spec["migration"]["subject_jobs"]
    status = s.states(jobs)
    if status.get(jobs["ct_acceptance"]) != ("COMPLETED", "0:0"):
        raise PermissionError("Original C acceptance must be COMPLETED/0:0 in accounting")
    rows = {task: dict(job=jobs[task], state=status.get(jobs[task], ("UNKNOWN", None))[0],
                       exit_code=status.get(jobs[task], (None, None))[1]) for task in TARGETS}
    for task, row in rows.items():
        if row["state"] not in s.TERMINAL | {"PENDING"} or row["state"] == "COMPLETED":
            raise PermissionError(f"Preserve original C job {row['job']} ({task}): {row['state']}; stop migration")
    return rows


def retirement_path(spec):
    from .dev_campaign import stage_dir
    return stage_dir(spec)/"retirement.json"


def verify_retirement(spec, *, live=False):
    from . import dev_campaign as c, dev_submission as s
    value = load_json(retirement_path(spec))
    plan = load_json(c.stage_dir(spec)/"command_plan.json")
    validate(value, "DEV_C_TOPOLOGY_DEBUG_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]})
    rows = value["tasks"]
    if (set(rows) != set(TARGETS) or value["acceptance_state"] != "COMPLETED/0:0"
            or value["cancellation_policy"] != "exact_pending_ids_only"
            or any(row["job"] != spec["migration"]["subject_jobs"][task]
                   or row["state"] not in s.TERMINAL - {"COMPLETED"} for task, row in rows.items())):
        raise ValueError("C debug retirement evidence differs")
    if live and any(r["state"] not in s.TERMINAL - {"COMPLETED"} for r in old_states(spec).values()):
        raise PermissionError("Old C jobs can still run; do not submit debug replacements")
    return value


def retire(spec, *, execute=False, authorization_phrase=None, reviewed_plan_hash=None):
    from . import dev_campaign as c, dev_submission as s
    study = validate_stage(spec)
    plan = s.submit(spec)  # Complete canonical dry validation; never submits.
    rows = old_states(spec)
    if not execute:
        return dict(read_only=True, tasks=rows, plan_sha256=plan["content_hash"],
                    required_phrase=RETIRE_PHRASE, final_test_accessed=False)
    if authorization_phrase != RETIRE_PHRASE or reviewed_plan_hash != plan["content_hash"]:
        raise PermissionError("Explicit C debug retirement phrase and reviewed plan hash required")
    if retirement_path(spec).exists():
        return verify_retirement(spec, live=True)
    s.site_checks(spec, study, plan)  # Do not cancel before debug admission succeeds.
    subject = load_json(checked_file(spec["subject_spec"]))
    old_study, _, _ = subject_metadata(subject)
    pending = []
    for task, row in rows.items():
        if row["state"] == "PENDING":
            fields = s.scheduler_identity(subject, old_study, task, row["job"], pending=True)
            if fields.get("JobState") != "PENDING" or fields.get("JobName") != "c2jd_"+task:
                raise PermissionError("Original C job changed state/identity; nothing cancelled")
            pending.append(row["job"])
    if pending:
        result = s.scheduler(["scancel", "--ctld", "--state=PENDING", "--partition=tier3",
                              "--account=reu-aisocial", *pending])
        if result.returncode:
            raise RuntimeError("C cancellation returned an error; inspect exact states before retry: "+result.stderr)
    rows = old_states(spec)
    if any(r["state"] not in s.TERMINAL - {"COMPLETED"} for r in rows.values()):
        raise RuntimeError("C cancellation not yet terminal in accounting; wait, inspect, then explicitly retry retirement")
    # Authenticate receipts again after the pending-to-running/completed race window.
    subject_metadata(subject)
    value = artifact("DEV_C_TOPOLOGY_DEBUG_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]},
        tasks=rows, acceptance_state="COMPLETED/0:0", cancellation_policy="exact_pending_ids_only")
    c.write(spec["root"], f"stages/{NAME}/retirement.json", value, "DEV_C_TOPOLOGY_DEBUG_RETIREMENT")
    return value
