"""Partition-only B_DZ replacement; retirement donor: c_topology_debug at 1cdf6b89."""
from pathlib import Path

from . import bdz_campaign as original
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES

KIND = "BDZ_TIER3"
CONTRACT = "CMS2JC2_RESPONSE_BDZ_TIER3/v1"
NAME = "bdz_compare_tier3_r1"
RETIRE_PHRASE = "AUTHORIZE CMS2JC2 BDZ TIER3 REPLACEMENT"
TARGETS = tuple([f"bz_eval_{i}" for i in range(4)] + ["bz_select"])


def subject_metadata(subject):
    from . import dev_campaign as c, dev_submission as s
    if subject.get("contract") != "CMS2JC2_RESPONSE_BDZ_STAGE/v1" or subject.get("stage") != "bdz_compare":
        raise ValueError("Tier3 replacement requires the original debug B_DZ comparison")
    study = original.validate_stage(subject, source=False)
    for task in TARGETS:
        for directory in ("receipts", "claims"):
            target = c.stage_dir(subject)/directory/(task+".json" if directory == "receipts" else task)
            if target.exists():
                raise PermissionError("Preserve original B_DZ work; stop migration: "+task)
    gate = original.gate(subject)
    accepted = c.product(gate, "bz_acceptance", "result")
    validate(accepted, "BDZ_ACCEPTANCE", parents={"stage": gate["content_hash"],
                                                 "protocol": original.protocol()["content_hash"]})
    if (any(accepted.get(k) is not True for k in (
            "serial_process_parity", "resource_envelope_ok", "nonidentity_maps_exercised"))
            or accepted["jets"] != min(32, COUNTS["residual"])
            or accepted["scientific_quality_gate"] is not False or accepted["confirmation_accessed"] is not False):
        raise ValueError("Original B_DZ acceptance differs")
    registry = original.registry(subject)
    plan = c.command_plan(subject, study)
    if load_json(c.stage_dir(subject)/"command_plan.json") != plan:
        raise ValueError("Original B_DZ plan differs")
    ledger = load_json(c.stage_dir(subject)/"submission_ledger.json")
    validate(ledger, "DEV_LEDGER", parents={"stage": subject["content_hash"], "plan": plan["content_hash"]})
    jobs = s.submitted_jobs(subject, plan)
    if ledger["dry_run"] is not False or ledger["jobs"] != jobs or set(jobs) != set(TARGETS):
        raise ValueError("Original B_DZ ledger/journals are incomplete or differ")
    return study, accepted, registry, ledger


def reuse(subject, study):
    from . import dev_campaign as c
    donor, accepted, registry, ledger = subject_metadata(subject)
    for field in ("imported", "review", "numerical_environment"):
        if donor[field] != study[field]:
            raise ValueError("B_DZ tier3 reuse changed "+field)
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("B_DZ tier3 reuse changed scientific source: "+name)
    a, b = Path(donor["root"]).resolve(), Path(study["root"]).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise PermissionError("B_DZ tier3 root must be disjoint")
    gate = original.gate(subject)
    return artifact("BDZ_TIER3_REUSE", parents={"study": study["content_hash"],
        "subject": subject["content_hash"], "subject_ledger": ledger["content_hash"],
        "acceptance": accepted["content_hash"], "registry": registry["content_hash"],
        "acceptance_receipt": c.verified_outputs(gate, "bz_acceptance")["content_hash"],
        "calibration_receipt": c.verified_outputs(gate, "bz_calibrate")["content_hash"]},
        subject_jobs=ledger["jobs"], acceptance_reused=True, calibration_reused=True,
        scientific_source_unchanged=True, old_outputs_modified=False, confirmation_accessed=False)


def create(*, parent_spec, project_dir, source_commit, root):
    from . import dev_campaign as c
    subject_ref = file_ref(parent_spec)
    subject = load_json(checked_file(subject_ref))
    donor, _, _, _ = subject_metadata(subject)
    root = Path(root).resolve()
    old = Path(donor["root"]).resolve()
    if root == old or root.is_relative_to(old) or old.is_relative_to(root):
        raise PermissionError("Use a fresh disjoint B_DZ tier3 root")
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in root.parents):
        raise PermissionError("Do not nest the B_DZ tier3 root inside another study")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="tier3")
    migration = reuse(subject, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "migration": migration["content_hash"],
                                  "protocol": original.protocol()["content_hash"]},
        study=file_ref(root/"study_spec.json"), root=study["root"], name=NAME, stage="bdz_compare",
        subject_spec=subject_ref, parent_spec=subject["parent_spec"], policy=subject["policy"],
        b_threads=1, protocol=original.protocol(), migration=migration,
        tasks=original.tasks("bdz_compare"), scientific_qualification=False,
        resources_are_development_envelopes=True)
    validate_stage(spec)
    c.write(root, f"stages/{NAME}/stage_spec.json", spec, KIND)
    c.write(root, f"stages/{NAME}/command_plan.json", c.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    from . import dev_campaign as c
    study = load_json(checked_file(spec["study"]))
    c.validate_study(study, source=source)
    subject = load_json(checked_file(spec["subject_spec"]))
    migration = reuse(subject, study)
    validate(spec, KIND, parents={"study": study["content_hash"], "migration": migration["content_hash"],
                                  "protocol": original.protocol()["content_hash"]})
    if (study["site"]["partition"] != "tier3" or spec["root"] != study["root"]
            or spec["name"] != NAME or spec["stage"] != "bdz_compare"
            or spec["tasks"] != original.tasks("bdz_compare") or spec["protocol"] != original.protocol()
            or spec["parent_spec"] != subject["parent_spec"] or spec["policy"] != subject["policy"]
            or spec["b_threads"] != 1 or spec["migration"] != migration
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("B_DZ tier3 registration differs")
    return study


def old_states(spec):
    from . import dev_submission as s
    jobs = spec["migration"]["subject_jobs"]
    status = s.states(jobs)
    rows = {t: dict(job=jobs[t], state=status.get(jobs[t], ("UNKNOWN", None))[0],
                    exit_code=status.get(jobs[t], (None, None))[1]) for t in TARGETS}
    for task, row in rows.items():
        if row["state"] not in (s.TERMINAL - {"COMPLETED"}) | {"PENDING"}:
            raise PermissionError(f"Preserve original B_DZ job {row['job']} ({task}): {row['state']}; stop migration")
    return rows


def retirement_path(spec):
    from .dev_campaign import stage_dir
    return stage_dir(spec)/"retirement.json"


def verify_retirement(spec, *, live=False):
    from . import dev_campaign as c, dev_submission as s
    value = load_json(retirement_path(spec))
    plan = load_json(c.stage_dir(spec)/"command_plan.json")
    validate(value, "BDZ_TIER3_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]})
    rows = value["tasks"]
    if (set(rows) != set(TARGETS) or value["cancellation_policy"] != "exact_pending_ids_only"
            or any(row["job"] != spec["migration"]["subject_jobs"][t]
                   or row["state"] not in s.TERMINAL - {"COMPLETED"} for t, row in rows.items())):
        raise ValueError("B_DZ tier3 retirement evidence differs")
    if live and any(r["state"] not in s.TERMINAL - {"COMPLETED"} for r in old_states(spec).values()):
        raise PermissionError("Old B_DZ jobs can still run; do not submit replacements")
    return value


def retire(spec, *, execute=False, authorization_phrase=None, reviewed_plan_hash=None):
    from . import dev_campaign as c, dev_submission as s
    study = validate_stage(spec)
    plan = s.submit(spec)  # Canonical dry validation; no scheduler mutation.
    rows = old_states(spec)
    if not execute:
        return dict(read_only=True, tasks=rows, plan_sha256=plan["content_hash"],
                    required_phrase=RETIRE_PHRASE, final_test_accessed=False)
    if authorization_phrase != RETIRE_PHRASE or reviewed_plan_hash != plan["content_hash"]:
        raise PermissionError("Explicit B_DZ tier3 retirement phrase and reviewed plan hash required")
    if retirement_path(spec).exists():
        return verify_retirement(spec, live=True)
    s.site_checks(spec, study, plan)  # Prove admission BEFORE any cancellation.
    subject = load_json(checked_file(spec["subject_spec"]))
    old_study, _, _, _ = subject_metadata(subject)
    pending = []
    for task, row in rows.items():
        if row["state"] == "PENDING":
            fields = s.scheduler_identity(subject, old_study, task, row["job"], pending=True)
            if fields.get("JobState") != "PENDING" or fields.get("JobName") != "c2jd_"+task:
                raise PermissionError("Original B_DZ state/identity changed; nothing cancelled")
            pending.append(row["job"])
    if pending:
        result = s.scheduler(["scancel", "--ctld", "--state=PENDING", "--partition=debug",
                              "--account=reu-aisocial", *pending])
        if result.returncode:
            raise RuntimeError("B_DZ cancellation error; inspect exact states before retry: "+result.stderr)
    rows = old_states(spec)
    if any(r["state"] not in s.TERMINAL - {"COMPLETED"} for r in rows.values()):
        raise RuntimeError("Cancellation not yet terminal in accounting; inspect then explicitly retry retirement")
    subject_metadata(subject)  # Catch a worker claim/output in the race window.
    value = artifact("BDZ_TIER3_RETIREMENT", parents={"stage": spec["content_hash"],
        "plan": plan["content_hash"], "subject_ledger": spec["migration"]["parents"]["subject_ledger"]},
        tasks=rows, cancellation_policy="exact_pending_ids_only")
    c.write(spec["root"], f"stages/{NAME}/retirement.json", value, "BDZ_TIER3_RETIREMENT")
    return value


def render(spec):
    """Authenticate the unchanged worker's selection without debug-only validation."""
    from . import dev_campaign as c, bdz_metrics as metrics, bdz_maps as maps
    validate_stage(spec, source=False)
    verify_retirement(spec)
    if not (c.stage_dir(spec)/"receipts/bz_select.json").is_file():
        return "No completed B_DZ tier3 report yet; use monitor."
    row, reg = c.product(spec, "bz_select", "result"), original.registry(spec)
    validate(row, "BDZ_SELECTION", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
                                            "protocol": original.protocol()["content_hash"]})
    winner, reasons = metrics.choose(row["scores"])
    if (row["selected"] != winner or row["guard_failures"] != reasons
            or row["selected_strength"] != maps.STRENGTHS[winner] or row["map_hash"] != reg["mapping"]["content_hash"]
            or row["jets"] != COUNTS["evaluation"] or row["confirmation_accessed"] is not False
            or row["production_qualified"] is not False
            or row["shard_hashes"] != [c.product(spec, t, "result")["content_hash"] for t in TARGETS[:4]]):
        raise ValueError("B_DZ tier3 selection differs")
    lines = ["candidate       score    core TV    tails     joint   guard failures"]
    for name in maps.CANDIDATES:
        score = row["scores"][name]
        lines.append(f"{name:<14} {score['score']:.5f} " + " ".join(
            f"{score['blocks'][k]:9.5f}" for k in ("core_tv", "tail_balanced", "joint"))
            + "   " + (", ".join(reasons[name]) or "none"))
    return "\n".join(lines + [f"Frozen development choice: {winner}",
        f"Jets: {row['jets']}; calibration reused unchanged; execution: tier3",
        f"Figures: {Path(spec['root'])/'figures'/spec['name']}",
        f"Full statistics: {Path(spec['root'])/'reports'/spec['name']/'bz_select.json'}",
        "Exploratory development only. Confirmation/test untouched. Not production qualified."])
