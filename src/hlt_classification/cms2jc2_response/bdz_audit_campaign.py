"""Isolated frozen-development diagnostic; no calibration or confirmation API."""
from pathlib import Path

from . import dev_campaign as dev, bdz_campaign as bdz, bdz_metrics as scores, bdz_maps as maps
from . import bdz_audit_metrics as metrics
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, file_ref, checked_file
from .dev_restart import EXECUTION_ONLY_FILES

KIND = "BDZ_AUDIT_STAGE"
CONTRACT = "CMS2JC2_RESPONSE_BDZ_AUDIT_STAGE/v1"
NAME = "bdz_audit_r1"


def protocol():
    return artifact("BDZ_AUDIT_PROTOCOL", jets=COUNTS["evaluation"], shards=4, replicas=[0, 1, 2],
        candidates=list(maps.CANDIDATES), fields=list(metrics.NAMES), pids=["all", *map(str, range(6))],
        signed_edges=metrics.SIGNED_EDGES, error_edges=metrics.ERROR_EDGES,
        magnitude_log10_edges=metrics.MAG_EDGES, error_log10_edges=metrics.ERR_EDGES,
        thresholds=[list(t) for t in scores.THRESHOLDS], top_per_cell=metrics.TOP,
        validity_order=["neither", "error_only", "value_only", "both"],
        selection=False, confirmation_accessed=False, transfer_authorized=False)


def tasks():
    names = [f"ba_eval_{i}" for i in range(4)]
    return [dev.task("ba_acceptance", "ba_acceptance", 2, 32, 2),
            *[dev.task(n, "ba_evaluate", 36, 128, 8, ["ba_acceptance"], shard=i) for i, n in enumerate(names)],
            dev.task("ba_report", "ba_report", 1, 32, 4, names)]


def completed(parent):
    kind = parent.get("contract")
    if kind not in ("CMS2JC2_RESPONSE_BDZ_STAGE/v1", "CMS2JC2_RESPONSE_BDZ_TIER3/v1") or parent.get("stage") != "bdz_compare":
        raise PermissionError("Audit requires a completed B_DZ development comparison")
    study = dev.validate_stage(parent, source=False)
    if kind == "CMS2JC2_RESPONSE_BDZ_TIER3/v1":
        from .bdz_tier3 import verify_retirement
        verify_retirement(parent)
    reg = bdz.registry(parent)
    row = dev.product(parent, "bz_select", "result")
    validate(row, "BDZ_SELECTION", parents={"stage": parent["content_hash"],
        "registry": reg["content_hash"], "protocol": bdz.protocol()["content_hash"]})
    choice, guards = scores.choose(row["scores"])
    if (row["selected"] != choice or row["guard_failures"] != guards
            or row["map_hash"] != reg["mapping"]["content_hash"]
            or row["selected_strength"] != maps.STRENGTHS[choice] or row["jets"] != COUNTS["evaluation"]
            or any(row[k] is not False for k in ("confirmation_accessed", "production_qualified", "transfer_authorized"))
            or row["shard_hashes"] != [dev.product(parent, f"bz_eval_{i}", "result")["content_hash"] for i in range(4)]):
        raise ValueError("Completed B_DZ selection differs")
    return study, reg, row


def reuse(parent, study):
    donor, reg, row = completed(parent)
    for name in ("imported", "review", "numerical_environment"):
        if study[name] != donor[name]:
            raise ValueError("Audit changed donor interface: "+name)
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("Audit changed scientific source: "+name)
    disjoint(study["root"], donor["root"])
    return artifact("BDZ_AUDIT_REUSE", parents={"study": study["content_hash"], "parent": parent["content_hash"],
        "selection": row["content_hash"], "registry": reg["content_hash"],
        "selection_receipt": dev.verified_outputs(parent, "bz_select")["content_hash"]},
        historical_choice=row["selected"], old_artifacts_modified=False,
        calibration_reused=True, confirmation_accessed=False)


def disjoint(root, donor):
    a, b = Path(root).resolve(), Path(donor).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise PermissionError("Use a disjoint new diagnostic root")


def create(*, parent_spec, project_dir, source_commit, root):
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    donor, _, _ = completed(parent)
    disjoint(root, donor["root"])
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in Path(root).resolve().parents):
        raise PermissionError("Do not nest an audit inside an existing campaign")
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="tier3")
    evidence = reuse(parent, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"],
        "reuse": evidence["content_hash"]}, study=file_ref(Path(study["root"])/"study_spec.json"),
        root=study["root"], name=NAME, stage="bdz_audit", parent_spec=ref, policy=parent["policy"],
        protocol=protocol(), reuse=evidence, b_threads=1, tasks=tasks(), scientific_qualification=False,
        resources_are_development_envelopes=True)
    validate_stage(spec)
    dev.write(spec["root"], f"stages/{NAME}/stage_spec.json", spec, KIND)
    dev.write(spec["root"], f"stages/{NAME}/command_plan.json", dev.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec["study"]))
    dev.validate_study(study, source=source)
    parent = load_json(checked_file(spec["parent_spec"]))
    evidence = reuse(parent, study)
    validate(spec, KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"],
                                 "reuse": evidence["content_hash"]})
    if (spec["root"] != study["root"] or spec["name"] != NAME or spec["stage"] != "bdz_audit"
            or study["site"]["partition"] != "tier3" or spec["protocol"] != protocol()
            or spec["reuse"] != evidence or spec["tasks"] != tasks() or spec["policy"] != parent["policy"]
            or spec["b_threads"] != 1 or spec["scientific_qualification"] is not False
            or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Frozen audit registration differs")
    return study


def read(spec):
    validate_stage(spec, source=False)
    row = dev.product(spec, "ba_report", "result")
    validate(row, "BDZ_AUDIT_REPORT", parents={"stage": spec["content_hash"], "protocol": protocol()["content_hash"],
        "selection": spec["reuse"]["parents"]["selection"],
        "acceptance": dev.product(spec, "ba_acceptance", "result")["content_hash"]})
    if (row["historical_choice"] != spec["reuse"]["historical_choice"] or row["jets"] != COUNTS["evaluation"]
            or row["shard_hashes"] != [dev.product(spec, f"ba_eval_{i}", "result")["content_hash"] for i in range(4)]
            or any(row[k] is not False for k in ("confirmation_accessed", "production_qualified", "transfer_authorized", "automatic_followup"))):
        raise ValueError("Frozen audit report differs")
    return row
