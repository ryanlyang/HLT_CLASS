"""Frozen B reuse and isolated validity-aware tracking audit registration."""
from pathlib import Path

from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref, validate_samples
from .b_tracking_generation import VARIANTS

NAME = "frozen_b_tracking_r1"
KIND = "DEV_B_TRACKING"


def tasks():
    from .dev_campaign import task
    return [task("bt_acceptance", "bt_acceptance", 2, 32, 2), *[
        task(f"bt_eval_{i}", "bt_evaluate", 36, 128, 24, ["bt_acceptance"], shard=i)
        for i in range(4)], task("bt_report", "bt_report", 1, 32, 4, [f"bt_eval_{i}" for i in range(4)])]


def reuse(parent, study):
    from . import dev_campaign as c
    from .dev_restart import EXECUTION_ONLY_FILES
    if parent.get("stage") != "compare":
        raise ValueError("Frozen B tracking requires a completed B_L comparison")
    donor = c.validate_stage(parent, source=False)
    if (study["imported"] != donor["imported"] or study["review"] != donor["review"]
            or study["numerical_environment"] != donor["numerical_environment"]):
        raise ValueError("Frozen B tracking inputs, conventions or numerical environment changed")
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("Frozen B tracking scientific source changed: "+name)
    old_root, new_root = Path(donor["root"]).resolve(), Path(study["root"]).resolve()
    if old_root == new_root or old_root.is_relative_to(new_root) or new_root.is_relative_to(old_root):
        raise PermissionError("Frozen B tracking destination overlaps donor")
    pilot = c.preparation_stage(parent)
    samples, ranges = (c.product(pilot, "prepare", key) for key in ("samples", "ranges"))
    inventory, roles = (load_json(checked_file(study["imported"]["files"][key]))
                        for key in ("cms_inventory.json", "response_roles.json"))
    validate_samples(samples, inventory, roles, canonical=True)
    validate(ranges, "DEV_RANGES", parents={"samples": samples["content_hash"]})
    response = c.product(parent, "candidate_B_L", "response")
    fit = c.product(parent, "candidate_B_L", "fit")
    report = c.product(parent, "report_B", "result")
    validate(response, "FITTED_RESPONSE")
    validate(fit, "DEV_FIT", parents={"response": response["content_hash"], "samples": samples["content_hash"]})
    parents = dict(response=response["content_hash"], samples=samples["content_hash"], ranges=ranges["content_hash"])
    validate(report, "DEV_EVALUATION", parents=parents)
    if (fit["candidate"] != "B_L" or report["candidate"] != "B_L" or response["candidate_id"] != "B_L"
            or response["rules"] != c.POLICIES[parent["policy"]]
            or response["parents"]["compatibility"] != study["review"]["content_hash"]
            or response["parents"]["source"] != donor["source"]["content_hash"] or not response["estimable"]
            or report["jets"] != COUNTS["evaluation"] or report["replicas"] != [0, 1, 2]
            or report["all_registered_jets_included"] is not True):
        raise ValueError("Completed B_L evaluation differs")
    receipts = {owner: c.verified_outputs(parent, owner)["content_hash"]
                for owner in ["candidate_B_L", "report_B", *[f"evaluate_B_{i}" for i in range(4)]]}
    for i in range(4):
        shard = c.product(parent, f"evaluate_B_{i}", "result")
        validate(shard, "DEV_HISTOGRAM_SHARD", parents=parents)
        if (shard["candidate"] != "B_L" or shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4
            or shard["replicas"] != [0, 1, 2] or shard["unresolved_jets_not_filtered"] is not True
            or shard["durable_proxy_arrays"] is not False):
            raise ValueError("Completed B shard differs")
    return artifact("DEV_B_REUSE", parents={**parents, "comparison": parent["content_hash"],
        "fit": fit["content_hash"], "report": report["content_hash"],
        "prepare_receipt": c.verified_outputs(pilot, "prepare")["content_hash"]},
        donor_receipts=receipts, new_study=study["content_hash"], association_recomputed=False,
        fitted_model_modified=False, old_jobs_modified=False, diagnostic_only=True)


def create(*, parent_spec, project_dir, source_commit, root):
    from . import dev_campaign as c
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    donor = c.validate_stage(parent, source=False)
    c.verified_outputs(parent, "report_B")
    destination, old_root = Path(root).resolve(), Path(donor["root"]).resolve()
    if destination == old_root or destination.is_relative_to(old_root) or old_root.is_relative_to(destination):
        raise PermissionError("Use a fresh disjoint B tracking audit root")
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in destination.parents):
        raise PermissionError("Do not nest the B audit inside another study or campaign")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="tier3")
    evidence = reuse(parent, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "reuse": evidence["content_hash"]},
        study=file_ref(destination/"study_spec.json"), root=study["root"], name=NAME, stage="btrack",
        parent_spec=parent_ref, policy=parent["policy"], b_threads=1, reuse=evidence,
        variants=list(VARIANTS), replicas=[0, 1, 2], tasks=tasks(), scientific_qualification=False,
        resources_are_development_envelopes=True)
    c.write(root, f"stages/{NAME}/stage_spec.json", spec, KIND)
    c.write(root, f"stages/{NAME}/command_plan.json", c.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_stage(spec, *, source=True):
    from . import dev_campaign as c
    study = load_json(checked_file(spec["study"]))
    c.validate_study(study, source=source)
    parent = load_json(checked_file(spec["parent_spec"]))
    evidence = reuse(parent, study)
    validate(spec, KIND, parents={"study": study["content_hash"], "reuse": evidence["content_hash"]})
    if (study["site"]["partition"] != "tier3" or spec["root"] != study["root"]
            or spec["name"] != NAME or spec["stage"] != "btrack" or spec["reuse"] != evidence
            or spec["variants"] != list(VARIANTS) or spec["replicas"] != [0, 1, 2]
            or spec["tasks"] != tasks() or spec["policy"] != parent["policy"] or spec["b_threads"] != 1
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Frozen B tracking diagnostic registration differs")
    return study
