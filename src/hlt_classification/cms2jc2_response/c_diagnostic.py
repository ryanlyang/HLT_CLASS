"""Read-only fitted-C reuse and the canonical CPU diagnostic registration."""
from pathlib import Path

from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref, validate_samples
from .c_diagnostic_generation import VARIANTS

NAME = "frozen_c_r1"
KIND = "DEV_C_DIAGNOSTIC"


def tasks():
    from .dev_campaign import task
    result = [task("c_acceptance", "c_acceptance", 1, 32, 2)]
    for variant in VARIANTS:
        names = [f"c_eval_{variant}_{i}" for i in range(4)]
        result += [task(name, "c_evaluate", 1, 32, 12, ["c_acceptance"], variant=variant, shard=i)
                   for i, name in enumerate(names)]
        result.append(task("c_report_"+variant, "c_report", 1, 16, 2, names, variant=variant))
    result.append(task("c_summary", "c_summary", 1, 16, 2, ["c_report_"+v for v in VARIANTS]))
    return result


def reuse(parent, study):
    from . import dev_campaign as c
    from .dev_restart import EXECUTION_ONLY_FILES
    if parent.get("stage") != "compare":
        raise ValueError("Frozen C requires a completed C_L comparison")
    donor = c.validate_stage(parent, source=False)
    if (study["imported"] != donor["imported"] or study["review"] != donor["review"]
            or study["numerical_environment"] != donor["numerical_environment"]):
        raise ValueError("Frozen C inputs, conventions or numerical environment changed")
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("Frozen C scientific source changed: "+name)
    old_root, new_root = Path(donor["root"]).resolve(), Path(study["root"]).resolve()
    if old_root == new_root or old_root.is_relative_to(new_root) or new_root.is_relative_to(old_root):
        raise PermissionError("Frozen C destination overlaps donor")
    pilot = c.preparation_stage(parent)
    samples, ranges = (c.product(pilot, "prepare", key) for key in ("samples", "ranges"))
    inventory, roles = (load_json(checked_file(study["imported"]["files"][key]))
                        for key in ("cms_inventory.json", "response_roles.json"))
    validate_samples(samples, inventory, roles, canonical=True)
    validate(ranges, "DEV_RANGES", parents={"samples": samples["content_hash"]})
    response = c.product(parent, "candidate_C_L", "response")
    fit = c.product(parent, "candidate_C_L", "fit")
    report = c.product(parent, "report_C", "result")
    validate(response, "FITTED_RESPONSE")
    validate(fit, "DEV_FIT", parents={"response": response["content_hash"], "samples": samples["content_hash"]})
    parents = dict(response=response["content_hash"], samples=samples["content_hash"], ranges=ranges["content_hash"])
    validate(report, "DEV_EVALUATION", parents=parents)
    if (fit["candidate"] != "C_L" or report["candidate"] != "C_L" or response["candidate_id"] != "C_L"
            or response["rules"] != c.POLICIES[parent["policy"]]
            or response["parents"]["compatibility"] != study["review"]["content_hash"]
            or response["parents"]["source"] != donor["source"]["content_hash"] or not response["estimable"]
            or report["jets"] != COUNTS["evaluation"] or report["replicas"] != [0, 1, 2]
            or report["all_registered_jets_included"] is not True):
        raise ValueError("Completed C_L evaluation differs")
    receipts = {owner: c.verified_outputs(parent, owner)["content_hash"]
                for owner in ["candidate_C_L", "report_C", *[f"evaluate_C_{i}" for i in range(4)]]}
    for i in range(4):
        shard = c.product(parent, f"evaluate_C_{i}", "result")
        validate(shard, "DEV_HISTOGRAM_SHARD", parents=parents)
        if shard["candidate"] != "C_L" or shard["shard"] != i or shard["jets"] != COUNTS["evaluation"]//4:
            raise ValueError("Completed C shard differs")
    return artifact("DEV_C_REUSE", parents={**parents, "comparison": parent["content_hash"],
        "fit": fit["content_hash"], "report": report["content_hash"],
        "prepare_receipt": c.verified_outputs(pilot, "prepare")["content_hash"]},
        donor_receipts=receipts, new_study=study["content_hash"], association_recomputed=False,
        fitted_model_modified=False, old_jobs_modified=False, diagnostic_only=True)


def create(*, parent_spec, project_dir, source_commit, root):
    from . import dev_campaign as c
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    donor = c.validate_stage(parent, source=False)
    c.verified_outputs(parent, "report_C")
    destination, old_root = Path(root).resolve(), Path(donor["root"]).resolve()
    if destination == old_root or destination.is_relative_to(old_root) or old_root.is_relative_to(destination):
        raise PermissionError("Use a fresh disjoint C diagnostic root")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="tier3")
    evidence = reuse(parent, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "reuse": evidence["content_hash"]},
        study=file_ref(destination/"study_spec.json"), root=study["root"], name=NAME, stage="cdiag",
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
            or spec["name"] != NAME or spec["stage"] != "cdiag" or spec["reuse"] != evidence
            or spec["variants"] != list(VARIANTS) or spec["replicas"] != [0, 1, 2]
            or spec["tasks"] != tasks() or spec["policy"] != parent["policy"] or spec["b_threads"] != 1
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Frozen C diagnostic registration differs")
    return study
