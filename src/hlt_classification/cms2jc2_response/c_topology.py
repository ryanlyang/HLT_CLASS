"""Isolated, privileged observed-topology diagnostic; never a fitted mapper."""
from pathlib import Path

from .contracts import artifact, load_json, validate
from .dev_data import checked_file, file_ref
from .c_diagnostic import reuse

NAME = "observed_topology_r1"
KIND = "DEV_C_TOPOLOGY"
VIEWS = ("FREE_ALL", "FREE_RESOLVED", "FREE_UNRESOLVED", "FREE_COMPARABLE", "FIXED_FULL", "FIXED_CENTRAL")


def tasks():
    from .dev_campaign import task
    return [task("ct_acceptance", "ct_acceptance", 2, 32, 2), *[
        task(f"ct_eval_{i}", "ct_evaluate", 36, 128, 24, ["ct_acceptance"], shard=i)
        for i in range(4)], task("ct_report", "ct_report", 1, 32, 4, [f"ct_eval_{i}" for i in range(4)])]


def create(*, parent_spec, project_dir, source_commit, root):
    from . import dev_campaign as c
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    donor = c.validate_stage(parent, source=False)
    if parent["stage"] != "compare":
        raise ValueError("Use the original completed C comparison as parent")
    c.verified_outputs(parent, "report_C")
    destination, old_root = Path(root).resolve(), Path(donor["root"]).resolve()
    if destination == old_root or destination.is_relative_to(old_root) or old_root.is_relative_to(destination):
        raise PermissionError("Observed topology needs a fresh disjoint root")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), partition="tier3")
    evidence = reuse(parent, study)
    spec = artifact(KIND, parents={"study": study["content_hash"], "reuse": evidence["content_hash"]},
        study=file_ref(destination/"study_spec.json"), root=study["root"], name=NAME, stage="ctopo",
        parent_spec=parent_ref, policy=parent["policy"], b_threads=1, reuse=evidence,
        views=list(VIEWS), replicas=[0, 1, 2], tasks=tasks(), scientific_qualification=False,
        resources_are_development_envelopes=True, observed_state_privileged=True,
        evaluation_association_recomputed=True, fitted_model_modified=False)
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
        or spec["name"] != NAME or spec["stage"] != "ctopo" or spec["reuse"] != evidence
        or spec["views"] != list(VIEWS) or spec["replicas"] != [0, 1, 2] or spec["tasks"] != tasks()
        or spec["policy"] != parent["policy"] or spec["b_threads"] != 1
        or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True
        or spec["observed_state_privileged"] is not True or spec["evaluation_association_recomputed"] is not True
        or spec["fitted_model_modified"] is not False):
        raise ValueError("Observed-topology registration differs")
    return study
