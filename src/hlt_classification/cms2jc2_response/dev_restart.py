"""Fresh CPU64 comparison with authenticated reuse of frozen development inputs.

Only metadata/reports are read from the parent. No old fit output or Slurm job
is modified, cancelled, or adopted as a new result.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import artifact, load_json, validate
from .dev_data import checked_file, file_ref, validate_samples
from .dev_parallel import execution_profile

# All other files in the original source surface must match byte-for-byte.
# This includes the matcher, quotas/reservoir, reader/bridge, features,
# response families, random keys, model equations and diagnostic definitions.
EXECUTION_ONLY_FILES = {
    "src/hlt_classification/cms2jc2_response/dev_campaign.py",
    "src/hlt_classification/cms2jc2_response/dev_submission.py",
    "src/hlt_classification/cms2jc2_response/dev_worker.py",
    "scripts/cms2jc2_response_dev.py",
    "sbatch/run_cms2jc2_response_dev_cpu.sh",
    "docs/plans/CMS2JC2_RESPONSE_CPU_DEVELOPMENT_PLAN.md",
    "docs/contracts/CMS2JC2_RESPONSE_CPU_DEVELOPMENT.md",
}


def reuse_evidence(parent, study):
    from . import dev_campaign as c
    if parent.get("stage") != "confirm":
        raise ValueError("CPU64 reuse requires a completed association confirmation")
    donor = c.validate_stage(parent, source=False)
    if study["root"] == donor["root"]:
        raise PermissionError("CPU64 comparison needs a fresh separate study")
    if (study["imported"] != donor["imported"] or study["review"] != donor["review"]
            or study["numerical_environment"] != donor["numerical_environment"]):
        raise ValueError("CPU64 reuse changed CMS inputs, conventions or numerical environment")
    old_source = donor["source"]
    new_source = study["source"]
    for name, digest in old_source["files"].items():
        if name not in EXECUTION_ONLY_FILES and new_source["files"].get(name) != digest:
            raise ValueError(f"CPU64 reuse changed scientific source: {name}")
    pilot = c.preparation_stage(parent)
    report = c.product(parent, "association_confirm", "result")
    samples = c.product(pilot, "prepare", "samples")
    ranges = c.product(pilot, "prepare", "ranges")
    from .dev_data import COUNTS
    inventory = load_json(checked_file(study["imported"]["files"]["cms_inventory.json"]))
    roles = load_json(checked_file(study["imported"]["files"]["response_roles.json"]))
    validate_samples(samples, inventory, roles, canonical=True)
    validate(report, "DEV_ASSOCIATION", parents={"samples": samples["content_hash"]})
    validate(ranges, "DEV_RANGES", parents={"samples": samples["content_hash"]})
    if (report["jets"] != COUNTS["location"]+COUNTS["residual"] or report["policy"] != parent["policy"]
            or report["rules"] != c.POLICIES[parent["policy"]]):
        raise ValueError("CPU64 confirmation population or matching policy changed")
    return artifact("DEV_CPU64_REUSE", parents={"donor_study": donor["content_hash"],
        "confirmation": parent["content_hash"], "confirmation_report": report["content_hash"],
        "prepare_receipt": c.verified_outputs(pilot, "prepare")["content_hash"],
        "samples": samples["content_hash"], "ranges": ranges["content_hash"]},
        new_study_sha256=study["content_hash"], matching_recomputed_only_for_ram_records=True,
        confirmation_rerun=False, old_models_imported=False, old_jobs_modified=False)


def create_compare64(*, parent_spec, project_dir, source_commit, root, partition="debug"):
    from . import dev_campaign as c
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    if parent.get("stage") != "confirm":
        raise ValueError("Use the completed confirm_search_r1 stage as parent")
    donor = c.validate_stage(parent, source=False)
    # Check output receipts before creating any new root.
    c.verified_outputs(parent, "association_confirm")
    destination = Path(root).resolve()
    old_root = Path(donor["root"]).resolve()
    if destination == old_root or destination.is_relative_to(old_root) or old_root.is_relative_to(destination):
        raise PermissionError("CPU64 destination overlaps the original study")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit,
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]), root=root, partition=partition)
    reuse = reuse_evidence(parent, study)
    profile = execution_profile()
    spec = artifact("DEV_STAGE64", parents={"study": study["content_hash"],
                    "reuse": reuse["content_hash"], "execution": profile["content_hash"]},
        study=file_ref(destination/"study_spec.json"), root=study["root"], stage="compare", name="compare64_r1",
        parent_spec=parent_ref, policy=parent["policy"], b_threads=1, execution=profile, reuse=reuse,
        tasks=c.tasks("compare", parent["policy"], 1, cpu64=True),
        scientific_qualification=False, resources_are_development_envelopes=True)
    directory = c.stage_dir(spec)
    directory.mkdir(parents=True, exist_ok=False)
    c.write(root, "stages/compare64_r1/stage_spec.json", spec, "DEV_STAGE64")
    c.write(root, "stages/compare64_r1/command_plan.json", c.command_plan(spec, study), "DEV_PLAN")
    return spec


def validate_compare64(spec, *, source=True):
    from . import dev_campaign as c
    study = load_json(checked_file(spec["study"]))
    c.validate_study(study, source=source)
    profile = execution_profile()
    parent = load_json(checked_file(spec["parent_spec"]))
    reuse = reuse_evidence(parent, study)
    validate(spec, "DEV_STAGE64", parents={"study": study["content_hash"],
             "reuse": reuse["content_hash"], "execution": profile["content_hash"]})
    if (spec["root"] != study["root"] or spec["name"] != "compare64_r1" or spec["stage"] != "compare"
            or spec["policy"] != parent["policy"] or spec["b_threads"] != 1
            or spec["execution"] != profile or spec["reuse"] != reuse
            or spec["tasks"] != c.tasks("compare", parent["policy"], 1, cpu64=True)
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("CPU64 comparison registration differs")
    return study
