"""Explicit debug-measured CORR_MID -> tier3 execution, with unchanged science.

Old gate/campaign/profile records remain immutable. A new executor imports the
passed gate only when its scientific source is byte-identical, except for the
single version-dispatch insertion documented below. No runtime monkeypatching.
"""
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.execution import DEBUG_PROFILE_TRANSFER, execution_site
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from . import correlated as original, gate as g, submission as s
from .contracts import artifact, file_ref, validate, validate_file_ref, write_json

BASE_COMMIT = "062713ead3e00721d9f3cb7073e88aef83ebd325"
AUTHORIZATION = "AUTHORIZE CORR MID 100K 50K TIER3 DIRECT COARSE EXACT PLAN"
PRODUCTION = "src/hlt_classification/cms_proxy_ladder/production.py"
ADDED = "src/hlt_classification/cms_proxy_ladder/correlated_tier3.py"
NEEDLE = '    version = spec.get("schema_version")\n'
INSERTION = (
    '    if version == 6:\n'
    '        from .correlated_tier3 import validate_campaign as validate_tier3_campaign\n'
    '        return validate_tier3_campaign(spec, check_source=check_source)\n'
)


def source_transfer(base, project, commit):
    """No blanket whitelist: the old production file may gain only this dispatch."""
    if base["source_commit"] != BASE_COMMIT:
        raise ValueError("Tier3 transfer requires the approved original source pin")
    source = g.source_lock(Path(project), commit, literature=True, correlated=True)
    old, new = base["source"]["files"], source["files"]
    if set(new) != set(old) | {ADDED}:
        raise ValueError("Executor source file set differs from the registered amendment")
    for name in old:
        if name != PRODUCTION and new[name] != old[name]:
            raise ValueError("Measured scientific source changed: " + name)
    before = (Path(base["project_dir"]) / PRODUCTION).read_text(encoding="utf-8")
    after = (Path(project) / PRODUCTION).read_text(encoding="utf-8")
    if before.count(NEEDLE) != 1 or after != before.replace(NEEDLE, NEEDLE + INSERTION, 1):
        raise ValueError("Only the registered version-dispatch insertion is allowed")
    transfer = artifact("TIER3_SOURCE_TRANSFER", parents={
        "measurement_source": base["source"]["content_hash"], "executor_source": source["content_hash"]},
        policy="unchanged_science_with_exact_v6_validation_dispatch_v1",
        modified=[PRODUCTION], added=[ADDED])
    return source, transfer


def execution_profile(measured):
    # Do not pretend that profiling was performed on tier3 or with the new pin.
    fields = {k: v for k, v in measured.items() if k not in {
        "contract", "schema_version", "content_hash", "parents", "final_test_accessed"}}
    fields.update(execution_site=execution_site("sporc_a100"),
                  measurement_site=measured["execution_site"],
                  site_transfer_policy=DEBUG_PROFILE_TRANSFER)
    return artifact("TIER3_RUNTIME_PROFILE", parents={"measurement": measured["content_hash"]}, **fields)


def _build(base, reference, project, commit, root, source, transfer):
    profile = execution_profile(base["runtime_profile"])
    fields = {k: v for k, v in base.items() if k not in {
        "contract", "schema_version", "content_hash", "parents", "final_test_accessed"}}
    fields.update(campaign_root=str(Path(root).resolve()), project_dir=str(Path(project).resolve()),
                  source_commit=commit, source=source, runtime_profile=profile,
                  measurement_campaign=reference, source_transfer=transfer)
    return artifact("CAMPAIGN_SPEC", version=6, parents={
        **base["parents"], "profile": profile["content_hash"],
        "measurement_campaign": base["content_hash"], "source_transfer": transfer["content_hash"]}, **fields)


def create_campaign(*, measurement_spec, project_dir, source_commit, campaign_root):
    reference = file_ref(Path(measurement_spec))
    base = load_json(validate_file_ref(reference))
    original.validate_campaign(base, check_source=True)
    root = Path(campaign_root).resolve()
    gate = load_json(Path(base["gate_root"]) / "gate_spec.json")
    protected = [base["campaign_root"], base["gate_root"], base["project_dir"],
                 project_dir, gate["request"]["study_root"], gate["request"]["offline_root"]]
    if root.exists() or any(root.is_relative_to(Path(p).resolve()) or
                            Path(p).resolve().is_relative_to(root) for p in protected):
        raise FileExistsError("Fresh separate tier3 science root required")
    source, transfer = source_transfer(base, project_dir, source_commit)
    spec = _build(base, reference, project_dir, source_commit, root, source, transfer)
    validate_campaign(spec, check_source=True)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "campaign_spec.json", spec)
    return spec


def validate_campaign(spec, *, check_source=False):
    digest = validate(spec, "CAMPAIGN_SPEC", version=6)
    base = load_json(validate_file_ref(spec["measurement_campaign"]))
    original.validate_campaign(base, check_source=check_source)
    source, transfer = source_transfer(base, spec["project_dir"], spec["source_commit"])
    expected = _build(base, spec["measurement_campaign"], spec["project_dir"], spec["source_commit"],
                      spec["campaign_root"], source, transfer)
    if spec != expected:
        raise ValueError("Tier3 transfer changed science, profile, source, or lineage")
    return digest


def plan(spec):
    validate_campaign(spec, check_source=True)
    # Shared command builder/worker, but an explicit separate job namespace.
    value = s.science_plan(spec)
    for row in value["commands"]:
        row["command"] = [f"--job-name=jc2crt_{row['task_id']}" if a.startswith("--job-name=") else a
                          for a in row["command"]]
    return artifact("COMMAND_PLAN", parents={"subject": spec["content_hash"]},
                    mode="science", commands=value["commands"])


def submit(spec, *, execute=False, plan_hash=None, authorization_phrase=None):
    value = plan(spec)
    root = Path(spec["campaign_root"])
    path = root / "command_plan.json"
    if path.exists():
        if load_json(path) != value:
            raise ValueError("Saved tier3 command plan differs")
    else:
        write_json(path, value)
    if not execute:
        submit_exact_dag(identity=spec["content_hash"], plan=value,
            output=root / "dry_run_submission_ledger.json",
            canonical_dry_run=root / "dry_run_submission_ledger.json", execute=False)
        return value
    if authorization_phrase != AUTHORIZATION or plan_hash != value["content_hash"]:
        raise PermissionError("Exact reviewed tier3 plan and authorization required")
    original.check_submission_site(value)
    return original.submit_claimed(spec, value, root)
