"""Read-only, cross-commit dz-fix preparation import; never model reuse."""
from __future__ import annotations

from pathlib import Path
import re

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from .contracts import validate as validate_parent
from .dzfix_salience_continuation import validate_continuation
from .dzfix_fusion_chain import COUNTS, artifact, registration, validate
from .inventory import validate_inventory
from .production import _source
from .salience_foundation import authenticate_preparation, assignment_source, validate_foundation_spec
from .salience_production import _screen_artifacts
from .salience_screen import _task_report as screen_task_report, task_graph as screen_task_graph

JOB = re.compile(r"[1-9][0-9]*")


def _parent(launch):
    parent = load_json(launch["continuation_spec_path"])
    validate_continuation(parent, check_source=False)
    ledger_path = Path(parent["continuation_root"]) / "launchers/after_screen/submission_ledger.json"
    ledger = load_json(ledger_path)
    validate_submission_ledger(ledger)
    if (ledger["dry_run"] or ledger["campaign_spec_sha256"] != parent["content_hash"]
            or set(ledger["jobs"]) != {"after_screen"}
            or JOB.fullmatch(ledger["jobs"]["after_screen"]) is None):
        raise ValueError("Exact continuation after_screen live ledger required")
    return parent, ledger


def create_launch(*, continuation_spec: Path, inventory_path: Path, launch_root: Path,
                  campaign_root: Path, project: Path, source_commit: str):
    _source(project, source_commit)
    parent, ledger = _parent({"continuation_spec_path": str(continuation_spec)})
    inventory = load_json(inventory_path)
    validate_inventory(inventory)
    # This explicit producer registration distinguishes dzfix from the old,
    # otherwise schema-compatible September-10 snapshot.
    if "20260918_dzfix" not in parent["data_root"]:
        raise ValueError("This campaign requires the registered September-18 dzfix dataset")
    root, campaign = Path(launch_root).resolve(), Path(campaign_root).resolve()
    protected = [Path(parent[k]).resolve() for k in ("continuation_root", "data_root", "project_dir")]
    protected += [Path(project).resolve()]
    for output in (root, campaign):
        if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
            raise FileExistsError("New launch/campaign roots must be fresh and outside all source trees")
    if root.is_relative_to(campaign) or campaign.is_relative_to(root):
        raise ValueError("Launch and campaign roots must be separate")
    value = artifact("LAUNCH_SPEC", source_commit=source_commit,
        project_dir=str(Path(project).resolve()), launch_root=str(root), campaign_root=str(campaign),
        continuation_spec_path=str(Path(continuation_spec).resolve()),
        continuation_sha256=parent["content_hash"],
        parent_ledger_sha256=ledger["content_hash"], parent_job_id=ledger["jobs"]["after_screen"],
        inventory_path=str(Path(inventory_path).resolve()), inventory_sha256=inventory["content_hash"],
        registration=registration(), final_test_accessed=False)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "launch_spec.json", value)
    return value


def validate_launch(spec, *, check_source=True):
    digest = validate(spec, "LAUNCH_SPEC")
    parent, ledger = _parent(spec)
    inventory = load_json(spec["inventory_path"])
    validate_inventory(inventory)
    if (spec["registration"] != registration() or spec["final_test_accessed"] is not False
            or spec["continuation_sha256"] != parent["content_hash"]
            or spec["parent_ledger_sha256"] != ledger["content_hash"]
            or spec["parent_job_id"] != ledger["jobs"]["after_screen"]
            or spec["inventory_sha256"] != inventory["content_hash"]
            or "20260918_dzfix" not in parent["data_root"]):
        raise ValueError("Fusion-chain launch lineage differs")
    protected = [Path(parent[k]).resolve() for k in ("continuation_root", "data_root", "project_dir")]
    protected += [Path(spec["project_dir"]).resolve()]
    root, campaign = Path(spec["launch_root"]).resolve(), Path(spec["campaign_root"]).resolve()
    if (root.is_relative_to(campaign) or campaign.is_relative_to(root)
            or any(output.is_relative_to(p) or p.is_relative_to(output)
                   for output in (root, campaign) for p in protected)):
        raise ValueError("Output roots overlap protected source trees")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def _completion(parent):
    root = Path(parent["continuation_root"])
    complete = load_json(root / "continuation_complete.json")
    receipt = load_json(root / "after_screen_receipt.json")
    validate_parent(complete, "DZFIX_SALIENCE_CONTINUATION_COMPLETE")
    validate_parent(receipt, "DZFIX_SALIENCE_CONTINUATION_RECEIPT")
    campaign = load_json(root / "production/campaign_spec.json")
    validate_parent(campaign, "SALIENCE_CAMPAIGN_SPEC")
    dry = load_json(root / "production/dry_run_submission_ledger.json")
    validate_submission_ledger(dry)
    if (complete["continuation_sha256"] != parent["content_hash"]
            or complete["after_screen_receipt_sha256"] != receipt["content_hash"]
            or receipt["continuation_sha256"] != parent["content_hash"]
            or receipt["phase"] != "after_screen"
            or receipt["source_commit"] != parent["source_commit"]
            or receipt["campaign_sha256"] != campaign["content_hash"]
            or complete["campaign_sha256"] != campaign["content_hash"]
            or dry["campaign_spec_sha256"] != campaign["content_hash"]
            or receipt["production_dry_ledger_sha256"] != dry["content_hash"]
            or not dry["dry_run"] or len(dry["jobs"]) != 30
            or complete["final_test_accessed"] or receipt["final_test_accessed"]
            or receipt["live_production"]
            or complete["live_production"] or not complete["production_dry_run"]):
        raise ValueError("Continuation completion/production dry-run provenance differs")
    return complete, receipt


def build_import(launch):
    validate_launch(launch)
    parent, _ = _parent(launch)
    complete, receipt = _completion(parent)
    screen_path = Path(parent["continuation_root"]) / "screen/screen_spec.json"
    screen, selection, profile = _screen_artifacts(screen_path, deep=True)
    screen_reports = {row["task_id"]: screen_task_report(screen, row["task_id"]) for row in screen_task_graph()}
    if not all(screen_reports.values()):
        raise ValueError("Completed screen lacks required authenticated task receipts")
    if (screen_reports["select"]["result"] != selection
            or screen_reports["preflight"]["result"] != profile):
        raise ValueError("Screen selection/profile differs from task receipts")
    done = load_json(Path(screen["screen_root"]) / "screen_complete.json")
    if (screen["source_commit"] != parent["source_commit"]
            or screen["data_root"] != parent["data_root"]
            or done["screen_sha256"] != screen["content_hash"]
            or done["final_test_accessed"] or profile["final_test_accessed"]
            or receipt["screen_complete_sha256"] != done["content_hash"]):
        raise ValueError("Screen is not this completed dzfix continuation's screen")
    foundation_root = Path(selection["winner_foundation_root"])
    foundation_path = foundation_root / "foundation_spec.json"
    foundation = load_json(foundation_path)
    validate_foundation_spec(foundation)
    lock = authenticate_preparation(foundation, foundation_root)
    if (foundation["splits"]["profile"] != "TRAIN_500K"
            or foundation["splits"]["role_counts"] != COUNTS
            or foundation["inventory"] != load_json(launch["inventory_path"])
            or foundation["inputs"]["capacity"] != 240
            or selection["winner"] != foundation["candidate"]
            or selection["winner_foundation_sha256"] != foundation["content_hash"]):
        raise ValueError("Wrong snapshot, population, input capacity or selected salience foundation")
    producer_campaign = load_json(Path(parent["continuation_root"]) / "production/campaign_spec.json")
    if (producer_campaign["foundation"] != foundation
            or producer_campaign["source_commit"] != parent["source_commit"]
            or producer_campaign["data_root"] != parent["data_root"]):
        raise ValueError("Continuation production preview differs from selected source")
    files = [Path(launch["continuation_spec_path"]),
             Path(parent["continuation_root"]) / "launchers/after_screen/submission_ledger.json",
             Path(parent["continuation_root"]) / "continuation_complete.json",
             Path(parent["continuation_root"]) / "after_screen_receipt.json",
             Path(parent["continuation_root"]) / "production/campaign_spec.json",
             Path(parent["continuation_root"]) / "production/dry_run_submission_ledger.json",
             screen_path, Path(screen["screen_root"]) / "selection_lock.json",
             Path(screen["screen_root"]) / "screen_complete.json",
             Path(screen["screen_root"]) / "runtime_profile.json",
             foundation_path, foundation_root / "foundation_lock.json", Path(launch["inventory_path"])]
    files += [Path(screen["screen_root"]) / "tasks" / (name + ".json") for name in screen_reports]
    source = artifact("SOURCE_IMPORT", data_root=parent["data_root"],
        continuation_sha256=parent["content_hash"], continuation_complete_sha256=complete["content_hash"],
        producer_commit=parent["source_commit"], consumer_commit=launch["source_commit"],
        foundation_spec_path=str(foundation_path), foundation_root=str(foundation_root),
        foundation_sha256=foundation["content_hash"], foundation_lock_sha256=lock["content_hash"],
        selection_sha256=selection["content_hash"], screen_profile_sha256=profile["content_hash"],
        selected_candidate=foundation["candidate"], preparation_file_sha256=assignment_source()["file_sha256"],
        files={str(p): sha256_file(p) for p in files}, models_imported=[], final_test_accessed=False)
    return source, foundation


def validate_import(source, *, deep=False):
    digest = validate(source, "SOURCE_IMPORT")
    if (source["models_imported"] != [] or source["final_test_accessed"] is not False
            or source["preparation_file_sha256"] != assignment_source()["file_sha256"]):
        raise ValueError("Unreviewed preparation-code change or forbidden model import")
    if any(sha256_file(Path(p)) != h for p, h in source["files"].items()):
        raise ValueError("Imported preparation artifact bytes changed")
    foundation = load_json(source["foundation_spec_path"])
    validate_foundation_spec(foundation)
    if foundation["content_hash"] != source["foundation_sha256"]:
        raise ValueError("Imported foundation changed")
    if deep:
        lock = authenticate_preparation(foundation, Path(source["foundation_root"]))
        if lock["content_hash"] != source["foundation_lock_sha256"]:
            raise ValueError("Imported foundation lock changed")
    return digest
