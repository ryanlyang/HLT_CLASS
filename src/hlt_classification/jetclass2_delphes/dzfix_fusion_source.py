"""Read-only v2 debug-screen import; no continuation or trained-model reuse."""
from __future__ import annotations

from pathlib import Path
import re

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from .contracts import validate as validate_parent
from .dzfix_fusion_chain import COUNTS, artifact, registration, validate
from .execution import execution_site
from .inputs import input_contract
from .inventory import validate_inventory
from .production import _source
from .salience_foundation import authenticate_preparation, assignment_source, validate_foundation_spec
from .salience_production import _screen_artifacts
from .salience_screen import (
    REGISTRY, _task_report as screen_task_report, command_plan as screen_command_plan,
    task_graph as screen_task_graph, validate_screen,
)

JOB = re.compile(r"[1-9][0-9]*")


def _parent(launch):
    path = Path(launch["screen_spec_path"]).resolve()
    screen = load_json(path)
    validate_screen(screen)
    if (screen.get("schema_version") != 2
            or screen.get("screen_execution_site") != execution_site("sporc_a100_debug")
            or screen.get("production_execution_site") != execution_site("sporc_a100")
            or "20260918_dzfix" not in screen["data_root"]
            or screen["split_profile"] != "TRAIN_500K" or screen["role_counts"] != COUNTS
            or path != (Path(screen["screen_root"]) / "screen_spec.json").resolve()
            or [r["candidate"] for r in screen["candidates"]] != REGISTRY):
        raise ValueError("Exact dzfix TRAIN_500K debug screen v2 required")
    ledger = load_json(Path(screen["screen_root"]) / "submission_ledger.json")
    validate_submission_ledger(ledger)
    tasks = {r["task_id"] for r in screen_task_graph()}
    if (ledger["dry_run"] is not False or ledger["campaign_spec_sha256"] != screen["content_hash"]
            or set(ledger["jobs"]) != tasks or len(set(ledger["jobs"].values())) != len(tasks)
            or any(JOB.fullmatch(j) is None for j in ledger["jobs"].values())):
        raise ValueError("Exact eight-task screen live ledger with complete job required")
    # Verify the complete dependency closure, not merely a plausible job number.
    for row in screen_command_plan(screen)["commands"]:
        expected = list(row["command"])
        for task, job in ledger["jobs"].items():
            expected = [token.replace("${JOB_" + task + "}", job) for token in expected]
        if ledger["commands"][row["task_id"]] != expected:
            raise ValueError("Screen ledger commands/dependencies differ from exact source plan")
    return screen, ledger


def _population(screen, inventory):
    rows = screen["candidates"] + [dict(foundation_root=screen["bottleneck_root"],
                                       foundation_sha256=screen["bottleneck_sha256"])]
    splits = None
    expected_inputs = None
    for row in rows:
        path = Path(row["foundation_root"]) / "foundation_spec.json"
        foundation = load_json(path)
        checks = {
            "foundation hash": foundation["content_hash"] == row["foundation_sha256"],
            "inventory": foundation["inventory"] == inventory,
            "split profile": foundation["splits"]["profile"] == "TRAIN_500K",
            "role counts": foundation["splits"]["role_counts"] == COUNTS,
            "split membership": splits is None or foundation["splits"] == splits,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Screen snapshot/population/foundation identity differs at {path}: {', '.join(failed)}")
        if expected_inputs is None:
            # Same metadata-only, round-UP rule as both foundation builders.
            # Never import the previous snapshot's fixed 240-token assumption,
            # change the published foundation, or truncate a longer real jet.
            maximum = max(max(r["max_selected_particles"].values()) for r in inventory["files"])
            capacity = max(16, ((maximum + 15) // 16) * 16)
            expected_inputs = input_contract(capacity=capacity)
        if foundation["inputs"] != expected_inputs:
            raise ValueError(
                f"Screen foundation input contract differs at {path}: "
                f"inventory-derived capacity={expected_inputs['capacity']}, "
                f"foundation capacity={foundation['inputs'].get('capacity')}; "
                "exact 17-feature/11-class no-truncation contract required"
            )
        splits = foundation["splits"]


def _protected(screen, project):
    return [Path(screen[k]).resolve() for k in ("screen_root", "data_root", "project_dir", "bottleneck_root")] + [
        Path(r["foundation_root"]).resolve() for r in screen["candidates"]
    ] + [Path(project).resolve()]


def create_launch(*, screen_spec: Path, inventory_path: Path, launch_root: Path,
                  campaign_root: Path, project: Path, source_commit: str):
    _source(project, source_commit)
    screen, ledger = _parent({"screen_spec_path": str(screen_spec)})
    inventory = load_json(inventory_path)
    validate_inventory(inventory)
    _population(screen, inventory)
    root, campaign = Path(launch_root).resolve(), Path(campaign_root).resolve()
    protected = _protected(screen, project)
    for output in (root, campaign):
        if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
            raise FileExistsError("New launch/campaign roots must be fresh and outside all source trees")
    if root.is_relative_to(campaign) or campaign.is_relative_to(root):
        raise ValueError("Launch and campaign roots must be separate")
    value = artifact("LAUNCH_SPEC", source_commit=source_commit,
        project_dir=str(Path(project).resolve()), launch_root=str(root), campaign_root=str(campaign),
        screen_spec_path=str(Path(screen_spec).resolve()), screen_sha256=screen["content_hash"],
        parent_task_id="complete", parent_ledger_sha256=ledger["content_hash"],
        parent_job_id=ledger["jobs"]["complete"],
        inventory_path=str(Path(inventory_path).resolve()), inventory_sha256=inventory["content_hash"],
        registration=registration(), final_test_accessed=False)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root / "launch_spec.json", value)
    return value


def validate_launch(spec, *, check_source=True):
    digest = validate(spec, "LAUNCH_SPEC")
    screen, ledger = _parent(spec)
    inventory = load_json(spec["inventory_path"])
    validate_inventory(inventory)
    _population(screen, inventory)
    if (spec["registration"] != registration() or spec["final_test_accessed"] is not False
            or spec["screen_sha256"] != screen["content_hash"] or spec["parent_task_id"] != "complete"
            or spec["parent_ledger_sha256"] != ledger["content_hash"]
            or spec["parent_job_id"] != ledger["jobs"]["complete"]
            or spec["inventory_sha256"] != inventory["content_hash"]):
        raise ValueError("Fusion-chain launch lineage differs")
    root, campaign = Path(spec["launch_root"]).resolve(), Path(spec["campaign_root"]).resolve()
    if (root.is_relative_to(campaign) or campaign.is_relative_to(root)
            or any(output.is_relative_to(p) or p.is_relative_to(output)
                   for output in (root, campaign) for p in _protected(screen, spec["project_dir"]))):
        raise ValueError("Output roots overlap protected source trees")
    if check_source:
        _source(Path(spec["project_dir"]), spec["source_commit"])
    return digest


def build_import(launch):
    validate_launch(launch)
    parent, ledger = _parent(launch)
    screen_path = Path(launch["screen_spec_path"])
    screen, selection, profile = _screen_artifacts(screen_path, deep=True)
    if screen != parent:
        raise ValueError("Screen changed during import")
    root = Path(screen["screen_root"])
    complete = load_json(root / "screen_complete.json")
    validate_parent(complete, "SALIENCE_SCREEN_COMPLETE")
    screen_reports = {row["task_id"]: screen_task_report(screen, row["task_id"]) for row in screen_task_graph()}
    if not all(screen_reports.values()):
        raise ValueError("Completed screen lacks required authenticated task receipts")
    if (screen_reports["select"]["result"] != selection
            or screen_reports["preflight"]["result"] != profile
            or screen_reports["complete"]["result"] != complete
            or complete["screen_sha256"] != screen["content_hash"]
            or complete["selection_lock_sha256"] != selection["content_hash"]
            or complete["scientific_fit_count"] != 4 or complete["final_test_accessed"] is not False
            or profile["final_test_accessed"] is not False or profile["passed"] is not True
            or profile["source_commit"] != screen["source_commit"]):
        raise ValueError("Screen completion/selection/profile differs from task receipts")
    chosen = next((r for r in screen["candidates"] if r["candidate"] == selection["winner"]), None)
    if (chosen is None or chosen["foundation_sha256"] != selection["winner_foundation_sha256"]
            or Path(chosen["foundation_root"]).resolve() != Path(selection["winner_foundation_root"]).resolve()):
        raise ValueError("Winner is not a foundation registered in the source screen")
    foundation_root = Path(chosen["foundation_root"]).resolve()
    foundation_path = foundation_root / "foundation_spec.json"
    foundation = load_json(foundation_path)
    validate_foundation_spec(foundation)
    lock = authenticate_preparation(foundation, foundation_root)
    if (foundation["content_hash"] != chosen["foundation_sha256"]
            or foundation["candidate"] != selection["winner"]):
        raise ValueError("Selected salience foundation differs")
    files = [screen_path, root / "submission_ledger.json", root / "selection_lock.json",
             root / "screen_complete.json", root / "runtime_profile.json",
             foundation_path, foundation_root / "foundation_lock.json", Path(launch["inventory_path"])]
    files += [root / "tasks" / (name + ".json") for name in screen_reports]
    source = artifact("SOURCE_IMPORT", data_root=screen["data_root"],
        screen_spec_path=str(screen_path), screen_sha256=screen["content_hash"],
        screen_complete_sha256=complete["content_hash"], parent_ledger_sha256=ledger["content_hash"],
        parent_task_id="complete", parent_job_id=ledger["jobs"]["complete"],
        producer_commit=screen["source_commit"], consumer_commit=launch["source_commit"],
        foundation_spec_path=str(foundation_path), foundation_root=str(foundation_root),
        foundation_sha256=foundation["content_hash"], foundation_lock_sha256=lock["content_hash"],
        selection_sha256=selection["content_hash"], screen_profile_sha256=profile["content_hash"],
        selected_candidate=foundation["candidate"], preparation_file_sha256=assignment_source()["file_sha256"],
        files={str(p): sha256_file(p) for p in files}, models_imported=[], final_test_accessed=False)
    return source, foundation


def validate_import(source, *, deep=False):
    digest = validate(source, "SOURCE_IMPORT")
    if (source["models_imported"] != [] or source["final_test_accessed"] is not False
            or source["parent_task_id"] != "complete"
            or source["preparation_file_sha256"] != assignment_source()["file_sha256"]):
        raise ValueError("Unreviewed preparation-code change or forbidden model import")
    if any(sha256_file(Path(p)) != h for p, h in source["files"].items()):
        raise ValueError("Imported preparation artifact bytes changed")
    foundation = load_json(source["foundation_spec_path"])
    validate_foundation_spec(foundation)
    if foundation["content_hash"] != source["foundation_sha256"]:
        raise ValueError("Imported foundation changed")
    if deep:
        screen, ledger = _parent(source)
        _population(screen, foundation["inventory"])
        if (screen["content_hash"] != source["screen_sha256"]
                or ledger["content_hash"] != source["parent_ledger_sha256"]
                or ledger["jobs"]["complete"] != source["parent_job_id"]):
            raise ValueError("Imported screen/ledger changed")
        lock = authenticate_preparation(foundation, Path(source["foundation_root"]))
        if lock["content_hash"] != source["foundation_lock_sha256"]:
            raise ValueError("Imported foundation lock changed")
    return digest
