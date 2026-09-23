"""Synthetic parent authentication and success-gated launch coverage for K2."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as chain, concat_k2_source as source,
    concat_k2_submit as scheduler, concat_k2_runtime as runtime,
)
from hlt_classification.jetclass2_delphes.contracts import artifact as parent_artifact
from hlt_classification.jetclass2_delphes.inputs import input_contract
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_jetclass2_dzfix_fusion_chain import rehash

def test_deferred_launcher_uses_exact_postscreen_boundary_and_no_expired_id(monkeypatch, tmp_path):
    spec = dict(content_hash="a"*64, project_dir=str(tmp_path / "project"), launch_root=str(tmp_path / "launch"),
                campaign_root=str(tmp_path / "campaign"), parent_job_id="21748725", registration=chain.registration())
    parent = {"screen_root": str(tmp_path / "parent")}
    monkeypatch.setattr(scheduler, "validate_launch", lambda s: s["content_hash"])
    monkeypatch.setattr(scheduler, "_parent", lambda s: (parent, {}))
    pending = scheduler.launcher_plan(spec, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:21748725" in pending and "--partition=tier3" in pending
    calls = []
    def sbatch(c, **kw):
        calls.append(c)
        return SimpleNamespace(stdout="12345")
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    first = scheduler.schedule(spec, execute=True, authorization=chain.AUTHORIZE)
    write_immutable_json(tmp_path / "parent/screen_complete.json", {})
    validated = []
    monkeypatch.setattr(scheduler, "build_import", lambda s: validated.append(True))
    done = scheduler.launcher_plan(spec, "after_matching")["commands"][0]["command"]
    assert validated == [True] and not any(x.startswith("--dependency") for x in done)
    assert scheduler.schedule(spec, execute=True, authorization=chain.AUTHORIZE) == first
    assert len(calls) == 1
    def corrupt(s):
        raise ValueError("corrupt completion")
    monkeypatch.setattr(scheduler, "build_import", corrupt)
    with pytest.raises(ValueError, match="corrupt"):
        scheduler.launcher_plan(spec, "after_matching")


@pytest.mark.parametrize("fault", ["individual_fit", "dry", "wrong_screen", "duplicate_job", "wrong_dependency"])
def test_source_parent_rejects_inexact_screen_ledgers(imported_source, fault):
    launch, _, root = imported_source
    path = root / "submission_ledger.json"
    value = load_json(path)
    if fault == "individual_fit":
        value["jobs"] = {"fit_LINEAR": "21748721"}
        value["commands"] = {"fit_LINEAR": ["sbatch", "a"]}
    elif fault == "dry":
        value["dry_run"] = True
    elif fault == "wrong_screen":
        value["campaign_spec_sha256"] = "f" * 64
    elif fault == "duplicate_job":
        value["jobs"]["complete"] = value["jobs"]["select"]
    else:
        value["commands"]["complete"] = [
            x.replace("afterok:21748724", "afterok:21741414")
            for x in value["commands"]["complete"]]
    path.unlink()
    write_immutable_json(path, rehash(value))
    with pytest.raises(ValueError, match="ledger"):
        source._parent(launch)



@pytest.fixture
def imported_source(tmp_path, monkeypatch):
    """Exercise actual v2 screen/ledger/receipt validators with synthetic payloads."""
    from hlt_classification.jetclass2_delphes import salience_screen as screen_module
    root = tmp_path / "debug_screen"
    inventory = parent_artifact("INVENTORY", test_fixture=True, files=[
        {"max_selected_particles": {"hlt": 311, "offline": 319}},
        {"max_selected_particles": {"hlt": 200, "offline": 220}},
    ])
    splits = {"profile": "TRAIN_500K", "role_counts": chain.COUNTS}
    candidates, foundations, locks = [], {}, {}
    for name in source.REGISTRY:
        foundation_root = tmp_path / "shared_maps" / name
        value = parent_artifact("SALIENCE_FOUNDATION_SPEC", inventory=inventory,
            splits=splits, inputs=input_contract(capacity=320), candidate=name, assignment_tasks=[dict(file_index=0, path='test.root', role='train', rows=500000)])
        lock = parent_artifact("SALIENCE_FOUNDATION_LOCK", foundation_sha256=value["content_hash"])
        write_immutable_json(foundation_root / "foundation_spec.json", value)
        write_immutable_json(foundation_root / "foundation_lock.json", lock)
        candidates.append(dict(candidate=name, foundation_root=str(foundation_root),
                               foundation_sha256=value["content_hash"]))
        foundations[name] = (value, foundation_root)
        locks[str(foundation_root)] = lock
    foundation, foundation_root = foundations[source.REGISTRY[0]]
    bottleneck_root = tmp_path / "bottleneck"
    bottleneck = parent_artifact("FOUNDATION_SPEC", inventory=inventory,
                                splits=splits, inputs=input_contract(capacity=320))
    write_immutable_json(bottleneck_root / "foundation_spec.json", bottleneck)
    foundations[screen_module.CONTEXT] = (bottleneck, bottleneck_root)
    template = parent_artifact("RUNTIME_PROFILE", execution_site=source.execution_site("sporc_a100"),
                              cpus=8, memory_mb=73728, workers=8, train_minutes=808)
    screen = parent_artifact("SALIENCE_SCREEN_SPEC", version=2, screen_root=str(root),
        source_commit="b"*40, project_dir=str(tmp_path / "producer_project"),
        data_root=str(tmp_path / "jetclass2_20260918_dzfix"),
        screen_execution_site=source.execution_site("sporc_a100_debug"),
        production_execution_site=source.execution_site("sporc_a100"),
        execution_policy=screen_module.DEBUG_SCREEN_POLICY, scientific_configuration_unchanged=True,
        execution_changes=["partition_tier3_to_debug", "bounded_debug_walltime"],
        debug_walltime_minutes=480, candidate_registry=source.REGISTRY,
        contextual_control=screen_module.CONTEXT, scientific_fit_count=4,
        coordinate="U100", final_test_accessed=False, existing_campaign_mutations=False,
        split_profile="TRAIN_500K", role_counts=chain.COUNTS, candidates=candidates,
        bottleneck_root=str(bottleneck_root), bottleneck_sha256=bottleneck["content_hash"],
        resource_template_path=str(tmp_path / "resource_template.json"),
        resource_template_sha256=template["content_hash"])
    selection = parent_artifact("SALIENCE_SELECTION_LOCK", winner=foundation["candidate"],
        winner_foundation_root=str(foundation_root), winner_foundation_sha256=foundation["content_hash"],
        screen_sha256=screen["content_hash"], final_test_accessed=False)
    profile = parent_artifact("SALIENCE_RUNTIME_PROFILE", version=2,
        screen_sha256=screen["content_hash"], source_commit=screen["source_commit"],
        execution_site=screen["production_execution_site"], screen_execution_site=screen["screen_execution_site"],
        execution_policy=screen["execution_policy"], passed=True, final_test_accessed=False)
    done = parent_artifact("SALIENCE_SCREEN_COMPLETE", screen_sha256=screen["content_hash"],
        selection_lock_sha256=selection["content_hash"], scientific_fit_count=4, final_test_accessed=False)
    for name, value in {
        "screen_spec.json": screen, "selection_lock.json": selection,
        "screen_complete.json": done, "runtime_profile.json": profile,
        "screen_split.json": {"test_fixture": True}, "inventory.json": inventory,
    }.items():
        write_immutable_json(root / name, value)
    monkeypatch.setattr(source, "_source", lambda *a: None)
    monkeypatch.setattr(screen_module, "_source", lambda *a: None)
    monkeypatch.setattr(screen_module, "_load_foundations", lambda *a, **k: foundations)
    monkeypatch.setattr(screen_module, "_template", lambda *a, **k: template)
    monkeypatch.setattr(source, "validate_inventory", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "validate_foundation_spec", lambda v: v["content_hash"])
    monkeypatch.setattr(source, "authenticate_preparation", lambda f, p: locks[str(p)])
    plan = screen_module.command_plan(screen)
    jobs = {r["task_id"]: str(21748718+i) for i, r in enumerate(source.screen_task_graph())}
    commands = {}
    for row in plan["commands"]:
        command = list(row["command"])
        for name, job in jobs.items():
            command = [x.replace("${JOB_" + name + "}", job) for x in command]
        commands[row["task_id"]] = command
    ledger = build_submission_ledger(campaign_spec_sha256=screen["content_hash"], dry_run=False,
                                     jobs=jobs, commands=commands)
    write_immutable_json(root / "submission_ledger.json", ledger)
    for row in source.screen_task_graph():
        name = row["task_id"]
        if name == "select":
            value, paths = selection, ["selection_lock.json"]
        elif name == "preflight":
            value, paths = profile, ["runtime_profile.json", "screen_split.json"]
        elif name == "complete":
            value, paths = done, ["screen_complete.json"]
        else:
            path = name + "/payload.json"
            write_immutable_json(root / path, {"test_fixture": name})
            value, paths = {"checkpoint": path}, [path]
        report = parent_artifact("SALIENCE_SCREEN_TASK", task_id=name,
            screen_sha256=screen["content_hash"], source_commit=screen["source_commit"],
            result=value, outputs=[dict(path=p, sha256=sha256_file(root / p)) for p in paths],
            final_test_accessed=False)
        write_immutable_json(root / "tasks" / (name + ".json"), report)
    launch = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=tmp_path / "launch",
        campaign_root=tmp_path / "campaign", project=tmp_path / "new_project", source_commit="a"*40)
    return launch, foundation, root

def test_completed_source_import_reuses_only_formula_and_authenticates_all_bytes(imported_source):
    launch, foundation, root = imported_source
    record, actual = source.build_import(launch)
    assert actual == foundation and record["models_imported"] == []
    assert record["salience_formula_only"] and record["assignments_imported"] is False
    assert actual["inputs"]["capacity"] == 320
    assert record["consumer_commit"] != record["producer_commit"]
    assert record["selected_candidate"] == "SALIENCE_PT_LINEAR"
    assert source.validate_import(record, deep=True) == record["content_hash"]
    assert record["parent_job_id"] == "21748725"
    assert Path(record["foundation_root"]).parent == root.parent / "shared_maps"
    assert not (root / "production").exists()
    assert not any("continuation" in name for name in record)
    (root / "selection_lock.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        source.validate_import(record)


def test_old_inventory_cannot_be_relabelled_as_dzfix(imported_source):
    launch, _, root = imported_source
    inventory = parent_artifact("INVENTORY", old_snapshot=True)
    write_immutable_json(root / "old_inventory.json", inventory)
    wrong = rehash(launch, inventory_path=str(root / "old_inventory.json"), inventory_sha256=inventory["content_hash"])
    with pytest.raises(ValueError, match="snapshot"):
        source.build_import(wrong)


@pytest.mark.parametrize("maximum,capacity", [(3, 16), (225, 240), (311, 320), (320, 320), (321, 336)])
def test_population_capacity_uses_inventory_round_up_not_a_fixed_snapshot(imported_source, maximum, capacity):
    _, _, root = imported_source
    screen = load_json(root / "screen_spec.json")
    inventory = rehash(load_json(root / "inventory.json"), files=[
        {"max_selected_particles": {"hlt": maximum, "offline": maximum - 1}},
    ])
    rows = screen["candidates"] + [{"foundation_root": screen["bottleneck_root"]}]
    for row in rows:
        path = Path(row["foundation_root"]) / "foundation_spec.json"
        foundation = rehash(load_json(path), inventory=inventory, inputs=input_contract(capacity=capacity))
        path.unlink()
        write_immutable_json(path, foundation)
        row["foundation_sha256"] = foundation["content_hash"]
    screen["bottleneck_sha256"] = rows[-1]["foundation_sha256"]
    source._population(screen, inventory)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("change", [
    {"capacity": 240}, {"capacity": 304}, {"capacity": 336},
    {"truncation": "allowed"}, {"class_count": 15},
])
def test_all_foundations_reject_wrong_capacity_or_input_semantics(imported_source, index, change):
    _, _, root = imported_source
    screen = load_json(root / "screen_spec.json")
    inventory = load_json(root / "inventory.json")
    rows = screen["candidates"] + [{"foundation_root": screen["bottleneck_root"]}]
    path = Path(rows[index]["foundation_root"]) / "foundation_spec.json"
    foundation = load_json(path)
    foundation = rehash(foundation, inputs=rehash(foundation["inputs"], **change))
    path.unlink()
    write_immutable_json(path, foundation)
    if index < 3:
        rows[index]["foundation_sha256"] = foundation["content_hash"]
    else:
        screen["bottleneck_sha256"] = foundation["content_hash"]
    with pytest.raises(ValueError, match="inventory-derived capacity=320") as error:
        source._population(screen, inventory)
    assert str(path) in str(error.value)


def test_partial_completion_is_not_a_ready_dependency(imported_source):
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        source.build_import(launch)


def test_pending_screen_can_be_queued_without_any_completion_or_production_preview(imported_source):
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    (root / "tasks/complete.json").unlink()
    pending = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=root.parent / "pending_launch",
        campaign_root=root.parent / "pending_campaign", project=Path(launch["project_dir"]),
        source_commit=launch["source_commit"])
    assert pending["parent_task_id"] == "complete" and pending["parent_job_id"] == "21748725"
    command = scheduler.launcher_plan(pending, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:21748725" in command
    assert not (root / "production").exists()
    assert not Path(pending["campaign_root"]).exists()
    assert not (root.parent / "pending_launch/submissions_after_matching").exists()


def test_parent_job_is_read_from_ledger_not_hardcoded(imported_source):
    launch, _, root = imported_source
    path = root / "submission_ledger.json"
    ledger = load_json(path)
    ledger["jobs"]["complete"] = "98765432"
    path.unlink()
    write_immutable_json(path, rehash(ledger))
    second = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=root.parent / "another_launch",
        campaign_root=root.parent / "another_campaign", project=Path(launch["project_dir"]),
        source_commit=launch["source_commit"])
    (root / "screen_complete.json").unlink()
    command = scheduler.launcher_plan(second, "after_matching")["commands"][0]["command"]
    assert "--dependency=afterok:98765432" in command
    with pytest.raises(ValueError, match="lineage"):
        source.validate_launch(launch)  # An existing launch may not silently retarget.


def test_cli_and_helper_expose_direct_screen_boundary(imported_source, monkeypatch, capsys):
    import runpy
    import sys
    launch, _, root = imported_source
    (root / "screen_complete.json").unlink()
    repo = Path(__file__).resolve().parents[1]
    cli = repo / "scripts/jetclass2_concat_k2.py"
    monkeypatch.setattr(sys, "argv", [str(cli), "schedule", "--spec",
                                     str(Path(launch["launch_root"]) / "launch_spec.json")])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(cli), run_name="__main__")
    assert result.value.code == 0
    printed = capsys.readouterr().out
    assert "--dependency=afterok:21748725" in printed
    assert "Initial short-job partition: tier3" in printed
    assert "Training: tier3 only, 96h requested, 95h maximum projected fit" in printed
    assert "tier3 <-> debug" in printed
    assert str(root / "screen_spec.json") in printed
    assert not (Path(launch["launch_root"]) / "submissions_after_matching/submission_ledger.json").exists()
    helper = (repo / "scripts/queue_jetclass2_concat_k2.sh").read_text()
    assert "jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json" in helper
    assert '--screen-spec "${SCREEN_SPEC}"' in helper
    assert "CONT_SPEC" not in helper and "21741416" not in helper



def test_old_tier3_screen_is_not_accepted_as_replacement(imported_source):
    launch, _, root = imported_source
    path = root / "screen_spec.json"
    value = rehash(load_json(path), schema_version=1,
                   contract="JETCLASS2_DELPHES_SALIENCE_SCREEN_SPEC/v1")
    path.unlink()
    write_immutable_json(path, value)
    with pytest.raises(ValueError, match="debug screen v2"):
        source._parent(launch)


def republish_screen_result(root, task, filename, value):
    """Make internally hashed synthetic evidence, to test semantic checks too."""
    path = root / filename
    path.unlink()
    write_immutable_json(path, value)
    report_path = root / "tasks" / (task + ".json")
    report = load_json(report_path)
    report["result"] = value
    for output in report["outputs"]:
        if output["path"] == filename:
            output["sha256"] = sha256_file(path)
    report_path.unlink()
    write_immutable_json(report_path, rehash(report))


@pytest.mark.parametrize("fault", ["unregistered_root", "context_control", "wrong_hash"])
def test_selection_must_name_registered_candidate_foundation(imported_source, fault):
    launch, _, root = imported_source
    selection = load_json(root / "selection_lock.json")
    if fault == "unregistered_root":
        selection["winner_foundation_root"] = str(root / "guessed_continuation/foundation")
    elif fault == "context_control":
        selection["winner"] = "BOTTLENECK_CONTEXT"
    else:
        selection["winner_foundation_sha256"] = "f" * 64
    selection = rehash(selection)
    republish_screen_result(root, "select", "selection_lock.json", selection)
    complete = rehash(load_json(root / "screen_complete.json"), selection_lock_sha256=selection["content_hash"])
    republish_screen_result(root, "complete", "screen_complete.json", complete)
    with pytest.raises(ValueError, match="registered"):
        source.build_import(launch)


def test_completion_file_without_task_attestation_is_not_ready(imported_source):
    launch, _, root = imported_source
    (root / "tasks/complete.json").unlink()
    with pytest.raises(ValueError, match="receipts"):
        source.build_import(launch)


def test_wrong_screen_completion_is_not_reusable(imported_source):
    launch, _, root = imported_source
    complete = rehash(load_json(root / "screen_complete.json"), screen_sha256="f" * 64)
    republish_screen_result(root, "complete", "screen_complete.json", complete)
    with pytest.raises(ValueError, match="completion"):
        source.build_import(launch)


def test_debug_screen_profile_cannot_redirect_consumer_or_change_producer_site(imported_source, monkeypatch):
    launch, _, root = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    spec = chain.create(launch=launch)
    assert load_json(root / "runtime_profile.json")["execution_site"]["partition"] == "tier3"
    assert spec["execution_site"]["partition"] == "tier3"
    assert all("--partition=tier3" in r["command"] for r in scheduler.plan(spec, "full")["commands"])
    profile = rehash(load_json(root / "runtime_profile.json"),
                     execution_site=source.execution_site("sporc_a100_debug"))
    republish_screen_result(root, "preflight", "runtime_profile.json", profile)
    with pytest.raises(ValueError, match="runtime/production site"):
        source.build_import(launch)


def test_registration_change_cannot_reuse_acceptance(imported_source, monkeypatch):
    launch, _, _ = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    spec = chain.create(launch=launch)
    assert chain.validate_campaign(spec) == spec["content_hash"]
    assert chain.create(launch=launch) == spec
    changed = deepcopy(spec)
    changed["resources"]["train"]["cpus"] = 16
    with pytest.raises(ValueError, match="registration"):
        chain.validate_campaign(rehash(changed))


def test_launcher_automates_gate_then_science_without_resubmitting_parent(imported_source, monkeypatch):
    launch, _, _ = imported_source
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    def authenticate(s, name, **kw):
        write_immutable_json(Path(s["launch_root"])/"execution"/name/"300.json",
                             chain.artifact("EXECUTION_RECORD", test_only=True))
        return "300"
    monkeypatch.setattr(scheduler, "authenticate_job", authenticate)
    original_source = source.assignment_source()
    calls = []
    def sbatch(c, **kw):
        assert c[0] == "sbatch"  # Never scancel/scontrol update or another queue family.
        calls.append(c)
        return SimpleNamespace(stdout=str(1000+len(calls)))
    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    # Provenance hashing uses git; keep its subprocess unaffected by fake sbatch.
    monkeypatch.setattr(source, "assignment_source", lambda: original_source)
    result = scheduler.run_launcher(launch, "after_matching")
    assert len(calls) == 8  # Seven preparation/gate tasks and the dependent launcher.
    assert "--dependency=afterok:1007" in calls[-1]
    monkeypatch.setattr(runtime, "science_gate", lambda s: {"passed": True})
    result = scheduler.run_launcher(launch, "after_gate")
    assert result["result"]["science_tasks"] == 17 and len(calls) == 25
    assert all("--partition=tier3" in c for c in calls)
