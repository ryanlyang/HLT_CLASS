"""K2-only preparation reuse: real synthetic maps, immutable lineage, fresh GPU gate."""
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as chain, concat_k2_data as data,
    concat_k2_preparation_import as reuse, concat_k2_runtime as runtime,
    concat_k2_source as source, concat_k2_submit as scheduler,
)
from hlt_classification.jetclass2_delphes.contracts import artifact as base_artifact
from hlt_classification.jetclass2_delphes.salience_foundation import build_foundation_spec
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES
from test_jetclass2_delphes import snapshot
from test_jetclass2_dzfix_fusion_chain import rehash
from test_jetclass2_concat_k2_source import imported_source


def rewrite(path, value):
    path.unlink()
    write_immutable_json(path, value)


@pytest.fixture
def completed_donor(snapshot, tmp_path, monkeypatch):
    raw, inventory, reservoirs = snapshot
    registry = build_registry(raw, inventory, reservoirs, training_sizes=(11,),
                              validation_size=11, test_size=11, step_size=3)
    splits = select_profile(registry, inventory, "TRAIN_11")
    parent = build_foundation_spec(inventory, splits, CANDIDATES[0])
    counts = splits["role_counts"]
    monkeypatch.setattr(chain, "COUNTS", counts)
    monkeypatch.setattr(reuse, "COUNTS", counts)
    monkeypatch.setattr(reuse, "SPLIT_PROFILE", "TRAIN_11")
    # The external selected salience screen is covered by the source test suite.
    # Here exercise real ROOT matching, all map validators and task receipts.
    monkeypatch.setattr(source, "validate_import", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "validate_import", lambda *a, **k: None)
    monkeypatch.setattr(chain, "_source", lambda *a: None)
    parent_path = tmp_path / "parent/foundation_spec.json"
    write_immutable_json(parent_path, parent)
    source_import = chain.artifact("SOURCE_IMPORT", foundation_spec_path=str(parent_path),
        foundation_sha256=parent["content_hash"], consumer_commit="a"*40, data_root=str(raw),
        screen_spec_path=str(tmp_path / "screen/screen_spec.json"), screen_sha256="b"*64,
        parent_ledger_sha256="c"*64, parent_job_id="123")
    foundation = chain.foundation_spec(parent, source_import["content_hash"])
    root = tmp_path / "old_k2"
    registered = dict(chain.registration(), split_profile="TRAIN_11")
    donor = base_artifact("CONCAT_K2_CAMPAIGN_SPEC", version=1, **registered,
        source_commit="a"*40, project_dir=str(tmp_path / "old_project"), campaign_root=str(root),
        data_root=str(raw), source_import=source_import, foundation=foundation,
        tasks=chain.task_graph(foundation))
    write_immutable_json(root / "campaign_spec.json", donor)
    for task in donor["tasks"]:
        name = task["task_id"]
        directory = root / "outputs" / name
        directory.mkdir(parents=True)
        if name == "authenticate":
            result = {"source_import_sha256": source_import["content_hash"]}
        elif name == "matcher_acceptance":
            value = data.matcher_acceptance(donor)
            write_immutable_json(directory / "matcher_acceptance.json", value)
            result = {"matcher_acceptance_sha256": value["content_hash"]}
        elif task["kind"] == "assign":
            value = data.assignment(donor, task["file_index"], directory)
            result = {"assignment_sha256": value["content_hash"]}
        else:
            assert name == "foundation_lock"
            value = data.foundation_lock(donor)
            write_immutable_json(directory / "foundation_lock.json", value)
            result = {"foundation_lock_sha256": value["content_hash"]}
        write_immutable_json(directory / "result.json", result)
        report = chain.artifact("TASK_REPORT", task_id=name, campaign_sha256=donor["content_hash"],
            source_commit=donor["source_commit"], result=result, final_test_accessed=False,
            parents={p: load_json(root / "tasks" / (p+".json"))["content_hash"] for p in task["dependencies"]},
            outputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha256_file(p)) for p in directory.iterdir()])
        write_immutable_json(root / "tasks" / (name+".json"), report)
        if name == "foundation_lock":
            break
    return donor


def target_spec(donor, tmp_path, partition="debug"):
    record = reuse.describe_reuse(Path(donor["campaign_root"]) / "campaign_spec.json")
    imported = rehash(donor["source_import"], consumer_commit="d"*40)
    parent = load_json(imported["foundation_spec_path"])
    foundation = chain.foundation_spec(parent, imported["content_hash"], reuse=True)
    return chain.artifact("CAMPAIGN_SPEC", **chain.registration(partition), foundation=foundation,
        tasks=chain.task_graph(foundation, reuse=True), preparation_import=record,
        source_import=imported, source_commit="d"*40, project_dir=str(tmp_path / "new_project"),
        campaign_root=str(tmp_path / "new_k2"), data_root=donor["data_root"])


def test_independent_copy_byte_parity_donor_untouched_and_fresh_gpu_gate(completed_donor, tmp_path, monkeypatch):
    donor = completed_donor
    root = Path(donor["campaign_root"])
    before = {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    spec = target_spec(donor, tmp_path)
    chain.validate_campaign(spec, check_source=False)
    assert chain.gates(spec) == ("authenticate", "import_preparation", "foundation_lock",
                                 "partition_validation", "audit_storage", "preflight")
    assert len(scheduler.plan(spec, "science")["commands"]) == 17
    assert not any(t["kind"] in ("assign", "matcher_acceptance") for t in spec["tasks"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Matching must not rerun")
    for module in (data, runtime):
        monkeypatch.setattr(module, "assignment", forbidden)
        monkeypatch.setattr(module, "matcher_acceptance", forbidden)
    def job(s, name):
        write_immutable_json(Path(s["campaign_root"]) / "execution" / name / "456.json", {"fixture": True})
        return "456"
    monkeypatch.setattr(scheduler, "authenticate_job", job)
    for name in ("authenticate", "import_preparation", "foundation_lock"):
        with monkeypatch.context() as guarded:
            guarded.setattr(data, "DatasetReader", forbidden)
            runtime.run_task(spec, name, device="cpu")
    receipt = runtime.completed(spec, "import_preparation")
    assert receipt["result"]["matching_recomputed"] is False
    assert receipt["result"]["gpu_acceptance_imported"] is False
    output_names = {r["path"] for r in receipt["outputs"]}
    for task in spec["foundation"]["assignment_tasks"]:
        suffix = Path("outputs") / f"assign_{task['file_index']:04d}" / "assignments.npz"
        new, old = Path(spec["campaign_root"]) / suffix, root / suffix
        assert new.read_bytes() == old.read_bytes() and not new.samefile(old)
        assert suffix.as_posix() in output_names
        report, _ = data.load_assignment(spec, task["file_index"])
        assert report["donor_foundation_sha256"] == donor["foundation"]["content_hash"]
    old_lock, new_lock = data.foundation_lock(donor), data.foundation_lock(spec)
    assert {k:v for k,v in old_lock.items() if k not in ("content_hash", "foundation_sha256", "shards")} == {
        k:v for k,v in new_lock.items() if k not in ("content_hash", "foundation_sha256", "shards")}
    # Real model input arrays, all rungs, both ordinary roles, not just map hashes.
    for s in (donor, spec):
        s["resources"]["train"]["cpus"] = 1  # fixture speed only, after contract checks
    for role in ("train", "validation"):
        for coordinate in ("D100", "D075", "D050", "D025", "D000", "HLT_X1"):
            a, b = data.prepare(donor, role, coordinate), data.prepare(spec, role, coordinate)
            np.testing.assert_array_equal(a.identities, b.identities)
            for key, value in a.batch(np.arange(len(a))).items():
                np.testing.assert_array_equal(value, b.batch(np.arange(len(b)))[key])
            np.testing.assert_array_equal(a.labels, b.labels)
    assert spec["validation_partition"] == donor["validation_partition"]
    assert before == {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(PermissionError, match="GPU acceptance"):
        runtime.science_gate(spec)
    new.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        runtime.completed(spec, "import_preparation")


@pytest.mark.parametrize("partition", ["debug", "tier3"])
def test_import_dry_plan_keeps_science_and_portability(completed_donor, tmp_path, partition):
    spec = target_spec(completed_donor, tmp_path, partition)
    plan = scheduler.plan(spec, "full")
    assert len(plan["commands"]) == 23
    assert all("--partition="+partition in row["command"] for row in plan["commands"])
    assert spec["training"]["batch_size"] == 256
    assert spec["batch_probe_policy"]["order"] == [128, 256]
    assert spec["execution_policy"]["mutable_scheduler_fields"] == ["Partition"]


@pytest.mark.parametrize("fault", ["wrong_family", "nested_import", "wrong_k", "population", "missing_lock",
    "missing_receipt", "wrong_parent", "producer", "wrong_array_receipt", "unattested_report", "matcher", "shard_coverage"])
def test_reuse_rejects_incompatible_or_incomplete_donors(completed_donor, fault):
    donor = completed_donor
    root = Path(donor["campaign_root"])
    path = root / "campaign_spec.json"
    if fault in ("wrong_family", "nested_import", "wrong_k", "population"):
        if fault == "wrong_family": donor = rehash(donor, contract="JETCLASS2_DELPHES_SALIENCE_FOUNDATION_SPEC/v1")
        if fault == "nested_import": donor = rehash(donor, preparation_import={"a": 1})
        if fault == "wrong_k": donor = rehash(donor, k=1)
        if fault == "population": donor = rehash(donor, role_counts=dict(train=1, validation=1, final_test=1))
        rewrite(path, donor)
    elif fault == "missing_lock":
        (root / "outputs/foundation_lock/foundation_lock.json").unlink()
    elif fault == "missing_receipt":
        (root / "tasks/matcher_acceptance.json").unlink()
    elif fault == "wrong_parent":
        p = root / "tasks/foundation_lock.json"
        rewrite(p, rehash(load_json(p), parents={}))
    else:
        index = donor["foundation"]["assignment_tasks"][0]["file_index"]
        name = f"assign_{index:04d}"
        if fault in ("producer", "unattested_report"):
            p = root / "outputs" / name / "assignment_report.json"
            value = load_json(p)
            if fault == "producer": value["producer"] = rehash(value["producer"], file_sha256={"wrong.py": "0"*64})
            else: value["elapsed_seconds"] += 1
            rewrite(p, rehash(value))
        elif fault == "wrong_array_receipt":
            p = root / "tasks" / (name+".json")
            value = load_json(p)
            for row in value["outputs"]:
                if row["path"].endswith("assignments.npz"): row["sha256"] = "0"*64
            rewrite(p, rehash(value))
        elif fault == "matcher":
            p = root / "outputs/matcher_acceptance/matcher_acceptance.json"
            rewrite(p, rehash(load_json(p), exhaustive_cases=0))
        else:
            p = root / "outputs/foundation_lock/foundation_lock.json"
            rewrite(p, rehash(load_json(p), shards={}))
    with pytest.raises((ValueError, FileNotFoundError)):
        reuse.describe_reuse(path)


def test_copy_rejects_same_size_corruption_before_publication(completed_donor, tmp_path):
    spec = target_spec(completed_donor, tmp_path)
    root = Path(completed_donor["campaign_root"])
    path = next(root.glob("outputs/assign_*/assignments.npz"))
    value = bytearray(path.read_bytes()); value[-1] ^= 1; path.write_bytes(value)
    with pytest.raises(ValueError, match="bytes changed"):
        reuse.import_preparation(spec, tmp_path / "unused")
    assert not Path(spec["campaign_root"]).exists()


@pytest.mark.parametrize("fault", ["source", "overlap", "pinned_record"])
def test_pinned_reuse_cannot_be_retargeted(completed_donor, tmp_path, fault):
    spec = target_spec(completed_donor, tmp_path)
    record, imported, destination = spec["preparation_import"], spec["source_import"], spec["campaign_root"]
    if fault == "source": imported = rehash(imported, foundation_sha256="0"*64)
    if fault == "overlap": destination = Path(completed_donor["campaign_root"]) / "new"
    if fault == "pinned_record": record = rehash(record, donor_foundation_lock_sha256="0"*64)
    with pytest.raises(ValueError):
        reuse.validate_reuse(record, source=imported, destination=destination)


def test_materialize_is_idempotent_and_does_not_import_preflight(completed_donor, tmp_path, monkeypatch):
    spec = target_spec(completed_donor, tmp_path)
    launch = dict(preparation_import=spec["preparation_import"], registration=chain.registration("debug"),
        campaign_root=spec["campaign_root"], source_commit=spec["source_commit"],
        project_dir=spec["project_dir"], content_hash="0"*64)
    monkeypatch.setattr(source, "validate_launch", lambda *a: None)
    parent = load_json(spec["source_import"]["foundation_spec_path"])
    monkeypatch.setattr(source, "build_import", lambda *a: (spec["source_import"], parent))
    created = chain.create(launch=launch)
    assert created == chain.create(launch=launch)
    chain.validate_campaign(created, check_source=False)
    assert not (Path(created["campaign_root"]) / "outputs/preflight").exists()
    assert len(chain.gates(created)) == 6


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_portable_and_memory_fixed_fresh_donors_supported(completed_donor, version):
    donor = completed_donor
    root = Path(donor["campaign_root"])
    updated = rehash(donor, schema_version=version,
                     contract=f"JETCLASS2_DELPHES_CONCAT_K2_CAMPAIGN_SPEC/v{version}")
    rewrite(root / "campaign_spec.json", updated)
    # Publish a consistent synthetic historical receipt chain, retaining maps.
    receipts = {}
    for row in updated["tasks"]:
        name = row["task_id"]
        path = root / "tasks" / (name+".json")
        report = rehash(load_json(path), campaign_sha256=updated["content_hash"],
                        parents={p: receipts[p]["content_hash"] for p in row["dependencies"]})
        rewrite(path, report)
        receipts[name] = report
        if name == "foundation_lock":
            break
    record = reuse.describe_reuse(root / "campaign_spec.json")
    assert record["donor_spec_sha256"] == updated["content_hash"]


def test_create_launch_refuses_uncompleted_one_to_one_donor(imported_source, tmp_path):
    launch, _, screen = imported_source
    with pytest.raises(ValueError, match="contract mismatch"):
        source.create_launch(screen_spec=screen / "screen_spec.json", inventory_path=screen / "inventory.json",
            launch_root=tmp_path / "new_launch", campaign_root=tmp_path / "new_campaign",
            project=Path(launch["project_dir"]), source_commit=launch["source_commit"],
            reuse_preparation_spec=Path(load_json(screen / "selection_lock.json")["winner_foundation_root"]) / "foundation_spec.json")
    assert not (tmp_path / "new_launch").exists()


def test_launch_pins_explicit_donor_and_rejects_changed_screen(completed_donor, tmp_path, monkeypatch):
    donor = completed_donor
    prior = donor["source_import"]
    inventory = donor["foundation"]["inventory"]
    inventory_path = tmp_path / "inventory.json"
    write_immutable_json(inventory_path, inventory)
    screen = dict(content_hash=prior["screen_sha256"], screen_root=str(tmp_path / "screen"),
        data_root=donor["data_root"], project_dir=donor["project_dir"],
        bottleneck_root=str(tmp_path / "bottleneck"), candidates=[])
    ledger = dict(content_hash=prior["parent_ledger_sha256"], jobs={"complete": prior["parent_job_id"]})
    monkeypatch.setattr(source, "_parent", lambda *a: (screen, ledger))
    monkeypatch.setattr(source, "_population", lambda *a: None)
    monkeypatch.setattr(source, "_source", lambda *a: None)
    launch = source.create_launch(screen_spec=Path(prior["screen_spec_path"]), inventory_path=inventory_path,
        launch_root=tmp_path / "launch_reuse", campaign_root=tmp_path / "campaign_reuse",
        project=tmp_path / "consumer_project", source_commit="d"*40, partition="debug",
        reuse_preparation_spec=Path(donor["campaign_root"]) / "campaign_spec.json")
    assert source.validate_launch(launch) == launch["content_hash"]
    assert launch["preparation_import"]["donor_spec_sha256"] == donor["content_hash"]
    assert not Path(launch["campaign_root"]).exists()  # no workers/submission from registration
    wrong = rehash(launch, parent_job_id="999")
    with pytest.raises(ValueError):
        reuse.validate_reuse(launch["preparation_import"], launch=wrong, destination=launch["campaign_root"])
