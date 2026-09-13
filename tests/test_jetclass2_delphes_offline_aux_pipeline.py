from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
import uproot

from test_jetclass2_delphes import root_file
from test_jetclass2_delphes_offline_aux_math import Toy
from hlt_classification.jetclass2_delphes.inventory import build_inventory
from hlt_classification.jetclass2_delphes.splits import build_splits
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.offline_aux import roles, banks, cache, training, reporting, campaign, submission
from hlt_classification.jetclass2_delphes.offline_aux.contracts import artifact, load_json, write_immutable_json, sha256_file
from hlt_classification.jetclass2_delphes.offline_aux.normalization import fit
from hlt_classification.jetclass2_delphes.offline_aux.model import create_model, state_hash


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("aux_native")
    data = root / "data"
    for source in ("train_higgs2p", "train_qcd"):
        for i in range(18):
            root_file(data / source / f"ntuple_{i}.root")
    inventory = build_inventory(data)
    reservoirs = build_splits(inventory)
    registry = build_registry(data, inventory, reservoirs, training_sizes=(33,), validation_size=22, test_size=11)
    profile = select_profile(registry, inventory, "TRAIN_33")
    split = roles.build_roles(data, inventory, profile, root / "roles", production=False, select_rows=11)
    return root, data, inventory, profile, split


def test_native_roles_targets_cache_and_train(prepared, monkeypatch):
    root, data, inventory, profile, split = prepared
    metadata = {r: roles.load_role(root / "roles", split, r) for r in roles.ROWS}
    ids = {r: {bytes(i) for i in m["identities"]} for r, m in metadata.items()}
    assert not ids["TRAIN"] & ids["VAL_SELECT"] and not ids["VAL_SELECT"] & ids["VAL_REPORT"]
    for role in ("VAL_REPORT", "final_test"):
        with pytest.raises(PermissionError):
            list(roles.read_rows(data, inventory, split, metadata["TRAIN"], role=role, include_offline=True))
    with pytest.raises(PermissionError, match="capability"):
        list(roles.read_rows(data, inventory, split, metadata["VAL_REPORT"], role="VAL_SELECT", include_offline=True))
    # HLT reader cannot request offline branches, even while aux targets exist.
    original = uproot.behaviors.TBranch.HasBranches.arrays
    def hlt_only(self, expressions=None, *a, **kw):
        assert not any(x.startswith("part_") for x in expressions)
        return original(self, expressions, *a, **kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", hlt_only)
    caches = {r: cache.prepare(data, inventory, split, metadata[r], role=r, workers=1, budget_bytes=2**30)
              for r in ("TRAIN", "VAL_SELECT")}
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", original)
    a = banks.build_shard(data, inventory, split, metadata["TRAIN"], root=root / "a", role="TRAIN", shard=0, workers=1)
    b = banks.build_shard(data, inventory, split, metadata["TRAIN"], root=root / "b", role="TRAIN", shard=0, workers=2)
    assert a == b  # exact array bytes and semantic manifests, not just float closeness
    bank, values, mask, extra = banks.load_targets(root / "a", split, metadata["TRAIN"], "TRAIN")
    assert values.shape == (33, 28) and extra["hlt_pt"].shape == (33,)
    bad = copy.deepcopy(metadata["TRAIN"]); bad["identities"] = bad["identities"][::-1]
    with pytest.raises(ValueError, match="join"):
        banks.load_targets(root / "a", split, bad, "TRAIN")
    normalizer = fit(values, mask, np.arange(33), role="TRAIN", train_bank_sha256=bank["content_hash"])
    model = create_model("BOTH", "DISCOVERY", factory=Toy)
    node = dict(node_id="LOCAL_SYNTHETIC_ONLY", arm="BOTH", replicate="DISCOVERY", **{"lambda": [3, 10]})
    report, state = training.train(model, caches["TRAIN"], caches["VAL_SELECT"], values, mask, normalizer,
                                   node=node, device="cpu", acceptance=True)
    assert report["acceptance_only"] and report["completed_passes"] == 1
    assert report["selected_state_sha256"] == state_hash(state) == state_hash(model.state_dict())
    assert not list(root.rglob("*.pt"))  # kernel writes neither optimizer nor checkpoint
    with pytest.raises(ValueError, match="population"):
        training.train(model, caches["TRAIN"], caches["VAL_SELECT"], values, mask, normalizer, node=node, device="cpu")
    with pytest.raises(MemoryError):
        cache.prepare(data, inventory, split, metadata["TRAIN"], role="TRAIN", workers=2, budget_bytes=1)


def test_role_replay_metadata_only(prepared, monkeypatch):
    root, data, inventory, profile, split = prepared
    original = uproot.behaviors.TBranch.HasBranches.arrays
    def scalars(self, expressions=None, *a, **kw):
        assert expressions == ["jet_label", "hlt_matched"]
        return original(self, expressions, *a, **kw)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", scalars)
    replay = roles.build_roles(data, inventory, profile, root / "roles_replay", production=False, select_rows=11)
    assert replay == split
    with pytest.raises(ValueError, match="exact frozen"):
        roles.authenticate_profile(inventory, profile)


def test_weighted_metrics_exact_repetition():
    rng = np.random.default_rng(99)
    y = np.repeat(np.arange(11), 15)
    raw = rng.integers(1, 5, size=(len(y), 11))
    p = (raw / raw.sum(1, keepdims=True)).astype(np.float32)
    weights = rng.integers(0, 4, size=len(y))
    fast = reporting.WeightedMetrics(y, p).evaluate(weights)
    expanded = evaluate_probabilities(np.repeat(y, weights), np.repeat(p, weights, axis=0))
    for k in ("accuracy", "macro_ovr_auc", "cross_entropy", "macro_r50"):
        assert fast[k] == pytest.approx(expanded[k], rel=1e-12)
    assert reporting.flatten_metrics(fast) == pytest.approx(reporting.flatten_metrics(expanded))
    weights[y == 3] = 0
    assert reporting.WeightedMetrics(y, p).evaluate(weights) is None
    files = np.arange(len(y))%7
    assert np.array_equal(reporting.bootstrap_draw(files, 12), reporting.bootstrap_draw(files, 12))


def metrics(auc=.5):
    return dict(macro_ovr_auc=auc, cross_entropy=1., macro_mean_log_qcd_rejection_at_50pct_signal=None)


def test_selection_small_gain_and_patience():
    state = training.Selection()
    for epoch in range(1, 61):
        better, stop = state.observe(metrics(.5+epoch*1e-7), epoch, epoch*1954)
        assert better  # retained even though each is below significant delta
        assert stop == (epoch == 60)
    assert state.last_significant == 1
    assert state.observe(metrics(.6), 61, 61*1954) == (True, False)


def test_four_arms_and_frozen_locks():
    nodes = campaign.discovery_nodes()
    assert len(nodes) == 10
    models = {n["node_id"]: dict(node=n, validation=metrics(), selected_update=10) for n in nodes}
    calibration = dict(normalizer=artifact("NORMALIZATION", hlt_pt_edges=[1, 2, 3, 4]), preparation_lock_sha256="n"*64)
    config = campaign.configuration_lock({"content_hash": "s"}, {"content_hash": "r"}, models, **calibration)
    assert all(v == [1, 10] for v in config["selected_lambda"].values())
    assert len(config["models"]) == 10
    with pytest.raises(ValueError, match="ten"):
        campaign.configuration_lock({"content_hash": "s"}, {"content_hash": "r"}, dict(list(models.items())[:-1]), **calibration)
    confirmation = campaign.confirmation_nodes(config)
    assert len(confirmation) == 12 and all(n["replicate"] != "DISCOVERY" for n in confirmation)
    m = {n["node_id"]: dict(node=n) for n in confirmation}
    lock = campaign.reporting_lock({"content_hash": "s"}, {"content_hash": "r"}, config, m)
    roles.authorize_role("VAL_REPORT", {"content_hash": "r"}, lock)
    with pytest.raises(ValueError, match="twelve"):
        campaign.reporting_lock({"content_hash": "s"}, {"content_hash": "r"}, config, dict(list(m.items())[:-1]))
    graph = campaign.task_graph("CONFIRMATION", configuration=config)
    assert len([t for t in graph if t["kind"] == "train"]) == 12
    assert len([t for t in graph if t["kind"] == "bootstrap"]) == 20
    assert all("reporting_lock" in t["dependencies"] for t in graph if t["kind"] in {"target", "evaluate"})


def test_exact_submission_journal_and_no_foreign_mutations(tmp_path, monkeypatch):
    stage = artifact("STAGE_SPEC", root=str(tmp_path / "stage"), name="GATE", tasks=campaign.task_graph("GATE"),
                     resources=dict(cpus=8, workers=8, memory_mb=73728))
    from hlt_classification.jetclass2_delphes.execution import execution_site
    study = dict(root=str(tmp_path), project_dir=str(tmp_path / "source"), site=execution_site("sporc_a100"))
    monkeypatch.setattr(submission, "validate_study", lambda *a, **k: None)
    monkeypatch.setattr(submission, "validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(submission.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*2**30))
    calls = []
    def run(command, **kw):
        calls.append(command)
        assert command[0] == "sbatch"
        return SimpleNamespace(stdout="12345\n")
    monkeypatch.setattr(submission.subprocess, "run", run)
    attempt = Path(stage["root"]) / "attempts" / "initial"
    dry = submission.submit(stage, study, attempt)
    assert dry["dry_run"] and not calls
    with pytest.raises(PermissionError):
        submission.submit(stage, study, attempt, execute=True, phrase="wrong")
    phrase = "AUTHORIZE JC2 OFFLINE AUX GATE EXACT STAGE"
    live = submission.submit(stage, study, attempt, execute=True, phrase=phrase)
    assert live["jobs"] == {"sample": "12345"} and len(calls) == 1
    assert submission.submit(stage, study, attempt, execute=True, phrase=phrase) == live
    assert len(calls) == 1
    assert "--partition=tier3" in calls[0] and "--no-requeue" in calls[0]
    assert not any("--gres" in v for v in calls[0])


def test_ambiguous_submission_refuses_retry(tmp_path, monkeypatch):
    stage = artifact("STAGE_SPEC", root=str(tmp_path / "stage"), name="GATE", tasks=campaign.task_graph("GATE"),
                     resources=dict(cpus=8, workers=8, memory_mb=73728))
    from hlt_classification.jetclass2_delphes.execution import execution_site
    study = dict(root=str(tmp_path), project_dir=str(tmp_path / "source"), site=execution_site("sporc_a100"))
    monkeypatch.setattr(submission, "validate_study", lambda *a, **k: None)
    monkeypatch.setattr(submission, "validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(submission.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*2**30))
    calls = []
    def lost(command, **kwargs):
        calls.append(command)
        raise RuntimeError("transport lost after acceptance")
    monkeypatch.setattr(submission.subprocess, "run", lost)
    attempt = Path(stage["root"]) / "attempts" / "initial"
    submission.submit(stage, study, attempt)
    with pytest.raises(RuntimeError):
        submission.submit(stage, study, attempt, execute=True, phrase="AUTHORIZE JC2 OFFLINE AUX GATE EXACT STAGE")
    with pytest.raises(PermissionError, match="acknowledgement"):
        submission.submit(stage, study, attempt, execute=True, phrase="AUTHORIZE JC2 OFFLINE AUX GATE EXACT STAGE")
    assert len(calls) == 1


def test_recovery_never_cancels_and_checks_all_old_attempts(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes.offline_aux import recovery
    stage = artifact("STAGE_SPEC", root=str(tmp_path / "stage"), name="GATE", tasks=campaign.task_graph("GATE"),
                     resources=dict(cpus=8, workers=8, memory_mb=73728))
    from hlt_classification.jetclass2_delphes.execution import execution_site
    study = dict(root=str(tmp_path), project_dir=str(tmp_path / "source"), site=execution_site("sporc_a100"))
    monkeypatch.setattr(recovery, "validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(submission, "validate_stage", lambda *a, **k: None)
    ledger = artifact("SUBMISSION_LEDGER", stage_sha256=stage["content_hash"], dry_run=False, jobs={"sample": "777"})
    write_immutable_json(Path(stage["root"]) / "attempts" / "initial" / "submission_ledger.json", ledger)
    calls = []
    state = ["RUNNING"]
    def accounting(command, **kwargs):
        calls.append(command)
        assert command[0] == "sacct" and command[command.index("-j")+1] == "777"
        return SimpleNamespace(stdout="777|"+state[0]+"\n")
    monkeypatch.setattr(submission.subprocess, "run", accounting)
    root = Path(stage["root"]) / "attempts" / "retry1"
    with pytest.raises(PermissionError, match="active jobs"):
        recovery.prepare(stage, study, root)
    assert not root.exists()
    state[0] = "FAILED"
    record = recovery.prepare(stage, study, root)
    assert record["restart_from_zero"] and record["jobs_cancelled"] == []
    assert record["remaining"] == ["sample"]
    assert all(c[0] == "sacct" for c in calls)


def test_bootstrap_missing_and_censored_draws_are_not_redrawn():
    draws = [dict(draw=i, contrasts=None) for i in range(1000)]
    intervals = reporting.bootstrap_intervals(draws)
    assert len(intervals) == 6
    assert intervals["BOTH_minus_CE"]["macro_r50"] == dict(valid_draws=0, interval=None)
    y = np.repeat(np.arange(11), 3)
    p = np.eye(11, dtype=np.float32)[y]
    m = reporting.WeightedMetrics(y, p).evaluate(np.ones(len(y)))
    assert m["macro_r50"] is None and m["macro_ovr_auc"] == 1


def test_cpu_fixture_cannot_certify_real_acceptance():
    from hlt_classification.jetclass2_delphes.offline_aux.execution import validate_acceptance
    from hlt_classification.jetclass2_delphes.offline_aux.contracts import seed_register
    assert len({s for r in seed_register().values() for s in r.values()}) == 432
    with pytest.raises((KeyError, ValueError)):
        validate_acceptance(artifact("EXECUTION_ACCEPTANCE", passed=True), {"content_hash": "a"*64})


def test_real_frozen_profile_can_create_only_first_unmeasured_gate(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    if not (repository / "artifacts/jetclass2_delphes_scaling_splits_20260911_v1/profiles/TRAIN_500K.json").is_file():
        pytest.skip("Optional frozen local manifests are not shipped as source code")
    inv = load_json(repository / "artifacts/jetclass2_delphes_20260910_provenance_v1/inventory.json")
    profile = load_json(repository / "artifacts/jetclass2_delphes_scaling_splits_20260911_v1/profiles/TRAIN_500K.json")
    checked = []
    monkeypatch.setattr(campaign, "_source", lambda project, commit: checked.append((project, commit)))
    raw = tmp_path / "raw"; raw.mkdir()
    study = campaign.create_study(root=tmp_path / "aux", data_root=raw, inventory=inv, profile=profile,
                                  project_dir=repository, source_commit="a"*40)
    assert campaign.validate_study(study) == study["content_hash"] and checked
    stage = campaign.create_stage(study, "GATE")
    plan = submission.command_plan(stage, study, Path(stage["root"])/"attempts"/"initial")
    assert len(plan["commands"]) == 1
    assert "--job-name=jc2aux_sample" in plan["commands"][0]["command"]
    with pytest.raises(FileNotFoundError):
        campaign.create_stage(study, "PREPARE")  # missing real GATE receipt
    with pytest.raises(FileNotFoundError):
        campaign.create_stage(study, "DISCOVERY")
    with pytest.raises(ValueError, match="semantics"):
        bad = {k: v for k, v in study.items() if k not in {"contract", "schema_version", "content_hash", "final_test_accessed"}}
        bad["fit_count"] = 21
        campaign.validate_study(artifact("STUDY_SPEC", **bad))


def test_cli_imports_only_readiness_metadata_without_waiting_for_jobs(prepared, monkeypatch, capsys):
    import importlib.util
    import sys
    from hlt_classification.jetclass2_delphes.contracts import artifact as native_artifact
    root, data, inv, profile, _ = prepared
    ready = native_artifact("READINESS_SPEC", foundation=dict(inventory=inv, splits=profile),
                            data_root=str(data), final_test_accessed=False)
    write_immutable_json(root / "readiness.json", ready)
    received = []
    def create(**kwargs):
        received.append(kwargs)
        return dict(metadata_imported=True)
    monkeypatch.setattr(campaign, "create_study", create)
    path = Path(__file__).resolve().parents[1] / "scripts/jetclass2_delphes_offline_aux.py"
    spec = importlib.util.spec_from_file_location("auxiliary_cli_test", path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", [str(path), "create", "--root", str(root / "new_study"),
        "--project-dir", str(path.parents[1]), "--source-commit", "a"*40,
        "--readiness-spec", str(root / "readiness.json")])
    assert cli.main() == 0
    assert received[0]["inventory"] == inv and received[0]["profile"] == profile
    assert received[0]["data_root"] == data
    assert "metadata_imported" in capsys.readouterr().out
