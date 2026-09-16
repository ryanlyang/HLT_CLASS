from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from test_jetclass2_delphes import particle_values, snapshot
from test_jetclass2_delphes_salience import registered
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.jetclass2_delphes import native_concat as study
from hlt_classification.jetclass2_delphes import native_concat_data as data
from hlt_classification.jetclass2_delphes import native_concat_model as model
from hlt_classification.jetclass2_delphes import model as ordinary
from hlt_classification.jetclass2_delphes.cache import RamBlock
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes.inputs import build_inputs
from hlt_classification.jetclass2_delphes.reader import Jet, Particles
from hlt_classification.jetclass2_delphes.salience_foundation import build_foundation_spec
from hlt_classification.jetclass2_delphes.salience_learned_data import partition_codes
from hlt_classification.jetclass2_delphes.salience_learned_graph import NODE_REGISTRY
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(17, 128)

    def forward(self, x):
        return self.linear(x.transpose(1, 2))


class FakeWeaver(nn.Module):
    def __init__(self, **config):
        super().__init__()
        assert config["input_dim"] == 17 and config["num_classes"] == 11
        assert config["trim"] is False
        self.embed = Embedding()
        self.cls_token = nn.Parameter(torch.randn(1, 128))
        self.fc = nn.Linear(128, 11)

    def forward(self, features, *, v, mask):
        hidden = self.embed(features) * mask.transpose(1, 2)
        return self.fc(hidden.sum(1) / mask.sum(2) + self.cls_token)


@pytest.fixture
def fake_weaver(monkeypatch):
    monkeypatch.setattr(ordinary, "load_weaver_particle_transformer_class", lambda: FakeWeaver)


def make_cache(role="train", repeats=3):
    offsets, xs, vs, ids, labels = [0], [], [], [], []
    for i in range(11 * repeats):
        offline, hlt = particle_values(3 + i % 2), particle_values(2 + i % 4)
        offline[:, 1] += .01 * (i % 11)
        hlt[:, 1] += .02 * (i % 11)
        jet = Jet(f"{i:064x}", i % 11, Particles(hlt), Particles(offline))
        x, v = data.jet_inputs(jet, 240)
        xs.append(x); vs.append(v); offsets.append(offsets[-1] + len(x))
        ids.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8)); labels.append(jet.label)
    block = RamBlock(0, np.asarray(offsets), np.concatenate(xs), np.concatenate(vs),
                     np.asarray(ids), np.asarray(labels))
    return data.ConcatCache([block], role=role, foundation_sha256="f"*64, capacity=480)


@pytest.mark.parametrize("no,nh", [(2, 5), (5, 2), (3, 3), (240, 240)])
def test_exact_native_support_and_each_own_normalization(no, nh):
    jet = Jet("a"*64, 1, Particles(particle_values(nh)), Particles(particle_values(no)))
    x, v = data.jet_inputs(jet, 240)
    assert x.shape == (no + nh, 18)
    np.testing.assert_array_equal(x[:no, :17], build_inputs(jet.offline, capacity=240).features[:, :no].T)
    np.testing.assert_array_equal(x[no:, :17], build_inputs(jet.hlt, capacity=240).features[:, :nh].T)
    np.testing.assert_array_equal(v, np.concatenate((jet.offline.p4, jet.hlt.p4)))
    assert (x[:no, 17] == 0).all() and (x[no:, 17] == 1).all()


def test_no_truncation_and_missing_offline_rejected():
    with pytest.raises(ValueError, match="capacity overflow"):
        data.jet_inputs(Jet("a"*64, 0, Particles(particle_values()), Particles(particle_values(241))), 240)
    with pytest.raises(ValueError, match="both"):
        data.jet_inputs(Jet("a"*64, 0, Particles(particle_values()), None), 240)


def test_padding_and_shuffle_preserve_source_codes():
    cache = make_cache()
    raw = cache.batch(np.asarray([9, 1, 4]))
    assert raw["features"].shape[1:] == (18, 16)
    np.testing.assert_array_equal(raw["identities"], cache.identities[[9, 1, 4]])
    assert (raw["features"][:, 17][~raw["mask"][:, 0]] == -1).all()
    for index in (np.asarray([-1]), np.asarray([len(cache)]), np.asarray([1.5])):
        with pytest.raises(ValueError):
            cache.batch(index)


def test_native_process_cache_without_any_matching_outputs(registered):
    root, inventory, splits = registered
    foundation = build_foundation_spec(inventory, splits, "SALIENCE_PT_LINEAR")
    one = data.prepare_cache(foundation, data_root=root, role="train", workers=1, max_ram_bytes=10**8)
    two = data.prepare_cache(foundation, data_root=root, role="train", workers=2, max_ram_bytes=10**8)
    assert len(one) == 11
    for key in ("features", "vectors", "mask", "labels", "identities"):
        np.testing.assert_array_equal(one.batch(np.arange(11))[key], two.batch(np.arange(11))[key])
    with pytest.raises(PermissionError):
        data.prepare_cache(foundation, data_root=root, role="final_test", workers=1, max_ram_bytes=10**8)
    with pytest.raises(MemoryError):
        data.prepare_cache(foundation, data_root=root, role="train", workers=1, max_ram_bytes=1)


def test_matched_backbone_and_new_embedding_do_not_change_rng(fake_weaver):
    tagged = model.new_model()
    next_tagged = torch.rand(10)
    torch.manual_seed(NODE_REGISTRY["CE_SINGLE_D000"].payload()["initialization_seed"])
    reference = ordinary.DelphesParticleTransformer()
    torch.testing.assert_close(next_tagged, torch.rand(10), rtol=0, atol=0)
    for name, tensor in reference.state_dict().items():
        mapped = name.replace("mod.embed.", "mod.embed.numerical.")
        torch.testing.assert_close(tensor, tagged.state_dict()[mapped], rtol=0, atol=0)
    study.base_validate(model.node_spec(), "NATIVE_CONCAT_NODE")
    assert model.node_spec()["teacher_distribution"] is None
    assert model.node_spec()["deployable"] is False


def test_forward_source_gradients_and_zero_tag_parity(fake_weaver):
    raw = make_cache().batch(np.arange(8))
    assert all(model.zero_tag_parity(raw, device="cpu").values())
    net = model.new_model()
    logits = net(*(torch.from_numpy(raw[k]) for k in ("features", "vectors", "mask")))
    nn.functional.cross_entropy(logits, torch.from_numpy(raw["labels"])).backward()
    assert logits.shape == (8, 11)
    assert (net.mod.embed.source.weight.grad.abs().sum(1) > 0).all()
    changed = raw["features"].copy()
    changed[:, 17, 0] = .5
    with pytest.raises(ValueError, match="source codes"):
        net(torch.from_numpy(changed), torch.from_numpy(raw["vectors"]), torch.from_numpy(raw["mask"]))


def test_partition_joins_and_heldout_disjoint():
    cache = make_cache("validation", repeats=6)
    arrays = dict(identities=cache.identities, labels=cache.labels,
                  partition=partition_codes(cache.identities, cache.labels))
    subsets = study.partition_views(cache, arrays)
    assert all(set(c.labels) == set(range(11)) for c in subsets.values())
    assert not set(subsets["checkpoint"].indices) & set(subsets["report"].indices)
    altered = dict(arrays, identities=arrays["identities"][::-1])
    with pytest.raises(ValueError, match="population"):
        study.partition_views(cache, altered)


def fake_source(tmp_path):
    cache = make_cache("validation")
    metrics = evaluate_probabilities(cache.labels, np.full((len(cache), 11), 1/11, np.float32))
    foundation = {"splits": {"profile": "TRAIN_500K", "role_counts": dict(train=500_000, validation=1_000_000, final_test=1_000_000)},
                  "inputs": {"capacity": 240}, "assignment_tasks": [
                      dict(role="train", rows=100), dict(role="validation", rows=100)]}
    source = artifact("SALIENCE_LEARNED_HANDOFF_CAMPAIGN_SPEC", campaign_root=str(tmp_path / "existing"),
                      data_root=str(tmp_path / "data"), source_lock={"foundation_root": str(tmp_path / "foundation")},
                      project_dir=str(tmp_path / "old_project"), foundation=foundation)
    source_path = tmp_path / "reference.json"
    write_immutable_json(source_path, source)
    references = [dict(node=name, metrics=metrics, selected_pass=12, passes=75, task_sha256="b"*64)
                  for name in study.REFERENCES]
    evidence = dict(campaign_sha256=source["content_hash"], gate_hashes={}, partition_sha256="a"*64, references=references)
    return source_path, source, evidence


@pytest.fixture
def specification(tmp_path, monkeypatch):
    path, source, evidence = fake_source(tmp_path)
    monkeypatch.setattr(study, "_source", lambda *args: None)
    monkeypatch.setattr(study, "source_evidence", lambda path: (source, evidence, {}))
    spec = study.create(reference_spec=path, output_root=tmp_path / "concat", project=tmp_path / "project", source_commit="a"*40)
    return spec


def test_exactly_one_job_and_dry_does_not_submit(specification, monkeypatch):
    from hlt_classification.jetclass2_delphes import submission
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run invoked subprocess")
    monkeypatch.setattr(submission.subprocess, "run", forbidden)
    spec = specification
    ledger = study.submit(spec)
    assert ledger["dry_run"] and list(ledger["jobs"]) == ["fit"]
    command = study.command_plan(spec)["commands"][0]
    assert command["dependencies"] == []
    assert "--partition=tier3" in command["command"]
    assert "--gres=gpu:a100:1" in command["command"]
    assert not any("dependency=" in token for token in command["command"])
    with pytest.raises(PermissionError):
        study.submit(spec, execute=True)


def test_semantic_tamper_and_root_overlap_rejected(specification, tmp_path):
    changed = deepcopy(specification)
    changed["recipe"]["training"]["batch_size"] = 128
    changed = with_content_hash(changed)
    with pytest.raises(ValueError):
        study.validate_spec(changed)
    source = load_json(specification["reference_spec_path"])
    for name in ("campaign_root", "data_root", "project_dir"):
        with pytest.raises(ValueError, match="overlaps"):
            study._isolated(Path(source[name]) / "new", source, tmp_path / "project")


def test_live_submission_receipt_idempotence_and_ambiguous_failure(specification, monkeypatch):
    from hlt_classification.jetclass2_delphes import submission
    from types import SimpleNamespace
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345;sporc\n")
    monkeypatch.setattr(submission.subprocess, "run", sbatch)
    ledger = study.submit(specification, execute=True, authorization_phrase=study.AUTHORIZE)
    assert ledger["jobs"] == {"fit": "12345"}
    assert study.submit(specification, execute=True, authorization_phrase=study.AUTHORIZE) == ledger
    assert len(calls) == 1
    study._receipt(specification, "12345")
    with pytest.raises(PermissionError, match="receipt"):
        study._receipt(specification, "12346")


def test_pending_sbatch_intent_never_retries(specification, monkeypatch):
    from hlt_classification.jetclass2_delphes import submission
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("lost acknowledgement")
    monkeypatch.setattr(submission.subprocess, "run", ambiguous)
    with pytest.raises(RuntimeError, match="lost"):
        study.submit(specification, execute=True, authorization_phrase=study.AUTHORIZE)
    with pytest.raises(PermissionError, match="Ambiguous"):
        study.submit(specification, execute=True, authorization_phrase=study.AUTHORIZE)
    assert len(calls) == 1


def test_full_fit_selected_checkpoint_report_and_tamper_guard(specification, fake_weaver, monkeypatch):
    from hlt_classification.jetclass2_delphes import salience_learned_training as training
    # Synthetic end-to-end test uses the same kernel; no short-fit CLI exists.
    monkeypatch.setitem(training.TRAINING, "maximum_passes", 2)
    train, val = make_cache(), make_cache("validation")
    arrays = dict(identities=val.identities, labels=val.labels,
                  partition=partition_codes(val.identities, val.labels))
    parts = study.partition_views(val, arrays)
    acceptance = study.artifact("ACCEPTANCE", parents={"spec": specification["content_hash"]}, passed=True)
    root = Path(specification["campaign_root"])
    write_immutable_json(root / "acceptance.json", acceptance)
    report = study.finish_fit(specification, train, parts["checkpoint"], parts["report"], acceptance, device="cpu")
    assert report["evaluation_role"] == "V_report"
    assert report["rows"][-1]["passes"] == 2
    assert report["rows"][-1]["metrics"]["rows"] == len(parts["report"])
    monkeypatch.setattr(study, "reference_row", lambda *args: None)
    assert study.results(specification)[-1]["node"] == model.NODE_ID
    state = torch.load(root / "selected.pt", weights_only=True)
    reloaded = model.new_model()
    reloaded.load_state_dict(state, strict=True)
    assert not list(root.glob("*optimizer*"))
    (root / "selected.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        study.results(specification)


def test_installed_weaver_native_concat():
    pytest.importorskip("weaver")
    torch.set_num_threads(1)
    raw = make_cache().batch(np.arange(3))
    assert all(model.zero_tag_parity(raw, device="cpu").values())
    net = model.new_model()
    logits = net(*(torch.from_numpy(raw[k]) for k in ("features", "vectors", "mask")))
    nn.functional.cross_entropy(logits, torch.from_numpy(raw["labels"])).backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(net.mod.embed.source.weight.grad).all()


def test_worker_gates_before_fit_and_does_not_rerun_claimed_root(specification, monkeypatch):
    spec = specification
    train, validation = make_cache(), make_cache("validation")
    arrays = dict(identities=validation.identities, labels=validation.labels,
                  partition=partition_codes(validation.identities, validation.labels))
    source = dict(runtime_profile={"installed_environment": {"test": True}})
    monkeypatch.setattr(study, "validate_spec", lambda spec: (source, arrays))
    monkeypatch.setattr(study, "allocation", lambda site: ("100", 8, 131072))
    monkeypatch.setattr(study, "installed_environment", lambda: {"test": True})
    monkeypatch.setattr(study, "_receipt", lambda *args: None)
    monkeypatch.setattr(study, "prepare_cache", lambda *args, **kw: train if kw["role"] == "train" else validation)
    calls = []
    def probe(spec, tr, val, checkpoint):
        calls.append("accept")
        assert len(checkpoint) < len(val)
        return study.artifact("ACCEPTANCE", parents={"spec": spec["content_hash"]},
                              passed=True, projected_loop_seconds=1)
    def fit(spec, tr, checkpoint, report, acceptance, **kwargs):
        calls.append("fit")
        assert acceptance["runtime_budget_passed"]
        assert not set(checkpoint.indices) & set(report.indices)
        return {"test": "finished"}
    monkeypatch.setattr(study, "accept", probe)
    monkeypatch.setattr(study, "finish_fit", fit)
    assert study.run(spec) == {"test": "finished"}
    assert calls == ["accept", "fit"]
    with pytest.raises(FileExistsError):
        study.run(spec)
    assert calls == ["accept", "fit"]


def test_runtime_rejection_cannot_start_science(specification, monkeypatch):
    cache = make_cache("validation")
    arrays = dict(identities=cache.identities, labels=cache.labels,
                  partition=partition_codes(cache.identities, cache.labels))
    monkeypatch.setattr(study, "validate_spec", lambda spec: ({"runtime_profile": {"installed_environment": {}}}, arrays))
    monkeypatch.setattr(study, "allocation", lambda site: ("100", 8, 131072))
    monkeypatch.setattr(study, "installed_environment", lambda: {})
    monkeypatch.setattr(study, "_receipt", lambda *args: None)
    monkeypatch.setattr(study, "prepare_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(study, "accept", lambda *args: study.artifact("ACCEPTANCE", projected_loop_seconds=10**10))
    monkeypatch.setattr(study, "finish_fit", lambda *a, **kw: pytest.fail("Science started after failed gate"))
    with pytest.raises(RuntimeError, match="walltime"):
        study.run(specification)
    assert not (Path(specification["campaign_root"]) / "acceptance.json").exists()
    assert (Path(specification["campaign_root"]) / "runtime_rejection.json").exists()


def test_resource_bound_and_empty_existing_root_fail_before_publication(tmp_path, monkeypatch):
    path, source, evidence = fake_source(tmp_path)
    monkeypatch.setattr(study, "_source", lambda *args: None)
    monkeypatch.setattr(study, "source_evidence", lambda path: (source, evidence, {}))
    args = dict(reference_spec=path, output_root=tmp_path / "concat", project=tmp_path / "project", source_commit="a"*40)
    with monkeypatch.context() as scoped:
        scoped.setattr(study, "memory_bounds", lambda *a: dict(train=2**30, validation=2**30))
        with pytest.raises(MemoryError):
            study.create(**args, memory_mb=1024)
    assert not (tmp_path / "concat").exists()
    with pytest.raises(ValueError):
        study.create(**args, workers=9, cpus=8)
    (tmp_path / "concat").mkdir()
    with pytest.raises(FileExistsError):
        study.create(**args)


def test_reference_reader_authenticates_bytes_and_uses_report_role(tmp_path):
    from hlt_classification.jetclass2_delphes.salience_learned_contracts import artifact as learned_artifact
    source = learned_artifact("CAMPAIGN_SPEC", campaign_root=str(tmp_path), source_commit="a"*40,
                              foundation={"content_hash": "f"*64}, tasks=[{"task_id": "train_M0HLT"}])
    metrics = evaluate_probabilities(np.arange(11), np.full((11, 11), 1/11, np.float32))
    training = learned_artifact("TRAINING_REPORT", node=NODE_REGISTRY["M0HLT"].payload(),
                                parents={"campaign_spec": source["content_hash"], "foundation": "f"*64},
                                source_commit="a"*40, scientific_fit=True, final_test_accessed=False,
                                passes=75, selected_pass=12, validation={"accuracy": -1})
    diagnostic = learned_artifact("DIAGNOSTIC", parents={"training_report": training["content_hash"]},
                                  report_metrics=metrics, final_test_accessed=False)
    write_immutable_json(tmp_path / "fit.json", training)
    write_immutable_json(tmp_path / "diagnostic.json", diagnostic)
    task = learned_artifact("TASK_REPORT", campaign_sha256=source["content_hash"], source_commit="a"*40,
                            task_id="train_M0HLT", final_test_accessed=False,
                            result={"training_report": "fit.json", "training_report_sha256": training["content_hash"],
                                    "diagnostic": "diagnostic.json", "diagnostic_sha256": diagnostic["content_hash"]},
                            outputs=[dict(path=p, sha256=sha256_file(tmp_path / p)) for p in ("fit.json", "diagnostic.json")])
    write_immutable_json(tmp_path / "tasks/train_M0HLT.json", task)
    assert study.reference_row(source, "M0HLT")["metrics"] == metrics
    (tmp_path / "diagnostic.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        study.reference_row(source, "M0HLT")
