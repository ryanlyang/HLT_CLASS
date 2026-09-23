"""Registered physical batching, not a spec-only or accumulated-batch change."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as campaign, concat_k2_runtime as runtime,
    salience_learned_graph as graph, salience_learned_training as training,
)
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.jetclass2_delphes.contracts import artifact as base_artifact
from test_jetclass2_dzfix_fusion_chain import fake_native, make_cache
from test_jetclass2_dzfix_fusion_chain import rehash
from test_jetclass2_concat_k2 import spec_at


class TrackedCache:
    def __init__(self, role):
        source = make_cache(role)
        block = source.blocks[0]
        count, length = 300, 4
        indices = np.arange(count) % len(source)
        identities = np.asarray([np.frombuffer(i.to_bytes(32, "big"), np.uint8)
                                 for i in range(count)])
        self.cache = RamCache([RamBlock(0, np.arange(count+1)*length,
            block.features.reshape(len(source), length, 17)[indices].reshape(-1, 17),
            block.vectors.reshape(len(source), length, 4)[indices].reshape(-1, 4),
            identities, np.arange(count) % 11)], role=role,
            foundation_sha256="f"*64, coordinate_name="D100")
        self.seen = []

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def __len__(self):
        return len(self.cache)

    def batch(self, indices):
        self.seen.append(indices.copy())
        return self.cache.batch(indices)


@pytest.mark.parametrize("kd", [False, True])
@pytest.mark.parametrize("size", [None, 128])
def test_kernel_uses_actual_registered_batches_including_final_partial(fake_native, monkeypatch, kd, size):
    train, validation = TrackedCache("train"), TrackedCache("validation")
    node = campaign.nodes()[4 if kd else 3]
    model = runtime.new_model(node)
    original_training = deepcopy(graph.TRAINING)
    teacher = np.eye(11, dtype=np.float32)[train.labels] if kd else None
    original_batch = training._train_batch
    received = []

    def step(model, raw, **kw):
        received.append(len(raw["labels"]))
        if kd:
            np.testing.assert_array_equal(kw["teacher"].cpu().numpy(), teacher[train.seen[-1]])
        else:
            assert kw["teacher"] is None
        return original_batch(model, raw, **kw)

    monkeypatch.setattr(training, "_train_batch", step)
    kwargs = {} if size is None else dict(batch_size=size, inference_batch_size=size)
    report, state = training.train_kernel(model, lambda _: train, lambda _: validation,
        node=node, device="cpu", acceptance_passes=1,
        teacher_probabilities=teacher, teacher_identities=train.identities if kd else None, **kwargs)
    sizes = [256, 44] if size is None else [128, 128, 44]
    assert received == [len(i) for i in train.seen] == sizes
    assert [len(i) for i in validation.seen] == sizes
    assert report["validation_history"][0]["update"] == len(sizes)
    expected_order = np.random.default_rng(np.random.SeedSequence([node["sampler_seed"], 1])).permutation(300)
    np.testing.assert_array_equal(np.concatenate(train.seen), expected_order)
    np.testing.assert_array_equal(np.concatenate(validation.seen), np.arange(300))
    assert report["validation_history"][0]["learning_rate"] == graph.learning_rate(1.)
    assert report["selected_weights_restored"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), state[key], rtol=0, atol=0)
    assert graph.TRAINING == original_training and graph.TRAINING["batch_size"] == 256
    if size is None:
        assert "batching" not in report  # Preserve the legacy report shape.
    else:
        runtime.validate_kernel_batching(campaign.registration(), report)


@pytest.mark.parametrize("field", ["batch_size", "inference_batch_size"])
@pytest.mark.parametrize("value", [0, -1, True, 128., "128"])
def test_invalid_batch_override_fails_before_reading_data(field, value):
    def forbidden(_):
        pytest.fail("Invalid batching must fail before loading caches")
    with pytest.raises(ValueError, match="positive integers"):
        training.train_kernel(None, forbidden, forbidden, node={}, device="cpu", **{field:value})


def test_k2_registration_is_independent_and_versions_old_acceptance():
    original = deepcopy(graph.TRAINING)
    registration = campaign.registration("debug")
    assert registration["training"] == {**original, "batch_size":128}
    assert registration["inference_batch_size"] == 128
    assert registration["batch_probe_policy"]["order"] == [128]
    assert registration["batch_probe_policy"]["production_batch_size"] == 128
    registration["training"]["betas"][0] = 0.
    assert graph.TRAINING == original
    assert campaign.registration()["training"]["betas"] == original["betas"]
    assert campaign.VERSIONS == dict(LAUNCH_SPEC=6, CAMPAIGN_SPEC=6, ACCEPTANCE=5,
                                      TRAINING_REPORT=2, BATCH_PROBE=2)
    for kind, version in (("LAUNCH_SPEC",5), ("CAMPAIGN_SPEC",5), ("ACCEPTANCE",4),
                          ("TRAINING_REPORT",1), ("BATCH_PROBE",1)):
        with pytest.raises(ValueError, match="contract mismatch"):
            campaign.validate(base_artifact("CONCAT_K2_"+kind, version=version), kind)


@pytest.mark.parametrize("field", ["training_batch_size", "inference_batch_size", "gradient_accumulation_steps"])
def test_kernel_evidence_rejects_wrong_batching(tmp_path, field):
    spec = spec_at(tmp_path)
    report = dict(batching=dict(training_batch_size=128, inference_batch_size=128,
                                 gradient_accumulation_steps=1))
    runtime.validate_kernel_batching(spec, report)
    report["batching"][field] = 2 if field == "gradient_accumulation_steps" else 256
    with pytest.raises(ValueError, match="batch acceptance"):
        runtime.validate_kernel_batching(spec, report)


@pytest.mark.parametrize("field", ["training", "inference", "probes"])
def test_rehashed_old_batch_policy_cannot_be_registered_as_v6(tmp_path, field):
    spec = spec_at(tmp_path)
    if field == "training":
        spec["training"]["batch_size"] = 256
    elif field == "inference":
        spec["inference_batch_size"] = 256
    else:
        spec["batch_probe_policy"]["order"] = [128, 256]
    with pytest.raises(ValueError, match="registration differs"):
        campaign.validate_campaign(rehash(spec), check_source=False)
