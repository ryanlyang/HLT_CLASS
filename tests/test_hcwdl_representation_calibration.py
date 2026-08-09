from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.scouting.hcwdl_representation_calibration import (
    CalibrationComponentRows,
    CalibrationForwardResult,
    build_calibration_selection_artifact,
    calibrate_representation_components,
    calibration_required_after_pass,
    calibration_seed_payload_bytes,
    early_backbone_parameters,
    select_calibration_identities,
    validate_calibration_selection_artifact,
)


class _CalibrationBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deployable_model = nn.Module()
        self.deployable_model.mod = nn.Module()
        mod = self.deployable_model.mod
        mod.embed = nn.Linear(3, 4)
        mod.pair_embed = nn.Linear(4, 4)
        mod.blocks = nn.ModuleList((nn.Linear(4, 4), nn.Linear(4, 4)))
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.long))

    def forward(self, value):
        self.forward_counter.add_(1)
        mod = self.deployable_model.mod
        hidden = torch.tanh(mod.embed(value))
        hidden = torch.tanh(mod.pair_embed(hidden))
        hidden = torch.tanh(mod.blocks[0](hidden))
        return mod.blocks[1](hidden)


def test_smallest_hash_calibration_selection_is_deterministic_and_canonical():
    identities = [f"{value:064x}" for value in range(30, 0, -1)]
    selection = select_calibration_identities(
        campaign_sha256="a" * 64,
        parent_logit_counterpart_node_id="M5c",
        identity_sha256s=identities,
        limit=12,
    )
    assert selection.actual_rows == 12
    assert list(selection.rows) == sorted(
        selection.rows,
        key=lambda row: (bytes.fromhex(row.selection_sha256), bytes.fromhex(row.identity_sha256)),
    )
    assert calibration_seed_payload_bytes(selection.rows[0]).startswith(b'{"campaign_sha256"')
    repeated = select_calibration_identities(
        campaign_sha256="a" * 64,
        parent_logit_counterpart_node_id="M5c",
        identity_sha256s=identities,
        limit=12,
    )
    assert repeated == selection
    with pytest.raises(ValueError, match="repeated"):
        select_calibration_identities(
            campaign_sha256="a" * 64,
            parent_logit_counterpart_node_id="M5c",
            identity_sha256s=[identities[0], identities[0]], limit=2,
        )


def test_calibration_selection_artifact_binds_campaign_counterpart_and_order():
    identities = [f"{value:064x}" for value in range(64)]
    artifact = build_calibration_selection_artifact(
        campaign_sha256="a" * 64,
        parent_logit_counterpart_node_id="M3c",
        identity_sha256s=list(reversed(identities)),
        limit=16,
    )
    assert validate_calibration_selection_artifact(
        artifact,
        expected_campaign_sha256="a" * 64,
        expected_parent_logit_counterpart_node_id="M3c",
    ) == artifact["content_hash"]
    assert artifact["actual_rows"] == 16
    assert artifact["ordered_identity_sha256s"] != sorted(
        artifact["ordered_identity_sha256s"]
    )
    changed = dict(artifact)
    changed["ordered_identity_sha256s"] = list(reversed(
        artifact["ordered_identity_sha256s"]
    ))
    from hlt_classification.data.cache_contracts import with_content_hash
    changed.pop("content_hash")
    with pytest.raises(ValueError, match="order/digests"):
        validate_calibration_selection_artifact(with_content_hash(changed))


def test_calibration_uses_one_forward_and_restores_every_runtime_state():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model = _CalibrationBackbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    python_state = random.getstate()
    external = {"cursor": 11, "window": [1, 2, 3]}
    calls = 0

    def snapshot():
        return copy.deepcopy(external)

    def restore(value):
        external.clear(); external.update(value)

    def student_forward(batch):
        nonlocal calls
        calls += 1
        external["cursor"] += 1
        # Consume all three RNG families; calibration must erase this use.
        random.random(); np.random.random(); torch.rand(())
        return model(batch)

    def losses_from_forward(batch, output):
        labels = torch.arange(len(batch)) % 3
        base = (output.square().sum(-1) + 0.2 * output[:, 0]).float()
        jet = (output[:, 0] - 0.25).square()
        set_rows = (output[:, 1] + output[:, 2]).square()
        return CalibrationForwardResult(
            base_rows=base,
            labels=labels,
            class_weights=torch.ones(15, dtype=torch.float32),
            components={
                "jet": CalibrationComponentRows(jet, torch.ones(len(batch), dtype=torch.bool), {"rows": len(batch)}),
                "set": CalibrationComponentRows(set_rows, torch.ones(len(batch), dtype=torch.bool), {"rows": len(batch)}),
            },
        )

    batches = [torch.full((8, 3), 0.1 + index / 100) for index in range(16)]
    result = calibrate_representation_components(
        model=model, batches=batches, student_forward=student_forward,
        losses_from_forward=losses_from_forward,
        component_names=("jet", "set"), optimizer=optimizer,
        external_snapshot=snapshot, external_restore=restore,
    )
    assert calls == result.forward_calls == 16
    assert all(component.status == "active" for component in result.components.values())
    assert all(component.scale > 0 for component in result.components.values())
    assert external == {"cursor": 11, "window": [1, 2, 3]}
    assert model.state_dict().keys() == model_state.keys()
    assert all(torch.equal(model.state_dict()[name], value) for name, value in model_state.items())
    assert optimizer.state_dict() == optimizer_state
    assert all(torch.equal(parameter.grad, gradients[name]) for name, parameter in model.named_parameters())
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    assert np.array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_calibration_marks_sparse_support_inactive_without_fabrication():
    model = _CalibrationBackbone()

    def student_forward(batch):
        return model(batch)

    def losses_from_forward(batch, output):
        return CalibrationForwardResult(
            base_rows=output.square().sum(-1),
            labels=torch.zeros(len(batch), dtype=torch.long),
            class_weights=torch.ones(15),
            components={
                "relation": CalibrationComponentRows(
                    output[:, 0].square(),
                    torch.tensor([True] + [False] * (len(batch) - 1)),
                ),
            },
        )

    # Only eleven supplied batches have support, below the frozen 12-batch
    # threshold. The node remains valid and the component freezes at zero.
    result = calibrate_representation_components(
        model=model, batches=[torch.ones(4, 3) for _ in range(11)],
        student_forward=student_forward, losses_from_forward=losses_from_forward,
        component_names=("relation",),
        expected_batches=11,
    )
    component = result.components["relation"]
    assert component.status == "inactive_valid_support"
    assert component.inactive_reason == "insufficient_valid_batches"
    assert component.scale == 0 and component.scale_hex == 0.0.hex()


def test_calibration_parameter_support_and_barrier_schedule_fail_closed():
    model = _CalibrationBackbone()
    names = tuple(name for name, _ in early_backbone_parameters(model))
    assert names == tuple(sorted(names))
    manifest = {name: list(parameter.shape) for name, parameter in model.named_parameters() if name in names}
    assert early_backbone_parameters(model, expected_manifest=manifest)
    wrong = dict(manifest); wrong[names[0]] = [999]
    with pytest.raises(ValueError, match="manifest"):
        early_backbone_parameters(model, expected_manifest=wrong)
    assert calibration_required_after_pass(strategy="RSET", completed_pass=2) == ("jet", "set")
    assert calibration_required_after_pass(strategy="RSET", completed_pass=4) == ()
    assert calibration_required_after_pass(strategy="RREL", completed_pass=4) == ("relation",)
