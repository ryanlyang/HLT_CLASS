"""Exercise the production loss boundary without requiring installed Weaver."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hlt_classification.jetclass2_delphes.salience_learned_model import (
    FusionOutput, WithdrawalOutput,
)
from hlt_classification.jetclass2_delphes.salience_learned_training import (
    _train_batch,
)
from hlt_classification.scouting.hcwdl_offline_hlt_withdrawal import (
    withdrawal_loss,
)


def _outputs(alpha):
    generator = torch.Generator().manual_seed(714)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def route():
        return FusionOutput(
            torch.randn(2, 11, generator=generator, requires_grad=True),
            tuple(torch.randn(2, 3, 5, generator=generator, requires_grad=True)
                  for _ in range(4)) if alpha else (),
            mask,
        )

    zero = route()
    output = WithdrawalOutput(zero, route() if alpha else zero, alpha)
    labels = torch.tensor([1, 4])
    teacher = torch.softmax(torch.randn(2, 11, generator=generator) / 2., dim=1)
    return output, labels, teacher


def test_fusion_loss_aliases_preserve_the_primary_tensors_and_autograd():
    output, _, _ = _outputs(.5)
    for route in (output.zero, output.privileged):
        assert route.hlt_states is route.primary_states
        assert route.hlt_mask is route.primary_mask
        assert all(state.requires_grad for state in route.hlt_states)


@pytest.mark.parametrize("alpha", [1., .5, 0.])
def test_preflight_loss_preserves_coefficients_masking_and_directed_gradients(alpha):
    # The same public loss is called directly by _run_preflight on FusionOutput.
    output, labels, teacher = _outputs(alpha)
    terms = withdrawal_loss(output, labels, teacher)
    zero, privileged = output.zero.logits, output.privileged.logits

    def kd(logits):
        return 4. * F.kl_div(F.log_softmax(logits / 2., dim=1), teacher,
                            reduction="batchmean")

    ce_zero = F.cross_entropy(zero, labels)
    expected = .25 * ce_zero + .30 * kd(zero)
    expected += .15 * F.cross_entropy(privileged, labels) + .20 * kd(privileged)
    if alpha:
        assert torch.autograd.grad(
            terms["logit_consistency"], privileged,
            allow_unused=True, retain_graph=True,
        )[0] is None
        logit_consistency = F.kl_div(
            F.log_softmax(zero, dim=1), F.softmax(privileged.detach(), dim=1),
            reduction="batchmean",
        )
        # Index only valid primary tokens, independently of the loss's mask sum.
        representation = torch.stack([
            F.smooth_l1_loss(
                F.layer_norm(left, (5,))[output.zero.primary_mask],
                F.layer_norm(right.detach(), (5,))[output.zero.primary_mask],
            )
            for left, right in zip(output.zero.primary_states,
                                   output.privileged.primary_states, strict=True)
        ]).mean()
        torch.testing.assert_close(terms["representation_consistency"], representation)
        expected += .05 * logit_consistency + .05 * representation
    else:
        assert output.zero is output.privileged
        assert terms["logit_consistency"].item() == 0.
        assert terms["representation_consistency"].item() == 0.
        torch.testing.assert_close(terms["total"], .40 * ce_zero + .50 * kd(zero))
    torch.testing.assert_close(terms["total"], expected)
    terms["total"].backward()
    assert torch.isfinite(zero.grad).all()
    if alpha:
        assert torch.isfinite(privileged.grad).all()
        for state in output.zero.primary_states:
            assert torch.isfinite(state.grad).all()
            assert state.grad[output.zero.primary_mask].abs().sum() > 0
            assert torch.count_nonzero(state.grad[~output.zero.primary_mask]) == 0
        assert all(state.grad is None for state in output.privileged.primary_states)


@pytest.mark.parametrize("damage", ["states", "mask"])
def test_withdrawal_still_rejects_mismatched_primary_surfaces(damage):
    output, labels, teacher = _outputs(.5)
    if damage == "states":
        privileged = replace(output.privileged,
                             primary_states=output.privileged.primary_states[:3])
    else:
        privileged = replace(output.privileged, primary_mask=~output.zero.primary_mask)
    with pytest.raises(ValueError, match="representation surfaces differ"):
        withdrawal_loss(replace(output, privileged=privileged), labels, teacher)


@pytest.mark.parametrize("role", ["fusion_withdrawal", "morph_withdrawal"])
@pytest.mark.parametrize("alpha", [1., .5, 0.])
def test_training_dispatch_uses_compatible_loss_and_no_context_at_zero(role, alpha):
    output, labels, teacher = _outputs(alpha)
    primary = {
        "features": np.zeros((2, 17, 3), np.float32),
        "vectors": np.zeros((2, 4, 3), np.float32),
        "mask": output.zero.primary_mask[:, None].numpy(),
        "labels": labels.numpy(),
    }
    raw = ({"primary": primary, "context": primary, "labels": labels.numpy()}
           if alpha else primary)
    calls = []

    class FusionProbe:
        def forward_withdrawal(self, *views, alpha):
            # The exact-zero training batch intentionally has no context keys.
            assert len(views) == (6 if alpha else 3)
            calls.append(alpha)
            return output

    loss, metrics = _train_batch(
        FusionProbe(), raw, node={"role": role}, device="cpu",
        teacher=teacher, alpha=alpha,
    )
    assert calls == [alpha]
    assert metrics["alpha"] == alpha
    assert all(np.isfinite(value) for value in metrics.values())
    loss.backward()
    assert torch.isfinite(output.zero.logits.grad).all()
