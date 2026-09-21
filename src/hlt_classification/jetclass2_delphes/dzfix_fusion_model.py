"""Same fusion computation with the accepted CMS temporary-mask optimization."""
import torch

from .salience_learned_model import DelphesAdjacentFusionParticleTransformer


class DzfixFusionParticleTransformer(DelphesAdjacentFusionParticleTransformer):
    def _prepare_injection_bias(self, pair_bias, context_padding):
        # Preserve the full Weaver pair-BN population. Only compact the output
        # rectangle and merge its padding once rather than at four injections.
        # No detach, input cropping, checkpointing or batch-size change.
        padding = torch.zeros(context_padding.shape, dtype=pair_bias.dtype,
                              device=pair_bias.device).masked_fill(context_padding, float("-inf"))
        return (pair_bias + padding[:, None, None, :]).contiguous(), None


def native_mask_parity(raw, *, device):
    """Nonzero residual forward/backward parity using actual installed Weaver."""
    from .salience_learned_training import _autocast
    inputs = tuple(torch.from_numpy(raw[k]).to(device) for k in ("features", "vectors", "mask"))
    with torch.random.fork_rng(devices=[0] if str(device).startswith("cuda") else []):
        torch.manual_seed(3701)
        old = DelphesAdjacentFusionParticleTransformer(context_initialization_seed=3702).to(device).eval()
        for injection in old.injections:
            torch.nn.init.normal_(injection.residual_projection.weight, std=.01)
        new = DzfixFusionParticleTransformer(context_initialization_seed=3702).to(device).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        outputs, gradients = [], []
        for model in (old, new):
            with _autocast(device):
                logits = model.forward_fused(*inputs, *inputs, alpha=1.).logits
            logits.float().square().mean().backward()
            outputs.append(logits.detach().float().cpu())
            gradients.append({k: p.grad.detach().float().cpu().clone()
                              for k, p in model.named_parameters() if p.grad is not None})
        tolerance = dict(rtol=.01, atol=5e-4) if str(device).startswith("cuda") else dict(rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(*outputs, **tolerance)
        if gradients[0].keys() != gradients[1].keys():
            raise ValueError("Compact-bias gradient coverage differs")
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key], gradients[1][key], **tolerance)
    return True
