"""Native fusion with compact masks and lossless pair-activation CPU storage."""
from contextlib import nullcontext
from copy import deepcopy

import torch

from .salience_learned_model import DelphesAdjacentFusionParticleTransformer


PAIR_OFFLOAD_POLICY = {
    "version": "pair_saved_tensors_cpu_v1",
    "scope": ["context", "primary", "cross"],
    "when": "training_grad_enabled_cuda",
    "pin_memory": True,
    "recompute": False,
    "pair_population": "full_combined_weaver",
}


class DzfixFusionParticleTransformer(DelphesAdjacentFusionParticleTransformer):
    def __init__(self, *, context_initialization_seed=None, pair_offload=True):
        super().__init__(context_initialization_seed=context_initialization_seed)
        if type(pair_offload) is not bool:
            raise TypeError("Pair offload switch must be Boolean")
        # False is for the reference parity run only. Production factories use True.
        self.pair_offload = pair_offload
        self.reset_pair_offload_stats()

    def reset_pair_offload_stats(self):
        self._pair_offload_stats = {
            name: dict(calls=0, saved_cuda_tensors=0, saved_cuda_bytes=0,
                       restored_cuda_tensors=0)
            for name in PAIR_OFFLOAD_POLICY["scope"]
        }

    def pair_offload_stats(self):
        # Counters are not buffers/checkpoint state and hold no tensor references.
        return deepcopy(self._pair_offload_stats)

    def _pair_embedding(self, mod, vectors, mask):
        if not (self.pair_offload and self.training and torch.is_grad_enabled()
                and vectors.device.type == "cuda"):
            return super()._pair_embedding(mod, vectors, mask)
        name = next(name for name, candidate in (
            ("context", self.context_mod), ("primary", self.primary_mod),
            ("cross", self.cross_pair_mod)) if candidate is mod)
        stats = self._pair_offload_stats[name]
        stats["calls"] += 1
        storage = torch.autograd.graph.save_on_cpu(pin_memory=True)

        def pack(tensor):
            packed = storage.pack_hook(tensor)
            if tensor.device.type == "cuda":
                stats["saved_cuda_tensors"] += 1
                stats["saved_cuda_bytes"] += tensor.numel() * tensor.element_size()
            return packed

        def unpack(packed):
            value = storage.unpack_hook(packed)
            if value.device.type == "cuda":
                stats["restored_cuda_tensors"] += 1
            return value

        # Only tensors autograd saves INSIDE pair_embed are moved, in their
        # original dtype. Outputs stay on GPU. No recomputation, pair chunking,
        # BN population change, detachment of outputs, or microbatching occurs.
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return super()._pair_embedding(mod, vectors, mask)

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


def validate_offload_stats(stats, *, calls):
    """Require actual CUDA packing AND backward retrieval at all three sites."""
    if not isinstance(stats, dict) or set(stats) != set(PAIR_OFFLOAD_POLICY["scope"]):
        raise ValueError("Pair offload evidence has different sites")
    for name, row in stats.items():
        keys = {"calls", "saved_cuda_tensors", "saved_cuda_bytes", "restored_cuda_tensors"}
        if (not isinstance(row, dict) or set(row) != keys
                or any(type(value) is not int for value in row.values())):
            raise ValueError(f"Invalid pair offload counters at {name}")
        if (row.get("calls") != calls or row.get("saved_cuda_bytes", 0) <= 0
                or row.get("saved_cuda_tensors", 0) <= 0
                or row.get("restored_cuda_tensors", 0) <= 0):
            raise ValueError(f"Pair offload was not exercised at {name}")


def native_offload_parity(primary_raw, context_raw, *, device, bf16=False):
    """Three training updates: logits/loss/gradients/BN/weights/AdamW parity.

    Same compact-mask model on both sides isolates storage from the separately
    checked native-mask transformation. CUDA exercises real pinned transfers;
    CPU is supplemental only. No production-size baseline is needed to prove
    the local storage transformation, so the reference does not itself OOM.
    """
    from .model import distillation_loss
    from .salience_learned_training import _optimizer

    cuda = torch.device(device).type == "cuda"
    if bf16 and not cuda:
        raise ValueError("BF16 offload parity requires CUDA")
    inputs = [tuple(torch.from_numpy(raw[k]).to(device)
                    for k in ("features", "vectors", "mask"))
              for raw in (primary_raw, context_raw)]
    labels = torch.from_numpy(primary_raw["labels"]).to(device).long()
    tolerance = dict(rtol=.01, atol=5e-4) if bf16 else dict(rtol=2e-5, atol=2e-6)

    def compare(left, right):
        if left.keys() != right.keys():
            raise ValueError("Offload parity tensor coverage differs")
        for name in left:
            if not torch.isfinite(left[name]).all() or not torch.isfinite(right[name]).all():
                raise ValueError(f"Nonfinite offload parity tensor: {name}")
            torch.testing.assert_close(left[name], right[name], **tolerance,
                                       msg=lambda message: f"Offload parity {name}: {message}")

    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if cuda else []):
        torch.manual_seed(3811)
        original = DzfixFusionParticleTransformer(context_initialization_seed=3812, pair_offload=False)
        # Exercise the full context/cross gradient path from the first update.
        for injection in original.injections:
            torch.nn.init.normal_(injection.residual_projection.weight, std=.01)
        offloaded = DzfixFusionParticleTransformer(context_initialization_seed=3812)
        offloaded.load_state_dict(original.state_dict(), strict=True)
        models = [original.to(device).train(), offloaded.to(device).train()]
        optimizers = [_optimizer(model) for model in models]
        teacher = torch.softmax(torch.randn(len(labels), 11, device=device), dim=-1)
        for step in range(3):
            snapshots = []
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(3813 + step)  # Identical training dropout draws.
                # FP32 on CUDA explicitly disables any enclosing autocast too.
                precision = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16) if cuda else nullcontext()
                with precision:
                    logits = model.forward_fused(*inputs[0], *inputs[1], alpha=1.).logits
                    loss = distillation_loss(logits, labels, teacher_probabilities=teacher)
                loss.backward()
                snapshot = {"logits": logits.detach().float().cpu(),
                            "loss": loss.detach().float().cpu()}
                snapshot.update({"grad/" + k: p.grad.detach().cpu().clone()
                                 for k, p in model.named_parameters() if p.grad is not None})
                optimizer.step()
                snapshot.update({"state/" + k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()})
                for name, parameter in model.named_parameters():
                    snapshot.update({f"optimizer/{name}/{key}": value.detach().cpu().clone()
                                     for key, value in optimizer.state.get(parameter, {}).items()
                                     if isinstance(value, torch.Tensor)})
                snapshots.append(snapshot)
                del logits, loss
            compare(*snapshots)
        stats = offloaded.pair_offload_stats()
        if cuda:
            validate_offload_stats(stats, calls=3)
        # Eval performs no packing, including eval-with-grad native mask checks.
        before = offloaded.pair_offload_stats()
        for model in models:
            model.eval()
        with torch.inference_mode():
            predictions = [model.forward_fused(*inputs[0], *inputs[1], alpha=1.).logits
                           for model in models]
        compare({"eval": predictions[0]}, {"eval": predictions[1]})
        if before != offloaded.pair_offload_stats():
            raise ValueError("Inference unexpectedly offloaded tensors")
    return dict(passed=True, device_type="cuda" if cuda else "cpu",
                precision="bf16" if bf16 else "fp32", steps=3,
                checks=["logits", "loss", "parameter_gradients", "batchnorm_buffers",
                        "updated_weights", "optimizer_state", "eval_logits"],
                tolerance=tolerance, offload_stats=stats)
