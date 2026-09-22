"""K2-only lossless CPU storage for native Weaver pair saved tensors.

The native pair forward and its full BatchNorm population are unchanged.
Only tensors retained by autograd inside that forward are stored on pinned
CPU memory; no recomputation, microbatching, casting or input trimming occurs.
"""
from contextlib import nullcontext
from copy import deepcopy
import time

import torch

from .model import DelphesParticleTransformer, distillation_loss


PAIR_STORAGE = dict(
    version="k2_pair_saved_tensors_cpu_v1", scope="native_pair_embed_only",
    when="training_grad_enabled_cuda", pin_memory=True, recompute=False,
    dtype="unchanged", batchnorm_population="unchanged", inference_offload=False,
)
BATCH_PROBE_POLICY = dict(
    order=[128, 256], steps=3, rows="longest_real_train_and_validation",
    fresh_model_per_batch=True, production_batch_size=256,
    automatic_batch_fallback=False, stop_on_oom=True,
)
PARITY_CHECKS = ["logits", "loss", "feature_gradients", "parameter_gradients",
                 "batchnorm_buffers", "updated_weights", "optimizer_state", "eval_logits"]


class K2ParticleTransformer(DelphesParticleTransformer):
    def __init__(self, *, pair_storage=True):
        super().__init__()
        if type(pair_storage) is not bool:
            raise TypeError("Pair storage switch must be Boolean")
        self.pair_storage = pair_storage
        # Keep exactly the native checkpoint keys. A wrapper nn.Module around
        # pair_embed would insert an extra namespace. This is instance-local:
        # neither the installed class nor any other campaign is monkeypatched.
        self._native_pair_forward = self.mod.pair_embed.forward
        self.mod.pair_embed.forward = self._pair_forward
        self.reset_pair_storage_stats()

    def reset_pair_storage_stats(self):
        self._pair_storage_stats = dict(calls=0, saved_cuda_tensors=0,
            saved_cuda_bytes=0, restored_cuda_tensors=0)

    def pair_storage_stats(self):
        # No tensors/graphs are retained by these diagnostic counters.
        return deepcopy(self._pair_storage_stats)

    def _pair_forward(self, vectors, *args, **kwargs):
        if not (self.pair_storage and self.training and torch.is_grad_enabled()
                and vectors.device.type == "cuda"):
            return self._native_pair_forward(vectors, *args, **kwargs)
        stats = self._pair_storage_stats
        stats["calls"] += 1
        storage = torch.autograd.graph.save_on_cpu(pin_memory=True)

        def pack(tensor):
            packed = storage.pack_hook(tensor)
            if tensor.device.type == "cuda":
                stats["saved_cuda_tensors"] += 1
                stats["saved_cuda_bytes"] += tensor.numel() * tensor.element_size()
            return packed

        def unpack(packed):
            tensor = storage.unpack_hook(packed)
            if tensor.device.type == "cuda":
                stats["restored_cuda_tensors"] += 1
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return self._native_pair_forward(vectors, *args, **kwargs)


def validate_storage_stats(stats, *, calls=3):
    keys = {"calls", "saved_cuda_tensors", "saved_cuda_bytes", "restored_cuda_tensors"}
    if (not isinstance(stats, dict) or set(stats) != keys
            or any(type(v) is not int or v <= 0 for v in stats.values())
            or stats["calls"] != calls
            or stats["restored_cuda_tensors"] > stats["saved_cuda_tensors"]):
        raise ValueError("K2 pair storage has no real CUDA save/backward evidence")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def storage_parity(raw, *, device, bf16=False):
    """Compare the unmodified adapter and K2 over three complete AdamW steps."""
    from .salience_learned_training import _optimizer
    cuda = torch.device(device).type == "cuda"
    if bf16 and not cuda:
        raise ValueError("BF16 storage parity requires CUDA")
    tolerance = dict(rtol=.01, atol=5e-4) if bf16 else dict(rtol=2e-5, atol=2e-6)

    def compare(left, right):
        if left.keys() != right.keys():
            raise ValueError("K2 storage parity tensor coverage differs")
        for name in left:
            if not torch.isfinite(left[name]).all() or not torch.isfinite(right[name]).all():
                raise ValueError("Nonfinite K2 storage parity tensor: " + name)
            torch.testing.assert_close(left[name], right[name], **tolerance,
                msg=lambda message: f"K2 storage parity {name}: {message}")

    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if cuda else []):
        torch.manual_seed(4711)
        native = DelphesParticleTransformer()
        optimized = K2ParticleTransformer()
        optimized.load_state_dict(native.state_dict(), strict=True)
        models = [native.to(device).train(), optimized.to(device).train()]
        optimizers = [_optimizer(model) for model in models]
        vectors, mask = (torch.from_numpy(raw[k]).to(device) for k in ("vectors", "mask"))
        features = [torch.tensor(raw["features"], device=device, requires_grad=True) for _ in models]
        labels = torch.from_numpy(raw["labels"]).to(device).long()
        teacher = torch.softmax(torch.randn(len(labels), 11, device=device), -1)
        timings = [[], []]
        for step in range(3):
            snapshots = []
            for i, (model, optimizer, x) in enumerate(zip(models, optimizers, features)):
                optimizer.zero_grad(set_to_none=True)
                x.grad = None
                torch.manual_seed(4712 + step)
                synchronize(device)
                started = time.monotonic()
                precision = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16) if cuda else nullcontext()
                with precision:
                    logits = model(x, vectors, mask)
                    loss = distillation_loss(logits, labels, teacher_probabilities=teacher)
                loss.backward()
                snapshot = {"logits": logits.detach().float().cpu(), "loss": loss.detach().cpu(),
                            "feature_grad": x.grad.detach().cpu().clone()}
                snapshot.update({"grad/" + k: p.grad.detach().cpu().clone()
                                 for k, p in model.named_parameters() if p.grad is not None})
                optimizer.step()
                synchronize(device)
                timings[i].append(time.monotonic() - started)
                snapshot.update({"state/" + k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                for name, p in model.named_parameters():
                    snapshot.update({f"optimizer/{name}/{key}": v.detach().cpu().clone()
                        for key, v in optimizer.state.get(p, {}).items() if isinstance(v, torch.Tensor)})
                snapshots.append(snapshot)
                del logits, loss
            compare(*snapshots)
        stats = optimized.pair_storage_stats()
        if cuda:
            validate_storage_stats(stats)
        with torch.inference_mode():
            outputs = [m.eval()(x, vectors, mask) for m, x in zip(models, features)]
        compare({"eval": outputs[0]}, {"eval": outputs[1]})
        if optimized.pair_storage_stats() != stats:
            raise ValueError("Inference must not offload pair tensors")
    return dict(passed=True, device_type="cuda" if cuda else "cpu", steps=3,
        precision="bf16" if bf16 else "fp32", checks=list(PARITY_CHECKS), tolerance=tolerance,
        storage_stats=stats, native_step_seconds=timings[0], optimized_step_seconds=timings[1],
        timing_scope="small_parity_fixture_including_diagnostic_copies_not_production_throughput")
