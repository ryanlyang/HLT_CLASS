"""15-class single-view ParT, with the current JC2 floor-tail recipe."""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.models.particle_transformer import canonical_particle_transformer_config, load_weaver_particle_transformer_class
from hlt_classification.jetclass2_delphes.campaign import recipe, learning_rate
from hlt_classification.cms_salience_learned.training import autocast, tensors, optimizer_for
from hlt_classification.scouting.evaluation import classification_metrics
from .contracts import artifact, ARMS


def model_config(arm):
    if arm not in ARMS:
        raise ValueError("Unknown model feature interface")
    return dict(canonical_particle_transformer_config(), input_dim=17 if arm == "SHARED17" else 21,
                num_classes=15, trim=False)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.mod = load_weaver_particle_transformer_class()(**model_config(arm))

    def forward(self, features, vectors, mask):
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self):
        return {"mod.cls_token"}


def fresh(node):
    torch.manual_seed(node["initialization_seed"])
    return Model(node["arm"])


def probabilities_valid(p, rows):
    if (p.shape != (rows, 15) or not np.isfinite(p).all() or np.any(p < 0)
        or not np.allclose(p.sum(1), 1., atol=2e-6, rtol=0)):
        raise ValueError("CMS probability simplex differs")


def metrics(labels, p):
    probabilities_valid(p, len(labels))
    if set(np.unique(labels)) != set(range(15)):
        raise ValueError("All fifteen classes required for validation")
    result = classification_metrics(np.log(np.maximum(p, 1e-30)), labels)
    finite_r = []
    for row in result["per_class"].values():
        if "qcd_rejection" not in row:
            continue
        r = row["qcd_rejection"]["50pct"]
        row["rejection_at_50pct"] = None if r["zero_background"] else r["rejection"]
        row["one_event_resolution_rejection"] = int((labels == 0).sum())
        finite_r.append(row["rejection_at_50pct"])
    result["macro_mean_log_qcd_rejection_at_50pct_signal"] = (
        float(np.mean(np.log(finite_r))) if all(x is not None for x in finite_r) else None)
    return result


def predict(model, cache, *, device, temperature=1., batch_size=256):
    if temperature not in (1., 2.):
        raise ValueError("Unregistered inference temperature")
    model.eval()
    result = np.empty((len(cache), 15), np.float32)
    with torch.inference_mode():
        for start in range(0, len(cache), batch_size):
            ix = np.arange(start, min(start + batch_size, len(cache)))
            with autocast(device):
                logits = model(*tensors(cache.batch_primary(ix), device))
            result[ix] = torch.softmax(logits.float() / temperature, -1).cpu().numpy()
    probabilities_valid(result, len(cache))
    return result


def loss(logits, labels, teacher=None):
    if logits.shape != (len(labels), 15) or not torch.isfinite(logits).all():
        raise ValueError("Invalid CMS logits")
    ce = F.cross_entropy(logits.float(), labels)
    if teacher is None:
        return ce
    q = teacher.float().detach()
    if (q.shape != logits.shape or not torch.isfinite(q).all() or (q < 0).any()
        or not torch.allclose(q.sum(1), torch.ones(len(q), device=q.device), atol=2e-6, rtol=0)):
        raise ValueError("Invalid CMS teacher probabilities")
    return .25 * ce + .75 * 4. * F.kl_div(F.log_softmax(logits.float() / 2., -1), q, reduction="batchmean")


def train(model, tr, va, *, node, device, teacher=None, teacher_ids=None, acceptance_passes=None):
    config = recipe()
    if (tr.role != "train" or va.role != "validation" or tr.foundation_sha256 != va.foundation_sha256
        or tr.primary != node["coordinate"] or va.primary != node["coordinate"] or not len(tr) or not len(va)):
        raise ValueError("CMS train/validation view lineage differs")
    if node["teacher"] is None:
        if teacher is not None or teacher_ids is not None:
            raise ValueError("CE control received a teacher")
    else:
        if teacher is None or teacher.dtype != np.float32 or not np.array_equal(teacher_ids, tr.identities):
            raise ValueError("CMS teacher/student identity join differs")
        probabilities_valid(teacher, len(tr))
    if acceptance_passes is not None and (type(acceptance_passes) is not int or not 1 <= acceptance_passes <= 3):
        raise ValueError("Miniature must use 1-3 passes")
    model.to(device)
    optimizer = optimizer_for(model)
    torch.manual_seed(node["initialization_seed"])
    history, best, best_key = [], None, None
    significant, significant_pass, updates = -math.inf, 0, 0
    started = time.monotonic()
    for epoch in range(1, (acceptance_passes or config["maximum_passes"]) + 1):
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([node["sampler_seed"], epoch])).permutation(len(tr))
        t0 = time.monotonic()
        for start in range(0, len(tr), config["batch_size"]):
            ix = order[start:start + config["batch_size"]]
            updates += 1
            lr = learning_rate(updates / math.ceil(len(tr) / config["batch_size"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            raw = tr.batch_primary(ix)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(*tensors(raw, device))
            target = None if teacher is None else torch.from_numpy(teacher[ix]).to(device)
            objective = loss(logits, torch.from_numpy(raw["labels"]).to(device), target)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite CMS loss")
            objective.backward()
            grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not grads or not torch.stack(grads).all():
                raise FloatingPointError("Nonfinite/absent CMS parameter gradients")
            optimizer.step()
        train_seconds = time.monotonic() - t0
        t0 = time.monotonic()
        current = metrics(va.labels, predict(model, va, device=device))
        r50 = current["macro_mean_log_qcd_rejection_at_50pct_signal"]
        key = (current["macro_ovr_auc"], -current["cross_entropy"], -math.inf if r50 is None else r50, -updates)
        if best_key is None or key > best_key:
            best_key, selected, best_metrics = key, epoch, current
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if current["macro_ovr_auc"] > significant + config["minimum_auc_delta"]:
            significant, significant_pass = current["macro_ovr_auc"], epoch
        history.append(dict(pass_number=epoch, train_seconds=train_seconds,
            validation_seconds=time.monotonic() - t0, validation=current, learning_rate=lr))
        print(f"CMS-FEATURE node={node['node_id']} pass={epoch} auc={current['macro_ovr_auc']:.7f} train_seconds={train_seconds:.2f}", flush=True)
        if acceptance_passes is None and epoch >= config["minimum_passes"] and epoch - significant_pass >= config["patience"]:
            break
    model.load_state_dict(best, strict=True)
    model.eval()
    report = artifact("TRAINING", parents={"foundation": tr.foundation_sha256, "recipe": config["content_hash"]},
        node=node, model=model_config(node["arm"]), passes=len(history), selected_pass=selected,
        validation=best_metrics, history=history, scientific_fit=acceptance_passes is None,
        selected_weights_restored=True, runtime_seconds=time.monotonic() - started)
    return report, best


def parity(cache, arm, device):
    torch.manual_seed(71)
    wrapper = Model(arm).to(device).eval()
    direct = load_weaver_particle_transformer_class()(**model_config(arm)).to(device).eval()
    direct.load_state_dict(wrapper.mod.state_dict(), strict=True)
    raw = cache.batch_primary(np.arange(min(4, len(cache))))
    x1 = torch.tensor(raw["features"], device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    vectors, mask = (torch.tensor(raw[k], device=device) for k in ("vectors", "mask"))
    y1, y2 = wrapper(x1, vectors, mask), direct(x2, v=vectors, mask=mask)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
    y1.square().mean().backward(); y2.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
    if not torch.isfinite(y1).all() or not torch.isfinite(x1.grad).all():
        raise ValueError("Nonfinite parity logits/features")
    for (ln, left), (rn, right) in zip(wrapper.mod.named_parameters(), direct.named_parameters(), strict=True):
        if ln != rn or (left.grad is None) != (right.grad is None):
            raise ValueError("Parity gradient topology differs")
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
            if not torch.isfinite(left.grad).all():
                raise ValueError("Nonfinite parity parameter gradients")
    return dict(passed=True, model=model_config(arm), forward_features_parameters=True, precision="fp32")
