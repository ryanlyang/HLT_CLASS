"""Fixed, explicitly normalized CE + auxiliary objectives (all reductions FP32)."""
import torch
import torch.nn.functional as F


def kl(q, logits):
    q, logits = q.float(), logits.float()
    return (torch.xlogy(q, q) - q * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def objectives(logits, heads, labels, targets, standardized, pair_valid, *, arm, coefficient):
    if arm not in {"CE", "COMP", "STRUCT", "BOTH"} or not 0 <= coefficient <= 1:
        raise ValueError("Unregistered loss")
    ce = F.cross_entropy(logits.float(), labels)
    zero = ce.new_zeros(())
    comp, struct = zero, zero
    if arm in {"COMP", "BOTH"}:
        counts = F.huber_loss(heads["counts"].float(), standardized[:, :5].float(), delta=1.)
        comp = .5 * (counts + kl(targets[:, 5:10], heads["fractions"]))
    if arm in {"STRUCT", "BOTH"}:
        scalars = F.huber_loss(heads["scalars"].float(), standardized[:, 5:].float(), delta=1.)
        # Deliberately no connection to pair-head parameters for an all-invalid
        # batch. Do not substitute a dummy target or change the group divisor.
        pair = kl(targets[pair_valid, 20:28], heads["pair"][pair_valid]) if bool(pair_valid.any()) else zero
        struct = (scalars + kl(targets[:, 12:20], heads["radial"]) + pair) / 3
    aux = .5 * (comp + struct) if arm == "BOTH" else comp if arm == "COMP" else struct
    weighted = coefficient * aux
    return ce + weighted, dict(ce=ce, composition=comp, structure=struct, weighted_aux=weighted)


def gradient_diagnostic(parts, representation):
    if representation is None:
        return dict(ce_norm=None, auxiliary_norm=None, cosine=None)
    left = torch.autograd.grad(parts["ce"], representation, retain_graph=True)[0].float()
    right = torch.autograd.grad(parts["weighted_aux"], representation, retain_graph=True, allow_unused=True)[0]
    right = torch.zeros_like(left) if right is None else right.float()
    a, b = torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
    cosine = None if a.item() == 0 or b.item() == 0 else float((left * right).sum() / (a * b))
    return dict(ce_norm=float(a), auxiliary_norm=float(b), cosine=cosine)
