"""One positive plus 1,024 negative frozen keys; independent of U100 regression."""
import torch
from torch.nn import functional as F
from hlt_classification.luka_u100_embedding.training import objective as u100_objective
from .contracts import TEMPERATURE, NEGATIVES


def scores(query, keys):
    if (query.ndim != 2 or query.shape[1] != 128 or keys.shape != (len(query), NEGATIVES+1, 128)
            or not torch.isfinite(query).all() or not torch.isfinite(keys).all()):
        raise ValueError('Invalid matching tensors')
    q = F.normalize(query.float(), dim=-1, eps=1e-12)
    return torch.bmm(keys.detach().float(), q.unsqueeze(-1)).squeeze(-1) / TEMPERATURE


def match(query, keys, positive=None):
    logits = scores(query, keys)
    if positive is None:
        positive = torch.zeros(len(query), device=query.device, dtype=torch.long)
    return F.cross_entropy(logits, positive)


def objective(logits, labels, prediction, target, query, keys, coefficient):
    base, ce, embed = u100_objective(logits, labels, prediction, target, coefficient=5.)
    contrastive = match(query, keys)
    total = base + coefficient*contrastive
    if not torch.isfinite(total):
        raise ValueError('Nonfinite training loss')
    return total, ce, embed, contrastive
