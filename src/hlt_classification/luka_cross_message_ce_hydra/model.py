"""Exact existing deployable architecture, with no embedding head or targets."""
import torch
from hlt_classification.luka_cross_message_hydra.model import MessageParT
from hlt_classification.luka_cross_message_hydra.contracts import seed
from hlt_classification.luka_cross_message_hydra import training as donor
from .contracts import settings


class CEMessageParT(MessageParT):
    def __init__(self, width):
        # The donor constructs auxiliary heads in forked RNG scopes, preserving
        # exact native/message initialization and subsequent dropout RNG states.
        super().__init__(width=width, mode='ce', embedding=5, correction_weight=0., seed_index=0)
        del self.predictor
        self.embedding = 0


def new_model(imported, task):
    cfg = settings(task)
    torch.manual_seed(seed(imported, cfg, 'initialization_seed'))
    return CEMessageParT(cfg['width'])


def step(model, optimizer, raw, device, *, lr):
    if (not isinstance(model, CEMessageParT) or model.embedding != 0
            or model.mode != 'ce' or hasattr(model, 'predictor')):
        raise PermissionError('CE-only model required')
    if set(raw) != {'features', 'vectors', 'mask', 'labels', 'identities'}:
        raise PermissionError('Only native HLT fields and ground truth may enter CE training')
    return donor.step(model, optimizer, raw, device, lr=lr)


def deployment(task, state, device):
    model = CEMessageParT(settings(task)['width'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
