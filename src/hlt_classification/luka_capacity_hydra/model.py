"""Installed Weaver encoder with width-specific, removable prediction heads."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from .contracts import cell, model_config, recipe, weights


class Native(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.mod = load_weaver_particle_transformer_class()(**model_config(width))

    def forward(self, features, vectors, mask):
        return self.mod(features, v=vectors, mask=mask)

    def no_weight_decay(self):
        return {'mod.cls_token'}


class Student(Native):
    def __init__(self, task):
        super().__init__(cell(task)['width'])
        self.heads = nn.ModuleDict()
        for name in weights(task):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe()['predictor_seeds'][name])
                self.heads[name] = nn.Sequential(nn.Linear(self.width, 128), nn.GELU(), nn.Linear(128, 128))
        self.capture = False
        self.predictions = None

    def forward(self, features, vectors, mask):
        self.predictions = None
        if not self.capture or not self.heads:
            return super().forward(features, vectors, mask)
        fc = self.mod.fc
        if (not isinstance(fc, nn.Sequential) or len(fc) != 1 or not isinstance(fc[0], nn.Linear)
                or fc[0].in_features != self.width or fc[0].out_features != 11):
            raise ValueError('Width-specific classification tap differs')
        captured = []
        handle = fc.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            logits = super().forward(features, vectors, mask)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (len(logits), self.width):
            raise ValueError('Expected one pooled representation')
        with torch.autocast(captured[0].device.type, enabled=False):
            self.predictions = {k: head(captured[0].float()) for k, head in self.heads.items()}
        return logits

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Student(task)


def objective(model, task, logits, labels, **targets):
    expected = weights(task)
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid classification logits')
    if set(targets) != set(expected):
        raise PermissionError('Target set differs from registered objective')
    terms = dict(ce=F.cross_entropy(logits.float(), labels.long()))
    for name, coefficient in expected.items():
        pred, target = model.predictions[name], targets[name]
        if pred.shape != (len(labels), 128) or target.shape != pred.shape:
            raise ValueError('Prediction/target dimensions differ')
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError('Nonfinite embedding')
        terms[name] = coefficient*F.smooth_l1_loss(pred.float(), target.detach().float(), beta=1, reduction='mean')
    loss = sum(terms.values())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite objective')
    return loss, terms
