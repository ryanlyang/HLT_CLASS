"""Three independent summaries sharing installed Weaver class-attention weights."""
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from .contracts import cell, model_config, recipe, weights, EXTRA_PARAMETERS


class Native(nn.Module):
    def __init__(self, summaries=1, query_seed=None):
        super().__init__()
        if summaries not in (1, 3):
            raise ValueError('Unregistered summary count')
        self.width, self.summary_count = 128, summaries
        self.mod = load_weaver_particle_transformer_class()(**model_config())
        self.capture = False
        self.summaries = None
        fc = self.mod.fc
        if (not isinstance(fc, nn.Sequential) or len(fc) != 1 or not isinstance(fc[0], nn.Linear)
                or (fc[0].in_features, fc[0].out_features) != (128, 11)):
            raise ValueError('Native classifier contract differs')
        self.native_parameters = sum(p.numel() for p in self.mod.parameters())
        if summaries == 3:
            # Neither the extra queries nor auxiliary heads perturb native initialization/RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe()['extra_query_seed'] if query_seed is None else query_seed)
                self.mod.extra_cls_tokens = nn.Parameter(torch.empty(1, 2, 128))
                nn.init.trunc_normal_(self.mod.extra_cls_tokens, std=.02)
                expanded = nn.Linear(384, 11)
                with torch.no_grad():
                    expanded.weight.zero_()
                    expanded.weight[:, :128].copy_(fc[0].weight)
                    expanded.bias.copy_(fc[0].bias)
                self.mod.fc = nn.Sequential(expanded)
        expected = EXTRA_PARAMETERS if summaries == 3 else 0
        if sum(p.numel() for p in self.mod.parameters()) - self.native_parameters != expected:
            raise ValueError('Deployed parameter delta differs')

    def forward(self, features, vectors, mask):
        self.summaries = None
        if self.summary_count == 1:
            # Preserve the installed original forward, including precision and pooling.
            handle = self.mod.fc.register_forward_pre_hook(self._capture) if self.capture else None
            try:
                return self.mod(features, v=vectors, mask=mask)
            finally:
                if handle is not None:
                    handle.remove()
        particles, padding = self.mod._forward_encoder(features, v=vectors, mask=mask)
        queries = [self.mod.cls_token, *self.mod.extra_cls_tokens.split(1, dim=1)]
        pooled = []
        for query in queries:
            h = query.expand(particles.shape[0], -1, -1)
            for block in self.mod.cls_blocks:
                h = block(particles, x_cls=h, padding_mask=padding)
            pooled.append(self.mod.norm(h).squeeze(1))
        if self.capture:
            self.summaries = pooled
        return self.mod.fc(torch.cat(pooled, dim=1))

    def _capture(self, module, args):
        self.summaries = [args[0]]

    def no_weight_decay(self):
        return {'mod.cls_token', 'mod.extra_cls_tokens'}


class Student(Native):
    def __init__(self, task):
        config = cell(task)
        super().__init__(config['summaries'], recipe()['extra_query_seed'] + config['seed']*recipe()['seed_offset'])
        self.sources = config['prediction_sources']
        self.heads = nn.ModuleDict()
        for name in weights(task):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe()['predictor_seeds'][name])
                self.heads[name] = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.predictions = None

    def forward(self, features, vectors, mask):
        self.predictions = None
        logits = super().forward(features, vectors, mask)
        if self.capture:
            if self.summaries is None or any(h.shape != (len(logits), 128) for h in self.summaries):
                raise ValueError('Summary shape differs')
            with torch.autocast(logits.device.type, enabled=False):
                self.predictions = {k: head(self.summaries[self.sources[k]].float()) for k, head in self.heads.items()}
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
