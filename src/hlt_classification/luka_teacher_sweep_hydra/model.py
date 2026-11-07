"""Native classifier with removable, strongly weighted teacher objectives."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_embedding_aux.model import head_input
from hlt_classification.luka_supervision_depth.model import seeded_head, supcon
from hlt_classification.luka_supcon5.model import Student
from .contracts import TEACHERS, STUDENTS, recipe, term_names
from . import targets


def variance_covariance(h):
    h = h.float()
    if len(h) < 2:
        return h.sum()*0, h.sum()*0
    z = h-h.mean(0); var = F.relu(1-torch.sqrt(z.square().mean(0)+1e-4)).mean()
    c = z.T@z/(len(h)-1); off = c[~torch.eye(h.shape[1], device=h.device, dtype=torch.bool)]
    return var, off.square().sum()/h.shape[1]


def barlow(a, b):
    if len(a) < 2:
        return (a.sum()+b.sum())*0
    def standardize(x):
        z = x.float()-x.float().mean(0)
        return z/torch.sqrt(z.square().mean(0)+1e-4)
    c = standardize(a).T@standardize(b)/len(a)
    eye = torch.eye(a.shape[1], device=a.device, dtype=torch.bool)
    return (c.diag()-1).square().sum()+.0051*c[~eye].square().sum()


def instance(a, b):
    n = len(a)
    if n < 2:
        return (a.sum()+b.sum())*0
    z = F.normalize(torch.cat((a, b)).float(), dim=1)
    scores = z@z.T/.1
    scores = scores.masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), -torch.inf)
    return F.cross_entropy(scores, (torch.arange(2*n, device=z.device)+n)%(2*n))


def thin(inputs, generator, fraction, *, removed_target=False):
    valid = inputs['mask'][:, 0].bool()
    keep = (torch.rand(valid.shape, generator=generator, device=valid.device) >= fraction) & valid
    first = valid.to(torch.int64).argmax(1)
    missing = ~keep.any(1); keep[missing, first[missing]] = True
    if removed_target:
        full = (keep == valid).all(1) & (valid.sum(1) > 1)
        last = valid.shape[1]-1-valid.flip(1).to(torch.int64).argmax(1)
        keep[full, last[full]] = False
    mask = keep[:, None]
    return {k: (v*mask if k != 'mask' else mask) for k, v in inputs.items()}, valid & ~keep


class SetDecoder(nn.Module):
    def __init__(self, maximum):
        super().__init__(); self.maximum = maximum
        self.queries = nn.Parameter(torch.randn(maximum, 128)*.02)
        self.network = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 22))
        self.count = nn.Linear(128, maximum+1)

    def forward(self, h):
        z = torch.cat((h[:, None].expand(-1, self.maximum, -1), self.queries[None].expand(len(h), -1, -1)), -1)
        return self.network(z), self.count(h)


def set_loss(output, counts, target, mask):
    """Exact detached assignment, differentiable matched regression/presence/count."""
    pred, presence = output[..., :21], output[..., 21]
    if int(mask.sum(1).max()) > pred.shape[1]:
        raise ValueError('No truncation of masked particles')
    # Squared L1-Huber costs are formed per jet to bound GPU/host transient memory.
    costs = []
    with torch.no_grad():
        for p, t, m in zip(pred, target, mask):
            costs.append(F.smooth_l1_loss(p[:, None].expand(-1, int(m.sum()), -1),
                t[m][None].expand(len(p), -1, -1), reduction='none').mean(-1).cpu().numpy())
    with ThreadPoolExecutor(max_workers=6) as pool:
        assignments = list(pool.map(linear_sum_assignment, costs))
    losses = []
    for p, logits, t, m, (i, j) in zip(pred, presence, target, mask, assignments):
        labels = torch.zeros_like(logits); labels[i] = 1
        pos = labels.bool(); neg = ~pos
        regression = F.smooth_l1_loss(p[i], t[m][j]) if len(i) else p.sum()*0
        presence_terms = []
        if pos.any(): presence_terms.append(F.softplus(-logits[pos]).mean())
        if neg.any(): presence_terms.append(F.softplus(logits[neg]).mean())
        losses.append(regression+sum(presence_terms)/len(presence_terms))
    return torch.stack(losses).mean()+F.cross_entropy(counts, mask.sum(1).long())


class Teacher(DelphesParticleTransformer):
    def __init__(self, task, statistics=None):
        super().__init__(); self.task = task; self.statistics = statistics
        self.capture = False; self.h = self.inputs = None
        self.projector = seeded_head(84221) if task in ('T01', 'T02', 'T04', 'T07') else None
        self.instance_projector = seeded_head(84222) if task == 'T04' else None
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(84223)
            self.decoder = SetDecoder(statistics['maximum']) if task == 'T08' else None
            self.summary = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 36)) if task == 'T09' else None

    def forward(self, features, vectors, mask):
        self.h = self.inputs = None
        if not self.capture:
            return super().forward(features, vectors, mask)
        self.inputs = dict(features=features, vectors=vectors, mask=mask)
        logits, self.h = head_input(self, features, vectors, mask)
        return logits

    def view(self, inputs):
        with torch.autocast(self.h.device.type, enabled=self.h.device.type == 'cuda', dtype=torch.bfloat16):
            return head_input(self, **inputs)

    def native_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith('mod.')}


def new_model(spec, task):
    if task not in FIT_KEYS:
        raise ValueError('Unknown fit')
    torch.manual_seed(spec['nodes'][task]['initialization_seed'])
    return Teacher(task, targets.load(spec) if task in ('T08', 'T09') else None) if task in TEACHERS else Student()


FIT_KEYS = (*TEACHERS, *STUDENTS)


def objective(model, task, logits, labels, u100=None, offline=None, *, seed=0):
    if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all():
        raise ValueError('Invalid classification logits')
    if task in STUDENTS:
        from hlt_classification.luka_supcon5.model import objective as original
        return original(model, 'student_supcon5', logits, labels, u100, offline)
    if task not in TEACHERS or u100 is not None or offline is not None:
        raise PermissionError('Teacher role/targets differ')
    h = model.h.float(); labels = labels.long(); terms = {'ce': F.cross_entropy(logits.float(), labels)}
    g = torch.Generator(device=h.device).manual_seed(seed)
    with torch.autocast(h.device.type, enabled=False):
        if task in ('T01', 'T02', 'T03', 'T04', 'T07'):
            z = h if task == 'T03' else model.projector(h)
            terms['supcon'] = recipe()['supcon_weights'][task]*supcon(z, labels, .1)
        if task in ('T04', 'T05', 'T06'):
            a, _ = thin(model.inputs, g, .3); b, _ = thin(model.inputs, g, .3)
            _, ha = model.view(a); _, hb = model.view(b); ha, hb = ha.float(), hb.float()
            if task == 'T04': terms['instance'] = 5*instance(model.instance_projector(ha), model.instance_projector(hb))
            elif task == 'T05':
                va, ca = variance_covariance(ha); vb, cb = variance_covariance(hb)
                terms.update(invariance=25*F.mse_loss(ha, hb), variance=25*(va+vb)/2, covariance=(ca+cb)/2)
            else: terms['barlow'] = 5*barlow(ha, hb)
        if task == 'T07':
            v, c = variance_covariance(h); terms.update(variance=25*v, covariance=c)
        if task == 'T08':
            view, mask = thin(model.inputs, g, .4, removed_target=True); _, hv = model.view(view)
            m = model.statistics['moments']['particle']
            target = torch.cat((model.inputs['features'], model.inputs['vectors']), 1).transpose(1, 2).float()
            target = (target-torch.tensor(m['mean'], device=h.device))/torch.tensor(m['scale'], device=h.device)
            pred, count = model.decoder(hv.float())
            terms['reconstruction'] = 5*set_loss(pred, count, target.detach(), mask)
        if task == 'T09':
            raw = {k: v.detach().cpu().numpy() for k, v in model.inputs.items()}
            m = model.statistics['moments']['summary']
            target = torch.tensor((targets.summaries(raw)-np.asarray(m['mean']))/np.asarray(m['scale']), device=h.device, dtype=torch.float32)
            terms['summary'] = 5*F.smooth_l1_loss(model.summary(h), target)
        if task == 'T10':
            view, _ = thin(model.inputs, g, .3); lv, hv = model.view(view)
            terms['ce'] = .5*(terms['ce']+F.cross_entropy(lv.float(), labels))
            terms['consistency'] = 5*128*F.mse_loss(F.normalize(h, dim=1), F.normalize(hv.float(), dim=1))
        if task == 'T11':
            noise = torch.randn(h.shape, generator=g, device=h.device)*h.detach().std(0, unbiased=False).clamp_min(.001)*.5
            terms['ce'] = F.cross_entropy(model.mod.fc(h+noise).float(), labels)
    if set(terms) != term_names(task) or not torch.isfinite(sum(terms.values())):
        raise ValueError('Invalid teacher objective')
    return sum(terms.values()), terms
