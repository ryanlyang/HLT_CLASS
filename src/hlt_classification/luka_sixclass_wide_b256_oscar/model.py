"""Six-output native graph; only true-label CE reaches classifier weights."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, model_config as old_config
from hlt_classification.models.particle_transformer import load_weaver_particle_transformer_class
from hlt_classification.luka_u100_contrastive.losses import scores
from . import contracts as c


def main_loss(parts, stage):
    c.main_policy(stage)  # Reject stages outside this experiment.
    return parts['ce']


def objective(net, raw, indices, banks, *, update, steps, device, sampler=None, candidates=None):
    names = c.TEACHERS[net.stage]
    if set(banks) != set(names):
        raise ValueError('Wrong stage target inventory')
    if (sampler is None) != (net.stage != 'DE000'):
        raise ValueError('Only final DE000 uses contrastive matching')
    net.capture = bool(names)
    logits = forward(net, raw, device, bf16=torch.device(device).type=='cuda').float()
    labels = torch.as_tensor(raw['labels'], device=device, dtype=torch.long)
    parts = {'ce': F.cross_entropy(logits, labels)}
    probabilities = {t:torch.as_tensor(banks[t].probabilities[indices], device=device).detach().float() for t in names}
    if any(p.shape != (len(indices),6) for p in probabilities.values()):
        raise ValueError('Native T2 teacher probability shape differs')
    total = main_loss(parts, net.stage)
    for teacher in names:
        target = torch.as_tensor(banks[teacher].targets[indices],device=device).detach().float()
        if target.shape != (len(indices),128):
            raise ValueError('Embedding target shape differs')
        e = F.smooth_l1_loss(net.predictions[teacher],target,beta=1,reduction='mean')
        k = 4*F.kl_div(F.log_softmax(net.auxiliary_logits[teacher]/2.,dim=-1),probabilities[teacher],reduction='batchmean')
        parts['embedding_'+teacher]=e; parts['kd_'+teacher]=k
        total = total + (10/len(names))*e + (1/len(names))*k
    if sampler is not None:
        if candidates is None or candidates.shape!=(len(indices),1025) or not np.array_equal(candidates[:,0],indices):
            raise ValueError('Contrastive positive identity differs')
        parts['matching'] = F.cross_entropy(scores(net.query,sampler.candidate_keys(candidates)),
                                            torch.zeros(len(indices),device=device,dtype=torch.long))
        parts['weighted_matching'] = c.coefficient(net.stage,update,steps)*parts['matching']
        total = total + parts['weighted_matching']
    parts['total'] = total
    if not torch.isfinite(torch.stack(list(parts.values()))).all():
        raise ValueError('Nonfinite objective')
    return total, parts

def model_config():
    return dict(old_config(), num_classes=len(c.CLASS_NAMES))


def model_contract():
    return c.artifact('MODEL', factory='installed_weaver_particle_transformer', config=model_config(),
                      class_names=list(c.CLASS_NAMES), input_keys=['features','vectors','mask'])


class NativeParticleTransformer(DelphesParticleTransformer):
    def __init__(self):
        nn.Module.__init__(self)
        self.mod = load_weaver_particle_transformer_class()(**model_config())


def classifier(net):
    fc = net.mod.fc
    if (not isinstance(fc, nn.Sequential) or len(fc) != 1 or not isinstance(fc[0], nn.Linear)
            or (fc[0].in_features, fc[0].out_features) != (128,6)):
        raise ValueError('Expected registered six-output native classifier')
    return fc


def head_input(net, features, vectors, mask):
    tapped = []
    handle = classifier(net).register_forward_pre_hook(lambda _m, args: tapped.append(args[0]))
    try:
        logits = NativeParticleTransformer.forward(net, features, vectors, mask)
    finally:
        handle.remove()
    if len(tapped) != 1 or tapped[0].shape != (len(logits),128):
        raise ValueError('Native representation tap differs')
    return logits, tapped[0]


def forward(net, raw, device, *, bf16, capture=False):
    tensors = [torch.from_numpy(raw[k]).to(device) for k in ('features','vectors','mask')]
    tapped = []
    handle = classifier(net).register_forward_pre_hook(lambda _m, args: tapped.append(args[0])) if capture else None
    try:
        with torch.autocast(device_type=torch.device(device).type, enabled=bf16, dtype=torch.bfloat16):
            logits = net(*tensors)
    finally:
        if handle is not None: handle.remove()
    if logits.shape != (len(raw['labels']),6):
        raise ValueError('Six-class native logits required')
    if capture:
        if len(tapped) != 1 or tapped[0].shape != (len(logits),128):
            raise ValueError('Teacher embedding tap differs')
        return logits, tapped[0]
    return logits


class Student(NativeParticleTransformer):
    def __init__(self, stage):
        config=c.node(stage)
        super().__init__()
        self.stage=stage
        self.predictors=nn.ModuleDict(); self.logit_heads=nn.ModuleDict()
        for teacher in c.TEACHERS[stage]:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['embedding_seeds'][teacher])
                self.predictors[teacher]=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['logit_seeds'][teacher])
                self.logit_heads[teacher]=nn.Linear(128,6)
        self.matching=None
        if stage=='DE000':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['matching_seed'])
                self.matching=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,128))
        self.capture=False
        self.clear()

    def clear(self):
        self.predictions={}; self.auxiliary_logits={}; self.query=None

    def forward(self, features, vectors, mask):
        self.clear()
        if not self.capture:
            return super().forward(features,vectors,mask)
        logits,h=head_input(self,features,vectors,mask)
        with torch.autocast(h.device.type,enabled=False):
            self.predictions={t:head(h.float()) for t,head in self.predictors.items()}
            self.auxiliary_logits={t:head(h.float()) for t,head in self.logit_heads.items()}
            if self.matching is not None: self.query=self.matching(h.float())
        return logits

    def native_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if not k.startswith(('predictors.','logit_heads.','matching.'))}

    def deployed(self):
        # Deployment construction must not perturb the scientific RNG stream.
        with torch.random.fork_rng(devices=[]):
            result=NativeParticleTransformer()
        result.load_state_dict(self.native_state(),strict=True)
        return result.eval()


def new_model(stage):
    torch.manual_seed(c.node(stage)['initialization_seed'])
    return Student(stage)



def installed_parity(cache, *, device) -> dict:
    torch.manual_seed(71)
    wrapper = NativeParticleTransformer().to(device).eval()
    direct = load_weaver_particle_transformer_class()(**model_config()).to(device).eval()
    direct.load_state_dict(wrapper.mod.state_dict())
    raw = cache.batch(np.arange(min(4, len(cache))))
    x1 = torch.tensor(raw["features"], device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    vectors, mask = (torch.tensor(raw[k], device=device) for k in ("vectors", "mask"))
    y1, y2 = wrapper(x1, vectors, mask), direct(x2, v=vectors, mask=mask)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
    y1.square().mean().backward()
    y2.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
    if not torch.isfinite(y1).all() or not torch.isfinite(x1.grad).all():
        raise ValueError("Nonfinite parity logits/input gradients")
    for (_, left), (_, right) in zip(wrapper.mod.named_parameters(), direct.named_parameters()):
        if left.grad is None or right.grad is None:
            if left.grad is not None or right.grad is not None:
                raise ValueError("Parity parameter gradient topology differs")
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
            if not torch.isfinite(left.grad).all():
                raise ValueError("Nonfinite parity parameter gradient")
    return c.artifact("WEAVER_PARITY", model=model_contract(), device=str(device), passed=True,
                    forward_and_feature_and_parameter_gradients=True)

