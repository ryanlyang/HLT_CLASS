import math
import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.jetclass2_delphes.reader import Particles
from hlt_classification.jetclass2_delphes.offline_aux.targets import summarize, histogram, PAIR
from hlt_classification.jetclass2_delphes.offline_aux.normalization import fit
from hlt_classification.jetclass2_delphes.offline_aux.contracts import learning_rate, seed
from hlt_classification.jetclass2_delphes.offline_aux.roles import selection_indices, authorize_role
from hlt_classification.jetclass2_delphes.offline_aux.model import create_model, state_hash
from hlt_classification.jetclass2_delphes.offline_aux.losses import objectives, kl


def particles(pt=(1., 2., 3.), phi=(0., .1, .3), species=(0, 1, 2)):
    a = np.zeros((len(pt), 14), np.float32)
    a[:, 0] = np.array(pt) * np.cos(phi)
    a[:, 1] = np.array(pt) * np.sin(phi)
    a[:, 3] = pt
    for i, k in enumerate(species):
        a[i, 5+k] = 1
        a[i, 4] = int(k in (0, 3, 4))
    return Particles(a)


def test_targets_hand_example_and_invariance():
    p = particles()
    v, valid, diag = summarize(p)
    assert valid
    np.testing.assert_allclose(v[:5], np.log1p([1, 1, 1, 0, 0]))
    np.testing.assert_allclose(v[5:10], [1/6, 2/6, 3/6, 0, 0])
    assert v[20:28].sum() == pytest.approx(1)
    scaled = p.values.copy(); scaled[:, :4] *= 2
    np.testing.assert_allclose(v, summarize(Particles(scaled))[0], atol=2e-6)
    np.testing.assert_allclose(v, summarize(Particles(p.values[::-1]))[0], atol=2e-6)
    rotated = particles(phi=(1., 1.1, 1.3))
    np.testing.assert_allclose(v, summarize(rotated)[0], atol=2e-6)
    np.testing.assert_allclose(histogram(np.array([0., .05, .1, 12.]), np.ones(4)/4, PAIR),
                               [.25, .25, .25, 0, 0, 0, 0, .25])
    single, valid, _ = summarize(particles((1.,), (0.,), (0,)))
    assert not valid and np.all(single[20:] == 0) and single[11] == 0
    bad = p.values.copy(); bad[:, 3] = .1
    with pytest.raises(ValueError, match="mass"):
        summarize(Particles(bad))


def test_normalization_train_only():
    v = np.stack([summarize(particles())[0]] * 20)
    n = fit(v, np.ones(20, bool), np.arange(20), role="TRAIN", train_bank_sha256="a"*64)
    assert n["scale"] == [.001]*7
    with pytest.raises(PermissionError):
        fit(v, np.ones(20, bool), np.arange(20), role="VAL_SELECT", train_bank_sha256="a"*64)


def test_schedule_roles_seeds():
    assert learning_rate(3*1954, 1954) == 3e-4
    assert learning_rate(45*1954, 1954) == 3e-4
    assert learning_rate(60*1954, 1954) == 1.5e-5
    assert learning_rate(100*1954, 1954) == 1.5e-5
    assert math.ceil(500000/256) == 1954 and 500000%256 == 32
    labels = np.repeat(np.arange(11), 10)
    ids = np.arange(110*32, dtype=np.uint8).reshape(110, 32)
    a = selection_indices(labels, ids, "a"*64, 33)
    assert len(a) == 33 and np.bincount(labels[a]).tolist() == [3]*11
    for role in ("VAL_REPORT", "final_test", "validation"):
        with pytest.raises(PermissionError):
            authorize_role(role, {})
    assert seed("DISCOVERY", "shared_init") != seed("CONFIRM_01", "shared_init")


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.Module()
        self.mod.fc = nn.Sequential(nn.Linear(128, 11))
        self.embed = nn.Linear(17, 128)
        self.calls = 0

    def forward(self, features, vectors, mask):
        self.calls += 1
        return self.mod.fc(self.embed(features.mean(-1)))


def test_paired_heads_single_forward_and_losses():
    comp = create_model("COMP", "DISCOVERY", factory=Toy)
    both = create_model("BOTH", "DISCOVERY", factory=Toy)
    assert state_hash(comp.backbone.state_dict()) == state_hash(both.backbone.state_dict())
    assert state_hash(comp.heads["counts"].state_dict()) == state_hash(both.heads["counts"].state_dict())
    f = torch.randn(4, 17, 16)
    args = dict(features=f, vectors=torch.zeros(4, 4, 16), mask=torch.ones(4, 1, 16, dtype=torch.bool))
    logits, heads, rep = both(**args)
    assert both.backbone.calls == 1 and not both.backbone.mod.fc._forward_pre_hooks
    targets = torch.tensor(np.stack([summarize(particles())[0]]*4))
    total, parts = objectives(logits, heads, torch.arange(4), targets, torch.zeros(4, 7),
                              torch.zeros(4, dtype=torch.bool), arm="BOTH", coefficient=.3)
    assert torch.allclose(total, parts["ce"] + .15*(parts["composition"]+parts["structure"]))
    total.backward()
    assert both.heads["pair"][0].weight.grad is None
    assert both.backbone.embed.weight.grad.abs().sum() > 0
    assert torch.allclose(logits, both.deployable()(**args))
    assert kl(torch.tensor([[1., 0.]]), torch.zeros(1, 2)) == pytest.approx(math.log(2))


def test_zero_lambda_shared_update_parity_and_rng_isolation():
    torch.manual_seed(761)
    before = torch.get_rng_state().clone()
    a = create_model("CE", "DISCOVERY", factory=Toy)
    b = create_model("BOTH", "DISCOVERY", factory=Toy)
    assert torch.equal(before, torch.get_rng_state())
    # Covers scalar buffers present in actual Weaver BatchNorm modules.
    assert state_hash({"counter": torch.tensor(1)}) != state_hash({"counter": torch.tensor(2)})
    x = torch.randn(7, 17, 16)
    args = dict(features=x, vectors=torch.zeros(7, 4, 16), mask=torch.ones(7, 1, 16, dtype=torch.bool))
    target = torch.tensor(np.stack([summarize(particles())[0]]*7))
    from hlt_classification.jetclass2_delphes.offline_aux.training import optimizer
    for model in (a, b):
        opt = optimizer(model)
        logits, heads, _ = model(**args)
        loss, _ = objectives(logits, heads, torch.arange(7), target, torch.zeros(7, 7),
                             torch.ones(7, dtype=torch.bool), arm=model.arm, coefficient=0.)
        loss.backward(); opt.step()
    for key in a.backbone.state_dict():
        torch.testing.assert_close(a.backbone.state_dict()[key], b.backbone.state_dict()[key], rtol=0, atol=0)


def test_phi_wrapping_and_tolerated_mass_roundoff():
    p = particles(pt=(2., 2.), phi=(np.pi-.01, -np.pi+.01), species=(0, 1))
    v, valid, _ = summarize(p)
    assert valid and v[20] == pytest.approx(1) and v[11] < .02
    a = particles((1.,), (0.,), (0,)).values.copy()
    a[:, 3] = np.nextafter(np.float32(1), np.float32(0))
    v, _, diagnostic = summarize(Particles(a))
    assert diagnostic["mass_clamped"] == 1 and v[10] == 0
    a[:, 3] = .99
    with pytest.raises(ValueError):
        summarize(Particles(a))
