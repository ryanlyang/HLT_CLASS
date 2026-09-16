"""Oracle-only source-tagged ParT, retaining installed Weaver's entire forward."""
from __future__ import annotations

import torch
from torch import nn

from .model import DelphesParticleTransformer
from .contracts import artifact
from .salience_learned_graph import NODE_REGISTRY


NODE_ID = "NATIVE_OFFLINE_HLT_CONCAT"
SOURCE_SEED = 2026091601


def node_spec():
    node = {k: v for k, v in NODE_REGISTRY["CE_SINGLE_D000"].payload().items()
            if k not in {"contract", "schema_version", "content_hash"}}
    node.update(node_id=NODE_ID, branch="ORACLE", primary_coordinate=NODE_ID,
                input_protocol="native_tagged_concat_v1", coordinate_exact=None,
                deployable=False, source_embedding_seed=SOURCE_SEED)
    return artifact("NATIVE_CONCAT_NODE", **node)


class SourceTaggedEmbedding(nn.Module):
    def __init__(self, numerical):
        super().__init__()
        self.numerical = numerical
        # New parameters do not consume the matched backbone/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SOURCE_SEED)
            self.source = nn.Embedding(2, 128)
            nn.init.trunc_normal_(self.source.weight, std=.02)

    def forward(self, inputs):
        hidden = self.numerical(inputs[:, :17])
        codes = inputs[:, 17].long()
        if hidden.shape != (len(inputs), inputs.shape[2], 128):
            raise TypeError("Installed Weaver embedding layout differs")
        source = self.source(codes.clamp_min(0)).masked_fill((codes < 0)[..., None], 0)
        return hidden + source.to(hidden.dtype)


class NativeConcatParticleTransformer(DelphesParticleTransformer):
    def __init__(self):
        super().__init__()
        self.mod.embed = SourceTaggedEmbedding(self.mod.embed)

    def forward(self, features, vectors, mask):
        if features.ndim != 3 or features.shape[1] != 18:
            raise ValueError("Concatenation requires 17 numerical fields plus a source code")
        b, _, n = features.shape
        if (vectors.shape != (b, 4, n) or mask.shape != (b, 1, n)
                or mask.dtype != torch.bool or not 2 <= n <= 480):
            raise ValueError("Concatenation shape/capacity differs")
        codes, visible = features[:, 17], mask[:, 0]
        if (not torch.isfinite(features).all() or not torch.isfinite(vectors).all()
                or not (((codes == 0) | (codes == 1) | ~visible).all())
                or not (codes[~visible] == -1).all()
                or not (visible.sum(1) >= 2).all()):
            raise ValueError("Invalid concatenation source codes/mask/values")
        return self.mod(features, v=vectors, mask=mask)


def new_model():
    torch.manual_seed(node_spec()["initialization_seed"])
    return NativeConcatParticleTransformer()


def zero_tag_parity(raw, *, device):
    """Installed forward and numerical-input/parameter gradients, with tags zeroed."""
    tagged = new_model().to(device).eval()
    torch.manual_seed(node_spec()["initialization_seed"])
    ordinary = DelphesParticleTransformer().to(device).eval()
    with torch.no_grad():
        tagged.mod.embed.source.weight.zero_()
    x = torch.from_numpy(raw["features"]).to(device).requires_grad_(True)
    y = x[:, :17].detach().clone().requires_grad_(True)
    v = torch.from_numpy(raw["vectors"]).to(device)
    m = torch.from_numpy(raw["mask"]).to(device)
    left, right = tagged(x, v, m), ordinary(y, v, m)
    torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-6)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(x.grad[:, :17], y.grad, rtol=1e-5, atol=2e-6)
    originals = dict(ordinary.named_parameters())
    for name, parameter in tagged.named_parameters():
        if name.startswith("mod.embed.source."):
            continue
        target = originals[name.replace("mod.embed.numerical.", "mod.embed.")]
        if (parameter.grad is None) != (target.grad is None):
            raise ValueError("Concatenation gradient participation differs")
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, target.grad, rtol=1e-5, atol=2e-6)
    return dict(logits=True, numerical_input_gradients=True, backbone_gradients=True)
