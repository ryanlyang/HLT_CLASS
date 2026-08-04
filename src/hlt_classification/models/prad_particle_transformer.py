"""Parity-oriented Weaver extension for Privileged Relational Attention."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable

import torch
from torch import nn

from hlt_classification.models.particle_transformer import (
    CanonicalParticleTransformer,
    build_particle_transformer,
)
from hlt_classification.prad.relation import (
    ContextualPairRelation,
    GatedRelationBias,
    RelationBiasProjector,
)

PRAD_PARTICLE_TRANSFORMER_CONTRACT = (
    "hlt_classification_prad_particle_transformer_v1"
)
PRAD_RUNTIME_VALIDATION_CONTRACT = (
    "hlt_classification_prad_runtime_validation_v2"
)

PRAD_SCALAR_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16)
PRAD_PID_FEATURE_SLICE = slice(6, 11)


@dataclass(frozen=True)
class PradForwardOutput:
    logits: torch.Tensor
    relation: torch.Tensor
    privileged_bias: torch.Tensor
    semantic_logits: torch.Tensor
    particle_mask: torch.Tensor
    standard_bias: torch.Tensor
    aligned_pair_payload: torch.Tensor | None = None


def standard_four_pair_features(vectors: torch.Tensor) -> torch.Tensor:
    """Reproduce Weaver's symmetric standard-four raw pair quantities."""

    if vectors.ndim != 3 or vectors.shape[1] != 4:
        raise ValueError("Lorentz vectors must have shape [B,4,N]")
    px, py, pz, energy = vectors.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt((px.square() + py.square()).clamp_min(0.0))
    rapidity = 0.5 * torch.log(
        1.0 + (2.0 * pz) / (energy - pz).clamp_min(1.0e-20)
    )
    phi = torch.atan2(py, px)
    pt_i, pt_j = pt.unsqueeze(-1), pt.unsqueeze(-2)
    rapidity_i, rapidity_j = rapidity.unsqueeze(-1), rapidity.unsqueeze(-2)
    phi_i, phi_j = phi.unsqueeze(-1), phi.unsqueeze(-2)
    delta_phi = (phi_i - phi_j + math.pi) % (2.0 * math.pi) - math.pi
    delta = torch.sqrt(
        (rapidity_i - rapidity_j).square() + delta_phi.square()
    )
    pt_min = torch.minimum(pt_i, pt_j)
    log_kt = torch.log((pt_min * delta).clamp_min(1.0e-8))
    log_z = torch.log(
        (pt_min / (pt_i + pt_j).clamp_min(1.0e-8)).clamp_min(1.0e-8)
    )
    log_delta = torch.log(delta.clamp_min(1.0e-8))
    vector_i = vectors.unsqueeze(-1)
    vector_j = vectors.unsqueeze(-2)
    vector_sum = vector_i + vector_j
    momentum2 = vector_sum[:, :3].square().sum(dim=1, keepdim=True)
    mass2 = (vector_sum[:, 3:4].square() - momentum2).clamp_min(1.0e-8)
    log_mass2 = torch.log(mass2)
    pair = torch.cat((log_delta, log_kt, log_z, log_mass2), dim=1).permute(
        0, 2, 3, 1
    ).contiguous()
    return 0.5 * (pair + pair.transpose(1, 2))


def _pid_category(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pid = features[:, PRAD_PID_FEATURE_SLICE, :].transpose(1, 2)
    counts = pid.sum(dim=-1)
    category = torch.full(
        counts.shape, 5, dtype=torch.long, device=features.device
    )
    known = counts == 1
    category[known] = pid.argmax(dim=-1)[known]
    category[~mask] = 5
    return category


class PradParticleTransformer(nn.Module):
    """Extend canonical Weaver ParT while retaining its modules and weights.

    The public ``forward`` consumes only the four canonical HLT tensors. The
    relation-returning method is also HLT-only and exists for training losses
    and diagnostics. Offline oracle bias has a separately named method and is
    therefore easy for deployability audits to reject.
    """

    def __init__(
        self,
        *,
        baseline: CanonicalParticleTransformer | None = None,
        context_depth: int = 2,
        relation_dim: int = 16,
        relation_dropout: float = 0.1,
        semantic_targets: int = 3,
        injection_depth: str = "after_context",
        gate_structure: str = "layer_head",
        retain_standard_pair_bias: bool = True,
        deploy_relation_attention: bool = True,
    ) -> None:
        super().__init__()
        self.baseline = baseline if baseline is not None else build_particle_transformer()
        mod = self.baseline.mod
        required = (
            "embed",
            "pair_embed",
            "blocks",
            "cls_blocks",
            "norm",
            "fc",
            "trimmer",
            "_forward_aggregator",
            "block_ids_with_attn_mask",
        )
        missing = [name for name in required if not hasattr(mod, name)]
        if missing:
            raise TypeError(f"installed Weaver lacks PRAD surfaces: {missing}")
        self.num_layers = len(mod.blocks)
        if self.num_layers < 4:
            raise ValueError("PRAD requires at least four particle blocks")
        if context_depth < 0 or context_depth >= self.num_layers:
            raise ValueError("PRAD context depth leaves no injection blocks")
        self.context_depth = int(context_depth)
        if injection_depth not in {"all", "after_context", "final_half"}:
            raise ValueError("PRAD injection depth differs")
        if injection_depth == "all" and self.context_depth != 0:
            raise ValueError("all-block injection requires context_depth=0")
        first_injection = {
            "all": 0,
            "after_context": self.context_depth,
            "final_half": max(self.context_depth, self.num_layers // 2),
        }[injection_depth]
        self.injection_depth = injection_depth
        self.injection_block_ids = tuple(range(first_injection, self.num_layers))
        self.retain_standard_pair_bias = bool(retain_standard_pair_bias)
        self.deploy_relation_attention = bool(deploy_relation_attention)
        first_block = mod.blocks[0]
        block_mask_policy = mod.block_ids_with_attn_mask
        if (
            isinstance(block_mask_policy, (list, tuple))
            and len(block_mask_policy) == self.num_layers
            and all(isinstance(value, bool) for value in block_mask_policy)
        ):
            self.attention_mask_blocks = tuple(
                index
                for index, enabled in enumerate(block_mask_policy)
                if enabled
            )
        else:
            try:
                self.attention_mask_blocks = tuple(
                    index
                    for index in range(self.num_layers)
                    if index in block_mask_policy
                )
            except TypeError as error:
                raise TypeError(
                    "installed Weaver attention-mask block policy differs"
                ) from error
        if not self.attention_mask_blocks:
            raise ValueError("canonical PRAD baseline applies no pair interaction bias")
        self.attention_heads = int(first_block.num_heads)
        self.context_dim = int(first_block.embed_dim)
        self.relation = ContextualPairRelation(
            context_dim=self.context_dim,
            scalar_dim=len(PRAD_SCALAR_FEATURE_INDICES),
            categorical_cardinalities=(6,),
            relation_dim=relation_dim,
            dropout=relation_dropout,
        )
        self.relation_to_bias = RelationBiasProjector(
            relation_dim, self.attention_heads
        )
        self.gated_bias = GatedRelationBias(
            len(self.injection_block_ids),
            self.attention_heads,
            structure=gate_structure,
        )
        self.semantic_heads = nn.Linear(relation_dim, semantic_targets)

    def no_weight_decay(self) -> set[str]:
        return {f"baseline.{name}" for name in self.baseline.no_weight_decay()}

    def deployable_parameter_count(self) -> int:
        modules = self if self.deploy_relation_attention else self.baseline
        return int(sum(parameter.numel() for parameter in modules.parameters()))

    def load_baseline_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load an ordinary canonical-wrapper state without key guessing."""

        result = self.baseline.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("baseline checkpoint did not load exactly")

    def _prepare(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        pair_payload: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        mod = self.baseline.mod
        if features.ndim != 3 or features.shape[1] != 17:
            raise ValueError("PRAD features must have shape [B,17,N]")
        if vectors.shape != (features.shape[0], 4, features.shape[2]):
            raise ValueError("PRAD Lorentz-vector shape differs")
        if mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("PRAD mask shape differs")
        if mask.dtype != torch.bool:
            raise ValueError("PRAD mask must have boolean dtype")
        # Weaver's own trimmer is retained. All relation-side features are
        # derived after its joint permutation/truncation of x, v, and mask.
        if pair_payload is not None:
            expected = (features.shape[0], features.shape[2], features.shape[2])
            if pair_payload.ndim != 4 or (
                pair_payload.shape[0],
                pair_payload.shape[2],
                pair_payload.shape[3],
            ) != expected:
                raise ValueError(
                    "training pair payload must have shape [B,C,N,N]"
                )
        features, vectors, mask, pair_payload = mod.trimmer(
            features, vectors, mask, pair_payload
        )
        particle_mask = mask.squeeze(1)
        padding_mask = ~particle_mask
        scalar_features = features[
            :, PRAD_SCALAR_FEATURE_INDICES, :
        ].transpose(1, 2).contiguous()
        categories = _pid_category(features, particle_mask)[..., None]
        embedded = mod.embed(features)
        expected_embedding_shape = (
            features.shape[0],
            features.shape[2],
            self.context_dim,
        )
        if embedded.shape != expected_embedding_shape:
            raise TypeError(
                "installed Weaver embedding layout differs from batch-first "
                f"PRAD contract: {tuple(embedded.shape)} != "
                f"{expected_embedding_shape}"
            )
        embedded = embedded.masked_fill(~mask.transpose(1, 2), 0)
        standard_bias = mod.pair_embed(vectors, uu=None, mask=mask)
        expected_bias_shape = (
            features.shape[0],
            self.attention_heads,
            features.shape[2],
            features.shape[2],
        )
        if standard_bias.shape != expected_bias_shape:
            raise TypeError(
                "installed Weaver pair-bias layout differs from PRAD contract: "
                f"{tuple(standard_bias.shape)} != {expected_bias_shape}"
            )
        raw_pair = standard_four_pair_features(vectors)
        return (
            embedded,
            padding_mask,
            particle_mask,
            scalar_features,
            categories,
            standard_bias,
            raw_pair,
            pair_payload,
        )

    def _forward_impl(
        self,
        features: torch.Tensor,
        vectors: torch.Tensor,
        mask: torch.Tensor,
        *,
        oracle_bias: torch.Tensor | None = None,
        training_pair_payload: torch.Tensor | None = None,
        remove_standard_bias: bool = False,
    ) -> PradForwardOutput:
        if oracle_bias is not None and training_pair_payload is not None:
            raise ValueError("oracle bias and training pair payload are exclusive")
        pair_payload = (
            oracle_bias if oracle_bias is not None else training_pair_payload
        )
        mod = self.baseline.mod
        (
            hidden,
            padding_mask,
            particle_mask,
            scalar_features,
            categories,
            standard_bias,
            raw_pair,
            aligned_pair_payload,
        ) = self._prepare(features, vectors, mask, pair_payload)
        for index in range(self.context_depth):
            hidden = mod.blocks[index](
                hidden,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=(
                    standard_bias
                    if index in self.attention_mask_blocks
                    else None
                ),
            )
        relation = self.relation(
            hidden,
            raw_pair,
            particle_mask,
            scalar_features=scalar_features,
            categorical_features=categories,
        )
        predicted_bias = self.relation_to_bias(relation, particle_mask)
        privileged_bias = (
            predicted_bias if oracle_bias is None else aligned_pair_payload
        )
        assert privileged_bias is not None
        if privileged_bias.shape != standard_bias.shape:
            raise ValueError("PRAD oracle/predicted bias shape differs from Weaver")
        injection_lookup = {
            block_id: gate_id
            for gate_id, block_id in enumerate(self.injection_block_ids)
        }
        for index in range(self.context_depth, self.num_layers):
            if index in injection_lookup:
                combined = self.gated_bias(
                    standard_bias,
                    privileged_bias,
                    injection_layer=injection_lookup[index],
                    remove_standard_bias=(
                        remove_standard_bias or not self.retain_standard_pair_bias
                    ),
                )
            else:
                combined = standard_bias
            hidden = mod.blocks[index](
                hidden,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=(
                    combined if index in self.attention_mask_blocks else None
                ),
            )
        pooled = mod._forward_aggregator(hidden, padding_mask)
        logits = mod.fc(pooled)
        return PradForwardOutput(
            logits=logits,
            relation=relation,
            privileged_bias=predicted_bias,
            semantic_logits=self.semantic_heads(relation),
            particle_mask=particle_mask,
            standard_bias=standard_bias,
            aligned_pair_payload=aligned_pair_payload,
        )

    def forward_with_relations(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> PradForwardOutput:
        del points
        return self._forward_impl(features, lorentz_vectors, mask)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.deploy_relation_attention:
            return self.baseline(points, features, lorentz_vectors, mask)
        return self.forward_with_relations(
            points, features, lorentz_vectors, mask
        ).logits

    def forward_training(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        *,
        pair_payload: torch.Tensor,
    ) -> PradForwardOutput:
        """Training-only path that aligns packed pair targets with trimming."""

        del points
        return self._forward_impl(
            features,
            lorentz_vectors,
            mask,
            training_pair_payload=pair_payload,
        )

    def forward_oracle(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        *,
        offline_teacher_bias: torch.Tensor,
    ) -> PradForwardOutput:
        """Explicitly nondeployable oracle diagnostic."""

        del points
        return self._forward_impl(
            features,
            lorentz_vectors,
            mask,
            oracle_bias=offline_teacher_bias,
        )


def build_prad_particle_transformer(**kwargs: Any) -> PradParticleTransformer:
    return PradParticleTransformer(**kwargs)


def validate_prad_runtime(
    *,
    device: str = "cpu",
    seed: int = 20260804,
    batch_size: int = 2,
    particles: int = 8,
    baseline_factory: Callable[[], CanonicalParticleTransformer] = build_particle_transformer,
) -> dict[str, Any]:
    """Exercise PRAD mechanics against the selected Weaver runtime.

    This is a synthetic mechanics attestation, not a data or physics result.
    Production uses the default installed-Weaver factory; tests may inject an
    interface-compatible double.
    """

    if batch_size < 2 or particles < 4:
        raise ValueError("PRAD runtime validation requires B>=2 and N>=4")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("PRAD runtime device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PRAD validation requested but CUDA is unavailable")
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    student = PradParticleTransformer(
        baseline=baseline_factory(), relation_dropout=0.0
    ).to(target)
    student.eval()
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    points = torch.randn(batch_size, 2, particles, generator=generator, device=target)
    features = torch.randn(
        batch_size, 17, particles, generator=generator, device=target
    )
    features[:, PRAD_PID_FEATURE_SLICE, :] = 0
    categories = torch.randint(
        0, 5, (batch_size, particles), generator=generator, device=target
    )
    features.scatter_(1, (categories + 6).unsqueeze(1), 1.0)
    vectors = torch.randn(
        batch_size, 4, particles, generator=generator, device=target
    )
    vectors[:, 3, :] = vectors[:, :3, :].square().sum(1).add(1.0).sqrt()
    mask = torch.ones(batch_size, 1, particles, dtype=torch.bool, device=target)
    mask[0, :, -2:] = False
    mask[1, :, -1:] = False
    labels = torch.arange(batch_size, device=target) % 10

    with torch.no_grad():
        baseline_logits = student.baseline(points, features, vectors, mask)
        zero_gate_output = student.forward_with_relations(
            points, features, vectors, mask
        )
    maximum_zero_gate_error = float(
        (baseline_logits - zero_gate_output.logits).abs().max().cpu()
    )

    teacher = PradParticleTransformer(
        baseline=baseline_factory(), relation_dropout=0.0
    ).to(target)
    teacher.eval()
    teacher.requires_grad_(False)
    with torch.no_grad():
        teacher_output = teacher.forward_with_relations(points, features, vectors, mask)

    student.zero_grad(set_to_none=True)
    with torch.no_grad():
        student.gated_bias.raw_gates.fill_(0.2)
    student_output = student.forward_with_relations(points, features, vectors, mask)
    mechanics_loss = (
        torch.nn.functional.cross_entropy(student_output.logits, labels)
        + torch.nn.functional.smooth_l1_loss(
            student_output.relation, teacher_output.relation.detach()
        )
        + torch.nn.functional.smooth_l1_loss(
            student_output.privileged_bias,
            teacher_output.privileged_bias.detach(),
        )
    )
    mechanics_loss.backward()
    relation_gradient_norm = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in student.relation.parameters()
            if parameter.grad is not None
        ).sqrt().cpu()
    )
    gate_gradient_norm = float(
        student.gated_bias.raw_gates.grad.detach().float().norm().cpu()
        if student.gated_bias.raw_gates.grad is not None
        else 0.0
    )
    teacher_has_gradient = any(
        parameter.grad is not None for parameter in teacher.parameters()
    )
    with torch.no_grad():
        public_logits = student(points, features, vectors, mask)

    # Reproduce the registered pair-supervised Stage-A freeze schedule.  This
    # is deliberately separate from the all-trainable mechanics loss above:
    # optimized CUDA attention kernels can have different backward surfaces
    # when only a differentiable additive attention bias remains.
    for parameter in student.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in student.relation.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        student.gated_bias.raw_gates.zero_()
    student.train()
    stage_a_output = student.forward_training(
        points,
        features,
        vectors,
        mask,
        pair_payload=torch.zeros(
            batch_size,
            1,
            particles,
            particles,
            device=target,
        ),
    )
    stage_a_output.logits.retain_grad()
    stage_a_valid = (
        stage_a_output.particle_mask[:, :, None]
        & stage_a_output.particle_mask[:, None, :]
    )
    stage_a_valid &= ~torch.eye(
        stage_a_valid.shape[-1], dtype=torch.bool, device=target
    )[None]
    stage_a_semantic_valid = stage_a_valid[..., None].expand_as(
        stage_a_output.semantic_logits
    )
    # Local imports avoid a model/training import cycle while ensuring the
    # standalone runtime worker exercises the production loss implementation.
    from hlt_classification.prad.experiments import CORE_EXPERIMENTS
    from hlt_classification.prad.training import student_loss

    stage_a_loss = student_loss(
        output=stage_a_output,
        labels=labels,
        experiment=CORE_EXPERIMENTS["E5"],
        stage="A",
        semantic_targets=torch.zeros_like(stage_a_output.semantic_logits),
        semantic_valid=stage_a_semantic_valid,
        semantic_positive_weights=torch.ones(3, device=target),
    ).total
    stage_a_loss.backward()
    stage_a_relation_gradient_norm = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in student.relation.parameters()
            if parameter.grad is not None
        ).sqrt().cpu()
    )
    stage_a_relation_gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in student.relation.parameters()
    )

    valid_keys = zero_gate_output.particle_mask[:, None, None, :]
    centered_sum = (zero_gate_output.privileged_bias * valid_keys).sum(dim=-1)
    checks = {
        "zero_gate_baseline_logits_below_1e-6": maximum_zero_gate_error < 1.0e-6,
        "public_forward_is_hlt_only": list(
            inspect.signature(student.forward).parameters
        ) == ["points", "features", "lorentz_vectors", "mask"],
        "public_logits_finite": bool(torch.isfinite(public_logits).all()),
        "relation_is_symmetric": bool(
            torch.allclose(
                student_output.relation,
                student_output.relation.transpose(1, 2),
                atol=1.0e-6,
                rtol=0.0,
            )
        ),
        "relation_bias_is_key_centered": bool(centered_sum.abs().max() < 1.0e-5),
        "relation_module_receives_gradient": relation_gradient_norm > 0.0,
        "nonzero_gate_receives_gradient": gate_gradient_norm > 0.0,
        "teacher_parameters_are_frozen": not teacher_has_gradient,
        "mechanics_loss_finite": bool(torch.isfinite(mechanics_loss)),
        "stage_a_logits_excluded_from_backward": stage_a_output.logits.grad is None,
        "stage_a_relation_gradient_finite_nonzero": (
            stage_a_relation_gradients_finite
            and stage_a_relation_gradient_norm > 0.0
        ),
    }
    return {
        "contract": PRAD_RUNTIME_VALIDATION_CONTRACT,
        "schema_version": 2,
        "scope": "synthetic_runtime_mechanics_not_scientific_performance",
        "device": str(target),
        "seed": seed,
        "batch_size": batch_size,
        "particles": particles,
        "torch_version": torch.__version__,
        "maximum_zero_gate_logit_error": maximum_zero_gate_error,
        "relation_gradient_norm": relation_gradient_norm,
        "gate_gradient_norm": gate_gradient_norm,
        "stage_a_relation_gradient_norm": stage_a_relation_gradient_norm,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "PRAD_PARTICLE_TRANSFORMER_CONTRACT",
    "PRAD_RUNTIME_VALIDATION_CONTRACT",
    "PradForwardOutput",
    "PradParticleTransformer",
    "build_prad_particle_transformer",
    "standard_four_pair_features",
    "validate_prad_runtime",
]
