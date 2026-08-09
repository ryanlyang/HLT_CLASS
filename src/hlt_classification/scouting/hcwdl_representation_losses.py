"""Exact FP32 HCWDL-RKD representation objectives.

The functions in this module consume live student surfaces and compact,
detached teacher summaries.  They never compare particle indices across
views.  Every reduction and eligibility rule is frozen by
``HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal, Mapping, Sequence

import numpy as np

from .hcwdl_representation_kernels import (
    SpectralKernelResources,
    cached_finite_mmd,
    finite_spectral_features,
    normalized_weights,
)
from .matching import p4_kinematics


RSET_CONTRACT: Final = "HCWDL_REP_SET/v1"
RREL_CONTRACT: Final = "HCWDL_REP_REL/v1"
FAMILY_CONTRACT: Final = "HCWDL_REP_TOKEN_FAMILY/v1"
CHARGED_FAMILY: Final = 0
NEUTRAL_FAMILY: Final = 1
CONTRADICTION_FAMILY: Final = 2
MALFORMED_FAMILY: Final = 3
PADDED_FAMILY: Final = -1
DIRECT_CLASSIFICATION: Final = 0
CHARGE_ONLY_CLASSIFICATION: Final = 1
CONTRADICTION_CLASSIFICATION: Final = 2
MALFORMED_CLASSIFICATION: Final = 3
PADDED_CLASSIFICATION: Final = -1
RELATION_STRATA: Final = ("local", "medium", "wide")
RELATION_EDGES: Final = (0.05, 0.20)
RHO_REPRESENTATION: Final = 0.10
ORTHOGONALITY_COEFFICIENT: Final = 1.0e-3


@dataclass(frozen=True)
class FamilyClassification:
    family_codes: object
    reason_codes: object


@dataclass(frozen=True)
class ReducedRows:
    loss: object
    per_jet: object
    eligible: object
    eligible_count: int
    denominator: object


@dataclass(frozen=True)
class JetLossResult:
    loss: object
    direct: object
    gram: object
    direct_rows: object
    gram_squared_errors: object
    gram_pair_weights: object


@dataclass(frozen=True)
class SetLossResult:
    reduction: ReducedRows
    active_family_count: object
    family_eligible: object
    family_losses: object


@dataclass(frozen=True)
class TokenKernelTargets:
    means: object
    present: object


@dataclass(frozen=True)
class RelationSketches:
    means: object
    eligible: object
    pair_counts: object
    effective_sample_sizes: object


@dataclass(frozen=True)
class RelationLossResult:
    reduction: ReducedRows
    active_family_count: object
    active_stratum_count: object
    family_stratum_eligible: object
    family_stratum_losses: object
    student_sketches: RelationSketches


@dataclass(frozen=True)
class ProjectionDiagnostic:
    name: str
    singular_values: tuple[float, ...]
    condition_number: float
    poorly_conditioned: bool
    orthogonality_loss: float


@dataclass(frozen=True)
class ScheduledRepresentationLoss:
    total: object
    scientific: object
    orthogonality: object
    jet_coefficient: float
    set_coefficient: float
    relation_coefficient: float
    ramp_jet_set: float
    ramp_relation: float


def classify_hlt_token_families(charge, pid_flags, mask) -> FamilyClassification:
    """Classify raw HLT tokens without silently resolving contradictions."""

    import torch

    charge_value = torch.as_tensor(charge)
    flags = torch.as_tensor(pid_flags, device=charge_value.device)
    visible = torch.as_tensor(mask, device=charge_value.device, dtype=torch.bool)
    if visible.ndim == 3 and visible.shape[1] == 1:
        visible = visible[:, 0]
    if charge_value.ndim != 2 or visible.shape != charge_value.shape:
        raise ValueError("charge/mask must be [batch,tokens]")
    if flags.shape != (*charge_value.shape, 5):
        raise ValueError("PID flags must be [batch,tokens,5]")
    if visible.any():
        if not torch.isfinite(charge_value[visible]).all():
            raise ValueError("visible raw charge is nonfinite")
        valid_charge = (
            (charge_value[visible] == -1)
            | (charge_value[visible] == 0)
            | (charge_value[visible] == 1)
        )
        if not valid_charge.all():
            raise ValueError("visible raw charge lies outside {-1,0,+1}")
        if not torch.isfinite(flags[visible]).all():
            raise ValueError("visible raw PID flags are nonfinite")

    family = torch.full(
        charge_value.shape, PADDED_FAMILY, dtype=torch.int8,
        device=charge_value.device,
    )
    reason = torch.full_like(family, PADDED_CLASSIFICATION)
    if not visible.any():
        return FamilyClassification(family, reason)

    binary = ((flags == 0) | (flags == 1)).all(-1)
    count = flags.sum(-1)
    known = binary & (count == 1)
    unknown = binary & (count == 0)
    malformed = ~(known | unknown)
    charge_family = torch.where(
        charge_value == 0,
        torch.full_like(family, NEUTRAL_FAMILY),
        torch.full_like(family, CHARGED_FAMILY),
    )
    pid_index = flags.argmax(-1)
    pid_family = torch.where(
        pid_index <= 2,
        torch.full_like(family, CHARGED_FAMILY),
        torch.full_like(family, NEUTRAL_FAMILY),
    )
    agree = known & (pid_family == charge_family)
    contradiction = known & ~agree
    family[visible & agree] = pid_family[visible & agree].to(torch.int8)
    reason[visible & agree] = DIRECT_CLASSIFICATION
    family[visible & unknown] = charge_family[visible & unknown]
    reason[visible & unknown] = CHARGE_ONLY_CLASSIFICATION
    family[visible & contradiction] = CONTRADICTION_FAMILY
    reason[visible & contradiction] = CONTRADICTION_CLASSIFICATION
    family[visible & malformed] = MALFORMED_FAMILY
    reason[visible & malformed] = MALFORMED_CLASSIFICATION
    return FamilyClassification(family, reason)


def token_weights(vectors, mask):
    """Return the exact half-uniform, half-softened-pT per-token weights."""

    import torch

    p4 = torch.as_tensor(vectors)
    visible = torch.as_tensor(mask, device=p4.device, dtype=torch.bool)
    if visible.ndim == 3 and visible.shape[1] == 1:
        visible = visible[:, 0]
    if p4.ndim != 3 or p4.shape[1] != 4 or visible.shape != (p4.shape[0], p4.shape[2]):
        raise ValueError("vectors/mask must be [batch,4,tokens]/[batch,tokens]")
    p4 = p4.float()
    pt = torch.sqrt(p4[:, 0].square() + p4[:, 1].square())
    if visible.any() and (
        not torch.isfinite(pt[visible]).all() or not (pt[visible] > 0).all()
    ):
        raise ValueError("visible token pT must be finite and positive")
    count = visible.sum(-1)
    safe_count = count.clamp_min(1).to(torch.float32)
    sqrt_pt = torch.where(visible, torch.sqrt(pt), torch.zeros_like(pt))
    denominator = sqrt_pt.sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    result = torch.where(
        visible,
        0.5 / safe_count[:, None] + 0.5 * sqrt_pt / denominator[:, None],
        torch.zeros_like(pt),
    )
    if not torch.isfinite(result).all():
        raise FloatingPointError("token weights are nonfinite")
    nonempty = count > 0
    if nonempty.any() and not torch.allclose(
        result[nonempty].sum(-1), torch.ones_like(safe_count[nonempty]),
        atol=1.0e-6, rtol=1.0e-6,
    ):
        raise RuntimeError("token weights do not normalize")
    return result


def class_weighted_eligible_mean(
    per_jet, labels, class_weights, eligible,
) -> ReducedRows:
    """Reduce eligible per-jet losses under authenticated class weights."""

    import torch

    rows = torch.as_tensor(per_jet).float()
    label = torch.as_tensor(labels, device=rows.device, dtype=torch.long)
    active = torch.as_tensor(eligible, device=rows.device, dtype=torch.bool)
    weights = torch.as_tensor(class_weights, device=rows.device, dtype=torch.float32)
    if rows.ndim != 1 or label.shape != rows.shape or active.shape != rows.shape:
        raise ValueError("representation rows/labels/eligibility must be [batch]")
    if weights.ndim != 1 or weights.numel() != 15:
        raise ValueError("representation class weights must contain 15 values")
    if label.numel() and (int(label.min()) < 0 or int(label.max()) >= 15):
        raise ValueError("representation labels lie outside 15 classes")
    if not torch.isfinite(rows).all() or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise FloatingPointError("representation rows/class weights are invalid")
    selected = weights[label] * active.to(torch.float32)
    denominator = selected.sum()
    if bool(active.any()):
        loss = (selected * rows).sum() / denominator
    else:
        loss = rows.sum() * 0.0
    return ReducedRows(loss, rows, active, int(active.sum()), denominator)


def _layer_norm(value):
    import torch.nn.functional as functional
    return functional.layer_norm(value.float(), (128,), weight=None, bias=None, eps=1.0e-5)


def _unit(value):
    import torch.nn.functional as functional
    return functional.normalize(value.float(), p=2, dim=-1, eps=1.0e-12)


def jet_representation_loss(
    student_jet,
    teacher_jet,
    projection,
    *,
    labels,
    class_weights,
) -> JetLossResult:
    """Paired jet cosine plus raw-space class-weighted Gram geometry."""

    import torch

    student = torch.as_tensor(student_jet).float()
    teacher = torch.as_tensor(teacher_jet, device=student.device).float()
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[1] != 128:
        raise ValueError("jet representations must be paired [batch,128]")
    if teacher.requires_grad:
        raise ValueError("teacher jet targets must be detached")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("jet representations are nonfinite")
    projected = projection(_layer_norm(student)).float()
    q_student = _unit(projected)
    q_teacher = _unit(_layer_norm(teacher))
    direct_rows = 1.0 - (q_student * q_teacher).sum(-1)
    direct = class_weighted_eligible_mean(
        direct_rows, labels, class_weights,
        torch.ones(len(student), dtype=torch.bool, device=student.device),
    ).loss

    normalized_student = _unit(_layer_norm(student))
    normalized_teacher = _unit(_layer_norm(teacher))
    gram_errors = (
        normalized_student @ normalized_student.transpose(0, 1)
        - normalized_teacher @ normalized_teacher.transpose(0, 1)
    ).square()
    batch = len(student)
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=student.device)
    labels_tensor = torch.as_tensor(labels, device=student.device, dtype=torch.long)
    weights = torch.as_tensor(class_weights, device=student.device, dtype=torch.float32)
    pair_weights = torch.sqrt(weights[labels_tensor, None] * weights[labels_tensor][None, :])
    pair_weights = torch.where(off_diagonal, pair_weights, torch.zeros_like(pair_weights))
    pair_denominator = pair_weights.sum()
    gram = (
        (pair_weights * gram_errors).sum() / pair_denominator
        if batch > 1 else gram_errors.sum() * 0.0
    )
    loss = 0.75 * direct + 0.25 * gram
    if not torch.isfinite(loss):
        raise FloatingPointError("jet representation loss is nonfinite")
    return JetLossResult(loss, direct, gram, direct_rows, gram_errors, pair_weights)


def ordinary_set_representation_loss(
    student_tokens,
    student_vectors,
    student_mask,
    teacher_means,
    teacher_eligible,
    projection,
    resources: SpectralKernelResources,
    *,
    labels,
    class_weights,
) -> SetLossResult:
    """Per-jet unordered finite-kernel loss for ordinary D teachers."""

    import torch

    tokens = torch.as_tensor(student_tokens).float()
    mask = torch.as_tensor(student_mask, device=tokens.device, dtype=torch.bool)
    target = torch.as_tensor(teacher_means, device=tokens.device).float()
    target_active = torch.as_tensor(teacher_eligible, device=tokens.device, dtype=torch.bool)
    if tokens.ndim != 3 or tokens.shape[-1] != 128 or mask.shape != tokens.shape[:2]:
        raise ValueError("student token surface/mask shapes differ")
    if target.shape != (len(tokens), resources.total_features) or target_active.shape != (len(tokens),):
        raise ValueError("ordinary set target shape differs")
    if target.requires_grad:
        raise ValueError("ordinary set targets must be detached")
    weights = token_weights(student_vectors, mask)
    per_jet = tokens.sum((1, 2)) * 0.0
    eligible = target_active & mask.any(-1)
    for row in range(len(tokens)):
        if bool(eligible[row]):
            projected = _unit(projection(tokens[row, mask[row]]).float())
            per_jet[row] = cached_finite_mmd(
                projected, weights[row, mask[row]], target[row], resources,
            )
    reduction = class_weighted_eligible_mean(
        per_jet, labels, class_weights, eligible,
    )
    return SetLossResult(
        reduction=reduction,
        active_family_count=eligible.to(torch.int64),
        family_eligible=eligible[:, None],
        family_losses=per_jet[:, None],
    )


def build_ordinary_token_targets(
    teacher_tokens,
    teacher_vectors,
    teacher_mask,
    resources: SpectralKernelResources,
) -> TokenKernelTargets:
    """Build detached compact per-jet targets from one ordinary teacher tap."""

    import torch

    tokens = torch.as_tensor(teacher_tokens)
    visible = torch.as_tensor(teacher_mask, device=tokens.device, dtype=torch.bool)
    if tokens.requires_grad:
        raise ValueError("teacher token target construction requires a detached forward")
    if tokens.ndim != 3 or tokens.shape[-1] != 128 or visible.shape != tokens.shape[:2]:
        raise ValueError("ordinary teacher token surface/mask shapes differ")
    weights = token_weights(teacher_vectors, visible)
    means = tokens.float().new_zeros((len(tokens), resources.total_features))
    present = visible.any(-1)
    for row in range(len(tokens)):
        if bool(present[row]):
            normalized = _unit(tokens[row, visible[row]].float())
            features = finite_spectral_features(normalized, resources)
            means[row] = (weights[row, visible[row], None] * features).sum(0)
    return TokenKernelTargets(means.detach(), present)


def native_offline_set_representation_loss(
    student_tokens,
    student_vectors,
    student_mask,
    family_codes,
    teacher_means,
    teacher_present,
    projections: Sequence,
    resources: SpectralKernelResources,
    *,
    labels,
    class_weights,
) -> SetLossResult:
    """TOFF set loss with independent charged/neutral latent bases."""

    import torch

    tokens = torch.as_tensor(student_tokens).float()
    mask = torch.as_tensor(student_mask, device=tokens.device, dtype=torch.bool)
    family = torch.as_tensor(family_codes, device=tokens.device)
    target = torch.as_tensor(teacher_means, device=tokens.device).float()
    present = torch.as_tensor(teacher_present, device=tokens.device, dtype=torch.bool)
    if tokens.ndim != 3 or tokens.shape[-1] != 128 or mask.shape != tokens.shape[:2] or family.shape != mask.shape:
        raise ValueError("TOFF student token surfaces differ")
    if target.shape != (len(tokens), 2, resources.total_features) or present.shape != (len(tokens), 2):
        raise ValueError("TOFF set targets differ")
    if target.requires_grad or len(projections) != 2:
        raise ValueError("TOFF targets/projections differ")
    family_weights = torch.stack(
        tuple(
            token_weights(student_vectors, mask & (family == family_index))
            for family_index in (CHARGED_FAMILY, NEUTRAL_FAMILY)
        ),
        dim=1,
    )
    family_losses = tokens.new_zeros((len(tokens), 2))
    family_eligible = torch.zeros((len(tokens), 2), dtype=torch.bool, device=tokens.device)
    for row in range(len(tokens)):
        for family_index in (CHARGED_FAMILY, NEUTRAL_FAMILY):
            selected = mask[row] & (family[row] == family_index)
            if bool(present[row, family_index]) and bool(selected.any()):
                projected = _unit(projections[family_index](tokens[row, selected]).float())
                family_losses[row, family_index] = cached_finite_mmd(
                    projected,
                    family_weights[row, family_index, selected],
                    target[row, family_index],
                    resources,
                )
                family_eligible[row, family_index] = True
    active_count = family_eligible.sum(-1)
    per_jet = (
        (family_losses * family_eligible).sum(-1)
        / active_count.clamp_min(1).to(torch.float32)
    )
    eligible = active_count > 0
    reduction = class_weighted_eligible_mean(per_jet, labels, class_weights, eligible)
    return SetLossResult(reduction, active_count, family_eligible, family_losses)


def build_native_offline_token_targets(
    charged_tokens,
    charged_vectors,
    charged_mask,
    neutral_tokens,
    neutral_vectors,
    neutral_mask,
    resources: SpectralKernelResources,
) -> TokenKernelTargets:
    """Build separate charged/neutral TOFF kernel means without concatenation."""

    import torch

    charged = build_ordinary_token_targets(
        charged_tokens, charged_vectors, charged_mask, resources,
    )
    neutral = build_ordinary_token_targets(
        neutral_tokens, neutral_vectors, neutral_mask, resources,
    )
    means = torch.stack((charged.means, neutral.means), dim=1)
    present = torch.stack((charged.present, neutral.present), dim=1)
    return TokenKernelTargets(means.detach(), present)


def _relation_stratum(delta_r: np.ndarray) -> np.ndarray:
    result = np.full(delta_r.shape, 2, dtype=np.int8)
    result[delta_r < RELATION_EDGES[1]] = 1
    result[delta_r < RELATION_EDGES[0]] = 0
    return result


def relation_population_eligibility(pair_weights) -> tuple[bool, float]:
    """Apply the exact four-pair/FP64 ESS gate to one candidate stratum."""

    value = np.asarray(
        pair_weights.detach().cpu().double().numpy()
        if hasattr(pair_weights, "detach") else pair_weights,
        dtype=np.float64,
    )
    if value.ndim != 1 or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("relation pair weights must be finite positive [pairs]")
    if value.size == 0:
        return False, 0.0
    normalized = value / value.sum(dtype=np.float64)
    ess = float(1.0 / np.square(normalized).sum(dtype=np.float64))
    if not np.isfinite(ess):
        raise FloatingPointError("relation effective sample size is nonfinite")
    return bool(value.size >= 4 and ess >= 3.0), ess


def build_student_relation_sketches(
    token_states,
    vectors,
    mask,
    visible_indices,
    resources: SpectralKernelResources,
    *,
    family_codes=None,
) -> RelationSketches:
    """Build differentiable top-32 latent-cosine sketches per jet/stratum."""

    import torch

    states = torch.as_tensor(token_states).float()
    p4 = torch.as_tensor(vectors, device=states.device).float()
    visible = torch.as_tensor(mask, device=states.device, dtype=torch.bool)
    identities = torch.as_tensor(visible_indices, device=states.device)
    if states.ndim != 3 or states.shape[-1] != 128 or visible.shape != states.shape[:2]:
        raise ValueError("relation student state/mask shapes differ")
    if p4.shape != (states.shape[0], 4, states.shape[1]) or identities.shape != visible.shape:
        raise ValueError("relation vectors/identity shapes differ")
    if family_codes is None:
        family = torch.zeros_like(identities, dtype=torch.int8)
        family_count = 1
    else:
        family = torch.as_tensor(family_codes, device=states.device)
        if family.shape != visible.shape:
            raise ValueError("relation family-code shape differs")
        family_count = 2
    if visible.any() and not torch.isfinite(states[visible]).all():
        raise FloatingPointError("visible student relation states are nonfinite")
    weights = torch.stack(
        tuple(
            token_weights(p4, visible & (family == family_index))
            for family_index in range(family_count)
        ),
        dim=1,
    )
    means = states.new_zeros((len(states), family_count, 3, resources.total_features))
    eligible = torch.zeros((len(states), family_count, 3), dtype=torch.bool, device=states.device)
    pair_counts = torch.zeros_like(eligible, dtype=torch.int64)
    ess = torch.zeros_like(means[..., 0], dtype=torch.float64)

    p4_cpu = p4.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float64, copy=False)
    ids_cpu = identities.detach().cpu().numpy()
    visible_cpu = visible.detach().cpu().numpy()
    family_cpu = family.detach().cpu().numpy()
    for row in range(len(states)):
        for family_index in range(family_count):
            selected = np.flatnonzero(
                visible_cpu[row] & (family_cpu[row] == family_index)
            )
            if selected.size < 2:
                continue
            pt, eta, phi, _ = p4_kinematics(p4_cpu[row, selected])
            if not np.isfinite(pt).all() or not np.isfinite(eta).all() or not np.isfinite(phi).all():
                raise FloatingPointError("student relation geometry is nonfinite")
            order = np.lexsort((ids_cpu[row, selected], -pt))[:32]
            selected = selected[order]
            _, eta, phi, _ = p4_kinematics(p4_cpu[row, selected])
            left, right = np.triu_indices(len(selected), k=1)
            delta_phi = np.arctan2(
                np.sin(phi[left] - phi[right]), np.cos(phi[left] - phi[right]),
            )
            delta_r = np.sqrt((eta[left] - eta[right]) ** 2 + delta_phi ** 2)
            if not np.isfinite(delta_r).all():
                raise FloatingPointError("student relation deltaR is nonfinite")
            strata = _relation_stratum(delta_r)
            selected_tensor = torch.as_tensor(selected, device=states.device, dtype=torch.long)
            normalized_states = _unit(states[row, selected_tensor])
            left_tensor = torch.as_tensor(left, device=states.device, dtype=torch.long)
            right_tensor = torch.as_tensor(right, device=states.device, dtype=torch.long)
            cosine = (normalized_states[left_tensor] * normalized_states[right_tensor]).sum(-1)
            pair_weight = (
                weights[row, family_index, selected_tensor[left_tensor]]
                * weights[row, family_index, selected_tensor[right_tensor]]
            )
            for stratum in range(3):
                chosen_np = np.flatnonzero(strata == stratum)
                count = len(chosen_np)
                pair_counts[row, family_index, stratum] = count
                if count == 0:
                    continue
                chosen = torch.as_tensor(chosen_np, device=states.device, dtype=torch.long)
                normalized_pair_weight = normalized_weights(pair_weight[chosen], expected_rows=count).to(states.device)
                active, current_ess = relation_population_eligibility(pair_weight[chosen])
                ess[row, family_index, stratum] = current_ess
                if active:
                    feature = finite_spectral_features(cosine[chosen], resources)
                    means[row, family_index, stratum] = (
                        normalized_pair_weight[:, None] * feature
                    ).sum(0)
                    eligible[row, family_index, stratum] = True
    return RelationSketches(means, eligible, pair_counts, ess)


def build_teacher_relation_targets(
    teacher_tokens,
    teacher_vectors,
    teacher_mask,
    visible_indices,
    resources: SpectralKernelResources,
    *,
    family_codes=None,
) -> RelationSketches:
    """Build compact detached teacher relations with the identical population rule."""

    import torch

    tokens = torch.as_tensor(teacher_tokens)
    if tokens.requires_grad:
        raise ValueError("teacher relation target construction requires a detached forward")
    result = build_student_relation_sketches(
        tokens, teacher_vectors, teacher_mask, visible_indices, resources,
        family_codes=family_codes,
    )
    return RelationSketches(
        result.means.detach(), result.eligible.detach(),
        result.pair_counts.detach(), result.effective_sample_sizes.detach(),
    )


def relation_representation_loss(
    student_tokens,
    student_vectors,
    student_mask,
    visible_indices,
    teacher_means,
    teacher_eligible,
    resources: SpectralKernelResources,
    *,
    labels,
    class_weights,
    family_codes=None,
) -> RelationLossResult:
    """Compare marginal latent-cosine sketches over fixed physical strata."""

    import torch

    sketches = build_student_relation_sketches(
        student_tokens, student_vectors, student_mask, visible_indices, resources,
        family_codes=family_codes,
    )
    target = torch.as_tensor(teacher_means, device=sketches.means.device).float()
    target_active = torch.as_tensor(
        teacher_eligible, device=sketches.means.device, dtype=torch.bool,
    )
    if target.shape != sketches.means.shape or target_active.shape != sketches.eligible.shape:
        raise ValueError("teacher relation sketch shape differs")
    if target.requires_grad:
        raise ValueError("teacher relation sketches must be detached")
    jointly_active = sketches.eligible & target_active
    losses = (sketches.means - target).square().sum(-1)
    active_strata = jointly_active.sum(-1)
    family_losses = (
        (losses * jointly_active).sum(-1)
        / active_strata.clamp_min(1).to(torch.float32)
    )
    active_families = active_strata > 0
    family_count = active_families.sum(-1)
    per_jet = (
        (family_losses * active_families).sum(-1)
        / family_count.clamp_min(1).to(torch.float32)
    )
    eligible = family_count > 0
    reduction = class_weighted_eligible_mean(per_jet, labels, class_weights, eligible)
    return RelationLossResult(
        reduction=reduction,
        active_family_count=family_count,
        active_stratum_count=active_strata,
        family_stratum_eligible=jointly_active,
        family_stratum_losses=losses,
        student_sketches=sketches,
    )


def projection_orthogonality(projections: Mapping[str, object]):
    """Mean explicit projection regularizer; finite ill-conditioning continues."""

    import torch

    if not projections:
        raise ValueError("at least one representation projection is required")
    values = []
    for name, projection in sorted(projections.items()):
        weight = projection.weight.float()
        if weight.shape != (128, 128) or not torch.isfinite(weight).all():
            raise FloatingPointError(f"projection {name} is nonfinite or has wrong shape")
        identity = torch.eye(128, device=weight.device, dtype=torch.float32)
        values.append((weight.transpose(0, 1) @ weight - identity).square().sum() / 128.0)
    result = torch.stack(values).mean()
    if not torch.isfinite(result):
        raise FloatingPointError("projection orthogonality is nonfinite")
    return result


def projection_diagnostics(projections: Mapping[str, object]) -> tuple[ProjectionDiagnostic, ...]:
    """Return validation diagnostics without treating finite conditioning as fatal."""

    import torch

    result = []
    for name, projection in sorted(projections.items()):
        weight = projection.weight.detach().float()
        if weight.shape != (128, 128) or not torch.isfinite(weight).all():
            raise FloatingPointError(f"projection {name} is nonfinite or has wrong shape")
        singular = torch.linalg.svdvals(weight)
        minimum = float(singular[-1].cpu())
        maximum = float(singular[0].cpu())
        condition = math.inf if minimum == 0 else maximum / minimum
        identity = torch.eye(128, device=weight.device)
        orth = float(((weight.T @ weight - identity).square().sum() / 128.0).cpu())
        result.append(ProjectionDiagnostic(
            name=name,
            singular_values=tuple(float(value) for value in singular.cpu()),
            condition_number=condition,
            poorly_conditioned=condition > 1.0e4,
            orthogonality_loss=orth,
        ))
    return tuple(result)


def jet_set_ramp(effective_pass: float) -> float:
    if not np.isfinite(effective_pass) or effective_pass < 0:
        raise ValueError("effective pass must be finite and nonnegative")
    if effective_pass <= 2.0:
        return 0.0
    if effective_pass >= 6.0:
        return 1.0
    return (effective_pass - 2.0) / 4.0


def relation_ramp(effective_pass: float) -> float:
    if not np.isfinite(effective_pass) or effective_pass < 0:
        raise ValueError("effective pass must be finite and nonnegative")
    if effective_pass <= 4.0:
        return 0.0
    if effective_pass >= 8.0:
        return 1.0
    return (effective_pass - 4.0) / 4.0


def effective_pass_for_update(update: int, updates_per_pass: int) -> float:
    if update < 0 or updates_per_pass <= 0:
        raise ValueError("update/pass size differs")
    return (update + 1) / updates_per_pass


def scheduled_representation_loss(
    *,
    strategy: Literal["RSET", "RREL"],
    effective_pass: float,
    scaled_jet,
    scaled_set,
    orthogonality,
    scaled_relation=None,
    rho: float = RHO_REPRESENTATION,
) -> ScheduledRepresentationLoss:
    """Apply the exact equal-exposure RSET/RREL pass schedule."""

    import torch

    if rho != RHO_REPRESENTATION:
        raise ValueError("HCWDL representation coefficient is frozen at 0.10")
    js = jet_set_ramp(effective_pass)
    rel = relation_ramp(effective_pass)
    jet = torch.as_tensor(scaled_jet).float()
    set_loss = torch.as_tensor(scaled_set, device=jet.device).float()
    orth = torch.as_tensor(orthogonality, device=jet.device).float()
    if strategy == "RSET":
        if scaled_relation is not None:
            raise ValueError("RSET cannot receive a relation component")
        jet_coefficient = 0.40 * js
        set_coefficient = 0.60 * js
        relation_coefficient = 0.0
    elif strategy == "RREL":
        if scaled_relation is None:
            raise ValueError("RREL requires a relation component")
        relation_value = torch.as_tensor(scaled_relation, device=jet.device).float()
        common = js - 0.25 * rel
        jet_coefficient = 0.40 * common
        set_coefficient = 0.60 * common
        relation_coefficient = 0.25 * rel
    else:
        raise ValueError("unknown representation strategy")
    scientific = jet_coefficient * jet + set_coefficient * set_loss
    if strategy == "RREL":
        scientific = scientific + relation_coefficient * relation_value
    orthogonal = js * ORTHOGONALITY_COEFFICIENT * orth
    scheduled = scientific + orthogonal
    total = rho * scheduled
    if not all(torch.isfinite(value) for value in (scientific, orthogonal, total)):
        raise FloatingPointError("scheduled representation loss is nonfinite")
    return ScheduledRepresentationLoss(
        total=total,
        scientific=scientific,
        orthogonality=orthogonal,
        jet_coefficient=jet_coefficient,
        set_coefficient=set_coefficient,
        relation_coefficient=relation_coefficient,
        ramp_jet_set=js,
        ramp_relation=rel,
    )


__all__ = [
    "CHARGED_FAMILY", "CHARGE_ONLY_CLASSIFICATION",
    "CONTRADICTION_CLASSIFICATION", "CONTRADICTION_FAMILY",
    "DIRECT_CLASSIFICATION", "FAMILY_CONTRACT", "FamilyClassification",
    "JetLossResult", "MALFORMED_CLASSIFICATION", "MALFORMED_FAMILY",
    "NEUTRAL_FAMILY", "ORTHOGONALITY_COEFFICIENT", "PADDED_FAMILY",
    "ProjectionDiagnostic", "RELATION_EDGES", "RELATION_STRATA",
    "RHO_REPRESENTATION", "RREL_CONTRACT", "RSET_CONTRACT",
    "ReducedRows", "RelationLossResult", "RelationSketches",
    "ScheduledRepresentationLoss", "SetLossResult", "TokenKernelTargets",
    "build_native_offline_token_targets", "build_ordinary_token_targets",
    "build_student_relation_sketches", "build_teacher_relation_targets",
    "class_weighted_eligible_mean",
    "classify_hlt_token_families", "effective_pass_for_update",
    "jet_representation_loss", "jet_set_ramp",
    "native_offline_set_representation_loss",
    "ordinary_set_representation_loss", "projection_diagnostics",
    "projection_orthogonality", "relation_ramp",
    "relation_population_eligibility", "relation_representation_loss",
    "scheduled_representation_loss",
    "token_weights",
]
