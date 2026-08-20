"""Train-only HCWDL representation gradient calibration.

Calibration observes one stochastic student forward per canonical batch,
derives the base and every component loss from those same tensors, and restores
all runtime state before returning immutable scalar results.  It performs no
optimizer or scheduler step and never accumulates ``parameter.grad``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
import random
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

from .hcwdl_representation_contracts import (
    CALIBRATION_SELECTION_CONTRACT,
    GRADIENT_CALIBRATION_CONTRACT,
)
from .hcwdl_representation_losses import class_weighted_eligible_mean


CALIBRATION_ROWS: Final = 4096
CALIBRATION_BATCHES: Final = 16
CALIBRATION_BATCH_SIZE: Final = 256
MINIMUM_VALID_BATCHES: Final = 12
MINIMUM_SCALE: Final = 1.0e-4
MAXIMUM_SCALE: Final = 1.0e4

EARLY_BACKBONE_PREFIXES: Final = (
    "deployable_model.mod.embed.",
    "deployable_model.mod.pair_embed.",
    "deployable_model.mod.blocks.0.",
    "deployable_model.mod.blocks.1.",
)


@dataclass(frozen=True)
class CalibrationIdentity:
    identity_sha256: str
    selection_sha256: str
    selection_payload: Mapping[str, object]


@dataclass(frozen=True)
class CalibrationSelection:
    rows: tuple[CalibrationIdentity, ...]
    ordered_identity_sha256: str
    requested_rows: int
    actual_rows: int


@dataclass(frozen=True)
class CalibrationComponentRows:
    per_jet: object
    eligible: object
    support: Mapping[str, int] | None = None
    # Some registered components have an exact reduction that cannot be
    # reconstructed from independent rows.  In particular the jet anchor owns
    # a recipe-vector off-diagonal Gram term (uniform under v4 exact ones).
    # Supplying ``loss`` keeps
    # that exact reduction while ``per_jet``/``eligible`` still define the
    # matched support used for the base-loss gradient.
    loss: object | None = None


@dataclass(frozen=True)
class CalibrationForwardResult:
    base_rows: object
    labels: object
    class_weights: object
    components: Mapping[str, CalibrationComponentRows]


@dataclass(frozen=True)
class GradientCalibrationComponent:
    name: str
    status: str
    inactive_reason: str | None
    scale: float
    scale_hex: str
    base_gradient_rms: tuple[float, ...]
    representation_gradient_rms: tuple[float, ...]
    median_base_gradient_rms: float | None
    median_representation_gradient_rms: float | None
    valid_batches: int
    support: tuple[Mapping[str, int], ...]


@dataclass(frozen=True)
class GradientCalibrationResult:
    contract: str
    components: Mapping[str, GradientCalibrationComponent]
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    parameter_scalar_count: int
    forward_calls: int


def select_calibration_identities(
    *,
    campaign_sha256: str,
    parent_logit_counterpart_node_id: str,
    identity_sha256s: Sequence[str],
    limit: int = CALIBRATION_ROWS,
) -> CalibrationSelection:
    """Select the exact smallest-hash train-only calibration population."""

    campaign = require_sha256(campaign_sha256, name="calibration campaign SHA-256")
    if not parent_logit_counterpart_node_id:
        raise ValueError("calibration parent counterpart node ID is empty")
    if limit <= 0 or limit > CALIBRATION_ROWS:
        raise ValueError("calibration selection limit differs")
    seen: set[str] = set()
    candidates: list[CalibrationIdentity] = []
    for raw_identity in identity_sha256s:
        identity = require_sha256(raw_identity, name="calibration identity SHA-256")
        if identity in seen:
            raise ValueError("repeated canonical calibration identity")
        seen.add(identity)
        payload = {
            "campaign_sha256": campaign,
            "contract": CALIBRATION_SELECTION_CONTRACT,
            "identity_sha256": identity,
            "parent_logit_counterpart_node_id": parent_logit_counterpart_node_id,
        }
        candidates.append(CalibrationIdentity(
            identity_sha256=identity,
            selection_sha256=canonical_sha256(payload),
            selection_payload=payload,
        ))
    candidates.sort(key=lambda row: (
        bytes.fromhex(row.selection_sha256), bytes.fromhex(row.identity_sha256),
    ))
    selected = tuple(candidates[:limit])
    ordered_hash = canonical_sha256({
        "contract": "HCWDL_REP_GRAD_CAL_ORDER/v1",
        "ordered_identity_sha256s": [row.identity_sha256 for row in selected],
        "ordered_selection_sha256s": [row.selection_sha256 for row in selected],
    })
    return CalibrationSelection(
        rows=selected,
        ordered_identity_sha256=ordered_hash,
        requested_rows=limit,
        actual_rows=len(selected),
    )


def calibration_seed_payload_bytes(row: CalibrationIdentity) -> bytes:
    """Expose canonical bytes for cross-language seed/hash fixtures."""

    if canonical_sha256(row.selection_payload) != row.selection_sha256:
        raise ValueError("calibration identity selection payload was mutated")
    return canonical_json_bytes(row.selection_payload)


def build_calibration_selection_artifact(
    *,
    campaign_sha256: str,
    parent_logit_counterpart_node_id: str,
    identity_sha256s: Sequence[str],
    limit: int = CALIBRATION_ROWS,
) -> dict[str, Any]:
    """Build the immutable ordered train-only calibration population.

    The artifact intentionally records both the canonical jet identities and
    their campaign/counterpart-bound selection digests.  The latter, rather
    than raw identity order or loader order, is the scientific ordering key.
    """

    selection = select_calibration_identities(
        campaign_sha256=campaign_sha256,
        parent_logit_counterpart_node_id=parent_logit_counterpart_node_id,
        identity_sha256s=identity_sha256s,
        limit=limit,
    )
    identities = [row.identity_sha256 for row in selection.rows]
    selection_digests = [row.selection_sha256 for row in selection.rows]
    return with_content_hash({
        "contract": CALIBRATION_SELECTION_CONTRACT,
        "schema_version": 1,
        "campaign_sha256": require_sha256(
            campaign_sha256, name="calibration campaign SHA-256",
        ),
        "parent_logit_counterpart_node_id": parent_logit_counterpart_node_id,
        "requested_rows": int(selection.requested_rows),
        "actual_rows": int(selection.actual_rows),
        "ordered_identity_sha256s": identities,
        "ordered_selection_sha256s": selection_digests,
        # ``CalibrationSelection`` predates the artifact schema and calls this
        # combined order digest ``ordered_identity_sha256``.  Give the durable
        # field its precise meaning while retaining exact numerical semantics.
        "ordered_selection_sha256": selection.ordered_identity_sha256,
        "canonical_identity_order_sha256": canonical_sha256(identities),
    })


def validate_calibration_selection_artifact(
    value: Mapping[str, Any],
    *,
    expected_campaign_sha256: str | None = None,
    expected_parent_logit_counterpart_node_id: str | None = None,
) -> str:
    """Re-derive every selection digest and the frozen total ordering."""

    digest = validate_content_hash(
        value,
        expected_contract=CALIBRATION_SELECTION_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "campaign_sha256", "parent_logit_counterpart_node_id",
        "requested_rows", "actual_rows", "ordered_identity_sha256s",
        "ordered_selection_sha256s", "ordered_selection_sha256",
        "canonical_identity_order_sha256",
    }
    if not required.issubset(value):
        raise ValueError("calibration-selection artifact fields differ")
    campaign = require_sha256(
        value["campaign_sha256"], name="calibration campaign SHA-256",
    )
    counterpart = str(value["parent_logit_counterpart_node_id"])
    if not counterpart:
        raise ValueError("calibration parent counterpart node ID is empty")
    if expected_campaign_sha256 is not None and campaign != require_sha256(
        expected_campaign_sha256, name="expected calibration campaign SHA-256",
    ):
        raise ValueError("calibration campaign lineage differs")
    if (
        expected_parent_logit_counterpart_node_id is not None
        and counterpart != expected_parent_logit_counterpart_node_id
    ):
        raise ValueError("calibration parent counterpart lineage differs")
    identities = list(value["ordered_identity_sha256s"])
    selection_digests = list(value["ordered_selection_sha256s"])
    actual = int(value["actual_rows"])
    requested = int(value["requested_rows"])
    if (
        requested <= 0
        or requested > CALIBRATION_ROWS
        or actual != len(identities)
        or actual != len(selection_digests)
        or actual <= 0
        or actual > requested
        or len(set(identities)) != actual
    ):
        raise ValueError("calibration-selection row counts differ")
    rebuilt = select_calibration_identities(
        campaign_sha256=campaign,
        parent_logit_counterpart_node_id=counterpart,
        identity_sha256s=identities,
        limit=actual,
    )
    if identities != [row.identity_sha256 for row in rebuilt.rows] or (
        selection_digests != [row.selection_sha256 for row in rebuilt.rows]
    ):
        raise ValueError("calibration-selection order/digests differ")
    if value["ordered_selection_sha256"] != rebuilt.ordered_identity_sha256:
        raise ValueError("calibration-selection combined order hash differs")
    if value["canonical_identity_order_sha256"] != canonical_sha256(identities):
        raise ValueError("calibration canonical identity-order hash differs")
    return digest


def early_backbone_parameters(
    model,
    *,
    expected_manifest: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[str, object], ...]:
    """Return the exact sorted early-backbone support used by every component."""

    selected = tuple(sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if any(name.startswith(prefix) for prefix in EARLY_BACKBONE_PREFIXES)
        ),
        key=lambda item: item[0],
    ))
    if not selected:
        raise ValueError("HCWDL calibration early-backbone support is empty")
    for prefix in EARLY_BACKBONE_PREFIXES:
        if not any(name.startswith(prefix) for name, _ in selected):
            raise ValueError(f"HCWDL calibration support is missing {prefix}")
    for name, parameter in selected:
        if not parameter.requires_grad:
            raise ValueError(f"HCWDL calibration parameter is frozen: {name}")
    if expected_manifest is not None:
        observed = {name: list(parameter.shape) for name, parameter in selected}
        expected = {str(name): list(shape) for name, shape in expected_manifest.items()}
        if observed != expected:
            raise ValueError("HCWDL calibration parameter manifest differs")
    return selected


def _gradient_rms(loss, parameters: Sequence, *, retain_graph: bool) -> float:
    import torch

    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    if len(gradients) != len(parameters):
        raise RuntimeError("HCWDL calibration gradient support differs")
    numerator = torch.zeros((), dtype=torch.float64, device=gradients[0].device)
    scalar_count = 0
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None or gradient.shape != parameter.shape:
            raise RuntimeError("HCWDL calibration gradient is disconnected")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("HCWDL calibration gradient is nonfinite")
        numerator = numerator + gradient.detach().double().square().sum()
        scalar_count += parameter.numel()
    result = torch.sqrt(numerator / scalar_count)
    if not torch.isfinite(result):
        raise FloatingPointError("HCWDL calibration gradient RMS is nonfinite")
    return float(result.cpu())


@dataclass
class _RuntimeSnapshot:
    model_state: Mapping[str, object]
    module_modes: Mapping[str, bool]
    model_buffers: Mapping[str, object]
    module_enabled: Mapping[str, bool]
    parameter_gradients: Mapping[str, object | None]
    optimizer_state: Mapping[str, object] | None
    python_random_state: object
    numpy_random_state: object
    torch_cpu_rng_state: object
    torch_cuda_rng_states: object | None
    external_state: object | None


def _snapshot_runtime(model, optimizer, external_snapshot):
    import torch

    return _RuntimeSnapshot(
        model_state=copy.deepcopy(model.state_dict()),
        module_modes={name: module.training for name, module in model.named_modules()},
        model_buffers={
            name: value.detach().clone() for name, value in model.named_buffers()
        },
        module_enabled={
            name: bool(module.enabled)
            for name, module in model.named_modules()
            if hasattr(module, "enabled")
        },
        parameter_gradients={
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        },
        optimizer_state=None if optimizer is None else copy.deepcopy(optimizer.state_dict()),
        python_random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_cpu_rng_state=torch.random.get_rng_state().clone(),
        torch_cuda_rng_states=(
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() else None
        ),
        external_state=None if external_snapshot is None else copy.deepcopy(external_snapshot()),
    )


def _restore_runtime(model, optimizer, snapshot, external_restore) -> None:
    import torch

    model.load_state_dict(snapshot.model_state, strict=True)
    for name, value in model.named_buffers():
        if name not in snapshot.model_buffers:
            raise RuntimeError("model buffer topology changed during calibration")
        value.copy_(snapshot.model_buffers[name].to(value.device))
    for name, module in model.named_modules():
        if name not in snapshot.module_modes:
            raise RuntimeError("model module topology changed during calibration")
        module.training = snapshot.module_modes[name]
        if name in snapshot.module_enabled:
            module.enabled = snapshot.module_enabled[name]
    for name, parameter in model.named_parameters():
        prior = snapshot.parameter_gradients[name]
        parameter.grad = None if prior is None else prior.to(parameter.device).clone()
    if optimizer is not None:
        if snapshot.optimizer_state is None:
            raise RuntimeError("optimizer snapshot is absent")
        optimizer.load_state_dict(snapshot.optimizer_state)
    random.setstate(snapshot.python_random_state)
    np.random.set_state(snapshot.numpy_random_state)
    torch.random.set_rng_state(snapshot.torch_cpu_rng_state)
    if snapshot.torch_cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_rng_states)
    if external_restore is not None:
        external_restore(copy.deepcopy(snapshot.external_state))


def calibrate_representation_components(
    *,
    model,
    batches: Iterable[Any],
    student_forward: Callable[[Any], Any],
    losses_from_forward: Callable[[Any, Any], CalibrationForwardResult],
    component_names: Sequence[str],
    optimizer=None,
    expected_batches: int | None = CALIBRATION_BATCHES,
    minimum_valid_batches: int = MINIMUM_VALID_BATCHES,
    expected_parameter_manifest: Mapping[str, Sequence[int]] | None = None,
    external_snapshot: Callable[[], object] | None = None,
    external_restore: Callable[[object], None] | None = None,
) -> GradientCalibrationResult:
    """Calibrate all named components from one shared forward per batch.

    ``student_forward`` is invoked exactly once per supplied batch.
    ``losses_from_forward`` receives that one result and must derive all base
    and component rows from it; it is deliberately a separate callback so the
    calibration orchestrator, rather than each component, owns the stochastic
    forward count.
    """

    import torch

    names = tuple(component_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("HCWDL calibration component names are empty/duplicated")
    if minimum_valid_batches <= 0:
        raise ValueError("HCWDL calibration minimum batch count differs")
    named_parameters = early_backbone_parameters(
        model, expected_manifest=expected_parameter_manifest,
    )
    parameters = tuple(parameter for _, parameter in named_parameters)
    observed_base: dict[str, list[float]] = {name: [] for name in names}
    observed_rep: dict[str, list[float]] = {name: [] for name in names}
    observed_support: dict[str, list[Mapping[str, int]]] = {name: [] for name in names}
    snapshot = _snapshot_runtime(model, optimizer, external_snapshot)
    forward_calls = 0
    try:
        for batch in batches:
            surfaces = student_forward(batch)
            forward_calls += 1
            result = losses_from_forward(batch, surfaces)
            if not isinstance(result, CalibrationForwardResult):
                raise TypeError("calibration callback returned the wrong result type")
            if set(result.components) != set(names):
                raise ValueError("calibration callback component registry differs")
            base_rows = torch.as_tensor(result.base_rows).float()
            labels = torch.as_tensor(result.labels, device=base_rows.device, dtype=torch.long)
            class_weights = torch.as_tensor(
                result.class_weights, device=base_rows.device, dtype=torch.float32,
            )
            if base_rows.ndim != 1 or labels.shape != base_rows.shape:
                raise ValueError("calibration base rows/labels differ")
            active_components = []
            for name in names:
                component = result.components[name]
                rows = torch.as_tensor(component.per_jet, device=base_rows.device).float()
                eligible = torch.as_tensor(
                    component.eligible, device=base_rows.device, dtype=torch.bool,
                )
                if rows.shape != base_rows.shape or eligible.shape != base_rows.shape:
                    raise ValueError("calibration component rows/support differ")
                if bool(eligible.any()):
                    active_components.append(
                        (name, rows, eligible, component.support or {}, component.loss)
                    )
            for component_index, (
                name, rows, eligible, support, explicit_loss,
            ) in enumerate(active_components):
                base_loss = class_weighted_eligible_mean(
                    base_rows, labels, class_weights, eligible,
                ).loss
                representation_loss = (
                    class_weighted_eligible_mean(
                        rows, labels, class_weights, eligible,
                    ).loss
                    if explicit_loss is None
                    else torch.as_tensor(
                        explicit_loss, device=base_rows.device,
                    ).float()
                )
                if representation_loss.ndim != 0 or not torch.isfinite(
                    representation_loss
                ):
                    raise FloatingPointError(
                        "HCWDL calibration component loss is nonfinite or nonscalar"
                    )
                # The graph is retained through every requested component and
                # released after the last representation-gradient query.
                base_rms = _gradient_rms(base_loss, parameters, retain_graph=True)
                rep_rms = _gradient_rms(
                    representation_loss,
                    parameters,
                    retain_graph=component_index < len(active_components) - 1,
                )
                observed_base[name].append(base_rms)
                observed_rep[name].append(rep_rms)
                observed_support[name].append(dict(support))
    finally:
        _restore_runtime(model, optimizer, snapshot, external_restore)

    if expected_batches is not None and forward_calls != expected_batches:
        raise ValueError(
            f"HCWDL calibration requires {expected_batches} batches, got {forward_calls}"
        )
    components: dict[str, GradientCalibrationComponent] = {}
    for name in names:
        base_values = np.asarray(observed_base[name], dtype=np.float64)
        rep_values = np.asarray(observed_rep[name], dtype=np.float64)
        if not np.isfinite(base_values).all() or not np.isfinite(rep_values).all():
            raise FloatingPointError("HCWDL calibration norm history is nonfinite")
        valid = len(base_values)
        median_base = float(np.median(base_values)) if valid else None
        median_rep = float(np.median(rep_values)) if valid else None
        status = "active"
        reason = None
        scale = 0.0
        if valid < minimum_valid_batches:
            status = "inactive_valid_support"; reason = "insufficient_valid_batches"
        elif median_base is None or median_rep is None or median_base <= 0 or median_rep <= 0:
            status = "inactive_valid_support"; reason = "nonpositive_gradient_median"
        else:
            implied = median_base / median_rep
            if not np.isfinite(implied):
                raise FloatingPointError("HCWDL calibration implied scale is nonfinite")
            if implied < MINIMUM_SCALE or implied > MAXIMUM_SCALE:
                status = "inactive_valid_support"; reason = "scale_outside_frozen_bounds"
            else:
                scale = float(implied)
        components[name] = GradientCalibrationComponent(
            name=name,
            status=status,
            inactive_reason=reason,
            scale=scale,
            scale_hex=float(scale).hex(),
            base_gradient_rms=tuple(float(value) for value in base_values),
            representation_gradient_rms=tuple(float(value) for value in rep_values),
            median_base_gradient_rms=median_base,
            median_representation_gradient_rms=median_rep,
            valid_batches=valid,
            support=tuple(observed_support[name]),
        )
    return GradientCalibrationResult(
        contract=GRADIENT_CALIBRATION_CONTRACT,
        components=components,
        parameter_names=tuple(name for name, _ in named_parameters),
        parameter_shapes=tuple(tuple(parameter.shape) for _, parameter in named_parameters),
        parameter_scalar_count=sum(parameter.numel() for parameter in parameters),
        forward_calls=forward_calls,
    )


def calibration_required_after_pass(*, strategy: str, completed_pass: int) -> tuple[str, ...]:
    """Return the exact component family activated at a validation barrier."""

    if completed_pass == 2:
        return ("jet", "set")
    if completed_pass == 4 and strategy == "RREL":
        return ("relation",)
    if strategy not in {"RSET", "RREL"}:
        raise ValueError("unknown HCWDL representation strategy")
    return ()


__all__ = [
    "CALIBRATION_BATCHES", "CALIBRATION_BATCH_SIZE", "CALIBRATION_ROWS",
    "CALIBRATION_SELECTION_CONTRACT", "CalibrationComponentRows",
    "CalibrationForwardResult", "CalibrationIdentity", "CalibrationSelection",
    "EARLY_BACKBONE_PREFIXES", "GRADIENT_CALIBRATION_CONTRACT",
    "GradientCalibrationComponent", "GradientCalibrationResult",
    "MAXIMUM_SCALE", "MINIMUM_SCALE", "MINIMUM_VALID_BATCHES",
    "calibrate_representation_components", "calibration_required_after_pass",
    "calibration_seed_payload_bytes", "build_calibration_selection_artifact",
    "validate_calibration_selection_artifact", "early_backbone_parameters",
    "select_calibration_identities",
]
