"""Streaming mixed-type PMARD repair paths with exact endpoint semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import numpy as np

from .inputs import ParticleInputs, build_hlt_inputs
from .matching import p4_kinematics, physical_p4_mask, wrapped_delta_phi
from .schema import HLT_FEATURE_SPECS, HLT_VECTOR_BRANCHES

ALPHA_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
REPAIR_FAMILY = "P4_ONLY/v1"
FULL_REPAIR_FAMILY = "FULL_PARTICLE_ENDPOINT/v1"
SELECTIVE_FULL_REPAIR_FAMILY = "SELECTIVE_FULL_PARTICLE_ENDPOINT/v1"
HIGHCOV_SHELL_EXACT_FAMILY = "HIGHCOV_SHELL_EXACT/v1"
HIGHCOV_SHELL_SOFT_FAMILY = "HIGHCOV_SHELL_SOFT/v1"
HIGHCOV_HC_EXACT_FAMILY = "HIGHCOV_HC_EXACT/v1"
HIGHCOV_HC_THRESHOLD = 0.958730161190033
REPAIR_FAMILIES = (
    "P4_ONLY", "FULL_PARTICLE_ENDPOINT", "SELECTIVE_FULL_PARTICLE_ENDPOINT",
    "HIGHCOV_SHELL_EXACT", "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
    "TRACK_ONLY", "P4_PLUS_TRACK", "DIRECTION_ONLY",
    "RESPONSE_ONLY", "WRONG_DIRECTION", "RANDOM_DIRECTION",
    "LOG_ANGULAR", "CONFIDENCE_WEIGHTED", "MATCH_SHUFFLED",
)


def runtime_repair_family(repair_family: str) -> str:
    """Translate an authorized versioned family into the tensor-builder selector."""

    aliases = {
        FULL_REPAIR_FAMILY: "FULL_PARTICLE_ENDPOINT",
        SELECTIVE_FULL_REPAIR_FAMILY: "SELECTIVE_FULL_PARTICLE_ENDPOINT",
        HIGHCOV_SHELL_EXACT_FAMILY: "HIGHCOV_SHELL_EXACT",
        HIGHCOV_SHELL_SOFT_FAMILY: "HIGHCOV_SHELL_SOFT",
        HIGHCOV_HC_EXACT_FAMILY: "HIGHCOV_HC_EXACT",
    }
    runtime = aliases.get(repair_family, repair_family)
    if runtime not in REPAIR_FAMILIES:
        raise ValueError("unknown repair family")
    return runtime


RECOMPUTED_CHANNELS = frozenset((7, 8, 9, 10, 19))
RETAINED_CHANNELS = frozenset(set(range(21)) - RECOMPUTED_CHANNELS)


@dataclass(frozen=True)
class FullEndpointField:
    """Projection and interpolation policy for one HLT-schema channel."""

    channel: int
    charged_suffix: str | None
    neutral_suffix: str | None
    interpolation: str
    discrete_group: str | None = None

    @property
    def hlt_branch(self) -> str:
        return HLT_FEATURE_SPECS[self.channel].branch


FULL_ENDPOINT_FIELDS = (
    FullEndpointField(0, "quality", None, "discrete", "quality"),
    FullEndpointField(1, "charge", None, "discrete", "identity"),
    FullEndpointField(2, "isEl", None, "discrete", "identity"),
    FullEndpointField(3, "isMu", None, "discrete", "identity"),
    FullEndpointField(4, "isChargedHad", None, "discrete", "identity"),
    FullEndpointField(5, None, "isGamma", "discrete", "identity"),
    FullEndpointField(6, None, "isNeutralHad", "discrete", "identity"),
    FullEndpointField(7, "phirel", "phirel", "angle"),
    FullEndpointField(8, "etarel", "etarel", "linear"),
    FullEndpointField(9, "abseta", "abseta", "linear"),
    FullEndpointField(10, "pt_log_nopuppi", "pt_log_nopuppi", "linear"),
    FullEndpointField(11, "normchi2", None, "linear"),
    FullEndpointField(12, "dz", None, "linear"),
    FullEndpointField(13, "dxy", None, "linear"),
    FullEndpointField(14, "dxysig", None, "linear"),
    FullEndpointField(15, "btagEtaRel", None, "linear"),
    FullEndpointField(16, "btagPtRatio", None, "linear"),
    FullEndpointField(17, "btagPParRatio", None, "linear"),
    FullEndpointField(18, "dzsig", None, "linear"),
    FullEndpointField(19, "e_log_nopuppi", "e_log_nopuppi", "linear"),
    FullEndpointField(20, "lostInnerHits", None, "discrete", "lost_inner_hits"),
)
FULL_TRACK_CHANNELS = frozenset(range(11, 19))
FULL_IDENTITY_CHANNELS = frozenset(range(1, 7))
FULL_VALIDITY_GROUPS = {
    "quality": (0,), "identity": tuple(range(1, 7)),
    "relative_kinematics": (7, 8, 9), "scale_kinematics": (10, 19),
    "track_fit": (11, 20), "track_dz": (12, 18),
    "track_dxy": (13, 14), "track_btag": (15, 16, 17),
}
FULL_VALIDITY_GROUP_BY_CHANNEL = {
    channel: group for group, channels in FULL_VALIDITY_GROUPS.items() for channel in channels
}

if tuple(field.channel for field in FULL_ENDPOINT_FIELDS) != tuple(range(len(HLT_FEATURE_SPECS))):
    raise RuntimeError("full endpoint field contract does not cover the HLT schema exactly")
if set(FULL_VALIDITY_GROUP_BY_CHANNEL) != set(range(len(HLT_FEATURE_SPECS))):
    raise RuntimeError("full endpoint validity groups do not cover the HLT schema exactly")


def _rows(value: object) -> list[np.ndarray]:
    try:
        import awkward as ak
        if isinstance(value, ak.Array):
            return [np.asarray(row) for row in ak.to_list(value)]
    except ImportError:
        pass
    return [np.asarray(row) for row in value]  # type: ignore[arg-type]


def combined_offline_p4(
    charged: Mapping[str, object], neutral: Mapping[str, object], row: int,
) -> np.ndarray:
    def collection(arrays: Mapping[str, object], prefix: str) -> np.ndarray:
        columns = [_rows(arrays[f"{prefix}_{name}"])[row] for name in ("px", "py", "pz", "energy")]
        if len({len(item) for item in columns}) != 1:
            raise ValueError(f"{prefix} p4 collection lengths differ")
        return np.stack(columns, axis=1).astype(np.float32, copy=False)
    cpf = collection(charged, "cpfcandlt")
    npf = collection(neutral, "npfcand")
    return np.concatenate((cpf, npf), axis=0)


def full_endpoint_required_branches() -> frozenset[str]:
    """Raw offline branches needed to project every endpoint into 21 channels."""

    branches = {
        f"{prefix}_{name}"
        for prefix in ("cpfcandlt", "npfcand")
        for name in ("px", "py", "pz", "energy")
    }
    for field in FULL_ENDPOINT_FIELDS:
        if field.charged_suffix is not None:
            branches.add(f"cpfcandlt_{field.charged_suffix}")
        if field.neutral_suffix is not None:
            branches.add(f"npfcand_{field.neutral_suffix}")
    return frozenset(branches)


def _unit_switch(
    identity_key: str, token_index: int, group: str, *, seed: int,
    repair_contract: str = FULL_REPAIR_FAMILY,
) -> float:
    payload = (
        f"{repair_contract}\0{int(seed)}\0{identity_key}\0"
        f"{int(token_index)}\0{group}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _full_endpoint_rows(
    offline_arrays: Mapping[str, object], rows: int,
) -> dict[str, list[np.ndarray]]:
    required = full_endpoint_required_branches()
    missing = sorted(required - set(offline_arrays))
    if missing:
        raise KeyError(f"full offline endpoint is missing branches: {missing}")
    result = {branch: _rows(offline_arrays[branch]) for branch in required}
    if any(len(values) != rows for values in result.values()):
        raise ValueError("full offline endpoint branch row counts differ")
    return result


def _combined_endpoint_features(
    offline_rows: Mapping[str, Sequence[np.ndarray]], *, row: int,
    charged_count: int, neutral_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((charged_count + neutral_count, len(FULL_ENDPOINT_FIELDS)), np.float64)
    validity = np.ones_like(features, np.bool_)
    for field in FULL_ENDPOINT_FIELDS:
        if field.charged_suffix is None:
            charged = np.zeros(charged_count, np.float64)
            charged_valid = np.ones(charged_count, np.bool_)
        else:
            charged = np.asarray(
                offline_rows[f"cpfcandlt_{field.charged_suffix}"][row], dtype=np.float64,
            )
            if len(charged) != charged_count:
                raise ValueError(f"charged endpoint length differs for channel {field.channel}")
            charged_valid = np.isfinite(charged) & (np.abs(charged) <= 1.0e32)
        if field.neutral_suffix is None:
            neutral = np.zeros(neutral_count, np.float64)
            neutral_valid = np.ones(neutral_count, np.bool_)
        else:
            neutral = np.asarray(
                offline_rows[f"npfcand_{field.neutral_suffix}"][row], dtype=np.float64,
            )
            if len(neutral) != neutral_count:
                raise ValueError(f"neutral endpoint length differs for channel {field.channel}")
            neutral_valid = np.isfinite(neutral) & (np.abs(neutral) <= 1.0e32)
        combined = np.concatenate((charged, neutral))
        combined_valid = np.concatenate((charged_valid, neutral_valid))
        features[:, field.channel] = np.where(combined_valid, combined, 0.0)
        validity[:, field.channel] = combined_valid
    return features, validity


def project_offline_endpoint_records(
    offline_arrays: Mapping[str, object], *, row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one complete native row into the frozen 21-field endpoint space.

    The returned arrays retain the repository's persistent native layout:
    all ``cpfcandlt`` entries followed by all ``npfcand`` entries.  Values are
    raw endpoint values (invalid registered measurements are represented by
    zero plus a false validity bit); no HLT normalization is applied here.
    This is the single public projection primitive used by Shell Exact and by
    the HCWDL structural-support homotopy.
    """

    if row < 0:
        raise ValueError("offline endpoint row must be nonnegative")
    rows = None
    for branch in full_endpoint_required_branches():
        if branch not in offline_arrays:
            raise KeyError(f"full offline endpoint is missing branch: {branch}")
        count = len(_rows(offline_arrays[branch]))
        rows = count if rows is None else rows
        if count != rows:
            raise ValueError("full offline endpoint branch row counts differ")
    if rows is None or row >= rows:
        raise IndexError("offline endpoint row is outside its input")
    projected = _full_endpoint_rows(offline_arrays, rows)
    charged_count = len(projected["cpfcandlt_px"][row])
    neutral_count = len(projected["npfcand_px"][row])
    features, validity = _combined_endpoint_features(
        projected, row=row, charged_count=charged_count,
        neutral_count=neutral_count,
    )
    p4 = combined_offline_p4(offline_arrays, offline_arrays, row).astype(
        np.float64, copy=False,
    )
    if len(p4) != len(features):
        raise ValueError("offline p4 and projected endpoint lengths differ")
    return features, validity, p4


def transform_endpoint_features(
    raw_features: np.ndarray, validity: np.ndarray,
) -> np.ndarray:
    """Apply the canonical HLT transform to projected endpoint records."""

    values = np.asarray(raw_features, dtype=np.float64)
    valid = np.asarray(validity, dtype=np.bool_)
    if values.ndim != 2 or values.shape[1] != len(HLT_FEATURE_SPECS):
        raise ValueError("endpoint features must be [particles,21]")
    if valid.shape != values.shape:
        raise ValueError("endpoint feature validity shape differs")
    transformed = np.empty_like(values, dtype=np.float32)
    for channel, spec in enumerate(HLT_FEATURE_SPECS):
        clean = np.where(valid[:, channel], values[:, channel], 0.0)
        transformed[:, channel] = np.clip(
            (clean - float(spec.median)) * float(spec.factor),
            float(spec.lower), float(spec.upper),
        ).astype(np.float32, copy=False)
    if not np.isfinite(transformed).all():
        raise FloatingPointError("canonical endpoint transform became nonfinite")
    return transformed


def _validate_full_endpoint_features(
    features: np.ndarray, validity: np.ndarray, *, row: int,
) -> np.ndarray:
    identity = features[:, 2:7]
    binary = (identity == 0) | (identity == 1)
    if not validity[:, 1:7].all() or not binary.all() or not np.all(identity.sum(axis=1) == 1):
        raise ValueError(f"invalid offline endpoint particle identity in row {row}")
    categories = np.argmax(identity, axis=1)
    charged = categories < 3
    charge = features[:, 1]
    if np.any(charge[charged] == 0) or np.any(charge[~charged] != 0):
        raise ValueError(f"offline endpoint charge/category incompatibility in row {row}")
    for channel in (0, 1, 20):
        active = validity[:, channel]
        if not np.equal(features[active, channel], np.rint(features[active, channel])).all():
            raise ValueError(f"offline endpoint channel {channel} is not discrete in row {row}")
    return charged


def _hlt_charged_mask(
    raw: Mapping[str, Sequence[np.ndarray]], *, row: int, visible: int,
    tokens: np.ndarray,
) -> np.ndarray:
    flags = np.stack([
        np.asarray(raw[HLT_FEATURE_SPECS[channel].branch][row][:visible], np.float64)
        for channel in range(2, 7)
    ], axis=1)[tokens]
    if not (((flags == 0) | (flags == 1)).all() and np.all(flags.sum(axis=1) == 1)):
        raise ValueError(f"invalid matched HLT particle identity in row {row}")
    return np.argmax(flags, axis=1) < 3


def _apply_full_endpoint_repair(
    raw: dict[str, list[np.ndarray]], canonical: ParticleInputs,
    offline_p4_by_row: Sequence[np.ndarray], assignments: np.ndarray, *,
    alpha: float, offline_arrays: Mapping[str, object],
    identity_keys: Sequence[str] | None, discrete_seed: int,
    require_complete: bool = True,
    strength_by_token: np.ndarray | None = None,
    repair_contract: str = FULL_REPAIR_FAMILY,
) -> None:
    rows = canonical.features.shape[0]
    strengths = None if strength_by_token is None else np.asarray(strength_by_token, np.float64)
    if strengths is not None:
        if strengths.shape != assignments.shape or not np.isfinite(strengths).all():
            raise ValueError("full endpoint strength shape or finiteness differs")
        if np.any((strengths < 0) | (strengths > 1)):
            raise ValueError("full endpoint strengths must lie in [0,1]")
    has_intermediate = 0 < alpha < 1 if strengths is None else bool(
        np.any((strengths > 0) & (strengths < 1))
    )
    if has_intermediate:
        if identity_keys is None or len(identity_keys) != rows:
            raise ValueError("intermediate full endpoint repair requires one identity key per row")
        if len(set(map(str, identity_keys))) != rows:
            raise ValueError("full endpoint repair identity keys must be unique")
    offline_rows = _full_endpoint_rows(offline_arrays, rows)
    for row in range(rows):
        visible = min(int(canonical.raw_lengths[row]), canonical.features.shape[2])
        row_assignment = np.asarray(assignments[row, :visible], dtype=np.int64)
        matched_tokens = np.flatnonzero(row_assignment >= 0)
        if require_complete and len(matched_tokens) != visible:
            missing = np.flatnonzero(row_assignment < 0)
            raise ValueError(
                f"full endpoint requires every visible HLT token to be matched; "
                f"row {row} is missing {len(missing)} assignments"
            )
        matched_assignment = row_assignment[matched_tokens]
        row_strength = (
            np.full(len(matched_tokens), alpha, np.float64)
            if strengths is None else strengths[row, matched_tokens]
        )
        if len(set(matched_assignment.tolist())) != len(matched_assignment):
            raise ValueError(f"full endpoint assignment is not one-to-one in row {row}")

        hlt_p4 = np.stack(
            [np.asarray(raw[name][row][:visible], np.float64) for name in HLT_VECTOR_BRANCHES],
            axis=1,
        )
        offline_p4 = np.asarray(offline_p4_by_row[row], dtype=np.float64)
        if np.any(matched_assignment >= len(offline_p4)):
            raise ValueError(f"full endpoint assignment is out of bounds in row {row}")
        if not physical_p4_mask(hlt_p4).all():
            raise ValueError(f"nonphysical full endpoint p4 in row {row}")
        endpoint_p4 = offline_p4[matched_assignment]
        if not physical_p4_mask(endpoint_p4).all():
            raise ValueError(f"nonphysical selected offline endpoint p4 in row {row}")

        charged_count = len(offline_rows["cpfcandlt_px"][row])
        neutral_count = len(offline_rows["npfcand_px"][row])
        if charged_count + neutral_count != len(offline_p4):
            raise ValueError(f"offline p4 and feature endpoint lengths differ in row {row}")
        all_endpoint_features, all_endpoint_validity = _combined_endpoint_features(
            offline_rows, row=row, charged_count=charged_count, neutral_count=neutral_count,
        )
        endpoint_features = all_endpoint_features[matched_assignment]
        endpoint_validity = all_endpoint_validity[matched_assignment]
        endpoint_charged = _validate_full_endpoint_features(
            endpoint_features, endpoint_validity, row=row,
        )
        hlt_charged = _hlt_charged_mask(
            raw, row=row, visible=visible, tokens=matched_tokens,
        )
        applicability_changes = hlt_charged != endpoint_charged

        hlt_features = np.stack([
            np.asarray(raw[field.hlt_branch][row][:visible], np.float64)
            for field in FULL_ENDPOINT_FIELDS
        ], axis=1)[matched_tokens]
        hlt_validity = np.isfinite(hlt_features) & (np.abs(hlt_features) <= 1.0e32)
        hlt_features = np.where(hlt_validity, hlt_features, 0.0)

        key = "" if identity_keys is None else str(identity_keys[row])
        choices: dict[str, np.ndarray] = {}
        for group in ("identity", "quality", "lost_inner_hits"):
            if np.all(row_strength == 1):
                choices[group] = np.ones(len(matched_tokens), np.bool_)
            else:
                choices[group] = np.asarray([
                    _unit_switch(
                        key, token, group, seed=discrete_seed,
                        repair_contract=repair_contract,
                    ) < strength
                    for token, strength in zip(matched_tokens, row_strength, strict=True)
                ], np.bool_)
        validity_changes = {
            group: np.any(
                hlt_validity[:, channels] != endpoint_validity[:, channels], axis=1,
            )
            for group, channels in FULL_VALIDITY_GROUPS.items()
        }
        validity_choices = {
            group: (
                np.ones(len(matched_tokens), np.bool_) if np.all(row_strength == 1) else np.asarray([
                    _unit_switch(
                        key, token, f"validity_{group}", seed=discrete_seed,
                        repair_contract=repair_contract,
                    ) < strength
                    for token, strength in zip(matched_tokens, row_strength, strict=True)
                ], np.bool_)
            )
            for group in FULL_VALIDITY_GROUPS
        }

        repaired_p4 = hlt_p4.copy()
        interpolated_p4 = (
            (1.0 - row_strength[:, None]) * hlt_p4[matched_tokens]
            + row_strength[:, None] * endpoint_p4
        )
        repaired_p4[matched_tokens] = np.where(
            (row_strength == 1)[:, None], endpoint_p4, interpolated_p4,
        )
        if not physical_p4_mask(repaired_p4).all():
            raise ValueError(f"interpolated full endpoint p4 became nonphysical in row {row}")
        for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
            raw[branch][row][:visible] = repaired_p4[:, channel].astype(
                raw[branch][row].dtype, copy=False,
            )

        for field in FULL_ENDPOINT_FIELDS:
            hlt_value = hlt_features[:, field.channel]
            endpoint_value = endpoint_features[:, field.channel]
            validity_group = FULL_VALIDITY_GROUP_BY_CHANNEL[field.channel]
            group_validity_changes = validity_changes[validity_group]
            validity_choice = validity_choices[validity_group]
            if field.interpolation == "linear":
                repaired_value = (1.0 - row_strength) * hlt_value + row_strength * endpoint_value
                repaired_value = np.where(row_strength == 1, endpoint_value, repaired_value)
                repaired_value = np.where(
                    group_validity_changes,
                    np.where(validity_choice, endpoint_value, hlt_value),
                    repaired_value,
                )
                if field.channel in FULL_TRACK_CHANNELS and np.any(applicability_changes):
                    use_endpoint = choices["identity"]
                    repaired_value = np.where(
                        applicability_changes,
                        np.where(use_endpoint, endpoint_value, hlt_value),
                        repaired_value,
                    )
            elif field.interpolation == "angle":
                displacement = wrapped_delta_phi(endpoint_value, hlt_value)
                repaired_value = wrapped_delta_phi(
                    hlt_value + row_strength * displacement, 0.0,
                )
                repaired_value = np.where(row_strength == 1, endpoint_value, repaired_value)
                repaired_value = np.where(
                    group_validity_changes,
                    np.where(validity_choice, endpoint_value, hlt_value),
                    repaired_value,
                )
            elif field.interpolation == "discrete":
                group = field.discrete_group
                if group is None:
                    raise RuntimeError("discrete endpoint field lacks a switch group")
                use_endpoint = choices[group]
                if field.channel in {0, 20}:
                    use_endpoint = np.where(applicability_changes, choices["identity"], use_endpoint)
                use_endpoint = np.where(group_validity_changes, validity_choice, use_endpoint)
                repaired_value = np.where(use_endpoint, endpoint_value, hlt_value)
            else:
                raise RuntimeError(f"unknown endpoint interpolation {field.interpolation!r}")
            if not np.isfinite(repaired_value).all():
                raise ValueError(f"nonfinite repaired channel {field.channel} in row {row}")
            destination = raw[field.hlt_branch][row][:visible]
            destination[matched_tokens] = repaired_value.astype(
                destination.dtype, copy=False,
            )


def build_alpha_repaired_inputs(
    hlt_arrays: Mapping[str, object], offline_p4_by_row: Sequence[np.ndarray],
    assignments: np.ndarray, *, alpha: float,
    repair_family: str = "P4_ONLY",
    confidence_weights: np.ndarray | None = None,
    offline_arrays: Mapping[str, object] | None = None,
    identity_keys: Sequence[str] | None = None,
    discrete_seed: int = 1337,
) -> ParticleInputs:
    repair_family = runtime_repair_family(repair_family)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("repair alpha must be a finite scalar in [0, 1]") from error
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("repair alpha must be a finite scalar in [0, 1]")
    # PMARD's registered repair arms retain their discrete screening grid.
    # Shell Exact is a continuous, confidence-warped coordinate: the dense
    # HCWDL contracts deliberately register intermediate D95/D90/... views.
    if repair_family != "HIGHCOV_SHELL_EXACT" and alpha not in ALPHA_GRID:
        raise ValueError(f"repair alpha must be one of the legacy grid {ALPHA_GRID}")
    if repair_family in {"TRACK_ONLY", "P4_PLUS_TRACK"}:
        raise PermissionError(
            "track repair is disabled until a locked branch-semantics compatibility audit exists"
        )
    canonical = build_hlt_inputs(hlt_arrays)
    if alpha == 0:
        return canonical
    mapping = np.asarray(assignments)
    rows = canonical.features.shape[0]
    if mapping.shape != (rows, canonical.features.shape[2]):
        raise ValueError("repair assignment shape differs")
    if len(offline_p4_by_row) != rows:
        raise ValueError("offline endpoint row count differs")
    confidence = None if confidence_weights is None else np.asarray(confidence_weights, np.float64)
    confidence_families = {
        "CONFIDENCE_WEIGHTED", "HIGHCOV_SHELL_EXACT",
        "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
    }
    if repair_family in confidence_families:
        if confidence is None or confidence.shape != mapping.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence-weighted repair requires finite aligned probabilities")
    raw = {name: [row.copy() for row in _rows(value)] for name, value in hlt_arrays.items()}
    highcov_families = {
        "HIGHCOV_SHELL_EXACT", "HIGHCOV_SHELL_SOFT", "HIGHCOV_HC_EXACT",
    }
    if repair_family in {"FULL_PARTICLE_ENDPOINT", "SELECTIVE_FULL_PARTICLE_ENDPOINT"} | highcov_families:
        if offline_arrays is None:
            raise ValueError("full endpoint repair requires native offline arrays")
        effective_mapping = mapping
        strength_by_token = None
        repair_contract = (
            FULL_REPAIR_FAMILY if repair_family == "FULL_PARTICLE_ENDPOINT"
            else SELECTIVE_FULL_REPAIR_FAMILY
        )
        if repair_family in highcov_families:
            assert confidence is not None
            if repair_family == "HIGHCOV_SHELL_EXACT":
                if alpha == 0:
                    strength_by_token = np.zeros_like(confidence)
                elif alpha == 1:
                    strength_by_token = np.where(mapping >= 0, 1.0, 0.0)
                else:
                    strength_by_token = np.where(
                        mapping >= 0, alpha ** (2.0 - 1.3 * confidence), 0.0,
                    )
                repair_contract = HIGHCOV_SHELL_EXACT_FAMILY
            elif repair_family == "HIGHCOV_SHELL_SOFT":
                strength_by_token = np.where(mapping >= 0, alpha * confidence, 0.0)
                repair_contract = HIGHCOV_SHELL_SOFT_FAMILY
            else:
                effective_mapping = mapping.copy()
                effective_mapping[confidence < HIGHCOV_HC_THRESHOLD] = -1
                if alpha == 0:
                    strength_by_token = np.zeros_like(confidence)
                elif alpha == 1:
                    strength_by_token = np.where(effective_mapping >= 0, 1.0, 0.0)
                else:
                    strength_by_token = np.where(
                        effective_mapping >= 0,
                        alpha ** (2.0 - 1.3 * confidence), 0.0,
                    )
                repair_contract = HIGHCOV_HC_EXACT_FAMILY
        _apply_full_endpoint_repair(
            raw, canonical, offline_p4_by_row, effective_mapping, alpha=alpha,
            offline_arrays=offline_arrays, identity_keys=identity_keys,
            discrete_seed=discrete_seed,
            require_complete=repair_family == "FULL_PARTICLE_ENDPOINT",
            strength_by_token=strength_by_token,
            repair_contract=repair_contract,
        )
        result = build_hlt_inputs(raw)
        if not np.array_equal(result.mask, canonical.mask) or not np.array_equal(
            result.raw_lengths, canonical.raw_lengths,
        ):
            raise RuntimeError("full endpoint repair changed HLT token identity")
        return result
    for row in range(rows):
        visible = min(int(canonical.raw_lengths[row]), canonical.features.shape[2])
        hlt_p4 = np.stack([raw[name][row][:visible] for name in HLT_VECTOR_BRANCHES], axis=1).astype(np.float64)
        offline = np.asarray(offline_p4_by_row[row], dtype=np.float64)
        if not physical_p4_mask(hlt_p4).all():
            raise ValueError(f"nonphysical HLT p4 endpoint in row {row}")
        offline_valid = physical_p4_mask(offline)
        repaired = hlt_p4.copy()
        accepted = mapping[row, :visible] >= 0
        for i in np.flatnonzero(accepted):
            j = int(mapping[row, i])
            if j >= len(offline) or not offline_valid[j]:
                raise ValueError(f"invalid offline repair endpoint in row {row}, token {i}")
            endpoint = offline[j].copy()
            if repair_family in {"DIRECTION_ONLY", "WRONG_DIRECTION", "RANDOM_DIRECTION", "RESPONSE_ONLY"}:
                h_pt, h_eta, h_phi, h_e = (item[0] for item in p4_kinematics(hlt_p4[i:i + 1]))
                o_pt, o_eta, o_phi, o_e = (item[0] for item in p4_kinematics(offline[j:j + 1]))
                if repair_family == "DIRECTION_ONLY":
                    magnitude = np.linalg.norm(hlt_p4[i, :3]); eta, phi, energy = o_eta, o_phi, h_e
                elif repair_family == "RESPONSE_ONLY":
                    magnitude = np.linalg.norm(offline[j, :3]); eta, phi, energy = h_eta, h_phi, o_e
                else:
                    magnitude = np.linalg.norm(offline[j, :3])
                    if repair_family == "WRONG_DIRECTION":
                        eta = h_eta - (o_eta - h_eta)
                        phi = h_phi - float(wrapped_delta_phi(o_phi, h_phi))
                    else:
                        dr = np.hypot(o_eta - h_eta, float(wrapped_delta_phi(o_phi, h_phi)))
                        theta = 2 * np.pi * (((row + 1) * 2654435761 + (i + 1) * 2246822519) % 2**32) / 2**32
                        eta = h_eta + dr * np.cos(theta); phi = h_phi + dr * np.sin(theta)
                    energy = o_e
                pt = magnitude / np.cosh(eta)
                endpoint = np.array(
                    [pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy],
                    dtype=np.float64,
                )
            effective_alpha = alpha * (confidence[row, i] if repair_family == "CONFIDENCE_WEIGHTED" else 1.0)
            if repair_family == "LOG_ANGULAR":
                h_pt, h_eta, h_phi, h_e = (item[0] for item in p4_kinematics(hlt_p4[i:i + 1]))
                o_pt, o_eta, o_phi, o_e = (item[0] for item in p4_kinematics(offline[j:j + 1]))
                pt = np.exp((1 - effective_alpha) * np.log(h_pt) + effective_alpha * np.log(o_pt))
                energy = np.exp((1 - effective_alpha) * np.log(h_e) + effective_alpha * np.log(o_e))
                eta = (1 - effective_alpha) * h_eta + effective_alpha * o_eta
                phi = h_phi + effective_alpha * float(wrapped_delta_phi(o_phi, h_phi))
                repaired[i] = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy])
            else:
                repaired[i] = (1.0 - effective_alpha) * hlt_p4[i] + effective_alpha * endpoint
        if not physical_p4_mask(repaired).all():
            raise ValueError(f"interpolated p4 became nonphysical in row {row}")
        hpt, heta, hphi, henergy = p4_kinematics(hlt_p4)
        rpt, reta, rphi, renergy = p4_kinematics(repaired)
        deltas = {
            "scoutpfcand_phirel": wrapped_delta_phi(rphi, hphi),
            "scoutpfcand_etarel": reta - heta,
            "scoutpfcand_abseta": np.abs(reta) - np.abs(heta),
            "scoutpfcand_pt_log": np.log(rpt / hpt),
            "scoutpfcand_e_log": np.log(renergy / henergy),
        }
        if any(not np.isfinite(value).all() for value in deltas.values()):
            raise ValueError(f"nonfinite repair delta in row {row}")
        for branch, delta in deltas.items():
            raw[branch][row][:visible] = raw[branch][row][:visible] + np.where(accepted, delta, 0)
        for channel, branch in enumerate(HLT_VECTOR_BRANCHES):
            raw[branch][row][:visible] = repaired[:, channel].astype(raw[branch][row].dtype, copy=False)
    result = build_hlt_inputs(raw)
    if not np.array_equal(result.mask, canonical.mask) or not np.array_equal(result.raw_lengths, canonical.raw_lengths):
        raise RuntimeError("alpha repair changed HLT token identity")
    for channel in RETAINED_CHANNELS:
        if not np.array_equal(result.features[:, channel], canonical.features[:, channel]):
            raise RuntimeError(f"repair changed retained channel {channel}")
    return result


def build_full_offline_endpoint_inputs(
    hlt_arrays: Mapping[str, object], offline_arrays: Mapping[str, object],
    offline_p4_by_row: Sequence[np.ndarray], assignments: np.ndarray,
) -> ParticleInputs:
    """Build the exact HLT-cardinality, HLT-ordered all-offline endpoint."""

    return build_alpha_repaired_inputs(
        hlt_arrays, offline_p4_by_row, assignments, alpha=1.0,
        repair_family="FULL_PARTICLE_ENDPOINT", offline_arrays=offline_arrays,
    )


def build_selective_matched_offline_endpoint_inputs(
    hlt_arrays: Mapping[str, object], offline_arrays: Mapping[str, object],
    offline_p4_by_row: Sequence[np.ndarray], assignments: np.ndarray,
) -> ParticleInputs:
    """Replace accepted tokens completely; retain unmatched tokens exactly as HLT."""

    return build_alpha_repaired_inputs(
        hlt_arrays, offline_p4_by_row, assignments, alpha=1.0,
        repair_family="SELECTIVE_FULL_PARTICLE_ENDPOINT", offline_arrays=offline_arrays,
    )


__all__ = [
    "ALPHA_GRID", "FULL_ENDPOINT_FIELDS", "FULL_IDENTITY_CHANNELS",
    "FULL_REPAIR_FAMILY", "SELECTIVE_FULL_REPAIR_FAMILY", "FULL_TRACK_CHANNELS", "FULL_VALIDITY_GROUPS",
    "HIGHCOV_HC_EXACT_FAMILY", "HIGHCOV_HC_THRESHOLD", "HIGHCOV_SHELL_EXACT_FAMILY",
    "HIGHCOV_SHELL_SOFT_FAMILY",
    "RECOMPUTED_CHANNELS",
    "REPAIR_FAMILIES", "REPAIR_FAMILY", "RETAINED_CHANNELS",
    "build_alpha_repaired_inputs", "build_full_offline_endpoint_inputs",
    "build_selective_matched_offline_endpoint_inputs",
    "combined_offline_p4",
    "full_endpoint_required_branches",
    "project_offline_endpoint_records", "transform_endpoint_features",
    "runtime_repair_family",
]
