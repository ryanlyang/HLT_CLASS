"""Authenticated durable representation-target banks for HCWDL-RKD.

The module owns target *artifacts*, not teacher inference.  Callers provide
already-computed arrays from the one canonical teacher surface forward; this
code validates, stages, commits, reloads, and identity-joins those arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import zipfile
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    identity_order_sha256_iter,
    load_json,
    load_npz_arrays,
    require_sha256,
    sha256_bytes,
    sha256_file,
    validate_content_hash,
)

from .hcwdl_representation_artifacts import (
    FailureHook,
    commit_staged_directory,
    fsync_directory,
    write_staged_immutable_bytes,
    write_staged_immutable_json,
)
from .hcwdl_representation_contracts import (
    TARGET_BUILD_INTENT_CONTRACT,
    TARGET_CONSUMER_REGISTRY_CONTRACT,
    TARGET_EXECUTION_ATTESTATION_CONTRACT,
    TARGET_FORWARD_SPEC_CONTRACT,
    TARGET_GENERATION_CONTRACT,
    TARGET_LOGICAL_BANK_CONTRACT,
    TARGET_MANIFEST_CONTRACT,
    TARGET_SHARD_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    STORAGE_ESTIMATE_CONTRACT,
    build_versioned_artifact,
    logical_array_sha256,
    logical_array_sha256_from_byte_hash,
    validate_parent_hashes,
    validate_versioned_artifact,
)
from .hcwdl_representation_graph import CONTROL_REGISTRY, NODE_REGISTRY
from .hcwdl_representation_reporting import derive_representation_execution_id
from .hcwdl_representation_resources import validate_measured_profile


ORDINARY_BANK: Final = "ordinary"
TOFF_BANK: Final = "toff"
BANK_KINDS: Final = (ORDINARY_BANK, TOFF_BANK)
LOGICAL_BANK_IDS: Final = (
    "D0c", "D0w", "D25c", "D25w", "D50c", "D50w", "D75c", "D75w",
    "D100", "TOFF",
)
ORDINARY_FAMILIES: Final = ("all",)
TOFF_FAMILIES: Final = ("charged", "neutral")
ORDINARY_REASONS: Final = ("ordinary_all",)
TOFF_REASONS: Final = (
    "known_pid_charge_agree_charged",
    "known_pid_charge_agree_neutral",
    "charge_only_charged",
    "charge_only_neutral",
    "unclassified_contradiction",
    "unclassified_malformed",
)
TOKEN_KERNEL_DIM: Final = 1024
RELATION_KERNEL_DIM: Final = 256
RELATION_STRATA: Final = ("local", "medium", "wide")
TARGET_FORWARD_BATCH_SIZE: Final = 256
NONFINAL_ACCEPTANCE_TARGET_PURPOSE: Final = "nonfinal_acceptance"
NONFINAL_ACCEPTANCE_TARGET_CONSUMER_CONTRACT: Final = (
    "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_TARGET_CONSUMER/v1"
)
NONFINAL_ACCEPTANCE_TARGET_ROWS: Final = 512
NONFINAL_ACCEPTANCE_TARGET_TRAJECTORIES: Final = {
    "D0c": {
        "rset_m1c_two_update": (("rset_m1c_two_update",), "RSET_M1c"),
        "rrel_m1c_two_update": (("rrel_m1c_two_update",), "RREL_M1c"),
        "rrel_m1c_usr1_exact_resume": (
            ("usr1_reference", "usr1_interrupt", "usr1_resume"),
            "RREL_M1c",
        ),
    },
    "D0w": {
        "rset_m1w_two_update": (("rset_m1w_two_update",), "RSET_M1w"),
        "rrel_m1w_two_update": (("rrel_m1w_two_update",), "RREL_M1w"),
    },
}
KERNEL_RESOURCE_NAMES: Final = (
    "token_rbf_sigma_0p10", "token_rbf_sigma_0p25", "token_rbf_sigma_0p50",
    "token_rbf_sigma_1p00", "relation_rbf_sigma_0p05", "relation_rbf_sigma_0p10",
    "relation_rbf_sigma_0p20", "relation_rbf_sigma_0p40",
)
_PARTITION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TEACHER_INPUT = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_FORBIDDEN_TEACHER_INPUT_FRAGMENTS: Final = (
    "label", "class_index", "target", "match", "assignment", "confidence",
    "degradation", "identity_digest", "source_entry",
)
_POPULATION_ROW_DTYPE: Final = np.dtype([
    ("source_file_id", "<u4"),
    ("source_entry", "<u8"),
    ("identity_digest", "u1", (32,)),
    ("label", "u1"),
], align=False)


def deterministic_compressed_npz_bytes(
    arrays: Mapping[str, np.ndarray],
) -> bytes:
    """Serialize the plan-required deterministically keyed compressed NPZ.

    The repository-wide helper is intentionally ZIP_STORED because older
    artifact byte identities depend on that behavior.  HCWDL-RKD target
    shards are a new contract and explicitly require compression, so their
    serializer lives here instead of silently changing legacy bytes.
    """

    output = BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=6, allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"unsafe target array name {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"object target array {name!r} is forbidden")
            npy = BytesIO()
            np.lib.format.write_array(npy, array, allow_pickle=False)
            member = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o644 << 16
            archive.writestr(member, npy.getvalue(), compresslevel=6)
    return output.getvalue()

LOGICAL_REQUIRED_PARENTS: Final = frozenset({
    "source", "split", "train_row_selection", "graph", "assignment", "repair",
    "architecture", "parent_recipe", "representation_recipe", "kernel_resources",
    "parent_import", "parent_loss_attestation",
})


def bank_kind_for_id(bank_id: str) -> str:
    if bank_id not in LOGICAL_BANK_IDS:
        raise ValueError(f"unknown HCWDL-RKD logical bank {bank_id!r}")
    return TOFF_BANK if bank_id == "TOFF" else ORDINARY_BANK


def family_vocabulary(bank_kind: str) -> tuple[str, ...]:
    if bank_kind == ORDINARY_BANK:
        return ORDINARY_FAMILIES
    if bank_kind == TOFF_BANK:
        return TOFF_FAMILIES
    raise ValueError("unknown HCWDL-RKD bank kind")


def reason_vocabulary(bank_kind: str) -> tuple[str, ...]:
    if bank_kind == ORDINARY_BANK:
        return ORDINARY_REASONS
    if bank_kind == TOFF_BANK:
        return TOFF_REASONS
    raise ValueError("unknown HCWDL-RKD bank kind")


def target_core_values_per_row(bank_kind: str) -> int:
    if bank_kind == ORDINARY_BANK:
        return 15 + 128 + TOKEN_KERNEL_DIM + 3 * RELATION_KERNEL_DIM
    if bank_kind == TOFF_BANK:
        return 15 + 128 + 2 * TOKEN_KERNEL_DIM + 2 * 3 * RELATION_KERNEL_DIM
    raise ValueError("unknown HCWDL-RKD bank kind")


def target_metadata_bytes_per_row(bank_kind: str) -> int:
    families = len(family_vocabulary(bank_kind))
    reasons = len(reason_vocabulary(bank_kind))
    return 45 + 28 * families + 2 * reasons


def target_logical_bytes_per_row(bank_kind: str) -> int:
    return 4 * target_core_values_per_row(bank_kind) + target_metadata_bytes_per_row(bank_kind)


def target_array_schema(bank_kind: str, rows: int) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    if isinstance(rows, bool) or rows < 0:
        raise ValueError("HCWDL-RKD target row count differs")
    families = len(family_vocabulary(bank_kind))
    reasons = len(reason_vocabulary(bank_kind))
    schema: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
        "source_file_id": (np.dtype("<u4"), (rows,)),
        "source_entry": (np.dtype("<u8"), (rows,)),
        "identity_digest": (np.dtype("u1"), (rows, 32)),
        "label": (np.dtype("u1"), (rows,)),
        "logits": (np.dtype("<f4"), (rows, 15)),
        "jet_penultimate": (np.dtype("<f4"), (rows, 128)),
        "token_family_eligibility": (np.dtype("u1"), (rows, families)),
        "relation_eligibility": (np.dtype("u1"), (rows, families, 3)),
        "token_count": (np.dtype("<u2"), (rows, families)),
        "token_scalar_pt_sum": (np.dtype("<f4"), (rows, families)),
        "relation_pair_count": (np.dtype("<u2"), (rows, families, 3)),
        "relation_effective_sample": (np.dtype("<f4"), (rows, families, 3)),
        "family_reason_counts": (np.dtype("<u2"), (rows, reasons)),
    }
    if bank_kind == ORDINARY_BANK:
        schema.update({
            "token_kernel_mean": (np.dtype("<f4"), (rows, TOKEN_KERNEL_DIM)),
            "relation_kernel_mean": (
                np.dtype("<f4"), (rows, 3, RELATION_KERNEL_DIM),
            ),
        })
    elif bank_kind == TOFF_BANK:
        schema.update({
            "token_kernel_mean_charged": (
                np.dtype("<f4"), (rows, TOKEN_KERNEL_DIM),
            ),
            "token_kernel_mean_neutral": (
                np.dtype("<f4"), (rows, TOKEN_KERNEL_DIM),
            ),
            "relation_kernel_mean_charged": (
                np.dtype("<f4"), (rows, 3, RELATION_KERNEL_DIM),
            ),
            "relation_kernel_mean_neutral": (
                np.dtype("<f4"), (rows, 3, RELATION_KERNEL_DIM),
            ),
        })
    else:
        raise ValueError("unknown HCWDL-RKD bank kind")
    return schema


def _identity_hex_rows(value: np.ndarray) -> tuple[str, ...]:
    return tuple(bytes(row).hex() for row in np.asarray(value, dtype=np.uint8))


def identity_order_sha256(value: np.ndarray) -> str:
    return canonical_sha256(list(_identity_hex_rows(value)))


def identity_set_sha256(value: np.ndarray) -> str:
    return canonical_sha256(sorted(_identity_hex_rows(value)))


class TargetPopulationHasher:
    """Streaming hash of the authenticated train row-to-label mapping."""

    def __init__(self, *, role: str = "train") -> None:
        if role != "train":
            raise ValueError("HCWDL-RKD target population role must be train")
        self._digest = hashlib.sha256()
        self._rows = 0

    @property
    def rows(self) -> int:
        return self._rows

    def update(
        self,
        *,
        source_file_id: np.ndarray,
        source_entry: np.ndarray,
        identity_digest: np.ndarray,
        label: np.ndarray,
    ) -> None:
        source_ids = np.asarray(source_file_id)
        entries = np.asarray(source_entry)
        identities = np.asarray(identity_digest)
        labels = np.asarray(label)
        rows = len(entries)
        if (
            source_ids.dtype != np.dtype("<u4") or source_ids.shape != (rows,)
            or entries.dtype != np.dtype("<u8") or entries.shape != (rows,)
            or identities.dtype != np.dtype("u1") or identities.shape != (rows, 32)
            or labels.dtype != np.dtype("u1") or labels.shape != (rows,)
        ):
            raise ValueError("HCWDL-RKD target population row schema differs")
        packed = np.empty(rows, dtype=_POPULATION_ROW_DTYPE)
        packed["source_file_id"] = source_ids
        packed["source_entry"] = entries
        packed["identity_digest"] = identities
        packed["label"] = labels
        self._digest.update(np.ascontiguousarray(packed).tobytes(order="C"))
        self._rows += rows

    def hexdigest(self) -> str:
        if self._rows <= 0:
            raise ValueError("HCWDL-RKD target population is empty")
        return canonical_sha256({
            "byte_length": int(self._rows * _POPULATION_ROW_DTYPE.itemsize),
            "contract": "HCWDL_REPRESENTATION_TARGET_POPULATION_ROWS/v1",
            "packed_c_order_byte_sha256": self._digest.hexdigest(),
            "role": "train",
            "row_schema": [
                {"dtype": "<u4", "name": "source_file_id", "shape": []},
                {"dtype": "<u8", "name": "source_entry", "shape": []},
                {"dtype": "|u1", "name": "identity_digest", "shape": [32]},
                {"dtype": "|u1", "name": "label", "shape": []},
            ],
            "rows": int(self._rows),
            "storage_order": "C",
        })


def target_population_rows_sha256(
    *,
    source_file_id: np.ndarray,
    source_entry: np.ndarray,
    identity_digest: np.ndarray,
    label: np.ndarray,
) -> str:
    hasher = TargetPopulationHasher()
    hasher.update(
        source_file_id=source_file_id,
        source_entry=source_entry,
        identity_digest=identity_digest,
        label=label,
    )
    return hasher.hexdigest()


def validate_target_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    bank_kind: str,
    expected_rows: int | None = None,
    expected_source_file_id: int | None = None,
) -> int:
    """Validate exact dtypes/shapes plus target eligibility semantics."""

    if not isinstance(arrays, Mapping) or "source_entry" not in arrays:
        raise ValueError("HCWDL-RKD target arrays are absent")
    rows = len(np.asarray(arrays["source_entry"]))
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("HCWDL-RKD target shard row count differs")
    if rows <= 0:
        raise ValueError("HCWDL-RKD target shard cannot be empty")
    schema = target_array_schema(bank_kind, rows)
    if set(arrays) != set(schema):
        raise ValueError("HCWDL-RKD target array names differ")
    for name, (dtype, shape) in schema.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"HCWDL-RKD target array shape/dtype/order differs: {name}")
        if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"HCWDL-RKD target array is nonfinite: {name}")
    labels = np.asarray(arrays["label"])
    if np.any(labels > 14):
        raise ValueError("HCWDL-RKD target label lies outside 0..14")
    source_ids = np.asarray(arrays["source_file_id"])
    entries = np.asarray(arrays["source_entry"])
    if expected_source_file_id is not None and np.any(source_ids != expected_source_file_id):
        raise ValueError("HCWDL-RKD shard crosses its source-file partition")
    if np.any(source_ids[1:] < source_ids[:-1]) or np.any(
        (source_ids[1:] == source_ids[:-1]) & (entries[1:] <= entries[:-1])
    ):
        raise ValueError("HCWDL-RKD target identities are not canonically ordered")
    identity_rows = _identity_hex_rows(np.asarray(arrays["identity_digest"]))
    if len(set(identity_rows)) != rows:
        raise ValueError("HCWDL-RKD target shard repeats an identity digest")
    token_eligible = np.asarray(arrays["token_family_eligibility"])
    relation_eligible = np.asarray(arrays["relation_eligibility"])
    if np.any(token_eligible > 1) or np.any(relation_eligible > 1):
        raise ValueError("HCWDL-RKD target eligibility is not binary")
    token_count = np.asarray(arrays["token_count"])
    if np.any(token_count > 128):
        raise ValueError("HCWDL-RKD target token count exceeds canonical trimming")
    if not np.array_equal(token_eligible.astype(bool), token_count > 0):
        raise ValueError("HCWDL-RKD token eligibility/count differs")
    token_pt = np.asarray(arrays["token_scalar_pt_sum"])
    if (
        np.any(token_pt < 0)
        or np.any((token_count == 0) & (token_pt != 0))
        or np.any((token_count > 0) & (token_pt <= 0))
    ):
        raise ValueError("HCWDL-RKD token scalar-pT metadata differs")
    pair_count = np.asarray(arrays["relation_pair_count"])
    ess = np.asarray(arrays["relation_effective_sample"])
    if np.any(pair_count > 496) or np.any(ess > pair_count.astype(np.float32) + 1.0e-4):
        raise ValueError("HCWDL-RKD relation population exceeds top-32 semantics")
    expected_relation = (pair_count >= 4) & (ess >= np.float32(3.0))
    if np.any(ess < 0) or not np.array_equal(relation_eligible.astype(bool), expected_relation):
        raise ValueError("HCWDL-RKD relation eligibility/count/ESS differs")
    if np.any(relation_eligible > token_eligible[:, :, None]):
        raise ValueError("HCWDL-RKD relation target is eligible for an empty token family")
    if np.any((pair_count == 0) & (ess != 0)):
        raise ValueError("HCWDL-RKD empty relation stratum has nonzero ESS")
    if bank_kind == ORDINARY_BANK:
        token = np.asarray(arrays["token_kernel_mean"])
        relation = np.asarray(arrays["relation_kernel_mean"])
        if np.any(token[~token_eligible[:, 0].astype(bool)] != 0):
            raise ValueError("HCWDL-RKD ineligible token sketch is nonzero")
        if np.any(relation[~relation_eligible[:, 0].astype(bool)] != 0):
            raise ValueError("HCWDL-RKD ineligible relation sketch is nonzero")
        if not np.array_equal(
            np.asarray(arrays["family_reason_counts"])[:, 0], token_count[:, 0],
        ):
            raise ValueError("HCWDL-RKD ordinary family reason conservation differs")
    else:
        reasons = np.asarray(arrays["family_reason_counts"])
        if np.any(np.sum(reasons, axis=1, dtype=np.uint32) > 128):
            raise ValueError("HCWDL-RKD companion-HLT reasons exceed canonical trimming")
        for family_index, family in enumerate(TOFF_FAMILIES):
            token = np.asarray(arrays[f"token_kernel_mean_{family}"])
            relation = np.asarray(arrays[f"relation_kernel_mean_{family}"])
            if np.any(token[~token_eligible[:, family_index].astype(bool)] != 0):
                raise ValueError(f"HCWDL-RKD ineligible {family} token sketch is nonzero")
            if np.any(relation[~relation_eligible[:, family_index].astype(bool)] != 0):
                raise ValueError(f"HCWDL-RKD ineligible {family} relation sketch is nonzero")
    return rows


def _expected_teacher(bank_id: str) -> tuple[str, str, str]:
    if bank_id == "TOFF":
        return "TOFF", "toff", "shared"
    if bank_id == "D100":
        return "D100", "d100", "shared"
    suffix = bank_id[-1]
    domain = bank_id[:-1].lower()
    return bank_id, domain, "cold" if suffix == "c" else "warm"


def build_logical_target_bank(
    *, bank_id: str, teacher: Mapping[str, Any], parents: Mapping[str, Any],
) -> dict[str, Any]:
    bank_kind = bank_kind_for_id(bank_id)
    normalized_parents = validate_parent_hashes(parents)
    if not LOGICAL_REQUIRED_PARENTS.issubset(normalized_parents):
        raise ValueError("HCWDL-RKD logical bank lacks required scientific parents")
    expected_node, expected_domain, expected_track = _expected_teacher(bank_id)
    required_teacher = {
        "node_id", "domain", "track", "selected_report_sha256",
        "checkpoint_byte_sha256", "checkpoint_logical_sha256", "tap_sha256",
        "installed_weaver_signature_sha256",
    }
    if not isinstance(teacher, Mapping) or set(teacher) != required_teacher:
        raise ValueError("HCWDL-RKD logical-bank teacher record differs")
    if (
        teacher["node_id"] != expected_node
        or teacher["domain"] != expected_domain
        or teacher["track"] != expected_track
    ):
        raise ValueError("HCWDL-RKD logical-bank teacher/domain/track differs")
    normalized_teacher = dict(teacher)
    for name in required_teacher - {"node_id", "domain", "track"}:
        normalized_teacher[name] = require_sha256(
            normalized_teacher[name], name=f"logical-bank teacher {name}",
        )
    artifact = build_versioned_artifact(
        TARGET_LOGICAL_BANK_CONTRACT, parents=normalized_parents,
        payload={
            "logical_bank_id": bank_id,
            "bank_kind": bank_kind,
            "teacher": normalized_teacher,
            "family_vocabulary": list(family_vocabulary(bank_kind)),
            "reason_vocabulary": list(reason_vocabulary(bank_kind)),
            "token_kernel_dimension": TOKEN_KERNEL_DIM,
            "relation_kernel_dimension": RELATION_KERNEL_DIM,
            "relation_strata": list(RELATION_STRATA),
            "role": "train",
            "scientific_target_identity_sha256": canonical_sha256({
                "bank_id": bank_id, "teacher": normalized_teacher,
                "scientific_parents": normalized_parents,
            }),
        },
    )
    return artifact


def validate_logical_target_bank(
    value: Mapping[str, Any], *, expected_bank_id: str | None = None,
) -> str:
    digest = validate_versioned_artifact(
        value, expected_contract=TARGET_LOGICAL_BANK_CONTRACT,
        required_payload_keys=(
            "logical_bank_id", "bank_kind", "teacher", "family_vocabulary",
            "reason_vocabulary", "role", "scientific_target_identity_sha256",
        ),
    )
    payload = value["payload"]
    bank_id = str(payload["logical_bank_id"])
    if expected_bank_id is not None and bank_id != expected_bank_id:
        raise ValueError("HCWDL-RKD logical bank ID differs")
    if payload["bank_kind"] != bank_kind_for_id(bank_id) or payload["role"] != "train":
        raise ValueError("HCWDL-RKD logical bank kind/role differs")
    if payload["family_vocabulary"] != list(family_vocabulary(payload["bank_kind"])):
        raise ValueError("HCWDL-RKD logical bank family vocabulary differs")
    if payload["reason_vocabulary"] != list(reason_vocabulary(payload["bank_kind"])):
        raise ValueError("HCWDL-RKD logical bank reason vocabulary differs")
    rebuilt = build_logical_target_bank(
        bank_id=bank_id, teacher=payload["teacher"], parents=value["parents"],
    )
    if rebuilt["content_hash"] != digest:
        raise ValueError("HCWDL-RKD logical bank semantics differ")
    return digest


def expected_screen_consumer_nodes(bank_id: str) -> tuple[str, ...]:
    rung = {"D0": 1, "D25": 2, "D50": 3, "D75": 4}
    if bank_id == "D100":
        return (
            "RSET_M5c", "RSET_M5w", "RREL_M5c", "RREL_M5w",
            "RSET_M5c_JET_ONLY_REP", "RREL_M5c_NO_REL_REP",
            "RSET_M5c_WITHIN_CLASS_SHUFFLED_REP",
            "RREL_M5c_WITHIN_CLASS_SHUFFLED_REP",
        )
    if bank_id == "TOFF":
        return ("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w")
    track = bank_id[-1]
    base = bank_id[:-1]
    if base not in rung:
        raise ValueError("unknown HCWDL-RKD target-bank consumer mapping")
    return (f"RSET_M{rung[base]}{track}", f"RREL_M{rung[base]}{track}")


def _consumer_node_spec(node_id: str) -> Any:
    if node_id in NODE_REGISTRY:
        return NODE_REGISTRY[node_id]
    if node_id in CONTROL_REGISTRY:
        return CONTROL_REGISTRY[node_id]
    raise ValueError("HCWDL-RKD target consumer is not a registered node/control")


def build_target_consumer_row(
    logical_bank: Mapping[str, Any],
    *,
    purpose: str,
    campaign_sha256: str,
    recipe_sha256: str,
    node_id: str,
    seed: int,
) -> dict[str, Any]:
    """Build the exact Section-24.4 execution identity for one consumer.

    The physical target generation is deliberately absent from the identity.
    Recovery therefore carries the original screen/confirmation identity row
    into a new consumer registry instead of inventing an execution after the
    old generation has been cleaned.
    """

    logical_hash = validate_logical_target_bank(logical_bank)
    if purpose not in {"screen", "confirmation"}:
        raise ValueError("HCWDL-RKD consumer identity purpose differs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise ValueError("HCWDL-RKD target consumer seed differs")
    spec = _consumer_node_spec(node_id)
    if spec.target_bank_identity != logical_bank["payload"]["logical_bank_id"]:
        raise ValueError("HCWDL-RKD target consumer uses the wrong logical bank")
    if purpose == "confirmation" and (
        node_id not in NODE_REGISTRY or int(spec.rung) != 6
    ):
        raise ValueError("HCWDL-RKD confirmation consumer is not a primary M6 node")
    execution_id, identity_payload = derive_representation_execution_id(
        campaign_sha256=campaign_sha256,
        strategy=str(spec.strategy),
        node_id=node_id,
        purpose=purpose,
        seed=seed,
        initialization_parent=spec.initialization_parent,
        teacher=str(spec.representation_logit_teacher),
        logical_target_bank_sha256=logical_hash,
        target_purpose=purpose,
        recipe_sha256=recipe_sha256,
    )
    short_strategy = "RSET" if str(spec.strategy).endswith("SET/v1") else "RREL"
    return {
        "execution_id": execution_id,
        "execution_identity_payload": identity_payload,
        "node_id": node_id,
        "strategy": short_strategy,
        "track": str(spec.track),
        "seed": seed,
    }


def build_miniature_target_consumer_row(
    logical_bank: Mapping[str, Any], *, campaign_sha256: str,
    recipe_sha256: str, bounded_row_limit: int,
) -> dict[str, Any]:
    """Build the sole non-training consumer for a bounded cache miniature.

    This row is intentionally incompatible with screen, confirmation and
    recovery consumers.  It authorizes only RAM materialization, an exact
    identity join and immediate cleanup of at most 4,096 train-role rows.
    """

    logical_hash = validate_logical_target_bank(logical_bank)
    if (
        isinstance(bounded_row_limit, bool)
        or not isinstance(bounded_row_limit, int)
        or not 1 <= bounded_row_limit <= 4096
    ):
        raise ValueError("cache-miniature bounded row limit differs")
    bank_id = str(logical_bank["payload"]["logical_bank_id"])
    payload = {
        "contract": "HCWDL_REPRESENTATION_CACHE_MINIATURE_CONSUMER/v1",
        "campaign": require_sha256(campaign_sha256, name="miniature campaign"),
        "recipe": require_sha256(recipe_sha256, name="miniature recipe"),
        "logical_target_bank": logical_hash,
        "logical_bank_id": bank_id,
        "purpose": "miniature",
        "bounded_row_limit": bounded_row_limit,
        "role": "train",
        "scientific_authorization": False,
        "training_consumer_authorized": False,
    }
    return {
        "execution_id": canonical_sha256(payload),
        "execution_identity_payload": payload,
        "node_id": f"CACHE_MINIATURE_{bank_id}",
        "strategy": "CACHE_MINIATURE",
        "track": "non_scientific",
        "seed": 0,
    }


def build_nonfinal_acceptance_target_consumer_row(
    logical_bank: Mapping[str, Any], *, trajectory_id: str,
    acceptance_bootstrap_sha256: str, runtime_binding_sha256: str,
    source_runtime_row_sha256: str, action_registry_sha256: str,
    recipe_sha256: str,
) -> dict[str, Any]:
    """Build one bounded training consumer without impersonating a campaign row."""

    logical_hash = validate_logical_target_bank(logical_bank)
    bank_id = str(logical_bank["payload"]["logical_bank_id"])
    trajectories = NONFINAL_ACCEPTANCE_TARGET_TRAJECTORIES.get(bank_id)
    if trajectories is None or trajectory_id not in trajectories:
        raise ValueError("non-final acceptance target trajectory differs")
    action_ids, node_id = trajectories[trajectory_id]
    spec = _consumer_node_spec(node_id)
    if spec.target_bank_identity != bank_id:
        raise ValueError("non-final acceptance target consumer uses the wrong bank")
    short_strategy = "RSET" if str(spec.strategy).endswith("SET/v1") else "RREL"
    payload = {
        "contract": NONFINAL_ACCEPTANCE_TARGET_CONSUMER_CONTRACT,
        "acceptance_bootstrap_sha256": require_sha256(
            acceptance_bootstrap_sha256, name="acceptance bootstrap",
        ),
        "runtime_binding_sha256": require_sha256(
            runtime_binding_sha256, name="acceptance runtime binding",
        ),
        "source_runtime_row_sha256": require_sha256(
            source_runtime_row_sha256, name="acceptance source runtime row",
        ),
        "action_registry_sha256": require_sha256(
            action_registry_sha256, name="acceptance action registry",
        ),
        "representation_recipe_sha256": require_sha256(
            recipe_sha256, name="acceptance representation recipe",
        ),
        "logical_target_bank": logical_hash,
        "logical_bank_id": bank_id,
        "purpose": NONFINAL_ACCEPTANCE_TARGET_PURPOSE,
        "trajectory_id": trajectory_id,
        "action_ids": list(action_ids),
        "node_id": node_id,
        "strategy": short_strategy,
        "track": str(spec.track),
        "seed": 1337,
        "role": "train",
        "bounded_row_limit": NONFINAL_ACCEPTANCE_TARGET_ROWS,
        "training_consumer_authorized": True,
        "authorization_scope": "bounded_nonfinal_acceptance_only",
        "scientific_campaign_training_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
        "shared_final_authorized": False,
    }
    return {
        "execution_id": canonical_sha256(payload),
        "execution_identity_payload": payload,
        "node_id": node_id,
        "strategy": short_strategy,
        "track": str(spec.track),
        "seed": 1337,
    }


def build_nonfinal_acceptance_target_consumer_registry(
    logical_bank: Mapping[str, Any], *, acceptance_bootstrap_sha256: str,
    runtime_binding_sha256: str, source_runtime_row_sha256: str,
    action_registry_sha256: str, recipe_sha256: str,
) -> dict[str, Any]:
    """Build the exact D0c/D0w bounded consumer set and generation parent."""

    bank_id = str(logical_bank["payload"]["logical_bank_id"])
    trajectories = NONFINAL_ACCEPTANCE_TARGET_TRAJECTORIES.get(bank_id)
    if trajectories is None:
        raise ValueError("non-final acceptance target bank differs")
    consumers = [
        build_nonfinal_acceptance_target_consumer_row(
            logical_bank, trajectory_id=trajectory_id,
            acceptance_bootstrap_sha256=acceptance_bootstrap_sha256,
            runtime_binding_sha256=runtime_binding_sha256,
            source_runtime_row_sha256=source_runtime_row_sha256,
            action_registry_sha256=action_registry_sha256,
            recipe_sha256=recipe_sha256,
        )
        for trajectory_id in trajectories
    ]
    generation_parent = canonical_sha256({
        "contract": "HCWDL_REPRESENTATION_NONFINAL_ACCEPTANCE_TARGET_PARENT/v1",
        "acceptance_bootstrap_sha256": acceptance_bootstrap_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "source_runtime_row_sha256": source_runtime_row_sha256,
        "action_registry_sha256": action_registry_sha256,
        "representation_recipe_sha256": recipe_sha256,
        "logical_bank_id": bank_id,
    })
    return build_target_consumer_registry(
        logical_bank, purpose=NONFINAL_ACCEPTANCE_TARGET_PURPOSE,
        consumers=consumers, generation_parent_sha256=generation_parent,
    )


def _validate_consumer_rows(
    rows: Sequence[Mapping[str, Any]], *, logical_bank: Mapping[str, Any], purpose: str,
) -> list[dict[str, Any]]:
    logical_hash = validate_logical_target_bank(logical_bank)
    bank_id = str(logical_bank["payload"]["logical_bank_id"])
    required = {
        "execution_id", "execution_identity_payload", "node_id", "strategy",
        "track", "seed",
    }
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("HCWDL-RKD target consumer row differs")
        item = dict(row)
        item["execution_id"] = require_sha256(
            item["execution_id"], name="target consumer execution ID",
        )
        if purpose == "miniature":
            payload = item.get("execution_identity_payload")
            bank_id = str(logical_bank["payload"]["logical_bank_id"])
            if (
                item.get("strategy") != "CACHE_MINIATURE"
                or item.get("track") != "non_scientific"
                or item.get("node_id") != f"CACHE_MINIATURE_{bank_id}"
                or item.get("seed") != 0
                or not isinstance(payload, Mapping)
                or set(payload) != {
                    "contract", "campaign", "recipe", "logical_target_bank",
                    "logical_bank_id", "purpose", "bounded_row_limit", "role",
                    "scientific_authorization", "training_consumer_authorized",
                }
                or payload.get("contract")
                != "HCWDL_REPRESENTATION_CACHE_MINIATURE_CONSUMER/v1"
                or payload.get("logical_target_bank") != logical_hash
                or payload.get("logical_bank_id") != bank_id
                or payload.get("purpose") != "miniature"
                or payload.get("role") != "train"
                or payload.get("scientific_authorization") is not False
                or payload.get("training_consumer_authorized") is not False
                or isinstance(payload.get("bounded_row_limit"), bool)
                or not isinstance(payload.get("bounded_row_limit"), int)
                or not 1 <= int(payload["bounded_row_limit"]) <= 4096
                or item.get("execution_id") != canonical_sha256(dict(payload))
            ):
                raise ValueError("HCWDL-RKD miniature consumer semantics differ")
            require_sha256(payload.get("campaign"), name="miniature campaign")
            require_sha256(payload.get("recipe"), name="miniature recipe")
            normalized.append(item)
            continue
        if purpose == NONFINAL_ACCEPTANCE_TARGET_PURPOSE:
            payload = item.get("execution_identity_payload")
            if not isinstance(payload, Mapping):
                raise ValueError("non-final acceptance consumer payload differs")
            expected = build_nonfinal_acceptance_target_consumer_row(
                logical_bank,
                trajectory_id=str(payload.get("trajectory_id", "")),
                acceptance_bootstrap_sha256=str(
                    payload.get("acceptance_bootstrap_sha256", "")
                ),
                runtime_binding_sha256=str(payload.get("runtime_binding_sha256", "")),
                source_runtime_row_sha256=str(
                    payload.get("source_runtime_row_sha256", "")
                ),
                action_registry_sha256=str(payload.get("action_registry_sha256", "")),
                recipe_sha256=str(payload.get("representation_recipe_sha256", "")),
            )
            if item != expected:
                raise ValueError("non-final acceptance target consumer semantics differ")
            normalized.append(item)
            continue
        if item["strategy"] not in {"RSET", "RREL"} or item["track"] not in {"cold", "warm"}:
            raise ValueError("HCWDL-RKD target consumer strategy/track differs")
        spec = _consumer_node_spec(str(item["node_id"]))
        expected_strategy = "RSET" if str(spec.strategy).endswith("SET/v1") else "RREL"
        if (
            item["strategy"] != expected_strategy
            or item["track"] != spec.track
            or spec.target_bank_identity != bank_id
        ):
            raise ValueError("HCWDL-RKD target consumer node metadata differs")
        if (
            isinstance(item["seed"], bool)
            or not isinstance(item["seed"], int)
            or item["seed"] <= 0
        ):
            raise ValueError("HCWDL-RKD target consumer seed differs")
        identity_payload = item["execution_identity_payload"]
        if not isinstance(identity_payload, Mapping):
            raise ValueError("HCWDL-RKD target consumer identity payload differs")
        identity_purpose = str(identity_payload.get("purpose", ""))
        if purpose in {"screen", "confirmation"} and identity_purpose != purpose:
            raise ValueError("HCWDL-RKD target consumer purpose differs")
        if purpose == "recovery" and identity_purpose not in {"screen", "confirmation"}:
            raise ValueError("HCWDL-RKD recovery consumer lacks an original purpose")
        expected_id, expected_payload = derive_representation_execution_id(
            campaign_sha256=str(identity_payload.get("campaign", "")),
            strategy=str(spec.strategy),
            node_id=str(item["node_id"]),
            purpose=identity_purpose,
            seed=item["seed"],
            initialization_parent=spec.initialization_parent,
            teacher=str(spec.representation_logit_teacher),
            logical_target_bank_sha256=logical_hash,
            target_purpose=identity_purpose,
            recipe_sha256=str(identity_payload.get("recipe", "")),
        )
        if item["execution_id"] != expected_id or dict(identity_payload) != expected_payload:
            raise ValueError("HCWDL-RKD target consumer execution identity differs")
        item["execution_identity_payload"] = expected_payload
        normalized.append(item)
    if len({row["execution_id"] for row in normalized}) != len(normalized):
        raise ValueError("HCWDL-RKD target consumer registry repeats an execution")
    if purpose == "screen":
        if {row["node_id"] for row in normalized} != set(expected_screen_consumer_nodes(bank_id)):
            raise ValueError("HCWDL-RKD screen target consumer set differs")
        if any(row["seed"] != 1337 for row in normalized):
            raise ValueError("HCWDL-RKD screen target consumer seed differs")
    elif purpose == "confirmation":
        expected_nodes = set(expected_screen_consumer_nodes("TOFF"))
        if bank_id != "TOFF" or len(normalized) != 20:
            raise ValueError("HCWDL-RKD confirmation target registry differs")
        if {row["node_id"] for row in normalized} != expected_nodes:
            raise ValueError("HCWDL-RKD confirmation target node set differs")
        for node in expected_nodes:
            if {row["seed"] for row in normalized if row["node_id"] == node} != {
                11, 22, 33, 44, 55,
            }:
                raise ValueError("HCWDL-RKD confirmation target seeds differ")
    elif purpose == "recovery":
        if not normalized:
            raise ValueError("HCWDL-RKD recovery target registry is empty")
    elif purpose == "miniature":
        if len(normalized) != 1:
            raise ValueError("HCWDL-RKD miniature target registry must have one verifier")
    elif purpose == NONFINAL_ACCEPTANCE_TARGET_PURPOSE:
        trajectories = NONFINAL_ACCEPTANCE_TARGET_TRAJECTORIES.get(bank_id)
        if trajectories is None or {
            row["execution_identity_payload"]["trajectory_id"] for row in normalized
        } != set(trajectories):
            raise ValueError("non-final acceptance target consumer set differs")
        if len(normalized) != len(trajectories):
            raise ValueError("non-final acceptance target registry repeats a trajectory")
    else:
        raise ValueError("unknown HCWDL-RKD target purpose")
    return sorted(normalized, key=lambda row: row["execution_id"])


def build_target_consumer_registry(
    logical_bank: Mapping[str, Any],
    *, purpose: str, consumers: Sequence[Mapping[str, Any]],
    generation_parent_sha256: str,
) -> dict[str, Any]:
    logical_hash = validate_logical_target_bank(logical_bank)
    bank_id = logical_bank["payload"]["logical_bank_id"]
    parent_hash = require_sha256(
        generation_parent_sha256, name="target generation-purpose parent SHA-256",
    )
    rows = _validate_consumer_rows(
        consumers, logical_bank=logical_bank, purpose=purpose,
    )
    return build_versioned_artifact(
        TARGET_CONSUMER_REGISTRY_CONTRACT,
        parents={"logical_bank": logical_hash, "generation_parent": parent_hash},
        payload={
            "logical_bank_id": bank_id, "purpose": purpose,
            "generation_parent_sha256": parent_hash, "consumers": rows,
            "consumer_execution_ids_sha256": canonical_sha256(
                [row["execution_id"] for row in rows]
            ),
        },
    )


def validate_target_consumer_registry(
    value: Mapping[str, Any], *, logical_bank: Mapping[str, Any],
) -> str:
    logical_hash = validate_logical_target_bank(logical_bank)
    digest = validate_versioned_artifact(
        value, expected_contract=TARGET_CONSUMER_REGISTRY_CONTRACT,
        required_payload_keys=(
            "logical_bank_id", "purpose", "generation_parent_sha256", "consumers",
            "consumer_execution_ids_sha256",
        ),
    )
    payload = value["payload"]
    if (
        payload["logical_bank_id"] != logical_bank["payload"]["logical_bank_id"]
        or value["parents"].get("logical_bank") != logical_hash
        or value["parents"].get("generation_parent") != payload["generation_parent_sha256"]
    ):
        raise ValueError("HCWDL-RKD target registry logical-bank lineage differs")
    rows = _validate_consumer_rows(
        payload["consumers"], logical_bank=logical_bank, purpose=payload["purpose"],
    )
    if rows != payload["consumers"] or payload["consumer_execution_ids_sha256"] != canonical_sha256(
        [row["execution_id"] for row in rows]
    ):
        raise ValueError("HCWDL-RKD target consumer registry semantics differ")
    return digest


_FORWARD_REQUIRED = frozenset({
    "teacher", "producer", "device", "precision", "determinism", "batching",
    "implementation", "source_partitions",
})


def validate_target_forward_spec_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _FORWARD_REQUIRED:
        raise ValueError("HCWDL-RKD target-forward specification fields differ")
    teacher = payload["teacher"]
    teacher_keys = {
        "checkpoint_byte_sha256", "checkpoint_logical_sha256", "model_config_sha256",
        "architecture_sha256", "tap_sha256", "kernel_resources_sha256",
        "kernel_array_logical_hashes",
    }
    if not isinstance(teacher, Mapping) or set(teacher) != teacher_keys:
        raise ValueError("HCWDL-RKD target-forward teacher fields differ")
    for key in teacher_keys - {"kernel_array_logical_hashes"}:
        require_sha256(teacher.get(key), name=f"target-forward teacher {key}")
    kernel_hashes = teacher["kernel_array_logical_hashes"]
    if not isinstance(kernel_hashes, Mapping) or set(kernel_hashes) != set(
        KERNEL_RESOURCE_NAMES
    ):
        raise ValueError("HCWDL-RKD target-forward kernel logical hashes differ")
    for name, digest in kernel_hashes.items():
        require_sha256(digest, name=f"target-forward kernel logical hash {name}")
    producer = payload["producer"]
    if not isinstance(producer, Mapping) or set(producer) != {
        "source_commit", "source_snapshot_sha256", "packages",
    }:
        raise ValueError("HCWDL-RKD target-forward producer fields differ")
    commit = str(producer.get("source_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("HCWDL-RKD target-forward producer commit differs")
    require_sha256(producer.get("source_snapshot_sha256"), name="target-forward source snapshot")
    packages = producer.get("packages")
    required_packages = {"python", "torch", "cuda", "cudnn", "numpy", "awkward", "uproot", "weaver"}
    if not isinstance(packages, Mapping) or set(packages) != required_packages or any(
        not isinstance(value, str) or not value for value in packages.values()
    ):
        raise ValueError("HCWDL-RKD target-forward package inventory differs")
    device = payload["device"]
    if (
        not isinstance(device, Mapping)
        or set(device) != {
            "request", "architecture", "model", "compute_capability", "driver", "runtime",
        }
        or
        device.get("request") != "gpu:gh200:1"
        or device.get("architecture") != "Hopper"
        or not all(isinstance(device.get(key), str) and device.get(key) for key in (
            "model", "compute_capability", "driver", "runtime",
        ))
    ):
        raise ValueError("HCWDL-RKD target-forward device signature differs")
    precision = payload["precision"]
    if precision != {
        "parameters": "float32", "inputs": "float32", "activations": "float32",
        "autocast": False, "matmul_tf32": False, "cudnn_tf32": False,
        "reduced_precision_fp32_reduction": False, "output_order": "C",
    }:
        raise ValueError("HCWDL-RKD target-forward precision policy differs")
    determinism = payload["determinism"]
    if (
        not isinstance(determinism, Mapping)
        or set(determinism) != {
            "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
            "cublas_workspace_config", "rng_states_sha256",
        }
        or
        determinism.get("deterministic_algorithms") is not True
        or determinism.get("cudnn_deterministic") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cublas_workspace_config") != ":4096:8"
    ):
        raise ValueError("HCWDL-RKD target-forward determinism policy differs")
    require_sha256(determinism.get("rng_states_sha256"), name="target-forward RNG states")
    batching = payload["batching"]
    if not isinstance(batching, Mapping):
        raise ValueError("HCWDL-RKD target-forward batch partition differs")
    if batching != {
        "batch_size": TARGET_FORWARD_BATCH_SIZE,
        "order": "source_file_id_then_source_entry_v1",
        "cross_source_batches": False,
        "final_short_batch_per_source": True,
        "padding": False,
        "row_duplication": False,
    }:
        raise ValueError("HCWDL-RKD target-forward batch partition differs")
    implementation = payload["implementation"]
    required_implementation = {
        "input_decoding_sha256", "feature_layout_sha256", "trimmer_sha256",
        "family_code_sha256", "surface_capture_sha256", "sketch_arithmetic_sha256",
        "teacher_input_fields",
    }
    if not isinstance(implementation, Mapping) or set(implementation) != required_implementation:
        raise ValueError("HCWDL-RKD target-forward implementation signature differs")
    for key, digest in implementation.items():
        if key == "teacher_input_fields":
            continue
        require_sha256(digest, name=f"target-forward implementation {key}")
    input_fields = implementation["teacher_input_fields"]
    if (
        not isinstance(input_fields, list) or not input_fields
        or input_fields != sorted(input_fields)
        or len(input_fields) != len(set(input_fields))
        or any(
            not isinstance(name, str) or not _TEACHER_INPUT.fullmatch(name)
            or any(fragment in name.lower() for fragment in _FORBIDDEN_TEACHER_INPUT_FRAGMENTS)
            for name in input_fields
        )
    ):
        raise ValueError("HCWDL-RKD target-forward teacher input allow-list differs")
    partitions = payload["source_partitions"]
    if (
        not isinstance(partitions, list) or not partitions
        or partitions != sorted(partitions)
        or len(partitions) != len(set(partitions))
        or any(not isinstance(name, str) or not _PARTITION.fullmatch(name) for name in partitions)
    ):
        raise ValueError("HCWDL-RKD target-forward source partitions differ")
    forbidden = {"logical_target_sha256", "sentinel_hashes", "output_array_hashes"}
    if forbidden & set(payload):
        raise ValueError("HCWDL-RKD pre-build forward spec contains output-derived data")


def build_target_forward_spec(
    *, parents: Mapping[str, Any], payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_forward_spec_payload(payload)
    return build_versioned_artifact(
        TARGET_FORWARD_SPEC_CONTRACT, parents=parents, payload=payload,
    )


def validate_target_forward_spec(
    value: Mapping[str, Any], *, expected_parents: Mapping[str, Any] | None = None,
) -> str:
    digest = validate_versioned_artifact(
        value, expected_contract=TARGET_FORWARD_SPEC_CONTRACT,
        expected_parents=expected_parents,
    )
    validate_target_forward_spec_payload(value["payload"])
    return digest


def _validate_forward_logical_lineage(
    forward_spec: Mapping[str, Any], logical_bank: Mapping[str, Any],
) -> None:
    logical_hash = logical_bank["content_hash"]
    teacher = logical_bank["payload"]["teacher"]
    forward_teacher = forward_spec["payload"]["teacher"]
    if (
        forward_spec["parents"].get("logical_bank") != logical_hash
        or forward_teacher["checkpoint_byte_sha256"]
        != teacher["checkpoint_byte_sha256"]
        or forward_teacher["checkpoint_logical_sha256"]
        != teacher["checkpoint_logical_sha256"]
        or forward_teacher["tap_sha256"] != teacher["tap_sha256"]
        or forward_teacher["architecture_sha256"]
        != logical_bank["parents"]["architecture"]
        or forward_teacher["kernel_resources_sha256"]
        != logical_bank["parents"]["kernel_resources"]
    ):
        raise ValueError("HCWDL-RKD target-forward teacher/logical-bank lineage differs")


def derive_target_generation_id(
    logical_bank_sha256: str, consumer_registry_sha256: str,
    *, purpose: str, generation_parent_sha256: str,
) -> str:
    if purpose not in {
        "screen", "confirmation", "recovery", "miniature",
        NONFINAL_ACCEPTANCE_TARGET_PURPOSE,
    }:
        raise ValueError("unknown HCWDL-RKD target generation purpose")
    return canonical_sha256({
        "logical_bank_sha256": require_sha256(
            logical_bank_sha256, name="logical bank SHA-256",
        ),
        "purpose": purpose,
        "consumer_registry_sha256": require_sha256(
            consumer_registry_sha256, name="consumer registry SHA-256",
        ),
        "generation_parent_sha256": require_sha256(
            generation_parent_sha256, name="generation parent SHA-256",
        ),
    })


def _normalize_partition_specs(
    partitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    if not isinstance(partitions, Mapping) or not partitions:
        raise ValueError("HCWDL-RKD target partition registry is empty")
    result = {}
    for raw_name, raw in partitions.items():
        name = str(raw_name)
        if name != raw_name or not _PARTITION.fullmatch(name) or not isinstance(raw, Mapping):
            raise ValueError("HCWDL-RKD target partition name/record differs")
        if set(raw) != {"rows", "source_file_id"}:
            raise ValueError("HCWDL-RKD target partition fields differ")
        rows = raw["rows"]
        source_id = raw["source_file_id"]
        if (
            isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            or isinstance(source_id, bool) or not isinstance(source_id, int)
            or not 0 <= source_id < 2**32
        ):
            raise ValueError("HCWDL-RKD target partition size/source ID differs")
        result[name] = {"rows": rows, "source_file_id": source_id}
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class TargetGenerationContext:
    bank_root: Path
    logical_bank: Mapping[str, Any]
    consumer_registry: Mapping[str, Any]
    forward_spec: Mapping[str, Any]
    build_intent: Mapping[str, Any]
    generation_id: str
    build_owner_id: str
    staging_directory: Path
    committed_directory: Path

    @property
    def bank_id(self) -> str:
        return str(self.logical_bank["payload"]["logical_bank_id"])

    @property
    def bank_kind(self) -> str:
        return str(self.logical_bank["payload"]["bank_kind"])


def _gib_request_bytes(value: Any) -> int:
    text = str(value)
    if not text.endswith("G") or not text[:-1].isdigit() or int(text[:-1]) <= 0:
        raise ValueError("HCWDL-RKD measured target memory request differs")
    return int(text[:-1]) * 1024**3


def _validate_preflight_artifacts(
    *,
    logical_bank: Mapping[str, Any],
    rows: int,
    storage_estimate: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
    slurm_mem_per_node_bytes: int,
    peak_runtime_bytes: int,
) -> tuple[str, str]:
    storage_hash = validate_content_hash(
        storage_estimate,
        expected_contract=STORAGE_ESTIMATE_CONTRACT,
        expected_schema_version=1,
    )
    profile_hash = validate_measured_profile(resource_profile)
    kind = str(logical_bank["payload"]["bank_kind"])
    ordinary_bytes = rows * target_logical_bytes_per_row(ORDINARY_BANK)
    toff_bytes = rows * target_logical_bytes_per_row(TOFF_BANK)
    expected_storage_fields = {
        "contract", "schema_version", "parent_import_sha256", "row_counts",
        "logical_bytes_per_ordinary_row", "logical_bytes_per_toff_row",
        "ordinary_bank_bytes", "toff_bank_bytes",
        "committed_target_generation_bytes",
        "interrupted_target_generation_reserve_bytes",
        "peak_target_staging_plus_committed_bytes",
        "prediction_identity_bytes_per_row", "prediction_logit_bytes_per_row",
        "prediction_payload_bytes_per_row", "prediction_finalists",
        "prediction_logits_bytes",
        "retained_resume_bytes", "selected_checkpoint_bytes",
        "final_assignment_bytes", "fixed_artifact_bytes",
        "subtotal_before_filesystem_headroom_bytes",
        "filesystem_headroom_numerator", "filesystem_headroom_denominator",
        "filesystem_headroom_bytes", "estimated_campaign_peak_durable_bytes",
        "simultaneously_committed_target_banks", "compression_saving_assumed",
        "operational_fixed_sizes_measured", "content_hash",
    }
    row_counts = storage_estimate.get("row_counts")
    integer_fields = (
        "logical_bytes_per_ordinary_row", "logical_bytes_per_toff_row",
        "ordinary_bank_bytes", "toff_bank_bytes",
        "committed_target_generation_bytes",
        "interrupted_target_generation_reserve_bytes",
        "peak_target_staging_plus_committed_bytes", "filesystem_headroom_bytes",
        "prediction_identity_bytes_per_row", "prediction_logit_bytes_per_row",
        "prediction_payload_bytes_per_row", "prediction_finalists",
        "prediction_logits_bytes",
        "retained_resume_bytes", "selected_checkpoint_bytes",
        "final_assignment_bytes", "fixed_artifact_bytes",
        "subtotal_before_filesystem_headroom_bytes",
        "filesystem_headroom_numerator", "filesystem_headroom_denominator",
        "estimated_campaign_peak_durable_bytes",
    )
    integer_values_are_valid = all(
        not isinstance(storage_estimate.get(name), bool)
        and isinstance(storage_estimate.get(name), int)
        and int(storage_estimate[name]) >= 0
        for name in integer_fields
    )
    committed = max(ordinary_bytes, toff_bytes)
    interrupted = storage_estimate.get(
        "interrupted_target_generation_reserve_bytes", -1,
    )
    fixed_values = tuple(
        storage_estimate.get(name, -1) for name in (
            "retained_resume_bytes", "selected_checkpoint_bytes",
            "final_assignment_bytes", "fixed_artifact_bytes",
        )
    )
    subtotal = (
        committed + int(interrupted)
        + int(storage_estimate.get("prediction_logits_bytes", -1))
        + sum(int(value) for value in fixed_values)
    ) if integer_values_are_valid else -1
    headroom = (subtotal + 3) // 4 if subtotal >= 0 else -1
    final_rows = row_counts.get("final", -1) if isinstance(row_counts, Mapping) else -1
    prediction_bytes = storage_estimate.get("prediction_logits_bytes", -1)
    finalists = storage_estimate.get("prediction_finalists", -1)
    if (
        set(storage_estimate) != expected_storage_fields
        or not isinstance(row_counts, Mapping)
        or set(row_counts) != {"train", "validation", "final"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in row_counts.values()
        )
        or row_counts.get("train") != rows
        or not integer_values_are_valid
        or storage_estimate.get("parent_import_sha256")
        != logical_bank["parents"]["parent_import"]
        or int(storage_estimate.get("logical_bytes_per_ordinary_row", -1)) != 7_815
        or int(storage_estimate.get("logical_bytes_per_toff_row", -1)) != 15_021
        or int(storage_estimate.get("ordinary_bank_bytes", -1)) != ordinary_bytes
        or int(storage_estimate.get("toff_bank_bytes", -1)) != toff_bytes
        or storage_estimate.get("committed_target_generation_bytes") != committed
        or int(interrupted) < committed
        or int(storage_estimate.get("peak_target_staging_plus_committed_bytes", -1))
        != committed + int(interrupted)
        or storage_estimate.get("prediction_identity_bytes_per_row") != 32
        or storage_estimate.get("prediction_logit_bytes_per_row") != 60
        or storage_estimate.get("prediction_payload_bytes_per_row") != 92
        or prediction_bytes != final_rows * 92 * finalists
        or storage_estimate.get("subtotal_before_filesystem_headroom_bytes")
        != subtotal
        or storage_estimate.get("filesystem_headroom_numerator") != 1
        or storage_estimate.get("filesystem_headroom_denominator") != 4
        or storage_estimate.get("filesystem_headroom_bytes") != headroom
        or storage_estimate.get("estimated_campaign_peak_durable_bytes")
        != subtotal + headroom
        or storage_estimate.get("simultaneously_committed_target_banks") != 1
        or storage_estimate.get("compression_saving_assumed") is not False
        or storage_estimate.get("operational_fixed_sizes_measured")
        != all(value > 0 for value in fixed_values)
    ):
        raise ValueError("HCWDL-RKD target storage-estimate evidence differs")
    requests = resource_profile.get("requests")
    measurements = resource_profile.get("measurements")
    if not isinstance(requests, Mapping) or not isinstance(measurements, Mapping):
        raise ValueError("HCWDL-RKD target measured-resource evidence differs")
    request = requests.get("gpu_target")
    measurement = measurements.get("gpu_target")
    if (
        not isinstance(request, Mapping)
        or not isinstance(measurement, Mapping)
        or resource_profile.get("schema_version") != 1
        or request.get("gpu") != "gpu:gh200:1"
        or _gib_request_bytes(request.get("memory")) != slurm_mem_per_node_bytes
        or isinstance(measurement.get("peak_rss_bytes"), bool)
        or not isinstance(measurement.get("peak_rss_bytes"), (int, float))
        or not math.isfinite(float(measurement["peak_rss_bytes"]))
        or float(measurement["peak_rss_bytes"]) <= 0
        or int(peak_runtime_bytes) < math.ceil(float(measurement["peak_rss_bytes"]))
        or int(peak_runtime_bytes) * 4 > int(slurm_mem_per_node_bytes) * 3
    ):
        raise ValueError("HCWDL-RKD target measured-resource evidence differs")
    required_for_kind = ordinary_bytes if kind == ORDINARY_BANK else toff_bytes
    if int(storage_estimate.get(
        "committed_target_generation_bytes", -1,
    )) < required_for_kind:
        raise ValueError("HCWDL-RKD target storage estimate omits this bank")
    return storage_hash, profile_hash


def _validate_build_intent_semantics(context: TargetGenerationContext) -> None:
    payload = context.build_intent["payload"]
    required = {
        "logical_bank_id", "bank_kind", "generation_id", "purpose", "build_owner_id",
        "partitions", "expected_rows", "expected_class_counts",
        "expected_identity_order_sha256", "expected_identity_set_sha256",
        "expected_population_rows_sha256",
        "expected_logical_bytes", "target_storage_cap_bytes",
        "container_overhead_bytes", "staging_recovery_reserve_bytes",
        "quarantine_reserve_bytes", "filesystem_headroom_bytes",
        "peak_runtime_bytes", "slurm_mem_per_node_bytes",
        "filesystem_available_bytes", "preflight_required_durable_bytes",
        "storage_estimate_sha256", "resource_profile_sha256", "array_schemas",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("HCWDL-RKD target build-intent fields differ")
    partitions = _normalize_partition_specs(payload["partitions"])
    rows = sum(record["rows"] for record in partitions.values())
    class_counts = payload["expected_class_counts"]
    if (
        payload["logical_bank_id"] != context.bank_id
        or payload["bank_kind"] != context.bank_kind
        or payload["generation_id"] != context.generation_id
        or payload["purpose"] != context.consumer_registry["payload"]["purpose"]
        or require_sha256(payload["build_owner_id"], name="target build owner ID")
        != context.build_owner_id
        or payload["partitions"] != partitions
        or isinstance(payload["expected_rows"], bool)
        or not isinstance(payload["expected_rows"], int)
        or payload["expected_rows"] != rows
        or not isinstance(class_counts, list)
        or len(class_counts) != 15
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in class_counts
        )
        or sum(class_counts) != rows
    ):
        raise ValueError("HCWDL-RKD target build-intent population differs")
    if payload["purpose"] in {"miniature", NONFINAL_ACCEPTANCE_TARGET_PURPOSE}:
        consumers = context.consumer_registry["payload"].get("consumers", ())
        if payload["purpose"] == "miniature" and len(consumers) != 1:
            raise ValueError("HCWDL-RKD miniature target registry differs")
        limits = {
            row.get("execution_identity_payload", {}).get("bounded_row_limit")
            for row in consumers if isinstance(row, Mapping)
        }
        limit = next(iter(limits)) if len(limits) == 1 else None
        if (
            isinstance(limit, bool) or not isinstance(limit, int)
            or not 1 <= limit <= 4096 or rows > limit
        ):
            raise ValueError(
                "HCWDL-RKD miniature target rows exceed the bounded row limit"
                if payload["purpose"] == "miniature"
                else "HCWDL-RKD non-final target rows exceed their row limit"
            )
    require_sha256(
        payload["expected_identity_order_sha256"],
        name="target build expected identity-order SHA-256",
    )
    require_sha256(
        payload["expected_identity_set_sha256"],
        name="target build expected identity-set SHA-256",
    )
    require_sha256(
        payload["expected_population_rows_sha256"],
        name="target build expected population-row SHA-256",
    )
    expected_bytes = rows * target_logical_bytes_per_row(context.bank_kind)
    cap = payload["target_storage_cap_bytes"]
    budget_names = (
        "container_overhead_bytes", "staging_recovery_reserve_bytes",
        "quarantine_reserve_bytes", "filesystem_headroom_bytes",
        "peak_runtime_bytes", "slurm_mem_per_node_bytes", "filesystem_available_bytes",
    )
    budgets = {name: payload[name] for name in budget_names}
    durable_required = expected_bytes + sum(
        int(budgets[name]) for name in (
            "container_overhead_bytes", "staging_recovery_reserve_bytes",
            "quarantine_reserve_bytes", "filesystem_headroom_bytes",
        )
    )
    if (
        isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in budgets.values()
        )
        or budgets["slurm_mem_per_node_bytes"] <= 0
        or budgets["filesystem_available_bytes"] <= 0
        or payload["expected_logical_bytes"] != expected_bytes
        or payload["preflight_required_durable_bytes"] != durable_required
        or durable_required > int(cap)
        or int(cap) > budgets["filesystem_available_bytes"]
        or budgets["peak_runtime_bytes"] * 4
        > budgets["slurm_mem_per_node_bytes"] * 3
    ):
        raise ValueError("HCWDL-RKD target build-intent storage budget differs")
    if context.build_intent["parents"].get("storage_estimate") != require_sha256(
        payload["storage_estimate_sha256"], name="target storage-estimate SHA-256",
    ) or context.build_intent["parents"].get("resource_profile") != require_sha256(
        payload["resource_profile_sha256"], name="target resource-profile SHA-256",
    ):
        raise ValueError("HCWDL-RKD target build-intent preflight lineage differs")
    expected_schemas = {
        name: {
            array_name: {"dtype": dtype.str, "shape": list(shape)}
            for array_name, (dtype, shape) in target_array_schema(
                context.bank_kind, record["rows"],
            ).items()
        }
        for name, record in partitions.items()
    }
    if payload["array_schemas"] != expected_schemas:
        raise ValueError("HCWDL-RKD target build-intent array schemas differ")


def begin_target_generation(
    bank_root: str | Path,
    *,
    logical_bank: Mapping[str, Any],
    consumer_registry: Mapping[str, Any],
    forward_spec: Mapping[str, Any],
    partitions: Mapping[str, Mapping[str, Any]],
    expected_class_counts: Sequence[int],
    expected_identity_order_sha256: str,
    expected_identity_set_sha256: str,
    expected_population_rows_sha256: str,
    build_owner: Mapping[str, Any],
    target_storage_cap_bytes: int,
    container_overhead_bytes: int,
    staging_recovery_reserve_bytes: int,
    quarantine_reserve_bytes: int,
    filesystem_headroom_bytes: int,
    peak_runtime_bytes: int,
    slurm_mem_per_node_bytes: int,
    filesystem_available_bytes: int,
    storage_estimate: Mapping[str, Any],
    resource_profile: Mapping[str, Any],
) -> TargetGenerationContext:
    """Preflight then durably publish an owner-scoped build intent."""

    logical_hash = validate_logical_target_bank(logical_bank)
    registry_hash = validate_target_consumer_registry(
        consumer_registry, logical_bank=logical_bank,
    )
    forward_hash = validate_target_forward_spec(forward_spec)
    _validate_forward_logical_lineage(forward_spec, logical_bank)
    payload = consumer_registry["payload"]
    partition_specs = _normalize_partition_specs(partitions)
    if list(partition_specs) != forward_spec["payload"]["source_partitions"]:
        raise ValueError("HCWDL-RKD build partitions differ from forward spec")
    if (
        len(expected_class_counts) != 15
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected_class_counts
        )
    ):
        raise ValueError("HCWDL-RKD expected class counts differ")
    class_counts = list(expected_class_counts)
    rows = sum(record["rows"] for record in partition_specs.values())
    if sum(class_counts) != rows:
        raise ValueError("HCWDL-RKD expected class counts do not conserve rows")
    if payload["purpose"] in {"miniature", NONFINAL_ACCEPTANCE_TARGET_PURPOSE}:
        consumers = payload.get("consumers", ())
        limits = {
            row.get("execution_identity_payload", {}).get("bounded_row_limit")
            for row in consumers if isinstance(row, Mapping)
        }
        limit = next(iter(limits)) if len(limits) == 1 else None
        if (
            isinstance(limit, bool) or not isinstance(limit, int)
            or not 1 <= limit <= 4096 or rows > limit
        ):
            raise ValueError(
                "HCWDL-RKD miniature target rows exceed the bounded row limit"
                if payload["purpose"] == "miniature"
                else "HCWDL-RKD non-final target rows exceed their row limit"
            )
    require_sha256(expected_identity_order_sha256, name="target expected identity-order SHA-256")
    require_sha256(expected_identity_set_sha256, name="target expected identity-set SHA-256")
    require_sha256(
        expected_population_rows_sha256,
        name="target expected population-row SHA-256",
    )
    bank_kind = logical_bank["payload"]["bank_kind"]
    required_bytes = rows * target_logical_bytes_per_row(bank_kind)
    storage_hash, profile_hash = _validate_preflight_artifacts(
        logical_bank=logical_bank,
        rows=rows,
        storage_estimate=storage_estimate,
        resource_profile=resource_profile,
        slurm_mem_per_node_bytes=slurm_mem_per_node_bytes,
        peak_runtime_bytes=peak_runtime_bytes,
    )
    resource_values = {
        "container_overhead_bytes": container_overhead_bytes,
        "staging_recovery_reserve_bytes": staging_recovery_reserve_bytes,
        "quarantine_reserve_bytes": quarantine_reserve_bytes,
        "filesystem_headroom_bytes": filesystem_headroom_bytes,
        "peak_runtime_bytes": peak_runtime_bytes,
        "slurm_mem_per_node_bytes": slurm_mem_per_node_bytes,
        "filesystem_available_bytes": filesystem_available_bytes,
    }
    durable_required = required_bytes + sum(
        resource_values[name] for name in (
            "container_overhead_bytes", "staging_recovery_reserve_bytes",
            "quarantine_reserve_bytes", "filesystem_headroom_bytes",
        )
    )
    if (
        isinstance(target_storage_cap_bytes, bool)
        or not isinstance(target_storage_cap_bytes, int)
        or target_storage_cap_bytes <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in resource_values.values()
        )
        or slurm_mem_per_node_bytes <= 0
        or isinstance(filesystem_available_bytes, bool)
        or not isinstance(filesystem_available_bytes, int)
        or filesystem_available_bytes <= 0
        or durable_required > target_storage_cap_bytes
        or target_storage_cap_bytes > filesystem_available_bytes
        or filesystem_available_bytes < int(
            storage_estimate["estimated_campaign_peak_durable_bytes"]
        )
        or staging_recovery_reserve_bytes < int(
            storage_estimate["interrupted_target_generation_reserve_bytes"]
        )
        or filesystem_headroom_bytes < int(
            storage_estimate["filesystem_headroom_bytes"]
        )
        or peak_runtime_bytes * 4 > slurm_mem_per_node_bytes * 3
    ):
        raise MemoryError("HCWDL-RKD target generation exceeds its predeclared storage cap")
    generation_id = derive_target_generation_id(
        logical_hash, registry_hash, purpose=payload["purpose"],
        generation_parent_sha256=payload["generation_parent_sha256"],
    )
    if not isinstance(build_owner, Mapping) or not build_owner:
        raise ValueError("HCWDL-RKD target build owner is empty")
    build_owner_id = canonical_sha256({
        "generation_id": generation_id, "logical_bank_sha256": logical_hash,
        "consumer_registry_sha256": registry_hash, "forward_spec_sha256": forward_hash,
        "owner": dict(build_owner),
    })
    root = Path(bank_root)
    logical_path = root / "logical_bank.json"
    # Preflight above intentionally precedes the first filesystem mutation.
    write_staged_immutable_json(logical_path, logical_bank)
    committed = root / "generations" / generation_id
    staging_parent = root / "staging" / generation_id
    staging = staging_parent / build_owner_id
    intent_parents = {
        "logical_bank": logical_hash,
        "consumer_registry": registry_hash,
        "target_forward_spec": forward_hash,
        "storage_estimate": storage_hash,
        "resource_profile": profile_hash,
    }
    intent = build_versioned_artifact(
        TARGET_BUILD_INTENT_CONTRACT, parents=intent_parents,
        payload={
            "logical_bank_id": logical_bank["payload"]["logical_bank_id"],
            "bank_kind": bank_kind,
            "generation_id": generation_id,
            "purpose": payload["purpose"],
            "build_owner_id": build_owner_id,
            "partitions": partition_specs,
            "expected_rows": rows,
            "expected_class_counts": class_counts,
            "expected_identity_order_sha256": expected_identity_order_sha256,
            "expected_identity_set_sha256": expected_identity_set_sha256,
            "expected_population_rows_sha256": expected_population_rows_sha256,
            "expected_logical_bytes": required_bytes,
            "target_storage_cap_bytes": int(target_storage_cap_bytes),
            "preflight_required_durable_bytes": durable_required,
            "storage_estimate_sha256": storage_hash,
            "resource_profile_sha256": profile_hash,
            **resource_values,
            "array_schemas": {
                name: {
                    array_name: {"dtype": dtype.str, "shape": list(shape)}
                    for array_name, (dtype, shape) in target_array_schema(
                        bank_kind, record["rows"],
                    ).items()
                }
                for name, record in partition_specs.items()
            },
        },
    )
    context = TargetGenerationContext(
        root, logical_bank, consumer_registry, forward_spec, intent,
        generation_id, build_owner_id, staging, committed,
    )
    _validate_build_intent_semantics(context)
    if committed.exists():
        validate_target_generation(root, generation_id)
        existing = _context_from_committed(root, generation_id)
        if (
            existing.logical_bank["content_hash"] != logical_hash
            or existing.consumer_registry["content_hash"] != registry_hash
            or existing.forward_spec["content_hash"] != forward_hash
            or existing.build_intent["parents"] != intent["parents"]
        ):
            raise ValueError("committed HCWDL-RKD target request lineage differs")
        requested_payload = dict(intent["payload"])
        committed_payload = dict(existing.build_intent["payload"])
        requested_payload.pop("build_owner_id", None)
        committed_payload.pop("build_owner_id", None)
        if requested_payload != committed_payload:
            raise ValueError("committed HCWDL-RKD target build intent differs")
        # A valid committed generation takes precedence over any orphan staging
        # tree and returns the exact committed context, never caller-supplied
        # metadata that happened to derive the same generation ID.
        return existing
    if staging_parent.exists():
        foreign = [path for path in staging_parent.iterdir() if path.name != build_owner_id]
        if foreign:
            raise PermissionError("another owner has staged this target generation")
    staging.mkdir(parents=True, exist_ok=True)
    write_staged_immutable_json(staging / "build_intent.json", intent)
    write_staged_immutable_json(staging / "consumer_registry.json", consumer_registry)
    write_staged_immutable_json(staging / "target_forward_spec.json", forward_spec)
    fsync_directory(staging)
    return context


def _quarantine_staged_shard(context: TargetGenerationContext, partition: str) -> None:
    shard_directory = context.staging_directory / "shards"
    paths = [shard_directory / f"{partition}.npz", shard_directory / f"{partition}.json"]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    # Corrupt owner-scoped work must never be renamed into an otherwise valid
    # committed generation.  Keep the quarantine beside staging so the sole
    # generation commit point contains only manifest-registered members.
    quarantine_root = (
        context.bank_root / "quarantine" / context.generation_id
        / context.build_owner_id
    )
    already_quarantined = sum(
        path.stat().st_size for path in quarantine_root.rglob("*") if path.is_file()
    ) if quarantine_root.exists() else 0
    incoming = sum(path.stat().st_size for path in existing if path.is_file())
    if already_quarantined + incoming > int(
        context.build_intent["payload"]["quarantine_reserve_bytes"]
    ):
        raise MemoryError("HCWDL-RKD target quarantine reserve would be exceeded")
    digest = canonical_sha256({
        path.name: sha256_file(path) if path.is_file() else "not_file" for path in existing
    })
    quarantine = quarantine_root / f"{partition}_{digest}"
    quarantine.mkdir(parents=True, exist_ok=True)
    for path in existing:
        target = quarantine / path.name
        if target.exists():
            if sha256_file(target) != sha256_file(path):
                raise FileExistsError("HCWDL-RKD staged quarantine collision differs")
            path.unlink()
        else:
            path.rename(target)
    fsync_directory(quarantine)
    fsync_directory(shard_directory)


def _shard_sidecar(
    context: TargetGenerationContext, partition: str, arrays: Mapping[str, np.ndarray],
    *, payload_sha256: str, runtime_audit: Mapping[str, Any],
) -> dict[str, Any]:
    rows = len(arrays["source_entry"])
    class_counts = np.bincount(np.asarray(arrays["label"], dtype=np.int64), minlength=15)
    return build_versioned_artifact(
        TARGET_SHARD_CONTRACT,
        parents={
            "logical_bank": context.logical_bank["content_hash"],
            "consumer_registry": context.consumer_registry["content_hash"],
            "target_forward_spec": context.forward_spec["content_hash"],
            "build_intent": context.build_intent["content_hash"],
        },
        payload={
            "generation_id": context.generation_id,
            "logical_bank_id": context.bank_id,
            "bank_kind": context.bank_kind,
            "role": "train",
            "source_partition": partition,
            "source_file_id": int(arrays["source_file_id"][0]),
            "rows": rows,
            "payload_filename": f"shards/{partition}.npz",
            "payload_byte_sha256": payload_sha256,
            "identity_order_sha256": identity_order_sha256(arrays["identity_digest"]),
            "identity_set_sha256": identity_set_sha256(arrays["identity_digest"]),
            "population_rows_sha256": target_population_rows_sha256(
                source_file_id=arrays["source_file_id"],
                source_entry=arrays["source_entry"],
                identity_digest=arrays["identity_digest"],
                label=arrays["label"],
            ),
            "first_identity_digest": bytes(arrays["identity_digest"][0]).hex(),
            "last_identity_digest": bytes(arrays["identity_digest"][-1]).hex(),
            "first_source_entry": int(arrays["source_entry"][0]),
            "last_source_entry": int(arrays["source_entry"][-1]),
            "class_counts": class_counts.astype(int).tolist(),
            "array_sha256": {
                name: logical_array_sha256(name, np.asarray(arrays[name]))
                for name in sorted(arrays)
            },
            "array_shapes": {
                name: list(np.asarray(arrays[name]).shape) for name in sorted(arrays)
            },
            "array_dtypes": {
                name: np.asarray(arrays[name]).dtype.str for name in sorted(arrays)
            },
            "family_vocabulary": list(family_vocabulary(context.bank_kind)),
            "reason_vocabulary": list(reason_vocabulary(context.bank_kind)),
            "teacher_forward_dtype": "float32",
            "target_runtime_audit": _validate_shard_runtime_audit(
                runtime_audit, partition=partition, expected_rows=rows,
                expected_source_file_id=int(arrays["source_file_id"][0]),
            ),
        },
    )


def _validate_shard_runtime_audit(
    value: Mapping[str, Any], *, partition: str, expected_rows: int,
    expected_source_file_id: int,
) -> dict[str, Any]:
    required = {
        "source_partition", "teacher_forward_calls", "batch_partition_sha256",
        "canonical_batches",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("HCWDL-RKD shard runtime-audit fields differ")
    records = value["canonical_batches"]
    if not isinstance(records, list) or not records:
        raise ValueError("HCWDL-RKD shard runtime-audit batches are empty")
    required_record = {
        "source_partition", "batch_index", "source_file_id", "rows",
        "first_source_entry", "last_source_entry", "first_identity_digest",
        "last_identity_digest", "logits", "surfaces", "sketches",
    }
    normalized = []
    previous_entry: int | None = None
    partition_payloads = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != required_record:
            raise ValueError("HCWDL-RKD canonical target-batch record differs")
        row = dict(raw)
        if (
            row["source_partition"] != partition
            or row["batch_index"] != index
            or isinstance(row["rows"], bool)
            or not 1 <= int(row["rows"]) <= TARGET_FORWARD_BATCH_SIZE
            or row["source_file_id"] != expected_source_file_id
            or isinstance(row["first_source_entry"], bool)
            or isinstance(row["last_source_entry"], bool)
            or int(row["last_source_entry"]) < int(row["first_source_entry"])
            or (
                previous_entry is not None
                and int(row["first_source_entry"]) <= previous_entry
            )
        ):
            raise ValueError("HCWDL-RKD canonical target-batch partition/order differs")
        if index < len(records) - 1 and int(row["rows"]) != TARGET_FORWARD_BATCH_SIZE:
            raise ValueError("HCWDL-RKD non-final target batch is short")
        for name in (
            "first_identity_digest", "last_identity_digest", "logits", "surfaces",
            "sketches",
        ):
            require_sha256(row[name], name=f"target runtime batch {name}")
        row["rows"] = int(row["rows"])
        row["first_source_entry"] = int(row["first_source_entry"])
        row["last_source_entry"] = int(row["last_source_entry"])
        previous_entry = row["last_source_entry"]
        normalized.append(row)
        partition_payloads.append({
            key: row[key] for key in (
                "source_partition", "batch_index", "source_file_id", "rows",
                "first_source_entry", "last_source_entry", "first_identity_digest",
                "last_identity_digest",
            )
        })
    if (
        value["source_partition"] != partition
        or isinstance(value["teacher_forward_calls"], bool)
        or int(value["teacher_forward_calls"]) != len(normalized)
        or sum(row["rows"] for row in normalized) != expected_rows
        or value["batch_partition_sha256"] != canonical_sha256(partition_payloads)
    ):
        raise ValueError("HCWDL-RKD shard runtime-audit conservation differs")
    return {
        "source_partition": partition,
        "teacher_forward_calls": len(normalized),
        "batch_partition_sha256": value["batch_partition_sha256"],
        "canonical_batches": normalized,
    }


def _load_shard_pair(
    directory: Path, sidecar: Mapping[str, Any], *, context: TargetGenerationContext | None = None,
) -> dict[str, np.ndarray]:
    validate_versioned_artifact(
        sidecar, expected_contract=TARGET_SHARD_CONTRACT,
        expected_parents=(
            None if context is None else {
                "logical_bank": context.logical_bank["content_hash"],
                "consumer_registry": context.consumer_registry["content_hash"],
                "target_forward_spec": context.forward_spec["content_hash"],
                "build_intent": context.build_intent["content_hash"],
            }
        ),
        required_payload_keys=(
            "generation_id", "bank_kind", "source_partition", "rows",
            "payload_filename", "payload_byte_sha256", "array_sha256",
            "population_rows_sha256",
        ),
    )
    payload = sidecar["payload"]
    if context is not None and (
        payload["generation_id"] != context.generation_id
        or payload["bank_kind"] != context.bank_kind
    ):
        raise ValueError("HCWDL-RKD target shard generation/kind differs")
    if "target_runtime_audit" not in payload:
        raise ValueError("HCWDL-RKD target shard lacks its one-forward runtime audit")
    _validate_shard_runtime_audit(
        payload["target_runtime_audit"], partition=payload["source_partition"],
        expected_rows=int(payload["rows"]),
        expected_source_file_id=int(payload["source_file_id"]),
    )
    path = directory / Path(payload["payload_filename"])
    if not path.is_file() or sha256_file(path) != payload["payload_byte_sha256"]:
        raise ValueError("HCWDL-RKD target shard payload is absent or corrupt")
    arrays = load_npz_arrays(path)
    validate_target_arrays(
        arrays, bank_kind=payload["bank_kind"], expected_rows=int(payload["rows"]),
        expected_source_file_id=int(payload["source_file_id"]),
    )
    for name, array in arrays.items():
        if (
            payload["array_sha256"].get(name) != logical_array_sha256(name, array)
            or payload["array_shapes"].get(name) != list(array.shape)
            or payload["array_dtypes"].get(name) != array.dtype.str
        ):
            raise ValueError(f"HCWDL-RKD target shard array lineage differs: {name}")
    if payload["population_rows_sha256"] != target_population_rows_sha256(
        source_file_id=arrays["source_file_id"],
        source_entry=arrays["source_entry"],
        identity_digest=arrays["identity_digest"],
        label=arrays["label"],
    ):
        raise ValueError("HCWDL-RKD target shard population-row lineage differs")
    if context is not None:
        rebuilt = _shard_sidecar(
            context, payload["source_partition"], arrays,
            payload_sha256=payload["payload_byte_sha256"],
            runtime_audit=payload["target_runtime_audit"],
        )
        if rebuilt["content_hash"] != sidecar["content_hash"]:
            raise ValueError("HCWDL-RKD target shard sidecar semantics differ")
    return arrays


def stage_target_shard(
    context: TargetGenerationContext,
    *, partition: str, arrays: Mapping[str, np.ndarray],
    runtime_audit: Mapping[str, Any],
    failure_hook: FailureHook | None = None,
) -> str:
    if partition not in context.build_intent["payload"]["partitions"]:
        raise ValueError("HCWDL-RKD target shard partition is unregistered")
    specification = context.build_intent["payload"]["partitions"][partition]
    normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    validate_target_arrays(
        normalized, bank_kind=context.bank_kind,
        expected_rows=int(specification["rows"]),
        expected_source_file_id=int(specification["source_file_id"]),
    )
    if context.committed_directory.exists():
        manifest = validate_target_generation(context.bank_root, context.generation_id)
        record = next(
            (
                row for row in manifest["payload"]["shards"]
                if row["source_partition"] == partition
            ),
            None,
        )
        if record is None:
            raise ValueError("committed HCWDL-RKD target lacks the requested partition")
        sidecar = load_json(context.committed_directory / record["sidecar_path"])
        committed_arrays = _load_shard_pair(
            context.committed_directory, sidecar, context=context,
        )
        if set(committed_arrays) != set(normalized) or any(
            not np.array_equal(committed_arrays[name], normalized[name], equal_nan=False)
            for name in normalized
        ):
            raise FileExistsError("committed HCWDL-RKD target shard request differs")
        requested_audit = _validate_shard_runtime_audit(
            runtime_audit,
            partition=partition,
            expected_rows=int(specification["rows"]),
            expected_source_file_id=int(specification["source_file_id"]),
        )
        if sidecar["payload"]["target_runtime_audit"] != requested_audit:
            raise FileExistsError("committed HCWDL-RKD target runtime audit differs")
        return "reused_committed"
    shard_directory = context.staging_directory / "shards"
    payload_path = shard_directory / f"{partition}.npz"
    sidecar_path = shard_directory / f"{partition}.json"
    payload_bytes = deterministic_compressed_npz_bytes(normalized)
    sidecar = _shard_sidecar(
        context, partition, normalized, payload_sha256=sha256_bytes(payload_bytes),
        runtime_audit=runtime_audit,
    )
    if payload_path.exists() or sidecar_path.exists():
        try:
            existing = load_json(sidecar_path)
            existing_arrays = _load_shard_pair(context.staging_directory, existing, context=context)
            if existing["content_hash"] != sidecar["content_hash"] or any(
                not np.array_equal(existing_arrays[name], normalized[name]) for name in normalized
            ):
                raise ValueError("staged target shard meaning differs")
            return "reused"
        except (
            FileNotFoundError, KeyError, TypeError, ValueError,
            FloatingPointError, EOFError, OSError, zipfile.BadZipFile,
        ):
            _quarantine_staged_shard(context, partition)
    write_staged_immutable_bytes(payload_path, payload_bytes)
    if failure_hook is not None:
        failure_hook(f"after_shard_payload:{partition}")
    write_staged_immutable_json(sidecar_path, sidecar)
    if failure_hook is not None:
        failure_hook(f"after_shard_sidecar:{partition}")
    return "published"


def load_staged_target_shard(
    context: TargetGenerationContext, *, partition: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate and return one same-owner staged shard for exact continuation."""

    if partition not in context.build_intent["payload"]["partitions"]:
        raise ValueError("HCWDL-RKD staged target partition is unregistered")
    sidecar = load_json(context.staging_directory / "shards" / f"{partition}.json")
    arrays = _load_shard_pair(context.staging_directory, sidecar, context=context)
    return sidecar, arrays


def _streaming_array_sha256(
    name: str, arrays: Sequence[np.ndarray], *, total_rows: int,
) -> str:
    if not arrays:
        raise ValueError("HCWDL-RKD global array hash has no chunks")
    first = np.asarray(arrays[0])
    shape = (total_rows, *first.shape[1:])
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        if value.dtype != first.dtype or value.shape[1:] != first.shape[1:]:
            raise ValueError("HCWDL-RKD global target array chunks differ")
        digest.update(value.tobytes(order="C"))
    return logical_array_sha256_from_byte_hash(
        name=name,
        dtype=first.dtype.str,
        shape=shape,
        c_order_byte_sha256=digest.hexdigest(),
        byte_length=int(first.dtype.itemsize * math.prod(shape)),
    )


def _load_all_staged_shards(
    context: TargetGenerationContext,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, np.ndarray]]]:
    """Load every registered staged pair in canonical partition order.

    Returning the sidecars and decoded arrays as two explicit collections is
    intentional.  Iterating a two-element sidecar list through tuple
    unpacking can otherwise masquerade as this API for a two-partition
    miniature and feed JSON metadata into the global array audit.
    """

    sidecars: list[Mapping[str, Any]] = []
    arrays: list[Mapping[str, np.ndarray]] = []
    for partition in context.build_intent["payload"]["partitions"]:
        path = context.staging_directory / "shards" / f"{partition}.json"
        if not path.is_file():
            raise FileNotFoundError(f"HCWDL-RKD target shard is absent: {partition}")
        sidecar = load_json(path)
        if sidecar["payload"].get("source_partition") != partition:
            raise ValueError("HCWDL-RKD target shard partition differs")
        sidecars.append(sidecar)
        arrays.append(
            _load_shard_pair(context.staging_directory, sidecar, context=context)
        )
    return sidecars, arrays


def _validate_global_coverage(
    context: TargetGenerationContext, arrays: Iterable[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    intent = context.build_intent["payload"]
    total_rows = int(intent["expected_rows"])
    schema = target_array_schema(context.bank_kind, total_rows)
    hashers: dict[str, Any] = {}
    for name, (dtype, shape) in schema.items():
        hashers[name] = hashlib.sha256()
    previous: tuple[int, int] | None = None
    identities: list[bytes] = []
    population_hasher = TargetPopulationHasher()
    class_counts = np.zeros(15, dtype=np.int64)
    teacher_family_token_counts = np.zeros(
        len(family_vocabulary(context.bank_kind)), dtype=np.uint64,
    )
    companion_reason_counts_array = np.zeros(
        len(reason_vocabulary(context.bank_kind)), dtype=np.uint64,
    )
    observed_rows = 0
    array_names = set(schema)
    for part in arrays:
        if set(part) != array_names:
            raise ValueError("HCWDL-RKD global target array names differ")
        part_rows = len(part["label"])
        observed_rows += part_rows
        source_ids = part["source_file_id"]
        entries = part["source_entry"]
        first = (int(source_ids[0]), int(entries[0]))
        last = (int(source_ids[-1]), int(entries[-1]))
        if previous is not None and first <= previous:
            raise ValueError("HCWDL-RKD source shards overlap or are globally reordered")
        previous = last
        identities.extend(bytes(row) for row in part["identity_digest"])
        population_hasher.update(
            source_file_id=part["source_file_id"],
            source_entry=part["source_entry"],
            identity_digest=part["identity_digest"],
            label=part["label"],
        )
        class_counts += np.bincount(part["label"].astype(np.int64), minlength=15)
        teacher_family_token_counts += np.sum(
            part["token_count"], axis=0, dtype=np.uint64,
        )
        companion_reason_counts_array += np.sum(
            part["family_reason_counts"], axis=0, dtype=np.uint64,
        )
        for name, value in part.items():
            array = np.ascontiguousarray(value)
            expected_dtype, expected_shape = schema[name]
            if array.dtype != expected_dtype or array.shape[1:] != expected_shape[1:]:
                raise ValueError("HCWDL-RKD global target array chunks differ")
            hashers[name].update(array.tobytes(order="C"))
    if observed_rows != total_rows:
        raise ValueError("HCWDL-RKD target generation row coverage differs")
    if len(identities) != len(set(identities)):
        raise ValueError("HCWDL-RKD target generation repeats identities across shards")
    order_hash = identity_order_sha256_iter(identity.hex() for identity in identities)
    set_hash = identity_order_sha256_iter(identity.hex() for identity in sorted(identities))
    population_hash = population_hasher.hexdigest()
    if (
        order_hash != intent["expected_identity_order_sha256"]
        or set_hash != intent["expected_identity_set_sha256"]
        or population_hash != intent["expected_population_rows_sha256"]
    ):
        raise ValueError("HCWDL-RKD target generation identity population differs")
    if class_counts.astype(int).tolist() != intent["expected_class_counts"]:
        raise ValueError("HCWDL-RKD target generation class counts differ")
    global_hashes = {
        name: logical_array_sha256_from_byte_hash(
            name=name,
            dtype=schema[name][0].str,
            shape=schema[name][1],
            c_order_byte_sha256=digest.hexdigest(),
            byte_length=int(schema[name][0].itemsize * math.prod(schema[name][1])),
        )
        for name, digest in sorted(hashers.items())
    }
    logical_target_sha256 = canonical_sha256({
        "logical_bank_sha256": context.logical_bank["content_hash"],
        "target_forward_spec_sha256": context.forward_spec["content_hash"],
        "scientific_parents": context.logical_bank["parents"],
        "array_sha256": global_hashes,
        "identity_order_sha256": order_hash,
        "identity_set_sha256": set_hash,
        "population_rows_sha256": population_hash,
    })
    teacher_family_token_counts = teacher_family_token_counts.astype(int).tolist()
    companion_reason_counts = companion_reason_counts_array.astype(int).tolist()
    if context.bank_kind == ORDINARY_BANK:
        companion_family_counts = [companion_reason_counts[0]]
        companion_unclassified_count = 0
    else:
        companion_family_counts = [
            companion_reason_counts[0] + companion_reason_counts[2],
            companion_reason_counts[1] + companion_reason_counts[3],
        ]
        companion_unclassified_count = (
            companion_reason_counts[4] + companion_reason_counts[5]
        )
    companion_total_tokens = sum(companion_reason_counts)
    if (
        sum(companion_family_counts) + companion_unclassified_count
        != companion_total_tokens
    ):
        raise ValueError("HCWDL-RKD companion-HLT family conservation differs")
    return {
        "rows": total_rows,
        "class_counts": class_counts.astype(int).tolist(),
        "identity_order_sha256": order_hash,
        "identity_set_sha256": set_hash,
        "population_rows_sha256": population_hash,
        "array_sha256": global_hashes,
        "logical_target_sha256": logical_target_sha256,
        "teacher_family_token_counts": teacher_family_token_counts,
        "teacher_total_tokens": sum(teacher_family_token_counts),
        "companion_hlt_reason_counts": companion_reason_counts,
        "companion_hlt_family_counts": companion_family_counts,
        "companion_hlt_unclassified_count": companion_unclassified_count,
        "companion_hlt_total_tokens": companion_total_tokens,
    }


def _validate_execution_facts(
    facts: Mapping[str, Any], *, context: TargetGenerationContext,
    coverage: Mapping[str, Any], sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "producer", "device", "precision", "determinism", "batch_count",
        "batch_partition_sha256", "sentinel_hashes", "construction_seconds",
    }
    if not isinstance(facts, Mapping) or set(facts) != required:
        raise ValueError("HCWDL-RKD target execution facts differ")
    spec = context.forward_spec["payload"]
    if facts["producer"] != spec["producer"] or facts["precision"] != spec["precision"]:
        raise ValueError("HCWDL-RKD target execution producer/precision differs from spec")
    actual_device = dict(facts["device"])
    gpu_uuid = actual_device.pop("gpu_uuid", None)
    if actual_device != spec["device"] or not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("HCWDL-RKD target execution device differs from spec")
    if facts["determinism"] != spec["determinism"]:
        raise ValueError("HCWDL-RKD target execution determinism differs from spec")
    construction_seconds = facts["construction_seconds"]
    if (
        isinstance(construction_seconds, bool)
        or not isinstance(construction_seconds, (int, float))
        or not math.isfinite(float(construction_seconds))
        or float(construction_seconds) < 0
    ):
        raise ValueError("HCWDL-RKD target construction timing differs")
    canonical_batches = [
        row
        for sidecar in sidecars
        for row in sidecar["payload"]["target_runtime_audit"]["canonical_batches"]
    ]
    partition_payloads = [{
        key: row[key] for key in (
            "source_partition", "batch_index", "source_file_id", "rows",
            "first_source_entry", "last_source_entry", "first_identity_digest",
            "last_identity_digest",
        )
    } for row in canonical_batches]
    expected_partition_hash = canonical_sha256(partition_payloads)
    if (
        isinstance(facts["batch_count"], bool)
        or int(facts["batch_count"]) != len(canonical_batches)
        or facts["batch_partition_sha256"] != expected_partition_hash
    ):
        raise ValueError("HCWDL-RKD target execution batch partition differs")
    sentinels = facts["sentinel_hashes"]
    if set(sentinels) != {"first", "middle", "last"}:
        raise ValueError("HCWDL-RKD target execution sentinel registry differs")
    for position, record in sentinels.items():
        if set(record) != {"logits", "surfaces", "sketches"}:
            raise ValueError(f"HCWDL-RKD {position} sentinel fields differ")
        for name, digest in record.items():
            require_sha256(digest, name=f"target {position} sentinel {name}")
    expected_sentinels = {
        "first": {
            name: canonical_batches[0][name] for name in ("logits", "surfaces", "sketches")
        },
        "middle": {
            name: canonical_batches[len(canonical_batches) // 2][name]
            for name in ("logits", "surfaces", "sketches")
        },
        "last": {
            name: canonical_batches[-1][name] for name in ("logits", "surfaces", "sketches")
        },
    }
    if sentinels != expected_sentinels:
        raise ValueError("HCWDL-RKD target execution sentinel hashes differ")
    return {
        **dict(facts),
        "device": {**actual_device, "gpu_uuid": gpu_uuid},
        "construction_seconds": float(construction_seconds),
        "batch_count": int(facts["batch_count"]),
        "output_array_sha256": dict(coverage["array_sha256"]),
        "logical_target_sha256": coverage["logical_target_sha256"],
    }


def finalize_target_generation(
    context: TargetGenerationContext,
    *, execution_facts: Mapping[str, Any],
    prior_execution_attestation: Mapping[str, Any] | None = None,
    failure_hook: FailureHook | None = None,
) -> dict[str, Any]:
    if context.committed_directory.exists():
        manifest = validate_target_generation(context.bank_root, context.generation_id)
        attestation = load_json(
            context.committed_directory / "target_execution_attestation.json",
        )
        supplied = dict(execution_facts)
        if set(supplied) != {
            "producer", "device", "precision", "determinism", "batch_count",
            "batch_partition_sha256", "sentinel_hashes", "construction_seconds",
        } or any(attestation["payload"].get(name) != value for name, value in supplied.items()):
            raise FileExistsError("committed HCWDL-RKD target execution request differs")
        if prior_execution_attestation is not None:
            validate_versioned_artifact(
                prior_execution_attestation,
                expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
                required_payload_keys=("sentinel_hashes", "logical_target_sha256"),
            )
            if (
                prior_execution_attestation["payload"]["sentinel_hashes"]
                != attestation["payload"]["sentinel_hashes"]
                or prior_execution_attestation["payload"]["logical_target_sha256"]
                != manifest["payload"]["logical_target_sha256"]
            ):
                raise FileExistsError("committed HCWDL-RKD reconstruction request differs")
        return manifest
    sidecars, arrays = _load_all_staged_shards(context)
    coverage = _validate_global_coverage(context, arrays)
    stable_execution_facts = dict(execution_facts)
    staged_attestation_path = (
        context.staging_directory / "target_execution_attestation.json"
    )
    if staged_attestation_path.is_file():
        # Wall-clock timing is operational evidence, not scientific identity.
        # Once the owner has staged a valid attestation, an idempotent retry
        # must reproduce its bytes instead of inventing a second elapsed time.
        staged_attestation = load_json(staged_attestation_path)
        validate_versioned_artifact(
            staged_attestation,
            expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
            expected_parents={
                "logical_bank": context.logical_bank["content_hash"],
                "target_forward_spec": context.forward_spec["content_hash"],
                "build_intent": context.build_intent["content_hash"],
            },
        )
        stable_execution_facts["construction_seconds"] = staged_attestation[
            "payload"
        ]["construction_seconds"]
    execution_payload = _validate_execution_facts(
        stable_execution_facts,
        context=context, coverage=coverage, sidecars=sidecars,
    )
    if prior_execution_attestation is not None:
        validate_versioned_artifact(
            prior_execution_attestation,
            expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
            required_payload_keys=("sentinel_hashes", "logical_target_sha256"),
        )
        prior = prior_execution_attestation["payload"]
        if (
            prior["sentinel_hashes"] != execution_payload["sentinel_hashes"]
            or prior["logical_target_sha256"] != coverage["logical_target_sha256"]
        ):
            raise ValueError("HCWDL-RKD reconstructed target bytes differ")
    attestation = build_versioned_artifact(
        TARGET_EXECUTION_ATTESTATION_CONTRACT,
        parents={
            "logical_bank": context.logical_bank["content_hash"],
            "target_forward_spec": context.forward_spec["content_hash"],
            "build_intent": context.build_intent["content_hash"],
        },
        payload=execution_payload,
    )
    generation = build_versioned_artifact(
        TARGET_GENERATION_CONTRACT,
        parents={
            "logical_bank": context.logical_bank["content_hash"],
            "consumer_registry": context.consumer_registry["content_hash"],
            "target_forward_spec": context.forward_spec["content_hash"],
            "target_execution_attestation": attestation["content_hash"],
            "build_intent": context.build_intent["content_hash"],
        },
        payload={
            "generation_id": context.generation_id,
            "build_owner_id": context.build_owner_id,
            "logical_bank_id": context.bank_id,
            "purpose": context.consumer_registry["payload"]["purpose"],
            "logical_target_sha256": coverage["logical_target_sha256"],
            "population_rows_sha256": coverage["population_rows_sha256"],
            "teacher_family_token_counts": coverage["teacher_family_token_counts"],
            "teacher_total_tokens": coverage["teacher_total_tokens"],
            "companion_hlt_reason_counts": coverage["companion_hlt_reason_counts"],
            "companion_hlt_family_counts": coverage["companion_hlt_family_counts"],
            "companion_hlt_unclassified_count": coverage[
                "companion_hlt_unclassified_count"
            ],
            "companion_hlt_total_tokens": coverage["companion_hlt_total_tokens"],
            "storage_mode": "transient_durable_sketch",
            "durable_particle_dataset_published": False,
        },
    )
    shard_records = []
    for sidecar in sidecars:
        record = sidecar["payload"]
        shard_records.append({
            "source_partition": record["source_partition"],
            "payload_path": record["payload_filename"],
            "payload_byte_sha256": record["payload_byte_sha256"],
            "payload_bytes": (
                context.staging_directory / record["payload_filename"]
            ).stat().st_size,
            "sidecar_path": f"shards/{record['source_partition']}.json",
            "sidecar_byte_sha256": sha256_file(
                context.staging_directory / "shards" / f"{record['source_partition']}.json"
            ),
            "sidecar_content_hash": sidecar["content_hash"],
            "rows": record["rows"],
        })
    manifest = build_versioned_artifact(
        TARGET_MANIFEST_CONTRACT,
        parents={
            "logical_bank": context.logical_bank["content_hash"],
            "consumer_registry": context.consumer_registry["content_hash"],
            "target_forward_spec": context.forward_spec["content_hash"],
            "target_execution_attestation": attestation["content_hash"],
            "target_generation": generation["content_hash"],
            "build_intent": context.build_intent["content_hash"],
        },
        payload={
            "generation_id": context.generation_id,
            "logical_bank_id": context.bank_id,
            "bank_kind": context.bank_kind,
            "purpose": context.consumer_registry["payload"]["purpose"],
            "authorized_consumers": context.consumer_registry["payload"]["consumers"],
            "rows": coverage["rows"],
            "class_counts": coverage["class_counts"],
            "identity_order_sha256": coverage["identity_order_sha256"],
            "identity_set_sha256": coverage["identity_set_sha256"],
            "population_rows_sha256": coverage["population_rows_sha256"],
            "array_sha256": coverage["array_sha256"],
            "logical_target_sha256": coverage["logical_target_sha256"],
            "teacher_family_token_counts": coverage["teacher_family_token_counts"],
            "teacher_total_tokens": coverage["teacher_total_tokens"],
            "companion_hlt_reason_counts": coverage["companion_hlt_reason_counts"],
            "companion_hlt_family_counts": coverage["companion_hlt_family_counts"],
            "companion_hlt_unclassified_count": coverage[
                "companion_hlt_unclassified_count"
            ],
            "companion_hlt_total_tokens": coverage["companion_hlt_total_tokens"],
            "family_vocabulary": list(family_vocabulary(context.bank_kind)),
            "reason_vocabulary": list(reason_vocabulary(context.bank_kind)),
            "teacher_forward_dtype": "float32",
            "storage_mode": "transient_durable_sketch",
            "durable_particle_dataset_published": False,
            "shards": shard_records,
        },
    )
    write_staged_immutable_json(
        context.staging_directory / "target_execution_attestation.json", attestation,
    )
    if failure_hook is not None:
        failure_hook("after_execution_attestation")
    write_staged_immutable_json(context.staging_directory / "generation.json", generation)
    write_staged_immutable_json(context.staging_directory / "manifest.json", manifest)
    if failure_hook is not None:
        failure_hook("after_manifest")

    def validator(path: Path) -> dict[str, Any]:
        del path
        return validate_target_generation(context.bank_root, context.generation_id)

    commit_staged_directory(
        context.staging_directory, context.committed_directory,
        validate_committed=validator, failure_hook=failure_hook,
    )
    return validator(context.committed_directory)


def _context_from_committed(bank_root: Path, generation_id: str) -> TargetGenerationContext:
    directory = bank_root / "generations" / generation_id
    logical = load_json(bank_root / "logical_bank.json")
    registry = load_json(directory / "consumer_registry.json")
    forward = load_json(directory / "target_forward_spec.json")
    intent = load_json(directory / "build_intent.json")
    return TargetGenerationContext(
        bank_root, logical, registry, forward, intent, generation_id,
        str(intent["payload"]["build_owner_id"]),
        bank_root / "staging" / generation_id / str(intent["payload"]["build_owner_id"]),
        directory,
    )


def validate_target_generation(
    bank_root: str | Path, generation_id: str,
    *, expected_logical_target_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(bank_root)
    identity = require_sha256(generation_id, name="target generation ID")
    directory = root / "generations" / identity
    if not directory.is_dir():
        raise FileNotFoundError("HCWDL-RKD target generation is absent")
    context = _context_from_committed(root, identity)
    logical_hash = validate_logical_target_bank(context.logical_bank)
    registry_hash = validate_target_consumer_registry(
        context.consumer_registry, logical_bank=context.logical_bank,
    )
    forward_hash = validate_target_forward_spec(context.forward_spec)
    _validate_forward_logical_lineage(context.forward_spec, context.logical_bank)
    intent = context.build_intent
    validate_versioned_artifact(
        intent, expected_contract=TARGET_BUILD_INTENT_CONTRACT,
        expected_parents={
            "logical_bank": logical_hash, "consumer_registry": registry_hash,
            "target_forward_spec": forward_hash,
            "storage_estimate": intent["payload"]["storage_estimate_sha256"],
            "resource_profile": intent["payload"]["resource_profile_sha256"],
        },
    )
    _validate_build_intent_semantics(context)
    if intent["payload"]["generation_id"] != identity:
        raise ValueError("HCWDL-RKD build intent generation differs")
    expected_generation_id = derive_target_generation_id(
        logical_hash, registry_hash,
        purpose=context.consumer_registry["payload"]["purpose"],
        generation_parent_sha256=context.consumer_registry["payload"]["generation_parent_sha256"],
    )
    if identity != expected_generation_id:
        raise ValueError("HCWDL-RKD target generation identity differs")
    attestation = load_json(directory / "target_execution_attestation.json")
    validate_versioned_artifact(
        attestation, expected_contract=TARGET_EXECUTION_ATTESTATION_CONTRACT,
        expected_parents={
            "logical_bank": logical_hash, "target_forward_spec": forward_hash,
            "build_intent": intent["content_hash"],
        },
    )
    generation = load_json(directory / "generation.json")
    validate_versioned_artifact(
        generation, expected_contract=TARGET_GENERATION_CONTRACT,
        expected_parents={
            "logical_bank": logical_hash, "consumer_registry": registry_hash,
            "target_forward_spec": forward_hash,
            "target_execution_attestation": attestation["content_hash"],
            "build_intent": intent["content_hash"],
        },
    )
    manifest = load_json(directory / "manifest.json")
    validate_versioned_artifact(
        manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
        expected_parents={
            "logical_bank": logical_hash, "consumer_registry": registry_hash,
            "target_forward_spec": forward_hash,
            "target_execution_attestation": attestation["content_hash"],
            "target_generation": generation["content_hash"],
            "build_intent": intent["content_hash"],
        },
    )
    if manifest["payload"]["generation_id"] != identity:
        raise ValueError("HCWDL-RKD target manifest generation differs")
    generation_payload = generation["payload"]
    manifest_payload = manifest["payload"]
    if (
        generation_payload["generation_id"] != identity
        or generation_payload["build_owner_id"] != intent["payload"]["build_owner_id"]
        or generation_payload["logical_bank_id"] != context.bank_id
        or generation_payload["purpose"] != context.consumer_registry["payload"]["purpose"]
        or generation_payload["logical_target_sha256"]
        != manifest_payload["logical_target_sha256"]
        or generation_payload["population_rows_sha256"]
        != manifest_payload["population_rows_sha256"]
        or manifest_payload["population_rows_sha256"]
        != intent["payload"]["expected_population_rows_sha256"]
        or generation_payload["storage_mode"] != "transient_durable_sketch"
        or generation_payload["durable_particle_dataset_published"] is not False
        or manifest_payload["logical_bank_id"] != context.bank_id
        or manifest_payload["bank_kind"] != context.bank_kind
        or manifest_payload["purpose"] != context.consumer_registry["payload"]["purpose"]
        or manifest_payload["authorized_consumers"]
        != context.consumer_registry["payload"]["consumers"]
        or manifest_payload["family_vocabulary"]
        != list(family_vocabulary(context.bank_kind))
        or manifest_payload["reason_vocabulary"]
        != list(reason_vocabulary(context.bank_kind))
        or manifest_payload["teacher_forward_dtype"] != "float32"
        or manifest_payload["storage_mode"] != "transient_durable_sketch"
        or manifest_payload["durable_particle_dataset_published"] is not False
    ):
        raise ValueError("HCWDL-RKD target generation/manifest semantics differ")
    if expected_logical_target_sha256 is not None and manifest["payload"][
        "logical_target_sha256"
    ] != require_sha256(expected_logical_target_sha256, name="expected logical target SHA-256"):
        raise ValueError("HCWDL-RKD target logical hash differs")
    sidecars = []
    registered = manifest["payload"]["shards"]
    if [row["source_partition"] for row in registered] != list(
        intent["payload"]["partitions"]
    ):
        raise ValueError("HCWDL-RKD target manifest shard registry differs")
    for record in registered:
        sidecar_path = directory / record["sidecar_path"]
        if not sidecar_path.is_file() or sha256_file(sidecar_path) != record["sidecar_byte_sha256"]:
            raise ValueError("HCWDL-RKD target shard sidecar is absent or corrupt")
        sidecar = load_json(sidecar_path)
        if sidecar["content_hash"] != record["sidecar_content_hash"]:
            raise ValueError("HCWDL-RKD target shard sidecar hash differs")
        sidecar_payload = sidecar["payload"]
        expected_record = {
            "source_partition": sidecar_payload["source_partition"],
            "payload_path": sidecar_payload["payload_filename"],
            "payload_byte_sha256": sidecar_payload["payload_byte_sha256"],
            "payload_bytes": (directory / sidecar_payload["payload_filename"]).stat().st_size,
            "sidecar_path": f"shards/{sidecar_payload['source_partition']}.json",
            "sidecar_byte_sha256": sha256_file(sidecar_path),
            "sidecar_content_hash": sidecar["content_hash"],
            "rows": sidecar_payload["rows"],
        }
        if record != expected_record:
            raise ValueError("HCWDL-RKD target manifest shard record differs")
        sidecars.append(sidecar)
    expected_files = {
        "build_intent.json", "consumer_registry.json", "target_forward_spec.json",
        "target_execution_attestation.json", "generation.json", "manifest.json",
        *(
            path
            for record in registered
            for path in (record["payload_path"], record["sidecar_path"])
        ),
    }
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("HCWDL-RKD target generation contains unregistered members")
    coverage = _validate_global_coverage(
        context,
        (_load_shard_pair(directory, sidecar, context=context) for sidecar in sidecars),
    )
    for key in (
        "rows", "class_counts", "identity_order_sha256", "identity_set_sha256",
        "array_sha256", "logical_target_sha256", "teacher_family_token_counts",
        "teacher_total_tokens", "companion_hlt_reason_counts",
        "companion_hlt_family_counts", "companion_hlt_unclassified_count",
        "companion_hlt_total_tokens",
    ):
        if manifest["payload"][key] != coverage[key]:
            raise ValueError(f"HCWDL-RKD target manifest coverage differs: {key}")
    if attestation["payload"]["logical_target_sha256"] != coverage["logical_target_sha256"]:
        raise ValueError("HCWDL-RKD target attestation logical hash differs")
    execution_fact_keys = {
        "producer", "device", "precision", "determinism", "batch_count",
        "batch_partition_sha256", "sentinel_hashes", "construction_seconds",
    }
    rebuilt_attestation = _validate_execution_facts(
        {key: attestation["payload"][key] for key in execution_fact_keys},
        context=context, coverage=coverage, sidecars=sidecars,
    )
    if rebuilt_attestation != attestation["payload"]:
        raise ValueError("HCWDL-RKD target execution attestation semantics differ")
    return manifest


@dataclass(frozen=True)
class RepresentationTargetBank:
    manifest: Mapping[str, Any]
    execution_attestation: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(
        cls, bank_root: str | Path, generation_id: str,
        *, strategy: str = "RREL",
        expected_logical_target_sha256: str | None = None,
    ) -> "RepresentationTargetBank":
        if strategy not in {"RSET", "RREL", "JET_ONLY"}:
            raise ValueError("unknown HCWDL-RKD target materialization strategy")
        root = Path(bank_root)
        manifest = validate_target_generation(
            root, generation_id,
            expected_logical_target_sha256=expected_logical_target_sha256,
        )
        directory = root / "generations" / generation_id
        execution_attestation = load_json(
            directory / "target_execution_attestation.json"
        )
        schema = target_array_schema(
            manifest["payload"]["bank_kind"], int(manifest["payload"]["rows"]),
        )
        identity_and_jet = {
            "source_file_id", "source_entry", "identity_digest", "label", "logits",
            "jet_penultimate",
        }
        if strategy == "RSET":
            selected = identity_and_jet | {"token_family_eligibility"} | {
                name for name in schema if name.startswith("token_kernel_mean")
            }
        elif strategy == "JET_ONLY":
            # The compact M5 control must not instantiate token metadata or a
            # set/relation target in RAM (plan Section 23.2).
            selected = identity_and_jet
        else:
            selected = set(schema)
        arrays = {
            name: np.empty(shape, dtype=dtype)
            for name, (dtype, shape) in schema.items() if name in selected
        }
        cursor = 0
        for record in manifest["payload"]["shards"]:
            sidecar = load_json(directory / record["sidecar_path"])
            part = _load_shard_pair(directory, sidecar)
            stop = cursor + int(record["rows"])
            for name in arrays:
                arrays[name][cursor:stop] = part[name]
            cursor = stop
            del part
        if cursor != int(manifest["payload"]["rows"]):
            raise ValueError("HCWDL-RKD RAM target materialization row count differs")
        identities = [bytes(row) for row in arrays["identity_digest"]]
        lookup = {identity: index for index, identity in enumerate(identities)}
        if len(lookup) != len(identities):
            raise ValueError("HCWDL-RKD RAM target bank repeats identities")
        return cls(manifest, execution_attestation, arrays, lookup)

    def join(self, identity_digests: np.ndarray) -> dict[str, np.ndarray]:
        requested = np.asarray(identity_digests)
        if requested.dtype != np.uint8 or requested.ndim != 2 or requested.shape[1] != 32:
            raise ValueError("HCWDL-RKD target join identities must be uint8 [rows,32]")
        keys = [bytes(row) for row in requested]
        if len(keys) != len(set(keys)):
            raise ValueError("HCWDL-RKD target join request repeats an identity")
        try:
            indexes = np.asarray([self._lookup[key] for key in keys], dtype=np.int64)
        except KeyError as error:
            raise KeyError("HCWDL-RKD target identity join is incomplete") from error
        return {
            name: np.ascontiguousarray(value[indexes])
            for name, value in self.arrays.items()
            if name not in {"source_file_id", "source_entry", "identity_digest", "label"}
        }


__all__ = [
    "BANK_KINDS", "KERNEL_RESOURCE_NAMES", "LOGICAL_BANK_IDS", "ORDINARY_BANK", "ORDINARY_FAMILIES",
    "ORDINARY_REASONS", "RELATION_KERNEL_DIM", "RELATION_STRATA",
    "NONFINAL_ACCEPTANCE_TARGET_CONSUMER_CONTRACT",
    "NONFINAL_ACCEPTANCE_TARGET_PURPOSE", "NONFINAL_ACCEPTANCE_TARGET_ROWS",
    "NONFINAL_ACCEPTANCE_TARGET_TRAJECTORIES",
    "RepresentationTargetBank", "TARGET_FORWARD_BATCH_SIZE", "TOKEN_KERNEL_DIM",
    "TOFF_BANK", "TOFF_FAMILIES", "TOFF_REASONS", "TargetGenerationContext",
    "bank_kind_for_id", "begin_target_generation", "build_logical_target_bank",
    "build_miniature_target_consumer_row",
    "build_nonfinal_acceptance_target_consumer_registry",
    "build_nonfinal_acceptance_target_consumer_row",
    "build_target_consumer_registry", "build_target_consumer_row",
    "build_target_forward_spec",
    "derive_target_generation_id", "expected_screen_consumer_nodes",
    "deterministic_compressed_npz_bytes",
    "family_vocabulary", "finalize_target_generation", "identity_order_sha256",
    "identity_set_sha256", "reason_vocabulary", "stage_target_shard",
    "load_staged_target_shard", "TargetPopulationHasher",
    "target_population_rows_sha256",
    "target_array_schema", "target_core_values_per_row", "target_logical_bytes_per_row",
    "target_metadata_bytes_per_row", "validate_logical_target_bank",
    "validate_target_arrays", "validate_target_consumer_registry",
    "validate_target_forward_spec", "validate_target_generation",
]
