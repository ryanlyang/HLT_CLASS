"""Content-authenticated RAM-only teacher logits/representations joined by identity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from hlt_classification.data.cache_contracts import array_sha256, require_sha256, validate_content_hash, with_content_hash

EPHEMERAL_TEACHER_TARGET_CONTRACT = "hlt_classification_pmard_ephemeral_teacher_targets_v1"


def validate_ram_root(
    value: str | Path, *, project_root: str | Path, data_root: str | Path,
    allowed_roots: Sequence[str | Path],
) -> Path:
    path = Path(value).resolve(); project = Path(project_root).resolve(); data = Path(data_root).resolve()
    for forbidden in (project, data):
        try: path.relative_to(forbidden)
        except ValueError: pass
        else: raise ValueError("ephemeral RAM root lies inside project or immutable data root")
    permitted = False
    for allowed in allowed_roots:
        root = Path(allowed).resolve()
        try: path.relative_to(root); permitted = True
        except ValueError: pass
    if not permitted: raise ValueError("ephemeral path is outside declared RAM-backed roots")
    return path


@dataclass(frozen=True)
class EphemeralTeacherTargets:
    identities: tuple[str, ...]
    logits: np.ndarray
    header: Mapping[str, object]
    _lookup: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        value = np.asarray(self.logits)
        if (value.dtype not in (np.float16, np.float32)
                or value.shape != (len(self.identities), 15)
                or len(set(self.identities)) != len(self.identities)
                or not np.isfinite(value).all()):
            raise ValueError("teacher target identities/logits are invalid")
        if (self.header.get("storage_mode") != "ram_ephemeral"
                or self.header.get("identity_sha256") != hashlib.sha256(
                    "\n".join(self.identities).encode()
                ).hexdigest()
                or self.header.get("logits_sha256") != array_sha256("logits", value)):
            raise ValueError("teacher target header differs from RAM content")
        validate_content_hash(
            self.header, expected_contract=EPHEMERAL_TEACHER_TARGET_CONTRACT,
            expected_schema_version=1,
        )
        object.__setattr__(self, "_lookup", {
            key: index for index, key in enumerate(self.identities)
        })

    @classmethod
    def create(
        cls, identities: Sequence[str], logits: np.ndarray, *,
        teacher_report_sha256: str, split_manifest_sha256: str,
    ) -> "EphemeralTeacherTargets":
        keys = tuple(map(str, identities)); value = np.ascontiguousarray(logits)
        if value.dtype not in (np.float16, np.float32) or value.shape != (len(keys), 15):
            raise ValueError("teacher logits must be float16/float32 [rows,15]")
        if len(set(keys)) != len(keys) or not np.isfinite(value).all():
            raise ValueError("teacher target identities/logits are invalid")
        header = with_content_hash({
            "contract": EPHEMERAL_TEACHER_TARGET_CONTRACT, "schema_version": 1,
            "storage_mode": "ram_ephemeral", "rows": len(keys), "dtype": value.dtype.str,
            "identity_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "logits_sha256": array_sha256("logits", value),
            "teacher_report_sha256": require_sha256(teacher_report_sha256, name="teacher_report_sha256"),
            "split_manifest_sha256": require_sha256(split_manifest_sha256, name="split_manifest_sha256"),
            "authoritative_forward_dtype": "float32",
        })
        return cls(keys, value, header)

    def join(self, requested_identities: Sequence[str]) -> np.ndarray:
        try: indexes = [self._lookup[str(key)] for key in requested_identities]
        except KeyError as error: raise KeyError("teacher target identity join is incomplete") from error
        result = self.logits[indexes].astype(np.float32, copy=False)
        if not np.isfinite(result).all(): raise FloatingPointError("joined teacher logits are nonfinite")
        return result


__all__ = ["EPHEMERAL_TEACHER_TARGET_CONTRACT", "EphemeralTeacherTargets", "validate_ram_root"]
