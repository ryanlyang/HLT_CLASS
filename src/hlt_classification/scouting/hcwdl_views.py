"""Process-local HCWDL view and teacher-target banks.

The bank deliberately has no matcher dependency.  It consumes authenticated
particle views built from the durable dense assignment store exactly once and
then replays them from RAM for every training pass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, require_sha256, with_content_hash

from .inputs import NativeOfflineInputs, ParticleInputs
from .targets import EphemeralTeacherTargets


VIEW_BANK_CONTRACT: Final = "HCWDL_EPHEMERAL_VIEW_BANK/v1"
TARGET_BANK_CONTRACT: Final = "HCWDL_EPHEMERAL_TARGET_BANK/v1"


def _copy_particle(view: ParticleInputs) -> ParticleInputs:
    return ParticleInputs(
        np.ascontiguousarray(view.features, dtype=np.float32),
        np.ascontiguousarray(view.vectors, dtype=np.float32),
        np.ascontiguousarray(view.mask, dtype=np.bool_),
        np.ascontiguousarray(view.raw_lengths, dtype=np.int32),
    )


def _take_particle(view: ParticleInputs, indexes: np.ndarray) -> ParticleInputs:
    return ParticleInputs(
        view.features[indexes], view.vectors[indexes], view.mask[indexes],
        view.raw_lengths[indexes],
    )


def _copy_view(view: ParticleInputs | NativeOfflineInputs) -> ParticleInputs | NativeOfflineInputs:
    if isinstance(view, ParticleInputs):
        return _copy_particle(view)
    if isinstance(view, NativeOfflineInputs):
        return NativeOfflineInputs(_copy_particle(view.charged), _copy_particle(view.neutral))
    raise TypeError("HCWDL view bank accepts canonical particle inputs only")


def _take_view(
    view: ParticleInputs | NativeOfflineInputs, indexes: np.ndarray,
) -> ParticleInputs | NativeOfflineInputs:
    if isinstance(view, ParticleInputs):
        return _take_particle(view, indexes)
    return NativeOfflineInputs(_take_particle(view.charged, indexes), _take_particle(view.neutral, indexes))


def _rows(view: ParticleInputs | NativeOfflineInputs) -> int:
    return len(view.raw_lengths) if isinstance(view, ParticleInputs) else len(view.charged.raw_lengths)


def _concatenate(views: Sequence[ParticleInputs | NativeOfflineInputs]):
    def particle(items: Sequence[ParticleInputs]) -> ParticleInputs:
        return ParticleInputs(*(
            np.concatenate([getattr(item, field) for item in items], axis=0)
            for field in ("features", "vectors", "mask", "raw_lengths")
        ))
    if all(isinstance(view, ParticleInputs) for view in views):
        return particle(views)  # type: ignore[arg-type]
    if all(isinstance(view, NativeOfflineInputs) for view in views):
        return NativeOfflineInputs(
            particle([view.charged for view in views]),  # type: ignore[union-attr]
            particle([view.neutral for view in views]),  # type: ignore[union-attr]
        )
    raise ValueError("HCWDL domain changed particle-view type")


@dataclass(frozen=True)
class DomainRows:
    identities: tuple[str, ...]
    labels: np.ndarray
    view: ParticleInputs | NativeOfflineInputs


class EphemeralHcwdlViewBank:
    """Identity-aligned RAM views, each constructed by its producer once."""

    def __init__(
        self, *, role: str, domains: Mapping[str, DomainRows],
        assignment_manifest_sha256: str, split_manifest_sha256: str,
        build_counts: Mapping[str, int],
    ) -> None:
        if role not in {"train", "validation"} or not domains:
            raise ValueError("HCWDL view bank role/domains differ")
        names = tuple(sorted(domains))
        reference = domains[names[0]]
        if not reference.identities or len(reference.identities) != len(set(reference.identities)):
            raise ValueError("HCWDL view-bank identities are empty or duplicated")
        for name, rows in domains.items():
            if rows.identities != reference.identities or not np.array_equal(rows.labels, reference.labels):
                raise ValueError(f"HCWDL domain {name} is not identity/label aligned")
            if len(rows.labels) != _rows(rows.view):
                raise ValueError(f"HCWDL domain {name} row count differs")
            if int(build_counts.get(name, 0)) != 1:
                raise ValueError(f"HCWDL domain {name} was not constructed exactly once")
        self.role = role
        self.domains = dict(domains)
        self.identities = reference.identities
        self.labels = np.asarray(reference.labels)
        self.header = with_content_hash({
            "contract": VIEW_BANK_CONTRACT, "schema_version": 1,
            "storage_mode": "process_local_ram_only_v1", "role": role,
            "rows": len(self.identities), "domains": names,
            "identity_sha256": canonical_sha256(list(self.identities)),
            "assignment_manifest_sha256": require_sha256(
                assignment_manifest_sha256, name="assignment manifest SHA-256",
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split manifest SHA-256",
            ),
            "build_counts": dict(sorted(build_counts.items())),
            "durable_repaired_dataset": False,
            "matcher_callable_present": False,
        })

    @classmethod
    def build(
        cls, *, role: str, domain_builders: Mapping[str, Callable[[], Iterable[Mapping[str, Any]]]],
        assignment_manifest_sha256: str, split_manifest_sha256: str,
    ) -> "EphemeralHcwdlViewBank":
        domains: dict[str, DomainRows] = {}
        counts: dict[str, int] = {}
        for name, builder in domain_builders.items():
            counts[name] = counts.get(name, 0) + 1
            identities: list[str] = []
            labels: list[np.ndarray] = []
            pieces: list[ParticleInputs | NativeOfflineInputs] = []
            for batch in builder():
                view = batch.get("view")
                if not isinstance(view, (ParticleInputs, NativeOfflineInputs)):
                    raise ValueError(f"HCWDL domain {name} lacks canonical particle inputs")
                keys = tuple(map(str, batch.get("identity_keys", ())))
                values = np.asarray(batch.get("labels"))
                if len(keys) == 0 or len(keys) != len(values) or len(keys) != _rows(view):
                    raise ValueError(f"HCWDL domain {name} emitted an invalid block")
                identities.extend(keys); labels.append(values.copy())
                pieces.append(_copy_view(view))
            if not labels:
                raise ValueError(f"HCWDL domain {name} emitted no rows")
            domains[name] = DomainRows(
                tuple(identities), np.concatenate(labels),
                _concatenate(pieces),
            )
        return cls(
            role=role, domains=domains, assignment_manifest_sha256=assignment_manifest_sha256,
            split_manifest_sha256=split_manifest_sha256, build_counts=counts,
        )

    def batches(
        self, domain: str, *, batch_size: int, epoch: int,
        shuffle: bool, seed: int,
    ) -> Iterable[dict[str, object]]:
        if domain not in self.domains or batch_size <= 0:
            raise ValueError("HCWDL replay domain/batch size differs")
        indexes = np.arange(len(self.identities), dtype=np.int64)
        if shuffle:
            indexes = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(indexes)
        rows = self.domains[domain]
        for start in range(0, len(indexes), batch_size):
            selected = indexes[start:start + batch_size]
            yield {
                "identity_keys": np.asarray(rows.identities, dtype=object)[selected],
                "labels": rows.labels[selected], "view": _take_view(rows.view, selected),
            }


class EphemeralHcwdlTargetBank:
    """Build each teacher/domain FP32 logit table once, then identity-join it."""

    def __init__(self, *, split_manifest_sha256: str) -> None:
        self.split_manifest_sha256 = require_sha256(
            split_manifest_sha256, name="split manifest SHA-256",
        )
        self._targets: dict[tuple[str, str], EphemeralTeacherTargets] = {}
        self._build_counts: dict[tuple[str, str], int] = {}

    def build_once(
        self, *, teacher_id: str, domain: str, teacher_report_sha256: str,
        batches: Iterable[Mapping[str, object]], forward: Callable[[ParticleInputs], np.ndarray],
    ) -> EphemeralTeacherTargets:
        key = (teacher_id, domain)
        if key in self._targets:
            raise RuntimeError("HCWDL teacher/domain targets were requested for reconstruction")
        identities: list[str] = []
        logits: list[np.ndarray] = []
        for batch in batches:
            view = batch.get("view")
            if not isinstance(view, ParticleInputs):
                raise ValueError("HCWDL teacher target block lacks a particle view")
            keys = tuple(map(str, batch.get("identity_keys", ())))
            output = np.asarray(forward(view), dtype=np.float32)
            if output.shape != (len(keys), 15) or not np.isfinite(output).all():
                raise ValueError("HCWDL teacher forward did not emit finite FP32 [rows,15] logits")
            identities.extend(keys); logits.append(output)
        if not logits:
            raise ValueError("HCWDL teacher target source is empty")
        target = EphemeralTeacherTargets.create(
            identities, np.concatenate(logits), teacher_report_sha256=teacher_report_sha256,
            split_manifest_sha256=self.split_manifest_sha256,
        )
        self._targets[key] = target
        self._build_counts[key] = 1
        return target

    def get(self, teacher_id: str, domain: str) -> EphemeralTeacherTargets:
        return self._targets[(teacher_id, domain)]

    @property
    def header(self) -> dict[str, Any]:
        return with_content_hash({
            "contract": TARGET_BANK_CONTRACT, "schema_version": 1,
            "storage_mode": "process_local_ram_only_fp32_v1",
            "split_manifest_sha256": self.split_manifest_sha256,
            "targets": [list(key) for key in sorted(self._targets)],
            "build_counts": {"::".join(key): value for key, value in sorted(self._build_counts.items())},
            "durable_logits": False,
        })


__all__ = [
    "DomainRows", "EphemeralHcwdlTargetBank", "EphemeralHcwdlViewBank",
    "TARGET_BANK_CONTRACT", "VIEW_BANK_CONTRACT",
]
