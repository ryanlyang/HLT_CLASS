"""RAM-only paired adjacent views and exact direct-view morph scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256
from .hcwdl_homotopy import HomotopyCoordinate
from .hcwdl_representation_data import HCWDLTaggedParticleInputs


def morph_context_for_pass(pass_number: int) -> tuple[str, HomotopyCoordinate]:
    """Return the exact context coordinate for one-indexed morph pass."""

    if not 1 <= int(pass_number) <= 100:
        raise ValueError("direct-view morph pass differs")
    if pass_number == 1:
        return "U100", HomotopyCoordinate(1, 1, 0, 1)
    numerator = min(50, pass_number - 1)
    strength = 100 - 2 * numerator
    return f"D{strength:03d}", HomotopyCoordinate(1, 1, numerator, 50)


def _tagged_pair(
    context: Mapping[str, Any], primary: Mapping[str, Any], *,
    require_identical_views: bool = False,
) -> dict[str, Any]:
    if (
        not np.array_equal(context["identity_digests"], primary["identity_digests"])
        or not np.array_equal(context["labels"], primary["labels"])
    ):
        raise ValueError("adjacent paired views lack exact identity/label equality")
    left_key = "hlt" if "hlt" in context else "privileged"
    right_key = "hlt" if "hlt" in primary else "privileged"
    left = context[left_key]; right = primary[right_key]
    if type(left) is not type(right):
        raise TypeError("adjacent paired particle-view types differ")
    if require_identical_views and any(
        not np.array_equal(getattr(left, name), getattr(right, name))
        for name in (
            "features", "vectors", "mask", "raw_lengths", "visible_indices",
            "family_codes", "family_reason_codes",
        )
    ):
        raise ValueError("low-low fusion branches are not identity-equal")
    arrays = {}
    for name in (
        "features", "vectors", "mask", "visible_indices", "family_codes",
        "family_reason_codes",
    ):
        arrays[name] = np.ascontiguousarray(np.concatenate(
            (getattr(left, name), getattr(right, name)), axis=-1,
        ))
    rows, left_width = left.mask.shape[0], left.mask.shape[-1]
    right_width = right.mask.shape[-1]
    sources = np.full((rows, left_width + right_width), -1, dtype=np.int8)
    sources[:, :left_width] = np.where(left.mask[:, 0], np.int8(0), np.int8(-1))
    sources[:, left_width:] = np.where(right.mask[:, 0], np.int8(1), np.int8(-1))
    tagged = HCWDLTaggedParticleInputs(
        arrays["features"], arrays["vectors"], arrays["mask"],
        np.ascontiguousarray(left.raw_lengths + right.raw_lengths, dtype=np.int32),
        arrays["visible_indices"], arrays["family_codes"],
        arrays["family_reason_codes"], sources,
    )
    return {
        "hlt": tagged, "labels": np.ascontiguousarray(primary["labels"]),
        "identity_digests": np.ascontiguousarray(primary["identity_digests"]),
    }


class PairedViewCache:
    """Zip two exact RAM caches under one tagged training surface."""

    def __init__(self, context, primary, *, role: str, lineage: Mapping[str, Any]):
        if (
            context.header["rows"] != primary.header["rows"]
            or context.header["identity_order_sha256"]
            != primary.header["identity_order_sha256"]
            or not np.array_equal(context.identity_digests, primary.identity_digests)
        ):
            raise ValueError("adjacent paired cache identity order differs")
        self.context = context; self.primary = primary; self.role = role
        self.require_identical_views = (
            lineage.get("context") is not None
            and lineage.get("context") == lineage.get("primary")
        )
        self.identity_digests = primary.identity_digests
        self.header = {
            "contract": "HCWDL_ADJACENT_PAIRED_RAM_CACHE/v1",
            "schema_version": 1, "role": role,
            "rows": int(primary.header["rows"]),
            "array_bytes": int(context.header["array_bytes"]) + int(primary.header["array_bytes"]),
            "identity_order_sha256": primary.header["identity_order_sha256"],
            "identity_set_sha256": primary.header["identity_set_sha256"],
            "storage_mode": "process_local_ram_two_view_v1",
            "durable_artifact_published": False, "lineage": dict(lineage),
        }
        self.header["content_hash"] = canonical_sha256(self.header)

    def _zip(self, left, right):
        sentinel = object()
        from itertools import zip_longest
        for context, primary in zip_longest(left, right, fillvalue=sentinel):
            if context is sentinel or primary is sentinel:
                raise RuntimeError("adjacent paired cache batch counts differ")
            yield _tagged_pair(
                context, primary,
                require_identical_views=self.require_identical_views,
            )

    def iterate_batches(self, **kwargs):
        return self._zip(
            self.context.iterate_batches(**kwargs),
            self.primary.iterate_batches(**kwargs),
        )

    def iterate_canonical_batches(self, *, batch_size: int):
        return self._zip(
            self.context.iterate_canonical_batches(batch_size=batch_size),
            self.primary.iterate_canonical_batches(batch_size=batch_size),
        )

    def iterate_identity_digest_batches(self, identities: Sequence[str], *, batch_size: int):
        return self._zip(
            self.context.iterate_identity_digest_batches(identities, batch_size=batch_size),
            self.primary.iterate_identity_digest_batches(identities, batch_size=batch_size),
        )


@dataclass(frozen=True)
class _CoordinateNode:
    coordinate_name: str
    coordinate: HomotopyCoordinate
    seed_alias: str


class MorphPairManager:
    """Hold one context coordinate in RAM at a time for the morph control."""

    def __init__(
        self, *, primary_caches: Mapping[str, Any],
        build_coordinate: Callable[[Any], Mapping[str, Any]], seed_alias: str,
    ) -> None:
        self.primary_caches = dict(primary_caches)
        self.build_coordinate = build_coordinate; self.seed_alias = seed_alias
        self.active_pass: int | None = None; self.active_coordinate: str | None = None
        self.context_caches: dict[str, Any] = {}

    def ensure(self, pass_number: int) -> None:
        if self.active_pass == pass_number:
            return
        name, coordinate = morph_context_for_pass(pass_number)
        if self.active_coordinate == name:
            self.active_pass = pass_number
            return
        self.context_caches.clear()
        self.context_caches = dict(self.build_coordinate(_CoordinateNode(
            name, coordinate, self.seed_alias,
        )))
        self.active_pass = pass_number; self.active_coordinate = name

    def clear(self) -> None:
        self.context_caches.clear(); self.primary_caches.clear()
        self.active_pass = None; self.active_coordinate = None


class MorphPairCache:
    def __init__(self, manager: MorphPairManager, *, role: str):
        self.manager = manager; self.role = role
        primary = manager.primary_caches[role]
        self.identity_digests = primary.identity_digests
        self.header = {
            **dict(primary.header),
            "contract": "HCWDL_ADJACENT_MORPH_RAM_CACHE/v1",
            "storage_mode": "process_local_ram_one_context_coordinate_at_a_time_v1",
            "morph_denominator": 50, "durable_artifact_published": False,
        }
        self.header["content_hash"] = canonical_sha256(self.header)

    def _pair(self) -> PairedViewCache:
        if self.manager.active_pass is None:
            raise RuntimeError("morph context pass is not active")
        return PairedViewCache(
            self.manager.context_caches[self.role], self.manager.primary_caches[self.role],
            role=self.role, lineage={"morph_pass": self.manager.active_pass},
        )

    def iterate_batches(self, *, epoch: int, **kwargs):
        if self.role == "train":
            self.manager.ensure(epoch + 1)
        elif self.manager.active_pass is None:
            self.manager.ensure(epoch + 1)
        return self._pair().iterate_batches(epoch=epoch, **kwargs)

    def iterate_identity_digest_batches(self, identities: Sequence[str], *, batch_size: int):
        return self._pair().iterate_identity_digest_batches(identities, batch_size=batch_size)


__all__ = [
    "MorphPairCache", "MorphPairManager", "PairedViewCache", "morph_context_for_pass",
]
