"""Read-only temperature views over the authenticated LOGIT_D000E bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .hcwdl_mhpe_tri60_probability import Tri60ProbabilityTargets
from .hcwdl_tri60_m1_screen_graph import TEACHER_ID


def _soften(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature not in {1.0, 2.0}:
        raise ValueError("TRI60 M1 screen temperature differs")
    source = np.ascontiguousarray(probabilities, dtype=np.float64)
    if temperature == 1.0:
        return np.ascontiguousarray(source, dtype=np.float32)
    # log(p) is an exact logit representative of a categorical distribution;
    # softmax(log(p)/T) therefore applies temperature to the authenticated
    # ensemble teacher without reconstructing or persisting component logits.
    powered = np.power(source, 1.0 / temperature)
    result = powered / powered.sum(axis=1, keepdims=True, dtype=np.float64)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if (
        not np.isfinite(result).all() or np.any(result < 0)
        or not np.allclose(result.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6)
    ):
        raise FloatingPointError("TRI60 M1 screen softened targets differ")
    return result


@dataclass(frozen=True)
class ScreenProbabilityTargets:
    identities: np.ndarray
    probabilities: np.ndarray
    manifest: Mapping[str, object]
    temperature: float
    _lookup: Mapping[bytes, int]

    @classmethod
    def load(
        cls, manifest_path: str | Path, *, temperature: float,
    ) -> "ScreenProbabilityTargets":
        source = Tri60ProbabilityTargets.load(
            manifest_path, distribution_id=TEACHER_ID,
        )
        if source.temperature != 1.0:
            raise ValueError("TRI60 M1 screen source teacher is not T1")
        return cls(
            identities=source.identities,
            probabilities=_soften(source.probabilities, temperature),
            manifest=source.manifest, temperature=temperature,
            _lookup=source._lookup,
        )

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = np.ascontiguousarray(identity_digests)
        if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1] != 32:
            raise ValueError("TRI60 M1 screen join identities differ")
        try:
            indexes = np.asarray(
                [self._lookup[bytes(row)] for row in identities], dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError("TRI60 M1 screen probability join is incomplete") from error
        return np.ascontiguousarray(self.probabilities[indexes])


__all__ = ["ScreenProbabilityTargets"]
