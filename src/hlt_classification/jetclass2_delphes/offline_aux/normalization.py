"""Train-only calibration from the persisted FP32 target representation."""
import numpy as np
from .contracts import artifact, validate
from .targets import SCALARS, validate_targets


def fit(values, pair_valid, hlt_pt, *, train_bank_sha256, role):
    if role != "TRAIN":
        raise PermissionError("Normalizer can only fit TRAIN")
    validate_targets(values, pair_valid)
    hlt_pt = np.asarray(hlt_pt, np.float64)
    if hlt_pt.shape != (len(values),) or not np.isfinite(hlt_pt).all() or np.any(hlt_pt < 0):
        raise ValueError("HLT pT calibration population differs")
    # Explicit two-pass population moments in canonical row order. Input has
    # already crossed the durable FP32 target boundary.
    x = values[:, SCALARS].astype(np.float64)
    mean = x.mean(axis=0, dtype=np.float64)
    std = np.sqrt(np.mean((x - mean)**2, axis=0, dtype=np.float64))
    edges = np.quantile(hlt_pt, [.2, .4, .6, .8], method="linear")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Duplicate TRAIN HLT pT diagnostic edges")
    constants = values.astype(np.float64).mean(axis=0)
    if pair_valid.any():
        constants[20:28] = values[pair_valid, 20:28].astype(np.float64).mean(axis=0)
    return artifact("NORMALIZATION", train_bank_sha256=train_bank_sha256, role="TRAIN",
                    rows=len(values), mean=mean.tolist(), std=std.tolist(),
                    scale=np.maximum(std, 1e-3).tolist(), floor=(std < 1e-3).tolist(),
                    algorithm="canonical_fp32_promoted_fp64_two_pass_population_ddof0",
                    hlt_pt_edges=edges.tolist(), constant_targets=constants.tolist(),
                    constant_pair_valid=bool(pair_valid.any()))


def standardize(values, normalizer):
    validate(normalizer, "NORMALIZATION")
    if normalizer["role"] != "TRAIN":
        raise PermissionError("Not a training-only normalizer")
    return ((values[..., SCALARS] - np.array(normalizer["mean"], np.float32)) /
            np.array(normalizer["scale"], np.float32)).astype(np.float32)
