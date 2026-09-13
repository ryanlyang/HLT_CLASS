"""Versioned semantics for JC2_OFFLINE_AUX_500K (not a logit-KD campaign)."""
from __future__ import annotations

import hashlib
import math
import re
import stat
from pathlib import Path

from hlt_classification.data.cache_contracts import (
    with_content_hash, validate_content_hash, require_sha256, sha256_file,
    load_json, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes,
    load_npz_arrays,
)
from ..contracts import relative_file

PREFIX = "JETCLASS2_DELPHES_OFFLINE_AUX_"
ARMS = ("CE", "COMP", "STRUCT", "BOTH")
LAMBDAS = {"L010": [1, 10], "L030": [3, 10], "L100": [1, 1]}
REGISTRY = "72e8b76555e3707e90aa7ea46d521c93a2b801e46ee94c35ecee978e324d7d01"
OUTER_VAL = "8f04f54797f7cd181812815eb409b90cb1cf7e503c322061e4718d23f06060b5"
OUTER_TEST = "995f747b09a47c2e7cbdc996ba552542562ef0e074b3cb7a9de2b87f7438106e"
ROLES = ("TRAIN", "VAL_SELECT", "VAL_REPORT")
ROWS = dict(TRAIN=500_000, VAL_SELECT=200_000, VAL_REPORT=800_000)
MAX_FILE = 64 * 2**20
MAX_TOTAL = 4 * 2**30
# Same naming convention as cache_contracts.atomic_publish_bytes/mkstemp.
_PUBLICATION_TEMP = re.compile(r"\.(?P<destination>.+)\.[a-z0-9_]{8}\.tmp")


def artifact(kind, **fields):
    return with_content_hash(dict(contract=PREFIX + kind + "/v1", schema_version=1,
                                  final_test_accessed=False, **fields))


def validate(value, kind):
    digest = validate_content_hash(value, expected_contract=PREFIX + kind + "/v1")
    if value.get("final_test_accessed") is not False:
        raise PermissionError("Auxiliary study has no final-test capability")
    return digest


def seed(replicate: str, domain: str) -> int:
    if replicate not in ("DISCOVERY", "CONFIRM_01", "CONFIRM_02", "CONFIRM_03"):
        raise ValueError("Unregistered replicate")
    return int.from_bytes(hashlib.sha256(
        f"JC2/offline-aux/v1/{replicate}/{domain}".encode()).digest()[:4], "big")


def seed_register():
    domains = ["shared_init", "sampler", "train_rng"] + ["aux_init/"+h for h in
               ("counts", "fractions", "scalars", "radial", "pair")]
    domains += [f"train_rng/pass/{p}" for p in range(1, 101)]
    register = {rep: {d: seed(rep, d) for d in domains}
                for rep in ("DISCOVERY", "CONFIRM_01", "CONFIRM_02", "CONFIRM_03")}
    values = [v for row in register.values() for v in row.values()]
    if len(set(values)) != len(values):
        raise ValueError("Independent seed domains collided")
    return register


def recipe():
    return artifact("RECIPE", batch_size=256, minimum_passes=60, maximum_passes=100,
                    warmup=3, hold=45, decay_end=60, peak_lr=3e-4, floor_lr=1.5e-5,
                    patience=15, patience_delta=5e-5, precision="bf16_forward_fp32_loss",
                    optimizer="AdamW", betas=[.9, .999], eps=1e-8, weight_decay=.01,
                    foreach=False, fused=False, amsgrad=False, maximize=False,
                    capturable=False, differentiable=False, accumulation=1,
                    rolling_resume=False, loss="CE_plus_registered_aux_no_KD",
                    validation_role="VAL_SELECT", selection="AUC,-CE,logR50,-update")


def learning_rate(update: int, updates_per_pass: int):
    if not 1 <= update <= 100 * updates_per_pass or updates_per_pass < 1:
        raise ValueError("Update outside registered schedule")
    x = update / updates_per_pass
    if x <= 3:
        return 3e-4 * x / 3
    if x <= 45:
        return 3e-4
    if x <= 60:
        return 1.5e-5 + (3e-4 - 1.5e-5) * (1 + math.cos(math.pi * (x - 45) / 15)) / 2
    return 1.5e-5


def selection_key(metrics, update):
    log_r = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
    return (metrics["macro_ovr_auc"], -metrics["cross_entropy"],
            -math.inf if log_r is None else log_r, -update)


def checked_payload(root, descriptor):
    path = relative_file(Path(root), descriptor["path"])
    if (path.stat().st_size != descriptor["bytes"] or path.stat().st_size > MAX_FILE
            or sha256_file(path) != descriptor["sha256"]):
        raise ValueError("Payload checksum/size differs")
    return path


def publish_arrays(root, name, arrays):
    raw = deterministic_npz_bytes(arrays)
    if len(raw) > MAX_FILE:
        raise ValueError("Generated payload exceeds 64 MiB")
    path = relative_file(Path(root), name)
    atomic_publish_bytes(path, raw)
    return dict(path=name, sha256=sha256_file(path), bytes=len(raw))


def storage_audit(root):
    """Observe bounded outputs without racing sibling atomic publications.

    This is a live inventory, not a transactional filesystem quota. Count live
    temporary files too; tolerate disappearance only for publisher temporaries
    and the stage submitter's transient claim. Durable output validation remains
    receipt- and checksum-based, separately from this storage observation.
    """
    files = {}
    for path in Path(root).rglob("*"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if (path.name == "submission_in_progress.claim"
                    and path.parent.parent.name == "attempts"
                    and path.parent.parent.parent.parent.name == "stages"):
                # The submitter removes this zero-byte claim in finally; a
                # worker may finish before all sibling sbatch calls return.
                continue
            temporary = _PUBLICATION_TEMP.fullmatch(path.name)
            if temporary is None:
                raise
            # Publication links the finished payload and unlinks its temporary.
            # The final name may not have appeared in this directory listing.
            path = path.with_name(temporary["destination"])
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                # The writer can also remove its temp after an aborted write.
                continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Symlinks forbidden in study outputs")
        if stat.S_ISREG(metadata.st_mode):
            # Use the same metadata for type and size (no is_file/stat race).
            # A recovered final name also enumerated normally is counted once.
            files[path] = metadata.st_size
    sizes = list(files.values())
    if max(sizes, default=0) > MAX_FILE or sum(sizes) > MAX_TOTAL:
        raise ValueError("Auxiliary output storage envelope exceeded")
    return dict(files=len(files), bytes=sum(sizes), largest_bytes=max(sizes, default=0))
