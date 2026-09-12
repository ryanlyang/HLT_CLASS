"""Bounded paired-jet reads, with an HLT-only capability and sealed final test."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import awkward as ak
import numpy as np
import uproot

from .contracts import row_identity
from .inventory import validate_inventory, verify_file
from .schema import PARTICLE_FIELDS, SCALARS
from .selection import selected_mask
from .splits import validate_splits, is_subset_profile, file_population


@dataclass(frozen=True)
class Particles:
    """Unpadded raw [N,14]: four-vector, charge, five PID flags, d0/error/dz/error.

    Index/order is construction metadata, never a model feature. There are no
    invented quality flags, zero-error significances, or silent invalid-row cuts.
    """
    values: np.ndarray

    def __post_init__(self):
        value = np.array(self.values, dtype=np.float32, copy=True)
        if value.ndim != 2 or value.shape[1] != 14 or len(value) == 0:
            raise ValueError("Particles must be nonempty [N,14]")
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite required particle field")
        if np.any(value[:, 3] <= 0) or np.any(np.hypot(value[:, 0], value[:, 1]) <= 0):
            raise ValueError("Particle energy and transverse momentum must be positive")
        flags = value[:, 5:10]
        if not np.all((flags == 0) | (flags == 1)) or not np.all(flags.sum(axis=1) == 1):
            raise ValueError("Particle PID must be exclusive one-hot")
        if not np.isin(value[:, 4], (-1, 0, 1)).all():
            raise ValueError("Particle charge is outside {-1,0,1}")
        charged = (flags[:, 0] + flags[:, 3] + flags[:, 4]) == 1
        if np.any(charged != (value[:, 4] != 0)):
            raise ValueError("Particle PID/charge applicability differs")
        if np.any(value[:, [11, 13]] < 0):
            raise ValueError("Negative impact-parameter uncertainty")
        value.setflags(write=False)
        object.__setattr__(self, "values", value)

    @property
    def uncertainty_valid(self) -> np.ndarray:
        return self.values[:, [11, 13]] > 0

    @property
    def p4(self) -> np.ndarray:
        return self.values[:, :4]

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Jet:
    identity: str
    label: int
    hlt: Particles
    offline: Particles | None


def _particles(arrays, row: int, prefix: str, expected: int) -> Particles:
    columns = [np.asarray(ak.to_numpy(arrays[prefix + f][row])) for f in PARTICLE_FIELDS]
    if any(c.ndim != 1 or len(c) != expected for c in columns):
        raise ValueError(f"Jagged length/count mismatch: {prefix} row {row}")
    return Particles(np.column_stack(columns))


class DatasetReader:
    def __init__(self, data_root: Path, inventory: dict, splits: dict, *, role: str,
                 include_offline: bool = False, step_size: int = 2048,
                 file_paths: tuple[str, ...] | None = None):
        # No public boolean unlock for test access. A separate future locked
        # finalist executor will supply that capability, not ordinary workers.
        if role not in {"train", "validation"}:
            raise PermissionError("Ordinary Delphes readers cannot access final_test")
        if step_size < 1:
            raise ValueError("Reader step_size must be positive")
        self.inventory_hash = validate_inventory(inventory)
        validate_splits(splits, inventory)
        self.root = Path(data_root).resolve(strict=True)
        self.inventory, self.splits = inventory, splits
        self.role, self.include_offline, self.step_size = role, include_offline, step_size
        allowed = {g["path"] for g in splits["groups"] if g["role"] == role}
        if file_paths is not None and (len(set(file_paths)) != len(file_paths) or not set(file_paths) <= allowed):
            raise PermissionError("Requested file subset escapes the declared role")
        self.file_paths = allowed if file_paths is None else set(file_paths)
        self.expected_rows = sum(file_population(splits, r, role)[0] for r in inventory["files"]
                                 if r["path"] in self.file_paths)
        self.memberships = ({r["path"]: r for r in splits["memberships"][role]["files"]}
                            if is_subset_profile(splits) else None)

    def __iter__(self) -> Iterator[Jet]:
        total = 0
        for file, group in zip(self.inventory["files"], self.splits["groups"]):
            if file["path"] not in self.file_paths:
                continue
            expected_count, expected_classes = file_population(self.splits, file, self.role)
            if expected_count == 0:
                continue
            path = verify_file(self.root, file, self.inventory)
            entries = None
            if self.memberships is not None:
                from .split_registry import unpack_entries
                entries = unpack_entries(self.memberships[file["path"]]["entry_mask"], file["entries"])
            # Even branch requests in deployable/HLT-only paths omit offline
            # arrays. Metadata for selection remains outside model_inputs.
            branches = ["jet_label", "hlt_matched", "hlt_jet_nparticles"]
            branches += ["hlt_part_" + f for f in PARTICLE_FIELDS]
            if self.include_offline:
                branches += ["jet_nparticles"] + ["part_" + f for f in PARTICLE_FIELDS]
            file_count = 0
            class_counts = np.zeros(len(expected_classes), np.int64)
            with uproot.open(path) as handle:
                tree = handle[file["tree_key"]]
                for start in range(0, tree.num_entries, self.step_size):
                    if entries is not None:
                        selected = entries[np.searchsorted(entries, start):np.searchsorted(entries, start + self.step_size)]
                        if len(selected) == 0:
                            continue
                    arrays = tree.arrays(branches, entry_start=start,
                                         entry_stop=start + self.step_size, library="ak", how=dict)
                    keep, labels = selected_mask(
                        ak.to_numpy(arrays["jet_label"]), ak.to_numpy(arrays["hlt_matched"]),
                        file["source"], self.inventory["selection"],
                    )
                    if entries is not None:
                        requested = selected - start
                        if not keep[requested].all():
                            raise ValueError("Split membership contains an ineligible/unmatched row")
                        mask = np.zeros(len(keep), bool)
                        mask[requested] = True
                        keep &= mask
                    class_counts += np.bincount(labels[keep], minlength=len(expected_classes))
                    # Validate only retained rows: unmatched HLT dummy arrays
                    # are deliberately not interpreted as physical particles.
                    for index in np.flatnonzero(keep):
                        hlt = _particles(arrays, int(index), "hlt_part_", int(arrays["hlt_jet_nparticles"][index]))
                        offline = None
                        if self.include_offline:
                            offline = _particles(arrays, int(index), "part_", int(arrays["jet_nparticles"][index]))
                        yield Jet(row_identity(self.inventory_hash, file["path"], file["tree_key"], start + int(index)),
                                  int(labels[index]), hlt, offline)
                        total += 1
                        file_count += 1
            if file_count != expected_count or class_counts.tolist() != expected_classes:
                raise ValueError("Reader/file selected counts differ")
            verify_file(self.root, file, self.inventory)
        if total != self.expected_rows:
            raise ValueError("Reader/role selected counts differ")
