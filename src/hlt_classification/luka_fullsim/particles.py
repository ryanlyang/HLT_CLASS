"""Selected paired rows in stored units; no implicit physical-unit convention."""
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from hlt_classification.data.cache_contracts import sha256_file
from .contracts import row_identity, source_path
from .splits import validate_splits

FIELDS = ("px", "py", "pz", "energy", "charge", "isChargedHadron",
          "isNeutralHadron", "isPhoton", "isElectron", "isMuon",
          "d0val", "dzval", "d0err", "dzerr", "deta", "dphi")


@dataclass(frozen=True)
class RawJet:
    identity: str
    label: int
    file_index: int
    entry: int
    hlt: np.ndarray
    offline: np.ndarray | None


def _matrix(arrays, prefix, row, count):
    columns = [np.asarray(ak.to_numpy(arrays[prefix + field][row])) for field in FIELDS]
    if count <= 0 or any(c.ndim != 1 or len(c) != count for c in columns):
        raise ValueError(f"Selected particle/count mismatch: {prefix} row {row}")
    return np.column_stack(columns).astype(np.float64)


def _ranges(entries, size):
    """Only contiguous selected entries; never decode unused/test rows logically."""
    start = previous = entries[0]
    for entry in entries[1:]:
        if entry != previous + 1 or entry - start >= size:
            yield start, previous + 1
            start = entry
        previous = entry
    yield start, previous + 1


class ParticleReader:
    def __init__(self, container, inventory, splits, *, role, include_offline=False, chunk=512):
        if role not in ("train", "validation"):
            raise PermissionError("Final-test particle access is sealed")
        if type(include_offline) is not bool or type(chunk) is not int or not 1 <= chunk <= 2048:
            raise ValueError("Invalid particle reader capability/chunk")
        validate_splits(splits, inventory)
        self.raw = Path(container).resolve(strict=True) / "raw"
        self.inventory, self.splits, self.role = inventory, splits, role
        self.include_offline, self.chunk = include_offline, chunk

    def __iter__(self):
        total = 0
        for member in self.splits["memberships"][self.role]:
            index = member["file_index"]
            record = self.inventory["files"][index]
            path, _ = source_path(self.raw, record["path"])
            def authenticate():
                if sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Particle source checksum differs: {path}")
            authenticate()
            entries = {entry: label for label, es in enumerate(member["entries_by_class"])
                       for entry in es}
            if not entries:
                continue
            branches = ["hlt_jet_nparticles"] + ["hlt_part_" + f for f in FIELDS]
            if self.include_offline:
                branches += ["jet_nparticles"] + ["part_" + f for f in FIELDS]
            with uproot.open(path) as handle:
                tree = handle[record["tree_key"]]
                for start, stop in _ranges(sorted(entries), self.chunk):
                    arrays = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak", how=dict)
                    for offset, entry in enumerate(range(start, stop)):
                        hlt = _matrix(arrays, "hlt_part_", offset, int(arrays["hlt_jet_nparticles"][offset]))
                        offline = None
                        if self.include_offline:
                            offline = _matrix(arrays, "part_", offset, int(arrays["jet_nparticles"][offset]))
                        yield RawJet(row_identity(record, entry), entries[entry], index, entry, hlt, offline)
                        total += 1
            authenticate()
        if total != self.splits["selected_counts"][self.role]:
            raise ValueError("Selected particle coverage differs")


def identity_categories(raw):
    flags, charge = raw[:, 5:10], raw[:, 4]
    if not np.isin(flags, (0, 1)).all() or not np.isin(charge, (-1, 0, 1)).all():
        raise ValueError("Invalid charge/PID flags")
    sums = flags.sum(axis=1)
    if np.any(sums > 1):
        raise ValueError("Ambiguous multiple PID flags")
    category = np.where(sums == 1, flags.argmax(axis=1), 5)
    if np.any((category != 5) & ((charge != 0) != np.isin(category, (0, 3, 4)))):
        raise ValueError("Inconsistent charge/PID applicability")
    return category


def physical(raw, conventions, side):
    """Only callable with a validated explicit convention (no inferred scales)."""
    from hlt_classification.cms2jc2_response.bridge import Particles
    from .stage2 import validate_conventions
    validate_conventions(conventions)
    if side not in ("hlt", "offline"):
        raise ValueError("Unknown physical side")
    if raw.ndim != 2 or raw.shape[1] != len(FIELDS) or not np.isfinite(raw).all():
        raise ValueError("Nonfinite required particle value")
    if np.any(raw[:, 12:14] < 0):
        raise ValueError("Negative source error: resolve convention before preparation")
    category = identity_categories(raw)
    tracking = raw[:, 10:14].copy()
    valid = np.broadcast_to(raw[:, 4:5] != 0, tracking.shape).copy()
    valid[:, 2:] &= tracking[:, 2:] > 0
    if conventions["zero_error"] == "value_and_error_unavailable":
        valid[:, :2] &= valid[:, 2:]
    tracking[~valid] = 0.
    tracking *= {"mm": 1., "cm": 10.}[conventions["length_units"][side]]
    return Particles(raw[:, :4], raw[:, 4], category, tracking, valid,
                     tuple(str(i) for i in range(len(raw))))
