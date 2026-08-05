"""Authenticated, source-file-disjoint PMARD split construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Final, Iterable, Mapping, Sequence
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from hlt_classification.data.cache_contracts import require_sha256, validate_content_hash, with_content_hash
from .identity import normalize_source_path, reject_case_aliases

SCOUTING_SPLIT_CONTRACT: Final = "hlt_classification_scouting_file_split_v2"
SCOUTING_SPLIT_VERSION: Final = 2
SPLIT_SEED: Final = 12345
SPLIT_FRACTIONS: Final = (0.6, 0.2, 0.2)
SPLIT_ROLES: Final = ("train", "validation", "final_test")
MAX_CLASS_FRACTION_DEVIATION: Final = 0.02
SPLIT_ALGORITHM: Final = "file_disjoint_multiclass_minimax_milp_v2"


@dataclass(frozen=True, order=True)
class SourceFileRecord:
    path: str
    stratum: str
    raw_entries: int
    sha256: str
    mapped_entries: int
    class_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_source_path(self.path))
        if not self.stratum or "/" in self.stratum or "\\" in self.stratum:
            raise ValueError("stratum must be a nonempty sample-directory name")
        if isinstance(self.raw_entries, bool) or self.raw_entries < 0:
            raise ValueError("raw_entries must be nonnegative")
        require_sha256(self.sha256, name="source_file_sha256")
        counts = tuple(int(value) for value in self.class_counts)
        object.__setattr__(self, "class_counts", counts)
        if len(counts) != 15 or any(value < 0 for value in counts):
            raise ValueError("source file class counts must contain 15 nonnegative values")
        if isinstance(self.mapped_entries, bool) or self.mapped_entries < 0 or sum(counts) != self.mapped_entries:
            raise ValueError("source file mapped count differs from its class counts")


def largest_remainder_allocation(
    count: int, fractions: Sequence[float] = SPLIT_FRACTIONS
) -> tuple[int, int, int]:
    if count < 0 or len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("split allocation requires a nonnegative count and 3 positive fractions")
    if abs(sum(fractions) - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to one")
    raw = [count * float(value) for value in fractions]
    result = [int(value) for value in raw]
    remaining = count - sum(result)
    order = sorted(range(3), key=lambda i: (-(raw[i] - result[i]), i))
    for index in order[:remaining]:
        result[index] += 1
    return tuple(result)  # type: ignore[return-value]


def build_split_manifest(
    records: Iterable[SourceFileRecord], *, source_manifest_sha256: str,
    seed: int = SPLIT_SEED, fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> dict[str, object]:
    source_hash = require_sha256(source_manifest_sha256, name="source_manifest_sha256")
    materialized = sorted(records)
    if not materialized:
        raise ValueError("cannot split an empty source inventory")
    paths = [item.path for item in materialized]
    reject_case_aliases(paths)
    if len(paths) != len(set(paths)):
        raise ValueError("source inventory contains duplicate paths")
    if seed != SPLIT_SEED:
        raise ValueError(f"canonical PMARD split seed must be {SPLIT_SEED}")
    fractions = tuple(map(float, fractions))
    if fractions != SPLIT_FRACTIONS:
        raise ValueError("canonical PMARD split fractions must be 60/20/20")
    file_allocation = largest_remainder_allocation(len(materialized), fractions)
    if any(count == 0 for count in file_allocation):
        raise ValueError("file-disjoint split requires at least one file in every role")
    total_counts = np.asarray([item.class_counts for item in materialized], np.float64).sum(axis=0)
    if np.any(total_counts <= 0):
        raise ValueError("class-balanced split requires every class in the authenticated inventory")
    file_counts = np.asarray([item.class_counts for item in materialized], np.float64)
    assignments, optimum = _minimax_assignments(
        file_counts, total_counts, fractions=fractions,
        file_allocation=file_allocation, seed=seed,
    )
    roles: dict[str, list[SourceFileRecord]] = {
        role: [materialized[index] for index in np.flatnonzero(assignments == role_index)]
        for role_index, role in enumerate(SPLIT_ROLES)
    }
    role_payload: dict[str, object] = {}
    for role in SPLIT_ROLES:
        rows = sorted(roles[role])
        role_payload[role] = {
            "file_count": len(rows),
            "raw_entries": sum(item.raw_entries for item in rows),
            "mapped_entries": sum(item.mapped_entries for item in rows),
            "class_counts": np.asarray([item.class_counts for item in rows], np.int64).sum(axis=0).tolist(),
            "files": [asdict(item) for item in rows],
        }
    achieved = np.asarray([role_payload[role]["class_counts"] for role in SPLIT_ROLES], np.float64)
    achieved_fractions = achieved / total_counts[None, :]
    deviations = achieved_fractions - np.asarray(fractions)[:, None]
    return with_content_hash({
        "contract": SCOUTING_SPLIT_CONTRACT,
        "schema_version": SCOUTING_SPLIT_VERSION,
        "seed": seed,
        "fractions": list(map(float, fractions)),
        "algorithm": SPLIT_ALGORITHM,
        "minimax_solver_optimum": optimum,
        "maximum_allowed_class_fraction_deviation": MAX_CLASS_FRACTION_DEVIATION,
        "maximum_observed_class_fraction_deviation": float(np.max(np.abs(deviations))),
        "class_fraction_by_role": achieved_fractions.tolist(),
        "class_fraction_deviation_by_role": deviations.tolist(),
        "source_manifest_sha256": source_hash,
        "roles": role_payload,
    })


def _minimax_assignments(
    file_counts: np.ndarray, total_counts: np.ndarray, *,
    fractions: tuple[float, float, float], file_allocation: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, float]:
    """Globally minimize the worst of the 45 class-fraction deviations."""
    n_files, n_classes = file_counts.shape
    n_x = len(SPLIT_ROLES) * n_files
    t_index = n_x
    d_start = t_index + 1
    n_d = len(SPLIT_ROLES) * n_classes
    n_variables = d_start + n_d
    rows: list[np.ndarray] = []
    lowers: list[float] = []
    uppers: list[float] = []

    def add(row: np.ndarray, lower: float, upper: float) -> None:
        rows.append(row); lowers.append(lower); uppers.append(upper)

    for file_index in range(n_files):
        row = np.zeros(n_variables)
        for role_index in range(len(SPLIT_ROLES)):
            row[role_index * n_files + file_index] = 1
        add(row, 1, 1)
    for role_index, count in enumerate(file_allocation):
        row = np.zeros(n_variables)
        row[role_index * n_files:(role_index + 1) * n_files] = 1
        add(row, count, count)
    for role_index, fraction in enumerate(fractions):
        for class_index in range(n_classes):
            d_index = d_start + role_index * n_classes + class_index
            normalized = file_counts[:, class_index] / total_counts[class_index]
            row = np.zeros(n_variables)
            row[role_index * n_files:(role_index + 1) * n_files] = normalized
            row[d_index] = -1
            add(row, -np.inf, fraction)
            row = -row
            row[d_index] = -1
            add(row, -np.inf, -fraction)
            row = np.zeros(n_variables); row[d_index] = 1; row[t_index] = -1
            add(row, -np.inf, 0)

    matrix = np.stack(rows)
    constraints = [LinearConstraint(matrix, np.asarray(lowers), np.asarray(uppers))]
    lower_bounds = np.zeros(n_variables)
    upper_bounds = np.full(n_variables, np.inf); upper_bounds[:n_x] = 1
    integrality = np.zeros(n_variables, np.int8); integrality[:n_x] = 1
    bounds = Bounds(lower_bounds, upper_bounds)
    objective = np.zeros(n_variables); objective[t_index] = 1
    primary = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints)
    if not primary.success or primary.x is None:
        raise RuntimeError(f"class-balanced split MILP failed: {primary.message}")
    optimum = float(primary.x[t_index])

    # Preserve the minimax optimum, then minimize aggregate imbalance.
    optimum_row = np.zeros(n_variables); optimum_row[t_index] = 1
    secondary_constraints = constraints + [LinearConstraint(optimum_row, -np.inf, optimum + 1e-9)]
    secondary_objective = np.zeros(n_variables); secondary_objective[d_start:] = 1
    secondary = milp(
        secondary_objective, integrality=integrality, bounds=bounds,
        constraints=secondary_constraints,
    )
    if not secondary.success or secondary.x is None:
        raise RuntimeError(f"class-balanced split tie-break MILP failed: {secondary.message}")
    # Finally select one reproducible member of the minimax/L1-optimal set.
    # This separate stage lets the seed break membership ties without trading
    # away either scientific balance objective.
    l1_optimum = float(secondary.x[d_start:].sum())
    l1_row = np.zeros(n_variables); l1_row[d_start:] = 1
    tie_constraints = secondary_constraints + [
        LinearConstraint(l1_row, -np.inf, l1_optimum + 1e-9)
    ]
    rng = random.Random(seed)
    tie_objective = np.zeros(n_variables)
    tie_objective[:n_x] = np.asarray([rng.random() for _ in range(n_x)])
    tied = milp(
        tie_objective, integrality=integrality, bounds=bounds,
        constraints=tie_constraints,
    )
    if not tied.success or tied.x is None:
        raise RuntimeError(f"class-balanced split seeded MILP failed: {tied.message}")
    assignment_matrix = tied.x[:n_x].reshape(len(SPLIT_ROLES), n_files)
    if not np.allclose(assignment_matrix.sum(axis=0), 1, rtol=0, atol=1e-7):
        raise RuntimeError("class-balanced split solver returned invalid assignments")
    return np.argmax(assignment_matrix, axis=0).astype(np.int8), optimum


def validate_split_manifest(
    manifest: Mapping[str, object], *, source_manifest_sha256: str,
    expected_inventory: Iterable[SourceFileRecord] | None = None,
) -> str:
    digest = validate_content_hash(
        manifest, expected_contract=SCOUTING_SPLIT_CONTRACT,
        expected_schema_version=SCOUTING_SPLIT_VERSION,
    )
    if manifest.get("source_manifest_sha256") != require_sha256(
        source_manifest_sha256, name="source_manifest_sha256"
    ):
        raise ValueError("split source lineage differs")
    if manifest.get("fractions") != list(SPLIT_FRACTIONS):
        raise ValueError("split fractions differ from the canonical 60/20/20 contract")
    if manifest.get("seed") != SPLIT_SEED:
        raise ValueError("split seed differs from the canonical contract")
    if manifest.get("algorithm") != SPLIT_ALGORITHM:
        raise ValueError("split allocation algorithm differs")
    if manifest.get("maximum_allowed_class_fraction_deviation") != MAX_CLASS_FRACTION_DEVIATION:
        raise ValueError("split class-balance tolerance differs")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != SPLIT_ROLES:
        raise ValueError("split roles or order differ")
    seen: set[str] = set()
    for role in SPLIT_ROLES:
        value = roles[role]
        if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
            raise ValueError(f"invalid {role} split payload")
        files = [SourceFileRecord(**item) for item in value["files"]]
        if files != sorted(files):
            raise ValueError(f"{role} files are not canonical-order")
        paths = {item.path for item in files}
        if seen & paths:
            raise ValueError("source file occurs in multiple split roles")
        seen |= paths
        if value.get("file_count") != len(files):
            raise ValueError(f"{role} file count differs")
        if value.get("raw_entries") != sum(item.raw_entries for item in files):
            raise ValueError(f"{role} entry count differs")
        counts = np.asarray([item.class_counts for item in files], np.int64).sum(axis=0)
        if value.get("mapped_entries") != int(counts.sum()) or value.get("class_counts") != counts.tolist():
            raise ValueError(f"{role} mapped/class counts differ")
    expected_file_counts = largest_remainder_allocation(len(seen), SPLIT_FRACTIONS)
    actual_file_counts = tuple(int(roles[role]["file_count"]) for role in SPLIT_ROLES)
    if actual_file_counts != expected_file_counts:
        raise ValueError("split role file counts differ from largest-remainder allocation")
    reject_case_aliases(tuple(seen))
    if expected_inventory is not None:
        expected = {item.path: item for item in expected_inventory}
        actual = {
            item.path: item
            for role in SPLIT_ROLES
            for item in (SourceFileRecord(**row) for row in roles[role]["files"])
        }
        if actual != expected:
            raise ValueError("split inventory does not exactly cover authenticated source")
    class_counts = np.asarray([roles[role]["class_counts"] for role in SPLIT_ROLES], np.float64)
    totals = class_counts.sum(axis=0)
    fractions = class_counts / totals[None, :]
    expected_fractions = np.asarray(SPLIT_FRACTIONS)[:, None]
    maximum = float(np.max(np.abs(fractions - expected_fractions)))
    solver_optimum = manifest.get("minimax_solver_optimum")
    if (isinstance(solver_optimum, bool) or not isinstance(solver_optimum, (int, float))
            or not np.isfinite(solver_optimum) or solver_optimum < 0
            or maximum > float(solver_optimum) + 1e-7):
        raise ValueError("split minimax solver diagnostic differs")
    if not np.isclose(
        maximum, manifest.get("maximum_observed_class_fraction_deviation"),
        rtol=0, atol=1e-15,
    ):
        raise ValueError("split class-balance diagnostic differs")
    if not np.allclose(fractions, manifest.get("class_fraction_by_role"), rtol=0, atol=1e-15):
        raise ValueError("split class-fraction matrix differs")
    deviations = fractions - np.asarray(SPLIT_FRACTIONS)[:, None]
    if not np.allclose(
        deviations, manifest.get("class_fraction_deviation_by_role"), rtol=0, atol=1e-15,
    ):
        raise ValueError("split class-fraction deviation matrix differs")
    if maximum > MAX_CLASS_FRACTION_DEVIATION:
        raise ValueError(
            f"file-disjoint split cannot meet class-balance tolerance: {maximum:.6f} > {MAX_CLASS_FRACTION_DEVIATION:.6f}"
        )
    return digest


def role_records(manifest: Mapping[str, object], role: str) -> tuple[SourceFileRecord, ...]:
    if role not in SPLIT_ROLES:
        raise ValueError("unknown split role")
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    payload = roles[role]
    assert isinstance(payload, Mapping)
    return tuple(SourceFileRecord(**item) for item in payload["files"])


__all__ = [
    "MAX_CLASS_FRACTION_DEVIATION", "SCOUTING_SPLIT_CONTRACT", "SPLIT_ALGORITHM",
    "SPLIT_FRACTIONS", "SPLIT_ROLES", "SPLIT_SEED",
    "SourceFileRecord", "build_split_manifest", "largest_remainder_allocation",
    "role_records", "validate_split_manifest",
]
