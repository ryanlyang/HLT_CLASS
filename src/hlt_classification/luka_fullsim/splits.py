"""File-disjoint reservoirs and exact globally proportional ordinary subsets."""
from __future__ import annotations

from fractions import Fraction
import hashlib

from .contracts import CLASS_NAMES, ROLES, SEED, artifact, row_identity, validate
from .inventory import validate_inventory


def proportional_quotas(counts: list[int], total: int) -> list[int]:
    if (type(total) is not int or total < 1 or total > sum(counts)
            or len(counts) != len(CLASS_NAMES)
            or any(type(n) is not int or n < 1 for n in counts)):
        raise ValueError("Invalid quota population/counts")
    population = sum(counts)
    quotas = [total * n // population for n in counts]
    order = sorted(range(len(counts)), key=lambda i: (-(total * counts[i] % population), i))
    for i in order[:total - sum(quotas)]:
        quotas[i] += 1
    if min(quotas) < 1:
        raise ValueError("Natural proportions give a zero class quota; no silent rebalancing")
    return quotas


def _split(inventory: dict, train: int, validation: int, seed: int) -> dict:
    digest = validate_inventory(inventory)
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid split seed")
    totals = inventory["selected_class_counts"]
    quotas = {"train": proportional_quotas(totals, train),
              "validation": proportional_quotas(totals, validation)}
    if train + validation >= sum(totals):
        raise ValueError("No capacity remains for sealed test files")
    files = inventory["files"]
    fractions = (Fraction(2, 5), Fraction(1, 5), Fraction(2, 5))
    def tie(record):
        return hashlib.sha256(f"LUKA_FULLSIM/group/v1/{seed}/{record['path']}".encode()).hexdigest()
    ordered = sorted(range(len(files)), key=lambda i: (
        -max(Fraction(n, d) for n, d in zip(files[i]["selected_class_counts"], totals)),
        tie(files[i]), files[i]["path"],
    ))
    counts = [[0] * len(totals) for _ in ROLES]
    roles = [None] * len(files)
    for i in ordered:
        mass = files[i]["selected_class_counts"]
        objectives = []
        for r in range(len(ROLES)):
            missing = sum(c == 0 and m > 0 for c, m in zip(counts[r], mass))
            # Exact-rational increment to global squared normalized deficit.
            delta = sum((Fraction(c + m, n) - fractions[r]) ** 2
                        - (Fraction(c, n) - fractions[r]) ** 2
                        for c, m, n in zip(counts[r], mass, totals))
            objectives.append((-missing, delta, r))
        role = min(objectives)[2]
        roles[i] = ROLES[role]
        counts[role] = [c + m for c, m in zip(counts[role], mass)]
    for r, role in enumerate(ROLES):
        required = quotas.get(role, [1] * len(totals))
        shortages = {CLASS_NAMES[c]: dict(required=q, available=counts[r][c])
                     for c, q in enumerate(required) if counts[r][c] < q}
        if shortages:
            raise ValueError(f"File-disjoint {role} capacity shortage: {shortages}")
    members = {}
    for role, quota in quotas.items():
        chosen = {}
        for label, take in enumerate(quota):
            candidates = []
            for i, record in enumerate(files):
                if roles[i] != role:
                    continue
                for entry in record["entries_by_class"][label]:
                    identity = row_identity(record, entry)
                    rank = hashlib.sha256(
                        f"LUKA_FULLSIM/subset/v1/{seed}/{digest}/{role}/{identity}".encode()).digest()
                    candidates.append((rank, i, entry))
            for _, i, entry in sorted(candidates)[:take]:
                chosen.setdefault(i, [[] for _ in CLASS_NAMES])[label].append(entry)
        members[role] = [dict(file_index=i, entries_by_class=[sorted(es) for es in chosen[i]])
                         for i in sorted(chosen)]
    return artifact(
        "SPLITS", parents=dict(inventory=digest), seed=seed,
        algorithm="coverage_then_exact_normalized_deficit_40_20_40_hamilton_global_v1",
        target_reservoir_fractions={role: str(f) for role, f in zip(ROLES, fractions)},
        groups=[dict(path=r["path"], sha256=r["sha256"], role=role) for r, role in zip(files, roles)],
        reservoir_class_counts=dict(zip(ROLES, counts)), quotas=quotas, memberships=members,
        selected_counts=dict(train=train, validation=validation, final_test=sum(counts[2])),
        unused_ordinary_class_counts={role: [n - q for n, q in zip(counts[i], quotas[role])]
                                      for i, role in enumerate(ROLES[:2])},
        final_test_membership="all_eligible_entries_in_final_test_files_only",
        proven_disjointness="file_paths_and_content_hashes",
        cross_file_event_independence="unverified_no_event_ids_in_audited_schema",
        final_test_accessed=False,
    )


def build_splits(inventory: dict, *, train: int = 100000, validation: int = 50000,
                 seed: int = SEED) -> dict:
    return _split(inventory, train, validation, seed)


def validate_splits(value: dict, inventory: dict) -> str:
    digest = validate(value, "SPLITS")
    expected = _split(inventory, value["selected_counts"]["train"],
                      value["selected_counts"]["validation"], value["seed"])
    if value != expected:
        raise ValueError("Frozen FullSim split replay differs")
    return digest


def ordinary_rows(inventory: dict, splits: dict, *, role: str):
    """Authenticated join metadata only; never model features or test access."""
    if role not in ROLES[:2]:
        raise PermissionError("Only train/validation metadata access; final test is sealed")
    validate_splits(splits, inventory)
    for member in splits["memberships"][role]:
        record = inventory["files"][member["file_index"]]
        ordered = sorted((entry, label) for label, es in enumerate(member["entries_by_class"])
                         for entry in es)
        for entry, label in ordered:
            yield dict(identity=row_identity(record, entry), file=record["path"],
                       file_sha256=record["sha256"], tree=record["tree_key"],
                       entry=entry, label=label)
