"""Grouped response splits; historical evaluation files never supply capacity."""
from __future__ import annotations

import base64
import hashlib
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .contracts import MINIMA, OUTER_ROLES, FIT_ROLES, SEED, artifact, validate


def pack_entries(entries, raw_entries: int) -> str:
    entries = np.asarray(entries, dtype=np.int64)
    if raw_entries < 0 or entries.ndim != 1 or np.any(entries < 0) or np.any(entries >= raw_entries):
        raise ValueError("Invalid selected entry indices")
    if len(entries) and np.any(np.diff(entries) <= 0):
        raise ValueError("Entries must be strictly increasing")
    bits = np.zeros(raw_entries, np.uint8)
    bits[entries] = 1
    return base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode("ascii")


def unpack_entries(text: str, raw_entries: int) -> np.ndarray:
    data = base64.b64decode(text, validate=True)
    if len(data) != (raw_entries + 7) // 8 or base64.b64encode(data).decode("ascii") != text:
        raise ValueError("Noncanonical entry mask")
    bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")
    if np.any(bits[raw_entries:]):
        raise ValueError("Nonzero mask padding")
    return np.flatnonzero(bits[:raw_entries])


def _allocate(files: list[dict], fractions: tuple[float, ...], minima: tuple[int, ...]):
    """Minimax normalized counts, then L1, then SHA file-order/role lexicography."""
    n, k = len(files), len(fractions)
    if n < k or sum(f["selected_entries"] for f in files) < sum(minima):
        raise ValueError("Insufficient grouped CMS capacity; a plan amendment is required")
    total = sum(f["selected_entries"] for f in files)
    target = np.asarray(fractions) * total
    nx, t, ds = n * k, n * k, n * k + 1
    size = nx + 1 + k
    rows, low, high = [], [], []

    def add(row, a, b):
        rows.append(row); low.append(a); high.append(b)

    for i in range(n):
        row = np.zeros(size); row[np.arange(k) * n + i] = 1
        add(row, 1, 1)
    for r in range(k):
        count = np.zeros(size)
        count[r*n:(r+1)*n] = [f["selected_entries"] for f in files]
        add(count, max(1, minima[r]), np.inf)
        for source in sorted({f["source"] for f in files}):
            row = np.zeros(size)
            row[r*n:(r+1)*n] = [f["source"] == source for f in files]
            add(row, 1, np.inf)
        for sign in (-1, 1):
            row = sign * count / target[r]; row[t] = -1
            add(row, -np.inf, sign)
            row = sign * count / target[r]; row[ds+r] = -1
            add(row, -np.inf, sign)
    bounds = Bounds(np.zeros(size), np.r_[np.ones(nx), np.full(1+k, np.inf)])
    integrality = np.r_[np.ones(nx), np.zeros(1+k)]

    def solve(objective):
        return milp(objective, integrality=integrality, bounds=bounds,
                    constraints=LinearConstraint(np.asarray(rows), low, high),
                    options={"time_limit": 120., "mip_rel_gap": 0.})

    objective = np.zeros(size); objective[t] = 1
    first = solve(objective)
    if first.status != 0:
        raise ValueError(f"Grouped split minimax not proved: {first.message}")
    row = np.zeros(size); row[t] = 1
    add(row, -np.inf, first.fun + 1e-9)
    objective[:] = 0; objective[ds:] = 1
    second = solve(objective)
    if second.status != 0:
        raise ValueError(f"Grouped split L1 not proved: {second.message}")
    row = np.zeros(size); row[ds:] = 1
    add(row, -np.inf, second.fun + 1e-9)
    order = sorted(range(n), key=lambda i: hashlib.sha256(
        f"CMS2JC2/split/{SEED}/{files[i]['sha256']}".encode()).digest())
    result = np.empty(n, np.int64)
    for i in order:
        for r in range(k):
            row = np.zeros(size); row[r*n+i] = 1
            add(row, 1, 1)
            trial = solve(np.zeros(size))
            if trial.status == 0:
                result[i] = r
                break
            rows.pop(); low.pop(); high.pop()
            if trial.status != 2:
                raise RuntimeError("Split tie-breaking did not prove feasibility/infeasibility")
        else:
            raise RuntimeError("No canonical split tie-break assignment")
    return result, dict(minimax=float(first.fun), l1=float(second.fun))


def build_roles(inventory: dict) -> dict:
    validate(inventory, "CMS_INVENTORY")
    files = inventory["files"]
    if any(f["original_role"] != "train" for f in files):
        raise PermissionError("Only historical CMS training files may be subdivided")
    if len({f["sha256"] for f in files}) != len(files) or len({f["path"].casefold() for f in files}) != len(files):
        raise ValueError("Aliased/duplicate source content")
    indices, outer_solver = _allocate(files, (.8, .1, .1), MINIMA)
    fit_files = [f for f, r in zip(files, indices) if r == 0]
    inner, inner_solver = _allocate(fit_files, (.8, .2), (1, 1))
    inner_by_hash = {f["sha256"]: FIT_ROLES[r] for f, r in zip(fit_files, inner)}
    records = [dict(f, response_role=OUTER_ROLES[r],
                    fit_role=inner_by_hash.get(f["sha256"])) for f, r in zip(files, indices)]
    result = artifact("ROLES", parents={"inventory": inventory["content_hash"]}, files=records,
                      seed=SEED, minima=list(MINIMA), outer_solver=outer_solver,
                      inner_solver=inner_solver,
                      counts={role: sum(f["selected_entries"] for f in records if f["response_role"] == role)
                              for role in OUTER_ROLES})
    validate_roles(result, inventory)
    return result


def validate_roles(value: dict, inventory: dict):
    validate(inventory, "CMS_INVENTORY")
    validate(value, "ROLES", parents={"inventory": inventory["content_hash"]})
    original = {f["sha256"]: f for f in inventory["files"]}
    if len(value["files"]) != len(original) or value["minima"] != list(MINIMA) or value["seed"] != SEED:
        raise ValueError("Role population/policy differs")
    seen = set()
    for row in value["files"]:
        digest = row["sha256"]
        if digest in seen or digest not in original:
            raise ValueError("Duplicate or unexpected role file")
        seen.add(digest)
        if {k: v for k, v in row.items() if k not in {"response_role", "fit_role"}} != original[digest]:
            raise ValueError("Role file identity changed")
        if row["response_role"] not in OUTER_ROLES:
            raise PermissionError("Unknown response role")
        if ((row["response_role"] == "response_fit" and row["fit_role"] not in FIT_ROLES)
                or (row["response_role"] != "response_fit" and row["fit_role"] is not None)):
            raise ValueError("Internal fit role differs")
    sources = {f["source"] for f in inventory["files"]}
    for role, minimum in zip(OUTER_ROLES, MINIMA):
        rows = [r for r in value["files"] if r["response_role"] == role]
        count = sum(r["selected_entries"] for r in rows)
        if count < minimum or count != value["counts"][role] or {r["source"] for r in rows} != sources:
            raise ValueError("Role minima/source coverage/counts differ")
    for role in FIT_ROLES:
        rows = [r for r in value["files"] if r["fit_role"] == role]
        if not rows or {r["source"] for r in rows} != sources:
            raise ValueError("Internal fit-role source coverage differs")


def _apportion(total: int, capacities: list[int]) -> list[int]:
    denominator = sum(capacities)
    if total < 0 or total > denominator:
        raise ValueError("Learning-curve role capacity insufficient")
    allocated = [total*c // denominator for c in capacities]
    left = total - sum(allocated)
    order = sorted(range(len(capacities)), key=lambda i: (-(total*capacities[i] % denominator), i))
    for i in order[:left]:
        allocated[i] += 1
    return allocated


def build_memberships(roles: dict) -> dict:
    validate(roles, "ROLES")
    result = {}
    for role in FIT_ROLES:
        files = [r for r in roles["files"] if r["fit_role"] == role]
        entries, orders = {}, {}
        for f in files:
            selected = unpack_entries(f["entry_mask"], f["raw_entries"])
            if len(selected) != f["selected_entries"]:
                raise ValueError("Selected counts differ from frozen mask")
            keys = [hashlib.sha256(f"CMS2JC2/subset/{SEED}/{f['sha256']}/{f['tree_key']}/{int(i)}".encode()).digest()
                    for i in selected]
            entries[f["path"]] = selected
            orders[f["path"]] = sorted(range(len(selected)), key=lambda i: (keys[i], int(selected[i])))
        previous = [0] * len(files)
        for name, total in (("250K", 200_000 if role == "fit_location" else 50_000),
                            ("1M", 800_000 if role == "fit_location" else 200_000),
                            ("FULL", sum(f["selected_entries"] for f in files))):
            increment = _apportion(total - sum(previous), [f["selected_entries"]-n for f,n in zip(files, previous)])
            counts = [a+b for a,b in zip(previous, increment)]
            result.setdefault(name, {})[role] = [dict(
                path=f["path"], selected_entries=count,
                entry_mask=pack_entries(np.sort(entries[f["path"]][orders[f["path"]][:count]]), f["raw_entries"]),
            ) for f, count in zip(files, counts)]
            previous = counts
    value = artifact("MEMBERSHIPS", parents={"roles": roles["content_hash"]}, budgets=result)
    validate_memberships(value, roles)
    return value


def validate_memberships(value: dict, roles: dict):
    """Require exact coverage, nesting and counts; a missing mask is not FULL."""
    validate(roles, "ROLES")
    validate(value, "MEMBERSHIPS", parents={"roles": roles["content_hash"]})
    if set(value.get("budgets", {})) != {"250K", "1M", "FULL"}:
        raise ValueError("Learning-curve budget registry differs")
    for budget in value["budgets"].values():
        if set(budget) != set(FIT_ROLES):
            raise ValueError("Learning-curve internal role coverage differs")
    for role in FIT_ROLES:
        files = [f for f in roles["files"] if f["fit_role"] == role]
        paths = [f["path"] for f in files]
        previous = {path: np.empty(0, np.int64) for path in paths}
        for budget, expected in (("250K", 200_000 if role == "fit_location" else 50_000),
                                 ("1M", 800_000 if role == "fit_location" else 200_000),
                                 ("FULL", sum(f["selected_entries"] for f in files))):
            rows = value["budgets"][budget][role]
            if [row["path"] for row in rows] != paths:
                raise ValueError("Learning-curve file coverage/order differs")
            if sum(row["selected_entries"] for row in rows) != expected:
                raise ValueError("Learning-curve budget count differs")
            for row, source in zip(rows, files):
                if set(row) != {"path", "selected_entries", "entry_mask"}:
                    raise ValueError("Learning-curve row schema differs")
                selected = unpack_entries(row["entry_mask"], source["raw_entries"])
                eligible = unpack_entries(source["entry_mask"], source["raw_entries"])
                if (len(selected) != row["selected_entries"]
                        or not np.isin(selected, eligible, assume_unique=True).all()
                        or not np.isin(previous[row["path"]], selected, assume_unique=True).all()
                        or (budget == "FULL" and not np.array_equal(selected, eligible))):
                    raise ValueError("Learning-curve mask nesting/population differs")
                previous[row["path"]] = selected
