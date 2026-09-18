"""Frozen separately applied physical conditioning axes (not a Cartesian grid)."""
from __future__ import annotations

import numpy as np

CONDITIONS = ("all", *(f"category:{i}" for i in range(6)),
              *(f"pt:{i}" for i in range(4)), *(f"eta:{i}" for i in range(4)),
              *(f"crowding:{i}" for i in range(3)), "empty_offline")


def particle_cells(row, edges):
    return ["all", f"category:{int(row[0])}", *[
        f"{name}:{int(np.searchsorted(edges[name], row[col], side='right'))}"
        for name, col in (("pt", 2), ("eta", 3), ("crowding", 17))]]


def jet_cells(x, edges):
    if not len(x):
        return ["all", "empty_offline"]
    row = np.median(x, axis=0)
    # A jet can contain several categories; that overlap is declared, not
    # silently converted into a label or a category with dominant momentum.
    return ["all", *[f"category:{i}" for i in sorted(set(map(int, x[:, 0])))],
            *particle_cells(row, edges)[2:]]


def eligible_cells(x, edges):
    keys = set(jet_cells(x, edges))
    for row in x:
        keys.update(particle_cells(row, edges))
    return [k for k in CONDITIONS if k in keys]
