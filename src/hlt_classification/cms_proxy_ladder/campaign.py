"""Registered controls, coordinates, seeds, and three-spine graph."""
from __future__ import annotations

from fractions import Fraction
import hashlib

from hlt_classification.jetclass2_delphes.campaign import recipe

from .contracts import artifact
from .data import validate_foundation

BRANCHES = {
    "DIRECT": ("D000",),
    "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    "DENSE": ("U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000"),
}
DIRECT_COARSE_BRANCHES = {
    name: BRANCHES[name] for name in ("DIRECT", "COARSE")
}


def coordinate(name: str) -> tuple[Fraction, Fraction]:
    if name in {"OFFLINE", "U000"}:
        return Fraction(0), Fraction(0)
    allowed = {item for path in BRANCHES.values() for item in path}
    if name not in allowed:
        raise ValueError(f"Unregistered CMS-proxy ladder coordinate: {name}")
    fraction = {"033": Fraction(1, 3), "066": Fraction(2, 3)}.get(
        name[1:], Fraction(int(name[1:]), 100),
    )
    return (fraction, Fraction(0)) if name.startswith("U") else (Fraction(1), 1 - fraction)


def paired_seed(coordinate_name: str, domain: str) -> int:
    coordinate(coordinate_name)
    if domain not in {"initialization", "sampler"}:
        raise ValueError("Unknown CMS-proxy ladder RNG domain")
    payload = f"JETCLASS2_CMS_PROXY_LADDER/v1/{domain}/{coordinate_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _build_scientific_plan(
    foundation: dict, *, registered_branches: dict, version: int,
    foundation_root=None, population_selection: dict | None = None,
    node_prefix: str = "CMSP",
) -> dict:
    if foundation_root is not None:
        validate_foundation(foundation, root=foundation_root)
    nodes = [
        {"node_id": "M0HLT", "coordinate": "D000", "teacher": None, "branch": "CONTROL"},
        {"node_id": "OFFLINE", "coordinate": "OFFLINE", "teacher": None, "branch": "CONTROL"},
        {"node_id": "U000", "coordinate": "U000", "teacher": None, "branch": "CONTROL"},
    ]
    publications = ["U000"]
    branch_nodes = {}
    for branch, path in registered_branches.items():
        teacher, previous, names = "U000", "U000", []
        for index, name in enumerate(path):
            node_id = f"{node_prefix}_{branch}_{name}_from_{previous}"
            nodes.append({
                "node_id": node_id, "coordinate": name,
                "teacher": teacher, "branch": branch,
            })
            if index + 1 < len(path):
                publications.append(node_id)
            names.append(node_id)
            teacher, previous = node_id, name
        branch_nodes[branch] = names
    for node in nodes:
        u, f = coordinate(node["coordinate"])
        node.update(
            u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(node["coordinate"], "initialization"),
            sampler_seed=paired_seed(node["coordinate"], "sampler"),
            deployable=node["coordinate"] == "D000",
        )
    parents = {"foundation": foundation["content_hash"]}
    fields = {}
    if population_selection is not None:
        parents["population_selection"] = population_selection["content_hash"]
        fields["population_selection"] = population_selection
    return artifact(
        "SCIENTIFIC_PLAN", version=version,
        parents=parents,
        recipe=recipe(), nodes=nodes, branches=branch_nodes,
        probability_publications=publications,
        fresh_fit_count=len(nodes),
        probability_publication_count=len(publications),
        controls=["M0HLT", "OFFLINE", "U000"],
        recovery_reference={"zero": "M0HLT", "hundred": "OFFLINE"},
        imported_models=[], final_test_evaluation=False, **fields,
    )


def build_scientific_plan(foundation: dict, *, foundation_root=None) -> dict:
    return _build_scientific_plan(
        foundation, registered_branches=BRANCHES, version=1,
        foundation_root=foundation_root,
    )


def build_direct_coarse_plan(
    foundation: dict, *, population_selection: dict, foundation_root=None,
) -> dict:
    if foundation_root is None:
        raise ValueError("Direct/coarse plan requires an authenticated foundation root")
    from .population import validate_direct_coarse_population
    validate_direct_coarse_population(
        population_selection, foundation=foundation,
        foundation_root=foundation_root,
    )
    return _build_scientific_plan(
        foundation, registered_branches=DIRECT_COARSE_BRANCHES, version=2,
        foundation_root=foundation_root,
        population_selection=population_selection,
    )


def task_graph(plan: dict) -> list[dict]:
    rows = []
    for node in plan["nodes"]:
        dependencies = [] if node["teacher"] is None else ["reduce_" + node["teacher"]]
        rows.append({
            "task_id": "train_" + node["node_id"], "kind": "train",
            "node_id": node["node_id"], "dependencies": dependencies,
        })
        if node["node_id"] in plan["probability_publications"]:
            rows.append({
                "task_id": "reduce_" + node["node_id"], "kind": "reduce",
                "node_id": node["node_id"],
                "dependencies": ["train_" + node["node_id"]],
            })
    rows.append({
        "task_id": "aggregate", "kind": "aggregate", "node_id": None,
        "dependencies": [row["task_id"] for row in rows],
    })
    rows.append({
        "task_id": "campaign_complete", "kind": "complete", "node_id": None,
        "dependencies": ["aggregate"],
    })
    return rows


__all__ = [
    "BRANCHES", "DIRECT_COARSE_BRANCHES", "build_direct_coarse_plan",
    "build_scientific_plan", "coordinate", "paired_seed", "task_graph",
]
