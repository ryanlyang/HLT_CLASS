"""Three-spine persistent-HLT salience campaign graph."""
from __future__ import annotations

from .campaign import coordinate, paired_seed, recipe
from .contracts import artifact
from .salience_foundation import validate_foundation_spec


PATHS = {
    "DIRECT": ("D000",),
    "COARSE": ("U050", "U100", "D066", "D033", "D000"),
    "DENSE": ("U033", "U066", "U100", "D080", "D060", "D040", "D020", "D000"),
}


def build_campaign_plan(foundation: dict) -> dict:
    parent = validate_foundation_spec(foundation)
    nodes = [
        dict(node_id="M0HLT", coordinate="D000", teacher=None, branch="REFERENCE"),
        dict(node_id="U000", coordinate="U000", teacher=None, branch="REFERENCE"),
    ]
    publications = ["U000"]
    branches: dict[str, list[str]] = {}
    for branch, path in PATHS.items():
        teacher, previous, names = "U000", "U000", []
        for index, name in enumerate(path):
            node_id = f"JC2S_{branch}_{name}_from_{previous}"
            nodes.append(
                dict(node_id=node_id, coordinate=name, teacher=teacher, branch=branch)
            )
            if index + 1 < len(path):
                publications.append(node_id)
            names.append(node_id)
            teacher, previous = node_id, name
        branches[branch] = names
    for node in nodes:
        u, f = coordinate(node["coordinate"])
        node.update(
            u=[u.numerator, u.denominator], f=[f.numerator, f.denominator],
            initialization_seed=paired_seed(node["coordinate"], "initialization"),
            sampler_seed=paired_seed(node["coordinate"], "sampler"),
            deployable=node["coordinate"] == "D000",
        )
    splits = foundation["splits"]
    return artifact(
        "SALIENCE_CAMPAIGN_PLAN", foundation_sha256=parent,
        matcher_sha256=foundation["matcher"]["content_hash"],
        candidate=foundation["candidate"], inputs_sha256=foundation["inputs"]["content_hash"],
        recipe=recipe(), nodes=nodes, branches=branches,
        probability_publications=publications,
        fresh_fit_count=len(nodes), probability_publication_count=len(publications),
        branch_order=list(PATHS), ultradense_present=False,
        persistent_hlt=True, pure_offline_u000=False,
        split_profile=splits["profile"], split_profile_sha256=splits["content_hash"],
        split_registry_sha256=splits["registry_sha256"],
        role_counts=splits["role_counts"],
        role_membership_sha256={
            role: membership["content_hash"]
            for role, membership in splits["memberships"].items()
        },
        imported_models=[], final_test_accessed=False, executable=True,
    )


__all__ = ["PATHS", "build_campaign_plan", "coordinate", "paired_seed", "recipe"]
