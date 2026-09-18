"""Partial/group association with deterministic, bounded exact component search."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
import numpy as np

from .bridge import Particles, wrap_phi
from .contracts import artifact, validate


def policy(*, gate: float = .10, max_component_objects: int = 64,
           max_component_hypotheses: int = 4096, search_nodes: int = 100_000) -> dict:
    if gate not in (.05, .1, .2) or min(max_component_objects, max_component_hypotheses, search_nodes) < 1:
        raise ValueError("Invalid association policy")
    return artifact("ASSOCIATION_POLICY", gate=gate, same_side_radius=gate/2,
                    neighbour_limit=6, pt_ratio=[.2, 5.], angular_cost_scale=.1,
                    group_penalty=.5, dustbin_cost=1., max_component_objects=max_component_objects,
                    max_component_hypotheses=max_component_hypotheses, search_nodes=search_nodes,
                    tie_rule="float64_fsum_cost_then_canonical_hypothesis_keys",
                    solver="exact_cost_dominance_vertex_lower_bound_v1",
                    wall_clock_is_not_a_scientific_limit=True)


def distances(p: Particles) -> np.ndarray:
    return np.hypot(p.eta[:, None]-p.eta, wrap_phi(p.phi[:, None]-p.phi))


def groups(p: Particles, radius: float, *, max_size: int = 3,
           reverse_ties: bool = False) -> tuple[tuple[int, ...], ...]:
    """Shared offline-only proposal, ordered independently of storage permutation."""
    dr = distances(p)
    proposals = set()
    for i in range(len(p)):
        nearest = sorted((j for j in range(len(p)) if j != i and dr[i, j] <= radius),
                         key=lambda j: (float(dr[i,j]), p.keys[j]))[:6]
        for extra in range(1, max_size):
            for others in combinations(nearest, extra):
                g = tuple(sorted((i, *others), key=lambda j: p.keys[j]))
                if max(dr[a,b] for a,b in combinations(g, 2)) <= radius:
                    proposals.add(g)
    # Stable sort reverses only exact ties in the two physical priorities.
    # This diagnostic never changes the proposal set or nearest-neighbour rule.
    canonical = sorted(proposals, key=lambda g: tuple(p.keys[j] for j in g), reverse=reverse_ties)
    return tuple(sorted(canonical, key=lambda g: (
        max(float(dr[a,b]) for a,b in combinations(g, 2)),
        -float(p.pt[list(g)].sum()))))


def _axis(vector):
    pt = float(np.hypot(vector[0], vector[1]))
    if pt == 0:
        return None
    return pt, float(np.arcsinh(vector[2]/pt)), float(np.arctan2(vector[1], vector[0]))


@dataclass(frozen=True)
class Hypothesis:
    offline: tuple[int, ...]
    hlt: tuple[int, ...]
    cost: float
    key: tuple[tuple[str, ...], tuple[str, ...]]


def hypotheses(offline: Particles, hlt: Particles, rules: dict) -> list[Hypothesis]:
    validate(rules, "ASSOCIATION_POLICY")
    expected = policy(**{k: rules[k] for k in ("gate", "max_component_objects",
                                              "max_component_hypotheses", "search_nodes")})
    if rules != expected:
        raise ValueError("Unregistered association semantics")
    result = []
    offline_axes, hlt_axes = {}, {}

    def cached_axis(particles, indexes, cache):
        if indexes not in cache:
            cache[indexes] = _axis(particles.p4[list(indexes)].sum(axis=0))
        return cache[indexes]

    def add(a, b):
        a = tuple(sorted(a, key=lambda i: offline.keys[i]))
        b = tuple(sorted(b, key=lambda i: hlt.keys[i]))
        oa = cached_axis(offline, a, offline_axes)
        ha = cached_axis(hlt, b, hlt_axes)
        if oa is None or ha is None:
            return
        ratio = ha[0]/oa[0]
        dr = math.hypot(ha[1]-oa[1], float(wrap_phi(ha[2]-oa[2])))
        if dr > rules["gate"] or not rules["pt_ratio"][0] <= ratio <= rules["pt_ratio"][1]:
            return
        cost = (dr/rules["angular_cost_scale"])**2 + (math.log(ratio)/math.log(2))**2
        cost += rules["group_penalty"] * (len(a)+len(b)-2)
        # A more expensive hypothesis cannot beat its disjoint singleton dustbins.
        if cost <= rules["dustbin_cost"]*(len(a)+len(b)):
            result.append(Hypothesis(a, b, cost,
                          (tuple(offline.keys[i] for i in a), tuple(hlt.keys[i] for i in b))))

    for i in range(len(offline)):
        for j in range(len(hlt)):
            add((i,), (j,))
    for a in groups(offline, rules["same_side_radius"]):
        for j in range(len(hlt)):
            add(a, (j,))
    for b in groups(hlt, rules["same_side_radius"], max_size=2):
        for i in range(len(offline)):
            add((i,), b)
    for i, key in enumerate(offline.keys):
        result.append(Hypothesis((i,), (), rules["dustbin_cost"], ((key,), ())))
    for j, key in enumerate(hlt.keys):
        result.append(Hypothesis((), (j,), rules["dustbin_cost"], ((), (key,))))
    return sorted(result, key=lambda h: h.key)


def associate(offline: Particles, hlt: Particles, rules: dict) -> dict:
    hs = hypotheses(offline, hlt, rules)
    n = len(offline)+len(hlt)
    vertices = [tuple((*h.offline, *(len(offline)+j for j in h.hlt))) for h in hs]
    adjacency = [set() for _ in range(n)]
    for v in vertices:
        for i in v:
            adjacency[i].update(v)
    components, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        stack, component = [i], set()
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            stack.extend(adjacency[item]-component)
        seen.update(component)
        components.append(sorted(component))
    chosen, unresolved, total_nodes = [], [], 0
    for component in components:
        vertex_set = set(component)
        indexes = [i for i,v in enumerate(vertices) if v and v[0] in vertex_set]
        reason = None
        if len(component) > rules["max_component_objects"]:
            reason = "object_limit"
        elif len(indexes) > rules["max_component_hypotheses"]:
            reason = "hypothesis_limit"
        if reason:
            unresolved.append(dict(objects=component, reason=reason)); continue
        local = {v:i for i,v in enumerate(component)}
        masks = {i: sum(1 << local[v] for v in vertices[i]) for i in indexes}
        options = {v: [i for i in indexes if masks[i] & (1 << v)] for v in range(len(component))}
        dustbins = tuple(i for i in indexes if len(vertices[i]) == 1)
        best_ids = tuple(sorted(dustbins))
        best_cost = math.fsum(hs[i].cost for i in best_ids)
        best_key = tuple(hs[i].key for i in best_ids)
        # A deterministic feasible warm start only supplies an upper bound;
        # it is never published as an optimum if the exact search exhausts.
        used, greedy = 0, []
        for i in sorted(indexes, key=lambda i: (
                -(rules["dustbin_cost"]*len(vertices[i])-hs[i].cost)/len(vertices[i]), hs[i].key)):
            if not masks[i] & used:
                used |= masks[i]
                greedy.append(i)
        greedy = tuple(sorted(greedy))
        greedy_cost = math.fsum(hs[i].cost for i in greedy)
        greedy_key = tuple(hs[i].key for i in greedy)
        if (greedy_cost, greedy_key) < (best_cost, best_key):
            best_ids, best_cost, best_key = greedy, greedy_cost, greedy_key
        search_count, exhausted = 0, False
        full = (1 << len(component))-1
        dominance = {}

        def visit(covered: int, selected: tuple[int, ...]):
            nonlocal best_ids, best_cost, best_key, search_count, exhausted
            if exhausted:
                return
            search_count += 1
            if search_count > rules["search_nodes"]:
                exhausted = True; return
            current = math.fsum(hs[i].cost for i in selected)
            if current > best_cost + 1e-12:
                return
            prior = dominance.get(covered, math.inf)
            if current > prior + 1e-12:
                return
            dominance[covered] = min(current, prior)
            if covered == full:
                ids = tuple(sorted(selected)); key = tuple(hs[i].key for i in ids)
                if (current, key) < (best_cost, best_key):
                    best_cost, best_key, best_ids = current, key, ids
                return
            available = [v for v in range(len(component)) if not (covered & (1 << v))]
            feasible = {v: [i for i in options[v] if not masks[i] & covered] for v in available}
            # Every feasible covering pays each object's share of some edge.
            # Minimizing each share independently is an admissible lower bound.
            lower = math.fsum(min(hs[i].cost/len(vertices[i]) for i in feasible[v]) for v in available)
            if current + lower > best_cost + 1e-12:
                return
            v = min(available, key=lambda v: (len(feasible[v]), v))
            for i in sorted(feasible[v], key=lambda i: (hs[i].cost, hs[i].key)):
                visit(covered | masks[i], (*selected, i))
        visit(0, ())
        total_nodes += search_count
        if exhausted:
            unresolved.append(dict(objects=component, reason="deterministic_search_limit"))
        else:
            chosen.extend(best_ids)
    # A partially solved jet is never presented as a fully labelled training jet.
    return artifact("ASSOCIATION", parents={"policy": rules["content_hash"]},
                    offline_keys=list(offline.keys), hlt_keys=list(hlt.keys),
                    resolved=not unresolved, unresolved_components=unresolved,
                    search_nodes=total_nodes, alternative_cost_margin=None,
                    hypotheses=[asdict(hs[i]) for i in sorted(chosen)],
                    optimum_cost=math.fsum(hs[i].cost for i in chosen) if not unresolved else None)
