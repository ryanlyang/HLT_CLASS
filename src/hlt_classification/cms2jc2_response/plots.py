"""Bounded hash-stratified and explicitly worst-case physical example panels."""
from __future__ import annotations

import hashlib
import heapq
import html
import numpy as np

from .bridge import wrap_phi
from .contracts import artifact
from .features import features
from .metrics import jet_summary
from .response import Generator

COLORS = ("#397ac0", "#d48824", "#3e9663", "#bb4b89", "#7852a1", "#555555")


def _stratum(particles):
    f = features(particles)
    scalar_pt = float(particles.pt.sum())
    charged = float(particles.pt[particles.charge != 0].sum())/scalar_pt if scalar_pt else 0.
    return (int(scalar_pt >= 500.) + 2*int(len(particles) >= 50) +
            4*int(charged >= .5) + 8*int(len(f) > 0 and np.median(f[:, 17]) >= 3))


def _keep(heap, priority, identity, value, cap):
    if not cap:
        return
    tie = int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big")
    item = (-priority, -tie, identity, value)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif (priority, tie) < (-heap[0][0], -heap[0][1]):
        heapq.heapreplace(heap, item)


def _particles(p):
    return dict(p4=p.p4.tolist(), category=p.category.tolist(), charge=p.charge.tolist(),
                tracking=p.tracking.tolist(), valid=p.valid.tolist(), keys=list(p.keys))


def panels(pairs, response, *, role: str, selection_hash: str, examples=200, worst=50):
    if role not in {"response_select", "response_confirm", "jc2_train"} or examples != 200:
        raise ValueError("Visual sample registry differs")
    if worst != (50 if role == "response_select" else 0):
        raise ValueError("Worst panels are selection-only")
    generator = Generator(response)
    # A global fill reservoir handles empty strata; no full population retained.
    global_heap, strata, worst_heap = [], {i: [] for i in range(16)}, []
    populations, jets = [0]*16, 0
    for pair in pairs:
        if (role == "jc2_train") != (pair.hlt is None):
            raise PermissionError("Visual source/role differs")
        priority = int.from_bytes(hashlib.sha256(("CMS2JC2_VISUAL/v1:"+pair.identity).encode()).digest(), "big")
        cell = _stratum(pair.offline)
        populations[cell] += 1; jets += 1
        _keep(global_heap, priority, pair.identity, pair, examples)
        _keep(strata[cell], priority, pair.identity, pair, examples//16+int(cell < examples % 16))
        if worst:
            proxy, _ = generator(pair.offline, jet=pair.identity)
            a, b = jet_summary(pair.hlt), jet_summary(proxy)
            a["scalar_pt"], b["scalar_pt"] = float(pair.hlt.pt.sum()), float(proxy.pt.sum())
            score = float(np.mean([abs(a.get(k, 0.)-b.get(k, 0.))/max(1., abs(a.get(k, 0.)))
                                   for k in ("multiplicity", "scalar_pt", "mass")]))
            _keep(worst_heap, -score, pair.identity, pair, worst)
        if jets % 1000 == 0:
            print(f"CMS2JC2 phase=visual_population role={role} jets={jets}", flush=True)
    chosen = {r[2]: r[3] for heap in strata.values() for r in heap}
    for row in sorted(global_heap, key=lambda r: (-r[0], -r[1])):
        if len(chosen) >= examples:
            break
        chosen[row[2]] = row[3]
    sets = {"stratified_hash": list(sorted(chosen.values(), key=lambda p: p.identity)),
            "worst50_not_random": [r[3] for r in sorted(worst_heap, key=lambda r: (-r[0], -r[1]))]}
    payloads = {}
    tie_deltas = []
    for rule, selected in sets.items():
        for pair in selected:
            if pair.identity not in payloads:
                proxy, trace = generator(pair.offline, jet=pair.identity, replica=0, trace=True)
                payloads[pair.identity] = dict(identity=pair.identity, source_group=pair.source_group,
                    offline=_particles(pair.offline), real_hlt=_particles(pair.hlt) if pair.hlt is not None else None,
                    proxy=_particles(proxy), trace=trace, stratum=_stratum(pair.offline))
            if rule == "stratified_hash" and role == "response_select":
                alternative, _ = generator(pair.offline, jet=pair.identity, replica=0, diagnostic_reverse_ties=True)
                nominal = payloads[pair.identity]["proxy"]
                nominal_p4 = np.asarray(nominal["p4"]).reshape(-1, 4)
                changed = not (nominal == _particles(alternative))
                tie_deltas.append(dict(jet=pair.identity, changed=changed,
                    multiplicity_delta=len(alternative)-len(nominal_p4),
                    scalar_pt_delta=float(alternative.pt.sum()-np.hypot(nominal_p4[:, 0], nominal_p4[:, 1]).sum())))
    return artifact("EXAMPLES", parents={"selection": selection_hash, "response": response["content_hash"]},
                    role=role, replica=0, population_jets=jets, stratum_populations=populations,
                    rule="16 fixed binary strata, bottom SHA256 in each, global bottom hash fills empty strata",
                    strata=dict(scalar_pt_gev=500., multiplicity=50, charged_pt_fraction=.5, median_crowding_005=3),
                    worst_rule="mean abs collection error / max(1,real) for multiplicity, scalar pT, mass",
                    sets={k: [p.identity for p in rows] for k, rows in sets.items()}, examples=payloads,
                    small_population=len(chosen) < examples,
                    reverse_tie_order_sensitivity=dict(nonselecting=True, replica=0,
                        scope="200 fixed stratified selection examples only; same frozen model and random keys",
                        rule="reverse canonical keys only for exactly equal radius and summed-pT merge priorities",
                        model_refitted=False, primary_output_unchanged=True, rows=tie_deltas),
                    physical_status=response["physical_status"], public_release_authorized=False,
                    native_jc2_hlt_accessed=False if role == "jc2_train" else None)


def example_svg(example, *, response_hash, role, selection_rule):
    """Vector-only physical plot; no raster assets or external plotting service."""
    sides = [("Offline", example["offline"])]
    if example["real_hlt"] is not None:
        sides.append(("Real CMS HLT", example["real_hlt"]))
    sides.append(("CMS-calibrated proxy", example["proxy"]))
    base = np.asarray(example["offline"]["p4"]).reshape(-1, 4).sum(axis=0)
    pt = float(np.hypot(*base[:2]))
    eta0 = float(np.arcsinh(base[2]/pt)) if pt else 0.
    phi0 = float(np.arctan2(base[1], base[0])) if pt else 0.
    points, extent = [], .5
    for name, collection in sides:
        p = np.asarray(collection["p4"]).reshape(-1, 4)
        pts = np.hypot(p[:, 0], p[:, 1])
        eta = np.arcsinh(np.divide(p[:, 2], pts, out=np.zeros(len(p)), where=pts > 0))-eta0
        phi = wrap_phi(np.arctan2(p[:, 1], p[:, 0])-phi0)
        if len(p):
            extent = max(extent, float(np.max(np.abs(eta))), float(np.max(np.abs(phi))))
        points.append((name, collection, pts, eta, phi))
    width = 380*len(sides)
    esc = html.escape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="490" viewBox="0 0 {width} 490">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="15" y="20" font-size="12">{esc(role)} | {esc(selection_rule)} | replica 0</text>',
           f'<text x="15" y="38" font-size="10">Response: {esc(response_hash)}</text>',
           f'<text x="15" y="53" font-size="10">Jet key: {esc(example["identity"])}</text>']
    for j, (name, collection, pts, eta, phi) in enumerate(points):
        x0 = 15+j*380
        svg.extend([f'<text x="{x0}" y="78" font-size="14">{esc(name)}; N={len(pts)}</text>',
                    f'<rect x="{x0}" y="90" width="350" height="330" fill="none" stroke="#999"/>'])
        for i in range(len(pts)):
            x, y = x0+175+160*phi[i]/extent, 255-150*eta[i]/extent
            radius = min(13., 2+float(np.sqrt(pts[i])))
            title = f"key={collection['keys'][i]} pt={pts[i]:.5g} GeV tracking(mm,mm,mm,mm)={collection['tracking'][i]} valid={collection['valid'][i]}"
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" fill="{COLORS[collection["category"][i]]}" opacity=".65"><title>{esc(title)}</title></circle>')
        svg.append(f'<text x="{x0}" y="438" font-size="11">delta phi horizontal; delta eta vertical; range ±{extent:.3g}</text>')
    svg.extend(['<text x="15" y="457" font-size="11">Colours: charged hadron blue; neutral hadron orange; photon green; electron pink; muon purple; unknown grey.</text>',
                '<text x="15" y="476" font-size="11">Marker radius = min(13, 2+sqrt(pT/GeV)); full operations/validity/support flags in companion JSON.</text>', '</svg>'])
    return "\n".join(svg).encode()
