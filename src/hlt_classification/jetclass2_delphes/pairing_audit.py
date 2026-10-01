"""Bounded raw jet-pair diagnostics, independent of constituent assignments."""
from __future__ import annotations

import hashlib
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from scipy.stats import rankdata

from .contracts import artifact, row_identity
from .inputs import eta_phi, wrap_phi
from .inventory import validate_inventory, verify_file
from .provenance import source_record
from .reader import Particles, _particles
from .schema import CLASS_NAMES, PARTICLE_FIELDS
from .selection import selected_mask
from .split_registry import unpack_entries, validate_split_profile

OPTIONAL_BRANCHES = (
    "hlt_jet_dr_offline", "jet_eta", "jet_phi", "hlt_jet_eta", "hlt_jet_phi",
)
QUANTITIES = ("pt", "mass", "count", "charged_count", "charged_pt_fraction")


def _seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"JC2/pair-audit/v1/{seed}/{key}".encode()).digest()[:8], "little")


def describe(particles: Particles) -> dict:
    raw = particles.values.astype(np.float64)
    p4 = raw[:, :4].sum(axis=0)
    eta, phi = eta_phi(p4)
    mass2 = float(p4[3] ** 2 - np.dot(p4[:3], p4[:3]))
    pt = np.hypot(raw[:, 0], raw[:, 1])
    charged = raw[:, 4] != 0
    return dict(
        pt=float(np.hypot(*p4[:2])), eta=float(eta), phi=float(phi),
        mass=float(np.sqrt(max(0., mass2))), negative_mass2=mass2 < 0,
        count=len(raw), charged_count=int(charged.sum()),
        charged_pt_fraction=float(pt[charged].sum() / pt.sum()),
    )


def delta_r(left: dict, right: dict) -> float:
    return float(np.hypot(left["eta"] - right["eta"], wrap_phi(left["phi"] - right["phi"])))


def sample_pairs(inventory: dict, profile: dict, *, data_root: Path, role="train",
                 files_per_source=16, rows_per_file=256, seed=20260916) -> tuple[list[dict], list[dict]]:
    """Hash-selected files; seeded entry sampling from the frozen role mask.

    Only bounded ROOT chunks are resident. No final-test branch is opened.
    Equal rows per selected file are a diagnostic sample, not population weights.
    """
    if role not in {"train", "validation"}:
        raise PermissionError("Pairing diagnostics cannot access final_test")
    if (type(seed) is not int or seed < 0 or type(files_per_source) is not int
            or not 1 <= files_per_source <= 64 or type(rows_per_file) is not int
            or not 2 <= rows_per_file <= 1024):
        raise ValueError("Invalid bounded diagnostic sample settings")
    inv_hash = validate_inventory(inventory)
    validate_split_profile(profile, inventory)
    memberships = {r["path"]: r for r in profile["memberships"][role]["files"]}
    candidates: dict[str, list[dict]] = {}
    for record in inventory["files"]:
        if record["path"] in memberships and memberships[record["path"]]["rows"]:
            candidates.setdefault(record["source"], []).append(record)
    chosen = []
    for source in sorted(candidates):
        chosen.extend(sorted(candidates[source], key=lambda r: _seed(seed, r["path"]))[:files_per_source])
    results, files = [], []
    root = Path(data_root).resolve(strict=True)
    for number, record in enumerate(chosen, 1):
        path = verify_file(root, record, inventory)
        members = unpack_entries(memberships[record["path"]]["entry_mask"], record["entries"])
        rng = np.random.default_rng(_seed(seed, record["path"] + "/entries"))
        entries = np.sort(rng.choice(members, min(rows_per_file, len(members)), replace=False))
        with uproot.open(path) as handle:
            tree = handle[record["tree_key"]]
            optional = [name for name in OPTIONAL_BRANCHES if name in tree.keys()]
            branches = ["jet_label", "hlt_matched", "jet_nparticles", "hlt_jet_nparticles"]
            branches += [prefix + field for prefix in ("part_", "hlt_part_") for field in PARTICLE_FIELDS]
            for chunk in np.unique(entries // 2048):
                start = int(chunk) * 2048
                selected = entries[(entries >= start) & (entries < start + 2048)]
                arrays = tree.arrays(branches + optional, entry_start=start,
                                     entry_stop=min(start + 2048, record["entries"]), library="ak", how=dict)
                keep, labels = selected_mask(ak.to_numpy(arrays["jet_label"]),
                                            ak.to_numpy(arrays["hlt_matched"]),
                                            record["source"], inventory["selection"])
                for entry in selected:
                    index = int(entry) - start
                    if not keep[index]:
                        raise ValueError("Diagnostic membership contains an ineligible row")
                    hlt = _particles(arrays, index, "hlt_part_", int(arrays["hlt_jet_nparticles"][index]))
                    offline = _particles(arrays, index, "part_", int(arrays["jet_nparticles"][index]))
                    extra = {name: float(arrays[name][index]) for name in optional}
                    producer_dr = extra.get("hlt_jet_dr_offline")
                    if producer_dr is not None and (not np.isfinite(producer_dr) or producer_dr < 0):
                        producer_dr = None
                    branch_dr = None
                    if all(name in extra and np.isfinite(extra[name]) for name in OPTIONAL_BRANCHES[1:]):
                        branch_dr = float(np.hypot(extra["jet_eta"] - extra["hlt_jet_eta"],
                                                  wrap_phi(extra["jet_phi"] - extra["hlt_jet_phi"])))
                    results.append(dict(
                        identity=row_identity(inv_hash, record["path"], record["tree_key"], int(entry)),
                        source=record["source"], file=record["path"], label=int(labels[index]),
                        hlt=describe(hlt), offline=describe(offline),
                        producer_dr=producer_dr, branch_axis_dr=branch_dr,
                    ))
        verify_file(root, record, inventory)
        files.append(dict(path=record["path"], sha256=record["sha256"], tree_key=record["tree_key"],
                          entries=entries.tolist(), optional_branches=optional))
        print(f"JC2 pair_audit file={number}/{len(chosen)} sampled={len(results)} path={record['path']}", flush=True)
    if not results or len({r["identity"] for r in results}) != len(results):
        raise ValueError("Empty or duplicate pairing sample")
    return results, files


def distribution(values) -> dict:
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    if not len(values):
        return dict(count=0, mean=None, q50=None, q90=None, q99=None, maximum=None)
    return dict(count=len(values), mean=float(values.mean()),
                **{f"q{p}": float(np.percentile(values, p)) for p in (50, 90, 99)},
                maximum=float(values.max()))


def spearman(left, right):
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def shuffled_partners(rows: list[dict], *, seed: int, control: str) -> np.ndarray:
    """Bijective, no-self partners inside class/source (and optional pT bins).

    Singletons are excluded, never treated as a shuffled self-pair. The
    kinematic control bins by recipient HLT pT (width .2 in log pT).
    """
    if control not in {"class_source", "class_source_pt", "class_file"}:
        raise ValueError("Unknown pairing control")
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        key = (row["label"], row["file"] if control == "class_file" else row["source"])
        if control == "class_source_pt":
            key += (int(np.floor(np.log(max(row["hlt"]["pt"], 1e-8)) / .2)),)
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    partners = np.full(len(rows), -1, dtype=np.int64)
    for key in sorted(groups):
        group = np.asarray(groups[key], np.int64)
        if len(group) > 1:
            order = rng.permutation(group)
            partners[order] = np.roll(order, 1)
    return partners


def pairing_metrics(rows: list[dict], indices: np.ndarray, partners: np.ndarray) -> dict:
    dr = [delta_r(rows[int(i)]["hlt"], rows[int(j)]["offline"]) for i, j in zip(indices, partners)]
    correlations, differences = {}, {}
    for name in QUANTITIES:
        left = np.array([rows[int(i)]["hlt"][name] for i in indices])
        right = np.array([rows[int(j)]["offline"][name] for j in partners])
        correlations[name] = spearman(left, right)
        differences[name] = distribution(np.abs(left - right))
    return dict(rows=len(indices), axis_dr=distribution(dr),
                dr_above={str(cut): float(np.mean(np.asarray(dr) > cut)) if len(dr) else None
                          for cut in (.05, .1, .2, .4, 1.)},
                spearman=correlations, absolute_difference=differences)


def summarize(rows: list[dict], *, seed=20260916, repeats=20) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("Control repeats must be 1..100")
    controls = {}
    for control in ("class_source", "class_source_pt", "class_file"):
        permutations = [shuffled_partners(rows, seed=_seed(seed, f"{control}/{r}"), control=control)
                        for r in range(repeats)]
        eligible = np.flatnonzero(permutations[0] >= 0)
        by_class = {}
        for name, indices in [("ALL", eligible)] + [
            (label, np.array([i for i in eligible if rows[int(i)]["label"] == c], dtype=np.int64))
            for c, label in enumerate(CLASS_NAMES)
        ]:
            actual = pairing_metrics(rows, indices, indices)
            shuffled = [pairing_metrics(rows, indices, p[indices]) for p in permutations]
            by_class[name] = dict(
                actual=actual,
                shuffled_median_dr=distribution([r["axis_dr"]["q50"] for r in shuffled if r["rows"]]),
                shuffled_spearman={key: distribution([r["spearman"][key] for r in shuffled
                                                      if r["spearman"][key] is not None]) for key in QUANTITIES},
                shuffled_median_absolute_difference={key: distribution([
                    r["absolute_difference"][key]["q50"] for r in shuffled if r["rows"]]) for key in QUANTITIES},
            )
        controls[control] = dict(excluded_singleton_rows=len(rows) - len(eligible), by_class=by_class)
    native_dr = np.array([delta_r(row["hlt"], row["offline"]) for row in rows])
    producer_rows = [r for r in rows if r["producer_dr"] is not None]
    both = [r for r in producer_rows if r["branch_axis_dr"] is not None]
    return dict(
        rows=len(rows), class_counts={name: sum(r["label"] == c for r in rows) for c, name in enumerate(CLASS_NAMES)},
        actual=pairing_metrics(rows, np.arange(len(rows)), np.arange(len(rows))), controls=controls,
        producer_dr=distribution([r["producer_dr"] for r in producer_rows]),
        branch_axis_dr=distribution([r["branch_axis_dr"] for r in rows if r["branch_axis_dr"] is not None]),
        producer_vs_branch_axis_absolute_difference=distribution([abs(r["producer_dr"] - r["branch_axis_dr"]) for r in both]),
        producer_vs_constituent_axis_absolute_difference=distribution([
            abs(r["producer_dr"] - delta_r(r["hlt"], r["offline"])) for r in producer_rows]),
        negative_mass2_counts={side: sum(r[side]["negative_mass2"] for r in rows) for side in ("hlt", "offline")},
        largest_dr_rows=[dict(identity=rows[i]["identity"], file=rows[i]["file"],
                             class_name=CLASS_NAMES[rows[i]["label"]], axis_dr=float(native_dr[i]),
                             producer_dr=rows[i]["producer_dr"])
                         for i in np.argsort(-native_dr, kind="stable")[:10]],
    )


def run_audit(inventory: dict, profile: dict, *, data_root: Path, role="train",
              files_per_source=16, rows_per_file=256, seed=20260916, repeats=20) -> dict:
    rows, files = sample_pairs(inventory, profile, data_root=data_root, role=role,
                              files_per_source=files_per_source, rows_per_file=rows_per_file, seed=seed)
    return artifact(
        "JET_PAIRING_AUDIT", inventory_sha256=inventory["content_hash"],
        split_profile_sha256=profile["content_hash"], profile=profile["profile"],
        role=role, role_membership_sha256=profile["memberships"][role]["content_hash"],
        sample_policy="hash_files_per_source_random_registered_entries_v1",
        seed=seed, files_per_source=files_per_source, rows_per_file=rows_per_file,
        shuffle_repeats=repeats, pt_bin_log_width=.2, files=files,
        ordered_identity_sha256=hashlib.sha256(b"".join(bytes.fromhex(r["identity"]) for r in rows)).hexdigest(),
        producer=source_record(*[f"src/hlt_classification/jetclass2_delphes/{name}.py" for name in (
            "pairing_audit", "reader", "inputs", "inventory", "schema", "selection", "split_registry", "splits", "contracts", "provenance",
        )]),
        diagnostics=summarize(rows, seed=seed, repeats=repeats),
        final_test_accessed=False, matching_recomputed=False, physical_pairing_certified=False,
        limitations=[
            "Bounded per-file sample; not a population-weighted mismatch-rate estimate.",
            "Geometric/kinematic agreement is not truth-level certification of jet or particle identity.",
            "Stored producer distance is corroboration, not independent truth.",
            "Shuffle ranges are permutation variability, not confidence intervals on a mismatch rate.",
            "Constituent-sum axes can differ from producer jet axes; disagreement needs interpretation.",
            "This audit does not run models or independently replay target-bank prediction alignment.",
        ],
    )


def print_report(report: dict) -> None:
    from .contracts import validate
    validate(report, "JET_PAIRING_AUDIT")
    def fmt(value):
        return "n/a" if value is None else f"{value:.5f}"
    diagnostics = report["diagnostics"]
    print(f"Jet-pair audit: {report['profile']} / {report['role']} / {diagnostics['rows']} jets / {len(report['files'])} files")
    print("Raw global directions; no per-jet centering and no particle matching.")
    dr = diagnostics["actual"]["axis_dr"]
    print(f"Actual axis deltaR: median={fmt(dr['q50'])} p90={fmt(dr['q90'])} p99={fmt(dr['q99'])} max={fmt(dr['maximum'])}")
    print(f"Stored producer deltaR: {diagnostics['producer_dr']}")
    print(f"Producer vs stored jet-axis deltaR absolute difference: {diagnostics['producer_vs_branch_axis_absolute_difference']}")
    for control, values in diagnostics["controls"].items():
        print(f"\nControl: {control}; excluded singletons={values['excluded_singleton_rows']}")
        print(f"{'class':<14} {'rows':>6} {'real dR50':>10} {'shuf dR50':>10} {'real rhoPt':>11} {'shuf rhoPt':>11} {'real rhoN':>10} {'shuf rhoN':>10}")
        for name, row in values["by_class"].items():
            actual = row["actual"]
            print(f"{name:<14} {actual['rows']:>6} {fmt(actual['axis_dr']['q50']):>10} "
                  f"{fmt(row['shuffled_median_dr']['q50']):>10} {fmt(actual['spearman']['pt']):>11} "
                  f"{fmt(row['shuffled_spearman']['pt']['q50']):>11} {fmt(actual['spearman']['count']):>10} "
                  f"{fmt(row['shuffled_spearman']['count']['q50']):>10}")
    print("\nFinal test accessed: False. Diagnostic completed; physical pairing is not automatically certified.")
