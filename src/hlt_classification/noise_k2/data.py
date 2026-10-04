"""Ordinary-role selection, fresh compact maps and RAM-only physical views."""
from collections import Counter
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time

import awkward as ak
import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_npz_arrays,
)
from hlt_classification.literature_proxy_consumer import RelocatedDataset
from hlt_classification.literature_proxy_production import population
from hlt_classification.literature_proxy.population import BRANCHES, JC2_FIELDS, from_columns
from hlt_classification.cms2jc2_response.readers import authenticated_open
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.jetclass2_delphes.inventory import latest_tree
from hlt_classification.jetclass2_delphes.selection import selected_mask
from hlt_classification.jetclass2_delphes.split_registry import nested_quotas
from hlt_classification.jetclass2_delphes.concat_k2_data import partition_codes
from hlt_classification.jetclass2_delphes.cache import RamBlock, RamCache
from hlt_classification.cms_proxy_ladder.inputs import build_inputs
from .contracts import artifact, require, publish, load, reference, checked
from .views import match, build_view, validate_mapping, STRENGTHS


def dataset(spec):
    return RelocatedDataset.from_oscar_copy(spec["dataset_root"],
        expected_manifest_sha256=spec["manifest_sha256"])


def ordinary(role):
    if role not in ("train", "validation"):
        raise PermissionError("Final-test particles are sealed")


def write_arrays(path, arrays):
    atomic_publish_bytes(path, deterministic_npz_bytes(arrays))


def select_population(spec):
    data = dataset(spec)
    root = Path(spec["campaign_root"])
    reports = {}
    all_ids = set()
    for role, target in spec["counts"].items():
        ordinary(role)
        ids, labels, entries, shard_indices = [], [], [], []
        shards = [s for s in spec["shards"] if s["role"] == role]
        for s in shards:
            source, raw_entries = population.entries_for(data.study["population"], s["source"])
            with authenticated_open(data.offline_root/source["path"], source["sha256"]) as handle:
                key, tree = latest_tree(handle)
                require(key == source["tree_key"] and tree.num_entries == source["raw_entries"],
                        "Population ROOT identity differs")
                # Scalar-only eligibility and targets; no particle decoding or native HLT.
                values = tree.arrays(["jet_label", "hlt_matched"], library="np")
                keep, target_labels = selected_mask(values["jet_label"], values["hlt_matched"],
                    source["source"], data.inventory["selection"])
                require(keep[raw_entries].all(), "Frozen population contains ineligible rows")
                labels.extend(target_labels[raw_entries].tolist())
            ids.extend(bytes.fromhex(v) for v in population.ids(data.inventory["content_hash"], source, raw_entries))
            entries.extend(raw_entries.tolist())
            shard_indices.extend([s["index"]]*len(raw_entries))
        labels = np.asarray(labels, np.int64)
        require(len(ids) == data.study["counts"][role] and len(set(ids)) == len(ids), "Parent role coverage differs")
        quotas = nested_quotas(np.bincount(labels, minlength=11).tolist(), (target,))[0]
        chosen = []
        prefix = ("NOISE_V3/K2/100K/v1/"+spec["manifest_sha256"]+"/"+role+"/").encode()
        for c, quota in enumerate(quotas):
            indexes = np.flatnonzero(labels == c).tolist()
            indexes.sort(key=lambda i: (hashlib.sha256(prefix+ids[i]).digest(), ids[i]))
            chosen.extend(indexes[:quota])
        chosen = np.asarray(sorted(chosen), np.int64)
        identities = np.asarray([np.frombuffer(ids[i], np.uint8) for i in chosen])
        require(not all_ids.intersection(ids[i] for i in chosen), "Train/validation overlap")
        all_ids.update(ids[i] for i in chosen)
        arrays = dict(identities=identities, labels=labels[chosen],
            entries=np.asarray(entries, np.int64)[chosen],
            shards=np.asarray(shard_indices, np.int32)[chosen])
        if role == "validation":
            arrays["codes"] = partition_codes(identities, arrays["labels"])
        path = root/"population"/(role+".npz")
        write_arrays(path, arrays)
        reports[role] = dict(rows=target, parent_rows=len(labels), class_counts=quotas,
                            arrays=reference(root, path))
        print(f"NOISE-K2 selection role={role} rows={target} parent={len(labels)}", flush=True)
    return publish(root/"population.json", artifact("POPULATION", campaign_sha256=spec["content_hash"],
        manifest_sha256=spec["manifest_sha256"], roles=reports,
        policy=spec["population_policy"], sealed_final_test_count=spec["final_test_count"]))


def selected(spec, role):
    ordinary(role)
    root = Path(spec["campaign_root"])
    report = load(root/"population.json", "POPULATION")
    require(report["campaign_sha256"] == spec["content_hash"] and report["manifest_sha256"] == spec["manifest_sha256"],
            "Population lineage differs")
    arrays = load_npz_arrays(checked(root, report["roles"][role]["arrays"]))
    n = spec["counts"][role]
    require(set(arrays) == {"identities", "labels", "entries", "shards"} | ({"codes"} if role == "validation" else set()),
            "Population columns differ")
    require(arrays["identities"].dtype == np.uint8 and arrays["identities"].shape == (n, 32)
        and len(np.unique(arrays["identities"], axis=0)) == n
        and all(arrays[k].shape == (n,) for k in ("labels", "entries", "shards"))
        and arrays["labels"].dtype == np.int64 and arrays["entries"].dtype == np.int64
        and arrays["shards"].dtype == np.int32 and np.all(arrays["entries"] >= 0)
        and np.all((arrays["labels"] >= 0) & (arrays["labels"] < 11)), "Population arrays differ")
    require(set(arrays["shards"].tolist()) <= {s["index"] for s in spec["shards"] if s["role"] == role},
            "Selection crosses role")
    require(report["roles"][role]["rows"] == n
        and np.bincount(arrays["labels"], minlength=11).tolist() == report["roles"][role]["class_counts"],
        "Population class quotas differ")
    if role == "validation":
        require(arrays["codes"].dtype == np.uint8
                and np.array_equal(arrays["codes"], partition_codes(arrays["identities"], arrays["labels"])),
                "Validation partition differs")
    return report, arrays


def iter_selected(data, shard, rows, *, paired):
    """Yield selected raw-entry joins; proxy-only path never opens ROOT."""
    ordinary(shard["role"])
    require(type(paired) is bool, "Explicit paired capability required")
    source, entries = population.entries_for(data.study["population"], shard["source"])
    require(np.isin(rows["entries"], entries).all() and len(np.unique(rows["entries"])) == len(rows["entries"]),
            "Requested raw entries escape frozen shard")
    expected = dict(zip(rows["entries"].tolist(), range(len(rows["entries"]))))
    selected_shard = data._shards(shard["role"], [shard["source"]["shard_id"]])[0]
    opened = authenticated_open(data.offline_root/source["path"], source["sha256"]) if paired else nullcontext(None)
    count = 0
    with opened as handle:
        if paired:
            key, tree = latest_tree(handle)
            require(key == source["tree_key"] and tree.num_entries == source["raw_entries"], "ROOT identity differs")
        for block_entries, block in data._proxy_blocks(selected_shard, entries):
            bucket_index, raw = None, None
            for bank_index, entry in enumerate(block_entries):
                i = expected.get(int(entry))
                if i is None:
                    continue
                require(i == count, "Selected shard order differs")
                # Whole bank bytes/schema/identities were authenticated above.
                # Decode physical objects only for the selected 100k/50k rows.
                lo, hi = block["offsets"][bank_index:bank_index+2]
                identity = bytes(block["jet_identity"][bank_index]).hex()
                proxy = Particles(*(block[k][lo:hi] for k in ("p4","charge","category","tracking","valid")),
                                  tuple(str(j) for j in range(hi-lo)))
                require(bytes(rows["identities"][i]).hex() == identity, "Selected proxy identity differs")
                offline = None
                if paired:
                    bucket = int(entry)//512
                    if bucket != bucket_index:
                        bucket_index = bucket
                        raw = tree.arrays(list(BRANCHES), entry_start=bucket*512,
                            entry_stop=min((bucket+1)*512, source["raw_entries"]), library="ak", how=dict)
                    local = int(entry)-bucket*512
                    cols = {f: ak.to_numpy(raw["part_"+f][local]) for f in JC2_FIELDS}
                    require(all(len(v) == int(raw["jet_nparticles"][local]) for v in cols.values()), "Offline lengths differ")
                    offline = from_columns(cols)
                yield i, identity, int(rows["labels"][i]), proxy, offline
                count += 1
    require(count == len(rows["entries"]), "Incomplete selected shard")


def shard_rows(arrays, index):
    chosen = np.flatnonzero(arrays["shards"] == index)
    return {k: v[chosen] for k, v in arrays.items()}


def assignment(spec, shard):
    role = shard["role"]
    pop, arrays = selected(spec, role)
    rows = shard_rows(arrays, shard["index"])
    root = Path(spec["campaign_root"])
    directory = root/"assignments"/str(shard["index"])
    maps, offsets, no = [], [0], []
    counts, per_class = Counter(), {str(c): Counter() for c in range(11)}
    angular, occupancy = Counter(), Counter()
    maxima = dict(proxy=0, offline=0, expanded=0)
    started = time.monotonic()
    for i, identity, label, proxy, offline in iter_selected(dataset(spec), shard, rows, paired=True):
        mapping, base = match(proxy, offline, spec["candidate"])
        kept = mapping[mapping >= 0]
        dropped = np.setdiff1d(np.arange(len(offline)), kept)
        values = dict(jets=1, proxy=len(proxy), offline=len(offline),
            cropped=len(dropped), overflow_jets=int(len(dropped)>0), fillers=max(0, 2*len(proxy)-len(offline)),
            offline_pt=float(offline.pt.sum()), cropped_pt=float(offline.pt[dropped].sum()),
            offline_salience=int(base["offline_salience"].sum()),
            cropped_salience=int(base["offline_salience"][dropped].sum()))
        counts.update(values); per_class[str(label)].update(values)
        owner, slot = np.nonzero(mapping >= 0)
        angular.update((base["qdr"][owner, mapping[owner, slot]]//100000).tolist())
        occupancy.update(np.sum(mapping >= 0, axis=1).tolist())
        for k, n in (("proxy", len(proxy)), ("offline", len(offline)), ("expanded", 3*len(proxy))):
            maxima[k] = max(maxima[k], n)
        for coordinate in STRENGTHS:
            v = build_view(proxy, coordinate=coordinate, offline=offline, identity=identity,
                candidate=spec["candidate"], mapping=mapping)
            require(len(v) == 3*len(proxy), "Fixed K2 support changed")
            for k in ("p4", "charge", "category", "tracking", "valid"):
                require(np.array_equal(getattr(v, k)[::3], getattr(proxy, k)), "Original proxy changed")
            build_inputs(v, capacity=max(16, len(v)))
        maps.append(mapping); offsets.append(offsets[-1]+len(proxy)); no.append(len(offline))
        if (i+1) % 500 == 0:
            print(f"NOISE-K2 matching shard={shard['index']} rows={i+1}/{len(rows['entries'])} seconds={time.monotonic()-started:.1f}", flush=True)
    a = dict(identities=rows["identities"], offsets=np.asarray(offsets, np.int64),
        offline_counts=np.asarray(no, np.int32), mapping=np.concatenate(maps) if maps else np.empty((0,2), np.int32))
    path = directory/"maps.npz"
    write_arrays(path, a)
    return publish(directory/"report.json", artifact("ASSIGNMENT", campaign_sha256=spec["content_hash"],
        population_sha256=pop["content_hash"], shard=shard, arrays=reference(root, path),
        counts=dict(counts), by_class={k: dict(v) for k, v in per_class.items()}, maximum_particles=maxima,
        angular_bins_width=.01, angular_bins=[[k,v] for k,v in sorted(angular.items())],
        occupancy={str(k): occupancy[k] for k in (0,1,2)}, all_coordinate_invariants=True))


def load_assignment(spec, shard):
    root = Path(spec["campaign_root"])
    pop, selection = selected(spec, shard["role"])
    rows = shard_rows(selection, shard["index"])
    r = load(root/"assignments"/str(shard["index"])/"report.json", "ASSIGNMENT")
    require(r["campaign_sha256"] == spec["content_hash"] and r["population_sha256"] == pop["content_hash"]
            and r["shard"] == shard and r["all_coordinate_invariants"] is True, "Assignment lineage differs")
    a = load_npz_arrays(checked(root, r["arrays"]))
    n = len(rows["entries"])
    require(set(a) == {"identities", "offsets", "offline_counts", "mapping"}
        and np.array_equal(a["identities"], rows["identities"])
        and a["offsets"].dtype == np.int64 and a["offsets"].shape == (n+1,)
        and a["offline_counts"].dtype == np.int32 and a["offline_counts"].shape == (n,)
        and a["offsets"][0] == 0 and a["offsets"][-1] == len(a["mapping"])
        and np.all(np.diff(a["offsets"]) > 0), "Assignment array coverage differs")
    for i in range(n):
        start, end = a["offsets"][i:i+2]
        validate_mapping(a["mapping"][start:end], nh=int(end-start), no=int(a["offline_counts"][i]))
    return r, a


def foundation(spec):
    reports = [load_assignment(spec, s)[0] for s in spec["shards"]]
    counts, classes = Counter(), {str(c): Counter() for c in range(11)}
    for r in reports:
        counts.update(r["counts"])
        for k, v in r["by_class"].items():
            classes[k].update(v)
    require(counts["jets"] == sum(spec["counts"].values()), "Foundation coverage differs")
    require(all(sum(r["counts"].get("jets",0) for r in reports if r["shard"]["role"] == role) == count
                for role,count in spec["counts"].items()), "Foundation role coverage differs")
    maxima = {k: max(r["maximum_particles"][k] for r in reports) for k in ("proxy", "offline", "expanded")}
    capacity = max(16, ((max(maxima.values())+15)//16)*16)
    ram_upper = sum(spec["counts"].values())*(capacity*84+1024)
    require(2*ram_upper < spec["resources"]["train"]["memory_mb"]*1024**2*spec["cache_fraction"],
            "Conservative RAM bound exceeds registration")
    return publish(Path(spec["campaign_root"])/"foundation.json", artifact("FOUNDATION",
        campaign_sha256=spec["content_hash"], population_sha256=load(Path(spec["campaign_root"])/"population.json", "POPULATION")["content_hash"],
        assignments={str(r["shard"]["index"]): r["content_hash"] for r in reports},
        capacity=capacity, ram_upper_bytes=ram_upper, maximum_particles=maxima,
        counts=dict(counts), by_class={k: dict(v) for k,v in classes.items()}))


def get_foundation(spec):
    value = load(Path(spec["campaign_root"])/"foundation.json", "FOUNDATION")
    require(value["campaign_sha256"] == spec["content_hash"], "Foundation parent differs")
    expected = max(16, ((max(value["maximum_particles"].values())+15)//16)*16)
    require(value["capacity"] == expected and value["counts"]["jets"] == sum(spec["counts"].values())
        and value["ram_upper_bytes"] == sum(spec["counts"].values())*(expected*84+1024)
        and set(value["assignments"]) == {str(s["index"]) for s in spec["shards"]}, "Foundation capacity/coverage differs")
    return value


def cache(spec, role, coordinate):
    ordinary(role)
    f = get_foundation(spec)
    _, selection = selected(spec, role)
    data = dataset(spec)
    blocks = []
    paired = coordinate not in ("HLT_X1", "HLT_X3", "D000")
    need_map = coordinate in ("D100", "D075", "D050", "D025")
    started = time.monotonic()
    for shard in spec["shards"]:
        if shard["role"] != role:
            continue
        rows = shard_rows(selection, shard["index"])
        if not len(rows["entries"]):
            continue
        maps = load_assignment(spec, shard)[1] if need_map else None
        offsets, features, vectors = [0], [], []
        for i, identity, label, proxy, offline in iter_selected(data, shard, rows, paired=paired):
            mapping = None
            if maps is not None:
                start, end = maps["offsets"][i:i+2]
                require(end-start == len(proxy) and maps["offline_counts"][i] == len(offline), "Map endpoint counts differ")
                mapping = maps["mapping"][start:end]
            view = build_view(proxy, coordinate=coordinate, offline=offline, identity=identity,
                candidate=spec["candidate"], mapping=mapping)
            value = build_inputs(view, capacity=f["capacity"])
            features.append(value.features); vectors.append(value.vectors); offsets.append(offsets[-1]+len(view))
        blocks.append(RamBlock(shard["index"], np.asarray(offsets, np.int64), np.concatenate(features),
            np.concatenate(vectors), rows["identities"], rows["labels"]))
        require(sum(b.nbytes for b in blocks) <= f["ram_upper_bytes"], "Measured cache exceeds bound")
    result = RamCache(blocks, role=role, foundation_sha256=f["content_hash"], coordinate_name=coordinate)
    require(np.array_equal(result.identities, selection["identities"]) and np.array_equal(result.labels, selection["labels"]),
            "Cache population/order differs")
    print(f"NOISE-K2 cache role={role} coordinate={coordinate} rows={len(result)} seconds={time.monotonic()-started:.1f}", flush=True)
    return result
