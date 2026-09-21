"""Compact K2 maps, all-row crop audits, bounded RAM caches and validation roles."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path
import time

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    sha256_file, write_immutable_json,
)
from .cache import RamBlock, RamCache, _limit_worker_threads
from .concat_k2_campaign import artifact, registration, validate
from .concat_k2_views import build_view, match_particles, validate_mapping, STRENGTHS
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import DR_QUANTUM
from .contracts import relative_file
from .inputs import build_inputs
from .provenance import source_record
from .reader import DatasetReader, Jet, Particles
from .salience_learned_data import IndexedRamCache
from .schema import CLASS_NAMES


def producer():
    return source_record(*[f"src/hlt_classification/jetclass2_delphes/{n}.py" for n in (
        "concat_k2_views", "concat_k2_data", "reader", "schema", "inputs", "salience_views",
        "views", "splits", "split_registry", "selection", "contracts")],
        "src/hlt_classification/scouting/hcwdl_fullcard_salience_matcher.py",
        "src/hlt_classification/scouting/hcwdl_fullcard_salience_contracts.py",
        "src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py",
        "src/hlt_classification/scouting/highcov_data.py",
        "src/hlt_classification/scouting/highcov_features.py")


def file_task(foundation, index):
    rows = [r for r in foundation["assignment_tasks"] if r["file_index"] == index]
    if len(rows) != 1 or rows[0]["role"] not in {"train", "validation"}:
        raise PermissionError("Only registered ordinary-role files can be read")
    return rows[0]


def assignment(spec, index, directory):
    f = spec["foundation"]
    task = file_task(f, index)
    directory, started = Path(directory), time.monotonic()
    ids, offsets, maps, offline_counts = [], [0], [], []
    counts = Counter()
    per_class = {name: Counter() for name in CLASS_NAMES}
    qdr_counts, ratios, occupancies, distant = Counter(), Counter(), Counter(), Counter()
    maximum_dr=0.
    lost_pt_bins, lost_salience_bins = Counter(), Counter()
    maxima = dict(hlt=0, offline=0, x3=0)
    weighted_dr, weight_total = 0., 0.
    scalar_pt, lost_pt, salience, lost_salience = 0., 0., 0, 0
    for number, jet in enumerate(DatasetReader(Path(spec["data_root"]), f["inventory"], f["splits"],
            role=task["role"], include_offline=True, file_paths=(task["path"],)), 1):
        mapping, full = match_particles(jet.hlt, jet.offline, f["candidate"], diagnostics=True)
        retained = mapping[mapping >= 0]
        discarded = np.setdiff1d(np.arange(len(jet.offline)), retained)
        nh, no = len(jet.hlt), len(jet.offline)
        values = dict(jets=1, hlt=nh, offline=no, retained=len(retained),
                      cropped=len(discarded), overflow_jets=int(no > 2*nh), fillers=max(0, 2*nh-no), x3=3*nh)
        counts.update(values)
        per_class[CLASS_NAMES[jet.label]].update(values)
        ratios[(nh,no)] += 1
        occupancies.update(map(int,np.sum(mapping>=0,axis=1)))
        for key,value in (("hlt",nh),("offline",no),("x3",3*nh)):
            maxima[key]=max(maxima[key],value)
        pt = np.hypot(jet.offline.p4[:,0].astype(np.float64), jet.offline.p4[:,1].astype(np.float64))
        scalar_pt += float(pt.sum()); lost_pt += float(pt[discarded].sum())
        salience += int(full["offline_salience"].sum()); lost_salience += int(full["offline_salience"][discarded].sum())
        per_class[CLASS_NAMES[jet.label]].update(dict(offline_scalar_pt=float(pt.sum()),
            cropped_scalar_pt=float(pt[discarded].sum()), offline_salience=int(full["offline_salience"].sum()),
            cropped_salience=int(full["offline_salience"][discarded].sum())))
        lost_pt_bins[int(np.floor(1000*float(pt[discarded].sum())/float(pt.sum())))]+=1
        lost_salience_bins[int(1000*int(full["offline_salience"][discarded].sum())//int(full["offline_salience"].sum()))]+=1
        owners, slots = np.nonzero(mapping >= 0)
        chosen=mapping[owners,slots]
        distances=full["qdr"][owners,chosen].astype(np.float64)*DR_QUANTUM
        maximum_dr=max(maximum_dr,float(distances.max()))
        for threshold in (.1,.2,.3,.5,1.):
            distant[str(threshold)]+=int(np.count_nonzero(full["qdr"][owners,chosen]>round(threshold/DR_QUANTUM)))
        weights=full["hlt_salience"][owners].astype(np.float64)+full["offline_salience"][chosen].astype(np.float64)
        weighted_dr+=float(np.dot(weights,distances)); weight_total+=float(weights.sum())
        # Fixed .01 delta-R bins bound report size; never censor the overflow tail.
        qdr_counts.update(map(int,np.floor(distances*100)))
        # All endpoints/intermediates must keep support and valid raw features.
        for coord in STRENGTHS:
            view = build_view(jet, coord, f["candidate"], mapping)
            if len(view) != 3*nh or not np.array_equal(view.values[::3], jet.hlt.values):
                raise ValueError("Fixed support/original HLT invariant differs")
            build_inputs(view, capacity=f["inputs"]["capacity"])
        ids.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
        maps.append(mapping); offsets.append(offsets[-1]+nh); offline_counts.append(no)
        if number % 1000 == 0:
            print(f"JC2-K2 phase=assign file={index} rows={number}/{task['rows']} seconds={time.monotonic()-started:.1f}", flush=True)
    if len(ids) != task["rows"]:
        raise ValueError("Assignment population differs")
    arrays = dict(identities=np.asarray(ids, np.uint8).reshape(-1,32), offsets=np.asarray(offsets,np.int64),
                  mapping=np.concatenate(maps) if maps else np.empty((0,2),np.int32),
                  offline_counts=np.asarray(offline_counts,np.int32))
    path = directory/"assignments.npz"
    atomic_publish_bytes(path, deterministic_npz_bytes(arrays))
    report = artifact("ASSIGNMENT_SHARD", foundation_sha256=f["content_hash"], file_task=task,
        views_sha256=f["views"]["content_hash"], array_sha256=sha256_file(path), array_bytes=path.stat().st_size,
        producer=producer(), counts=dict(counts), by_class={k:dict(v) for k,v in per_class.items()},
        delta_r_histogram_step=.01, delta_r_histogram=[[k,v] for k,v in sorted(qdr_counts.items())],
        native_count_joint_histogram=[[h,o,n] for (h,o),n in sorted(ratios.items())],
        owner_occupancy={str(k):occupancies[k] for k in (0,1,2)}, maximum_particles=maxima,
        crop_fraction_histogram_step=.001, cropped_pt_fraction_histogram=[list(r) for r in sorted(lost_pt_bins.items())],
        cropped_salience_fraction_histogram=[list(r) for r in sorted(lost_salience_bins.items())],
        pair_salience_weighted_delta_r_sum=weighted_dr, pair_salience_weight_sum=weight_total,
        selected_delta_r_gt=dict(distant), maximum_delta_r=maximum_dr,
        offline_scalar_pt=scalar_pt, cropped_scalar_pt=lost_pt,
        offline_salience=salience, cropped_salience=lost_salience,
        interpolation_invariants_all_rows=True, elapsed_seconds=time.monotonic()-started, final_test_accessed=False)
    write_immutable_json(directory/"assignment_report.json", report)
    return report


def load_assignment(spec, index):
    f = spec["foundation"]
    task = file_task(f, index)
    directory = Path(spec["campaign_root"])/"outputs"/f"assign_{index:04d}"
    report = load_json(directory/"assignment_report.json")
    validate(report,"ASSIGNMENT_SHARD")
    path = directory/"assignments.npz"
    if (report["foundation_sha256"] != f["content_hash"] or report["file_task"] != task
            or report["views_sha256"] != f["views"]["content_hash"] or report["final_test_accessed"] is not False
            or report["producer"]["file_sha256"] != producer()["file_sha256"]
            or path.stat().st_size != report["array_bytes"] or sha256_file(path) != report["array_sha256"]):
        raise ValueError("K2 assignment lineage/source/bytes differ")
    a = load_npz_arrays(path)
    if set(a) != {"identities","offsets","mapping","offline_counts"}:
        raise ValueError("K2 assignment layout differs")
    ids, off, m, no = (a[k] for k in ("identities","offsets","mapping","offline_counts"))
    n = task["rows"]
    if (ids.dtype != np.uint8 or ids.shape != (n,32) or len(np.unique(ids,axis=0)) != n
            or off.dtype != np.int64 or off.shape != (n+1,) or off[0] != 0 or off[-1] != len(m)
            or np.any(np.diff(off) < 1) or m.dtype != np.int32 or m.shape != (int(off[-1]),2)
            or no.dtype != np.int32 or no.shape != (n,) or np.any(no < 1)):
        raise ValueError("K2 compact array type/coverage differs")
    for j,(start,end) in enumerate(zip(off[:-1],off[1:])):
        validate_mapping(m[start:end], nh=int(end-start), no=int(no[j]))
    return report,a


def foundation_lock(spec):
    reports = [load_assignment(spec,t["file_index"])[0] for t in spec["foundation"]["assignment_tasks"]]
    totals=Counter()
    roles={r:Counter() for r in ("train","validation")}
    classes={name:Counter() for name in CLASS_NAMES}
    for report in reports:
        totals.update(report["counts"])
        roles[report["file_task"]["role"]].update(report["counts"])
        for name,values in report["by_class"].items(): classes[name].update(values)
    if totals["jets"] != spec["role_counts"]["train"]+spec["role_counts"]["validation"]:
        raise ValueError("Foundation row coverage differs")
    if any(roles[r]["jets"]!=spec["role_counts"][r] for r in roles):
        raise ValueError("Foundation per-role row coverage differs")
    return artifact("FOUNDATION_LOCK", foundation_sha256=spec["foundation"]["content_hash"],
        shards={str(r["file_task"]["file_index"]):r["content_hash"] for r in reports}, counts=dict(totals),
        by_role={k:dict(v) for k,v in roles.items()}, by_class={k:dict(v) for k,v in classes.items()},
        maximum_particles={k:max(r["maximum_particles"][k] for r in reports) for k in ("hlt","offline","x3")},
        pair_salience_weighted_delta_r_sum=sum(r["pair_salience_weighted_delta_r_sum"] for r in reports),
        pair_salience_weight_sum=sum(r["pair_salience_weight_sum"] for r in reports),
        selected_delta_r_gt={str(t):sum(r["selected_delta_r_gt"][str(t)] for r in reports) for t in (.1,.2,.3,.5,1.)},
        maximum_delta_r=max(r["maximum_delta_r"] for r in reports),
        cropped_scalar_pt=sum(r["cropped_scalar_pt"] for r in reports),
        offline_scalar_pt=sum(r["offline_scalar_pt"] for r in reports),
        cropped_salience=sum(r["cropped_salience"] for r in reports),
        offline_salience=sum(r["offline_salience"] for r in reports), final_test_accessed=False)


def cache_bounds(spec):
    f=spec["foundation"]; workers=spec["resources"]["train"]["cpus"]
    result={}
    for role in ("train","validation"):
        sizes=[]
        for task in f["assignment_tasks"]:
            if task["role"] != role: continue
            native=f["inventory"]["files"][task["file_index"]]["max_selected_particles"]
            sizes.append(task["rows"]*(max(native["offline"],3*native["hlt"])*84+1024))
        result[role]=sum(sizes)+4*sum(sorted(sizes,reverse=True)[:workers])
    if sum(result.values()) > spec["resources"]["train"]["memory_mb"]*1024**2*spec["cache_fraction_limit"]:
        raise MemoryError(f"Expanded K2 RAM bound exceeds registration: {result}")
    return result


def _file_cache(args):
    f, data_root, campaign_root, task, coordinate = args
    privileged=coordinate not in {"HLT_X1","HLT_X3","D000"}
    maps = (load_assignment(dict(foundation=f,campaign_root=campaign_root),task["file_index"])[1]
            if coordinate in {"D100","D075","D050","D025"} else None)
    offsets, features, vectors, ids, labels=[0],[],[],[],[]
    for row,jet in enumerate(DatasetReader(Path(data_root), f["inventory"], f["splits"], role=task["role"],
            include_offline=privileged, file_paths=(task["path"],))):
        identity=np.frombuffer(bytes.fromhex(jet.identity),np.uint8)
        mapping=None
        if maps is not None:
            if not np.array_equal(identity,maps["identities"][row]) or len(jet.offline) != maps["offline_counts"][row]:
                raise ValueError("K2 map/native particle identity join differs")
            start,end=maps["offsets"][row:row+2]; mapping=maps["mapping"][start:end]
        view=build_view(jet,coordinate,f["candidate"],mapping)
        value=build_inputs(view,capacity=f["inputs"]["capacity"])
        length=len(view)
        features.append(value.features[:,:length].T.copy()); vectors.append(value.vectors[:,:length].T.copy())
        offsets.append(offsets[-1]+length); ids.append(identity); labels.append(jet.label)
    if len(ids) != task["rows"]: raise ValueError("RAM block coverage differs")
    return RamBlock(task["file_index"],np.asarray(offsets,np.int64),np.concatenate(features),np.concatenate(vectors),
                    np.asarray(ids,np.uint8).reshape(-1,32),np.asarray(labels,np.int64))


def prepare(spec,role,coordinate):
    if role not in {"train","validation"}: raise PermissionError("Final test is sealed")
    if coordinate not in {*STRENGTHS,"HLT_X1","HLT_X3","OFFLINE"}: raise ValueError("Unknown K2 view")
    limit=cache_bounds(spec)[role]; f=spec["foundation"]
    tasks=[t for t in f["assignment_tasks"] if t["role"]==role]
    args=[(f,spec["data_root"],spec["campaign_root"],t,coordinate) for t in tasks]
    blocks=[]; resident=0; started=time.monotonic(); workers=spec["resources"]["train"]["cpus"]
    def accept(block):
        nonlocal resident
        resident+=block.nbytes+len(block.labels)*40
        if resident>limit: raise MemoryError("K2 RAM ceiling exceeded; no disk spill")
        blocks.append(block)
        print(f"JC2-K2 phase=cache role={role} view={coordinate} files={len(blocks)}/{len(tasks)} seconds={time.monotonic()-started:.1f}",flush=True)
    if workers==1:
        for arg in args: accept(_file_cache(arg))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=_limit_worker_threads) as pool:
            iterator=iter(args)
            pending=[pool.submit(_file_cache,a) for a in (next(iterator,None) for _ in range(workers)) if a is not None]
            while pending:
                accept(pending.pop(0).result()); a=next(iterator,None)
                if a is not None: pending.append(pool.submit(_file_cache,a))
    result=RamCache(blocks,role=role,foundation_sha256=f["content_hash"],coordinate_name=coordinate)
    if len(result)!=f["splits"]["role_counts"][role]: raise ValueError("Cache population differs")
    return result


def partition_codes(identities, labels):
    if (identities.dtype!=np.uint8 or identities.shape!=(len(labels),32)
            or len(np.unique(identities,axis=0))!=len(labels) or np.any((labels<0)|(labels>=11))):
        raise ValueError("Partition identity/class population differs")
    codes=np.empty(len(labels),np.uint8); prefix=registration()["validation_partition"]["domain"].encode()
    for c in range(11):
        rows=np.flatnonzero(labels==c).tolist()
        if len(rows)<4: raise ValueError("Each validation class needs at least four rows")
        rows.sort(key=lambda i:hashlib.sha256(prefix+bytes(identities[i])).digest())
        for offset,i in enumerate(rows): codes[i]=(0,0,1,2)[offset%4]
    return codes


def publish_partition(spec,validation):
    root=Path(spec["campaign_root"]); path=root/"validation_partition.npz"
    a=dict(identities=validation.identities,labels=validation.labels,codes=partition_codes(validation.identities,validation.labels))
    atomic_publish_bytes(path,deterministic_npz_bytes(a))
    report=artifact("VALIDATION_PARTITION",campaign_sha256=spec["content_hash"],protocol=spec["validation_partition"],
        rows=len(validation),data_path=path.name,data_sha256=sha256_file(path),
        counts=[int(np.sum(a["codes"]==i)) for i in range(3)],final_test_accessed=False)
    write_immutable_json(root/"validation_partition.json",report)
    return report


def partition_indices(spec,validation):
    root=Path(spec["campaign_root"]); report=load_json(root/"validation_partition.json")
    validate(report,"VALIDATION_PARTITION"); path=relative_file(root,report["data_path"])
    if (report["campaign_sha256"]!=spec["content_hash"] or report["protocol"]!=spec["validation_partition"]
            or sha256_file(path)!=report["data_sha256"] or report["final_test_accessed"]):
        raise ValueError("Validation partition lineage differs")
    a=load_npz_arrays(path)
    if (not np.array_equal(a["identities"],validation.identities) or not np.array_equal(a["labels"],validation.labels)
            or not np.array_equal(a["codes"],partition_codes(validation.identities,validation.labels))):
        raise ValueError("Partition rows/codes differ")
    return {name:np.flatnonzero(a["codes"]==i) for i,name in enumerate(("checkpoint","diagnostic","report"))}


def caches(spec,node,*,train_only=False):
    result={"train":prepare(spec,"train",node["primary_coordinate"])}
    if not train_only:
        validation=prepare(spec,"validation",node["primary_coordinate"])
        result.update({name:IndexedRamCache(validation,index,role="validation")
                       for name,index in partition_indices(spec,validation).items()})
    return result


def matcher_acceptance(spec):
    # Small bounded exhaustive cases; never enumerate production multiplicity.
    rng=np.random.default_rng(20260921); checked=0
    for nh,no in ((1,1),(1,4),(2,2),(2,3),(2,5),(3,3),(3,4),(4,3)):
        for _ in range(3):
            def particles(n):
                p=np.zeros((n,14),np.float32); p[:,:3]=rng.normal(size=(n,3)); p[:,0]+=4
                p[:,3]=np.sqrt((p[:,:3]**2).sum(1))+1; p[:,4:6]=1
                return Particles(p)
            h,o=particles(nh),particles(no)
            prod=match_particles(h,o,spec["foundation"]["candidate"])
            ref=match_particles(h,o,spec["foundation"]["candidate"],reference=True)
            if not np.array_equal(prod,ref): raise ValueError("Capacity-two solver differs from exhaustive reference")
            jet=Jet("a"*64,0,h,o)
            for coordinate in STRENGTHS:
                value=build_view(jet,coordinate,spec["foundation"]["candidate"],prod)
                if not np.array_equal(value.values[::3],h.values): raise ValueError("Original HLT changed")
                build_inputs(value,capacity=max(16,3*nh))
            endpoint=build_view(Jet(jet.identity,jet.label,h,None),"D000",None)
            if not np.array_equal(endpoint.values,np.repeat(h.values,3,axis=0)): raise ValueError("HLT-only endpoint differs")
            checked+=1
    return artifact("MATCHER_ACCEPTANCE",foundation_sha256=spec["foundation"]["content_hash"],
        exhaustive_cases=checked, maximum_reference_side=8, endpoint_offline_free=True,
        all_registered_coordinates=True, acceptance_only=True, final_test_accessed=False)
