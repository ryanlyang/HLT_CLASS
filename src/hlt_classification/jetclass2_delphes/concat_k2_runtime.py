"""One-encoder K2 training and expanded-input SPORC A100 acceptance."""
from __future__ import annotations

from io import BytesIO
import gc
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from .banks import load_bank, publish_bank
from .contracts import relative_file
from .concat_k2_campaign import gates, artifact, nodes, validate, validate_campaign
from .concat_k2_data import caches, cache_bounds, prepare, publish_partition, assignment, foundation_lock, matcher_acceptance
from .concat_k2_source import validate_import
from .execution import allocation, gpu_identity
from .concat_k2_execution import runtime_site, validate_acceptance_site
from .model import installed_environment
from .concat_k2_model import (K2ParticleTransformer, storage_parity, synchronize,
    validate_storage_stats, PAIR_STORAGE, BATCH_PROBE_POLICY, PARITY_CHECKS)
from .reporting import evaluate_probabilities, recovery
from .salience_learned_data import IndexedRamCache
from .salience_learned_training import predict, train_kernel, _optimizer, _train_batch


def task(spec, name):
    found = [r for r in spec["tasks"] if r["task_id"] == name]
    if len(found) != 1:
        raise ValueError("Unknown fusion-chain task")
    return found[0]


def completed(spec, name):
    row = task(spec, name)
    path = Path(spec["campaign_root"]) / "tasks" / (name + ".json")
    if not path.is_file():
        return None
    report = load_json(path)
    validate(report, "TASK_REPORT")
    if (report["campaign_sha256"] != spec["content_hash"] or report["task_id"] != name
            or report["source_commit"] != spec["source_commit"] or report["final_test_accessed"]
            or set(report["parents"]) != set(row["dependencies"]) or not report["outputs"]):
        raise ValueError("Task report lineage differs")
    root = Path(spec["campaign_root"])
    for parent, digest in report["parents"].items():
        parent_report = load_json(root / "tasks" / (parent + ".json"))
        validate(parent_report, "TASK_REPORT")
        if parent_report["content_hash"] != digest or parent_report["campaign_sha256"] != spec["content_hash"]:
            raise ValueError("Task dependency attestation differs")
    for output in report["outputs"]:
        if sha256_file(relative_file(root, output["path"])) != output["sha256"]:
            raise ValueError("Task output bytes changed")
    return report


def science_gate(spec):
    reports = {name: completed(spec, name) for name in gates(spec)}
    if not all(reports.values()):
        raise PermissionError("Fresh K2 preparation (or verified import) and GPU acceptance must complete before science")
    acceptance = load_json(relative_file(Path(spec["campaign_root"]), reports["preflight"]["result"]["acceptance"]))
    validate(acceptance, "ACCEPTANCE")
    validate_acceptance_site(spec, acceptance)
    validate_memory_evidence(spec, acceptance)
    if (acceptance["campaign_sha256"] != spec["content_hash"] or acceptance["passed"] is not True
            or acceptance["final_test_accessed"] is not False
            or acceptance["resource"] != spec["resources"]["preflight"] or acceptance["acceptance_only"] is not True
            or acceptance["ordinary_rows"] != {r: spec["role_counts"][r] for r in ("train", "validation")}
            or acceptance["batch_size"] != spec["training"]["batch_size"]
            or acceptance["capacity"] != spec["foundation"]["inputs"]["capacity"]
            or not acceptance["checkpoint_round_trip"] or not acceptance["bank_round_trip"]
            or not acceptance["installed_weaver_fp32_parity"] or not acceptance["worst_population_batch_stress"]
            or not acceptance["hlt_only_endpoint"] or not acceptance["duplicate_pair_finiteness"]
            or not 0 < acceptance["peak_cuda_bytes"] <= acceptance["gpu"]["total_memory_bytes"] * spec["gpu_peak_fraction_limit"]
            or not 0 < acceptance["peak_rss_bytes"] <= spec["resources"]["train"]["memory_mb"] * 1024**2 * spec["cpu_peak_fraction_limit"]
            or not 0 < acceptance["projected_max_fit_seconds"] <= 23 * 3600
            or len(acceptance["native_execution"]) != 4
            or any(not row["kernel_report"]["acceptance_only"] or row["kernel_report"]["scientific_fit"]
                   for row in acceptance["native_execution"])):
        raise ValueError("Fresh K2 execution acceptance differs")
    return acceptance


def execution_gate(spec, *, science):
    job, cpus, memory = allocation(runtime_site(spec))
    resource = spec["resources"]["train"]
    if cpus != resource["cpus"] or memory != resource["memory_mb"]:
        raise ValueError("Worker CPU/RAM allocation differs")
    if science:
        acceptance = science_gate(spec)
        if acceptance["gpu"] != gpu_identity() or acceptance["environment"] != installed_environment():
            raise ValueError("Worker GPU/software differs from accepted execution")
    return job


def new_model(node):
    torch.manual_seed(node["initialization_seed"])
    return K2ParticleTransformer()


def save_state(path, state):
    buffer = BytesIO()
    torch.save(state, buffer)
    atomic_publish_bytes(path, buffer.getvalue())


def teacher(spec, node, identities):
    name = node["teacher_distribution"]
    if name is None:
        return None, None
    parent = completed(spec, "reduce_" + name)
    if parent is None:
        raise ValueError("Teacher reducer is not complete")
    result = parent["result"]
    values = load_bank(relative_file(Path(spec["campaign_root"]), result["bank"]),
        foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=result["training_report_sha256"], teacher_node=name,
        role="train", expected_identities=identities)
    return values, dict(task_sha256=parent["content_hash"], teacher_node=name,
                        bank_manifest_sha256=result["bank_manifest_sha256"])


def fit(spec, row, directory, device):
    node = next(n for n in spec["nodes"] if n["node_id"] == row["node_id"])
    values = caches(spec, node)
    model = new_model(node)
    q, lineage = teacher(spec, node, values["train"].identities)
    report, state = train_kernel(model, lambda _: values["train"], lambda _: values["checkpoint"],
        node=node, device=device, teacher_probabilities=q,
        teacher_identities=None if q is None else values["train"].identities)
    probabilities = predict(model, values["report"], node=node, device=device)
    path = directory / "selected.pt"
    save_state(path, state)
    outer = artifact("TRAINING_REPORT", campaign_sha256=spec["content_hash"], node=node,
        kernel_report=report, teacher_lineage=lineage, selected_checkpoint_sha256=sha256_file(path),
        checkpoint_validation=report["validation"], report_validation=evaluate_probabilities(values["report"].labels, probabilities),
        validation_partition_sha256=load_json(Path(spec["campaign_root"]) / "validation_partition.json")["content_hash"],
        validation_report_not_final_test=True, matching_selection_used_validation=True,
        final_test_accessed=False)
    write_immutable_json(directory / "training_report.json", outer)
    if node["deployable"]:
        from .model import model_contract
        write_immutable_json(directory/"deployment.json",artifact("DEPLOYMENT_EXPORT",
            campaign_sha256=spec["content_hash"], training_report_sha256=outer["content_hash"],
            checkpoint="selected.pt", checkpoint_sha256=sha256_file(path),model=model_contract(),
            input_features=spec["foundation"]["inputs"], native_hlt_copies=1 if node["primary_coordinate"]=="HLT_X1" else 3,
            preprocessing="concat_k2_views.deployment_inputs", offline_inputs=False,
            assignment_inputs=False, source_or_slot_embedding=False, final_test_accessed=False))
    return dict(checkpoint=path, training_report=directory / "training_report.json",
                training_report_sha256=outer["content_hash"])


def reduce(spec, row, directory, device):
    node = next(n for n in spec["nodes"] if n["node_id"] == row["node_id"])
    parent = completed(spec, "train_" + node["node_id"])
    if parent is None:
        raise ValueError("Reducer lacks a scientific fit")
    root = Path(spec["campaign_root"])
    report = load_json(relative_file(root, parent["result"]["training_report"]))
    validate(report, "TRAINING_REPORT")
    if not report["kernel_report"]["scientific_fit"] or report["kernel_report"]["acceptance_only"]:
        raise ValueError("Acceptance weights cannot be a science teacher")
    model = new_model(node)
    model.load_state_dict(torch.load(relative_file(root, parent["result"]["checkpoint"]),
                         map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    values = caches(spec, node, train_only=True)["train"]
    probabilities = predict(model, values, node=node, device=device, temperature=2.)
    bank = directory / "train_bank"
    manifest = publish_bank(bank, foundation_sha256=spec["foundation"]["content_hash"],
        teacher_report_sha256=report["content_hash"], teacher_node=node["node_id"],
        role="train", identities=values.identities, probabilities=probabilities)
    return dict(bank=bank, bank_manifest_sha256=manifest["content_hash"], training_report_sha256=report["content_hash"])


def result_rows(spec):
    reports = {}
    for node in spec["nodes"]:
        done = completed(spec, "train_" + node["node_id"])
        if done:
            value = load_json(relative_file(Path(spec["campaign_root"]), done["result"]["training_report"]))
            validate(value, "TRAINING_REPORT")
            reports[node["node_id"]] = value
    rows = []
    for node in spec["nodes"]:
        name = node["node_id"]
        value = reports.get(name)
        metrics = None if value is None else value["report_validation"]
        recovered = (None if value is None or not {"HLT_X1_CE", "OFFLINE_CE"} <= reports.keys() else
                     recovery(metrics, reports["HLT_X1_CE"]["report_validation"], reports["OFFLINE_CE"]["report_validation"]))
        rows.append(dict(node_id=name, deployable=node["deployable"], validation=metrics, recovery=recovered,
                         selected_pass=None if value is None else value["kernel_report"]["selected_pass"],
                         passes=None if value is None else value["kernel_report"]["passes"]))
    return rows


def representative(cache, size):
    # Include every class, then spread across all registered ordinary files.
    # This is execution sampling, never a scored scientific subset.
    selected = []
    for c in range(11):
        selected.extend(np.flatnonzero(cache.labels == c)[:min(16, size // 11)].tolist())
    used = set(selected)
    for i in np.linspace(0,len(cache)-1,min(2*size,len(cache)),dtype=np.int64):
        if len(selected) >= size:
            break
        if i not in used:
            selected.append(i)
            used.add(i)
    return np.asarray(selected[:size], np.int64)


def longest_indices(cache, size=256):
    lengths = np.concatenate([np.diff(block.offsets) for block in cache.blocks])
    if len(lengths) != len(cache) or np.any(lengths < 1):
        raise ValueError("Invalid ragged cache lengths")
    # Masked padding alone does not stress valid-pair intermediate buffers.
    return np.lexsort((np.arange(len(cache)), lengths))[-size:].astype(np.int64)


def _cuda_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stress(model, raw, node, device, capacity):
    raw=dict(raw)
    for key in ("features","vectors","mask"):
        value=raw[key]
        if value.shape[-1]>capacity: raise ValueError("Input exceeds expanded capacity")
        raw[key]=np.pad(value,((0,0),(0,0),(0,capacity-value.shape[-1])))
    optimizer=_optimizer(model); model.train()
    q=None if node["teacher_distribution"] is None else torch.full((len(raw["labels"]),11),1/11,device=device)
    timings=[]
    for step in range(3):
        synchronize(device); started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss,_=_train_batch(model,raw,node=node,device=device,teacher=q,alpha=1.)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite longest-batch gradient")
        optimizer.step()
        synchronize(device); timings.append(time.monotonic()-started)
        print(f"JC2-K2 phase=batch_probe view={node['primary_coordinate']} batch={len(raw['labels'])} step={step+1}/3 seconds={timings[-1]:.3f}",flush=True)
    optimizer.zero_grad(set_to_none=True)
    stats=model.pair_storage_stats() if hasattr(model,"pair_storage_stats") else None
    if torch.device(device).type=="cuda": validate_storage_stats(stats)
    return dict(step_seconds=timings,storage_stats=stats)


PROBE_CASES=("CONCAT_K2_D100","CONCAT_K2_D075","CONCAT_K2_D000","HLT_X1_COMPRESSED")


def _exercise_probe(spec,node,train,validation,device,batch_size):
    # A new optimizer/BN state for each probe; these weights never become teachers.
    model=new_model(node).to(device)
    ti=longest_indices(train,batch_size); vi=longest_indices(validation,batch_size)
    if len(ti)!=batch_size or len(vi)!=batch_size:
        raise ValueError("Batch probe needs the requested number of distinct real rows")
    result=_stress(model,train.batch(ti),node,device,spec["foundation"]["inputs"]["capacity"])
    synchronize(device); started=time.monotonic()
    predict(model,IndexedRamCache(validation,vi,role="validation"),node=node,device=device,batch_size=batch_size)
    synchronize(device)
    return dict(**result,validation_seconds=time.monotonic()-started,
        train_rows=len(ti),validation_rows=len(vi),
        steady_train_jets_per_second=2*batch_size/sum(result["step_seconds"][1:]))


def _probe_attempt(spec,node,train,validation,device,batch_size):
    _cuda_clear(); torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try:
        result=dict(status="COMPLETED",**_exercise_probe(spec,node,train,validation,device,batch_size))
    except torch.OutOfMemoryError as error:
        # Exit the except block before cleanup: do not retain its CUDA traceback.
        result=dict(status="CUDA_OOM",error=str(error),error_type=type(error).__name__)
    result.update(elapsed_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved())
    _cuda_clear()
    return result


def batch_probes(spec,directory,node,train,validation,device):
    records=[]
    for size in BATCH_PROBE_POLICY["order"]:
        result=_probe_attempt(spec,node,train,validation,device,size)
        record=artifact("BATCH_PROBE",**result,node_id=node["node_id"],
            campaign_sha256=spec["content_hash"],device_type=torch.device(device).type,
            batch_size=size,capacity=spec["foundation"]["inputs"]["capacity"],
            pair_storage=PAIR_STORAGE,policy=BATCH_PROBE_POLICY,acceptance_only=True,final_test_accessed=False)
        write_immutable_json(directory/f"batch_probe_{node['node_id']}_{size}.json",record)
        records.append(record)
        print(f"JC2-K2 phase=batch_probe_result view={node['primary_coordinate']} batch={size} status={record['status']} peak_cuda_GiB={record['peak_cuda_bytes']/2**30:.3f}",flush=True)
        if record["status"]!="COMPLETED":
            raise MemoryError(f"K2 batch {size} CUDA OOM. Probe evidence saved; no acceptance or science submission. "
                              "Batch 128 is diagnostic only; production remains 256 until explicitly re-registered.")
    return records


def validate_memory_evidence(spec,value):
    if (value.get("pair_storage")!=PAIR_STORAGE or value.get("batch_probe_policy")!=BATCH_PROBE_POLICY
            or spec.get("pair_storage")!=PAIR_STORAGE or spec.get("batch_probe_policy")!=BATCH_PROBE_POLICY):
        raise ValueError("K2 memory acceptance policy differs")
    reports=value.get("storage_parity_reports",[])
    expected=[(name,precision) for name in ("CONCAT_K2_D100","CONCAT_K2_D000") for precision in ("fp32","bf16")]
    if [(r.get("node_id"),r.get("precision")) for r in reports]!=expected:
        raise ValueError("K2 memory acceptance parity coverage differs")
    for row in reports:
        if (row.get("passed") is not True or row.get("device_type")!="cuda"
                or row.get("steps")!=3 or row.get("checks")!=PARITY_CHECKS):
            raise ValueError("K2 memory acceptance needs real native training parity")
        validate_storage_stats(row.get("storage_stats"))
    probes=value.get("batch_probes",[])
    if [(r.get("node_id"),r.get("batch_size")) for r in probes]!=[
            ("ACCEPTANCE_"+name,size) for name in PROBE_CASES for size in (128,256)]:
        raise ValueError("K2 memory acceptance batch coverage/order differs")
    for row in probes:
        validate(row,"BATCH_PROBE")
        if (row.get("campaign_sha256")!=spec["content_hash"] or row.get("status")!="COMPLETED"
                or row.get("device_type")!="cuda" or row.get("capacity")!=value["capacity"]
                or row.get("pair_storage")!=PAIR_STORAGE or row.get("policy")!=BATCH_PROBE_POLICY
                or row.get("acceptance_only") is not True or row.get("final_test_accessed") is not False
                or row.get("train_rows")!=row["batch_size"] or row.get("validation_rows")!=row["batch_size"]
                or not 0<row.get("peak_cuda_bytes",0)<=value["peak_cuda_bytes"]
                or len(row.get("step_seconds",[]))!=3
                or any(not math.isfinite(t) or t<=0 for t in row["step_seconds"]+[row.get("validation_seconds",0)])):
            raise ValueError("K2 memory acceptance batch execution differs")
        validate_storage_stats(row.get("storage_stats"))


def preflight(spec,directory,device):
    if str(device) not in {"cuda","cuda:0"}:
        raise PermissionError("Real installed-Weaver A100 acceptance is mandatory")
    import resource
    from .acceptance import installed_parity
    job=execution_gate(spec,science=False); environment=installed_environment(); started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    evidence=[]; q=None; cache_seconds=0.; prior_ids=None; parity=[]; storage_reports=[]; probes=[]
    peak=reserved=0
    for name in PROBE_CASES:
        node=dict(next(n for n in nodes() if n["node_id"]==name)); node["node_id"]="ACCEPTANCE_"+name
        cached=time.monotonic()
        train=prepare(spec,"train",node["primary_coordinate"])
        validation=prepare(spec,"validation",node["primary_coordinate"])
        cache_seconds+=time.monotonic()-cached
        ti=representative(train,2048); vi=representative(validation,1024)
        t=IndexedRamCache(train,ti,role="train"); v=IndexedRamCache(validation,vi,role="validation")
        if prior_ids is not None and not np.array_equal(prior_ids,t.identities):
            raise ValueError("Acceptance coordinate changed teacher row identities")
        if name in {"CONCAT_K2_D100","CONCAT_K2_D000"}:
            # FP32, eval mode: actual native factory versus repository wrapper,
            # including duplicate-p4 pairs on the x3 endpoint.
            parity.append(installed_parity(t,device=device))
            _cuda_clear()
            for bf16 in (False,True):
                storage_reports.append(dict(node_id=name,**storage_parity(t.batch(np.arange(4)),device=device,bf16=bf16)))
                _cuda_clear()
        peak=max(peak,torch.cuda.max_memory_allocated()); reserved=max(reserved,torch.cuda.max_memory_reserved())
        # Try 128 before the first 256 mini-fit/stress. A failure preserves the
        # successful smaller probe, but cannot release any scientific jobs.
        probes.extend(batch_probes(spec,directory,node,train,validation,device))
        peak=max(peak,*(r["peak_cuda_bytes"] for r in probes))
        reserved=max(reserved,*(r["peak_reserved_cuda_bytes"] for r in probes))
        model=new_model(node)
        report,state=train_kernel(model,lambda _:t,lambda _:v,node=node,device=device,
            teacher_probabilities=q,teacher_identities=None if q is None else t.identities,acceptance_passes=2)
        expected=predict(model,v,node=node,device=device)
        buffer=BytesIO(); torch.save(state,buffer); buffer.seek(0)
        model.load_state_dict(torch.load(buffer,map_location="cpu",weights_only=True),strict=True)
        if not np.array_equal(expected,predict(model,v,node=node,device=device)):
            raise ValueError("Checkpoint round trip differs")
        q=predict(model,t,node=node,device=device,temperature=2.); prior_ids=t.identities.copy()
        bank=directory/(name+"_acceptance_bank")
        kwargs=dict(foundation_sha256=t.foundation_sha256,teacher_report_sha256=report["content_hash"],
                    teacher_node=node["node_id"],role="train")
        manifest=publish_bank(bank,identities=t.identities,probabilities=q,**kwargs)
        if not np.array_equal(q,load_bank(bank,expected_identities=t.identities,**kwargs)):
            raise ValueError("Probability bank round trip differs")
        epoch=report["validation_history"][-1]
        evidence.append(dict(node=node,kernel_report=report,bank_sha256=manifest["content_hash"],
            train_seconds_per_row=epoch["train_seconds"]/len(t),validation_seconds_per_row=epoch["validation_seconds"]/len(v)))
        print(f"JC2-K2 phase=stress view={node['primary_coordinate']} peak_cuda_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}",flush=True)
        del train,validation,t,v,model,state,buffer,expected,report
        _cuda_clear()
    gpu=gpu_identity(); peak=max(peak,torch.cuda.max_memory_allocated()); reserved=max(reserved,torch.cuda.max_memory_reserved())
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    checkpoint_rows=load_json(Path(spec["campaign_root"])/"validation_partition.json")["counts"][0]
    projected=max(100*(spec["role_counts"]["train"]*e["train_seconds_per_row"]+checkpoint_rows*e["validation_seconds_per_row"])
                  for e in evidence)*spec["runtime_projection_margin"]+cache_seconds
    measured=dict(peak_cuda_bytes=peak,peak_reserved_cuda_bytes=reserved,peak_rss_bytes=rss,gpu=gpu,
                  projected_max_fit_seconds=projected,cache_seconds=cache_seconds)
    write_immutable_json(directory/"resource_measurements.json",artifact("RESOURCE_MEASUREMENTS",**measured,
        campaign_sha256=spec["content_hash"],final_test_accessed=False))
    if peak>gpu["total_memory_bytes"]*spec["gpu_peak_fraction_limit"]:
        raise MemoryError(f"K2 GPU peak {peak/2**30:.2f} GiB exceeds registered 90% headroom; no automatic batch changes")
    if rss>spec["resources"]["train"]["memory_mb"]*1024**2*spec["cpu_peak_fraction_limit"]:
        raise MemoryError("K2 CPU peak exceeds registered headroom")
    if projected>23*3600:
        raise RuntimeError(f"Projected K2 100-pass fit {projected/3600:.2f}h exceeds the portable 23h bound")
    value=artifact("ACCEPTANCE",campaign_sha256=spec["content_hash"],passed=True,acceptance_only=True,
        job_id=job,site=runtime_site(spec),requested_site=spec["execution_site"],
        execution_policy_sha256=spec["execution_policy"]["content_hash"],
        resource=spec["resources"]["preflight"],environment=environment,
        elapsed_seconds=time.monotonic()-started,**measured,cache_bounds=cache_bounds(spec),
        ordinary_rows={r:spec["role_counts"][r] for r in ("train","validation")},
        capacity=spec["foundation"]["inputs"]["capacity"],batch_size=256,native_execution=evidence,
        pair_storage=PAIR_STORAGE,batch_probe_policy=BATCH_PROBE_POLICY,
        storage_parity_reports=storage_reports,batch_probes=probes,
        checkpoint_round_trip=True,bank_round_trip=True,installed_weaver_fp32_parity=True,
        parity_reports=parity,worst_population_batch_stress=True,duplicate_pair_finiteness=True,
        hlt_only_endpoint=True,final_test_accessed=False)
    validate_memory_evidence(spec,value)
    return value


def run_task(spec,name,*,device="cuda"):
    validate_campaign(spec); row=task(spec,name)
    old=completed(spec,name)
    if old: return old
    parents={p:completed(spec,p) for p in row["dependencies"]}
    if not all(parents.values()): raise PermissionError("Required parent artifacts are incomplete")
    from .concat_k2_submit import authenticate_job
    job_id=authenticate_job(spec,name)
    root=Path(spec["campaign_root"]); directory=root/"outputs"/name
    directory.mkdir(parents=True,exist_ok=False); kind=row["kind"]; imported_paths=[]
    try:
        if kind in {"train","reduce"}:
            execution_gate(spec,science=True)
            result=fit(spec,row,directory,device) if kind=="train" else reduce(spec,row,directory,device)
        elif kind=="authenticate":
            validate_import(spec["source_import"],deep=True)
            from .inventory import verify_snapshot
            verify_snapshot(Path(spec["data_root"]),spec["foundation"]["inventory"])
            result=dict(source_import_sha256=spec["source_import"]["content_hash"])
        elif kind=="import_preparation":
            from .concat_k2_preparation_import import import_preparation
            result,imported_paths=import_preparation(spec,directory)
        elif kind=="matcher_acceptance":
            report=matcher_acceptance(spec)
            write_immutable_json(directory/"matcher_acceptance.json",report)
            result=dict(matcher_acceptance_sha256=report["content_hash"])
        elif kind=="assign":
            report=assignment(spec,row["file_index"],directory)
            result=dict(assignment_sha256=report["content_hash"])
        elif kind=="foundation":
            report=foundation_lock(spec)
            write_immutable_json(directory/"foundation_lock.json",report)
            result=dict(foundation_lock_sha256=report["content_hash"])
        elif kind=="partition":
            report=publish_partition(spec,prepare(spec,"validation","HLT_X1"))
            result=dict(partition_sha256=report["content_hash"])
        elif kind=="storage":
            bounds=cache_bounds(spec); free=shutil.disk_usage(root).free
            if free<spec["minimum_free_disk_bytes"]: raise OSError("Insufficient durable storage headroom")
            result=dict(cache_bounds=bounds,free_bytes=free)
        elif kind=="preflight":
            report=preflight(spec,directory,device)
            write_immutable_json(directory/"acceptance.json",report)
            result=dict(acceptance=directory/"acceptance.json")
        elif kind=="aggregate":
            result=dict(rows=result_rows(spec),recovery_reference="HLT_X1_CE=0%, OFFLINE_CE=100%")
        elif kind=="complete":
            result=dict(scientific_fits=10,reducers=5,sealed_test=True)
        else: raise ValueError("Unknown K2 task kind")
        result={k:v.relative_to(root).as_posix() if isinstance(v,Path) else v for k,v in result.items()}
        write_immutable_json(directory/"result.json",artifact("RESULT",result=result,campaign_sha256=spec["content_hash"],task_id=name,final_test_accessed=False))
        paths=sorted(p for p in directory.rglob("*") if p.is_file())
        paths += imported_paths
        paths += [root/"execution"/name/(job_id+".json")]
        if kind=="partition": paths += [root/"validation_partition.json",root/"validation_partition.npz"]
        value=artifact("TASK_REPORT",campaign_sha256=spec["content_hash"],task_id=name,source_commit=spec["source_commit"],
            parents={p:r["content_hash"] for p,r in parents.items()},result=result,
            outputs=[dict(path=p.relative_to(root).as_posix(),sha256=sha256_file(p)) for p in paths],final_test_accessed=False)
        write_immutable_json(root/"tasks"/(name+".json"),value)
        return value
    except Exception as error:
        diagnostic=dict(error_type=type(error).__name__,message=str(error))
        if torch.cuda.is_available():
            diagnostic.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved())
        write_immutable_json(directory/"failure.json",artifact("FAILURE",campaign_sha256=spec["content_hash"],
            task_id=name,diagnostic=diagnostic,final_test_accessed=False))
        raise
    finally:
        _cuda_clear()
