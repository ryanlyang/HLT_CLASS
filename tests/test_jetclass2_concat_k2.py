"""K2 physics semantics, authenticated ROOT preparation and queue isolation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_views as views, concat_k2_data as data, concat_k2_campaign as chain,
    concat_k2_submit as scheduler, concat_k2_runtime as runtime,
)
from hlt_classification.jetclass2_delphes.reader import Jet, Particles
from hlt_classification.jetclass2_delphes.inputs import build_inputs
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES
from test_jetclass2_delphes import particle_values, snapshot
from test_jetclass2_dzfix_fusion_chain import fake_native, make_cache, rehash


def spec_at(tmp_path):
    f = dict(content_hash="f"*64, inputs={"capacity": 32}, candidate=CANDIDATES[0],
             assignment_tasks=[dict(file_index=0, role="train", path="a.root", rows=88),
                               dict(file_index=1, role="validation", path="b.root", rows=88)])
    return chain.artifact("CAMPAIGN_SPEC", **chain.registration(),
        source_commit="a"*40, campaign_root=str(tmp_path/"campaign"), project_dir=str(tmp_path/"project"),
        data_root=str(tmp_path/"data"), foundation=f, tasks=chain.task_graph(f),
        source_import={"content_hash":"b"*64}, launch_sha256="c"*64)


def jet(nh,no):
    h,o=particle_values(nh),particle_values(no)
    h[:,1]=np.linspace(-.5,.4,nh); o[:,1]=np.linspace(.1,.5,no)
    h[:,3]+=3; o[:,3]+=4
    return Jet("a"*64,1,Particles(h),Particles(o))


@pytest.mark.parametrize("candidate",CANDIDATES)
@pytest.mark.parametrize("nh,no",[(1,1),(1,3),(2,3),(2,6),(3,2),(3,4),(4,3)])
def test_exact_capacity_two_matches_exhaustive_and_keeps_every_hlt(candidate,nh,no):
    j=jet(nh,no)
    mapping=views.match_particles(j.hlt,j.offline,candidate)
    np.testing.assert_array_equal(mapping,views.match_particles(j.hlt,j.offline,candidate,reference=True))
    assert mapping.shape==(nh,2) and (mapping>=0).sum()==min(no,2*nh)
    retained=mapping[mapping>=0]
    assert len(np.unique(retained))==len(retained)
    for coordinate in views.STRENGTHS:
        v=views.build_view(j,coordinate,candidate,mapping)
        assert len(v)==3*nh
        np.testing.assert_array_equal(v.values[::3],j.hlt.values)
        inputs=build_inputs(v,capacity=max(16,3*nh))
        assert inputs.mask.sum()==3*nh
        assert np.isfinite(inputs.features).all()
    rich=views.build_view(j,"D100",candidate,mapping)
    for owner in range(nh):
        for slot in range(2):
            expected=j.offline.values[mapping[owner,slot]] if mapping[owner,slot]>=0 else j.hlt.values[owner]
            np.testing.assert_array_equal(rich.values[owner*3+slot+1],expected)


def test_retention_precedes_geometry_and_frozen_salience_not_renormalized():
    j=jet(2,7); c=CANDIDATES[0]
    a,kept=views.assignment_problem(j.hlt,j.offline,c)
    base=views.pairing_matrices(j.hlt,j.offline,c)
    expect=sorted(sorted(range(7),key=lambda i:(-int(base["offline_salience"][i]),i))[:4])
    assert kept.tolist()==expect
    np.testing.assert_array_equal(a["offline_salience"],base["offline_salience"][kept])
    np.testing.assert_array_equal(a["hlt_salience"],np.repeat(base["hlt_salience"],2))
    np.testing.assert_array_equal(a["utility"],np.repeat(base["utility"][:,kept],2,axis=0))


def test_equal_salience_ties_and_exchangeable_slot_canonicalization():
    h=Particles(np.repeat(particle_values(1),2,axis=0))
    o=Particles(np.repeat(particle_values(1),5,axis=0))
    m=views.match_particles(h,o,CANDIDATES[0])
    assert m.tolist()==[[0,1],[2,3]]
    assert views.match_particles(h,Particles(o.values[:1]),CANDIDATES[0]).tolist()==[[0,-1],[-1,-1]]


def test_hlt_only_endpoint_does_not_inspect_offline_matching_candidate_or_identity():
    h=jet(3,7).hlt
    class HLTOnly:
        hlt=h
        def __getattr__(self,key): raise AssertionError("Forbidden input: "+key)
    for coordinate in ("D000","HLT_X3"):
        result=views.build_view(HLTOnly(),coordinate,object(),object())
        np.testing.assert_array_equal(result.values,np.repeat(h.values,3,axis=0))
    assert views.build_view(HLTOnly(),"HLT_X1",object()) is h
    raw=build_inputs(views.hlt_x3(h),capacity=16)
    assert raw.features.shape==(17,16) and raw.mask.sum()==9


def test_mixed_type_switches_nested_finite_atomic_and_order_independent():
    h,o=particle_values(3),particle_values(4)
    o[:,4:10]=0; o[:,6]=1; o[:,10:14]=0
    h[:,11]=.04; h[:,13]=.1
    j=Jet("f"*64,2,Particles(h),Particles(o)); c=CANDIDATES[2]
    m=views.match_particles(j.hlt,j.offline,c)
    outputs={s:views.build_view(j,s,c,m).values for s in views.STRENGTHS}
    for owner,slot in zip(*np.nonzero(m>=0)):
        index=3*owner+slot+1
        neutral=[outputs[s][index,4]==0 for s in views.STRENGTHS]
        assert neutral==sorted(neutral,reverse=True)
        for s in ("D075","D050","D025"):
            value=outputs[s][index]
            assert (value[11]>0)==(value[4]!=0)
            strength=float(views.STRENGTHS[s])
            np.testing.assert_allclose(value[:4],strength*o[m[owner,slot],:4]+(1-strength)*h[owner,:4],rtol=1e-6)
    for s in reversed(views.STRENGTHS):
        np.testing.assert_array_equal(outputs[s],views.build_view(j,s,c,m).values)


def test_invalid_maps_fail_closed():
    for m in (np.array([[0,0]],np.int32),np.array([[-2,0]],np.int32),
              np.array([[1,0]],np.int32),np.array([[0,1]],np.int64)):
        with pytest.raises(ValueError): views.validate_mapping(m,nh=1,no=2)


@pytest.mark.parametrize("partition", ["tier3", "debug"])
def test_requested_graph_controls_seed_pairing_and_portable_dry_run(tmp_path,monkeypatch,partition):
    spec=spec_at(tmp_path); nodes=spec["nodes"]
    spec=rehash(spec,**chain.registration(partition))
    assert len(nodes)==10 and len({n["initialization_seed"] for n in nodes})==1
    assert all(n["initialization"]=="fresh" and n["context_coordinate"] is None for n in nodes)
    assert [(n["node_id"],n["teacher_distribution"]) for n in nodes[5:]]==[
        ("CONCAT_K2_D075","CONCAT_K2_D100"),("CONCAT_K2_D050","CONCAT_K2_D075"),
        ("CONCAT_K2_D025","CONCAT_K2_D050"),("CONCAT_K2_D000","CONCAT_K2_D025"),
        ("HLT_X1_COMPRESSED","CONCAT_K2_D000")]
    assert len(scheduler.plan(spec,"science")["commands"])==17
    seen=set()
    for row in scheduler.plan(spec,"full")["commands"]:
        assert set(row["dependencies"])<=seen; seen.add(row["task_id"])
        command=row["command"]
        assert "--partition="+partition in command and "--no-requeue" in command
        assert int(next(t.split("=")[1] for t in command if t.startswith("--time=")))<=1440
    monkeypatch.setattr(scheduler,"validate_campaign",lambda s:s["content_hash"])
    ledger=scheduler.submit(spec,stage="full")
    assert ledger["dry_run"] and len(ledger["jobs"])==len(spec["tasks"])
    with pytest.raises(PermissionError,match="Full-DAG"):
        scheduler.submit(spec,stage="full",execute=True,authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError,match="fresh|Fresh"):
        scheduler.submit(spec,stage="science",execute=True,authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError,match="authorization"):
        scheduler.submit(spec,stage="gate",execute=True)


def test_exact_gate_submission_idempotent_and_journal_failure_closed(tmp_path,monkeypatch):
    spec=spec_at(tmp_path)
    monkeypatch.setattr(scheduler,"validate_campaign",lambda s:s["content_hash"])
    scheduler.submit(spec,stage="full"); calls=[]
    def sbatch(c,**kw):
        calls.append(c); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(scheduler.subprocess,"run",sbatch)
    first=scheduler.submit(spec,stage="gate",execute=True,authorization=chain.AUTHORIZE)
    assert first==scheduler.submit(spec,stage="gate",execute=True,authorization=chain.AUTHORIZE)
    assert len(calls)==len(chain.gates(spec))
    assert all(not any("${JOB_" in x for x in c) for c in calls)
    assert "--dependency=afterok:102" in calls[2]
    assert "--dependency=afterok:103:104" in calls[4]


def test_partition_stable_disjoint_stratified_and_sealed(tmp_path):
    spec=spec_at(tmp_path); v=make_cache("validation")
    p=data.publish_partition(spec,v); assert p["counts"]==[44,22,22]
    roles=data.partition_indices(spec,v)
    assert len(set(np.concatenate(list(roles.values()))))==len(v)
    codes=data.partition_codes(v.identities,v.labels)
    assert np.array_equal(codes[::-1],data.partition_codes(v.identities[::-1],v.labels[::-1]))
    with pytest.raises(PermissionError,match="sealed"): data.prepare(spec,"final_test","D100")
    with pytest.raises(PermissionError,match="Real installed-Weaver"): runtime.preflight(spec,tmp_path,"cpu")


def test_capacity_rounds_up_expanded_hlt_not_old_320():
    p=dict(inventory={"files":[dict(max_selected_particles=dict(hlt=311,offline=319))]},
           splits={},assignment_tasks=[],candidate=CANDIDATES[0])
    f=chain.foundation_spec(p,"a"*64)
    assert f["inputs"]["capacity"]==944 and f["native_capacity"]==320
    assert f["reused_assignments"] is False


def test_real_synthetic_root_assignment_roundtrip_corruption_and_endpoint_cache(snapshot,tmp_path):
    from hlt_classification.jetclass2_delphes.split_registry import build_registry,select_profile
    from hlt_classification.jetclass2_delphes.salience_foundation import build_foundation_spec
    raw,inv,reservoirs=snapshot
    registry=build_registry(raw,inv,reservoirs,training_sizes=(11,),validation_size=11,test_size=11,step_size=3)
    parent=build_foundation_spec(inv,select_profile(registry,inv,"TRAIN_11"),CANDIDATES[0])
    f=chain.foundation_spec(parent,"a"*64); spec=spec_at(tmp_path)
    spec.update(foundation=f,data_root=str(raw),role_counts=f["splits"]["role_counts"])
    spec["resources"]["train"]["cpus"]=1
    for task in f["assignment_tasks"]:
        directory=Path(spec["campaign_root"])/"outputs"/f"assign_{task['file_index']:04d}"
        directory.mkdir(parents=True)
        report=data.assignment(spec,task["file_index"],directory)
        restored,arrays=data.load_assignment(spec,task["file_index"])
        assert report==restored and arrays["mapping"].shape[1]==2
        assert report["counts"]["cropped"]==0
    assert data.foundation_lock(spec)["counts"]["jets"]==22
    rich=data.prepare(spec,"train","D100"); h3=data.prepare(spec,"train","D000"); h1=data.prepare(spec,"train","HLT_X1")
    np.testing.assert_array_equal(rich.identities,h3.identities)
    assert np.all(np.diff(h3.blocks[0].offsets)==3*np.diff(h1.blocks[0].offsets))
    assert rich.batch(np.arange(11))["mask"].sum()==66
    first=f["assignment_tasks"][0]
    path=Path(spec["campaign_root"])/"outputs"/f"assign_{first['file_index']:04d}"/"assignments.npz"
    with path.open("ab") as stream: stream.write(b"corrupt")
    with pytest.raises(ValueError,match="bytes differ"): data.load_assignment(spec,first["file_index"])


def test_full_cpu_test_double_dispatch_teacher_banks_and_endpoint_compression(fake_native,tmp_path,monkeypatch):
    spec=spec_at(tmp_path); root=Path(spec["campaign_root"]); root.mkdir()
    monkeypatch.setattr(runtime,"validate_campaign",lambda s:s["content_hash"])
    monkeypatch.setattr(runtime,"validate_import",lambda *a,**k:None)
    def authenticate(s,name):
        write_immutable_json(root/"execution"/name/"123.json",chain.artifact("EXECUTION_RECORD",test_only=True))
        return "123"
    monkeypatch.setattr(scheduler,"authenticate_job",authenticate)
    monkeypatch.setattr(runtime,"execution_gate",lambda *a,**k:"123")
    monkeypatch.setattr(runtime,"cache_bounds",lambda s:{"train":1,"validation":1})
    monkeypatch.setattr(data,"prepare",lambda s,role,coordinate:make_cache(role,coordinate))
    monkeypatch.setattr(runtime,"prepare",data.prepare)
    from hlt_classification.jetclass2_delphes import inventory
    monkeypatch.setattr(inventory,"verify_snapshot",lambda *a:None)
    spec["foundation"]["inventory"]={}
    monkeypatch.setattr(runtime,"assignment",lambda *a:chain.artifact("ASSIGNMENT_SHARD",test_only=True))
    monkeypatch.setattr(runtime,"foundation_lock",lambda *a:chain.artifact("FOUNDATION_LOCK",test_only=True))
    monkeypatch.setattr(runtime,"preflight",lambda *a:chain.artifact("ACCEPTANCE",test_only=True))
    original=runtime.train_kernel
    def miniature(*a,**kw):
        r,s=original(*a,**kw,acceptance_passes=1)
        return rehash(r,scientific_fit=True,acceptance_only=False),s
    monkeypatch.setattr(runtime,"train_kernel",miniature)
    for row in spec["tasks"]:
        done=runtime.run_task(spec,row["task_id"],device="cpu")
        assert runtime.completed(spec,row["task_id"])==done
    assert len(runtime.result_rows(spec))==10
    for row in runtime.result_rows(spec):
        assert row["validation"]["rows"]==22
    done=runtime.completed(spec,"train_HLT_X1_COMPRESSED")
    r=load_json(root/done["result"]["training_report"])
    assert r["teacher_lineage"]["teacher_node"]=="CONCAT_K2_D000"
    assert r["checkpoint_validation"]["rows"]==44
    (root/done["result"]["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="bytes changed"): runtime.completed(spec,"train_HLT_X1_COMPRESSED")


def test_installed_weaver_duplicate_input_parity():
    pytest.importorskip("weaver")
    from hlt_classification.jetclass2_delphes.acceptance import installed_parity
    torch.set_num_threads(1)
    from hlt_classification.jetclass2_delphes.cache import RamBlock,RamCache
    value=views.deployment_inputs(jet(4,5).hlt,copies=3,capacity=16)
    cache=make_cache("train")
    b=RamBlock(0,np.arange(len(cache)+1)*12,np.tile(value.features[:,:12].T,(len(cache),1)),
        np.tile(value.vectors[:,:12].T,(len(cache),1)),cache.identities,cache.labels)
    assert installed_parity(RamCache([b],role="train",foundation_sha256="f"*64,coordinate_name="D000"),device="cpu")["passed"]


def test_export_adapter_exactly_matches_hlt_endpoint_and_has_no_owner_flags():
    j=jet(4,9)
    x3=views.deployment_inputs(j.hlt,copies=3,capacity=32)
    endpoint=build_inputs(views.build_view(Jet(j.identity,j.label,j.hlt,None),"D000",None),capacity=32)
    for name in ("features","vectors","mask"):
        np.testing.assert_array_equal(getattr(x3,name),getattr(endpoint,name))
    assert x3.features.shape[0]==17
    for owner in range(4):
        for slot in (1,2):
            np.testing.assert_array_equal(x3.features[:,3*owner],x3.features[:,3*owner+slot])
    with pytest.raises(ValueError): views.deployment_inputs(j.hlt,copies=2,capacity=32)


@pytest.mark.parametrize("coordinate",["HLT_X1","HLT_X3","D000","OFFLINE"])
def test_unmatched_endpoint_readers_do_not_open_assignment_files(coordinate,tmp_path,monkeypatch):
    j=jet(3,8); seen=[]
    def reader(*a,**kw):
        seen.append(kw["include_offline"])
        return iter([j if kw["include_offline"] else Jet(j.identity,j.label,j.hlt,None)])
    monkeypatch.setattr(data,"DatasetReader",reader)
    monkeypatch.setattr(data,"load_assignment",lambda *a,**k:pytest.fail("Unexpected old/new assignment access"))
    f=dict(inventory={},splits={},inputs={"capacity":32},candidate=CANDIDATES[0])
    t=dict(file_index=0,path="a.root",role="train",rows=1)
    block=data._file_cache((f,str(tmp_path),str(tmp_path),t,coordinate))
    assert seen==[coordinate=="OFFLINE"]
    assert block.offsets.tolist()==[0,{"HLT_X1":3,"HLT_X3":9,"D000":9,"OFFLINE":8}[coordinate]]


@pytest.mark.parametrize("accepted_partition", ["tier3", "debug"])
def test_memory_batch_capacity_execution_attestations_cannot_be_bypassed(tmp_path,monkeypatch,accepted_partition):
    from test_jetclass2_concat_k2_memory import memory_evidence
    from hlt_classification.jetclass2_delphes.concat_k2_execution import site_for_partition
    spec=spec_at(tmp_path); root=Path(spec["campaign_root"])
    evidence=dict(campaign_sha256=spec["content_hash"],passed=True,final_test_accessed=False,
        site=site_for_partition(accepted_partition),resource=spec["resources"]["preflight"],acceptance_only=True,
        requested_site=spec["execution_site"],execution_policy_sha256=spec["execution_policy"]["content_hash"],
        ordinary_rows={r:spec["role_counts"][r] for r in ("train","validation")},
        batch_size=256,capacity=32,checkpoint_round_trip=True,bank_round_trip=True,
        installed_weaver_fp32_parity=True,worst_population_batch_stress=True,
        hlt_only_endpoint=True,duplicate_pair_finiteness=True,peak_cuda_bytes=800,
        gpu={"total_memory_bytes":1000},peak_rss_bytes=1000,projected_max_fit_seconds=100,
        native_execution=[{"kernel_report":{"acceptance_only":True,"scientific_fit":False}}]*4,
        **memory_evidence(spec))
    for name,changes in (("good",{}),("memory",{"peak_cuda_bytes":901}),("batch",{"batch_size":128}),
            ("test",{"final_test_accessed":True}),("time",{"projected_max_fit_seconds":24*3600}),
            ("shape",{"capacity":16}),("duplicates",{"duplicate_pair_finiteness":False})):
        v=chain.artifact("ACCEPTANCE",**{**evidence,**changes}); write_immutable_json(root/(name+".json"),v)
        monkeypatch.setattr(runtime,"completed",lambda *a:{"result":{"acceptance":name+".json"}})
        if changes:
            with pytest.raises(ValueError,match="acceptance"): runtime.science_gate(spec)
        else: assert runtime.science_gate(spec)==v


def test_gpu_stress_uses_full_batch_capacity_and_leaves_raw_batch_unchanged(monkeypatch):
    raw=make_cache("train").batch(np.arange(80)); original=deepcopy(raw); shapes=[]
    model=torch.nn.Linear(1,1)
    monkeypatch.setattr(runtime,"_optimizer",lambda m:torch.optim.SGD(m.parameters(),lr=.01))
    def forward(m,batch,**kw):
        shapes.append(batch["features"].shape)
        assert batch["mask"].sum()==80*4
        return m(torch.ones(1,1)).sum(),{}
    monkeypatch.setattr(runtime,"_train_batch",forward)
    runtime._stress(model,raw,chain.nodes()[0],"cpu",64)
    assert shapes==[(80,17,64)]*3
    for k in raw: np.testing.assert_array_equal(raw[k],original[k])


def test_ambiguous_submission_cannot_be_blindly_retried(tmp_path,monkeypatch):
    spec=spec_at(tmp_path)
    monkeypatch.setattr(scheduler,"validate_campaign",lambda s:s["content_hash"])
    scheduler.submit(spec,stage="full")
    def lost(*a,**kw): raise RuntimeError("lost acknowledgement")
    monkeypatch.setattr(scheduler.subprocess,"run",lost)
    with pytest.raises(RuntimeError): scheduler.submit(spec,stage="gate",execute=True,authorization=chain.AUTHORIZE)
    with pytest.raises(PermissionError,match="Ambiguous"):
        scheduler.submit(spec,stage="gate",execute=True,authorization=chain.AUTHORIZE)


def test_monitor_reads_only_authenticated_own_job_ids(tmp_path,monkeypatch):
    spec=spec_at(tmp_path)
    monkeypatch.setattr(scheduler,"validate_campaign",lambda s:s["content_hash"])
    scheduler.submit(spec,stage="full"); calls=[]
    def sbatch(c,**kw):
        calls.append(c); return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(scheduler.subprocess,"run",sbatch)
    ledger=scheduler.submit(spec,stage="gate",execute=True,authorization=chain.AUTHORIZE)
    def sacct(c,**kw):
        assert c[0]=="sacct" and set(c[c.index('-j')+1].split(','))==set(ledger["jobs"].values())
        return SimpleNamespace(stdout="101|COMPLETED|00:01:00|debug|\n")
    monkeypatch.setattr(scheduler.subprocess,"run",sacct)
    rows=scheduler.monitor(spec)["rows"]
    assert next(r for r in rows if r["job_id"]=="101")["state"]=="COMPLETED"
    observed=next(r for r in rows if r["job_id"]=="101")
    assert observed["requested_partition"]=="tier3" and observed["actual_partition"]=="debug"
    assert next(r for r in rows if r["job_id"]=="102")["state"]=="UNKNOWN"


def test_preflight_dispatch_cpu_double_exercises_every_stage_not_remote_acceptance(fake_native,tmp_path,monkeypatch):
    import sys
    from hlt_classification.jetclass2_delphes import acceptance, concat_k2_parity, model as model_module
    from test_jetclass2_concat_k2_memory import memory_evidence
    spec=spec_at(tmp_path); root=Path(spec["campaign_root"]); root.mkdir()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "debug")  # Actual site may differ from tier3 submission.
    directory=root/"local_double"; directory.mkdir()
    data.publish_partition(spec,make_cache("validation"))
    monkeypatch.setitem(sys.modules,"resource",SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda *a:SimpleNamespace(ru_maxrss=1024)))
    monkeypatch.setattr(runtime,"execution_gate",lambda *a,**k:"123")
    monkeypatch.setattr(runtime,"installed_environment",lambda:{"test_only":True})
    monkeypatch.setattr(runtime,"gpu_identity",lambda:{"total_memory_bytes":10**12,"test_only":True})
    monkeypatch.setattr(runtime,"prepare",lambda s,role,c:make_cache(role,c))
    monkeypatch.setattr(runtime,"cache_bounds",lambda s:{"train":1,"validation":1})
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:1000)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda:2000)
    parity=acceptance.installed_parity
    monkeypatch.setattr(acceptance,"load_weaver_particle_transformer_class",model_module.load_weaver_particle_transformer_class)
    monkeypatch.setattr(acceptance,"installed_parity",lambda c,**kw:parity(c,device="cpu"))
    original_train=runtime.train_kernel; original_predict=runtime.predict
    def train(*a,**kw):
        kw["device"]="cpu"
        report,state=original_train(*a,**kw)
        # Test orchestration, not wall-clock extrapolation from a CPU double.
        report["validation_history"][-1].update(train_seconds=.01,validation_seconds=.01)
        return rehash(report),state
    def predict(*a,**kw): kw["device"]="cpu"; return original_predict(*a,**kw)
    monkeypatch.setattr(runtime,"train_kernel",train)
    monkeypatch.setattr(runtime,"predict",predict)
    fabricated=memory_evidence(spec)
    monkeypatch.setattr(concat_k2_parity,"early_parity",lambda *a:fabricated["early_parity_reports"])
    monkeypatch.setattr(runtime,"storage_parity",lambda *a,**kw: {
        k:v for k,v in fabricated["storage_parity_reports"][int(kw["bf16"])].items() if k!="node_id"})
    monkeypatch.setattr(runtime,"batch_probes",lambda s,d,n,*a:[
        r for r in fabricated["batch_probes"] if r["node_id"]==n["node_id"]])
    value=runtime.preflight(spec,directory,"cuda")
    assert value["site"]["partition"] == "debug" and value["requested_site"]["partition"] == "tier3"
    assert value["environment"]["test_only"] and len(value["native_execution"])==4
    assert value["bank_round_trip"] and value["checkpoint_round_trip"]
    assert (directory/"resource_measurements.json").is_file()
    assert all(r["kernel_report"]["acceptance_only"] for r in value["native_execution"])
