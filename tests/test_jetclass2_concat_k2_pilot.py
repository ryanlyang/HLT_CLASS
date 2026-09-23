"""Small K2 pilot registration, real ROOT subsets, training and queue isolation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as campaign, concat_k2_pilot as pilot,
    concat_k2_pilot_data as data, concat_k2_data as parent_data,
    concat_k2_runtime as runtime, concat_k2_submit as scheduler,
    concat_k2_execution as execution, concat_k2_source as source,
    salience_learned_training as training, salience_learned_graph as graph,
)
from hlt_classification.jetclass2_delphes.salience_foundation import build_foundation_spec
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES
from test_jetclass2_delphes import snapshot
from test_jetclass2_concat_k2 import spec_at
from test_jetclass2_concat_k2_source import imported_source
from test_jetclass2_concat_k2_reuse import completed_donor, target_spec
from test_jetclass2_dzfix_fusion_chain import rehash, fake_native, make_cache


def pilot_spec(tmp_path, partition="debug"):
    old = spec_at(tmp_path)
    fields = {k:v for k,v in old.items() if k not in {"contract","schema_version","content_hash"}}
    fields.update(campaign.registration(partition,pilot.PROFILE))
    fields["tasks"] = campaign.task_graph(fields["foundation"],reuse=True,profile=pilot.PROFILE)
    fields["preparation_import"] = {"test_only":True}
    return campaign.artifact("CAMPAIGN_SPEC",**fields)


def test_pilot_graph_recipe_versions_and_old_profile_unchanged(tmp_path):
    old = campaign.registration()
    shared = deepcopy(graph.TRAINING)
    spec = pilot_spec(tmp_path)
    assert spec["schema_version"] == 8
    assert spec["role_counts"] == dict(train=100000,validation=50000,final_test=1000000)
    assert (spec["fresh_fit_count"],spec["reducer_count"],spec["science_task_count"]) == (8,3,13)
    assert [n["primary_coordinate"] for n in spec["nodes"][5:]] == ["D050","D000","HLT_X1"]
    assert [n["teacher_distribution"] for n in spec["nodes"][5:]] == ["CONCAT_K2_D100","CONCAT_K2_D050","CONCAT_K2_D000"]
    assert len({n["initialization_seed"] for n in spec["nodes"]}) == 1
    assert len(campaign.gates(spec)) == 7
    assert not any(t["kind"] in {"assign","matcher_acceptance"} for t in spec["tasks"])
    assert "select_population" in campaign.gates(spec)
    assert len(scheduler.plan(spec,"science")["commands"]) == 13
    assert runtime.probe_cases(spec)[1] == "CONCAT_K2_D050"
    assert graph.TRAINING == shared and campaign.registration() == old
    assert old["fresh_fit_count"] == 10 and old["training"]["maximum_passes"] == 100
    assert old["resources"]["train"]["minutes"] == 5760
    assert campaign.artifact("ACCEPTANCE",experiment_profile=pilot.PROFILE)["schema_version"] == 7


@pytest.mark.parametrize("partition",["debug","tier3"])
def test_all_pilot_commands_fit_debug_and_allow_partition_moves(tmp_path,partition):
    spec = pilot_spec(tmp_path,partition)
    commands = scheduler.plan(spec,"full")["commands"]
    assert len(commands) == 20
    seen = set()
    for row in commands:
        assert set(row["dependencies"]) <= seen
        seen.add(row["task_id"])
        c = row["command"]
        assert "--partition="+partition in c and "--qos=qos_tier3" in c
        assert "--job-name=jc2k2p_"+row["task_id"] in c
        assert int(next(v.split("=")[1] for v in c if v.startswith("--time="))) <= 1440
        task = next(t for t in spec["tasks"] if t["task_id"] == row["task_id"])
        resource = spec["resources"][task["resource"]]
        for actual in ("debug","tier3"):
            assert execution.admit_site(spec,actual,resource=resource)["partition"] == actual
    assert execution.runtime_projection_limit_seconds(spec) == 23*3600
    execution.validate_runtime_projection(spec,23*3600)
    for invalid in (0,float("nan"),float("inf"),23*3600+.1,262478.22):
        with pytest.raises(ValueError): execution.validate_runtime_projection(spec,invalid)


def test_pilot_live_submission_needs_own_authorization_and_fresh_gate(tmp_path,monkeypatch):
    spec = pilot_spec(tmp_path)
    monkeypatch.setattr(scheduler,"validate_campaign",lambda s:s["content_hash"])
    ledger = scheduler.submit(spec,stage="full")
    assert ledger["dry_run"] and len(ledger["jobs"]) == 20
    with pytest.raises(PermissionError,match="authorization"):
        scheduler.submit(spec,stage="gate",execute=True,authorization=campaign.AUTHORIZE)
    with pytest.raises(PermissionError,match="Full-DAG"):
        scheduler.submit(spec,stage="full",execute=True,authorization=pilot.AUTHORIZE)
    with pytest.raises(PermissionError,match="Fresh"):
        scheduler.submit(spec,stage="science",execute=True,authorization=pilot.AUTHORIZE)


@pytest.mark.parametrize("position,expected",[(1,1e-4),(3,3e-4),(20,3e-4),
    (25,(3e-4+1.5e-5)/2),(30,1.5e-5),(45,1.5e-5),(60,1.5e-5)])
def test_pilot_schedule_landmarks(position,expected):
    assert pilot.learning_rate(position) == pytest.approx(expected)


@pytest.mark.parametrize("improve,expected_passes",[(False,45),(True,60)])
def test_real_kernel_pilot_clock_maximum_and_restore(fake_native,monkeypatch,improve,expected_passes):
    torch.set_num_threads(1)
    original = deepcopy(graph.TRAINING)
    train,validation = make_cache("train"),make_cache("validation")
    node = campaign.nodes(pilot.PROFILE)[0]
    model = runtime.new_model(node)
    real_metrics = training.evaluate_probabilities
    calls = []
    def metrics(*args):
        result = real_metrics(*args)
        calls.append(1)
        result["macro_ovr_auc"] = .6+len(calls)*.001 if improve else .8 if len(calls)==1 else .7
        return result
    monkeypatch.setattr(training,"evaluate_probabilities",metrics)
    report,state = training.train_kernel(model,lambda _:train,lambda _:validation,
        node=node,device="cpu",batch_size=128,inference_batch_size=128,
        training_recipe=pilot.training_recipe())
    assert report["passes"] == expected_passes
    assert report["selected_pass"] == (expected_passes if improve else 1)
    assert report["selected_weights_restored"] and report["scientific_fit"]
    assert report["training_recipe"] == pilot.training_recipe()
    assert report["validation_history"][29]["learning_rate"] == 1.5e-5
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(),state[name],rtol=0,atol=0)
    assert graph.TRAINING == original


def test_kernel_rejects_unregistered_recipe_before_opening_data():
    recipe = pilot.training_recipe(); recipe["maximum_passes"] = 61
    def forbidden(_): pytest.fail("Must reject before reading data")
    with pytest.raises(ValueError,match="recipe"):
        training.train_kernel(None,forbidden,forbidden,node={"role":"reference_ce"},device="cpu",
            batch_size=128,inference_batch_size=128,training_recipe=recipe)


def test_hash_subselection_exact_stratified_and_order_independent():
    rows = [(f"{i:064x}",i//30,i,i%11) for i in range(330)]
    quotas = [7]*11
    selected = data.select_rows(rows,quotas,role="train")
    assert selected == data.select_rows(list(reversed(rows)),quotas,role="train")
    assert len(selected) == 77 and np.bincount([r[3] for r in selected]).tolist() == quotas
    assert set(selected) <= set(rows)
    assert selected != data.select_rows(rows,quotas,role="validation")
    with pytest.raises(PermissionError): data.select_rows(rows,quotas,role="final_test")
    with pytest.raises(ValueError): data.select_rows(rows+rows[:1],quotas,role="train")


def test_projection_uses_pilot_rows_passes_and_checkpoint_role(tmp_path):
    spec = pilot_spec(tmp_path)
    evidence = [dict(train_seconds_per_row=.002,validation_seconds_per_row=.001),
                dict(train_seconds_per_row=.003,validation_seconds_per_row=.002)]
    expected = 60*(100000*.003+25000*.002)*1.30+900
    assert runtime.projected_fit_seconds(spec,evidence,25000,900) == expected
    full = campaign.registration()
    assert runtime.projected_fit_seconds(full,evidence,500000,900) == 100*(500000*.003+500000*.002)*1.30+900


@pytest.fixture
def root_pilot(snapshot,tmp_path):
    raw,inv,reservoirs = snapshot
    registry = build_registry(raw,inv,reservoirs,training_sizes=(22,),validation_size=11,test_size=11)
    parent = build_foundation_spec(inv,select_profile(registry,inv,"TRAIN_22"),CANDIDATES[0])
    foundation = campaign.foundation_spec(parent,"a"*64)
    spec = pilot_spec(tmp_path)
    spec = rehash(spec,foundation=foundation,data_root=str(raw),role_counts=dict(train=11,validation=11,final_test=11))
    spec["resources"]["train"]["cpus"] = 1  # synthetic fixture only
    spec = rehash(spec)
    for task in foundation["assignment_tasks"]:
        output = Path(spec["campaign_root"])/"outputs"/f"assign_{task['file_index']:04d}"
        output.mkdir(parents=True)
        parent_data.assignment(spec,task["file_index"],output)
    return spec


def test_root_subset_keeps_maps_and_test_sealed_and_matches_parent_inputs(root_pilot,monkeypatch):
    spec = root_pilot; f = spec["foundation"]
    test_paths = {f["inventory"]["files"][i]["path"] for i,g in enumerate(f["splits"]["groups"]) if g["role"]=="final_test"}
    real_open = data.uproot.open
    def guarded(path,*a,**kw):
        assert not any(str(path).replace("\\","/").endswith(p) for p in test_paths)
        return real_open(path,*a,**kw)
    monkeypatch.setattr(data.uproot,"open",guarded)
    directory = Path(spec["campaign_root"])/"outputs/select_population"
    pop = data.build_population(spec,directory)
    assert data.population(spec) == pop["profile"]
    assert pop["profile"]["memberships"]["final_test"]["files"] == f["splits"]["memberships"]["final_test"]["files"]
    for role in ("train","validation"):
        seen = None
        for coordinate in ("D100","D050","D000","HLT_X1","HLT_X3","OFFLINE"):
            # The full-population and subset workers produce identical retained views.
            full = parent_data.prepare(spec,role,coordinate)
            small = data.prepare(spec,role,coordinate)
            assert len(small) == 11
            if seen is not None: np.testing.assert_array_equal(small.identities,seen)
            seen = small.identities
            lookup = {bytes(identity):i for i,identity in enumerate(full.identities)}
            indices = np.asarray([lookup[bytes(i)] for i in small.identities])
            for key,value in small.batch(np.arange(len(small))).items():
                np.testing.assert_array_equal(value,full.batch(indices)[key])
    from hlt_classification.jetclass2_delphes.concat_k2_parity import parity_sample
    caches,sample = parity_sample(spec)
    assert set(sample["identities"]) <= {bytes(i).hex() for i in data.prepare(spec,"train","HLT_X1").identities}
    assert caches["D000"].coordinate_name == "D000"
    with pytest.raises(PermissionError): data.prepare(spec,"final_test","D000")
    with pytest.raises(ValueError): data.validate_population(spec,rehash(pop,rank_domain="wrong"))


def test_hlt_endpoint_pilot_does_not_load_maps_or_offline(root_pilot,monkeypatch):
    spec = root_pilot
    data.build_population(spec,Path(spec["campaign_root"])/"outputs/select_population")
    monkeypatch.setattr(data,"load_assignment",lambda *a:pytest.fail("HLT endpoint must not load maps"))
    real_reader = data.DatasetReader
    def reader(*a,**kw):
        assert kw["include_offline"] is False
        return real_reader(*a,**kw)
    monkeypatch.setattr(data,"DatasetReader",reader)
    for coord in ("D000","HLT_X1","HLT_X3"):
        assert len(data.prepare(spec,"train",coord)) == 11


def test_pilot_requires_donor_even_before_source_io(tmp_path):
    with pytest.raises(ValueError,match="explicit completed"):
        source.create_launch(screen_spec=tmp_path/"x",inventory_path=tmp_path/"i",
            launch_root=tmp_path/"l",campaign_root=tmp_path/"c",project=tmp_path,
            source_commit="a"*40,partition="debug",profile=pilot.PROFILE)


def test_pilot_materialization_validates_source_and_rejects_rehashed_changes(imported_source,tmp_path,monkeypatch):
    from hlt_classification.jetclass2_delphes import concat_k2_preparation_import as reuse
    launch,_,root = imported_source
    monkeypatch.setattr(reuse,"describe_reuse",lambda _: {"fixture":True})
    monkeypatch.setattr(reuse,"validate_reuse",lambda *a,**kw: None)
    monkeypatch.setattr(campaign,"_source",lambda *a: None)
    result = source.create_launch(screen_spec=root/"screen_spec.json",inventory_path=root/"inventory.json",
        launch_root=tmp_path/"pilot_launch",campaign_root=tmp_path/"pilot_campaign",
        project=Path(launch["project_dir"]),source_commit=launch["source_commit"],partition="debug",
        profile=pilot.PROFILE,reuse_preparation_spec=tmp_path/"donor.json")
    assert result["schema_version"] == 8
    source.validate_launch(result)
    spec = campaign.create(launch=result)
    campaign.validate_campaign(spec)
    assert spec["experiment_profile"] == pilot.PROFILE
    assert all("--partition=debug" in r["command"] for r in scheduler.plan(spec,"full")["commands"])
    for field,value in (("training",graph.TRAINING),("role_counts",campaign.COUNTS),
                         ("preparation_import",None),("experiment_profile","unknown")):
        with pytest.raises((ValueError,PermissionError)):
            campaign.validate_campaign(rehash(spec,**{field:value}))


def test_pilot_gate_checks_recipe_probes_and_23h_limit(tmp_path,monkeypatch):
    from test_jetclass2_concat_k2_memory import memory_evidence
    spec = pilot_spec(tmp_path)
    evidence = memory_evidence(spec)
    evidence["batch_probes"] = [rehash(p,node_id=p["node_id"].replace("D075","D050"))
                                 for p in evidence["batch_probes"]]
    value = campaign.artifact("ACCEPTANCE",experiment_profile=pilot.PROFILE,
        campaign_sha256=spec["content_hash"],passed=True,final_test_accessed=False,
        site=spec["execution_site"],requested_site=spec["execution_site"],
        execution_policy_sha256=spec["execution_policy"]["content_hash"],
        resource=spec["resources"]["preflight"],acceptance_only=True,
        ordinary_rows={r:spec["role_counts"][r] for r in ("train","validation")},
        batch_size=128,inference_batch_size=128,capacity=32,
        checkpoint_round_trip=True,bank_round_trip=True,installed_weaver_fp32_parity=True,
        worst_population_batch_stress=True,hlt_only_endpoint=True,duplicate_pair_finiteness=True,
        peak_cuda_bytes=800,gpu={"total_memory_bytes":1000},peak_rss_bytes=1000,
        projected_max_fit_seconds=8*3600,runtime_projection_limit_seconds=23*3600,
        native_execution=[{"kernel_report":{"acceptance_only":True,"scientific_fit":False,
            "training_recipe":pilot.training_recipe(),
            "batching":{"training_batch_size":128,"inference_batch_size":128,"gradient_accumulation_steps":1}}}]*4,
        **evidence)
    root = Path(spec["campaign_root"])
    for name,changes in (("valid",{}),("time",{"projected_max_fit_seconds":24*3600}),
                         ("rows",{"ordinary_rows":{"train":500000,"validation":1000000}}),
                         ("probes",memory_evidence(spec))):
        entry = rehash(value,**changes)
        write_immutable_json(root/(name+".json"),entry)
        monkeypatch.setattr(runtime,"completed",lambda *a:{"result":{"acceptance":name+".json"}})
        if changes:
            with pytest.raises(ValueError): runtime.science_gate(spec)
        else:
            assert runtime.science_gate(spec) == entry
    bad = {"batching":value["native_execution"][0]["kernel_report"]["batching"]}
    with pytest.raises(ValueError,match="recipe"):
        runtime.validate_kernel_batching(spec,bad)


def test_full_pilot_cpu_dispatch_runs_eight_fits_and_three_banks(fake_native,tmp_path,monkeypatch):
    spec = pilot_spec(tmp_path); root = Path(spec["campaign_root"]); root.mkdir()
    spec["foundation"]["splits"] = {"role_counts":campaign.COUNTS}
    spec["foundation"]["inventory"] = {}
    monkeypatch.setattr(runtime,"validate_campaign",lambda s:s["content_hash"])
    monkeypatch.setattr(runtime,"validate_import",lambda *a,**kw:None)
    from hlt_classification.jetclass2_delphes import inventory, concat_k2_preparation_import as reuse
    monkeypatch.setattr(inventory,"verify_snapshot",lambda *a:None)
    monkeypatch.setattr(reuse,"import_preparation",lambda *a:({"matching_recomputed":False},[]))
    monkeypatch.setattr(runtime,"foundation_lock",lambda s:campaign.artifact("FOUNDATION_LOCK",parent_rows=s["role_counts"]))
    monkeypatch.setattr(data,"build_population",lambda *a:campaign.artifact("PILOT_POPULATION",test_only=True))
    monkeypatch.setattr(runtime,"preflight",lambda *a:campaign.artifact("ACCEPTANCE",test_only=True))
    monkeypatch.setattr(runtime,"execution_gate",lambda *a,**kw:"123")
    monkeypatch.setattr(runtime,"cache_bounds",lambda s:{"train":1,"validation":1})
    monkeypatch.setattr(runtime,"prepare",lambda s,role,coordinate:make_cache(role,coordinate))
    def authenticate(s,name):
        write_immutable_json(root/"execution"/name/"123.json",{"test_only":True})
        return "123"
    monkeypatch.setattr(scheduler,"authenticate_job",authenticate)
    original = runtime.train_kernel
    def miniature(*a,**kw):
        assert kw["training_recipe"] == pilot.training_recipe()
        r,s = original(*a,**kw,acceptance_passes=1)
        return rehash(r,scientific_fit=True,acceptance_only=False),s
    monkeypatch.setattr(runtime,"train_kernel",miniature)
    for row in spec["tasks"]:
        done = runtime.run_task(spec,row["task_id"],device="cpu")
        assert runtime.completed(spec,row["task_id"]) == done
    assert len(runtime.result_rows(spec)) == 8
    assert runtime.completed(spec,"complete")["result"] == {"scientific_fits":8,"reducers":3,"sealed_test":True}
    parent = runtime.completed(spec,"train_HLT_X1_COMPRESSED")
    report = load_json(root/parent["result"]["training_report"])
    assert report["schema_version"] == 3
    assert report["teacher_lineage"]["teacher_node"] == "CONCAT_K2_D000"
    assert report["training"] == pilot.training_recipe()
    assert report["report_validation"]["rows"] == 22
    assert load_json(root/"outputs/foundation_lock/foundation_lock.json")["parent_rows"] == campaign.COUNTS


def test_real_pilot_import_and_population_do_not_rematch(completed_donor,tmp_path,monkeypatch):
    donor = completed_donor
    monkeypatch.setattr(pilot,"COUNTS",dict(donor["role_counts"]))
    old = target_spec(donor,tmp_path)
    fields = {k:v for k,v in old.items() if k not in {"contract","schema_version","content_hash"}}
    fields.update(campaign.registration("debug",pilot.PROFILE))
    fields["tasks"] = campaign.task_graph(fields["foundation"],reuse=True,profile=pilot.PROFILE)
    spec = campaign.artifact("CAMPAIGN_SPEC",**fields)
    campaign.validate_campaign(spec,check_source=False)
    root = Path(spec["campaign_root"])
    def forbidden(*a,**kw):
        raise AssertionError("Pilot must not recompute matching")
    monkeypatch.setattr(runtime,"assignment",forbidden)
    monkeypatch.setattr(runtime,"matcher_acceptance",forbidden)
    def authenticate(s,name):
        write_immutable_json(root/"execution"/name/"123.json",{"test_only":True})
        return "123"
    monkeypatch.setattr(scheduler,"authenticate_job",authenticate)
    for name in ("authenticate","import_preparation","foundation_lock","select_population"):
        runtime.run_task(spec,name,device="cpu")
        assert runtime.completed(spec,name)
    assert runtime.completed(spec,"import_preparation")["result"]["matching_recomputed"] is False
    assert data.population(spec)["role_counts"] == spec["role_counts"]
    for task in spec["foundation"]["assignment_tasks"]:
        suffix = Path("outputs")/f"assign_{task['file_index']:04d}"/"assignments.npz"
        assert (root/suffix).read_bytes() == (Path(donor["campaign_root"])/suffix).read_bytes()
    with pytest.raises(PermissionError,match="GPU acceptance"):
        runtime.science_gate(spec)
