from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.bridge import (
    Particles, from_cms, from_jc2, p4_from_coordinates, wrap_phi,
)
from hlt_classification.cms2jc2_response.contracts import (
    artifact, candidates, compatibility_template, validate, validate_compatibility, with_content_hash,
)
from hlt_classification.cms2jc2_response.rng import categorical, group_key, normals, uniform
from hlt_classification.cms2jc2_response.splits import _allocate, pack_entries, unpack_entries


@pytest.fixture
def review():
    """Synthetic-only producer evidence, never accepted as remote/readiness proof."""
    value = compatibility_template(inventory_hash="a" * 64)
    for row in value["decisions"].values():
        row.update(status="verified", decision="Synthetic fixture convention", evidence=[dict(
            source="synthetic_test_fixture", locator="test_cms2jc2_response", sha256="b" * 64)])
    value.update(reviewer="synthetic fixture", cms_length_to_mm=10., jc2_length_to_mm=1.,
                 momentum_to_gev=1., cms_d0_sign=1, jc2_d0_sign=-1)
    return with_content_hash(value)


def particles(pt=(1., 2.), eta=(0., .02), phi=(0., 0.), *, keys=None):
    n = len(pt)
    return Particles(p4_from_coordinates(pt, eta, phi, np.zeros(n)), np.ones(n), np.zeros(n),
                     np.zeros((n, 4)), np.zeros((n, 4), bool),
                     tuple(str(i) for i in range(n)) if keys is None else keys)


def test_registry_and_artifact_failure():
    rows = candidates()
    assert validate(rows, "CANDIDATES")
    assert [r["id"] for r in rows["candidates"]] == [f"{f}_{l}" for f in "ABC" for l in "LMH"]
    assert rows["primary_fits"] == 27 and rows["sensitivity_fits"] == 6
    rows["pruning"] = True
    with pytest.raises(ValueError, match="hash"):
        validate(rows, "CANDIDATES")
    with pytest.raises(ValueError):
        artifact("TEST", final_test_accessed=True)


def test_compatibility_is_not_assumed(review):
    assert validate_compatibility(review, inventory_hash="a" * 64)
    pending = compatibility_template(inventory_hash="a" * 64)
    with pytest.raises(PermissionError, match="Unresolved"):
        validate_compatibility(pending)
    pending = dict(review, reviewer=None)
    with pytest.raises(PermissionError, match="Human"):
        validate_compatibility(with_content_hash(pending))
    with pytest.raises(ValueError, match="inventory"):
        validate_compatibility(review, inventory_hash="c" * 64)


def test_keyed_rng_order_independence_and_domain_isolation():
    keys = ["jet_A", "jet_B", "jet_C"]
    draws = {k: normals(k, 0, "kinematics", "p:2", 12).tobytes() for k in keys}
    assert draws == {k: normals(k, 0, "kinematics", "p:2", 12).tobytes() for k in reversed(keys)}
    assert group_key(["1", "2"]) == group_key(["2", "1"])
    assert group_key(["12", "3"]) != group_key(["1", "23"])
    for i in range(100):
        assert 0 < uniform("j", 0, "identity", "1", i) < 1
    assert normals("j", 0, "tracking", "1", 4).tolist() != normals("j", 1, "tracking", "1", 4).tolist()
    with pytest.raises(ValueError):
        uniform("j", 0, "jet_label", "1")
    assert categorical([0, 1, 0], "j", 0, "identity", "1") == 1


def test_physicality_empty_and_immutable_particles():
    p = particles()
    assert np.allclose(p.pt, [1., 2.])
    assert len(p.take([])) == 0
    assert np.allclose(wrap_phi([np.pi + .01]), [-np.pi + .01])
    with pytest.raises(ValueError):
        p.p4[0, 0] = 6
    with pytest.raises(ValueError, match="Spacelike"):
        replace(p, p4=np.array([[2, 0, 0, 1], [2, 0, 0, 2.]]))
    with pytest.raises(ValueError, match="Lossy"):
        replace(p, charge=np.array([.5, 1.]))
    with pytest.raises(ValueError, match="applicability"):
        replace(p, tracking=np.ones((2, 4)))


def test_jc2_bridge_no_hlt_or_labels_and_missing_error(review):
    p = particles()
    cols = dict(zip(("px", "py", "pz", "energy"), p.p4.T))
    cols.update(charge=np.ones(2), isChargedHadron=np.ones(2), isNeutralHadron=np.zeros(2),
                isPhoton=np.zeros(2), isElectron=np.zeros(2), isMuon=np.zeros(2),
                d0val=np.array([.2, .3]), dzval=np.array([.4, .5]),
                d0err=np.array([0., .1]), dzerr=np.array([.2, 0.]))
    a = from_jc2(cols, review, keys=p.keys)
    assert a.tracking[0, 0] == -.2
    assert a.tracking[0, 2] == 0 and not a.valid[0, 2]
    assert a.valid[0, 0]  # An absent error does not erase a measured displacement.
    cols.update(jet_label="forbidden", hlt_part_px=np.array([np.nan]))
    b = from_jc2(cols, review, keys=p.keys)
    assert a.p4.tobytes() == b.p4.tobytes() and a.tracking.tobytes() == b.tracking.tobytes()


def test_cms_bridge_regular_pf_uncertainty_and_ambiguous_pid(review):
    p = particles()
    cols = {"cpfcandlt_"+f: v for f,v in zip(("px", "py", "pz", "energy"), p.p4.T)}
    cols.update({"cpfcandlt_"+f: np.asarray(v) for f,v in dict(
        charge=[1,1], isChargedHad=[1,1], isEl=[1,0], isMu=[0,0], isLostTrack=[0,1],
        dxy=[.2,.3], dxysig=[2,0], dz=[0,.4], dzsig=[0,2]).items()})
    for field in ("px", "py", "pz", "energy", "isNeutralHad", "isGamma"):
        cols["npfcand_"+field] = np.empty(0)
    result = from_cms(cols, review, side="offline")
    assert len(result) == 1 and result.category[0] == 5
    assert result.keys == ("cpfcandlt:0",)
    assert result.tracking[0, 0] == 2 and result.tracking[0, 2] == 1
    assert not result.valid[0, 3]


def test_canonical_masks_and_padding():
    value = pack_entries([0, 3, 8], 10)
    assert unpack_entries(value, 10).tolist() == [0, 3, 8]
    with pytest.raises(ValueError):
        pack_entries([1, 1], 10)
    with pytest.raises(ValueError):
        unpack_entries("//8=", 10)


def test_grouped_minimax_matches_exhaustive_and_is_replayable():
    files = [dict(sha256=f"{i:064x}", source="AB"[i%2], selected_entries=c)
             for i,c in enumerate([5, 6, 7, 8, 9, 10])]
    result, scores = _allocate(files, (.5, .25, .25), (1,1,1))
    best = float("inf")
    total = sum(f["selected_entries"] for f in files)
    for assignment in itertools.product(range(3), repeat=6):
        if any({files[i]["source"] for i,r in enumerate(assignment) if r == k} != {"A", "B"}
               for k in range(3)):
            continue
        counts = np.array([sum(f["selected_entries"] for f,r in zip(files, assignment) if r == k)
                           for k in range(3)])
        best = min(best, np.max(np.abs(counts / (np.array([.5,.25,.25])*total)-1)))
    assert scores["minimax"] == pytest.approx(best)
    assert np.array_equal(result, _allocate(files, (.5,.25,.25), (1,1,1))[0])
    with pytest.raises(ValueError, match="capacity"):
        _allocate(files, (.8,.1,.1), (2_000_000, 250_000, 250_000))


def test_offline_reader_rejects_test_before_opening_anything(review, tmp_path):
    from hlt_classification.cms2jc2_response.readers import iter_jc2_offline
    with pytest.raises(PermissionError, match="sealed"):
        next(iter_jc2_offline(tmp_path, {}, {}, review, role="final_test"))


def test_header_audit_reads_no_particle_arrays(tmp_path):
    import awkward as ak
    import uproot
    from hlt_classification.cms2jc2_response.audit import inspect_headers
    with uproot.recreate(tmp_path / "sample.root") as f:
        f.mktree("tree", {"example": "var * float32"}).extend({"example": ak.Array([[1., 2.]])})
    result = inspect_headers(tmp_path)
    assert result["particle_arrays_read"] is False
    assert result["authenticated_for_fitting"] is False
    assert result["files"][0]["missing_branches"]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_association_identity_and_dustbins(count):
    from hlt_classification.cms2jc2_response.association import associate, policy
    p = particles(tuple(1.+i for i in range(count)), tuple(.3*i for i in range(count)), (0.,)*count)
    result = associate(p, p, policy())
    assert result["resolved"] and result["optimum_cost"] == 0
    assert all(len(h["offline"]) == len(h["hlt"]) == 1 for h in result["hypotheses"])
    result = associate(p, particles((), (), ()), policy())
    assert result["optimum_cost"] == count
    assert all(h["hlt"] == () for h in result["hypotheses"])


def test_association_merge_split_and_budget_is_deterministic():
    from hlt_classification.cms2jc2_response.association import associate, policy
    two = particles((1.,1.), (0.,.01), (0.,0.))
    summed = two.p4.sum(axis=0, keepdims=True)
    one = Particles(summed, np.ones(1), np.zeros(1), np.zeros((1,4)), np.zeros((1,4),bool), ("s",))
    for a,b in ((two,one),(one,two)):
        out = associate(a,b,policy())
        assert out["resolved"] and out["optimum_cost"] == .5
        assert len(out["hypotheses"]) == 1
    limited = policy(search_nodes=1)
    out = associate(two, one, limited)
    assert not out["resolved"] and out["optimum_cost"] is None
    assert out == associate(two,one,limited)
    wrong = with_content_hash(dict(policy(), group_penalty=0.))
    with pytest.raises(ValueError, match="Unregistered"):
        associate(two,one,wrong)


def test_association_agrees_with_exhaustive_small_set_cover():
    from hlt_classification.cms2jc2_response.association import associate, hypotheses, policy
    a = particles((1.,2.), (0.,.02), (0.,0.))
    b = particles((1.2,1.5), (.01,.03), (0.,0.))
    hs = hypotheses(a,b,policy())
    best = float("inf")
    for keep in itertools.product((False, True), repeat=len(hs)):
        o, h, cost = [], [], 0.
        for use, hypothesis in zip(keep,hs):
            if use:
                o.extend(hypothesis.offline); h.extend(hypothesis.hlt); cost += hypothesis.cost
        if sorted(o) == [0,1] and sorted(h) == [0,1]:
            best = min(best,cost)
    assert associate(a,b,policy())["optimum_cost"] == pytest.approx(best)


def test_feature_schema_and_fit_only_support():
    from hlt_classification.cms2jc2_response.features import features, fit_preprocessing, transform, FEATURE_NAMES
    x = features(particles())
    assert x.shape == (2,len(FEATURE_NAMES)) and np.isfinite(x).all()
    single = features(particles((2.,), (0.,), (0.,)))
    assert single[0,26:28].tolist() == [1.,1.]
    pre = fit_preprocessing(np.vstack([x]*10), membership_hash="d"*64)
    normalized, clamped = transform(x,pre,family="smooth")
    assert normalized.shape == clamped.shape == x.shape
    assert features(particles((),(),())).shape == (0,len(FEATURE_NAMES))


@pytest.mark.parametrize("candidate", ["A_L","B_L","C_L"])
def test_conditional_families_portable_roundtrip(candidate):
    import json
    from hlt_classification.cms2jc2_response.features import features
    from hlt_classification.cms2jc2_response.families import fit_conditional, predict
    x = np.vstack([features(particles())]*20)
    y = np.full((len(x),1),.25)
    fitted = fit_conditional(x,y,np.ones(len(x)),candidate_id=candidate,task="continuous",membership_hash="a"*64)
    assert fitted["candidate"]["id"] == candidate
    p = predict(x,fitted)
    np.testing.assert_allclose(p,.25,atol=1e-7)
    portable = json.loads(json.dumps(fitted,allow_nan=False))
    assert predict(x,portable).tobytes() == p.tobytes()
    if candidate.startswith("C"):
        assert len(fitted["model"]["trees"]) == 200


def test_temperature_requires_separate_residual_role():
    from hlt_classification.cms2jc2_response.features import features
    from hlt_classification.cms2jc2_response.families import fit_conditional, predict, calibrate_temperature
    x = np.vstack([features(particles())]*20)
    y = np.arange(len(x))%2
    fitted = fit_conditional(x,y,np.ones(len(x)),candidate_id="A_L",task="categorical",classes=2,membership_hash="a"*64)
    assert np.allclose(predict(x,fitted).sum(axis=1),1.)
    with pytest.raises(PermissionError):
        calibrate_temperature(fitted,x,y,np.ones(len(x)),residual_membership_hash="a"*64)
    calibrated = calibrate_temperature(fitted,x,y,np.ones(len(x)),residual_membership_hash="b"*64)
    assert calibrated["temperature_calibrated"]


def test_simple_family_preserves_category_tracking_slope():
    from hlt_classification.cms2jc2_response.features import features
    from hlt_classification.cms2jc2_response.families import fit_conditional, predict
    x = np.vstack([features(particles())]*20)
    x[:,5] = np.arange(len(x))/10
    x[:,9] = 1
    y = (2*x[:,5]+.5)[:,None]
    fitted = fit_conditional(x,y,np.ones(len(x)),candidate_id="A_L",task="continuous",
                            tracking_linear=True,membership_hash="a"*64)
    np.testing.assert_allclose(predict(x,fitted),y,atol=1e-8)


def test_residual_copula_does_not_double_count_jet_variance():
    from hlt_classification.cms2jc2_response.residuals import fit_cell
    gen = np.random.default_rng(27)
    ids = np.repeat(np.arange(1200),3)
    shared = np.repeat(gen.normal(size=(1200,2)),3,axis=0)
    residual = shared*.6+gen.normal(size=shared.shape)*.8
    cell = fit_cell(residual,np.ones(len(ids)),ids)
    total = np.asarray(cell["correlation"])
    common = np.asarray(cell["shared_covariance"])
    independent = np.asarray(cell["independent_covariance"])
    np.testing.assert_allclose(common+independent,total,atol=1e-10)
    assert np.linalg.eigvalsh(common).min() > -1e-9
    assert np.linalg.eigvalsh(independent).min() > -1e-9
    assert np.diag(common).min() > 0


def test_residual_backend_replay_sparse_support_and_leakage():
    from hlt_classification.cms2jc2_response.residuals import fit_backend, sample
    errors = np.column_stack((np.linspace(-1,1,1200),np.linspace(0,2,1200)))
    conditions = np.column_stack((np.zeros(1200),np.ones(1200),np.ones(1200),np.zeros(1200)))
    args = dict(location_membership_hash="a"*64,residual_membership_hash="b"*64,
                with_crowding=True,edges={"pt":[0.],"eta":[.5],"crowding":[1.]},coordinates=["pt","eta"])
    fitted = fit_backend(errors,np.ones(1200),np.arange(1200),conditions,**args)
    params = dict(jet="j",replica=0,object_key="p:1",module="test")
    value,flags = sample(fitted,conditions[0],**params)
    assert flags["supported"]
    assert value.tobytes() == sample(fitted,conditions[0],**params)[0].tobytes()
    assert sample(fitted,[5,1,1,0],**params)[0] is None
    args["residual_membership_hash"] = "a"*64
    with pytest.raises(PermissionError):
        fit_backend(errors,np.ones(1200),np.arange(1200),conditions,**args)


def _prep_spec(tmp_path):
    from hlt_classification.cms2jc2_response.preparation import SITE, command_plan
    from hlt_classification.cms2jc2_response.contracts import publish
    spec = artifact("PREPARATION_SPEC",project_dir=str(tmp_path),campaign_root=str(tmp_path),site=SITE,
                    source={"content_hash":"a"*64},scientific_fits=0,automatic_science_submission=False)
    plan = command_plan(spec)
    publish(tmp_path/"command_plan.json",plan,"COMMAND_PLAN")
    publish(tmp_path/"dry_run_submission_ledger.json",artifact("DRY_LEDGER",parents={"spec":spec["content_hash"],
            "plan":plan["content_hash"]},argv=plan["argv"],dry_run=True),"DRY_LEDGER")
    return spec,plan


def test_preparation_dry_run_no_subprocess_or_mutation(tmp_path,monkeypatch):
    from hlt_classification.cms2jc2_response import preparation as module
    spec,plan = _prep_spec(tmp_path)
    monkeypatch.setattr(module,"validate_spec",lambda value: None)
    def fail(*args,**kwargs):
        pytest.fail("Dry submission called an external process")
    monkeypatch.setattr(module.subprocess,"run",fail)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert module.submit(spec)["dry_run"]
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not any("gres" in x or "gpus" in x for x in plan["argv"])
    assert "--partition=debug" in plan["argv"]
    assert "--qos=qos_tier3" in plan["argv"]
    assert plan["scientific_fits"] == 0
    with pytest.raises(PermissionError):
        module.submit(spec,execute=True,authorization_phrase="science")


def test_preparation_receipt_and_ambiguous_submission(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.cms2jc2_response import preparation as module
    spec,_ = _prep_spec(tmp_path)
    monkeypatch.setattr(module,"validate_spec",lambda value: None)
    monkeypatch.setattr(module.shutil,"disk_usage",lambda path: SimpleNamespace(free=100*1024**3))
    calls = []
    def run(argv,**kwargs):
        calls.append(argv)
        if argv[0] == "scontrol":
            assert argv == ["scontrol", "show", "partition", "debug", "-o"]
        return SimpleNamespace(returncode=0,stderr="",stdout=("PartitionName=debug State=UP" if argv[0]=="scontrol" else "12125;sporc\n"))
    monkeypatch.setattr(module.subprocess,"run",run)
    result = module.submit(spec,execute=True,authorization_phrase=module.AUTHORIZATION)
    assert result["jobs"] == {"read_only_preparation":"12125"}
    assert module.submit(spec,execute=True,authorization_phrase=module.AUTHORIZATION) == result
    assert sum(c[0]=="sbatch" for c in calls) == 1
    other = tmp_path/"lost"; other.mkdir(); spec,_ = _prep_spec(other)
    def ambiguous(argv,**kwargs):
        return SimpleNamespace(returncode=0,stderr="",stdout=("PartitionName=debug State=UP" if argv[0]=="scontrol" else "Acknowledgement lost"))
    monkeypatch.setattr(module.subprocess,"run",ambiguous)
    with pytest.raises(RuntimeError,match="ambiguous"):
        module.submit(spec,execute=True,authorization_phrase=module.AUTHORIZATION)
    assert (other/"submission_intent"/"intent.json").is_file()
    with pytest.raises(FileExistsError):
        module.submit(spec,execute=True,authorization_phrase=module.AUTHORIZATION)


def test_cpu_worker_and_readiness_are_not_full_science_claims():
    root = Path(__file__).resolve().parents[1]
    worker = (root/"sbatch/run_cms2jc2_response_cpu.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/cms2jc2_response.py"' in worker
    assert "PYTHONNOUSERSITE=1" in worker and "LD_LIBRARY_PATH" in worker
    assert "run-preparation" in worker and "scancel" not in worker


@pytest.mark.parametrize("candidate",["A_L","B_L","C_L"])
def test_families_learn_nonconstant_known_response(candidate):
    from hlt_classification.cms2jc2_response.features import features
    from hlt_classification.cms2jc2_response.families import fit_conditional,predict
    base = features(particles((1.,),(0.,),(0.,)))
    x = np.repeat(base,12000,axis=0)
    x[:,2] = np.linspace(-2,2,len(x)); x[:,4] = x[:,2]
    target = (.4*x[:,2]+.2)[:,None]
    fitted = fit_conditional(x,target,np.ones(len(x)),candidate_id=candidate,task="continuous",membership_hash="a"*64)
    held = np.repeat(base,301,axis=0)
    held[:,2] = np.linspace(-1.9,1.9,len(held)); held[:,4] = held[:,2]
    truth = (.4*held[:,2]+.2)[:,None]
    error = np.mean((predict(held,fitted)-truth)**2)
    assert error < .45*np.var(truth)
    assert predict(held[:0],fitted).shape == (0,1)


def test_jc2_root_reader_works_without_any_hlt_or_label_branches(tmp_path,review,monkeypatch):
    import awkward as ak
    import uproot
    from hlt_classification.cms2jc2_response.bridge import JC2_FIELDS
    from hlt_classification.cms2jc2_response.readers import iter_jc2_offline
    from hlt_classification.jetclass2_delphes import inventory as inv_module, splits as split_module
    from hlt_classification.jetclass2_delphes.split_registry import pack_entries as jc2_pack
    p = particles()
    columns = dict(zip(("px","py","pz","energy"),p.p4.T))
    columns.update(charge=np.ones(2),isChargedHadron=np.ones(2),isNeutralHadron=np.zeros(2),
                   isPhoton=np.zeros(2),isElectron=np.zeros(2),isMuon=np.zeros(2),
                   d0val=np.zeros(2),dzval=np.zeros(2),d0err=np.zeros(2),dzerr=np.zeros(2))
    path = tmp_path/"offline.root"
    with uproot.recreate(path) as handle:
        tree = handle.mktree("tree",{"jet_nparticles":"int32",**{"part_"+f:"var * float64" for f in JC2_FIELDS}})
        tree.extend({"jet_nparticles":np.array([2],np.int32),**{"part_"+f:ak.Array([columns[f]]) for f in JC2_FIELDS}})
    from hlt_classification.cms2jc2_response.contracts import sha256_file
    inv = {"files":[dict(path="offline.root",tree_key="tree;1",entries=1,sha256=sha256_file(path))]}
    profile = {"profile":"TRAIN_500K","groups":[dict(path="offline.root",role="train")],"memberships":{"train":{"files":[dict(
        path="offline.root",entry_mask=jc2_pack([0],1))]}}}
    # Only metadata authentication is synthetic here. The ROOT read and branch
    # capability are real, so any native-HLT/label request fails this test.
    monkeypatch.setattr(inv_module,"validate_inventory",lambda value:"d"*64)
    monkeypatch.setattr(inv_module,"verify_file",lambda *args:path)
    monkeypatch.setattr(split_module,"validate_splits",lambda *args:None)
    monkeypatch.setattr(split_module,"is_subset_profile",lambda *args:True)
    rows = list(iter_jc2_offline(tmp_path,inv,profile,review,role="train"))
    assert len(rows) == 1 and rows[0].hlt is None and len(rows[0].offline) == 2


def test_memberships_require_complete_nested_exact_budget_masks():
    from copy import deepcopy
    from hlt_classification.cms2jc2_response.splits import validate_memberships
    files = [dict(path=role+".root", fit_role=role, raw_entries=n,
                  selected_entries=n, entry_mask=pack_entries(np.arange(n), n))
             for role,n in (("fit_location",1_000_000),("fit_residual",250_000))]
    roles = artifact("ROLES", files=files)
    budgets = {}
    for name, fraction in (("250K",.2),("1M",.8),("FULL",1.)):
        budgets[name] = {f["fit_role"]:[dict(path=f["path"], selected_entries=int(f["raw_entries"]*fraction),
            entry_mask=pack_entries(np.arange(int(f["raw_entries"]*fraction)),f["raw_entries"]))] for f in files}
    good = artifact("MEMBERSHIPS",parents={"roles":roles["content_hash"]},budgets=budgets)
    validate_memberships(good,roles)
    bad = deepcopy(good); bad["budgets"]["250K"]["fit_location"] = []
    with pytest.raises(ValueError,match="coverage"):
        validate_memberships(with_content_hash(bad),roles)
    bad = deepcopy(good)
    bad["budgets"]["1M"]["fit_location"][0]["entry_mask"] = pack_entries(np.arange(200_000,1_000_000),1_000_000)
    with pytest.raises(ValueError,match="nesting"):
        validate_memberships(with_content_hash(bad),roles)
    bad = with_content_hash(dict(good,schema_version=2))
    with pytest.raises(ValueError,match="version|registry"):
        validate_memberships(bad,roles)
