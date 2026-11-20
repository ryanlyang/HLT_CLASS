"""Hbb epoch selection, pooled binary metrics, and sealed particle-role guards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import uproot
from hlt_classification.operations import sixclass_hbb_reselection as a
from hlt_classification.luka_sixclass_balanced_oscar import views, model, training, contracts as c
from hlt_classification.data.cache_contracts import sha256_file, atomic_publish_bytes, deterministic_npz_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.jetclass2_delphes.cache import RamCache


def probs(scores):
    scores=np.asarray(scores)
    return np.column_stack((1-scores, scores, np.zeros((len(scores),4))))


def test_binary_pair_score_excludes_other_signals_and_normalizes():
    labels=np.array([0,1,0,1,2])
    p=probs([.2,.6,.7,.8,.99])
    p[1,:2]*=.01;p[1,2]=.99  # Pair score remains .6; raw p_bb would be wrong.
    result=a.binary_metrics(labels,p)
    assert result['auc']==.75
    assert (result['background_count'],result['signal_count'])==(2,2)
    assert result['r50']['censored'] and result['r50']['rejection'] is None


def test_pooled_auc_is_not_mean_of_regional_aucs_and_ties_pass():
    y=np.array([0,1]);left=probs([.1,.2]);right=probs([.8,.9])
    assert a.binary_metrics(y,left)['auc']==a.binary_metrics(y,right)['auc']==1
    assert a.binary_metrics(np.tile(y,2),np.concatenate((left,right)))['auc']==.75
    r=a.binary_metrics(np.array([0,0,1,1]),probs([.4,.5,.5,.5]))
    assert r['r50']['background_passing']==1 and r['r50']['signal_passing']==2
    assert r['r50']['rejection']==2


@pytest.mark.parametrize('bad', ['nan','negative','zero_pair','missing_class','bad_sum'])
def test_bad_metrics_fail(bad):
    y=np.array([0,1]);p=probs([.2,.7])
    if bad=='nan':p[0,0]=np.nan
    elif bad=='negative':p[0,0]=-1
    elif bad=='zero_pair':p[0]=[0,0,1,0,0,0]
    elif bad=='missing_class':y[:]=0
    else:p[0]*=2
    with pytest.raises(ValueError):a.binary_metrics(y,p)


def test_independent_selection_earliest_tie_and_recovery():
    histories={s:[dict(epoch=e,metrics=dict(auc=.6+e/1000)) for e in range(1,21)] for s in a.MODELS}
    for s,epoch,auc in [('M0HLT',7,.8),('OFFLINE',11,.9),('DE000',17,.85)]:
        histories[s][epoch-1]['metrics']['auc']=auc
    histories['DE000'][19]['metrics']['auc']=.85
    picked={s:a.selected(r) for s,r in histories.items()}
    assert [picked[s]['epoch'] for s in a.MODELS]==[7,11,17]
    assert a.recovery({s:r['metrics']['auc'] for s,r in picked.items()})==pytest.approx(50)
    assert a.recovery(dict(M0HLT=.9,OFFLINE=.9,DE000=.95)) is None
    with pytest.raises(ValueError):a.selected(histories['M0HLT'][:-1])
    broken=deepcopy(histories['M0HLT']);broken[0]['epoch']=2
    with pytest.raises(ValueError):a.selected(broken)


@pytest.mark.parametrize('stage', ['M0HLT','DE000','OFFLINE'])
@pytest.mark.parametrize('workers', [1,4])
def test_validation_only_reader_exactly_matches_original_batches(tmp_path,monkeypatch,stage,workers):
    from test_luka_fullstat import write_root
    path=tmp_path/'raw/s.root';write_root(path,[161]*8)
    record=dict(path='raw/s.root',sha256=sha256_file(path),entries=8,tree_key='tree',category='signal')
    roles=np.array([0,1,2,3,1,0,1,2],np.uint8)
    member=dict(labels=np.zeros(8,np.int8),jet_nparticles=np.full(8,3),hlt_jet_nparticles=np.full(8,2),
                jet_pt=np.array([800,800,800,800,3000,800,800,800]))
    # Original operational code0=train,1=in-val,2=out-val,255=forbidden.
    original_codes=np.array([0,1,255,255,2,0,1,255],np.uint8)
    original,_=views.work((tmp_path,record,0,0,member,original_codes,c.COORDINATES[stage],'b'*64))
    arrays={r+'_'+k:getattr(original[r],k) for r in a.ROLES for k in ('identities','labels')}
    loaded=({'content_hash':'b'*64},{'files':[record]},[member],[roles],arrays,None)
    spec=dict(dataset=str(tmp_path),memory_mb=192000)
    real=uproot.behaviors.TBranch.HasBranches.arrays;reads=[]
    def guard(tree,expressions,*args,**kwargs):
        lo,hi=kwargs['entry_start'],kwargs['entry_stop']
        assert np.all(roles[lo:hi]==1)
        if stage!='OFFLINE':assert all(k.startswith('hlt_') for k in expressions)
        reads.append((lo,hi))
        return real(tree,expressions,*args,**kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches,'arrays',guard)
    caches=a.load_validation(spec,loaded,stage,workers=workers)
    assert (reads or workers==4) and set(caches)==set(a.ROLES)
    for r,cache in caches.items():
        gold=RamCache([original[r]],role='validation',foundation_sha256='b'*64,coordinate_name=c.COORDINATES[stage])
        for k,v in gold.batch(np.arange(len(gold))).items():
            assert np.array_equal(v,cache.batch(np.arange(len(cache)))[k])


@pytest.mark.parametrize('stage',['M0HLT','OFFLINE','DE000'])
def test_real_weaver_epoch_state_native_replay(tmp_path,stage):
    from test_luka_fullstat import write_root
    torch.set_num_threads(1)
    path=tmp_path/'raw/s.root';write_root(path,[161]*4)
    record=dict(path='raw/s.root',sha256=sha256_file(path),entries=4,tree_key='tree',category='signal')
    member=dict(labels=np.array([0,1,0,1]),jet_nparticles=np.full(4,3),hlt_jet_nparticles=np.full(4,2))
    blocks,_=views.work((tmp_path,record,0,0,member,np.array([1,1,2,2],np.uint8),c.COORDINATES[stage],'b'*64))
    cache=RamCache([blocks['validation']],role='validation',foundation_sha256='b'*64,coordinate_name=c.COORDINATES[stage])
    student=model.new_model(stage).eval()
    expected=training.predict(student,cache,'cpu')
    state=tmp_path/'state.pt';torch.save(dict(student=student.state_dict(),epoch=7,update=7*c.STEPS),state)
    receipt=dict(state=file_ref(state),epoch=7)
    native=a.native_from_epoch(stage,receipt,device='cpu')
    actual=training.predict(native,cache,'cpu')
    np.testing.assert_array_equal(actual,expected)
    assert all(not k.startswith(('predictors.','logit_heads.','matching.')) for k in native.state_dict())
    with pytest.raises(ValueError):a.native_from_epoch(stage,dict(receipt,epoch=8),device='cpu')


def test_saved_prediction_parity_checks_both_regions_and_ids(tmp_path):
    caches={r:SimpleNamespace(identities=np.arange(64,dtype=np.uint8).reshape(2,32),labels=np.array([0,1])) for r in a.ROLES}
    arrays={r+'_pt':np.array([800.,900.]) for r in a.ROLES}
    p={r:probs([.1,.8]) for r in a.ROLES};files={}
    for r,key in zip(a.ROLES,('predictions','outside_predictions')):
        path=tmp_path/(r+'.npz')
        atomic_publish_bytes(path,deterministic_npz_bytes(dict(identities=caches[r].identities,
            labels=caches[r].labels,pt=arrays[r+'_pt'],probabilities=p[r])))
        files[key]=file_ref(path)
    result=dict(receipt=dict(files=files))
    a.check_saved_predictions(result,caches,p,arrays)
    bad=deepcopy(p);bad['outside_validation'][0]=probs([.11])[0]
    with pytest.raises(ValueError,match='replay differs'):a.check_saved_predictions(result,caches,bad,arrays)


def test_receipts_bind_analysis_epoch_and_policy():
    r=a.artifact('EPOCH',dict(analysis='a'*64,epoch_commit='b'*64),epoch=1)
    a.validate(r,'EPOCH',r['parents'])
    with pytest.raises(ValueError):a.validate(r,'EPOCH',dict(analysis='c'*64))
    r['policy']=dict(r['policy'],population='in_range')
    with pytest.raises(ValueError):a.validate(r,'EPOCH',r['parents'])


def test_twenty_epoch_evaluation_resume_and_corrupt_receipt(tmp_path,monkeypatch):
    from hlt_classification.luka_sixclass_balanced_oscar import artifacts, reporting
    from hlt_classification.data.cache_contracts import load_json
    commits=[dict(epoch=e,content_hash=f'{e:064x}',row=dict(validation=dict(epoch=e))) for e in range(1,21)]
    monkeypatch.setattr(artifacts,'history',lambda *args,**kwargs:([],commits))
    evaluated=[]
    def native(stage,receipt,device):
        evaluated.append(receipt['epoch']);return receipt['epoch']
    monkeypatch.setattr(a,'native_from_epoch',native)
    monkeypatch.setattr(training,'predict',lambda epoch,cache,device:probs([.3,.5+epoch/1000]))
    monkeypatch.setattr(reporting,'evaluate_probabilities',lambda y,p:dict(epoch=int(round((p[1,1]-.5)*1000))))
    parity=[]
    monkeypatch.setattr(a,'check_saved_predictions',lambda *args:parity.append(True))
    monkeypatch.setenv('SLURM_JOB_ID','12345')
    result=dict(receipt=dict(fit_parents={}),report=dict(selected_pass=5))
    caches={r:SimpleNamespace(labels=np.array([0,1])) for r in a.ROLES}
    audit=dict(content_hash='a'*64)
    first=a.evaluate_stage({},audit,'M0HLT',result,caches,{},tmp_path)
    assert evaluated==[5]+[e for e in range(1,21) if e!=5] and len(first)==20 and parity==[True]
    evaluated.clear();parity.clear()
    second=a.evaluate_stage({},audit,'M0HLT',result,caches,{},tmp_path)
    assert evaluated==[5] and second==first and parity==[True]
    target=tmp_path/'M0HLT/20.json';broken=load_json(target);broken['metrics']['auc']=.3
    import json
    target.write_text(json.dumps(broken))
    with pytest.raises(ValueError):a.evaluate_stage({},audit,'M0HLT',result,caches,{},tmp_path)


def test_cli_imports_original_science_not_new_checkout():
    root=Path(__file__).resolve().parents[1]
    script=(root/'scripts/reselect_luka_sixclass_hbb.py').read_text()
    assert "Path(spec['project'])/'src'" in script and 'spec_from_file_location' in script
    queue=(root/'scripts/queue_luka_sixclass_hbb.sh').read_text()
    assert 'sbatch --test-only' in queue and queue.index('sbatch --test-only') < queue.index('sbatch --parsable')
    assert '--gres=gpu:l40s:1' in queue and 'scancel' not in queue


def test_real_queue_shell_dry_review_duplicate_and_resume(tmp_path):
    import os
    import subprocess
    import sys
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sp(path):
        p=Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    base=tmp_path/'base';project=tmp_path/'project'
    (project/'scripts').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts/queue_luka_sixclass_hbb.sh'
    script=project/'scripts'/source.name
    script.write_bytes(source.read_text().replace('/oscar/home/rlyang/hlt_classification',sp(base)).encode())
    spec=base/'checkpoints/luka_sixclass_balanced_oscar_8a25f126_r1/campaign_spec.json'
    spec.parent.mkdir(parents=True);spec.write_text('{}')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    bodies={
        'git':'if [[ "$*" == *"rev-parse HEAD"* ]]; then printf "%040d\\n" 1; fi\n',
        'squeue':':\n', 'sacct':'echo "FAILED|"\n',
        'sbatch':'[[ -z "${SBATCH_QOS:-}" ]] || exit 3\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n'
                 'if [[ "$*" == *"--test-only"* ]]; then echo ESTIMATE; else echo "$FIXTURE_JOB"; fi\n'}
    for name,body in bodies.items():
        path=bindir/name;path.write_bytes(('#!/usr/bin/env bash\n'+body).encode());path.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls),
             FIXTURE_JOB='12345',SBATCH_QOS='unwanted')
    args=[bash,str(script),'0'*39+'1']
    first=subprocess.run(args,env=env,capture_output=True,text=True,timeout=15)
    assert first.returncode==0,first.stderr
    assert 'QUEUED validation-only' in first.stdout
    assert subprocess.run(args,env=env,capture_output=True).returncode!=0
    assert len(calls.read_text().splitlines())==2
    resumed=subprocess.run(args+['resume'],env=dict(env,FIXTURE_JOB='12346'),capture_output=True,text=True,timeout=15)
    assert resumed.returncode==0,resumed.stderr
    lines=calls.read_text().splitlines()
    assert len(lines)==4 and all('--test-only' in lines[i] for i in (0,2))
