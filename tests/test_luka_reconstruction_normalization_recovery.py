"""Frozen TRAIN moments, exact legacy recovery and narrowly bound continuation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np
import pytest
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash,sha256_file
from hlt_classification.luka_balanced_reconstruction_hydra import normalization as norms,contracts as c,inputs,population
from hlt_classification.operations import hydra_reconstruction_normalization_recovery as op
from test_luka_fullstat import write_root


def fixture():
    spec=dict(root='/study',content_hash='a'*64,maximum=4,population=dict(content_hash='b'*64,
        counts=dict(train=120,validation=12),particles=dict(train=dict(offline=360))))
    record=c.artifact('TARGET_NORMALIZATION',parents=dict(population='b'*64),role='train',
        mean=[0.]*22,scale=[1.]*22,support=[360]*22,maximum=4,rows=120,tokens=360,
        frame='paired_HLT_sum_p4_axis_and_vector_pt')
    return spec,record


def test_captured_r22_record_matches_exact_qualification_and_is_launcher_default():
    project=Path(__file__).resolve().parents[1]
    relative='configs/operations/balanced_reco_dc5920f0_normalization.json'
    record=op.read(project/relative)
    spec=dict(maximum=316,population=dict(content_hash=record['parents']['population'],
        counts=dict(train=600000),particles=dict(train=dict(offline=38397921))))
    assert norms.validate(record,spec,expected_hash=op.NORMAL_HASH) is record
    assert relative in (project/'scripts/queue_luka_reconstruction_normalization_recovery.sh').read_text()


def test_only_frozen_values_are_applied_when_hashes_differ_by_numerical_roundoff(tmp_path):
    spec,record=fixture();observed=deepcopy(record);observed['mean'][0]=2e-13;observed=with_content_hash(observed)
    evidence=tmp_path/'comparison.json'
    assert observed['content_hash']!=record['content_hash']
    assert norms.choose(spec,observed,record,evidence=evidence) is record
    row=load_json(evidence);assert row['observed']==observed and row['frozen']==record
    assert row['applied_sha256']==record['content_hash'] and row['moments_within_tolerance']


@pytest.mark.parametrize('field,value',[('mean',.01),('scale',.01),('support',-1),('maximum',1),('rows',1),('role','validation')])
def test_substantive_or_scope_drift_cannot_be_hidden(tmp_path,field,value):
    spec,record=fixture();other=deepcopy(record)
    if field in ('mean','scale','support'):other[field][0]+=value
    else:other[field]=value
    other=with_content_hash(other)
    with pytest.raises(ValueError):norms.choose(spec,other,record,evidence=tmp_path/'evidence.json')
    row=load_json(tmp_path/'evidence.json');assert row['applied_sha256'] is None
    assert row['observed']==other and row['frozen']==record


def test_frozen_record_hash_and_published_file_remain_exact(tmp_path):
    spec,record=fixture();spec['root']=str(tmp_path)
    with pytest.raises(ValueError,match='qualified hash'):norms.validate(record,spec,expected_hash='c'*64)
    ref=norms.publish(spec,record);assert load_json(ref['path'])==record
    assert norms.publish(spec,record)==ref
    Path(ref['path']).write_text('{}')
    with pytest.raises(ValueError):norms.publish(spec,record)


def test_real_root_builder_applies_frozen_FP64_statistics_before_float32_cast(tmp_path,monkeypatch):
    path=tmp_path/'raw/test.root';write_root(path,[1]*492)
    record=dict(path='raw/test.root',sha256=sha256_file(path),entries=492,tree_key='tree',category='signal')
    labels=np.tile(np.arange(6),82);pt=np.tile(np.repeat(np.arange(650,2600,100),6),5)[:492]
    roles=np.r_[np.zeros(240,np.uint8),np.ones(240,np.uint8),np.full(12,2,np.uint8)]
    member=dict(labels=labels,jet_pt=pt,jet_nparticles=np.full(492,3),hlt_jet_nparticles=np.full(492,2))
    arrays={}
    for role,ix in [('train',np.arange(240)),('validation',np.arange(240,480))]:
        arrays[role+'_identities']=np.array([np.frombuffer(bytes.fromhex(population.donor.original.identity(record,int(i))),np.uint8) for i in ix])
        arrays[role+'_labels']=labels[ix];arrays[role+'_pt']=pt[ix]
    parent=(dict(content_hash='a'*64,counts=dict(train=240,validation=240)),dict(files=[record]),[member],[roles],arrays,None)
    monkeypatch.setattr(c,'COUNTS',dict(train=120,validation=120))
    selected,fields=population.describe(parent);p=c.artifact('POPULATION',parents=dict(parent='a'*64),**fields)
    spec=dict(root=str(tmp_path/'study'),dataset=str(tmp_path),content_hash='b'*64,population=p,maximum=3)
    baseline=inputs.prepare_node(spec,tmp_path/'baseline','D000',targets=True,workers=1,loaded=(p,parent,selected))
    fixed=deepcopy(baseline['normalization']);fixed['mean'][0]+=5e-13;fixed=with_content_hash(fixed)
    captured=[];measured=inputs.measured
    def observe(raw,frame):
        value,mask=measured(raw,frame);captured.append(value.copy());return value,mask
    monkeypatch.setattr(inputs,'measured',observe)
    repaired=inputs.prepare_node(spec,tmp_path/'repaired','D000',targets=True,workers=1,loaded=(p,parent,selected),
        frozen_normalization=fixed,normalization_evidence=tmp_path/'moments.json')
    assert repaired['normalization']==fixed
    actual=np.load(tmp_path/'repaired/target_values.npy')
    expected=((np.concatenate(captured)-np.asarray(fixed['mean']))/np.asarray(fixed['scale'])).astype(np.float32)
    assert np.array_equal(actual,expected)
    assert load_json(tmp_path/'moments.json')['frozen']==fixed
    for name in ('train_features','train_vectors','train_offsets','train_identities','validation_features'):
        assert (tmp_path/f'baseline/{name}.npy').read_bytes()==(tmp_path/f'repaired/{name}.npy').read_bytes()
    isolated=op.load_file('hlt_classification.luka_balanced_reconstruction_hydra._repair_test_inputs',Path(inputs.__file__))
    isolated.decoded=inputs.decoded
    isolated.prepare_node(spec,tmp_path/'parallel','D000',targets=True,workers=4,loaded=(p,parent,selected),
        frozen_normalization=fixed,normalization_evidence=tmp_path/'parallel_moments.json')
    for path in (tmp_path/'repaired').glob('*.npy'):
        assert path.read_bytes()==(tmp_path/'parallel'/path.name).read_bytes()


def test_recovery_bootstrap_replays_on_original_node_and_release_waits_for_success():
    spec=dict(root='/repair',project='/new',content_hash='a'*64,qualification_node='gpu1905',
        parent_spec=dict(path='/parent'))
    from unittest.mock import patch
    with patch.object(op,'referenced',lambda r:Path(r['path'])),patch.object(op,'read',lambda p:dict(project='/old')):
        rows=op.bootstrap_commands(spec)
        candidate_rows=op.bootstrap_commands(dict(spec,candidate=dict(path='/candidate')))
    assert '--nodelist=gpu1905' in rows[0]['command']
    assert '--gres=gpu:geforce_gtx_1080_ti:1' in rows[0]['command']
    assert '--time=240' in rows[0]['command']
    assert '--dependency=afterok:{normalize}' in rows[1]['command']
    assert not any(a.startswith('--dependency=') for a in candidate_rows[0]['command'])
    assert '--partition=compute' in candidate_rows[0]['command']
    assert '--dependency=afterok:{normalize}' in candidate_rows[1]['command']


def test_only_campaign_bound_pending_reconstruction_and_report_are_cancelled(tmp_path,monkeypatch):
    rows=[dict(job=str(i),task=t) for i,t in enumerate(('R24_s10','D20_s10','U100_s11','report'),1)]
    spec=dict(root=str(tmp_path),content_hash='a'*64,original_jobs={r['job']:r['task'] for r in rows})
    old=SimpleNamespace(spec=dict(root='/old',content_hash='b'*64),c=c,
        campaign=SimpleNamespace(accepted_jobs=lambda s:rows))
    monkeypatch.setattr(op,'states',lambda jobs:{j:('RUNNING' if j=='2' else 'PENDING','0:0') for j in jobs})
    checked=[];cancelled=[]
    def guard(spec,row):checked.append(row['job']);return dict(JobState='PENDING')
    monkeypatch.setattr(op,'pending_fields',guard)
    monkeypatch.setattr(op.subprocess,'run',lambda argv,**kw:cancelled.append(argv))
    op.cancel_pending(spec,old)
    assert checked==['1','4'] and cancelled==[['scancel','--state=PENDING','1'],['scancel','--state=PENDING','4']]
    assert not (tmp_path/'scheduler/2.intent.json').exists()


def test_cancel_race_does_not_cancel_a_newly_running_job(tmp_path,monkeypatch):
    row=dict(job='1',task='R24_s10');spec=dict(root=str(tmp_path),content_hash='a'*64,original_jobs={'1':'R24_s10'})
    old=SimpleNamespace(spec={},c=c,campaign=SimpleNamespace(accepted_jobs=lambda s:[row]))
    monkeypatch.setattr(op,'states',lambda jobs:{'1':('PENDING','0:0')})
    monkeypatch.setattr(op,'pending_fields',lambda *a:dict(JobState='RUNNING'))
    monkeypatch.setattr(op.subprocess,'run',lambda *a,**kw:pytest.fail('Running job cancelled'))
    op.cancel_pending(spec,old)


def test_frozen_replay_requires_original_qualified_hash_and_preserves_mismatch(tmp_path,monkeypatch):
    spec0,record=fixture();spec0['root']=str(tmp_path/'old')
    spec=dict(root=str(tmp_path/'repair'),content_hash='d'*64,candidate=dict(path='/candidate'),
        normalization_sha256='0'*64,qualification_node='gpu1905')
    old=SimpleNamespace(spec=spec0,q=dict(content_hash='c'*64))
    monkeypatch.setattr(op,'check',lambda s:old);monkeypatch.setattr(op,'bound',lambda *a:'123')
    monkeypatch.setattr(op,'referenced',lambda r:Path(r['path']))
    original_read=op.read;monkeypatch.setattr(op,'read',lambda p:record if Path(p)==Path('/candidate') else original_read(p))
    with pytest.raises(ValueError,match='no fits released'):op.normalize(spec)
    assert load_json(tmp_path/'repair/normalization/replayed.json')==record
    assert not (tmp_path/'repair/normalization/complete.json').exists()


def test_adapter_uses_new_rank_entrypoint_and_declares_repair_in_fit_parents(tmp_path,monkeypatch):
    spec0,record=fixture();root=tmp_path/'repair'
    spec=dict(root=str(root),project='/new',content_hash='c'*64,modules=dict(normalization=dict(path='/norm'),inputs=dict(path='/inputs')))
    monkeypatch.setattr(op,'frozen',lambda *a:record);monkeypatch.setattr(op,'referenced',lambda r:Path(r['path']))
    seen=[]
    def prepare(*a,**kw):seen.append(kw);return 'input-result'
    monkeypatch.setattr(op,'load_file',lambda name,path:norms if Path(path)==Path('/norm') else SimpleNamespace(prepare_node=prepare))
    old=SimpleNamespace(inputs=SimpleNamespace(prepare_node=None,decoded=inputs.decoded),launch=SimpleNamespace(command=lambda *a,**kw:
        ['python','-m','torch.distributed.run','--standalone','--','/old/script','rank','--node','/dev/shm/test']),
        training=SimpleNamespace(fit_parents=lambda *a,**kw:dict(spec='a'*64)))
    monkeypatch.setenv('LUKA_NORMALIZATION_TASK','R24_s10');monkeypatch.setenv('SLURM_JOB_ID','123')
    op.install(spec,old)
    assert old.inputs.prepare_node(spec0,'/node','D000',targets=True)=='input-result'
    assert seen[0]['frozen_normalization']==record
    argv=old.launch.command('/p','/r','R24_s10','/node')
    assert Path(argv[argv.index('--')+1])==Path('/new/scripts/recover_luka_reconstruction_normalization.py')
    assert argv[-2:]==['--repair-spec',str(root/'repair_spec.json')]
    evidence=root/'comparisons/R24_s10-123.json';write_immutable_json(evidence,c.artifact('NORMALIZATION_COMPARISON'))
    parents=old.training.fit_parents(spec0,'R24_s10',None,record)
    assert parents['execution_recovery']==spec['content_hash']
    assert parents['normalization_comparison']==load_json(evidence)['content_hash']


def test_replacement_preview_preserves_live_and_completed_models_and_only_requires_teachers(tmp_path,monkeypatch):
    from hlt_classification.luka_balanced_reconstruction_hydra import campaign
    tasks=('OFFLINE_s10','M0HLT_s10','U100_s10','R24_s10','D20_s10','R22_s10')
    monkeypatch.setattr(c,'FITS',tasks)
    rows=[dict(job=str(i+1),task=t) for i,t in enumerate(tasks)]
    statuses={str(i+1):(('COMPLETED' if i<3 else 'FAILED' if i==3 else 'PENDING' if i==4 else 'RUNNING'),'0:0') for i in range(6)}
    oldspec=dict(root=str(tmp_path/'old'),project='/original',content_hash='a'*64)
    def fit_root(spec,task):return Path(spec['root'])/'fits'/task
    for row in rows[:3]:
        path=fit_root(oldspec,row['task'])/'result.json';path.parent.mkdir(parents=True);path.write_text('{}')
    old=SimpleNamespace(spec=oldspec,c=c,campaign=SimpleNamespace(accepted_jobs=lambda s:rows,commands=campaign.commands),
        artifacts=SimpleNamespace(fit_root=fit_root,completed=lambda s,t:dict(receipt=dict(job_id=next(r['job'] for r in rows if r['task']==t)))))
    monkeypatch.setattr(op,'states',lambda jobs:{j:statuses[j] for j in jobs})
    spec=dict(root='/repair',project='/new',content_hash='b'*64,original_jobs={r['job']:r['task'] for r in rows})
    preview=op.replacements(spec,old,preview=True)
    assert [r['task'] for r in preview]==['R24_s10','D20_s10','report']
    assert not any(a.startswith('--dependency=') for a in preview[0]['command'])
    assert not any(a.startswith('--dependency=') for a in preview[1]['command'])
    assert '--dependency=afterany:6:{R24_s10}:{D20_s10}' in preview[2]['command']
    actual=op.replacements(spec,old)
    assert [r['task'] for r in actual]==['R24_s10','report']
    assert '--dependency=afterany:5:6:{R24_s10}' in actual[1]['command']
    (fit_root(oldspec,'U100_s10')/'result.json').unlink();statuses['3']=('RUNNING','0:0')
    assert '--dependency=afterok:3' in op.replacements(spec,old)[0]['command']


def test_new_original_format_attempt_retains_recovery_lineage_and_uncertainty_guard(tmp_path,monkeypatch):
    from hlt_classification.luka_balanced_reconstruction_hydra import campaign
    oldspec=dict(root=str(tmp_path/'original'),content_hash='a'*64)
    old=SimpleNamespace(spec=oldspec,c=c)
    spec=dict(content_hash='b'*64)
    rows=[dict(task='R24_s10',command=['sbatch','--wrap=true'])]
    monkeypatch.setenv('SBATCH_GRES','gpu:unexpected:8');monkeypatch.setenv('SLURM_JOB_ID','999')
    def submit(*args,**kwargs):
        assert 'SBATCH_GRES' not in kwargs['env'] and 'SLURM_JOB_ID' not in kwargs['env']
        return SimpleNamespace(returncode=0,stdout='123\n',stderr='')
    monkeypatch.setattr(op.subprocess,'run',submit)
    folder=Path(oldspec['root'])/'submission/attempt_002'
    assert op.submit_rows(spec,rows,folder,old=old)=={'R24_s10':'123'}
    accepted=campaign.accepted_jobs(oldspec)
    assert len(accepted)==1 and accepted[0]['execution_recovery']==spec['content_hash']
    assert accepted[0]['parents']==dict(spec=oldspec['content_hash'])
    with pytest.raises(FileExistsError):op.submit_rows(spec,rows,folder,old=old)
    folder=folder.with_name('attempt_003')
    monkeypatch.setattr(op.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr='lost connection'))
    with pytest.raises(ValueError,match='Uncertain submission'):op.submit_rows(spec,rows,folder,old=old)
    assert (folder/'R24_s10.intent.json').is_file() and not (folder/'R24_s10.submitted.json').exists()
    with pytest.raises(ValueError,match='Ambiguous submission'):campaign.accepted_jobs(oldspec)


def test_exact_candidate_can_be_frozen_without_reading_particles(tmp_path,monkeypatch):
    oldspec,record=fixture();root=tmp_path/'repair';candidate=tmp_path/'candidate.json'
    write_immutable_json(candidate,record)
    spec=dict(root=str(root),content_hash='b'*64,candidate=op.ref(candidate),
        normalization_sha256=record['content_hash'],qualification_node='gpu1905',
        modules=dict(normalization=op.ref(Path(norms.__file__))))
    old=SimpleNamespace(spec=oldspec,q=dict(content_hash='c'*64))
    monkeypatch.setattr(op,'check',lambda s:old);monkeypatch.setattr(op,'bound',lambda *a:'123')
    monkeypatch.setattr(op,'NORMAL_HASH',record['content_hash'])
    monkeypatch.setattr(inputs,'prepare_node',lambda *a,**kw:pytest.fail('Candidate recovery must not decode particles'))
    write_immutable_json(root/'submission/bootstrap/normalize.submitted.json',
        op.artifact('SUBMITTED',job='123',execution_recovery=spec['content_hash']))
    op.normalize(spec)
    assert op.frozen(spec,old)==record
    assert op.read(root/'normalization/complete.json')['method']=='qualified_hash_candidate'


def test_replay_never_removes_preexisting_scratch(tmp_path,monkeypatch):
    oldspec,record=fixture();root=tmp_path/'repair';ram=tmp_path/'shm'
    scratch=ram/'rlyang_lukabnorm_123';scratch.mkdir(parents=True);(scratch/'keep').write_text('existing')
    spec=dict(root=str(root),content_hash='b'*64,candidate=None)
    monkeypatch.setattr(op,'check',lambda s:SimpleNamespace(spec=oldspec))
    monkeypatch.setattr(op,'bound',lambda *a:'123')
    monkeypatch.setattr(op,'Path',lambda value:ram if str(value)=='/dev/shm' else Path(value))
    monkeypatch.setattr(op.shutil,'rmtree',lambda *a:pytest.fail('Preexisting scratch was removed'))
    with pytest.raises(ValueError,match='scratch already exists'):op.normalize(spec)
    assert (scratch/'keep').read_text()=='existing'
