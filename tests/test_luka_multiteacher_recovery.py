"""Replacement provenance, eight-job dependencies and unchanged real kernels."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import re
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch
from test_luka_embedding_aux import tiny, threads
from test_luka_attention import cache, parent_nodes
from test_luka_multiteacher_trial import fixture as multi_fixture, authenticated_fixture
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, with_content_hash,
    write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_attention import contracts as ac, model as am
from hlt_classification.luka_multiteacher_trial import contracts as mc, campaign as original_campaign
from hlt_classification.luka_multiteacher_recovery import contracts as c, source, campaign, submission, worker


def archive_fixture(anchor, parent, resources):
    return c.artifact('ARCHIVE', parents=dict(anchor=anchor['content_hash']), spec=anchor['spec'],
        anchor=anchor, prepared=anchor['evidence']['prepared'], archive=dict(verified_files=[anchor['spec']]),
        nodes=[n for n in ac.nodes(parent['nodes']) if n['node_id'] in c.REPLACEMENTS],
        recorded_checkpoints={n:dict(path='train_'+n+'/selected.pt',bytes=123,sha256='f'*64) for n in c.REPLACEMENTS},
        prior_request=dict(train_minutes=190,export_minutes=30,environment=resources.get('environment'),
                           gpu=resources.get('gpu')))


def runtime_fixture(path, rows=22):
    old, tr, val, views, _ = multi_fixture(path, rows=rows)
    parent = load_json(old['anchor']['spec']['path'])
    archive = archive_fixture(old['anchor'], parent, old['resources'])
    fields = {k:old[k] for k in ('root','project_dir','source_commit','source_snapshot','anchor','reference','resources')}
    fields.update(archive=archive, survivors={n:old['teachers'][n] for n in ('OFFLINE','COARSE_U100')},
        available_quota_gib=5, u100_reference=None, **campaign.policy(parent,old['resources'],archive))
    fields.update(counts=dict(train=rows,validation=len(val)),original50k_rows=len(val))
    spec = c.artifact('SPEC', parents=campaign.parents(fields), **fields)
    (Path(spec['root'])/'campaign_spec.json').write_text(json.dumps(spec))
    return spec,tr,val,views


def test_eight_jobs_exact_two_replacements_and_unchanged_five_recipes():
    parent = dict(nodes=parent_nodes()); resources = dict(measured_w5_minutes=190)
    a = dict(nodes=[n for n in ac.nodes(parent['nodes']) if n['node_id'] in c.REPLACEMENTS],
             prior_request=dict(train_minutes=190,export_minutes=30))
    policy = campaign.policy(parent,resources,a)
    donor = original_campaign.policy(parent,resources)
    for key in ('nodes','recipes','counts','original50k_rows','execution_site','cpus','memory_mb','train_minutes','export_minutes'):
        assert policy[key] == donor[key]
    assert policy['scientific_fits']==7 and policy['teacher_fits']==2 and policy['student_fits']==5
    assert policy['preflight_jobs']==0 and policy['teacher_minutes']==360 and policy['storage_budget_bytes']==5*2**30
    assert policy['execution_site']['partition']=='tier3'
    assert policy['teacher_nodes']==a['nodes']
    assert [n['teacher'] for n in policy['teacher_nodes']]==['ANCHOR','LOGIT_D066']
    assert all(n['branch']=='LOGIT' and n['attention_coefficient']==0 for n in policy['teacher_nodes'])
    assert len(c.tasks())==8 and all(t['dependencies']==['export_targets'] for t in c.tasks()[3:])
    for minutes in (True,0,1500):
        bad=deepcopy(a);bad['prior_request']['train_minutes']=minutes
        with pytest.raises(ValueError):campaign.policy(parent,resources,bad)
    with pytest.raises(PermissionError):c.validate(with_content_hash(dict(c.artifact('TASK'),final_test_accessed=True)),'TASK')


def test_source_keeps_old_runtime_and_requires_complete_recovery(monkeypatch):
    project=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(source,'_source',lambda *a:dict(content_hash='a'*64))
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else '\n'.join('A\t'+p for p in source.ADDITIONS)))
    assert source.current(project,'b'*40)['content_hash']=='a'*64
    monkeypatch.setattr(subprocess,'run',lambda cmd,**kw:SimpleNamespace(stdout='\n'.join(source.REQUIRED)
        if 'ls-files' in cmd else 'M\tsrc/hlt_classification/luka_attention/training.py'))
    with pytest.raises(ValueError,match='runtime changed'):source.current(project,'b'*40)
    compile(source.ARCHIVE_CHECK,'archival-reader','exec')
    assert 'read_archive' in source.ARCHIVE_CHECK and 'validate_file_ref' not in source.ARCHIVE_CHECK
    assert 'recorded_checkpoints' in source.ARCHIVE_CHECK


def test_create_authenticates_before_root_and_retains_originals(tmp_path,monkeypatch):
    anchor,parent,ref,res,teachers=authenticated_fixture(tmp_path/'original')
    survivors={n:with_content_hash(dict(teachers[n],origin='original_fixed30',
        node=next(x for x in parent['nodes'] if x['node_id']==n))) for n in ('OFFLINE','COARSE_U100')}
    archive=archive_fixture(anchor,parent,res)
    monkeypatch.setattr(source,'current',lambda *a:dict(content_hash='c'*64))
    monkeypatch.setattr(source.donor,'authenticate',lambda *a:(anchor,parent))
    monkeypatch.setattr(source,'survivors',lambda *a:survivors)
    monkeypatch.setattr(source,'archived',lambda *a:archive)
    monkeypatch.setattr(source.donor,'reference',lambda *a:ref)
    monkeypatch.setattr(source.donor,'resources',lambda *a:res)
    checked=[];monkeypatch.setattr(source,'inputs',lambda s:checked.append(s['root']))
    args=[anchor['spec']['path'],tmp_path/'old_attention/science/campaign_spec.json',
          ref['spec']['path'],tmp_path/'project','f'*40]
    before={p:p.read_bytes() for p in (tmp_path/'original').rglob('*') if p.is_file()}
    for quota in (4,True):
        with pytest.raises(ValueError):campaign.create(tmp_path/f'bad{quota}',*args,available_quota_gib=quota)
    spec=campaign.create(tmp_path/'recovery',*args,available_quota_gib=5)
    campaign.check(spec);assert checked==[spec['root']]
    assert before=={p:p.read_bytes() for p in before}
    bad=deepcopy(spec);bad['archive']=with_content_hash(dict(bad['archive'],nodes=[]))
    with pytest.raises(ValueError,match='recipe'):source.imports(bad)
    with pytest.raises(FileExistsError):campaign.create(tmp_path/'recovery',*args,available_quota_gib=5)
    monkeypatch.setattr(source,'inputs',lambda s:(_ for _ in ()).throw(ValueError('missing validation/D033')))
    with pytest.raises(ValueError,match='validation/D033'):
        campaign.create(tmp_path/'missing',*args,available_quota_gib=5)
    assert not (tmp_path/'missing').exists()


def test_archived_original_control_schema_and_no_missing_weight_access(tmp_path,monkeypatch):
    anchor,parent,ref,res,_=authenticated_fixture(tmp_path/'original')
    old=tmp_path/'old/science/campaign_spec.json'
    write_immutable_json(old.parent.parent/'gate_spec.json',dict(source_snapshot=dict(fixture=True)))
    write_immutable_json(old,dict(source_commit='94f560259dd04253c0aa47b31dec3aedc105ac14',
        gate_path=str(old.parent.parent/'gate_spec.json'),project_dir=str(tmp_path/'old_source')))
    data=archive_fixture(anchor,parent,res)
    evidence={k:deepcopy(data[k]) for k in ('anchor','prepared','archive','nodes','recorded_checkpoints','prior_request')}
    # The original attention anchor does not have prediction references, while
    # the later embedding anchor does. Match shared authenticated descriptors.
    for row in evidence['anchor']['evidence']['models'].values(): row.pop('predictions',None)
    monkeypatch.setattr(source,'validate_source_snapshot',lambda *a,**kw:None)
    monkeypatch.setattr(source.donor,'replay',lambda *a,**kw:(file_ref(old),deepcopy(evidence)))
    result=source.archived(old,anchor,parent)
    assert result['recorded_checkpoints']==data['recorded_checkpoints']
    assert not (old.parent/'train_LOGIT_D066/selected.pt').exists()
    assert not (old.parent/'train_LOGIT_D033/selected.pt').exists()
    evidence['nodes'][0]['attention_coefficient']=.1
    with pytest.raises(ValueError,match='recipe'):source.archived(old,anchor,parent)


def test_inputs_require_teacher_validation_not_only_student(tmp_path,monkeypatch):
    blocks={}
    for role,coords in [('train',('D000','U100','D066','D033')),('validation',('D000','D066','D033'))]:
        blocks[role]={}
        for co in coords:
            p=tmp_path/role/(co+'.npz');p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture')
            blocks[role][co]=[file_ref(p,root=tmp_path)]
    raw=tmp_path/'container/raw/train.root';raw.parent.mkdir(parents=True);raw.write_bytes(b'raw fixture')
    meta=(dict(content_hash='b'*64,blocks=blocks),dict(content_hash='a'*64),
          dict(files=[file_ref(raw,root=raw.parent)]),dict(memberships=dict(train=[dict(file_index=0)])))
    monkeypatch.setattr(source.regular_cache,'metadata',lambda _:meta)
    spec=dict(anchor=dict(evidence=dict(prepared_root=str(tmp_path),cache='b'*64,prepared='a'*64,
                                       raw_container=str(tmp_path/'container'))))
    source.inputs(spec)
    raw.write_bytes(b'corrupt raw')
    with pytest.raises(ValueError,match='OFFLINE TRAIN raw'):source.inputs(spec)
    raw.write_bytes(b'raw fixture')
    (tmp_path/'validation/D033.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='validation/D033'):source.inputs(spec)


def test_exact_eight_job_submit_dependencies_and_duplicate_refusal(tmp_path,monkeypatch):
    tiny(monkeypatch);spec,*_=runtime_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(spec);assert plan['jobs']==8 and plan['preflight_jobs']==0
    names=[t['task_id'] for t in c.tasks()]
    assert [r['task_id'] for r in plan['commands']]==names
    for r in plan['commands']:
        assert '--partition=tier3' in r['command'] and '--gres=gpu:a100:1' in r['command']
        assert '--kill-on-invalid-dep=yes' in r['command'] and '--no-requeue' in r['command']
    assert '--dependency=afterok:${JOB_train_LOGIT_D066}' in plan['commands'][1]['command']
    assert '--dependency=afterok:${JOB_train_LOGIT_D033}' in plan['commands'][2]['command']
    with pytest.raises(PermissionError):submission.submit(spec,execute=True,plan_hash='wrong',authorization=c.AUTHORIZATION)
    calls=[]
    def fake(args,**kw):
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls))+'\n',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert submission.accepted_jobs(spec)==dict(zip(names,map(str,range(101,109))))
    assert '--dependency=afterok:101' in calls[1] and '--dependency=afterok:102' in calls[2]
    assert all('--dependency=afterok:103' in args for args in calls[3:])
    with pytest.raises(FileExistsError):submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==8


def test_ambiguous_submission_keeps_claim_no_retry(tmp_path,monkeypatch):
    tiny(monkeypatch);spec,*_=runtime_fixture(tmp_path)
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(submission,'check_site',lambda _:None)
    plan=submission.submit(spec);calls=[]
    def lost(args,**kw):
        calls.append(args);return SimpleNamespace(returncode=1,stdout='',stderr='lost acknowledgement')
    monkeypatch.setattr(subprocess,'run',lost)
    with pytest.raises(Exception):submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    with pytest.raises(FileExistsError):submission.submit(spec,execute=True,plan_hash=plan['content_hash'],authorization=c.AUTHORIZATION)
    assert len(calls)==1 and (Path(spec['root'])/'submission/live_claim/intent.json').is_file()


def test_accounting_and_incomplete_parent_stop_before_work(tmp_path,monkeypatch):
    tiny(monkeypatch);spec,*_=runtime_fixture(tmp_path)
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:dict(train_LOGIT_D066='101',train_LOGIT_D033='102'))
    monkeypatch.setattr(submission.getpass,'getuser',lambda:'ryreu')
    good='101|lukamr_train_LOGIT_D066|ryreu|reu-aisocial|COMPLETED|0:0\n'
    for bad in ('',good.replace('COMPLETED','CANCELLED'),good.replace('0:0','1:0'),good.replace('ryreu','other')):
        monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=bad))
        with pytest.raises(ValueError):submission.successful(spec,'train_LOGIT_D066','101')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=good))
    submission.successful(spec,'train_LOGIT_D066','101')
    monkeypatch.setattr(campaign,'check',lambda _:None)
    monkeypatch.setattr(worker.logit,'execution',lambda *a:'102')
    monkeypatch.setattr(worker,'train_teacher',lambda *a:pytest.fail('Do not fit without teacher receipt'))
    with pytest.raises(ValueError,match='incomplete'):worker.run(spec,'train_LOGIT_D033')
    assert not (Path(spec['root'])/'train_LOGIT_D033_attempt').exists()


def test_real_teacher_chain_export_context_and_student_full30(tmp_path,monkeypatch,capsys):
    # Genuine installed Weaver/CUDA, exact30 teacher/student epochs, actual8
    # particle blocks and128-vector. Tiny jet population/width and FF/pair sizes.
    device='cuda' if torch.cuda.is_available() else 'cpu'
    from hlt_classification.jetclass2_delphes import model
    config=model.model_config()
    config.update(embed_dims=[128],pair_embed_dims=[4],num_heads=2,num_layers=8,num_cls_layers=1,fc_params=[])
    monkeypatch.setattr(model,'model_config',lambda:deepcopy(config))
    spec,tr,val,views=runtime_fixture(tmp_path)
    parent=load_json(spec['anchor']['spec']['path'])
    refs={}
    for i,name in enumerate(('M0HLT','OFFLINE','OFFLINE_EMBED_W5')):
        torch.manual_seed(i+45);native=DelphesParticleTransformer().eval()
        p=predict(native,val,device='cpu',batch_size=128);directory=tmp_path/'refs'/name
        write_immutable_json(directory/'report.json',dict(validation=evaluate_probabilities(val.labels,p)))
        atomic_publish_bytes(directory/'predictions.npz',deterministic_npz_bytes(dict(
            identities=val.identities,labels=val.labels,probabilities=p)))
        refs[name]=dict(report=file_ref(directory/'report.json'),predictions=file_ref(directory/'predictions.npz'))
    spec['anchor']['evidence']['models']={n:refs[n] for n in ('M0HLT','OFFLINE')}
    spec['reference']['files']=refs['OFFLINE_EMBED_W5']
    # The surviving U100 must have a report hash for its T2 target parent.
    spec['survivors']['COARSE_U100']['files'].update(report=refs['M0HLT']['report'])
    spec['survivors']['COARSE_U100']=with_content_hash(spec['survivors']['COARSE_U100'])
    spec=with_content_hash(spec);root=Path(spec['root'])
    (root/'campaign_spec.json').write_text(json.dumps(spec))
    jobs=dict(zip([t['task_id'] for t in c.tasks()],map(str,range(101,109))));active=['train_LOGIT_D066']
    monkeypatch.setattr(campaign,'check',lambda _:None);monkeypatch.setattr(campaign,'validate_spec',lambda _:None)
    monkeypatch.setattr(worker.donor,'metadata',lambda _: (None,None,None,None))
    valviews={co:cache('validation',coordinate=co) for co in ('D000','D066','D033')}
    for v in valviews.values():
        v.identities[:,-1]=1
        for b in v.blocks:b.identities[:,-1]=1
    monkeypatch.setattr(worker.banks,'data',lambda s,role,co,meta:
        valviews[co] if role=='validation' else tr if co=='D000' else views[co])
    monkeypatch.setattr(worker.donor.alljets_data,'comparison_indices',lambda *a:np.arange(len(val),dtype=np.int64))
    monkeypatch.setattr(worker.logit,'execution',lambda *a:jobs[active[0]])
    monkeypatch.setattr(submission,'accepted_jobs',lambda _:jobs)
    monkeypatch.setattr(submission,'successful',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=2**30,rss_bytes=2**30,gpu_peak_bytes=1))
    # Export's >=187/class admission is separately tested with2068 actual rows
    # in donor tests. This integration uses22 rows and a non-relational student.
    monkeypatch.setattr(worker.donor.training,'AnchorSampler',lambda labels:None)
    first=worker.run(spec,'train_LOGIT_D066',device=device)
    active[0]='train_LOGIT_D033';second=worker.run(spec,active[0],device=device)
    assert second['dependencies']==dict(train_LOGIT_D066=first['content_hash'])
    for name in c.REPLACEMENTS:
        r=load_json(root/('train_'+name)/'training_report.json')
        assert r['passes']==30 and r['node']['attention_coefficient']==0
        assert r['node']==worker.teacher_node(spec,name)
        assert r['validation_history'][-1]['update']==30
        assert (root/('train_'+name)/'selected.pt').is_file()
    b=load_json(root/'train_LOGIT_D033/teacher_bank/manifest.json')
    assert b['parents']['checkpoint']==file_ref(root/'train_LOGIT_D066/selected.pt')['sha256']
    assert b['attention'] is False and not b['shards']
    interim=worker.results(spec)
    assert [r['model'] for r in interim['rows']][-2:]==['NEW_LOGIT_D066','NEW_LOGIT_D033']
    active[0]='export_targets';export=worker.run(spec,active[0],device=device)
    ctx=worker.context(spec)
    assert export['execution_context']==ctx['content_hash']
    assert ctx['parents']['spec']==spec['content_hash']
    for n in c.REPLACEMENTS:
        assert ctx['teachers'][n]['origin']=='new_LOGIT_replacement'
        assert ctx['teachers'][n]['files']['checkpoint']['path']==str((root/('train_'+n)/'selected.pt').resolve())
    active[0]='multi_w5';worker.run(spec,active[0],device=device)
    result=worker.results(spec);assert len(result['rows'])==6 and len(result['pending'])==4
    assert result['rows'][-1]['model']=='multi_w5'
    r=load_json(root/'multi_w5/training_report.json');assert r['passes']==30
    # Durable reporting needs neither model deserialization nor particle loading.
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('Saved-only results'))
    monkeypatch.setattr(worker.banks,'data',lambda *a,**kw:pytest.fail('No particle arrays for results'))
    assert worker.results(spec)==result
    worker.print_results(result);assert 'NEW_LOGIT' in capsys.readouterr().out
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert worker.results(spec)==result and before=={p:p.read_bytes() for p in before}
    wrong=with_content_hash(dict(ctx,parents=dict(ctx['parents'],spec='e'*64)))
    (root/'execution_context.json').write_text(json.dumps(wrong))
    with pytest.raises(ValueError):worker.completed(spec,'multi_w5')
    (root/'execution_context.json').write_text(json.dumps(ctx))
    record=load_json(root/'train_LOGIT_D066_receipt.json')
    (root/'train_LOGIT_D066_receipt.json').write_text(json.dumps(with_content_hash(dict(record,job_id='999'))))
    with pytest.raises(ValueError,match='job'):worker.results(spec)


def test_shell_cli_docs_and_no_legacy_source_edits():
    project=Path(__file__).resolve().parents[1];bash=Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.is_file():bash=shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    for name in ('scripts/queue_luka_multiteacher_recovery.sh','sbatch/run_luka_multiteacher_recovery.sh'):
        subprocess.run([str(bash),'-n',name],cwd=project,check=True,capture_output=True,timeout=30)
    subprocess.run([sys.executable,'-B','scripts/luka_multiteacher_recovery.py','--help'],
        cwd=project,check=True,capture_output=True,timeout=60)
    for name in source.REQUIRED:
        text=(project/name).read_text();assert all(line==line.rstrip() for line in text.splitlines()),name
        if name.endswith('.md'):
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
                assert (project/name).parent.joinpath(link).is_file(),(name,link)
            for block in re.findall(r'^```bash\n(.*?)^```',text,re.M|re.S):
                subprocess.run([str(bash),'-n'],input=block,check=True,text=True,capture_output=True,timeout=30)
