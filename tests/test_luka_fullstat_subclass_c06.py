"""Combined science, exact optimized sampling, durable state and queue boundaries."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import subprocess
import os
import sys
import time
import numpy as np
import pytest
import torch
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_u100_contrastive import sampler_fixture
from hlt_classification.luka_fullstat_multi3_contrastive import model as old_model
from hlt_classification.luka_u100_multiview import teachers
from hlt_classification.luka_fullstat_subclass_c06 import contracts as c, model, sampling, training
from hlt_classification.luka_fullstat_subclass_c06 import campaign,submission,launch,worker,startup,banks,recovery
from hlt_classification.data.cache_contracts import load_json,write_immutable_json,with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def node(): return dict(initialization_seed=33,sampler_seed=44,coordinate='D000',context_coordinate=None)


def fixture(device='cpu'):
    reference=sampler_fixture(device)
    fast=sampling.Sampler(reference.keys.cpu().numpy(),reference.identities,reference.labels,device=device)
    return reference,fast


def test_scientific_recipe():
    r=c.recipe()
    assert (r['epochs'],r['batch_size'],r['ce_weight'],r['matching_coefficient'])==(20,128,1.,.6)
    assert set(r['embedding_weights'])=={'U100','CONCAT','FUSION'}
    assert r['matching_teacher']=='OFFLINE' and r['preflight_jobs']==0 and r['teacher_fits']==0
    assert r['subclass']['coefficient']==.5 and r['subclass']['class_counts']==[10]+[5]*10
    assert r['subclass']['head']==[128,60] and r['subclass']['temperature_goal']==.8
    assert c.coefficient(70314,23438)==.6
    assert c.coefficient(1,23438)==pytest.approx(.6/70314)


def test_only_contrastive_scientific_coefficient_changes():
    from hlt_classification.luka_fullstat_subclass import contracts as old
    a,b=old.recipe(),c.recipe()
    differences={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    assert differences == {'contract','content_hash','parents','matching_coefficient',
                            'historical_control','subclass_targets'}
    for update in (1,128,23438,46876,70313,70314,20*23438):
        assert c.coefficient(update,23438)==2*old.coefficient(update,23438)
        assert c.subclass_coefficient(update,23438)==.5*(old.coefficient(update,23438)/.3)
    for update in (70314,20*23438):assert c.subclass_coefficient(update,23438)==.5
    # The former artifact must not be accepted as new science.
    with pytest.raises(ValueError):c.validate(a,'RECIPE')


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('rows',[3,128])
def test_loss_gradient_delta_is_only_extra_matching(monkeypatch,device,rows):
    from hlt_classification.luka_fullstat_subclass import model as previous
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tiny(monkeypatch)
    tr=cache(coordinate='D000',rows=132)
    ids=np.arange(rows);raw=tr.batch(ids)
    banks={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32))
           for i,n in enumerate(c.TEACHERS)}
    keys=torch.nn.functional.normalize(torch.randn(rows,1025,128,device=device),dim=-1)
    sampler=SimpleNamespace(subclass_bank=subbank(tr),candidate_keys=lambda ix:keys)
    candidates=np.column_stack((ids,np.tile(np.arange(1024),(rows,1))))
    baseline=previous.new_model(node()).to(device);state=torch.get_rng_state().clone()
    new=model.new_model(node()).to(device)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in baseline.state_dict().items())
    baseline.train();new.train()
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if device=='cuda' else []
    original,oldparts=previous.objective(baseline,raw,ids,banks,sampler,candidates,.3,device)
    torch.set_rng_state(cpu)
    if cuda:torch.cuda.set_rng_state_all(cuda)
    total,parts=model.objective(new,raw,ids,banks,sampler,candidates,.6,device,subclass_weight=.5)
    for key in ('ce',*c.TEACHERS,'matching','subclass','weighted_subclass'):
        torch.testing.assert_close(parts[key],oldparts[key],rtol=0,atol=0)
    torch.testing.assert_close(parts['weighted_matching'],2*oldparts['weighted_matching'],rtol=0,atol=0)
    expected=original+.3*oldparts['matching']
    torch.testing.assert_close(total,expected)
    expected.backward();total.backward()
    for name,p in baseline.named_parameters():
        torch.testing.assert_close(p.grad,dict(new.named_parameters())[name].grad,rtol=2e-5,atol=2e-6)
    # At the old weights this independent implementation reproduces its update.
    baseline.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu)
    if cuda:torch.cuda.set_rng_state_all(cuda)
    original,_=previous.objective(baseline,raw,ids,banks,sampler,candidates,.3,device)
    torch.set_rng_state(cpu)
    if cuda:torch.cuda.set_rng_state_all(cuda)
    replay,_=model.objective(new,raw,ids,banks,sampler,candidates,.3,device,subclass_weight=.5)
    torch.testing.assert_close(replay,original,rtol=0,atol=0)
    original.backward();replay.backward()
    for name,p in baseline.named_parameters():
        torch.testing.assert_close(p.grad,dict(new.named_parameters())[name].grad,rtol=0,atol=0)


@pytest.mark.parametrize('matching,subclass',[(.61,.5),(.6,1.),(float('nan'),.5),(.6,float('nan'))])
def test_out_of_scope_coefficients_rejected(matching,subclass):
    with pytest.raises(ValueError,match='weights'):
        model.objective(None,None,None,{t:None for t in c.TEACHERS},None,None,matching,'cpu',subclass_weight=subclass)


def registered(tmp_path):
    from test_luka_fullstat_subclass import registered as previous
    from hlt_classification.luka_fullstat_subclass_c06 import source
    old=previous(tmp_path)
    old=with_content_hash(dict(old,source_commit=source.PARENT_COMMIT))
    Path(old['root']).mkdir()
    write_immutable_json(Path(old['root'])/'campaign_spec.json',old)
    reuse=c.artifact('REUSE',parents=dict(spec=old['content_hash']),parent_spec=file_ref(Path(old['root'])/'campaign_spec.json'),
        reference=old['references']['MULTI3_MIX'],u100=old['reuse']['u100'],banks=old['reuse']['banks'],
        job='8761518',subclasses=dict(manifest_hash='b'*64,files=[]))
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=reuse['content_hash']),
        root=str(tmp_path/'luka_fullstat_subclass_c06_fixture'),project_dir='/project',source_commit='f'*40,
        source_snapshot=dict(content_hash='c'*64),reuse=reuse,resume=None,available_quota_gib=8,
        **campaign.registration(old,reuse))


def subbank(tr):
    q=np.zeros((len(tr),10),np.float32)
    q[tr.labels==0]=.1;q[tr.labels!=0,:5]=.2
    # Nonuniform soft targets exercise connected nonzero subclass gradients.
    for i in range(len(q)):
        q[i,0]+=.05;q[i,1]-=.05
    return SimpleNamespace(targets=q,identities=tr.identities,labels=tr.labels)


@pytest.mark.parametrize('bad',['none','epochs','coefficient','teacher','coordinate','quota','memory','test','tasks'])
def test_registration_guards(tmp_path,bad):
    spec=registered(tmp_path)
    if bad=='epochs':spec['recipe']['epochs']=30
    elif bad=='coefficient':spec['recipe']['matching_coefficient']=1.
    elif bad=='teacher':spec['recipe']['matching_teacher']='FUSION'
    elif bad=='coordinate':spec['nodes'][c.TASK]['coordinate']='U100'
    elif bad=='quota':spec['available_quota_gib']=7
    elif bad=='memory':spec['memory_mb']=256000
    elif bad=='test':spec['final_test_accessed']=True
    elif bad=='tasks':spec['tasks']={'teacher_OFFLINE':[]}
    if bad=='none':
        campaign.validate_spec(spec)
        plan=submission.render(spec)
        assert plan['jobs']==plan['scientific_fits']==1
        assert plan['preflight_jobs']==plan['teacher_fits']==0
        assert '--time=2-00:00:00' in plan['commands'][0]['command']
    else:
        with pytest.raises((ValueError,PermissionError)):campaign.validate_spec(with_content_hash(spec))


@pytest.mark.parametrize('failure',['none','site','ambiguous','authorization'])
def test_detached_launch_exact_one_job_no_implicit_retry(tmp_path,monkeypatch,failure):
    s=registered(tmp_path);root=Path(s['root']);folder=Path(str(root)+'.launch');folder.mkdir()
    def create(*a,**kw):root.mkdir();write_immutable_json(root/'campaign_spec.json',s);return s
    monkeypatch.setattr(campaign,'create',create);monkeypatch.setattr(campaign,'check',lambda *a:None)
    def site(plan):
        if failure=='site':raise PermissionError('Site rejected')
    monkeypatch.setattr(submission,'check_site',site)
    calls=[]
    def sbatch(args,**kw):
        assert (folder/'approval.json').is_file();calls.append(args)
        return SimpleNamespace(returncode=0,stdout='ambiguous\n' if failure=='ambiguous' else '9000001\n',stderr='')
    monkeypatch.setattr(subprocess,'run',sbatch)
    params=dict(root=root,path=s['reuse']['parent_spec']['path'],project='/project',commit='f'*40,quota_gib=8,
        authorization='wrong' if failure=='authorization' else c.LAUNCH_AUTHORIZATION)
    if failure=='none':
        got=launch.run(**params);assert got['jobs']=={c.TASK:'9000001'}
        assert len(calls)==1 and submission.accepted_jobs(s)==got['jobs']
        assert not any('teacher_' in x for x in calls[0])
    else:
        with pytest.raises((ValueError,PermissionError)):launch.run(**params)
        assert len(calls)==(1 if failure=='ambiguous' else 0)
        if failure=='authorization':assert not (folder/'process_claim').exists();return
        assert (folder/'failure.json').exists()
    count=len(calls)
    with pytest.raises(FileExistsError):launch.run(**params)
    assert len(calls)==count


def fit_fixture(monkeypatch,device):
    tiny(monkeypatch);tr=cache(coordinate='D000',rows=22);val=cache('validation','D000',rows=22)
    targets={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities,labels=tr.labels) for i,n in enumerate(c.TEACHERS)}
    # Exercise the real 1025-key objective/worker over a small TRAIN epoch;
    # sampling laws over a capacity-valid population are tested independently.
    frozen=torch.nn.functional.normalize(torch.randn(22,1025,128,device=device),dim=-1)
    sampler=SimpleNamespace(subclass_bank=subbank(tr),identities=tr.identities,labels=tr.labels,keys=frozen,
        candidate_keys=lambda ix:frozen[ix[:,0]])
    class Batches:
        def __init__(self,s,order,epoch):self.order=order
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):
            for start in range(0,len(self.order),128):
                ids=self.order[start:start+128];yield ids,np.column_stack((ids,np.tile(np.arange(1024),(len(ids),1))))
    monkeypatch.setattr(training,'Lookahead',Batches)
    return tr,val,targets,sampler


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full20_resume_matches_uninterrupted_and_finished_fit_does_no_updates(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tr,val,targets,sampler=fit_fixture(monkeypatch,device)
    saved=[]
    net,report,best=training.train(tr,val,banks=targets,sampler=sampler,node=node(),parents={},device=device,
        save_epoch=lambda e,r,s:saved.append((deepcopy(r),deepcopy(s))))
    spec=dict(nodes={c.TASK:node()},counts=dict(train=len(tr),validation=len(val)))
    training.validate_report(report,spec,{})
    history=[v[0] for v in saved[:7]];selected=max(range(7),key=lambda i:training.selection_key(history[i]['validation'],history[i]['update']))
    resume=dict(history=history,state=saved[6][1],best=saved[selected][1]);continued=[]
    net2,report2,best2=training.train(tr,val,banks=targets,sampler=sampler,node=node(),parents={},device=device,
        save_epoch=lambda e,r,s:continued.append((e,r,s)),resume=resume)
    assert [e for e,_,_ in continued]==list(range(8,21))
    assert report2['validation']==report['validation'] and report2['selected_pass']==report['selected_pass']
    assert all(torch.equal(v,net2.state_dict()[k]) for k,v in net.state_dict().items())
    for old,new in zip(report['validation_history'],report2['validation_history']):
        assert old['validation']==new['validation'] and old['losses']==new['losses']
    monkeypatch.setattr(model,'objective',lambda *a,**k:pytest.fail('Finished fit must not optimize'))
    done=dict(history=report['validation_history'],state=saved[-1][1],best=best)
    final,r3,_=training.train(tr,val,banks=targets,sampler=sampler,node=node(),parents={},device=device,
        save_epoch=lambda *a:pytest.fail('Unexpected new epoch'),resume=done)
    assert r3==report and all(torch.equal(v,final.state_dict()[k]) for k,v in net.state_dict().items())


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('late_failure',[False,True])
def test_full_worker_twenty_epochs_publication_and_tamper_detection(tmp_path,monkeypatch,device,late_failure):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tr,val,targets,sampler=fit_fixture(monkeypatch,device)
    spec=registered(tmp_path);spec=with_content_hash(dict(spec,counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    root=Path(spec['root']);root.mkdir();write_immutable_json(root/'campaign_spec.json',spec)
    monkeypatch.setattr(campaign,'check',lambda *a:None);monkeypatch.setattr(campaign,'storage',lambda *a:None)
    monkeypatch.setattr(worker,'execution',lambda *a:'9000001')
    monkeypatch.setattr(submission,'accepted_jobs',lambda *a,**k:{c.TASK:'9000001'})
    monkeypatch.setattr(submission,'accounting',lambda *a,**k:None)
    monkeypatch.setattr(worker.views,'load',lambda *a:(tr,val,None))
    monkeypatch.setattr(banks,'load',lambda *a:(targets,sampler))
    monkeypatch.setattr(worker.targets,'check_manifest',lambda *a:None)
    monkeypatch.setattr(worker,'parents',lambda spec:dict(spec=spec['content_hash']))
    monkeypatch.setattr(worker,'output_names',lambda:[c.TASK+'/'+n for n in worker.FIT_FILES]+[c.TASK+'/epochs/%02d/%s'%(e,n) for e in range(1,21) for n in ('state.pt','receipt.json')])
    def probe(s,*a):
        proof=c.artifact('STARTUP',parents=dict(spec=s['content_hash']))
        write_immutable_json(Path(s['root'])/c.TASK/'startup_measurements.json',proof);return proof
    monkeypatch.setattr(startup,'run',probe);monkeypatch.setattr(startup,'check',lambda *a:None)
    monkeypatch.setattr(worker,'memory_snapshot',lambda *a:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=1))
    monkeypatch.setattr(worker,'check_memory',lambda *a:None)
    from contextlib import nullcontext
    monkeypatch.setattr(worker,'CPUWorkers',lambda *a:nullcontext(SimpleNamespace(memory=lambda:0)))
    monkeypatch.setattr(worker,'reference_arrays',lambda *a:(None,dict(identities=val.identities,labels=val.labels)))
    if late_failure:
        publish=worker.publish
        monkeypatch.setattr(worker,'publish',lambda *a:(_ for _ in ()).throw(RuntimeError('simulated publication failure')))
        with pytest.raises(RuntimeError,match='publication failure'):worker.run(spec,device=device)
        assert len(worker.epoch_records(spec))==20
        monkeypatch.setattr(submission,'accounting',lambda *a,**k:['9000001','failed'])
        resume=recovery.inspect(root/'campaign_spec.json',spec['project_dir'],spec['source_commit'])
        spec=with_content_hash(dict(spec,root=str(tmp_path/'luka_fullstat_subclass_c06_retry'),resume=resume))
        root=Path(spec['root']);root.mkdir();write_immutable_json(root/'campaign_spec.json',spec)
        recovery.check(spec);recovery.claim(spec,submission.render(spec))
        with pytest.raises(FileExistsError):recovery.claim(spec,submission.render(spec))
        monkeypatch.setattr(worker,'publish',publish)
        monkeypatch.setattr(model,'objective',lambda *a,**k:pytest.fail('Completed fit must not retrain'))
    result=worker.run(spec,device=device);assert result['task_id']==c.TASK
    report,arrays=worker.validate_fit(spec);assert len(report['validation_history'])==20
    assert arrays['probabilities'].shape==(22,11)
    assert len(worker.epoch_records(spec))==20
    (root/c.TASK/'epochs/01/state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):worker.completed(spec)


def test_full_installed_weaver_integrated_full_candidate_startup(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # No reduced architecture: use installed Weaver and the production model.
    tr=cache(coordinate='D000',rows=6006);val=cache('validation','D000',rows=22)
    _,sampler=fixture('cuda');sampler.subclass_bank=subbank(tr)
    targets={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities,labels=tr.labels) for i,n in enumerate(c.TEACHERS)}
    spec=registered(tmp_path);spec=with_content_hash(dict(spec,counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    (Path(spec['root'])/c.TASK).mkdir(parents=True)
    monkeypatch.setattr(startup,'memory_snapshot',lambda *a:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    import time
    with sampling.CPUWorkers(sampler):proof=startup.run(spec,tr,val,targets,sampler,'cuda',time.monotonic())
    startup.check(spec,proof)
    assert proof['measured_steps']==64 and proof['sampler_capacity']['rows']==6006
    assert proof['timing']['conservative_minutes']<spec['minutes'][c.TASK]
    bad=deepcopy(proof);bad['timing']=startup.timing(spec,1000.,1.,0.)
    with pytest.raises(ValueError,match='exceeds'):startup.check(spec,with_content_hash(bad))


def test_sampler_key_transform_matches_donor():
    from hlt_classification.luka_u100_contrastive.targets import normalized_keys
    from hlt_classification.luka_embedding_aux.targets import standardize
    raw=np.random.default_rng(1).normal(size=(67,128)).astype(np.float32)
    expected,_,_=normalized_keys(raw);standardized,_,_=standardize(raw)
    actual=standardized/np.maximum(np.linalg.norm(standardized,axis=1,keepdims=True),1e-12)
    np.testing.assert_array_equal(actual,expected)




def test_pinned_queue_ancestry_still_rejects_donor_changes(monkeypatch):
    from hlt_classification.luka_fullstat_subclass_c06 import source
    project=Path(source.__file__).resolve().parents[3]
    files=[str(p.relative_to(project)).replace('\\','/') for p in (project/source.PACKAGE).glob('*.py')]
    files+=list(source.ENTRYPOINTS)+['tests/test_luka_fullstat_subclass_c06.py',
        'tests/test_luka_fullstat_subclass_c06_targets.py',
        'tests/test_luka_fullstat_subclass_c06_lifecycle.py',
        'docs/contracts/LUKA_FULLSTAT_SUBCLASS_C06.md',
        'docs/LUKA_FULLSTAT_SUBCLASS_C06_RUNBOOK.md',
        'docs/plans/LUKA_FULLSTAT_SUBCLASS_C06_PLAN.md']
    changed=[source.PACKAGE+'source.py'];calls=[]
    def git(p,*args):
        calls.append(args)
        if args[0]=='diff':return '\n'.join(changed)
        if args[0]=='ls-files':return '\n'.join(files)
        return ''
    monkeypatch.setattr(source,'git',git)
    monkeypatch.setattr(source,'_source',lambda *a:{'pinned':True})
    assert source.current(project,'f'*40)=={'pinned':True}
    assert ('merge-base','--is-ancestor',source.BASELINE,'f'*40) in calls
    assert ('diff','--name-only',source.BASELINE,'f'*40,'--','src','scripts','sbatch') in calls
    changed.append('src/hlt_classification/luka_u100_contrastive/sampling.py')
    with pytest.raises(ValueError,match='Only additive'):source.current(project,'f'*40)


def test_exact_registered_worker_allocation(tmp_path,monkeypatch):
    from test_luka_fullstat_multiview_launch import allocation
    from hlt_classification.luka_fullstat_subclass_c06 import execution
    spec,evidence,fields=allocation(tmp_path,monkeypatch)
    spec.update(reuse=dict(u100=evidence),minutes={c.TASK:2880})
    monkeypatch.setattr(execution,'RUNTIME',spec['runtime_prefix'])
    monkeypatch.setattr(execution,'runtime',lambda e:None if e==evidence['environment'] else pytest.fail('Wrong evidence'))
    monkeypatch.setattr(execution,'installed_environment',lambda:evidence['environment'])
    monkeypatch.setattr(execution,'gpu_identity',lambda:evidence['gpu'])
    fields['TimeLimit']='1-00:00:00'
    with pytest.raises(PermissionError):execution.execution(spec)
    fields['TimeLimit']='2-00:00:00'
    assert execution.execution(spec)=='8300001'
    monkeypatch.setenv('SLURM_MEM_PER_NODE','180000')
    with pytest.raises(PermissionError):execution.execution(spec)


def test_shell_tail_exit_does_not_stop_detached_launch(tmp_path):
    # Windows setsid stub proves asynchronous completion, not Linux SSH survival.
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def shellpath(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    name='launch_luka_fullstat_subclass_c06.sh';script=repo/'scripts'/name
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/name).read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_text('# fixture runtime\n')
    binpath=tmp_path/'bin';binpath.mkdir()
    bodies={
        'git':'case "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n',
        'setsid':'exec "$@"\n','tail':'printf "TAIL_ATTACHED\\n"\n',
        'python':'sleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n'}
    for name,body in bodies.items():
        p=binpath/name;p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    parent=tmp_path/'parent.json';parent.write_text('{}');root=tmp_path/'output';done=tmp_path/'done'
    startup_env=tmp_path/'env.sh';startup_env.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],FIXTURE_PROJECT=shellpath(repo),
        FIXTURE_DONE=shellpath(done),BASH_ENV=shellpath(startup_env),FIXTURE_BIN=shellpath(binpath))
    args=[bash,str(script),'0'*39+'1',shellpath(root),shellpath(parent),'8','--authorize',c.LAUNCH_AUTHORIZATION]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'DETACHED:' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline:time.sleep(.05)
    log=Path(str(root)+'.launch/launcher.log')
    assert done.read_text()=='done',log.read_text()
    assert 'DETACHED_FINISHED' in log.read_text()
    assert subprocess.run(args,env=env,capture_output=True,text=True,timeout=5).returncode!=0
