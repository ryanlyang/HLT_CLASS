"""Matched science, zero-gradient isolation, real training and one-job submission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from test_luka_attention import cache
from test_luka_embedding_aux import tiny
from test_luka_fullstat_six_teacher import (
    node, probabilities, subbank, registered as donor_registered, fit_fixture)
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_six_teacher import contracts as old, model as old_model
from hlt_classification.luka_fullstat_six_teacher import training as old_training, sampling as old_sampling
from hlt_classification.luka_fullstat_six_teacher_s0 import (
    contracts as c, model,training,startup,campaign,submission,launch,source,sampling)


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def registered(tmp_path):
    parent=donor_registered(tmp_path)
    root=Path(parent['root']);root.mkdir()
    write_immutable_json(root/'campaign_spec.json',parent)
    baseline=c.artifact('BASELINE',parents=dict(spec=parent['content_hash']),
        spec=file_ref(root/'campaign_spec.json'),reference={},exports={})
    return c.artifact('SPEC',parents=dict(source='c'*64,reuse=parent['reuse']['content_hash'],baseline=baseline['content_hash']),
        root=str(tmp_path/'luka_fullstat_six_teacher_s0_fixture'),project_dir='/project',source_commit='f'*40,
        source_snapshot=dict(content_hash='c'*64),reuse=parent['reuse'],baseline=baseline,resume=None,
        available_quota_gib=12,**campaign.registration(parent,baseline))


def test_recipe_changes_only_subclass_and_experiment_identity():
    before=old.recipe();after=c.recipe()
    omit={'contract','schema_version','parents','content_hash',*c.FLAGS}
    a={k:v for k,v in before.items() if k not in omit}
    b={k:v for k,v in after.items() if k not in omit}
    sa={k:v for k,v in a.pop('subclass').items() if k not in omit}
    sb={k:v for k,v in b.pop('subclass').items() if k not in omit}
    assert sa.pop('coefficient')==.5 and sb.pop('coefficient')==0.
    assert sa==sb
    assert a.pop('historical_control')=='completed_subclass_c03'
    assert b.pop('historical_control')=='completed_six_teacher_subclass05'
    assert a==b
    assert old.recipe()==before
    assert after['parents']==dict(previous=before['content_hash'])
    for update in (1,23438,70314,20*23438):
        assert c.coefficient(update,23438)==old.coefficient(update,23438)
        assert c.auxiliary_ramp(update,23438)==old.auxiliary_ramp(update,23438)
        assert c.subclass_coefficient(update,23438)==0.
    with pytest.raises(ValueError):c.subclass_coefficient(0,23438)
    with pytest.raises(ValueError):c.validate(before,'RECIPE')


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('rows',[3,128])
def test_identical_initialization_zero_subclass_gradient_and_native_graph(monkeypatch,device,rows):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tiny(monkeypatch);tr=cache(coordinate='D000',rows=132);ids=np.arange(rows)
    parent=old_model.new_model(node());rng=torch.get_rng_state()
    net=model.new_model(node()).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    assert parent.state_dict().keys()==net.state_dict().keys()
    for k,v in parent.state_dict().items():assert torch.equal(v,net.state_dict()[k].cpu())
    b={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        probabilities=probabilities(len(tr),i)) for i,n in enumerate(c.TEACHERS)}
    keys=torch.nn.functional.normalize(torch.randn(rows,1025,128,device=device),dim=-1)
    sampler=SimpleNamespace(subclass_bank=subbank(tr),candidate_keys=lambda x:keys)
    candidates=np.column_stack((ids,np.tile(np.arange(1024),(rows,1))))
    net.train()
    total,p=model.objective(net,tr.batch(ids),ids,b,sampler,candidates,.3,device,subclass_weight=0.,ramp=1.)
    assert p['subclass']>0 and p['weighted_subclass']==0
    expected=p['ce']+sum(v for k,v in p.items() if k.startswith('weighted_') and k!='weighted_subclass')
    torch.testing.assert_close(total,expected,rtol=0,atol=0)
    params=list(net.mod.parameters());sg=torch.autograd.grad(p['weighted_subclass'],params,retain_graph=True,allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in sg)
    fc=list(net.mod.fc.parameters());cegrad=torch.autograd.grad(p['ce'],fc,retain_graph=True)
    d=training.diagnostics(p,net,10*23438,23438)
    assert d['norms']['subclass']>0 and d['norms']['weighted_subclass']==0
    assert d['cosine_with_ce']['weighted_subclass'] is None
    total.backward()
    for param,g in zip(fc,cegrad):torch.testing.assert_close(param.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in net.subclass.parameters())
    for head in (*net.predictors.values(),*net.logit_heads.values(),net.matching):
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in head.parameters())
    net.capture=False;net.eval();native=net.deployed().to(device)
    assert all(k.startswith('mod.') for k in native.state_dict())
    from hlt_classification.luka_u100_multiview import teachers
    with torch.no_grad():torch.testing.assert_close(teachers.forward(net,tr.batch(ids),device,bf16=False),
        teachers.forward(native,tr.batch(ids),device,bf16=False),rtol=0,atol=0)
    with pytest.raises(ValueError,match='zero'):
        model.objective(net,tr.batch(ids),ids,b,sampler,candidates,.3,device,subclass_weight=.5,ramp=1.)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_twenty_epochs_resume_and_selected_publication_state(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tr,val,banks,sampler=fit_fixture(monkeypatch,device)
    monkeypatch.setattr(training,'Lookahead',old_training.Lookahead)
    saved=[]
    def save(epoch,row,state):saved.append(dict(epoch=epoch,row=deepcopy(row),state=deepcopy(state)))
    net,report,best=training.train(tr,val,banks=banks,sampler=sampler,node=node(),parents={},device=device,save_epoch=save)
    assert len(saved)==report['passes']==20
    for e,r in enumerate(report['validation_history'],1):
        training.check_row(r,e,dict(train=len(tr),validation=len(val)))
        assert r['losses']['weighted_subclass']==r['subclass_coefficient']==0
    chosen=max(saved[:7],key=lambda x:training.selection_key(x['row']['validation'],x['row']['update']))
    resume=dict(state=saved[6]['state'],history=[x['row'] for x in saved[:7]],best=chosen['state'])
    continued=[]
    _,report2,best2=training.train(tr,val,banks=banks,sampler=sampler,node=node(),parents={},device=device,
        save_epoch=lambda *a:continued.append(a),resume=resume)
    assert [x[0] for x in continued]==list(range(8,21))
    assert report2['selected_pass']==report['selected_pass'] and report2['validation']==report['validation']
    for k,v in best['student'].items():assert torch.equal(v,best2['student'][k])
    final=dict(state=saved[-1]['state'],history=report['validation_history'],best=best)
    monkeypatch.setattr(model,'objective',lambda *a,**k:pytest.fail('Finished fit repeated updates'))
    _,r3,b3=training.train(tr,val,banks=banks,sampler=sampler,node=node(),parents={},device=device,
        save_epoch=lambda *a:pytest.fail('Finished fit repeated epoch'),resume=final)
    assert r3==report
    bad=deepcopy(saved[9]['row']);bad['gradient_diagnostic']['norms']['weighted_subclass']=.01
    with pytest.raises(ValueError,match='Inactive subclass'):training.check_row(bad,10,dict(train=len(tr),validation=len(val)))


@pytest.mark.parametrize('bad',['none','coefficient','teacher','ce','epochs','quota','memory','test','tasks'])
def test_registration_preserves_control_and_fixed_resources(tmp_path,monkeypatch,bad):
    s=registered(tmp_path)
    monkeypatch.setattr(source,'check_evidence',lambda value:load_json(value['spec']['path']))
    if bad=='coefficient':s['recipe']['subclass']['coefficient']=.5
    elif bad=='teacher':s['nodes'][c.TASK]['embedding_targets'][0]='U100'
    elif bad=='ce':s['recipe']['ce_weight']=.5
    elif bad=='epochs':s['recipe']['epochs']=30
    elif bad=='quota':s['available_quota_gib']=7
    elif bad=='memory':s['memory_mb']=256000
    elif bad=='test':s['final_test_accessed']=True
    elif bad=='tasks':s['tasks']={'teacher_OFFLINE':[]}
    if bad!='none':
        with pytest.raises((ValueError,PermissionError)):campaign.validate_spec(with_content_hash(s))
        return
    campaign.validate_spec(s)
    plan=submission.render(s)
    assert plan['jobs']==plan['scientific_fits']==1 and plan['preflight_jobs']==plan['teacher_fits']==0
    assert '--job-name=lukafs6s0_train_six' in plan['commands'][0]['command']
    assert '--time=2-00:00:00' in plan['commands'][0]['command']
    assert s['teacher_exports']==0


@pytest.mark.parametrize('failure',['none','site','ambiguous','authorization'])
def test_detached_launch_one_exact_submission(tmp_path,monkeypatch,failure):
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
    params=dict(root=root,path=s['baseline']['spec']['path'],project='/project',commit='f'*40,quota_gib=12,
        authorization='wrong' if failure=='authorization' else c.LAUNCH_AUTHORIZATION)
    if failure=='none':
        result=launch.run(**params);assert result['jobs']=={c.TASK:'9000001'} and len(calls)==1
    else:
        with pytest.raises((ValueError,PermissionError)):launch.run(**params)
        assert len(calls)==(1 if failure=='ambiguous' else 0)
        if failure=='authorization':assert not (folder/'process_claim').exists();return
        assert (folder/'failure.json').exists()
    count=len(calls)
    with pytest.raises(FileExistsError):launch.run(**params)
    assert len(calls)==count


def test_real_installed_weaver_startup_and_zero_subclass(monkeypatch,tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_luka_fullstat_six_teacher import fixture
    _,sampler=fixture('cuda');tr=cache(coordinate='D000',rows=6006);val=cache('validation','D000',rows=128)
    tr.identities=sampler.identities;tr.labels=sampler.labels;sampler.subclass_bank=subbank(tr)
    banks={n:SimpleNamespace(targets=np.random.default_rng(i).normal(size=(len(tr),128)).astype(np.float32),
        identities=tr.identities,labels=tr.labels,probabilities=probabilities(len(tr),i)) for i,n in enumerate(c.TEACHERS)}
    s=registered(tmp_path);s=with_content_hash(dict(s,counts=dict(c.COUNTS,train=len(tr),validation=len(val))))
    (Path(s['root'])/c.TASK).mkdir(parents=True)
    monkeypatch.setattr(startup,'memory_snapshot',lambda *a:dict(peak_rss_bytes=2**30,rss_bytes=2**30,
        gpu_peak_bytes=int(torch.cuda.max_memory_allocated())))
    if sys.platform!='linux':monkeypatch.setattr(old_sampling,'worker_peak',lambda pid:2**20)
    with sampling.CPUWorkers(sampler):proof=startup.run(s,tr,val,banks,sampler,'cuda',time.monotonic())
    startup.check(s,proof)
    assert proof['subclass_gradient_zero'] and proof['measured_steps']==64
    bad=with_content_hash(dict(proof,subclass_gradient_zero=False))
    with pytest.raises(ValueError,match='subclass_gradient_zero'):startup.check(s,bad)


def test_launcher_survives_tail_exit_and_rejects_repeat(tmp_path):
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def shellpath(p):
        p=Path(p).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'sbatch').mkdir()
    name='launch_luka_fullstat_six_teacher_s0.sh';script=repo/'scripts'/name
    script.write_bytes((Path(__file__).resolve().parents[1]/'scripts'/name).read_bytes())
    (repo/'sbatch/luka_splithead_finalize_common.sh').write_text('# fixture runtime\n')
    binpath=tmp_path/'bin';binpath.mkdir()
    bodies={'git':'case "$*" in\n*--show-toplevel*) printf "%s\\n" "$FIXTURE_PROJECT";;\n*"rev-parse HEAD"*) printf "%040d\\n" 1;;\nesac\n',
        'setsid':'exec "$@"\n','tail':'printf "TAIL_ATTACHED\\n"\n',
        'python':'sleep 1\nprintf "DETACHED_FINISHED\\n"\nprintf done > "$FIXTURE_DONE"\n'}
    for name,body in bodies.items():
        p=binpath/name;p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    parent=tmp_path/'parent.json';parent.write_text('{}');root=tmp_path/'output';done=tmp_path/'done'
    startup_env=tmp_path/'env.sh';startup_env.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],FIXTURE_PROJECT=shellpath(repo),
        FIXTURE_DONE=shellpath(done),BASH_ENV=shellpath(startup_env),FIXTURE_BIN=shellpath(binpath))
    args=[bash,str(script),'0'*39+'1',shellpath(root),shellpath(parent),'12','--authorize',c.LAUNCH_AUTHORIZATION]
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'DETACHED:' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    deadline=time.monotonic()+10
    while not done.exists() and time.monotonic()<deadline:time.sleep(.05)
    log=Path(str(root)+'.launch/launcher.log')
    assert done.read_text()=='done',log.read_text()
    assert 'DETACHED_FINISHED' in log.read_text()
    assert subprocess.run(args,env=env,capture_output=True,text=True,timeout=5).returncode!=0
