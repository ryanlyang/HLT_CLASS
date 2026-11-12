"""Completed-control identity, original-source replay and immutable-source guards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import pytest
from test_luka_fullstat_six_teacher import registered
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullstat_six_teacher import contracts as old, worker as old_worker, recovery as old_recovery
from hlt_classification.luka_fullstat.recipe import learning_rate
from hlt_classification.luka_fullstat_six_teacher_s0 import source,contracts as c


def completed(tmp_path,monkeypatch):
    s=registered(tmp_path);root=Path(s['root']);root.mkdir()
    project=tmp_path/'six-control-source';project.mkdir()
    s=with_content_hash(dict(s,source_commit=source.PARENT_COMMIT,project_dir=str(project)))
    path=root/'campaign_spec.json';write_immutable_json(path,s)
    folder=root/old.TASK;folder.mkdir()
    (folder/'selected.pt').write_bytes(b'fixture checkpoint')
    (folder/'validation_predictions.npz').write_bytes(b'fixture predictions')
    raw,weighted=old.loss_names();history=[];steps=23438
    for epoch in range(1,21):
        losses={k:.1 for k in [*raw,*weighted]};losses['total']=losses['ce']+sum(losses[k] for k in weighted)
        metrics=dict(rows=500000,accuracy=.8,macro_ovr_auc=.9+epoch*.001,cross_entropy=.2,
                     macro_mean_log_qcd_rejection_at_50pct_signal=1.)
        diag=dict(norms={k:.1 for k in [*raw,*weighted]},cosine_with_ce={k:.1 for k in [*raw,*weighted]}) if epoch in (1,3,10,20) else None
        history.append(dict(pass_number=epoch,update=epoch*steps,learning_rate=learning_rate(epoch),
            validation=metrics,losses=losses,matching_coefficient=old.coefficient(epoch*steps,steps),
            subclass_coefficient=old.subclass_coefficient(epoch*steps,steps),auxiliary_ramp=old.auxiliary_ramp(epoch*steps,steps),
            train_seconds=1.,validation_seconds=1.,gradient_diagnostic=diag))
    monkeypatch.setattr(old_worker,'parents',lambda p:dict(spec=p['content_hash']))
    report=old.artifact('TRAINING_REPORT',parents=dict(spec=s['content_hash'],recipe=old.recipe()['content_hash']),
        task_id=old.TASK,node=s['nodes'][old.TASK],passes=20,scientific_fit=True,selected_weights_restored=True,
        deployed_graph='native_HLT',selected_pass=20,validation=history[-1]['validation'],validation_history=history)
    write_immutable_json(folder/'training_report.json',report)
    rec=old.artifact('TASK',parents=dict(spec=s['content_hash']),job_id=source.PARENT_JOB,task_id=old.TASK,
        source_commit=source.PARENT_COMMIT,report_hash=report['content_hash'],
        outputs=[file_ref(folder/n,root=root) for n in ('selected.pt','training_report.json','validation_predictions.npz')])
    write_immutable_json(folder/'task_receipt.json',rec)
    reference={k:file_ref(folder/n) for k,n in [('checkpoint','selected.pt'),('report','training_report.json'),
        ('predictions','validation_predictions.npz'),('receipt','task_receipt.json')]}
    exports={t:[] for t in ('CONCAT','FUSION')}
    monkeypatch.setattr(old_recovery,'export_records',lambda p:exports)
    proof=c.artifact('BASELINE',parents=dict(spec=s['content_hash']),spec=file_ref(path),reference=reference,
        exports=exports,accounting=[source.PARENT_JOB,'lukafs6_train_six','rlyang',s['execution_site']['account'],'COMPLETED','0:0'],
        job_id=source.PARENT_JOB,report_hash=report['content_hash'])
    return s,proof


@pytest.mark.parametrize('bad',['none','failed','running','job','report','checkpoint','predictions','exports','spec'])
def test_baseline_requires_completed_exact_job_and_bound_outputs(tmp_path,monkeypatch,bad):
    parent,proof=completed(tmp_path,monkeypatch)
    if bad=='failed':proof['accounting'][4:]=['FAILED','1:0']
    elif bad=='running':proof['accounting'][4]='RUNNING'
    elif bad=='job':proof['job_id']='8848996'
    elif bad=='report':proof['report_hash']='0'*64
    elif bad in ('checkpoint','predictions'):Path(proof['reference'][bad]['path']).write_bytes(b'changed')
    elif bad=='exports':proof['exports']={'CONCAT':[]}
    elif bad=='spec':Path(proof['spec']['path']).write_text('{}')
    if bad=='none':assert source.check_evidence(proof)==parent
    else:
        with pytest.raises(ValueError):source.check_evidence(with_content_hash(proof))


@pytest.mark.parametrize('failure',['none','subprocess','missing_marker','wrong_job'])
def test_authentication_replays_original_source_before_reuse(tmp_path,monkeypatch,failure):
    parent,proof=completed(tmp_path,monkeypatch)
    original=Path(parent['project_dir']);calls=[]
    monkeypatch.setattr(source,'original_source',lambda p:original)
    fields={k:proof[k] for k in ('reference','exports','accounting','job_id','report_hash')}
    if failure=='wrong_job':fields['job_id']='1'
    def run(args,**kw):
        calls.append(args)
        assert args[0]==sys.executable and args[-1]==proof['spec']['path']
        assert kw['cwd']==original and kw['env']['PYTHONPATH']=='' and kw['env']['PYTHONNOUSERSITE']=='1'
        assert 'campaign.check(s);receipt=worker.completed(s)' in args[-2]
        assert 'submission.accounting(s)' in args[-2]
        return SimpleNamespace(returncode=1 if failure=='subprocess' else 0,stderr='fixture failure',
            stdout='' if failure=='missing_marker' else 'SIX_S0_BASELINE='+json.dumps(fields))
    monkeypatch.setattr(source.subprocess,'run',run)
    if failure=='none':assert source.authenticate(proof['spec']['path'])==proof
    else:
        with pytest.raises(ValueError):source.authenticate(proof['spec']['path'])
    assert len(calls)==1


def test_foreign_worktree_authentication_uses_exact_clean_snapshot(tmp_path,monkeypatch):
    original=tmp_path/'old';original.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(original),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    (original/'module.py').write_text('value = 1\n');git('add','.');git('commit','-m','original')
    commit=git('rev-parse','HEAD');snapshot=capture_source_snapshot(original,require_clean=True)
    monkeypatch.setattr(source,'PARENT_COMMIT',commit)
    p=dict(project_dir=str(original),source_snapshot=snapshot)
    assert source.original_source(p)==original
    (original/'module.py').write_text('value = 2\n')
    with pytest.raises(ValueError):source.original_source(p)
    (original/'module.py').write_text('value = 1\n');(original/'unexpected.py').write_text('pass\n')
    with pytest.raises(ValueError):source.original_source(p)


def test_new_source_guard_forbids_changes_to_donor_numerics(monkeypatch):
    project=Path(source.__file__).resolve().parents[3];changed=[source.PACKAGE+'contracts.py'];calls=[]
    def git(p,*args):
        calls.append(args)
        if args[0]=='diff':return '\n'.join(changed)
        if args[0]=='ls-files':return '\n'.join(source.REQUIRED)
        return ''
    monkeypatch.setattr(source,'git',git);monkeypatch.setattr(source,'_source',lambda *a:dict(pinned=True))
    assert source.current(project,'f'*40)==dict(pinned=True)
    assert ('merge-base','--is-ancestor',source.BASELINE,'f'*40) in calls
    changed.append('src/hlt_classification/luka_fullstat_six_teacher/model.py')
    with pytest.raises(ValueError,match='Only additive'):source.current(project,'f'*40)


def test_recipe_and_node_reject_wrong_original_teacher_or_weight(tmp_path):
    parent=registered(tmp_path)
    expected=dict(parent['nodes'][c.TASK],branch='FULLSTAT_SIX_TEACHER_S0')
    assert c.node(parent)==expected
    for field,value in [('total_embedding_weight',10.),('total_logit_weight',.5),('coordinate','U100')]:
        bad=deepcopy(parent);bad['nodes'][c.TASK][field]=value
        with pytest.raises(ValueError):c.node(bad)
