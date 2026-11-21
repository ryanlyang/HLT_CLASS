"""Pooled macro/binary recovery, frozen anchors, live snapshots and native replay."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.operations import sixclass_full_recovery as a
from hlt_classification.data.cache_contracts import (
    load_json, sha256_file, atomic_publish_bytes, deterministic_npz_bytes,
)
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_sixclass_balanced_oscar import training, reporting, views
from hlt_classification.luka_sixclass_low_lr_oscar import model as new_model
from hlt_classification.jetclass2_delphes.cache import RamCache


def fixture_arrays():
    rng = np.random.default_rng(42)
    arrays = {}; probabilities = {}
    for i, role in enumerate(a.hbb.ROLES):
        y = np.tile(np.arange(6, dtype=np.int8), 3)
        p = rng.uniform(.1, 1, (18, 6)); p /= p.sum(1, keepdims=True)
        arrays.update({role+'_labels':y, role+'_identities':np.full((18,32),i,np.uint8),
                       role+'_pt':np.full(18,800. if i == 0 else 300.)})
        probabilities[role] = p
    return arrays, probabilities


def saved_result(tmp_path, arrays, probabilities):
    files = {}
    for role, key in zip(a.hbb.ROLES, ('predictions','outside_predictions')):
        path = tmp_path/(role+'.npz')
        data = {k:arrays[role+'_'+k] for k in ('identities','labels','pt')}
        atomic_publish_bytes(path, deterministic_npz_bytes(dict(data, probabilities=probabilities[role])))
        files[key] = file_ref(path)
    return dict(receipt=dict(files=files,selected_epoch=4,content_hash='c'*64),
                report=dict(selected_pass=4,full_hbb_selected_pass=17))


def test_pooled_macro_and_hbb_are_separate_and_not_regional_averages():
    arrays, p = fixture_arrays()
    metrics = a.pooled_metrics(arrays,p)
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([arrays[r+'_labels'] for r in a.hbb.ROLES])
    probs = np.concatenate([p[r] for r in a.hbb.ROLES])
    assert metrics['macro']['macro_ovr_auc'] == roc_auc_score(y,probs/probs.sum(1,keepdims=True),multi_class='ovr')
    regional = [reporting.evaluate_probabilities(arrays[r+'_labels'],p[r])['macro_ovr_auc'] for r in a.hbb.ROLES]
    assert metrics['macro']['macro_ovr_auc'] != np.mean(regional)
    assert metrics['hbb'] == a.hbb.binary_metrics(y,probs)
    assert metrics['hbb']['signal_count'] == metrics['hbb']['background_count'] == 6


def test_anchor_uses_same_primary_checkpoint_for_both_metrics(tmp_path):
    arrays,p = fixture_arrays();result = saved_result(tmp_path,arrays,p)
    value = a.anchor(result,arrays)
    assert value['epoch'] == 4  # The exploratory Hbb selection is 17.
    assert value['metrics'] == a.pooled_metrics(arrays,p)
    broken = deepcopy(result);broken['receipt']['selected_epoch'] = 17
    with pytest.raises(ValueError,match='Anchor'):a.anchor(broken,arrays)


@pytest.mark.parametrize('field',['labels','identities','pt'])
def test_prediction_population_joins_are_checked(tmp_path,field):
    arrays,p = fixture_arrays();result = saved_result(tmp_path,arrays,p)
    bad = deepcopy(arrays);bad['outside_validation_'+field][0] += 1
    with pytest.raises(ValueError,match='population'):a.anchor(result,bad)


def test_recovery_uses_metric_specific_fixed_anchors_and_accepts_negative():
    assert a.recovery(.82,.8,.9) == pytest.approx(20)
    assert a.recovery(.79,.8,.9) == pytest.approx(-10)
    assert a.recovery(.95,.8,.9) == pytest.approx(150)
    assert a.recovery(.82,.8,.8) is None
    assert a.recovery(.82,.8,.7) is None


def test_snapshot_freezes_committed_prefix_and_rejects_replacement():
    rows = [dict(epoch=e,content_hash=f'{e:064x}') for e in range(1,7)]
    saved = a.snapshot(rows[:3]);assert a.snapshot(rows,saved) == saved
    bad = deepcopy(rows);bad[0]['content_hash'] = '0'*64
    with pytest.raises(ValueError,match='Snapshot'):a.snapshot(bad,saved)
    with pytest.raises(ValueError):a.snapshot([])
    with pytest.raises(ValueError):a.snapshot(rows[1:])


def test_receipts_reject_changed_scope_and_parents():
    row = a.artifact('EPOCH',dict(analysis='a'*64),epoch=1)
    a.validate(row,'EPOCH',row['parents'])
    with pytest.raises(ValueError):a.validate(row,'EPOCH',dict(analysis='b'*64))
    row['policy']['checkpoint_reselection'] = True
    with pytest.raises(ValueError):a.validate(row,'EPOCH',row['parents'])
    assert a.POLICY['checkpoint_reselection'] is False


@pytest.mark.parametrize('family',['Previous','New'])
def test_epoch_replay_resume_parity_and_corruption(tmp_path,monkeypatch,family):
    arrays,p = fixture_arrays();metrics = a.pooled_metrics(arrays,p)
    from hlt_classification.luka_sixclass_low_lr_oscar.training import binary_hbb
    y = np.concatenate([arrays[r+'_labels'] for r in a.hbb.ROLES])
    joined = np.concatenate([p[r] for r in a.hbb.ROLES])
    row = dict(validation=reporting.evaluate_probabilities(arrays['validation_labels'],p['validation']),
               full_hbb=binary_hbb(y,joined))
    commits = [dict(epoch=e,content_hash=f'{e:064x}',row=row) for e in range(1,4)]
    calls=[]
    monkeypatch.setattr(a,'native_from_epoch',lambda family,r,device:calls.append(r['epoch']))
    caches = {r:SimpleNamespace(labels=arrays[r+'_labels'],role_name=r) for r in a.hbb.ROLES}
    monkeypatch.setattr(training,'predict',lambda native,cache,device:p[cache.role_name])
    monkeypatch.setenv('SLURM_JOB_ID','1234')
    parity=[];monkeypatch.setattr(a.hbb,'check_saved_predictions',lambda *args:parity.append(True))
    published = dict(report=dict(selected_pass=2)) if family == 'Previous' else None
    audit = dict(content_hash='a'*64)
    rows = a.evaluate_family(audit,family,commits,caches,arrays,tmp_path,published=published,device='cpu')
    assert all(r['metrics']==metrics for r in rows)
    assert calls == ([2,1,3] if published else [1,2,3])
    calls.clear()
    assert a.evaluate_family(audit,family,commits,caches,arrays,tmp_path,published=published,device='cpu') == rows
    assert calls == ([2] if published else [1])
    if published:assert parity == [True,True]
    bad = deepcopy(commits);bad[0]['row']['validation']['accuracy'] = -1
    with pytest.raises(ValueError,match='In-range'):
        a.evaluate_family(audit,family,bad,caches,arrays,tmp_path,published=published,device='cpu')


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_real_new_weaver_checkpoint_native_prediction_parity(tmp_path,device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_luka_fullstat import write_root
    torch.set_num_threads(1)
    path=tmp_path/'raw/s.root';write_root(path,[161]*6)
    record=dict(path='raw/s.root',sha256=sha256_file(path),entries=6,tree_key='tree',category='signal')
    member=dict(labels=np.arange(6,dtype=np.int8),jet_nparticles=np.full(6,3),hlt_jet_nparticles=np.full(6,2))
    blocks,_=views.work((tmp_path,record,0,0,member,np.ones(6,np.uint8),'D000','b'*64))
    cache=RamCache([blocks['validation']],role='validation',foundation_sha256='b'*64,coordinate_name='D000')
    student=new_model.new_model('DE000').to(device).eval()
    from hlt_classification.luka_sixclass_low_lr_oscar import training as nt
    expected=nt.predict(student,cache,device)
    state=tmp_path/'state.pt';torch.save(dict(student=student.state_dict(),epoch=3,update=3*12961),state)
    receipt=dict(state=file_ref(state),epoch=3)
    native=a.native_from_epoch('New',receipt,device)
    actual=training.predict(native,cache,device)
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError):a.native_from_epoch('New',dict(receipt,epoch=4),device)


def test_tables_use_same_anchors_both_columns_macro_first(capsys):
    def metric(m,h):return dict(macro=dict(macro_ovr_auc=m),hbb=dict(auc=h))
    report=dict(anchors=dict(M0HLT=dict(epoch=5,metrics=metric(.8,.9)),
                            OFFLINE=dict(epoch=7,metrics=metric(.9,.95))),new_epochs=1,
        epoch_history=dict(Previous=[dict(epoch=1,metrics=metric(.82,.91))],
                           New=[dict(epoch=1,metrics=metric(.85,.92))]))
    a.print_report(report);text=capsys.readouterr().out
    assert text.index('SIX-CLASS MACRO') < text.index('HBB-VERSUS-QCD')
    for v in ('20.00','50.00','40.00','566,032','epoch 5','epoch 7'):assert v in text
    assert 'no pT cut' in text and 'no training' in text


def test_queue_real_shell_dry_run_duplicate_and_explicit_resume(tmp_path):
    import subprocess
    import sys
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    def sp(path):
        p=Path(path).resolve()
        return '/'+p.drive[0].lower()+p.as_posix()[2:] if sys.platform=='win32' else str(p)
    base=tmp_path/'base';project=tmp_path/'project';(project/'scripts').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts/queue_luka_sixclass_full_validation.sh'
    script=project/'scripts'/source.name
    script.write_bytes(source.read_text().replace('/oscar/home/rlyang/hlt_classification',sp(base)).encode())
    root=base/'checkpoints/luka_sixclass_low_lr_oscar_fixture_r1';root.mkdir(parents=True)
    (root/'campaign_spec.json').write_text('{}');(root/'train_OFFLINE-9442678.out').write_text('')
    bindir=tmp_path/'bin';bindir.mkdir();calls=tmp_path/'calls'
    bodies={
        'git':'if [[ "$*" == *"rev-parse HEAD"* ]]; then printf "%040d\\n" 1; fi\n',
        'squeue':':\n','sacct':'echo "FAILED|"\n',
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
    assert subprocess.run(args,env=env,capture_output=True).returncode!=0
    resumed=subprocess.run(args+['resume'],env=dict(env,FIXTURE_JOB='12346'),capture_output=True,text=True,timeout=15)
    assert resumed.returncode==0,resumed.stderr
    lines=calls.read_text().splitlines()
    assert len(lines)==4 and all('--test-only' in lines[i] for i in (0,2))
    assert all('--gres=gpu:l40s:1' in line and 'scancel' not in line for line in lines)
