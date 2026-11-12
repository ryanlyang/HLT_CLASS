"""Original export batch contexts and authenticated pre-training recovery."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_embedding_bridge_hydra import banks, contracts as c, distributed as d, inputs, recovery


class BatchSensitive(torch.nn.Module):
    """Deliberate batch-context dependence makes regrouping failures unambiguous."""
    def encode(self, features, vectors, mask):
        rows=features[:,0,0]
        values=(rows+rows.max()/1000+len(rows)/100).unsqueeze(1).expand(-1,128)
        return torch.zeros((len(rows),11),device=rows.device),values


class Rows:
    def __init__(self,n=267):
        self.n=n;self.calls=[]
        self.identities=np.array([np.frombuffer(__import__('hashlib').sha256(str(i).encode()).digest(),np.uint8) for i in range(n)])
    def __len__(self):return self.n
    def batch(self,ix):
        self.calls.append(np.array(ix))
        return dict(features=np.array(ix,dtype=np.float32).reshape(-1,1,1),
            vectors=np.zeros((len(ix),4,1),np.float32),mask=np.ones((len(ix),1,1),bool))


def reference(view):
    m=BatchSensitive();ctx=d.Context(0,1,torch.device('cpu'));chunks=[]
    for start in range(0,len(view),128):
        chunks.append(m.encode(**d.arrays(view.batch(np.arange(start,min(start+128,len(view)))),ctx))[1].numpy())
    return np.concatenate(chunks)


def test_frozen_panel_keeps_original_neighbours_and_whole_batches_on_each_rank():
    view=Rows();expected=reference(view);indices=np.array([250,1,266,128,2]);view.calls.clear()
    actual=d.original_u100(BatchSensitive(),view,d.Context(0,1,torch.device('cpu')),indices)
    np.testing.assert_array_equal(actual,expected[indices]);assert [len(x) for x in view.calls]==[128,128,11]
    for rank in (0,1):
        view.calls.clear();ctx=d.Context(rank,2,torch.device('cpu'))
        def gather(local):
            missing=np.setdiff1d(np.arange(len(indices)),local[0])
            return [local,(missing,expected[indices[missing]])]
        ctx.all=gather
        actual=d.original_u100(BatchSensitive(),view,ctx,indices)
        np.testing.assert_array_equal(actual,expected[indices])
        assert [len(x) for x in view.calls]==([128,11] if rank==0 else [128])
    np.testing.assert_array_equal(d.original_u100(BatchSensitive(),view,d.Context(0,1,torch.device('cpu'))),expected)


@pytest.mark.parametrize('indices',([1,1],[-1],[267],[],[1.5]))
def test_invalid_replay_rows_fail(indices):
    with pytest.raises(ValueError):d.original_u100(BatchSensitive(),Rows(),d.Context(0,1,torch.device('cpu')),np.array(indices))


@pytest.mark.parametrize('corrupt',[False,True])
def test_replay_preserves_bank_and_rejects_large_errors(tmp_path,monkeypatch,corrupt):
    from hlt_classification.luka_embedding_bridge_hydra import model
    from test_luka_embedding_bridge_numerical import probe_record
    view=Rows();targets=reference(view)
    if corrupt:targets[0,0]+=.1
    mean=np.zeros(128,np.float64);scale=np.ones(128,np.float64)
    ds=SimpleNamespace(view=lambda role,coord:view,u100=lambda role:dict(targets=targets,mean=mean,scale=scale))
    (tmp_path/'setup').mkdir();checkpoint=tmp_path/'teacher.pt';torch.save({},checkpoint)
    spec=dict(root=str(tmp_path),content_hash='a'*64,embedding_hash='b'*64,u100_numerical_policy=c.u100_numerical_policy())
    monkeypatch.setattr(model,'Model',lambda task:BatchSensitive())
    monkeypatch.setattr(inputs,'transferred',lambda spec:SimpleNamespace(path=lambda ref:checkpoint,manifest=dict(teacher=file_ref(checkpoint))))
    monkeypatch.setattr(banks.numerical,'local_probe',lambda *a:probe_record(len(view)))
    saved=targets.copy()
    if corrupt:
        with pytest.raises(ValueError,match='Original-batch'):banks.check_u100(spec,ds,d.Context(0,1,torch.device('cpu')))
    else:banks.check_u100(spec,ds,d.Context(0,1,torch.device('cpu')))
    record=load_json(tmp_path/'setup/u100_replay.json');c.validate(record,'U100_REPLAY')
    assert record['passed'] is (not corrupt)
    assert (record['atol'],record['rtol'])==(1e-4,1e-4)
    assert record['legacy_elementwise_diagnostic_only']is True
    assert (record['exceeding_coordinates']>0)==corrupt
    np.testing.assert_array_equal(targets,saved)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    from test_luka_embedding_bridge_hydra import test_original_physical_preparation_and_quarter_views,synthetic_dataset
    root=tmp_path/'old';root.mkdir()
    test_original_physical_preparation_and_quarter_views(root,monkeypatch)
    old=synthetic_dataset().spec
    old.update(root=str(root),anchor=dict(evidence=dict(prepared='c'*64)),transfer_hash='d'*64,native_hash='e'*64,
               bundle_root=str(root),memory_mb=48000,recipe=c.recipe(),source_commit=recovery.SOURCE)
    write_immutable_json(root/'campaign_spec.json',old)
    t=inputs.transferred(old);t.manifest['teacher']=dict(sha256='f'*64)
    write_immutable_json(root/'setup/preparation.json',c.artifact('PREPARATION',parents=dict(spec=old['content_hash']),
        seconds=8000.,host_peak_bytes=2**30,data=load_json(root/'setup/data/manifest.json')['content_hash']))
    failed=dict(contract='LUKA_EMBEDDING_BRIDGE_HYDRA_U100_REPLAY/v1',schema_version=1,
        parents=dict(spec=old['content_hash'],teacher='f'*64,original=old['embedding_hash']),passed=False,rows=4096,
        atol=1e-4,rtol=1e-4,maximum_error=.0014567)
    write_immutable_json(root/'setup/u100_replay.json',with_content_hash(failed))
    monkeypatch.setattr(recovery,'source_spec',lambda path:old)
    monkeypatch.setattr(recovery,'terminal_jobs',lambda spec:'162131')
    return old


def test_recovery_preserves_payload_inodes_rebinds_manifests_and_credits_only_missing_links(prepared,tmp_path):
    old=prepared;record=recovery.inspect(Path(old['root'])/'campaign_spec.json')
    new=dict(old,root=str(tmp_path/'new'),content_hash='7'*64,preparation_reuse=record)
    (Path(new['root'])/'setup').mkdir(parents=True)
    assert recovery.credit(new)==record['payload_bytes']>0
    before=file_ref(Path(old['root'])/'setup/data/manifest.json')
    result=recovery.materialize(new)
    assert recovery.credit(new)==0
    ds=inputs.Dataset(new);assert result['parents']==inputs.data_parents(new)
    assert file_ref(Path(old['root'])/'setup/data/manifest.json')==before
    for ref in record['arrays']:
        a=Path(old['root'])/'setup/data'/ref['path'];b=Path(new['root'])/'setup/data'/ref['path']
        assert os.path.samefile(a,b)
    np.testing.assert_array_equal(ds.u100('train')['targets'],inputs.Dataset(old).u100('train')['targets'])
    assert not (Path(new['root'])/'setup/u100_replay.json').exists()
    assert not (Path(new['root'])/'setup/acceptance.json').exists()
    with pytest.raises(FileExistsError):recovery.materialize(new)


@pytest.mark.parametrize('failure',('array','manifest','partial','acceptance','teacher','bank','parent'))
def test_recovery_rejects_changed_incomplete_or_wrong_lineage(prepared,tmp_path,failure):
    old=prepared;root=Path(old['root']);record=recovery.inspect(root/'campaign_spec.json')
    if failure=='array':
        with (root/'setup/data/train/D000/features.npy').open('r+b') as handle:
            handle.seek(-4,2);handle.write(b'bad!')
    elif failure=='manifest':
        path=root/'setup/data/train/D000/manifest.json';a=load_json(path);a['role']='validation';path.write_text(__import__('json').dumps(a))
    elif failure=='partial':(root/'setup/data/extra.partial.npy').write_bytes(b'partial')
    elif failure=='acceptance':write_immutable_json(root/'setup/acceptance.json',dict(fake=True))
    elif failure=='teacher':inputs.transferred(old).manifest['teacher']['sha256']='8'*64
    elif failure=='bank':
        new=dict(old,root=str(tmp_path/'new'),embedding_hash='9'*64,preparation_reuse=record)
        with pytest.raises(ValueError,match='different'):recovery.check(record,new)
        return
    else:
        new=dict(old,root=str(tmp_path/'new'),transfer_hash='9'*64,preparation_reuse=record)
        with pytest.raises(ValueError,match='different'):recovery.check(record,new)
        return
    with pytest.raises((ValueError,PermissionError)):recovery.inspect(root/'campaign_spec.json')


def test_accounting_and_original_source_are_required(monkeypatch,tmp_path):
    old=dict(root=str(tmp_path),project_dir=str(tmp_path/'code'),source_commit=recovery.SOURCE,
             source_snapshot={'git_commit':recovery.SOURCE})
    monkeypatch.setattr(recovery,'validate_source_snapshot',lambda *args,**kwargs:None)
    states={'accounting':'162131|FAILED|1:0|\n','active':''}
    def run(cmd,**kw):
        if 'inspect-submission' in cmd:return SimpleNamespace(stdout='{"bootstrap":{"setup":"162131","release":"162132"},"science":{}}')
        return SimpleNamespace(stdout=states['accounting' if cmd[0]=='sacct' else 'active'])
    monkeypatch.setattr(recovery.subprocess,'run',run)
    assert recovery.terminal_jobs(old)=='162131'
    states['accounting']='162131|RUNNING|0:0|\n'
    with pytest.raises(ValueError,match='not confirmed'):recovery.terminal_jobs(old)
    states['accounting']='162131|FAILED|1:0|\n';states['active']='162132\n'
    with pytest.raises(ValueError,match='still active'):recovery.terminal_jobs(old)


@pytest.fixture
def foreign_source(tmp_path,monkeypatch):
    """Two real Git revisions/worktrees; only the scheduler is simulated."""
    import subprocess
    from hlt_classification.provenance import capture_source_snapshot
    current=tmp_path/'current';current.mkdir();old_project=tmp_path/'original'
    def git(*args):
        return subprocess.run(['git','-C',str(current),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init','--quiet');git('config','user.name','Recovery Test');git('config','user.email','recovery@example.invalid')
    git('config','core.autocrlf','false');git('config','commit.gpgsign','false')
    scripts=current/'scripts';scripts.mkdir();cli=scripts/'luka_embedding_bridge_hydra.py'
    cli.write_text('''import json,sys
from pathlib import Path
assert sys.argv[1:3] == ['inspect-submission', '--spec']
old = json.loads(Path(sys.argv[3]).read_text())
assert Path(__file__).resolve().parents[1] == Path(old['project_dir']).resolve()
print(json.dumps({'bootstrap': {'setup': '162131', 'release': '162132'}, 'science': {}}))
''',encoding='utf-8')
    git('add','scripts');git('commit','--quiet','-m','Original CLI');commit=git('rev-parse','HEAD')
    cli.write_text('raise RuntimeError("Executed newer CLI for an old campaign")\n',encoding='utf-8')
    git('add','scripts');git('commit','--quiet','-m','Different current CLI')
    git('worktree','add','--quiet','--detach',str(old_project),commit)
    root=tmp_path/'run';root.mkdir()
    old=dict(root=str(root),project_dir=str(old_project),source_commit=commit,
             source_snapshot=capture_source_snapshot(old_project))
    write_immutable_json(root/'campaign_spec.json',old)
    monkeypatch.setattr(recovery,'SOURCE',commit)
    real_run=subprocess.run;calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]=='sacct':return SimpleNamespace(stdout='162131|FAILED|1:0|\n')
        if cmd[0]=='squeue':return SimpleNamespace(stdout='')
        return real_run(cmd,**kwargs)
    monkeypatch.setattr(recovery.subprocess,'run',run)
    return old,current,calls


def test_terminal_jobs_authenticates_and_executes_real_foreign_checkout(foreign_source):
    from hlt_classification.luka_fullsim.foundation import _source
    old,current,calls=foreign_source
    # Reproduce the exact previous failure without mocking the source checker.
    with pytest.raises(ValueError,match='outside the pinned project'):
        _source(Path(old['project_dir']),old['source_commit'])
    assert recovery.terminal_jobs(old)=='162131'
    executed=[cmd for cmd in calls if 'inspect-submission' in cmd]
    assert len(executed)==1
    assert str(Path(old['project_dir'])/'scripts/luka_embedding_bridge_hydra.py') in executed[0]
    assert str(current/'scripts/luka_embedding_bridge_hydra.py') not in executed[0]


@pytest.mark.parametrize('change',('tracked','untracked','wrong_checkout','wrong_commit','wrong_snapshot'))
def test_foreign_source_tampering_stops_before_old_cli_or_scheduler(foreign_source,change):
    old,current,calls=foreign_source;old=copy.deepcopy(old);root=Path(old['project_dir'])
    if change=='tracked':(root/'scripts/luka_embedding_bridge_hydra.py').write_text('changed\n',encoding='utf-8')
    elif change=='untracked':(root/'unexpected.txt').write_text('unexpected\n',encoding='utf-8')
    elif change=='wrong_checkout':old['project_dir']=str(current)
    elif change=='wrong_commit':old['source_commit']='0'*40
    else:old['source_snapshot']['tracked_files_sha256']='0'*64
    with pytest.raises(ValueError):recovery.terminal_jobs(old)
    assert not any('inspect-submission' in cmd or cmd[0] in ('sacct','squeue') for cmd in calls)


def test_no_reuse_credit_for_unregistered_preparation():
    assert recovery.credit(dict(root='/unused'))==0


def test_native_teacher_matches_original_export_loop_on_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from test_luka_embedding_bridge_hydra import synthetic_dataset
    from hlt_classification.luka_embedding_bridge_hydra.model import Model
    from hlt_classification.luka_embedding_aux.model import head_input
    ds=synthetic_dataset();base=ds.view('train','U100');ctx=d.Context(0,1,torch.device('cuda',0))
    class Repeated:
        def __len__(self):return 267
        def batch(self,ix):return base.batch(np.asarray(ix)%len(base))
    view=Repeated()
    torch.manual_seed(23);m=Model('TR').to(ctx.device).eval()
    ix=np.array([250,3,266,128,0]);chunks=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for start in range(0,len(view),128):
            raw=view.batch(np.arange(start,min(start+128,len(view))))
            _,expected=head_input(m,**d.arrays(raw,ctx));chunks.append(expected.float().cpu().numpy())
    actual=d.original_u100(m,view,ctx,ix)
    np.testing.assert_array_equal(actual,np.concatenate(chunks)[ix])


@pytest.mark.parametrize('reuse',[False,True])
def test_setup_checks_before_construction_or_uses_recovery_first(tmp_path,monkeypatch,reuse):
    from hlt_classification.luka_embedding_bridge_hydra import worker
    spec=dict(root=str(tmp_path),content_hash='a'*64,project_dir=str(tmp_path/'code'),preparation_reuse={} if reuse else None)
    monkeypatch.setattr(worker.campaign,'check',lambda spec:None)
    monkeypatch.setattr(worker.execution,'allocation',lambda spec:[])
    monkeypatch.setattr(worker,'require_job',lambda spec,task:'100')
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if 'rank' in cmd:raise RuntimeError('probe stops')
    monkeypatch.setattr(worker.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='probe stops'):worker.run(spec,'setup')
    if reuse:
        assert len(calls)==2 and 'prepare' in calls[0] and calls[1][-1]=='setup'
    else:
        assert len(calls)==1 and calls[0][-1]=='input_probe'
        assert all('prepare' not in call for call in calls)
