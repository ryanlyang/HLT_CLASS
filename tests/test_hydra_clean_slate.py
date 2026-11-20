"""Keep-set retirement: archives, current dependencies, and bounded old roots."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tarfile
import pytest

from hlt_classification.operations import hydra_clean_slate as op


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    base=tmp_path/'workspace';base.mkdir()
    commit='a'*40;project=base/('code_balanced_reco_'+commit[:8]);project.mkdir()
    primary=base/'code_65952ed3';common=primary/'.git';common.mkdir(parents=True)
    keep=dict(dataset='datasets/luka_qcdrun2com_higgs2p_sb_v1',environment='environment_pair_hydra_v1',
              run_prefix='balanced_reco_600k_')
    protected=[]
    for path in (project/'src/new.py',common/'objects/KEEP',base/keep['dataset']/'raw/new.root',
                 base/keep['environment']/'venv/bin/python',base/'runs/balanced_reco_600k_ebd15b7e_r1.launch/log.out',
                 base/'runs/balanced_reco_600k_new_r1/campaign_spec.json',
                 base/'retired_campaigns/previous/complete.json'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CURRENT DEPENDENCY');protected.append(path)
    old=[];reports=[]
    for name in ('bundle_65952ed3','raw_next_0a6721ed','imports'):
        path=base/name;path.mkdir();(path/'old.root').write_bytes(b'old payload');old.append(path)
        report=path/'manifest.json';report.write_text('{"source":"old dataset"}');reports.append(report)
    for name in ('reconstruction_stability_3ccc35ae_4x1080ti_r1','new_balanced_4a61a34c_r1','sixclass_tail_504a8314_r1'):
        path=base/'runs'/name;path.mkdir();(path/'selected.pt').write_bytes(b'old model');old.append(path)
        report=path/'report.json';report.write_text('{"auc":0.965}');reports.append(report)
    source=base/'code_old';source.mkdir();(source/'source.py').write_bytes(b'old pushed source');old.append(source)
    dirty=base/'code_dirty';dirty.mkdir();(dirty/'user.py').write_bytes(b'UNCOMMITTED USER WORK');protected.append(dirty/'user.py')
    unknown=base/'personal_notes';unknown.mkdir();(unknown/'KEEP').write_bytes(b'UNKNOWN USER FILE');protected.append(unknown/'KEEP')
    monkeypatch.setattr(op,'idle',lambda:None)
    monkeypatch.setattr(op,'verified_keep',lambda *args:(keep,common))
    def candidate(b,p,c):
        if p.name=='code_dirty':return None,'uncommitted work retained'
        return dict(path=str(p),kind='worktree',commit='b'*40,signature=op.signature(p)),None
    monkeypatch.setattr(op,'code_candidate',candidate)
    def git(args,**kw):
        assert args[-3:-1]==['worktree','remove'];shutil.rmtree(Path(args[-1]));return SimpleNamespace(stdout='',returncode=0)
    monkeypatch.setattr(op,'run',git)
    return base,project,commit,old,protected,reports


def execute(w):
    base,project,commit,*_=w
    return op.retire(base,project,commit,execute=True,authorization=op.AUTH,helper_sha256='c'*64)


def test_preview_lists_old_inputs_and_all_prior_campaigns_without_deleting(workspace):
    base,project,commit,old,protected,_=workspace
    plan=op.retire(base,project,commit)
    assert {Path(r['path']) for r in plan['candidates']}==set(old)
    assert all(p.exists() for p in old+protected)
    assert not (base/'retired_campaigns'/op.OPERATION).exists()


def test_execute_archives_reports_and_keeps_new_data_runtime_source_git_and_runs(workspace):
    base,_,_,old,protected,reports=workspace;before={p:p.read_bytes() for p in protected}
    expected={p.relative_to(base).as_posix():p.read_bytes() for p in reports}
    result=execute(workspace)
    assert all(not p.exists() for p in old)
    assert all(p.read_bytes()==v for p,v in before.items())
    folder=base/'retired_campaigns'/op.OPERATION
    with tarfile.open(folder/'historical_metadata.tar.gz') as archive:
        for name,data in expected.items():assert archive.extractfile(name).read()==data
    assert result['removed_directories']==len(old)
    assert (folder/'complete.json').exists()
    with pytest.raises(ValueError,match='Existing clean-slate record'):execute(workspace)


def test_payload_files_are_never_read_for_archive_or_hash(workspace,monkeypatch):
    original=Path.open
    def guarded(path,*args,**kw):
        if path.suffix in ('.root','.pt','.npz','.npy'):pytest.fail('Binary payload was opened')
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    execute(workspace)


def test_queue_change_prevents_deletion(workspace,monkeypatch):
    calls=[]
    def idle():
        calls.append(1)
        if len(calls)==2:raise ValueError('Queue is not empty')
    monkeypatch.setattr(op,'idle',idle)
    with pytest.raises(ValueError,match='Queue is not empty'):execute(workspace)
    assert all(p.exists() for p in workspace[3])


def test_wrong_authorization_stops_before_archive_or_delete(workspace):
    base,project,commit,old,*_=workspace
    with pytest.raises(ValueError,match='authorization'):
        op.retire(base,project,commit,execute=True,authorization='WRONG',helper_sha256='c'*64)
    assert all(p.exists() for p in old)


def test_new_keep_contract_failure_stops_all_removal(workspace,monkeypatch):
    def rejected(*args):raise ValueError('New source still depends on retired campaign artifacts')
    monkeypatch.setattr(op,'verified_keep',rejected)
    with pytest.raises(ValueError,match='still depends'):execute(workspace)
    assert all(p.exists() for p in workspace[3])


def test_changed_root_after_inventory_stops_with_archive_preserved(workspace,monkeypatch):
    base,_,_,old,*_=workspace;original=op.archive_metadata
    def change(*args):
        result=original(*args);(old[0]/'new_file').write_text('changed');return result
    monkeypatch.setattr(op,'archive_metadata',change)
    with pytest.raises(ValueError,match='directory changed'):execute(workspace)
    folder=base/'retired_campaigns'/op.OPERATION
    assert (folder/'archive.json').exists() and not (folder/'complete.json').exists()
    assert all(p.exists() for p in old)


def test_partial_delete_has_archive_and_journal_and_no_blind_retry(workspace,monkeypatch):
    base,_,_,old,*_=workspace;original=op.shutil.rmtree;calls=[]
    def fail(path,*args,**kw):
        calls.append(path)
        if len(calls)==2:raise OSError('injected unlink error')
        return original(path,*args,**kw)
    monkeypatch.setattr(op.shutil,'rmtree',fail)
    with pytest.raises(OSError,match='injected unlink'):execute(workspace)
    folder=base/'retired_campaigns'/op.OPERATION
    assert (folder/'archive.json').exists() and not (folder/'complete.json').exists()
    assert len((folder/'deleted.jsonl').read_text().splitlines())==1
    with pytest.raises(ValueError,match='Existing clean-slate record'):execute(workspace)


def test_removal_plan_cannot_overlap_kept_subtree(workspace,monkeypatch):
    base,project,_,_,_,_=workspace;common=base/'code_65952ed3/.git'
    keep=dict(environment='imports',run_prefix='balanced_reco_600k_')
    # An explicit dependency is kept even when its name is an old-input name.
    candidates,retained,_=op.inventory(base,project,keep,common)
    assert str(base/'imports') in retained
    assert str(base/'imports') not in [r['path'] for r in candidates]


def test_archive_bytes_have_authenticated_index(workspace):
    execute(workspace);folder=workspace[0]/'retired_campaigns'/op.OPERATION
    index=json.loads((folder/'archive.json').read_text())
    assert hashlib.sha256((folder/'historical_metadata.tar.gz').read_bytes()).hexdigest()==index['archive_sha256']
    assert all('sha256' in row for row in index['files'])


def test_corrupt_archive_stops_before_any_payload_deletion(workspace,monkeypatch):
    original=op.archive_metadata
    def corrupt(base,candidates,path):
        rows=original(base,candidates,path);path.write_bytes(b'corrupt archive');return rows
    monkeypatch.setattr(op,'archive_metadata',corrupt)
    with pytest.raises(tarfile.ReadError):execute(workspace)
    assert all(p.exists() for p in workspace[3])


def test_changed_report_after_archiving_is_not_deleted(workspace,monkeypatch):
    original=op.archive_metadata;report=workspace[-1][0]
    def changed(*args):
        rows=original(*args);report.write_text('{"modified":"after archival"}');return rows
    monkeypatch.setattr(op,'archive_metadata',changed)
    with pytest.raises(ValueError,match='source metadata changed'):execute(workspace)
    assert all(p.exists() for p in workspace[3])


def test_unreadable_directory_prevents_deletion_before_archive_is_complete(workspace,monkeypatch):
    def blocked(root,*,onerror,**kwargs):
        onerror(PermissionError('unreadable historical metadata'))
        return iter(())
    monkeypatch.setattr(op.os,'walk',blocked)
    with pytest.raises(PermissionError,match='unreadable'):execute(workspace)
    assert all(p.exists() for p in workspace[3])


@pytest.fixture
def keep_probe(tmp_path,monkeypatch):
    base=tmp_path/'keep';base.mkdir();commit='a'*40
    project=base/('code_balanced_reco_'+commit[:8]);project.mkdir()
    common=base/'code_65952ed3/.git';common.mkdir(parents=True)
    keep_bytes=(Path(__file__).parents[1]/op.KEEP_FILE).read_bytes();keep=json.loads(keep_bytes)
    config=project/op.KEEP_FILE;config.parent.mkdir(parents=True);config.write_bytes(keep_bytes)
    dataset=base/keep['dataset'];(dataset/'foundation').mkdir(parents=True)
    (dataset/'transfer_complete.json').write_text('{}');(dataset/'foundation/producer.json').write_text('{}')
    env=base/keep['environment'];(env/'venv/bin').mkdir(parents=True);(env/'python').mkdir()
    (env/'venv/bin/python').write_bytes(b'KEEP interpreter')
    (env/'venv/pyvenv.cfg').write_text('home = '+str(env/'python')+'\n')
    state=dict(common=str(common),dirty='',policy=dict(contract=keep['storage_policy'],schema_version=2,
        other_campaign_dependency=None,protected_initial_allowance_bytes=0),paths=[str(env/'venv/lib'),str(project/'src')])
    def command(args,**kw):
        if args[0]=='git':
            sub=args[3:]
            if sub==['rev-parse','HEAD']:out=commit
            elif sub[:1]==['status']:out=state['dirty']
            elif sub==['rev-parse','--git-common-dir']:out=state['common']
            elif sub[:1]==['show']:out=keep_bytes.decode()
            else:assert sub[:1]==['merge-base'];out=''
        else:
            assert args[0]==str(env/'venv/bin/python') and kw['timeout']==300
            out=json.dumps(dict(storage_policy=state['policy'],paths=state['paths']))
        return SimpleNamespace(stdout=out,returncode=0)
    monkeypatch.setattr(op,'run',command)
    return base,project,commit,state,env


def test_real_keep_contract_retains_common_git_interpreter_and_imported_dependencies(keep_probe):
    base,project,commit,_,_=keep_probe
    keep,common=op.verified_keep(base,project,commit)
    assert keep['content_hash']==op.KEEP_HASH and common==base/'code_65952ed3/.git'


@pytest.mark.parametrize('failure,expected',[
    ('dirty','New source is dirty'),('common','Unexpected shared Git'),
    ('v1','still depends'),('import','Imported runtime path depends'),('pth','metadata references'),
    ('interpreter','Runtime interpreter depends')])
def test_required_dependency_failures_block_clean_slate(keep_probe,failure,expected):
    base,project,commit,state,env=keep_probe
    if failure=='dirty':state['dirty']=' M src/user.py'
    elif failure=='common':state['common']=str(base/'another.git');Path(state['common']).mkdir()
    elif failure=='v1':state['policy']['contract']=state['policy']['contract'].replace('/v2','/v1')
    elif failure=='import':state['paths'].append(str(base/'bundle_65952ed3/module.py'))
    elif failure=='pth':
        p=env/'venv/lib/python3.10/site-packages/old.pth';p.parent.mkdir(parents=True)
        p.write_text(str(base/'bundle_65952ed3')+'\n')
    elif failure=='interpreter':(env/'venv/pyvenv.cfg').write_text('home = '+str(base/'bundle_65952ed3')+'\n')
    with pytest.raises(ValueError,match=expected):op.verified_keep(base,project,commit)


def test_altered_keep_manifest_is_rejected(keep_probe):
    base,project,commit,*_=keep_probe
    p=project/op.KEEP_FILE;value=json.loads(p.read_bytes());value['historical_artifacts_required']=True
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='keep contract differs'):op.verified_keep(base,project,commit)
