"""Exact one-file retirement, verified manifests and live dependency protection."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from hlt_classification.operations import hydra_contrast_key_retirement as op


def hashed(value):
    return dict(value,content_hash=op.hashlib.sha256(op.canonical(value)).hexdigest())


def fixture(base,monkeypatch):
    folder=base/op.DELTA;(folder/'keys').mkdir(parents=True);(base/'runs').mkdir()
    payload=b'old TRAIN keys'*100;target=folder/op.MEMBER;target.write_bytes(payload)
    monkeypatch.setattr(op,'EXPECTED_BYTES',len(payload))
    teacher=dict(path='/original/teacher.pt',sha256='d'*64,bytes=100)
    owner=hashed(dict(contract='owner/v1'))
    old=hashed(dict(counts=dict(train=12),imported=dict(content_hash='a'*64),
        anchor=dict(evidence=dict(prepared='b'*64,models=dict(OFFLINE=dict(checkpoint=teacher))))))
    replay=hashed(dict(contract='replay/v1'))
    ref=op.file_ref(target,base);ref={k:ref[k] for k in ('bytes','sha256')}
    keys=hashed(dict(contract='LUKA_U100_CONTRASTIVE_KEYS/v1',role='train',coordinate='OFFLINE',width=128,rows=12,
        payload=dict(path='keys.npz',**ref),teacher=teacher,parents=dict(spec=owner['content_hash'],
        imported='a'*64,prepared='b'*64)))
    (folder/'keys/manifest.json').write_text(json.dumps(keys))
    m=op.file_ref(folder/'keys/manifest.json',base)
    transfer=hashed(dict(contract='LUKA_U100_CONTRASTIVE_HYDRA_TRANSFER/v1',origin_spec=old,key_owner=owner,
        original_replay=replay,parents=dict(origin=old['content_hash'],replay=replay['content_hash']),
        files=[dict(path=op.MEMBER,**ref),dict(path='keys/manifest.json',bytes=m['bytes'],sha256=m['sha256'])]))
    (folder/'manifest.json').write_text(json.dumps(transfer))
    (folder/'selected.pt').write_bytes(b'keep model');(folder/'predictions.npz').write_bytes(b'keep predictions')
    return target


def clear():return dict(jobs={},metadata=[],blocked=False)


def test_preview_and_exact_retirement_preserve_every_other_file(tmp_path,monkeypatch):
    target=fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in (tmp_path/op.DELTA).rglob('*') if p.is_file() and p!=target}
    op.retire(tmp_path,guard=clear);assert target.exists() and not (tmp_path/'retired_campaigns').exists()
    result=op.retire(tmp_path,guard=clear,execute=True,authorization=op.AUTH,source_commit='f'*40)
    assert not target.exists() and result['jobs_changed'] is False
    for p,b in before.items():assert p.read_bytes()==b
    directory=tmp_path/'retired_campaigns'/op.OPERATION
    op.hashed(json.loads((directory/'intent.json').read_bytes()));op.hashed(json.loads((directory/'complete.json').read_bytes()))
    with pytest.raises(ValueError,match='Existing'):op.retire(tmp_path,guard=clear,execute=True,authorization=op.AUTH)


@pytest.mark.parametrize('problem',['checksum','manifest_hash','payload_path','role','owner','transfer_member'])
def test_corrupt_or_wrong_targets_are_never_deleted(tmp_path,monkeypatch,problem):
    target=fixture(tmp_path,monkeypatch);m=tmp_path/op.DELTA/'keys/manifest.json'
    if problem=='checksum':target.write_bytes(b'x'*target.stat().st_size)
    elif problem=='manifest_hash':m.write_text(m.read_text()+'broken')
    else:
        keys=json.loads(m.read_text());keys.pop('content_hash')
        if problem=='payload_path':keys['payload']['path']='../../important.npy'
        elif problem=='role':keys['role']='validation'
        elif problem=='owner':keys['parents']['spec']='e'*64
        else:
            p=tmp_path/op.DELTA/'manifest.json';value=json.loads(p.read_text());value.pop('content_hash')
            value['files'][0]['path']='different.npy';p.write_text(json.dumps(hashed(value)))
        if problem!='transfer_member':m.write_text(json.dumps(hashed(keys)))
    with pytest.raises((ValueError,KeyError)):op.retire(tmp_path,guard=clear,execute=True,authorization=op.AUTH,source_commit='f'*40)
    assert target.exists()


def test_live_and_newly_arriving_consumers_block_retirement(tmp_path,monkeypatch):
    target=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='live dependency'):
        op.retire(tmp_path,guard=lambda:dict(blocked=True),execute=True,authorization=op.AUTH,source_commit='f'*40)
    answers=iter([clear(),clear(),dict(blocked=True)])
    with pytest.raises(ValueError,match='New live'):
        op.retire(tmp_path,guard=lambda:next(answers),execute=True,authorization=op.AUTH,source_commit='f'*40)
    assert target.exists() and (tmp_path/'retired_campaigns'/op.OPERATION/'intent.json').exists()


def test_transitive_job_graph_finds_alias_paths_and_unknown_jobs_fail_closed(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);run=tmp_path/'runs/current';run.mkdir()
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'manifest.json').write_text(json.dumps(dict(keys='/home/rlyang/hlt_u100_pairs/'+op.DELTA)))
    (run/'campaign_spec.json').write_text(json.dumps(hashed(dict(root=str(run),parent=str(parent)))))
    jobs={'123':'JobId=123 StdOut='+str(run/'worker.out')}
    result=op.audit(tmp_path,jobs);assert result['blocked'] and len(result['metadata'])==3
    with pytest.raises(ValueError,match='Cannot resolve'):op.audit(tmp_path,{'124':'JobId=124 WorkDir=/unknown'})


def test_escape_paths_and_wrong_authorization_are_rejected(tmp_path,monkeypatch):
    target=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Noncanonical'):op.safe(tmp_path/'runs/../runs',tmp_path,directory=True)
    with pytest.raises(ValueError,match='authorization'):op.retire(tmp_path,guard=clear,execute=True,authorization='wrong')
    assert target.exists()
