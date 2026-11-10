"""C06 must reuse exactly the completed C03 TRAIN targets without reclustering."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from hlt_classification.data.cache_contracts import write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_embedding_sweep.banks import Bank as Offline
from hlt_classification.luka_fullstat_subclass import targets as previous
from hlt_classification.luka_fullstat_subclass_c06 import targets,banks
from test_luka_fullstat_subclass_targets import target_spec


def frozen(tmp_path,rows=330):
    tmp_path.mkdir(exist_ok=True)
    old,tr=target_spec(tmp_path,rows=rows)
    rec=old['reuse']['banks']['OFFLINE']
    offline=Offline(Path(rec['manifest']['path']).parent,tr,refs=rec['refs'])
    previous.prepare(old,tr,offline)
    path=Path(old['root'])/'campaign_spec.json';write_immutable_json(path,old)
    folder=Path(old['root'])/'subclasses'
    proof=dict(manifest_hash=previous.check_manifest(old)['content_hash'],files=[file_ref(folder/n) for n in targets.FILES])
    spec=with_content_hash(dict(old,root=str(tmp_path/'luka_fullstat_subclass_c06_fresh'),
        reuse=dict(old['reuse'],parent_spec=file_ref(path),subclasses=proof),resume=None))
    Path(spec['root']).mkdir()
    return spec,tr,offline,old


def test_fresh_and_recovery_keep_exact_bank_bytes_no_fit(tmp_path,monkeypatch):
    spec,tr,offline,old=frozen(tmp_path)
    monkeypatch.setattr(previous,'fit',lambda *a:pytest.fail('No subclass reclustering'))
    bank=targets.prepare(spec,tr,offline)
    for name in targets.FILES:
        assert (Path(spec['root'])/'subclasses'/name).read_bytes()==(Path(old['root'])/'subclasses'/name).read_bytes()
    assert bank.manifest['content_hash']==spec['reuse']['subclasses']['manifest_hash']
    assert not bank.targets.flags.writeable
    np.testing.assert_array_equal(bank.identities,tr.identities)
    with pytest.raises(FileExistsError):targets.prepare(spec,tr,offline)
    path=Path(spec['root'])/'campaign_spec.json';write_immutable_json(path,spec)
    resumed=with_content_hash(dict(spec,root=str(tmp_path/'luka_fullstat_subclass_c06_retry'),resume=dict(spec=file_ref(path))))
    Path(resumed['root']).mkdir()
    again=targets.prepare(resumed,tr,offline)
    np.testing.assert_array_equal(again.targets,bank.targets)


@pytest.mark.parametrize('bad',['validation','coordinate','row_order','prepared','parent_bytes','target_bytes','manifest_hash','files'])
def test_wrong_role_population_or_target_identity_stops(tmp_path,bad):
    spec,tr,offline,old=frozen(tmp_path)
    tr=deepcopy(tr)
    if bad=='validation':tr.role='validation'
    elif bad=='coordinate':tr.coordinate_name='U100'
    elif bad=='row_order':tr.identities=tr.identities[::-1].copy()
    elif bad=='prepared':tr.foundation_sha256='f'*64
    elif bad=='parent_bytes':Path(spec['reuse']['parent_spec']['path']).write_bytes(b'{}')
    elif bad=='target_bytes':(Path(old['root'])/'subclasses/targets.npz').write_bytes(b'bad')
    elif bad=='manifest_hash':spec['reuse']['subclasses']['manifest_hash']='f'*64
    elif bad=='files':spec['reuse']['subclasses']['files']=[]
    with pytest.raises((ValueError,PermissionError)):targets.prepare(spec,tr,offline)
    assert not (Path(spec['root'])/'subclasses').exists()


def test_all_four_bank_readers_keep_original_keys_and_probabilities(tmp_path,monkeypatch):
    spec,tr,offline,old=frozen(tmp_path,6006)
    monkeypatch.setattr(previous,'fit',lambda *a:pytest.fail('No target refit'))
    loaded,sampler=banks.load(spec,tr,'cpu')
    assert set(loaded)=={'U100','CONCAT','FUSION'}
    expected=offline.targets/np.maximum(np.linalg.norm(offline.targets,axis=1,keepdims=True),1e-12)
    np.testing.assert_array_equal(sampler.keys.cpu().numpy(),expected)
    np.testing.assert_array_equal(sampler.subclass_bank.targets,previous.Bank(old,tr).targets)
    (Path(spec['root'])/'subclasses/prototypes.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):targets.check_manifest(spec)


def source_fixture(tmp_path,monkeypatch):
    import json
    from hlt_classification.luka_fullstat_subclass import contracts as oldc
    from hlt_classification.luka_fullstat_subclass_c06 import contracts as c,source
    spec,tr,offline,old=frozen(tmp_path)
    inventory=tmp_path/'historical.py';inventory.write_bytes(b'# frozen historical runtime\n')
    parent=oldc.artifact('SPEC',root=old['root'],project_dir=str(tmp_path),source_commit=source.PARENT_COMMIT,
        counts=old['counts'],recipe=oldc.recipe(),anchor=old['anchor'],references={},
        reuse=dict(old['reuse'],u100={},historical_source={str(inventory):file_ref(inventory)['sha256']}))
    path=Path(old['root'])/'campaign_spec.json';path.write_text(json.dumps(parent))
    report=oldc.artifact('TRAINING_REPORT',selected_pass=17)
    rp=tmp_path/'report.json';write_immutable_json(rp,report)
    receipt=oldc.artifact('TASK',parents=dict(spec=parent['content_hash']),job_id='8761518',
        task_id='train_subclass',source_commit=source.PARENT_COMMIT,report_hash=report['content_hash'])
    cp=tmp_path/'receipt.json';write_immutable_json(cp,receipt)
    proof=c.artifact('REUSE',parents=dict(spec=parent['content_hash']),parent_spec=file_ref(path),
        historical_source=parent['reuse']['historical_source'],reference=dict(report=file_ref(rp),receipt=file_ref(cp)),
        banks=parent['reuse']['banks'],u100={},job='8761518',report_hash=report['content_hash'],
        subclasses=spec['reuse']['subclasses'])
    monkeypatch.setattr(source,'COUNTS',old['counts'])
    return parent,proof,inventory


@pytest.mark.parametrize('bad',['none','runtime','job','report','targets'])
def test_completed_reference_binding_and_payload_drift(tmp_path,monkeypatch,bad):
    from hlt_classification.luka_fullstat_subclass_c06 import source
    parent,proof,inventory=source_fixture(tmp_path,monkeypatch)
    if bad=='runtime':inventory.write_bytes(b'changed')
    elif bad=='job':proof['job']='8761519'
    elif bad=='report':proof['report_hash']='f'*64
    elif bad=='targets':Path(proof['subclasses']['files'][1]['path']).write_bytes(b'changed')
    proof=with_content_hash(proof)
    if bad=='none':assert source.check(proof)==parent
    else:
        with pytest.raises(ValueError):source.check(proof)


@pytest.mark.parametrize('bad',['none','job','failed_replay'])
def test_authenticate_uses_original_source_and_exact_completed_job(tmp_path,monkeypatch,bad):
    import json
    from types import SimpleNamespace
    from hlt_classification.luka_fullstat_subclass_c06 import source
    parent,proof,inventory=source_fixture(tmp_path,monkeypatch)
    def git(project,*args):
        if args==('rev-parse','HEAD'):return source.PARENT_COMMIT
        if args[0]=='ls-files':return inventory.name
        return ''
    monkeypatch.setattr(source,'git',git)
    evidence={k:proof[k] for k in ('reference','banks','job','report_hash','u100','subclasses')}
    if bad=='job':evidence['job']='8761519'
    def replay(args,**kw):
        assert args[-2]==source.REPLAY and args[-1]==proof['parent_spec']['path']
        assert kw['env']['PYTHONPATH']=='' and kw['cwd']==Path(parent['project_dir'])
        assert 'from hlt_classification.luka_fullstat_subclass import campaign,worker,targets' in args[-2]
        return SimpleNamespace(returncode=1 if bad=='failed_replay' else 0,stderr='fixture replay failed',
            stdout='SUBCLASS_C06_REUSE='+json.dumps(evidence))
    monkeypatch.setattr(source.subprocess,'run',replay)
    if bad=='none':
        result=source.authenticate(proof['parent_spec']['path'],tmp_path)
        assert result==proof and source.check(result)==parent
    else:
        with pytest.raises(ValueError):source.authenticate(proof['parent_spec']['path'],tmp_path)
