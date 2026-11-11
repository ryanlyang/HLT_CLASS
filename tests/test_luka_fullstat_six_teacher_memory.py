"""Regression for inherited worker peaks and recovery of finished native exports."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.luka_fullstat_six_teacher import (
    sampling, recovery, export_recovery as er, kd_banks, source, contracts as c)


def status(pid=101, peak=102400, rss=96000):
    return f'Name:\tpython\nPid:\t{pid}\nVmHWM:\t{peak} kB\nVmRSS:\t{rss} kB\n'


@pytest.mark.parametrize('bad', ['missing', 'pid', 'unit', 'negative', 'bounds', 'duplicate'])
def test_worker_status_fails_closed(bad):
    text = status()
    if bad == 'missing':text = 'Pid:\t101\n'
    if bad == 'pid':text = status(102)
    if bad == 'unit':text = text.replace('kB', 'bytes')
    if bad == 'negative':text = status(peak=-1)
    if bad == 'bounds':text = status(peak=1)
    if bad == 'duplicate':text += 'VmHWM:\t102400 kB\n'
    with pytest.raises(ValueError):sampling.parse_worker_status(text, 101)


def test_worker_peak_reads_linux_post_exec_address_space(monkeypatch):
    monkeypatch.setattr(sampling.sys, 'platform', 'linux')
    reads = []
    def read(path):
        reads.append(str(path).replace('\\', '/'))
        return status()
    monkeypatch.setattr(Path, 'read_text', read)
    assert sampling.worker_peak(101) == 102400*1024
    assert reads == ['/proc/101/status']


def test_inherited_45gib_peaks_do_not_false_reject_or_hide_real_oversize(monkeypatch):
    from hlt_classification.luka_fullstat_multiview.resources import check_memory
    cpu = sampling.CPUWorkers(SimpleNamespace())
    parent = 48393801728
    actual = {101:256*2**20, 102:260*2**20, 103:270*2**20}
    monkeypatch.setattr(sampling, 'worker_peak', lambda pid:actual[pid])
    def collect():
        jobs=[]
        for pid in actual:
            future=Future();future.set_result((str(pid),pid,parent));jobs.append(future)
        return cpu.collect(jobs)
    assert collect() == ['101', '102', '103']
    assert cpu.memory() == sum(actual.values())
    assert sum(cpu.lifetime_peaks.values()) == 3*parent
    spec=dict(memory_mb=192000, reuse=dict(u100=dict(gpu=dict(total_memory_bytes=48*2**30))))
    check_memory(spec, dict(peak_rss_bytes=parent+cpu.memory(), gpu_peak_bytes=10*2**30))
    with pytest.raises(MemoryError):
        check_memory(spec, dict(peak_rss_bytes=4*parent, gpu_peak_bytes=10*2**30))
    actual[101]=140*2**30;collect()
    with pytest.raises(MemoryError):
        check_memory(spec, dict(peak_rss_bytes=parent+cpu.memory(), gpu_peak_bytes=10*2**30))


def test_recovery_source_guard_rejects_loss_or_sampler_changes(monkeypatch):
    changed=[er.PREFIX+'sampling.py',er.PREFIX+'export_recovery.py']
    monkeypatch.setattr(source,'git',lambda p,*a:'\n'.join(changed) if a[0]=='diff' else '')
    er.source_delta('/project','f'*40)
    for forbidden in (er.PREFIX+'model.py',er.PREFIX+'training.py',
                      'src/hlt_classification/luka_fullstat_multi3_contrastive/sampling.py'):
        changed.append(forbidden)
        with pytest.raises(ValueError,match='reviewed memory'):er.source_delta('/project','f'*40)
        changed.pop()


def failure_fixture(tmp_path,monkeypatch):
    from test_luka_fullstat_six_teacher_targets import frozen,add_ladder
    spec,tr,_,_=frozen(tmp_path)
    add_ladder(spec,tr)
    # Export fixtures were generated under this test commit; production remains
    # pinned to 15808976 and the full3M/500k metadata.
    monkeypatch.setattr(er,'ORIGINAL_COMMIT',spec['source_commit'])
    monkeypatch.setattr(er,'COUNTS',spec['counts'])
    spec.update(project_dir=str(tmp_path/'original'),recipe=c.recipe(),
                source_snapshot={'fixture':True},nodes={c.TASK:{'seed':33}})
    spec=c.artifact('SPEC',**{k:v for k,v in spec.items() if k not in
        ('contract','schema_version','parents','content_hash',*c.previous.FLAGS)})
    path=Path(spec['root'])/'campaign_spec.json';write_immutable_json(path,spec)
    folder=Path(spec['root'])/c.TASK;folder.mkdir()
    write_immutable_json(folder/'startup_failure.json',c.artifact('FAILURE',error_type='MemoryError',
        message='Multiview peak_rss_bytes: measured 180.28 GiB; limit 151.38 GiB.'))
    write_immutable_json(folder/'startup_measurements.json',c.artifact('STARTUP',parents=dict(spec=spec['content_hash']),
        **{k:True for k in ('installed_weaver_parity','exact_sampler_parity','connected_gradients',
                          'classifier_ce_only','native_reload_parity','probe_discarded')}))
    original=c.artifact('RESUME',parents=dict(spec=spec['content_hash'],recipe=c.recipe()['content_hash']),
        spec=file_ref(path),accounting=[er.ORIGINAL_JOB,'lukafs6_train_six','fixture','default','FAILED','1:0'],
        epochs=[],exports=recovery.export_records(spec))
    monkeypatch.setattr(er,'source_delta',lambda *a:None)
    monkeypatch.setattr(er,'replay',lambda *a:original)
    return spec,tr,path,original


def test_changed_source_export_recovery_keeps_all_bytes_and_exporter(tmp_path,monkeypatch):
    from hlt_classification import provenance
    spec,tr,path,original=failure_fixture(tmp_path,monkeypatch)
    reviewed=recovery.inspect(path,tmp_path/'newproject','e'*40)
    assert reviewed['contract']=='LUKA_FULLSTAT_SIX_TEACHER_EXPORT_RECOVERY/v1'
    new=with_content_hash(dict(spec,root=str(tmp_path/'new'),source_commit='e'*40,
        project_dir=str(tmp_path/'newproject'),resume=reviewed,worker_memory_policy=sampling.memory_policy()))
    Path(new['root']).mkdir()
    monkeypatch.setattr(source,'git',lambda p,*a:er.ORIGINAL_COMMIT if a[0]=='rev-parse' else '')
    monkeypatch.setattr(provenance,'capture_source_snapshot',lambda *a,**k:spec['source_snapshot'])
    monkeypatch.setattr(kd_banks.views,'load',lambda *a:pytest.fail('Must reuse exports'))
    recovery.check(new);kd_banks.prepare(new,'cpu',0)
    for t in kd_banks.NAMES:
        assert kd_banks.manifest(new,t)['parents']['source']==spec['source_commit']
        for name in kd_banks.FILES:
            assert (kd_banks.folder(new,t)/name).read_bytes()==(kd_banks.folder(spec,t)/name).read_bytes()
        kd_banks.arrays(new,t,tr)
    assert recovery.import_epochs(new) is None
    # A later same-source continuation must retain the original exporter too.
    newpath=Path(new['root'])/'campaign_spec.json';write_immutable_json(newpath,new)
    again=dict(new,resume=dict(spec=file_ref(newpath),exports=reviewed['exports']))
    assert kd_banks.parents(again,'CONCAT')==kd_banks.parents(spec,'CONCAT')
    (kd_banks.folder(spec,'FUSION')/'probabilities.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):recovery.check(new)


@pytest.mark.parametrize('bad',['epoch','startup_accepted','different_failure','job','partial','recipe'])
def test_export_only_recovery_rejects_wrong_failure_or_training(tmp_path,monkeypatch,bad):
    spec,tr,path,original=failure_fixture(tmp_path,monkeypatch)
    folder=Path(spec['root'])/c.TASK
    if bad=='epoch':(folder/'epochs').mkdir()
    if bad=='startup_accepted':(folder/'startup.json').write_text('{}')
    if bad=='different_failure':
        failure=load_json(folder/'startup_failure.json');failure['error_type']='RuntimeError'
        (folder/'startup_failure.json').write_text(json.dumps(with_content_hash(failure)))
    if bad=='job':original['accounting'][0]='9999999'
    if bad=='partial':original['exports'].pop('FUSION')
    if bad=='recipe':
        spec['recipe']['ce_weight']=2
        path.write_text(json.dumps(with_content_hash(spec)))
    with pytest.raises(ValueError):recovery.inspect(path,tmp_path/'newproject','e'*40)


def test_export_replay_uses_original_source_process(tmp_path,monkeypatch):
    old=dict(project_dir=str(tmp_path/'original'))
    monkeypatch.setattr(source,'git',lambda *a:er.ORIGINAL_COMMIT)
    expected={'original':'receipt'};calls=[]
    def run(args,**kw):
        calls.append((args,kw))
        return SimpleNamespace(returncode=0,stdout='SIX_EXPORT_RECOVERY='+json.dumps(expected),stderr='')
    monkeypatch.setattr(er.subprocess,'run',run)
    assert er.replay(tmp_path/'campaign_spec.json',old)==expected
    args,kw=calls[0]
    assert args[0]==sys.executable and args[-2]==er.REPLAY
    assert kw['cwd']==Path(old['project_dir']) and kw['env']['PYTHONPATH']==''
    monkeypatch.setattr(er.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='corrupt'))
    with pytest.raises(ValueError,match='authentication failed'):er.replay(tmp_path/'spec',old)
