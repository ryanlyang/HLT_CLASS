"""Portable teacher closure, unchanged tail workers, transfer and exact L40 jobs."""
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tarfile
import numpy as np
import pytest
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash, sha256_file
from hlt_classification.luka_sixclass_tail_hydra import contracts as c, campaign, bundle, artifacts, worker, inspect
from hlt_classification.luka_sixclass_balanced_oscar import contracts as s, banks, training, startup
import test_luka_sixclass_balanced_oscar as old
from test_luka_fullstat_balanced_ladder import threads

ROOT=Path(__file__).resolve().parents[1]
module_spec=importlib.util.spec_from_file_location('tail_transfer',ROOT/'scripts/transfer_luka_sixclass_tail.py')
transfer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(transfer)


def spec_fixture(root):
    source=old.spec_fixture(root)
    body={k:v for k,v in source.items() if k not in ('contract','content_hash','schema_version','parents',*s.FLAGS)}
    body.update(gpu='l40',nodes={t:c.node(t) for t in c.STAGES},tasks=c.tasks(),execution_site=c.SITE,
        minutes={t:2880 for t in c.tasks()},import_root=str(root.parent/'import'),bundle_sha256='b'*64,
        imported_teachers={t:t for t in c.IMPORTED},population_root=str(root.parent/'import/population'))
    return c.artifact('SPEC',parents=dict(oscar_spec='a'*64,bundle='b'*64),**body)


def test_recipe_and_two_job_dag_are_unchanged_science():
    assert c.recipe()==s.recipe()
    assert c.STAGES==('DE050','DE000') and c.IMPORTED==s.STAGES[:4]
    assert c.tasks()=={'train_DE050':[],'train_DE000':['train_DE050']}
    for stage in c.STAGES:
        assert c.node(stage)==s.node(stage)
        assert c.main_policy(stage)==dict(teacher='UE100' if stage=='DE050' else 'DE050',ce_weight=.25,
            kd_weight=.75,temperature=2.,reduction='T2_forward_KL_class_sum_batch_mean')
    assert worker.training is old.training and worker.views is old.views and worker.startup is old.startup
    assert worker.banks is old.banks and worker.contrastive is old.contrastive
    assert c.COUNTS==dict(train=1658937,validation=328589,final_test=822313)


def test_l40_queue_is_only_two_fresh_fits_and_journaled(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(campaign,'storage',lambda *a,**k:{})
    monkeypatch.setattr(campaign,'site',lambda *a:None)
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(215200+len(calls)),stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    plan=campaign.queue(spec);assert not calls and plan['scientific_fits']==2 and plan['preflight_jobs']==0
    jobs=campaign.queue(spec,execute=True,authorization=c.AUTH)
    assert jobs=={'train_DE050':'215201','train_DE000':'215202'}
    assert len(campaign.accepted_jobs(spec))==2
    for row in calls:
        assert all(flag in row for flag in ('--partition=gpus','--account=default','--qos=normal',
            '--gres=gpu:nvidia_l40:1','--cpus-per-task=6','--mem=192000M','--time=2880','--export=NONE'))
        assert not any('919821' in x or '180224' in x for x in row)
    assert '--dependency=afterok:215201' in calls[1]
    assert not any(x.startswith('--dependency=') for x in calls[0])
    with pytest.raises(FileExistsError):campaign.queue(spec,execute=True,authorization=c.AUTH)
    assert len(calls)==2


def test_unknown_submission_and_live_jobs_never_requeued(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path);calls=[]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None);monkeypatch.setattr(campaign,'storage',lambda *a,**k:{})
    monkeypatch.setattr(campaign,'site',lambda *a:None)
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout='lost response',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Uncertain'):campaign.queue(spec,execute=True,authorization=c.AUTH)
    with pytest.raises(ValueError,match='Ambiguous'):campaign.queue(spec,execute=True,authorization=c.AUTH,resume=True)
    assert len(calls)==1
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a:[dict(task='train_DE050',job='215201')])
    def live(job):raise ValueError('still RUNNING')
    monkeypatch.setattr(campaign,'terminal',live)
    with pytest.raises(ValueError,match='RUNNING'):campaign.remaining(spec,True)


def test_resume_skips_completed_de050_and_does_not_need_cross_cluster_dependencies(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path)
    jobs=[dict(task='train_DE050',job='215201'),dict(task='train_DE000',job='215202')]
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**k:jobs)
    monkeypatch.setattr(campaign,'terminal',lambda job:[job,'COMPLETED' if job=='215201' else 'FAILED','0:0' if job=='215201' else '1:0'])
    path=tmp_path/'train_DE050/result.json';path.parent.mkdir();path.write_text('{}')
    monkeypatch.setattr(artifacts,'completed',lambda *a,**k:dict(receipt=dict(job_id='215201',content_hash='d'*64)))
    tasks,complete=campaign.remaining(spec,True)
    assert tasks==['train_DE000'] and complete=={'DE050':'d'*64}
    row=campaign.commands(spec,tasks)[0]
    assert row['dependencies']==[] and not any(x.startswith('--dependency=') for x in row['command'])


def test_registration_checks_runtime_import_population_and_protects_recipe(tmp_path,monkeypatch):
    dataset=tmp_path/'dataset';dataset.mkdir();project=tmp_path/'code';project.mkdir()
    imported_root=tmp_path/'imports/bundle';imported_root.mkdir(parents=True)
    parent=old.spec_fixture(tmp_path/'oscar');population=parent['population'];seen=[]
    m=dict(foundation_sha256=c.PARENT_HASH);write_immutable_json(dataset/'transfer_manifest.json',m)
    transfer_hash=campaign.canonical_sha256(m)
    write_immutable_json(dataset/'transfer_complete.json',dict(manifest_sha256=transfer_hash,all_files_verified=True))
    env=dict(versions={'torch':'2.5.1+cu121','weaver-core':'0.5.3'},weaver_python_sources={'ParticleTransformer.py':'a'*64})
    parent['environment']=dict(env,versions=dict(env['versions'],torch='2.5.1+cu118'))
    inherited=dict(donor_sources={},teachers={t:'f'*64 for t in c.IMPORTED})
    class Imported:
        spec=parent;manifest=inherited
        def __init__(self,root,expected,**kw):
            assert root==imported_root or str(root)==str(imported_root)
            assert expected=='b'*64
            seen.append(('import',kw))
        def authenticate_population(self,dataset):
            return population,None,None,None,dict(train_identities=np.zeros((1,32),np.uint8),train_labels=np.zeros(1,np.int64)),None
        def bank(self,stage,rows):seen.append(('bank',stage))
    monkeypatch.setattr(bundle,'Import',Imported);monkeypatch.setattr(campaign,'locations',lambda *a:None)
    monkeypatch.setattr(campaign,'source',lambda *a:{})
    monkeypatch.setattr(campaign,'runtime',lambda:env)
    monkeypatch.setattr(campaign,'donor_sources',lambda *a:{})
    monkeypatch.setattr(campaign.lineage,'verify',lambda *a:{})
    monkeypatch.setattr(c,'BASE',str(tmp_path));monkeypatch.setattr(c,'TRANSFER_HASH',transfer_hash)
    monkeypatch.setattr(c,'POPULATION_HASH',population['content_hash'])
    spec=campaign.create(dataset,tmp_path/'hydra',project,'a'*40,__import__('sys').prefix,
        import_root=imported_root,bundle_hash='b'*64,available_gib=20)
    campaign.check(spec,live=False)
    assert ('import',dict(verify_payloads=True)) in seen
    assert [r[1] for r in seen if r[0]=='bank']==list(c.IMPORTED)
    assert spec['population']==population and spec['recipe']==s.recipe() and spec['counts']==c.COUNTS
    for field,value in [('gpu','l40s'),('memory_mb',64000),('tasks',s.tasks()),('counts',dict(train=3000000)),
                        ('bundle_sha256','c'*64),('imported_teachers',{}),('recipe',{})]:
        bad=deepcopy(spec);bad[field]=value;bad=with_content_hash(bad)
        with pytest.raises((ValueError,AssertionError)):campaign.check(bad,live=False)
    with pytest.raises(FileExistsError):campaign.create(dataset,tmp_path/'hydra',project,'a'*40,__import__('sys').prefix,
        import_root=imported_root,bundle_hash='b'*64,available_gib=20)
    env['versions']['torch']='2.6.0'
    with pytest.raises(ValueError,match='Incompatible'):campaign.create(dataset,tmp_path/'bad',project,'a'*40,__import__('sys').prefix,
        import_root=imported_root,bundle_hash='b'*64,available_gib=20)


def test_hydra_dry_review_and_allocation_reject_other_site_gpu(tmp_path,monkeypatch):
    spec=spec_fixture(tmp_path);seen=[];cluster='hydra'
    def run(args,**kw):
        seen.append(args)
        return SimpleNamespace(returncode=0,stdout='ClusterName = '+cluster if args[-1]=='config' else 'Job 1',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    campaign.site(campaign.commands(spec,list(c.tasks())))
    assert len(seen)==3 and all('--test-only' in x and '--wrap=true' in x for x in seen[1:])
    cluster='slurmctld'
    with pytest.raises(PermissionError):campaign.site(campaign.commands(spec,list(c.tasks())))
    spec['environment_dir']=__import__('sys').prefix
    monkeypatch.setenv('SLURM_JOB_ID','215201');monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setenv('LD_LIBRARY_PATH',str(Path(spec['environment_dir'])/'lib'))
    monkeypatch.setattr(campaign,'binding',lambda *a:None);monkeypatch.setattr(campaign,'runtime',lambda:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40')
    fields=(f'JobId=215201 JobState=RUNNING Account=default QOS=normal Partition=gpus JobName=luka6tail_train_DE050 '
        f'Comment={spec["content_hash"]}:train_DE050 WorkDir=/project NumCPUs=6 NumNodes=1 NumTasks=1 CPUs/Task=6 '
        'AllocTRES=cpu=6,mem=187.5G,node=1,gres/gpu=1 TimeLimit=2-00:00:00')
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:SimpleNamespace(stdout='ClusterName = hydra' if args[-1]=='config' else fields))
    assert worker.allocation(spec,'train_DE050')=='215201'
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'NVIDIA L40S')
    with pytest.raises(PermissionError):worker.allocation(spec,'train_DE050')


def protocol_fixture(root):
    values={'campaign_spec.json':b'{}','population/population.json':b'{}','train_OFFLINE/x.npy':bytes(range(256))*100}
    files=[]
    for name,data in sorted(values.items()):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        files.append(dict(path=name,bytes=len(data),sha256=sha256_file(path)))
    m=c.artifact('BUNDLE',source_root='/oscar/home/rlyang/hlt_classification/checkpoints/example',files=files,bytes=sum(x['bytes'] for x in files))
    return m


def stream_bytes(root,m):
    out=io.BytesIO();transfer.stream(root,m,out);return out.getvalue()


def test_binary_transport_retry_preserves_verified_bytes(tmp_path):
    src=tmp_path/'source';m=protocol_fixture(src);raw=stream_bytes(src,m);dest=tmp_path/'dest'
    transfer.receive(dest,m['content_hash'],io.BytesIO(raw))
    original=(dest/'train_OFFLINE/x.npy').stat().st_mtime_ns
    transfer.receive(dest,m['content_hash'],io.BytesIO(raw))
    assert (dest/'train_OFFLINE/x.npy').stat().st_mtime_ns==original
    assert load_json(dest/'transfer_complete.json')['all_files_verified'] is True
    (dest/'train_OFFLINE/x.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Existing destination'):transfer.receive(dest,m['content_hash'],io.BytesIO(raw))
    assert (dest/'train_OFFLINE/x.npy').read_bytes()==b'corrupt'


def test_standalone_remote_stream_limits_source_paths(tmp_path,monkeypatch):
    import sys
    original=protocol_fixture(tmp_path/'source');path=tmp_path/'manifest.json';calls=[]
    monkeypatch.setattr(transfer,'stream',lambda root,m,out:calls.append(root))
    monkeypatch.setattr(sys,'stdout',SimpleNamespace(buffer=io.BytesIO()))
    for root in ('/etc','/oscar/home/rlyang/hlt_classification/checkpoints/../private','relative'):
        m=with_content_hash(dict(original,source_root=root));path.write_text(json.dumps(m))
        monkeypatch.setattr(sys,'argv',['transfer','stream-source','--manifest',str(path),'--sha256',m['content_hash']])
        with pytest.raises(ValueError):transfer.main()
    path.write_text(json.dumps(original))
    monkeypatch.setattr(sys,'argv',['transfer','stream-source','--manifest',str(path),'--sha256',original['content_hash']])
    transfer.main();assert calls==[original['source_root']]


@pytest.mark.parametrize('name',['../escape','/absolute','a/../x','a//x','a\\b','C:/x','train_DE050/selected.pt','train_OFFLINE/epochs/state.pt'])
def test_transfer_rejects_unsafe_or_unauthorized_paths(tmp_path,name):
    m=protocol_fixture(tmp_path/'source');m['files'][0]['path']=name;m['files'].sort(key=lambda r:r['path']);m=with_content_hash(m)
    with pytest.raises(ValueError):transfer.validate(m,m['content_hash'])


@pytest.mark.parametrize('kind',['symlink','hardlink','extra','truncated','wrong_hash'])
def test_transfer_rejects_invalid_archive_without_completion(tmp_path,kind):
    src=tmp_path/'source';m=protocol_fixture(src);out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as tar:
        data=transfer.canonical(m);info=tarfile.TarInfo('bundle.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        for i,r in enumerate(m['files']):
            info=tarfile.TarInfo(r['path']);data=(src/r['path']).read_bytes();info.size=len(data)
            if i==0 and kind in ('symlink','hardlink'):
                info.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE;info.linkname='../escape';info.size=0;tar.addfile(info);continue
            if kind=='wrong_hash' and i==2:data=b'x'*len(data)
            tar.addfile(info,io.BytesIO(data))
        if kind=='extra':tar.addfile(tarfile.TarInfo('extra'))
    data=out.getvalue()
    if kind=='truncated':data=data[:4000]
    with pytest.raises((ValueError,tarfile.TarError,OSError)):
        transfer.receive(tmp_path/'dest',m['content_hash'],io.BytesIO(data))
    assert not (tmp_path/'dest/transfer_complete.json').exists()


def test_frozen_donors_match_committed_scientific_sources():
    pins=load_json(ROOT/'src/hlt_classification/luka_sixclass_tail_hydra/donors.json')
    import hashlib
    for name,digest in pins['files'].items():
        blob=subprocess.check_output(['git','show',pins['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==digest,name
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==blob,name


def make_import(tmp_path,monkeypatch):
    parent=tmp_path/'oscar';spec,tr,val,sampler,accepted=old.worker_fixture(parent,monkeypatch)
    monkeypatch.setattr(c,'JOBS',{t:str(100+i) for i,t in enumerate(c.IMPORTED)})
    monkeypatch.setattr(c,'COUNTS',s.COUNTS)
    for stage in c.IMPORTED:old.worker.run(spec,'train_'+stage,device='cpu')
    # Durable original journals, copied byte-for-byte; optimizer states stay behind.
    for row in accepted:
        fields=dict(parents=dict(spec=spec['content_hash']),task=row['task'],plan='f'*64,command=['sbatch',row['task']])
        write_immutable_json(parent/'submission/attempt_001'/(row['task']+'.intent.json'),s.artifact('SUBMISSION_INTENT',**fields))
        write_immutable_json(parent/'submission/attempt_001'/(row['task']+'.submitted.json'),s.artifact('SUBMITTED',job=row['job'],**fields))
    write_immutable_json(parent/'population/population.json',spec['population'])
    arrays=dict(train_identities=tr.identities,train_labels=tr.labels,
                validation_pt=np.full(len(val),650.),outside_validation_pt=np.full(len(val),2800.))
    monkeypatch.setattr(old.population,'load',lambda *a,**k:(spec['population'],None,None,None,arrays,sampler))
    project=tmp_path/'project';project.mkdir()
    monkeypatch.setattr(bundle,'capture_source_snapshot',lambda p,**k:dict(git_commit='a'*40) if Path(p)==project else spec['source'])
    monkeypatch.setattr(campaign,'source',lambda *a:dict(git_commit='a'*40))
    monkeypatch.setattr(campaign,'donor_sources',lambda *a:{})
    monkeypatch.setattr(bundle,'oscar_root',lambda *a:None)
    monkeypatch.setattr(bundle.original,'runtime',lambda:spec['environment'])
    monkeypatch.setattr(bundle.original,'terminal',lambda j:[j,'COMPLETED','0:0'])
    m=bundle.export(spec,project,tmp_path/'manifest.json')
    assert len([x for x in m['files'] if '/epochs/' in x['path']])==80
    assert all(not ('/epochs/' in x['path'] and x['path'].endswith('.pt')) for x in m['files'])
    dest=tmp_path/'import';transfer.receive(dest,m['content_hash'],io.BytesIO(stream_bytes(parent,m)))
    # Windows fixture refs use native drive paths. Production refs are POSIX.
    if os.name=='nt':monkeypatch.setattr(bundle,'PurePosixPath',Path)
    imported=bundle.Import(dest,m['content_hash'],verify_payloads=True)
    for row in m['files']:assert (dest/row['path']).read_bytes()==(parent/row['path']).read_bytes()
    parent.rename(tmp_path/'oscar_unavailable')
    return imported,tr,val,sampler,project


def test_import_then_both_workers_and_export_recovery_without_oscar_paths(tmp_path,monkeypatch,capsys):
    imported,tr,val,sampler,project=make_import(tmp_path,monkeypatch)
    assert not Path(imported.spec['root']).exists()
    for stage in c.IMPORTED:
        target=imported.bank(stage,tr)
        assert target.targets.shape==(120,128) and target.probabilities.shape==(120,6)
        bad=deepcopy(tr);bad.identities=bad.identities[::-1]
        with pytest.raises(ValueError,match='rows'):imported.bank(stage,bad)
        with pytest.raises(PermissionError):imported.bank(stage,val)
    spec=spec_fixture(tmp_path/'hydra');spec.update(import_root=str(imported.root),bundle_sha256=imported.manifest['content_hash'],
        population=imported.spec['population'],population_root=str(imported.root/'population'),
        source=dict(git_commit='a'*40),counts=s.COUNTS)
    spec=with_content_hash(spec);jobs=[dict(task='train_DE050',job='215201'),dict(task='train_DE000',job='215202')]
    monkeypatch.setattr(campaign,'check',lambda *a,**k:None)
    monkeypatch.setattr(campaign,'accepted_jobs',lambda *a,**k:jobs)
    monkeypatch.setattr(campaign,'storage',lambda *a,**k:{})
    monkeypatch.setattr(worker,'allocation',lambda spec,task:next(r['job'] for r in reversed(jobs) if r['task']==task))
    monkeypatch.setattr(worker,'memory_snapshot',lambda _:dict(peak_rss_bytes=1024,rss_bytes=1024,gpu_peak_bytes=0))
    original=banks.export
    def fail(*a,**k):raise RuntimeError('export interrupted')
    monkeypatch.setattr(banks,'export',fail)
    with pytest.raises(RuntimeError,match='export interrupted'):worker.run(spec,'train_DE050',device='cpu')
    assert not (Path(spec['root'])/'train_DE050/result.json').exists()
    jobs.append(dict(task='train_DE050',job='215203'));monkeypatch.setattr(banks,'export',original)
    saved=artifacts.save_epoch
    monkeypatch.setattr(artifacts,'save_epoch',lambda *a,**k:pytest.fail('No repeated training after 20 committed epochs'))
    worker.run(spec,'train_DE050',device='cpu')
    monkeypatch.setattr(artifacts,'save_epoch',saved)
    worker.run(spec,'train_DE000',device='cpu')
    result=inspect.results(spec)
    assert set(result['models'])==set(c.ALL_STAGES)
    for stage in c.STAGES:
        r=artifacts.completed(spec,stage);assert r['report']['passes']==20
        assert r['report']['node']==s.node(stage)
        assert r['receipt']['fit_parents']['teacher_OFFLINE']==imported.manifest['teachers']['OFFLINE']
        state=torch.load(r['receipt']['files']['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert not any(x.startswith(('predictors.','logit_heads.','matching.')) for x in state)
    tail=artifacts.completed(spec,'DE000')
    assert tail['receipt']['fit_parents']['teacher_DE050']==artifacts.completed(spec,'DE050')['receipt']['content_hash']
    assert 'same' not in result or result['final_test_accessed'] is False
    inspect.progress(spec,'DE000');assert '20/20 epochs' in capsys.readouterr().out
    # Checksum tampering cannot be hidden behind the transfer completion marker.
    path=imported.root/'train_OFFLINE/result.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='size'):bundle.Import(imported.root,imported.manifest['content_hash'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual local CUDA acceptance')
@pytest.mark.parametrize('stage',c.STAGES)
def test_real_installed_weaver_acceptance_for_hydra_tail(stage,tmp_path,monkeypatch):
    # Identical startup function is used by the new Hydra worker. Original test
    # exercises real production architecture, gradients, miniature, AMP, export.
    old.test_real_weaver_integrated_qualification(stage,tmp_path,monkeypatch)


def test_detached_bash_continues_after_tail_exits(tmp_path):
    import sys,time
    bash='C:/Program Files/Git/bin/bash.exe' if sys.platform=='win32' else 'bash'
    posix=subprocess.check_output([bash,'-c','cd "$1" && pwd','fixture',str(tmp_path)],text=True).strip()
    def sp(p):return posix+'/'+Path(p).relative_to(tmp_path).as_posix()
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True);(project/'sbatch').mkdir()
    script=project/'scripts/launch_luka_sixclass_tail_hydra.sh';script.write_bytes((ROOT/'scripts'/script.name).read_bytes())
    (project/'sbatch/luka_sixclass_tail_hydra_common.sh').write_bytes(b':\n')
    env=tmp_path/'runtime';(env/'bin').mkdir(parents=True);bindir=tmp_path/'bin';bindir.mkdir()
    imported=tmp_path/'import';imported.mkdir();(imported/'transfer_complete.json').write_text('{}')
    calls=tmp_path/'calls';root=tmp_path/'runs/tail'
    for p,body in {env/'bin/python':'sleep 0.2\nprintf "%s\\n" "$*" >> "$FIXTURE_CALLS"\n',
                   bindir/'git':'printf "%040d\\n" 1\n',bindir/'tail':'echo TAIL_EXITED\n'}.items():
        p.write_bytes(('#!/usr/bin/env bash\n'+body).encode());p.chmod(0o755)
    envfile=tmp_path/'env.sh';envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    environment=dict(os.environ,BASH_ENV=sp(envfile),FIXTURE_BIN=sp(bindir),FIXTURE_CALLS=sp(calls))
    args=[bash,str(script),'detach',sp(project),sp(root),'0'*39+'1','20',sp(env),sp(imported),'b'*64,'--authorize',c.AUTH]
    result=subprocess.run(args,env=environment,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'DETACHED Hydra tail' in result.stdout and 'TAIL_EXITED' in result.stdout
    log=Path(str(root)+'.launch')/'launcher.log';deadline=time.monotonic()+12
    while (not log.exists() or 'LAUNCHER_EXIT=0' not in log.read_text()) and time.monotonic()<deadline:time.sleep(.05)
    assert 'LAUNCHER_EXIT=0' in log.read_text()
    rows=calls.read_text().splitlines();assert len(rows)==2 and ' register ' in rows[0] and ' queue ' in rows[1]
    assert '--import-root' in rows[0] and '--bundle-hash' in rows[0] and '--execute' in rows[1]
    duplicate=subprocess.run(args,env=environment,capture_output=True,text=True,timeout=10)
    assert duplicate.returncode!=0
