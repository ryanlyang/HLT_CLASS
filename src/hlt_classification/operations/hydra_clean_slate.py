"""Keep the new dataset/runtime/Git/campaign; retire previous workspace artifacts.

Standalone standard-library deletion helper with a bounded runtime import probe.
Old payloads are never deserialized or hashed; compact text evidence is archived
and authenticated before deletion.
"""
import argparse
import getpass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

BASE=Path('/cs/home/rlyang/hlt_u100_pairs')
AUTH='AUTHORIZE_HYDRA_NEW_DATASET_CLEAN_SLATE_V1'
OPERATION='new_dataset_clean_slate_v1'
KEEP_HASH='8d5077aa3d15e303756e03aa55476bee1c2abac2692ae83dac77613ed1ba6822'
KEEP_FILE='src/hlt_classification/luka_balanced_reconstruction_hydra/workspace_keep.json'
OLD_INPUTS={'bundle_65952ed3','cache_new_fullstat_offline_a865e5b6_r1',
    'contrast_delta_54473f10','cross_message_encoder_73abecfd','frozen_adapter_delta_4b9630fa',
    'imports','inputs_65952ed3','offline_teacher_6fe7e349','pretext_validation_54ff594d',
    'raw_next_0a6721ed'}
TEXT={'.json','.jsonl','.csv','.tsv','.txt','.log','.out','.md','.yaml','.yml','.sha256'}


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def sealed(kind,**fields):
    body=dict(contract='LUKA_HYDRA_CLEAN_SLATE_'+kind+'/v1',schema_version=1,
              parents=dict(fields.pop('parents',{})),final_test_inference=False,**fields)
    return dict(body,content_hash=hashlib.sha256(canonical(body)).hexdigest())


def run(args,*,check=True,timeout=90,extra_env=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
    env.update(extra_env or {})
    return subprocess.run(args,check=check,capture_output=True,text=True,env=env,timeout=timeout)


def idle():
    require(not run(['squeue','--me','--noheader','--format=%i']).stdout.strip(),
            'Queue is not empty; no new submissions during cleanup')


def safe(path,base):
    path=Path(path).absolute();base=Path(base).absolute()
    require(path==base or base in path.parents,'Path outside workspace: '+str(path))
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'Symlink path: '+str(path))
    require(path.resolve(strict=True)==path,'Noncanonical path: '+str(path))
    if hasattr(os,'getuid'):require(path.stat().st_uid==os.getuid(),'Foreign owner: '+str(path))
    return path


def signature(path):
    s=path.lstat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def digest_file(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):result.update(chunk)
    return result.hexdigest()


def publish(path,value):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def verified_keep(base,project,commit):
    base=safe(base,base);project=safe(project,base)
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'Exact pushed commit required')
    require(project.name=='code_balanced_reco_'+commit[:8] and project.parent==base,'Unexpected new checkout')
    require(run(['git','-C',str(project),'rev-parse','HEAD']).stdout.strip()==commit,'New checkout commit differs')
    require(not run(['git','-C',str(project),'status','--porcelain','--untracked-files=all']).stdout.strip(),
            'New source is dirty')
    run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'])
    common=Path(run(['git','-C',str(project),'rev-parse','--git-common-dir']).stdout.strip())
    if not common.is_absolute():common=project/common
    common=common.resolve(strict=True)
    require(common==base/'code_65952ed3/.git','Unexpected shared Git store; preserve it and inspect')
    data=(project/KEEP_FILE).read_bytes();value=json.loads(data)
    body={k:v for k,v in value.items() if k!='content_hash'}
    require(value.get('content_hash')==hashlib.sha256(canonical(body)).hexdigest()==KEEP_HASH,
            'New source keep contract differs')
    committed=run(['git','-C',str(project),'show','HEAD:'+KEEP_FILE]).stdout
    require(json.loads(committed)==value,'Keep contract differs from committed source')
    require(value['historical_artifacts_required'] is False and value['storage_policy'].endswith('/v2'),
            'Use the standalone storage-policy revision before retiring old ladder records')
    dataset=safe(base/value['dataset'],base);environment=safe(base/value['environment'],base)
    require((dataset/'transfer_complete.json').is_file() and (dataset/'foundation/producer.json').is_file(),
            'New dataset transfer/foundation metadata absent')
    cfg=dict(line.split(' = ',1) for line in (environment/'venv/pyvenv.cfg').read_text().splitlines() if ' = ' in line)
    require(Path(cfg['home']).resolve().is_relative_to(environment),'Runtime interpreter depends on another directory')
    require((environment/'venv/bin/python').resolve(strict=True).is_relative_to(environment),
            'Runtime Python points outside its retained environment')
    # Editable installations can depend on other workspace checkouts.
    allowed={'code_65952ed3',project.name,value['environment']}
    for pattern in ('venv/lib/python*/site-packages/*.pth','venv/lib/python*/site-packages/__editable__*_finder.py'):
        for path in environment.glob(pattern):
            text=path.read_text(errors='replace').replace('\\','/').replace('/home/rlyang/hlt_u100_pairs',base.as_posix())
            for found in re.findall(re.escape(base.as_posix())+r'/[^\s\x22\x27;,\)\]]+',text):
                name=Path(found).relative_to(base).parts[0]
                require(name in allowed,'Runtime metadata references another checkout: '+found)
    probe=('import json,sys; sys.path.insert(0,sys.argv[1]); '
           'from hlt_classification.luka_balanced_reconstruction_hydra import storage; '
           'import torch.nn.functional,uproot,scipy,threadpoolctl; '
           'from weaver.nn.model.ParticleTransformer import ParticleTransformer; '
           'paths=list(sys.path)+[getattr(m,"__file__",None) for m in list(sys.modules.values())]; '
           'print(json.dumps(dict(storage_policy=storage.policy(),paths=[p for p in paths if p])))')
    runtime=run([str(environment/'venv/bin/python'),'-I','-B','-c',probe,str(project/'src')],
        timeout=300,extra_env=dict(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH=str(environment/'venv/lib')))
    runtime_proof=json.loads(runtime.stdout.strip().splitlines()[-1]);policy=runtime_proof['storage_policy']
    require(policy['contract']==value['storage_policy'] and policy['schema_version']==2
            and policy['other_campaign_dependency'] is None and policy['protected_initial_allowance_bytes']==0,
            'New source still depends on retired campaign artifacts')
    for name in runtime_proof['paths']:
        location=Path(name.replace('/home/rlyang/hlt_u100_pairs',str(base))).resolve()
        if location.is_relative_to(base):
            require(location.relative_to(base).parts[0] in allowed,
                    'Imported runtime path depends on a directory to retire: '+str(location))
    return value,common


def code_candidate(base,path,common):
    other=run(['git','-C',str(path),'rev-parse','--git-common-dir'],check=False)
    if other.returncode:return None,'unrecognized source directory'
    location=Path(other.stdout.strip())
    if not location.is_absolute():location=path/location
    if location.resolve()!=common:return None,'separate Git store retained'
    if run(['git','-C',str(path),'status','--porcelain','--untracked-files=all']).stdout.strip():
        return None,'uncommitted work retained'
    head=run(['git','-C',str(path),'rev-parse','HEAD']).stdout.strip()
    if run(['git','-C',str(path),'merge-base','--is-ancestor',head,'origin/main'],check=False).returncode:
        return None,'unpublished/other-branch source retained'
    return dict(path=str(path),kind='worktree',commit=head,signature=signature(path)),None


def inventory(base,project,keep,common):
    candidates=[];retained=[str(project),str(common.parent),str(base/'datasets'),
        str(base/keep['environment']),str(base/'retired_campaigns')];skipped=[]
    for path in sorted(base.iterdir()):
        if not path.is_dir() or path.is_symlink():continue
        safe(path,base)
        if str(path) in retained or path.name=='runs':continue
        if path.name.startswith('code_'):
            item,reason=code_candidate(base,path,common)
            if item:candidates.append(item)
            else:skipped.append(dict(path=str(path),reason=reason));retained.append(str(path))
        elif path.name in OLD_INPUTS:
            candidates.append(dict(path=str(path),kind='old_inputs',signature=signature(path)))
        else:skipped.append(dict(path=str(path),reason='not in supplied old-input allowlist'));retained.append(str(path))
    for path in sorted((base/'runs').iterdir()):
        if not path.is_dir() or path.is_symlink():continue
        safe(path,base)
        if path.name.startswith(keep['run_prefix']):retained.append(str(path))
        else:candidates.append(dict(path=str(path),kind='old_run',signature=signature(path)))
    # Every removal is disjoint from every kept subtree, including Git storage.
    for row in candidates:
        target=Path(row['path'])
        for name in retained:
            protected=Path(name)
            require(not (target==protected or target in protected.parents or protected in target.parents),
                    'Removal overlaps a kept path: '+str(target))
    return candidates,retained,skipped


def archive_metadata(base,candidates,path):
    rows=[]
    def failed_walk(error):raise error
    with path.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for candidate in candidates:
                root=Path(candidate['path'])
                if candidate['kind']=='worktree':continue # Sources remain in retained pushed Git history.
                for folder,dirs,files in os.walk(root,followlinks=False,onerror=failed_walk):
                    dirs[:]=sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
                    for name in sorted(files):
                        source=Path(folder)/name
                        if source.is_symlink() or (source.suffix.lower() not in TEXT and name!='README'):continue
                        safe(source,base);before=signature(source)
                        if before[2]>32*2**20:continue # Compact evidence only; oversized logs are retired.
                        data=source.read_bytes();require(signature(source)==before,'Metadata changed: '+str(source))
                        relative=source.relative_to(base).as_posix()
                        info=tarfile.TarInfo(relative);info.size=len(data);info.mode=0o600;info.mtime=0
                        archive.addfile(info,io.BytesIO(data))
                        rows.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                                         signature=before))
    verify_archive(path,rows)
    return rows


def verify_archive(path,rows):
    """Read every retained text byte back; fail before deleting any old directory."""
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();by_name={m.name:m for m in members}
        require(len(members)==len(rows) and set(by_name)=={r['path'] for r in rows},
                'Archive member inventory differs')
        for row in rows:
            stream=archive.extractfile(by_name[row['path']]);require(stream is not None,'Archived metadata absent')
            require(hashlib.sha256(stream.read()).hexdigest()==row['sha256'],'Archived metadata differs')


def retire(base,project,commit,*,execute=False,authorization=None,helper_sha256=None):
    base=Path(base).absolute();project=Path(project).absolute();idle()
    keep,common=verified_keep(base,project,commit)
    operation=base/'retired_campaigns'/OPERATION
    require(not operation.exists(),'Existing clean-slate record; inspect before retrying')
    candidates,retained,skipped=inventory(base,project,keep,common)
    for name in retained:print('KEEP '+name,flush=True)
    for row in candidates:print('RETIRE '+row['path']+' ['+row['kind']+']',flush=True)
    for row in skipped:print('RETAIN '+row['path']+': '+row['reason'],flush=True)
    plan=sealed('INTENT',project=str(project),commit=commit,helper_sha256=helper_sha256,
        keep_contract=keep,candidates=candidates,retained=retained,skipped=skipped,
        deletion_scope='all_previous_campaigns_including_models_predictions_and_old_inputs')
    if not execute:return plan
    require(authorization==AUTH and helper_sha256 and len(helper_sha256)==64,'Explicit clean-slate authorization required')
    idle();operation.parent.mkdir(exist_ok=True);safe(operation.parent,base);operation.mkdir()
    publish(operation/'intent.json',plan)
    archive=operation/'historical_metadata.tar.gz'
    rows=archive_metadata(base,candidates,archive)
    record=sealed('ARCHIVE',parents=dict(intent=plan['content_hash']),files=rows,
                  archive_sha256=digest_file(archive),archive_bytes=archive.stat().st_size)
    publish(operation/'archive.json',record)
    verify_archive(archive,rows)
    print(f'ARCHIVE VERIFIED: {len(rows)} text/metadata files; retiring old payloads.',flush=True)
    with (operation/'deleted.jsonl').open('x',encoding='utf-8') as journal:
        for row in candidates:
            idle();target=safe(Path(row['path']),base)
            require(signature(target)==row['signature'],'Target directory changed: '+str(target))
            for metadata in rows:
                original=base/metadata['path']
                if original.is_relative_to(target):
                    safe(original,base)
                    require(signature(original)==metadata['signature'],'Archived source metadata changed: '+str(original))
            if row['kind']=='worktree':
                current,reason=code_candidate(base,target,common)
                require(current is not None and current['commit']==row['commit'],'Source changed: '+str(reason))
                run(['git','-C',str(common.parent),'worktree','remove',str(target)],timeout=900)
            else:
                # Canonical, owned, non-symlink target roots and keep disjointness
                # were checked above. shutil.rmtree does not follow child symlinks.
                shutil.rmtree(target)
            event=sealed('DELETED',parents=dict(intent=plan['content_hash'],archive=record['content_hash']),target=row)
            journal.write(canonical(event).decode()+'\n');journal.flush();os.fsync(journal.fileno())
            print('RETIRED '+str(target),flush=True)
    require(digest_file(archive)==record['archive_sha256'],'Metadata archive changed')
    verified_keep(base,project,commit);idle()
    result=sealed('COMPLETE',parents=dict(intent=plan['content_hash'],archive=record['content_hash']),
        removed_directories=len(candidates),retained=retained,skipped=skipped,
        filesystem_free_bytes=shutil.disk_usage(base).free)
    publish(operation/'complete.json',result)
    print(f'COMPLETE: retired {len(candidates)} old directories; '
          f'{result["filesystem_free_bytes"]/2**30:.2f} GiB free. New dataset/runtime/Git/campaign retained.',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorize');args=parser.parse_args()
    digest=digest_file(Path(__file__))
    require(digest==args.expected_sha256,'Uploaded helper SHA256 differs')
    require(getpass.getuser()=='rlyang','Expected Brown user rlyang')
    retire(BASE,args.project,args.commit,execute=args.execute,authorization=args.authorize,helper_sha256=digest)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('STOP: '+str(error),flush=True)
        raise SystemExit(1)
