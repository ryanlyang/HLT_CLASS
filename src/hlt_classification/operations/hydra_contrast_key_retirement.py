"""Retire one unused derived TRAIN key export; standard library only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

BASE=Path('/cs/home/rlyang/hlt_u100_pairs')
DELTA='contrast_delta_54473f10'
MEMBER='keys/keys.npz'
EXPECTED_BYTES=169968554
OPERATION='contrast_delta_54473f10_keys_v1'
AUTH='RETIRE_UNUSED_HYDRA_CONTRAST_KEYS_V1'


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def hashed(value):
    require(isinstance(value,dict),'Expected metadata object')
    body={k:v for k,v in value.items() if k!='content_hash'}
    require(value.get('content_hash')==hashlib.sha256(canonical(body)).hexdigest(),'Metadata content hash differs')
    return value


def seal(kind,**fields):
    body=dict(contract='LUKA_HYDRA_CONTRAST_KEY_RETIREMENT_'+kind+'/v1',schema_version=1,
        final_test_accessed=False,**fields)
    return dict(body,content_hash=hashlib.sha256(canonical(body)).hexdigest())


def safe(path,base,*,directory=False):
    path=Path(path).absolute();base=Path(base).absolute()
    require(path==base or path.is_relative_to(base),'Path escapes workspace')
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink forbidden')
    require(path.resolve(strict=True)==path,'Noncanonical path')
    info=path.stat()
    if hasattr(os,'getuid'):require(info.st_uid==os.getuid(),'Foreign owner')
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode) and info.st_nlink==1,
        'Unexpected file type or hardlink')
    return path


def signature(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def file_ref(path,base):
    path=safe(path,base);before=signature(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    require(signature(path)==before,'File changed during checksum')
    s=path.stat()
    return dict(path=str(path),bytes=s.st_size,sha256=digest.hexdigest(),
        allocated_bytes=getattr(s,'st_blocks',(s.st_size+511)//512)*512)


def read(path,base,*,verify=False):
    path=safe(path,base);require(path.stat().st_size<=64*2**20,'Oversized metadata')
    value=json.loads(path.read_bytes());require(isinstance(value,dict),'Expected JSON object')
    return hashed(value) if verify or 'content_hash' in value else value


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)


def normal(text,base):
    return re.sub(r'(?<![\w/.-])/(?:cs/)?home/rlyang/hlt_u100_pairs(?=/|$|[\s\'\"])',
        lambda _:str(base),text)


def scheduler():
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
    def run(args):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=45,env=env).stdout
    jobs=run(['squeue','--me','--noheader','--format=%i']).split();result={}
    require(all(re.fullmatch('[1-9][0-9]*(?:_[0-9]+)?',j) for j in jobs),'Unknown scheduler ID')
    for job in jobs:
        details=run(['scontrol','show','job','-o',job])
        require('JobId='+job+' ' in details+' ','Job identity differs');result[job]=details
    return result


def audit(base,jobs):
    """Follow current specs and local absolute JSON/directory parents, fail closed."""
    base=safe(base,base,directory=True);catalog=[]
    for folder in sorted((base/'runs').iterdir()):
        if not folder.is_dir() or folder.is_symlink():continue
        for path in folder.glob('*spec*.json'):
            if 'test' in path.name.lower():continue
            value=read(path,base,verify=True)
            if normal(value.get('root',''),base)==str(folder):catalog.append((path,value))
    pending=[];texts=[normal(t,base) for t in jobs.values()]
    for job,details in jobs.items():
        details=normal(details,base)
        matches=[p for p,_ in catalog if str(p.parent)+os.sep in details]
        if not matches:
            matches=[p for p,v in catalog if any('WorkDir='+normal(v.get(k,'<absent>'),base)+' ' in details+' '
                for k in ('project','project_dir'))]
        require(matches,'Cannot resolve live job '+job+'; no deletion');pending.extend(matches)
    seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        require(len(seen)<8192,'Dependency graph exceeded bound');seen.add(path)
        value=read(path,base);values=[normal(t,base) for t in strings(value)];texts.extend(values)
        for text in values:
            linked=Path(text)
            if not linked.is_absolute() or not linked.is_relative_to(base) or '\n' in text:continue
            if 'test' in [p.lower() for p in linked.parts]:continue
            if linked.suffix=='.json':
                require(linked.is_file(),'Missing live metadata: '+str(linked));pending.append(linked)
            elif linked.is_dir():
                safe(linked,base,directory=True)
                for name in ('campaign_spec.json','recovery_spec.json','benchmark_spec.json','manifest.json','origin/campaign_spec.json'):
                    if (linked/name).is_file():pending.append(linked/name)
    return dict(jobs=jobs,metadata=sorted(map(str,seen)),blocked=any(DELTA in t for t in texts))


def inventory(base):
    folder=safe(base/DELTA,base,directory=True);target=safe(folder/MEMBER,base)
    transfer=read(folder/'manifest.json',base,verify=True)
    keys=read(folder/'keys/manifest.json',base,verify=True)
    require(transfer['contract']=='LUKA_U100_CONTRASTIVE_HYDRA_TRANSFER/v1'
        and keys['contract']=='LUKA_U100_CONTRASTIVE_KEYS/v1','Wrong export contracts')
    old=hashed(transfer['origin_spec']);owner=hashed(transfer['key_owner']);replay=hashed(transfer['original_replay'])
    require(transfer['parents']==dict(origin=old['content_hash'],replay=replay['content_hash']),
        'Transfer lineage differs')
    require(keys['role']=='train' and keys['coordinate']=='OFFLINE' and keys['width']==128
        and keys['rows']==old['counts']['train'] and keys['payload']['path']=='keys.npz'
        and keys['parents']['spec']==owner['content_hash']
        and keys['parents']['imported']==old['imported']['content_hash']
        and keys['parents']['prepared']==old['anchor']['evidence']['prepared']
        and keys['teacher']==old['anchor']['evidence']['models']['OFFLINE']['checkpoint'],
        'Wrong TRAIN key bank or teacher binding')
    refs=[r for r in transfer['files'] if r['path']==MEMBER]
    manifests=[r for r in transfer['files'] if r['path']=='keys/manifest.json']
    require(len(refs)==len(manifests)==1,'Missing or duplicate transfer member')
    payload=file_ref(target,base);metadata=file_ref(folder/'keys/manifest.json',base)
    require(payload['bytes']==EXPECTED_BYTES,'Unexpected registered key payload size')
    require(all(payload[k]==refs[0][k]==keys['payload'][k] for k in ('bytes','sha256')),
        'Key payload checksum differs')
    require(all(metadata[k]==manifests[0][k] for k in ('bytes','sha256')),'Key manifest differs')
    return dict(remove=payload,retain=[file_ref(folder/'manifest.json',base),metadata],
        transfer_sha256=transfer['content_hash'],key_manifest_sha256=keys['content_hash'])


def publish(path,value):
    temp=path.with_name(path.name+'.partial')
    with temp.open('xb') as stream:stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    os.link(temp,path);temp.unlink()


def retire(base=BASE,*,execute=False,authorization=None,guard=None,source_commit=None):
    base=safe(base,base,directory=True);operation=base/'retired_campaigns'/OPERATION
    require(not os.path.lexists(operation),'Existing retirement record; preserve it and inspect before retry')
    guard=guard or (lambda:audit(base,scheduler()))
    live=guard();require(not live['blocked'],'Old contrastive keys are a live dependency; nothing deleted')
    proof=inventory(base)
    plan=seal('INTENT',operation=OPERATION,source_commit=source_commit,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),live=live,**proof,
        consequence='old_contrastive_training_replay_requires_restoring_or_reexporting_keys')
    print('TARGET:',proof['remove']['path'],flush=True)
    print(f'ELIGIBLE: {proof["remove"]["allocated_bytes"]/2**20:.1f} allocated MiB; one derived TRAIN export.',flush=True)
    if not execute:print('PREVIEW: no files or jobs changed.',flush=True);return plan
    require(authorization==AUTH,'Exact key-retirement authorization required')
    require(source_commit and re.fullmatch('[0-9a-f]{40}',source_commit),'Committed helper source required')
    require(not guard()['blocked'],'New live dependency; nothing deleted')
    if not operation.parent.exists():operation.parent.mkdir()
    safe(operation.parent,base,directory=True);operation.mkdir();publish(operation/'intent.json',plan)
    require(not guard()['blocked'],'New live dependency; retain retirement record for inspection')
    for ref in [*proof['retain'],proof['remove']]:require(file_ref(Path(ref['path']),base)==ref,'Payload changed before retirement')
    target=safe(Path(proof['remove']['path']),base)
    require(target==base/DELTA/MEMBER,'Target outside exact allowlist');target.unlink()
    for ref in proof['retain']:require(file_ref(Path(ref['path']),base)==ref,'Retained metadata changed')
    require(not os.path.lexists(target),'Retired payload reappeared')
    complete=seal('COMPLETE',parents=dict(intent=plan['content_hash']),retired=proof['remove'],
        raw_datasets_changed=False,models_changed=False,predictions_changed=False,jobs_changed=False,
        free_bytes=shutil.disk_usage(base).free)
    publish(operation/'complete.json',complete)
    print(f'RETIRED: {proof["remove"]["allocated_bytes"]/2**20:.1f} MiB. Current free: {complete["free_bytes"]/2**30:.3f} GiB.',flush=True)
    return complete
