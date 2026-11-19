"""Binary-safe selected-teacher transfer: Oscar -> local disk -> Hydra.

Standalone stdlib receiver, bounded manifest, no shell binary pipes, no extraction
of untrusted paths. A retry streams again and reuses verified complete files.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

PREFIX='LUKA_SIXCLASS_TAIL_HYDRA'
OSCAR='rlyang@ssh.ccv.brown.edu'
HYDRA='rlyang@ssh.cs.brown.edu'
IMPORTS='/cs/home/rlyang/hlt_u100_pairs/imports'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def safe(root,name):
    root=Path(root).absolute()
    if (not isinstance(name,str) or not name or '\\' in name or ':' in name
            or any(x in ('','.','..') for x in name.split('/'))):raise ValueError('Unsafe transfer path')
    if any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('Destination ancestor is a symlink')
    p=root
    for part in name.split('/'):
        p=p/part
        if p.is_symlink():raise ValueError('Transfer symlink forbidden')
    return p


def validate(m,expected):
    if (not re.fullmatch('[0-9a-f]{64}',expected or '') or m.get('contract')!=PREFIX+'_BUNDLE/v1'
            or m.get('content_hash')!=expected or hashlib.sha256(canonical({k:v for k,v in m.items() if k!='content_hash'})).hexdigest()!=expected):
        raise ValueError('Bundle identity differs')
    names=[r['path'] for r in m['files']]
    if names!=sorted(set(names)) or not names or sum(r['bytes'] for r in m['files'])!=m['bytes']:
        raise ValueError('Inventory/byte total differs')
    for r in m['files']:
        safe(Path.cwd(),r['path'])
        if (set(r)!=set(('path','bytes','sha256')) or type(r['bytes'])!=int or r['bytes']<0
                or not re.fullmatch('[0-9a-f]{64}',r['sha256'])):raise ValueError('Malformed file record')
        name=r['path']
        if (name!='campaign_spec.json' and not name.startswith(('population/','submission/','train_OFFLINE/',
                'train_M0HLT/','train_CONCAT/','train_UE100/'))):raise ValueError('Unauthorized bundle member')
        if '/epochs/' in name and not name.endswith('.json'):raise ValueError('Do not transfer optimizer states')
    if 'campaign_spec.json' not in names or 'population/population.json' not in names:
        raise ValueError('Missing population/spec')
    return m


def manifest(path,expected):
    path=Path(path)
    if path.stat().st_size>16*1024**2:raise ValueError('Manifest too large')
    return validate(json.loads(path.read_bytes()),expected)


def publish(path,data):
    # No overwrite, completion last; one transfer writer must own the directory.
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing record differs')
        return
    temp=path.with_name('.'+path.name+'.partial')
    if temp.is_symlink():raise ValueError('Partial record symlink')
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def receive(root,expected,stream):
    root=Path(root).absolute();safe(root,'bundle.json');root.mkdir(parents=True,exist_ok=True)
    # Serialize concurrent retries; releases automatically on SSH interruption.
    lock=safe(root,'.transfer.lock')
    with lock.open('a+b') as handle:
        if os.name=='nt':
            import msvcrt
            if lock.stat().st_size==0:handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        _receive(root,expected,stream)


def _receive(root,expected,stream):
    with tarfile.open(fileobj=stream,mode='r|') as tar:
        member=tar.next()
        if member is None or member.name!='bundle.json' or not member.isfile() or member.size>16*1024**2:
            raise ValueError('Missing bounded bundle manifest')
        m=validate(json.loads(tar.extractfile(member).read()),expected)
        marker=safe(root,'bundle.json')
        if not marker.exists() and any(p.name not in ('.transfer.lock','.bundle.json.partial') for p in root.iterdir()):
            raise FileExistsError('Nonempty destination without matching identity')
        if marker.exists() and marker.read_bytes()!=canonical(m):raise ValueError('Destination belongs to another bundle')
        remaining=0
        for r in m['files']:
            p=safe(root,r['path'])
            if p.exists():
                if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
                    raise ValueError('Existing destination differs; refusing overwrite: '+r['path'])
            else:remaining+=r['bytes']
        if shutil.disk_usage(root).free<remaining+2*1024**3:raise OSError('Insufficient destination disk space')
        publish(marker,canonical(m))
        for i,r in enumerate(m['files'],1):
            member=tar.next()
            if member is None or not member.isfile() or member.name!=r['path'] or member.size!=r['bytes']:
                raise ValueError('Archive inventory differs')
            target=safe(root,r['path']);target.parent.mkdir(parents=True,exist_ok=True)
            exists=target.exists();temp=safe(root,str(PurePosixPath(r['path']).with_name('.'+target.name+'.partial')))
            out=None if exists else temp.open('wb');h=hashlib.sha256();size=0
            try:
                inp=tar.extractfile(member)
                for block in iter(lambda:inp.read(8*1024**2),b''):
                    size+=len(block);h.update(block)
                    if out:out.write(block)
                if out:out.flush();os.fsync(out.fileno())
            finally:
                if out:out.close()
            if size!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('Transferred checksum differs: '+r['path'])
            if not exists:os.link(temp,target);temp.unlink()
            if i%500==0:print(f'Verified {i:,}/{len(m["files"]):,} files',flush=True)
        if tar.next() is not None:raise ValueError('Unexpected archive member')
    publish(safe(root,'transfer_complete.json'),canonical(dict(contract=PREFIX+'_TRANSFER_COMPLETE/v1',
        bundle_sha256=expected,files=len(m['files']),bytes=m['bytes'],all_files_verified=True)))
    print('TEACHER TRANSFER VERIFIED: '+str(root),flush=True)


def stream(root,m,output):
    with tarfile.open(fileobj=output,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
        data=canonical(m);info=tarfile.TarInfo('bundle.json');info.size=len(data);info.mode=0o600
        tar.addfile(info,io.BytesIO(data))
        for r in m['files']:
            p=safe(root,r['path'])
            if not p.is_file() or p.stat().st_size!=r['bytes']:raise ValueError('Source size changed')
            info=tarfile.TarInfo(r['path']);info.size=r['bytes'];info.mode=0o600
            with p.open('rb') as f:tar.addfile(info,f)
    # Receiver hashes every byte; never trusts the source's size check alone.


def ssh(host,identity=None):
    args=['ssh','-T','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    if identity:
        p=Path(identity).expanduser().absolute()
        if not p.is_file():raise ValueError('SSH identity file not found')
        args+=['-i',str(p)]  # Only SSH reads the private key.
    return args+[host]


def remote(args):
    code=base64.b64encode(Path(__file__).read_bytes()).decode()
    bootstrap=f'import base64; exec(compile(base64.b64decode({code!r}),"teacher_transfer.py","exec"))'
    return shlex.join(['python3','-I','-u','-B','-s','-c',bootstrap,*args])


def fetch(root,path,expected,identity=None):
    print('Downloading selected teachers from Oscar. Keep this terminal open.',flush=True)
    p=subprocess.Popen(ssh(OSCAR,identity)+[remote(['stream-source','--manifest',path,'--sha256',expected])],stdout=subprocess.PIPE)
    try:
        receive(root,expected,p.stdout);p.stdout.close()
        if p.wait()!=0:raise RuntimeError('Oscar SSH failed; inspect its diagnostic above')
    finally:
        if p.poll() is None:p.terminate()
        p.wait()


def send(root,expected,identity=None):
    root=Path(root);m=manifest(root/'bundle.json',expected)
    done=json.loads((root/'transfer_complete.json').read_bytes())
    if done.get('bundle_sha256')!=expected or done.get('all_files_verified') is not True:raise ValueError('Download incomplete')
    destination=IMPORTS+'/sixclass_oscar_'+expected[:16]
    print(f'Uploading {m["bytes"]/2**30:.2f} GiB to {destination}; keep terminal open.',flush=True)
    p=subprocess.Popen(ssh(HYDRA,identity)+[remote(['receive','--root',destination,'--sha256',expected])],stdin=subprocess.PIPE)
    try:
        stream(root,m,p.stdin);p.stdin.close()
        if p.wait()!=0:raise RuntimeError('Hydra SSH/receiver failed; see its diagnostic above')
    finally:
        if p.stdin and not p.stdin.closed:
            try:p.stdin.close()
            except OSError:pass
        if p.poll() is None:p.terminate()
        p.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('fetch','send','receive','stream-source'))
    p.add_argument('--root');p.add_argument('--manifest');p.add_argument('--sha256',required=True);p.add_argument('--identity-file')
    a=p.parse_args()
    if a.command=='stream-source':
        m=manifest(a.manifest,a.sha256);root=PurePosixPath(m['source_root'])
        try:root.relative_to('/oscar/home/rlyang/hlt_classification/checkpoints')
        except ValueError:raise ValueError('Use original Oscar checkpoint root') from None
        if '..' in root.parts:raise ValueError('Original source path traverses outside the campaign')
        stream(str(root),m,sys.stdout.buffer)
    elif a.command=='receive':receive(a.root,a.sha256,sys.stdin.buffer)
    elif a.command=='fetch':fetch(a.root,a.manifest,a.sha256,a.identity_file)
    else:send(a.root,a.sha256,a.identity_file)


if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError,tarfile.TarError) as exc:
        raise SystemExit('STOP: '+str(exc)+'; verified files are preserved. Use the same identity after resolving the error.') from None
