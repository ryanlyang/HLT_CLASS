"""Standard-library, binary-safe SSH stream. Run locally; receiver needs Python3 only."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def safe(root,name):
    if not isinstance(name,str) or '\\' in name or ':' in name or any(x in ('','.','..') for x in name.split('/')):
        raise ValueError('Unsafe transfer path')
    root=Path(root).resolve()
    for i in range(1,len(name.split('/'))+1):
        if root.joinpath(*name.split('/')[:i]).is_symlink():raise ValueError('Transfer symlink forbidden')
    return root/name


def inventory(root):
    root=Path(root).resolve();f=json.loads((root/'foundation/foundation.json').read_bytes())
    if (f.get('contract')!='LUKA_NEW_FULLSTAT_HYDRA_FOUNDATION/v1' or f.get('production') is not True
            or hashlib.sha256(canonical({k:v for k,v in f.items() if k!='content_hash'})).hexdigest()!=f['content_hash']):
        raise ValueError('Need authenticated production foundation')
    records=[dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256']) for r in f['files']]
    records+=[dict(path='foundation/'+r['membership']['path'],bytes=r['membership']['bytes'],sha256=r['membership']['sha256']) for r in f['files']]
    for name in ('source.sha256','foundation/foundation.json','foundation/producer.json'):
        p=root/name;records.append(dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
    if digest(root/'source.sha256')!=f['source_manifest_sha256']:raise ValueError('Source manifest drift')
    records.sort(key=lambda r:r['path'])
    for r in records:
        if safe(root,r['path']).stat().st_size!=r['bytes']:raise ValueError('Local input size changed')
    value=dict(contract='LUKA_NEW_FULLSTAT_TRANSFER/v1',foundation_sha256=f['content_hash'],files=records,
               bytes=sum(r['bytes'] for r in records))
    return value,hashlib.sha256(canonical(value)).hexdigest()


def receive(root,expected,stream):
    root=Path(root)
    if root.is_symlink():raise ValueError('Destination is a symlink')
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=stream,mode='r|') as tar:
        member=tar.next()
        if member is None or member.name!='transfer_manifest.json' or not member.isfile() or member.size>12*1024**2:
            raise ValueError('Missing bounded transfer manifest')
        data=tar.extractfile(member).read();value=json.loads(data)
        if hashlib.sha256(canonical(value)).hexdigest()!=expected or value.get('contract')!='LUKA_NEW_FULLSTAT_TRANSFER/v1':
            raise ValueError('Transfer identity differs')
        marker=root/'transfer_manifest.json'
        if marker.exists():
            if marker.read_bytes()!=data:raise FileExistsError('Destination belongs to another transfer')
        elif any(root.iterdir()):raise FileExistsError('Destination is nonempty without transfer identity')
        else:
            if shutil.disk_usage(root).free<value['bytes']+2*1024**3:raise OSError('Insufficient destination space')
            with marker.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        names=[r['path'] for r in value['files']]
        if names!=sorted(set(names)):raise ValueError('Transfer path duplicates/order')
        for i,r in enumerate(value['files'],1):
            member=tar.next()
            if member is None or not member.isfile() or member.name!=r['path'] or member.size!=r['bytes']:
                raise ValueError('Archive inventory differs')
            target=safe(root,r['path']);target.parent.mkdir(parents=True,exist_ok=True)
            exists=target.exists()
            if exists and (target.stat().st_size!=r['bytes'] or digest(target)!=r['sha256']):
                raise ValueError('Existing destination file differs; not overwriting')
            temp=target.with_name('.'+target.name+'.transfer-partial')
            if temp.is_symlink():raise ValueError('Partial transfer symlink')
            out=None if exists else temp.open('wb')
            h=hashlib.sha256();size=0
            try:
                inp=tar.extractfile(member)
                for block in iter(lambda:inp.read(8*1024**2),b''):
                    h.update(block);size+=len(block)
                    if out:out.write(block)
                if out:out.flush();os.fsync(out.fileno())
            finally:
                if out:out.close()
            if size!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('Stream checksum differs: '+r['path'])
            if not exists:os.link(temp,target);temp.unlink()
            if i%500==0:print(f'Verified received files {i:,}/{len(names):,}',flush=True)
        if tar.next() is not None:raise ValueError('Unexpected archive member')
    receipt=root/'transfer_complete.json'
    result=canonical(dict(contract='LUKA_NEW_FULLSTAT_TRANSFER_COMPLETE/v1',manifest_sha256=expected,
                          files=len(names),bytes=value['bytes'],all_files_verified=True))
    if receipt.exists():
        if receipt.read_bytes()!=result:raise ValueError('Existing completion differs')
    else:
        with receipt.open('xb') as f:f.write(result);f.flush();os.fsync(f.fileno())
    print('TRANSFER VERIFIED: '+str(root),flush=True)


def ssh_command(host,identity_file=None):
    args=['ssh','-T','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    if identity_file is not None:
        identity=Path(identity_file).expanduser().absolute()
        if not identity.is_file():raise ValueError('SSH identity file not found: '+str(identity))
        # Pass the path to OpenSSH; never read, copy or record private-key bytes.
        args+=['-i',str(identity)]
    return args+[host]


def close_input(process):
    if process.stdin and not process.stdin.closed:
        try:process.stdin.close()
        except OSError:pass  # Windows may report EINVAL as well as EPIPE.


def send(root,host,destination,*,identity_file=None):
    root=Path(root);value,sha=inventory(root)
    if host!='rlyang@ssh.cs.brown.edu' or destination!='/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1':
        raise ValueError('Explicit registered Brown destination only')
    command=ssh_command(host,identity_file)
    # SSH prompts are handled by the user's terminal; no credentials enter Python.
    code=base64.b64encode(Path(__file__).read_bytes()).decode()
    bootstrap=f'import base64; exec(compile(base64.b64decode({code!r}),"transfer_receiver.py","exec"))'
    remote=shlex.join(['python3','-I','-u','-B','-s','-c',bootstrap,'receive','--root',destination,'--sha256',sha])
    print(f'Streaming {value["bytes"]/2**30:.2f} GiB; manifest {sha}. Keep this upload terminal open.',flush=True)
    process=subprocess.Popen(command+[remote],stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
            data=canonical(value);info=tarfile.TarInfo('transfer_manifest.json');info.size=len(data);info.mode=0o600
            tar.addfile(info,io.BytesIO(data))
            for r in value['files']:
                info=tarfile.TarInfo(r['path']);info.size=r['bytes'];info.mode=0o600
                with safe(root,r['path']).open('rb') as f:tar.addfile(info,f)
        process.stdin.close()
        status=process.wait()
        if status!=0:raise RuntimeError(f'SSH/receiver exited {status}; see its message above. Same transfer may be retried after fixing the cause.')
    except (OSError,tarfile.TarError) as exc:
        close_input(process)
        status=process.wait()
        if status==255:
            raise RuntimeError('SSH failed (exit 255); see its message above. For Permission denied (publickey), '
                               'use --identity-file with the same key as your working SSH login.') from None
        raise RuntimeError(f'Transfer stopped; SSH/receiver exit {status}: {exc}. '
                           'Existing verified files are preserved; fix the cause before retrying.') from None
    except BaseException:
        close_input(process)
        if process.poll() is None:process.terminate()
        process.wait()
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['send','receive','inspect'])
    p.add_argument('--root',required=True);p.add_argument('--sha256')
    p.add_argument('--host',default='rlyang@ssh.cs.brown.edu')
    p.add_argument('--identity-file',help='OpenSSH private-key path, as in ssh -i; only SSH reads the key')
    p.add_argument('--destination',default='/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1')
    a=p.parse_args()
    if a.command=='receive':receive(a.root,a.sha256,sys.stdin.buffer)
    elif a.command=='send':send(a.root,a.host,a.destination,identity_file=a.identity_file)
    else:
        value,sha=inventory(a.root);print(json.dumps(dict(files=len(value['files']),bytes=value['bytes'],manifest_sha256=sha),indent=2))


if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError,ValueError,tarfile.TarError) as exc:
        raise SystemExit('STOP: '+str(exc)) from None
