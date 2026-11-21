"""Bounded four-GPU diagnostics; never decode data or release scientific fits."""
import argparse
from decimal import Decimal
import faulthandler
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import traceback

BASE=Path('/cs/home/rlyang/hlt_u100_pairs')
SPEC=BASE/'runs/balanced_reco_600k_a2b634c7_r1/campaign_spec.json'
COMMIT='a2b634c7a4406f24f6a11422f24fdb76b5a1c7ca'
AUTH='AUTHORIZE_HYDRA_RECONSTRUCTION_GPU_DIAGNOSTIC_V1'
PROFILES=('original','gloo_control','gloo_no_p2p')
MINUTES=5


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def sealed(kind,**fields):
    body=dict(contract='LUKA_HYDRA_GPU_DIAGNOSTIC_'+kind+'/v1',schema_version=1,
        final_test_accessed=False,raw_particle_reads=False,scientific_training=False,
        **fields)
    return dict(body,content_hash=hashlib.sha256(canonical(body)).hexdigest())


def read(path):
    value=json.loads(Path(path).read_bytes());body={k:v for k,v in value.items() if k!='content_hash'}
    require(value.get('content_hash')==hashlib.sha256(canonical(body)).hexdigest(),'Metadata checksum differs: '+str(path))
    return value


def publish(path,value):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec():
    value=read(SPEC)
    require(value['contract']=='LUKA_BALANCED_RECONSTRUCTION_HYDRA_SPEC/v1'
        and value['root']==str(SPEC.parent) and value['commit']==COMMIT
        and value['project']==str(BASE/'code_balanced_reco_a2b634c7')
        and value['environment_dir']==str(BASE/'environment_pair_hydra_v1/venv')
        and value['population']['counts']==dict(train=600000,validation=150000),
        'Use the exact failed a2b634c7 execution')
    return value


def commands(value,helper,output,sha,node):
    rows=[]
    for profile in PROFILES:
        program=[value['environment_dir']+'/bin/python','-I','-u','-B',str(helper),
            'supervise','--output',str(output),'--profile',profile,'--expected-sha256',sha]
        exports=dict(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',LD_LIBRARY_PATH=value['environment_dir']+'/lib',
            NCCL_DEBUG='INFO',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        env=['env','-u','PYTHONHOME','-u','PYTHONPATH','-u','LD_PRELOAD','-u','NCCL_P2P_DISABLE']
        if profile=='gloo_no_p2p':exports['NCCL_P2P_DISABLE']='1'
        env += [k+'='+v for k,v in exports.items()]
        argv=['sbatch','--parsable','--partition=gpus','--account=default','--qos=normal',
            '--nodes=1','--ntasks=1','--cpus-per-task=12','--mem=64000M',
            '--gres=gpu:geforce_gtx_1080_ti:4',f'--time=00:{MINUTES:02d}:00',
            '--nodelist='+node,'--export=NONE','--job-name=lukabgpu_'+profile,
            '--output='+str(output/(profile+'-%j.out')),'--chdir='+value['project'],
            '--kill-on-invalid-dep=yes','--comment='+sha+':'+profile]
        if rows:argv+=['--dependency=afterany:{previous}']
        argv+=['--wrap='+shlex.join(env+program)]
        rows.append(dict(profile=profile,command=argv))
    return rows


def submit(helper,sha,node,authorization):
    require(authorization==AUTH,'Exact diagnostic authorization required')
    require(node=='gpu1905','This diagnostic targets the node in the failure log')
    value=spec();output=BASE/'reports'/('reconstruction_gpu_diagnostic_'+sha[:12])
    require(not output.exists() and not output.is_symlink(),'Diagnostic already exists; inspect saved jobs rather than submitting twice')
    require(BASE.resolve(strict=True)==BASE,'Unexpected workspace mapping')
    head=subprocess.check_output(['git','-C',value['project'],'rev-parse','HEAD'],text=True,timeout=30).strip()
    dirty=subprocess.check_output(['git','-C',value['project'],'status','--porcelain','--untracked-files=all'],text=True,timeout=30)
    require(head==COMMIT and not dirty.strip(),'Original source changed/dirty; inspect before GPU submission: '+dirty.strip())
    config=subprocess.check_output(['scontrol','show','config'],text=True,timeout=30)
    require(re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',config) is not None,'Hydra required')
    rows=commands(value,helper,output,sha,node)
    for row in rows:
        review=[a for a in row['command'] if not a.startswith('--dependency=')]+['--test-only']
        subprocess.run(review,check=True,timeout=30)
    output.mkdir(parents=True)
    for profile in PROFILES:(output/profile).mkdir()
    plan=sealed('PLAN',parents=dict(spec=value['content_hash']),helper_sha256=sha,
        source_commit=COMMIT,node=node,minutes_per_job=MINUTES,profiles=rows)
    publish(output/'plan.json',plan);previous=None
    for row in rows:
        args=[a.replace('{previous}',previous) if previous else a for a in row['command']]
        intent=sealed('INTENT',parents=dict(plan=plan['content_hash']),profile=row['profile'],command=args)
        publish(output/(row['profile']+'.intent.json'),intent)
        result=subprocess.run(args,capture_output=True,text=True,timeout=30)
        job=result.stdout.strip().split(';')[0]
        require(result.returncode==0 and job.isdecimal() and int(job)>0,
                'Uncertain submission; inspect intent before retry: '+result.stdout+result.stderr)
        publish(output/(row['profile']+'.submitted.json'),sealed('SUBMITTED',
            parents=dict(intent=intent['content_hash']),profile=row['profile'],job=job))
        previous=job;print('SUBMITTED '+row['profile']+': '+job,flush=True)
    print('DIAGNOSTIC OUTPUT: '+str(output),flush=True)
    print('Only three five-minute diagnostic jobs submitted. No data preparation or science submission.',flush=True)


def bound(value,output,profile,sha):
    require(profile in PROFILES and output==BASE/'reports'/('reconstruction_gpu_diagnostic_'+sha[:12]),'Diagnostic location/profile differs')
    plan=read(output/'plan.json');require(plan['helper_sha256']==sha and plan['parents']==dict(spec=value['content_hash']),
        'Diagnostic source/spec binding differs')
    job=os.environ.get('SLURM_JOB_ID','');require(job.isdecimal(),'Slurm allocation required')
    import time
    for _ in range(12):
        receipt=output/(profile+'.submitted.json')
        if receipt.exists():break
        time.sleep(1)
    record=read(receipt)
    intent=read(output/(profile+'.intent.json'))
    require(record['job']==job and record['profile']==profile and record['parents']==dict(intent=intent['content_hash'])
        and intent['parents']==dict(plan=plan['content_hash']),'Saved diagnostic job differs')
    fields=dict(x.split('=',1) for x in subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30).split() if '=' in x)
    require(fields.get('JobState')=='RUNNING' and fields.get('Comment')==sha+':'+profile
        and fields.get('Partition')=='gpus' and fields.get('NumNodes')=='1'
        and fields.get('TimeLimit') in ('00:05:00','5:00') and fields.get('NodeList')=='gpu1905'
        and fields.get('CPUs/Task')=='12' and fields.get('NumTasks')=='1', 'Diagnostic allocation differs')
    tres=dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    memory=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    memory_mb=None if memory is None else Decimal(memory[1])*{'K':Decimal(1)/1024,'':1,'M':1,'G':1024,'T':1024**2}[memory[2]]
    require(tres.get('gres/gpu')=='4' and memory_mb==64000
        and all(k=='gres/gpu:geforce_gtx_1080_ti' and v=='4' for k,v in tres.items() if k.startswith('gres/gpu:')),
        'Diagnostic GPU/memory request differs')
    return job


def context(base_class,control,dist):
    """Gloo object transport plus synchronized handoff; tensors retain their group."""
    class HostContext(base_class):
        def main(self,fn):
            result=None;error=None
            if self.rank==0:
                try:result=fn()
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            box=[error,result]
            if self.world>1:dist.broadcast_object_list(box,src=0,group=control)
            self.barrier()
            if box[0] is not None:raise RuntimeError(box[0])
            return box[1]
        def all(self,value):
            if self.world==1:return [value]
            values=[None]*self.world;dist.all_gather_object(values,value,group=control);return values
        def barrier(self):
            if self.world>1:dist.barrier(group=control)
    return HostContext


class SyntheticCache:
    """Technical-only inputs. No particle source or split capability exists."""
    def __len__(self):return 16
    def batch(self,ix):
        import numpy as np
        rng=np.random.default_rng(71);features=rng.normal(0,.25,(16,17,8)).astype(np.float32)
        vectors=rng.normal(0,1,(16,4,8)).astype(np.float32)
        vectors[:,3]=np.sqrt((vectors[:,:3]**2).sum(1)+1)
        return dict(features=features[ix],vectors=vectors[ix],mask=np.ones((len(ix),1,8),np.float32),
            labels=(np.arange(16,dtype=np.int64)%6)[ix])


def device_for_rank(torch,rank):
    return torch.device('cuda',rank)


def rank(value,output,profile,sha):
    import numpy as np
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    r=int(os.environ['RANK']);require(r in range(4) and os.environ['WORLD_SIZE']=='4','Exactly four ranks required')
    stream=(output/profile/f'rank_{r}.stack.log').open('x')
    faulthandler.enable(file=stream);faulthandler.dump_traceback_later(40,repeat=True,file=stream)
    stage='startup'
    def mark(message):
        nonlocal stage
        stage=message;print(f'GPU_DIAGNOSTIC rank={r} BEGIN {message}',flush=True)
        stream.write(message+'\n');stream.flush()
    try:
        mark('source/runtime verification')
        sys.path.insert(0,str(Path(value['project'])/'src'))
        from hlt_classification.luka_balanced_reconstruction_hydra import campaign,kernel,resources
        from hlt_classification.luka_subclass_ddp_hydra.distributed import Context
        torch.cuda.set_device(r);kernel.configure(4,12);campaign.check(value)
        require(torch.cuda.device_count()==4 and torch.cuda.get_device_capability(r)==(6,1)
            and torch.cuda.get_device_name(r).removeprefix('NVIDIA ')=='GeForce GTX 1080 Ti',
            'Four GTX1080Ti GPUs required')
        device=device_for_rank(torch,r)
        mark('NCCL initialization;45-second collective timeout')
        dist.init_process_group('nccl',timeout=timedelta(seconds=45))
        group=None
        if profile!='original':
            mark('Gloo control initialization')
            group=dist.new_group(backend='gloo',timeout=timedelta(seconds=45))
            Context=context(Context,group,dist)
        ctx=Context(r,4,device)
        mark('first Context.main object broadcast')
        ctx.main(lambda:None);ctx.sync()
        mark('NCCL tensor sum')
        tensor=torch.full((8,),float(r+1),device=device);dist.all_reduce(tensor);ctx.sync()
        require(bool(torch.all(tensor==10)),'Tensor sum differs')
        mark('NCCL tensor broadcast')
        tensor.fill_(42 if r==0 else 0);dist.broadcast(tensor,src=0);ctx.sync()
        require(bool(torch.all(tensor==42)),'Tensor broadcast differs')
        mark('CPU-object gather')
        gathered=ctx.all(dict(rank=r));require(sorted(x['rank'] for x in gathered)==list(range(4)),'Object gather differs')
        mark('original production OFFLINE full-weight parity; synthetic rows only')
        proof=resources.parity(value,'OFFLINE_s10',SyntheticCache(),None,None,ctx)
        result=sealed('RANK',parents=dict(spec=value['content_hash']),helper_sha256=sha,
            profile=profile,job=os.environ['SLURM_JOB_ID'],rank=r,host=socket.gethostname(),
            gpu=torch.cuda.get_device_name(r),torch_version=torch.__version__,
            nccl_version=list(torch.cuda.nccl.version()),parity=proof,passed=True,discarded=True)
        mark('successful collective barrier')
        ctx.barrier();ctx.sync()
        publish(output/profile/f'rank_{r}.json',result)
    except Exception:
        publish(output/profile/f'rank_{r}.error.json',sealed('ERROR',parents=dict(spec=value['content_hash']),
            profile=profile,rank=r,stage=stage,traceback=traceback.format_exc()))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()
        stream.close()


def supervise(value,output,profile,sha,helper):
    job=bound(value,output,profile,sha)
    # Failed NCCL jobs must not leave huge core files inside the frozen source.
    if os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    args=[sys.executable,'-X','faulthandler','-u','-B','-s','-m','torch.distributed.run',
        '--standalone','--nnodes=1','--nproc-per-node=4','--max-restarts=0','--',str(helper),
        'rank','--output',str(output),'--profile',profile,'--expected-sha256',sha]
    result=subprocess.run(args,cwd=output/profile)
    if result.returncode:
        publish(output/profile/'failed.json',sealed('FAILED',parents=dict(spec=value['content_hash']),
            profile=profile,job=job,returncode=result.returncode))
        raise RuntimeError('Diagnostic failed; inspect individual rank stack/error files')
    rows=[read(output/profile/f'rank_{r}.json') for r in range(4)]
    require(all(x['passed'] and x['job']==job and x['profile']==profile and x['rank']==r
        and x['helper_sha256']==sha and x['parents']==dict(spec=value['content_hash'])
        and x['raw_particle_reads'] is False and x['scientific_training'] is False for r,x in enumerate(rows)),
            'Incomplete four-rank evidence')
    publish(output/profile/'complete.json',sealed('COMPLETE',parents=dict(spec=value['content_hash']),
        profile=profile,job=job,ranks=[x['content_hash'] for x in rows],passed=True))
    print('DIAGNOSTIC PASSED '+profile,flush=True)


def status(output):
    print('Diagnostic directory: '+str(output))
    jobs=[]
    for profile in PROFILES:
        receipt=output/(profile+'.submitted.json')
        if not receipt.exists():print(profile+': not submitted');continue
        job=read(receipt)['job'];jobs.append(job)
        folder=output/profile
        state='saved pass' if (folder/'complete.json').exists() else 'saved failure' if (folder/'failed.json').exists() else 'no completion yet'
        print(profile+' | job '+job+' | '+state)
        for path in sorted(folder.glob('rank_*.error.json')):
            row=read(path);print(path.name+' stage='+row['stage']);print(row['traceback'][-6000:])
        if not (folder/'complete.json').exists():
            stack=folder/'rank_0.stack.log'
            if stack.exists():print('RANK 0 STACK:\n'+'\n'.join(stack.read_text(errors='replace').splitlines()[-60:]))
    if jobs:subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs),
        '-o','JobIDRaw,JobName%32,State,ExitCode,Elapsed,Timelimit,NodeList'],check=True,timeout=30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['submit','supervise','rank','status'])
    p.add_argument('--expected-sha256',required=True);p.add_argument('--authorize');p.add_argument('--node',default='gpu1905')
    p.add_argument('--output',type=Path);p.add_argument('--profile',choices=PROFILES);a=p.parse_args()
    helper=Path(__file__).resolve();sha=digest(helper)
    require(sha==a.expected_sha256 and getpass.getuser()=='rlyang','Helper identity/user differs')
    if a.mode=='submit':submit(helper,sha,a.node,a.authorize);return
    if a.mode=='status':status(BASE/'reports'/('reconstruction_gpu_diagnostic_'+sha[:12]));return
    value=spec();bound(value,a.output,a.profile,sha)
    if a.mode=='supervise':supervise(value,a.output,a.profile,sha,helper)
    else:rank(value,a.output,a.profile,sha)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('GPU_DIAGNOSTIC FAILED: '+str(error),flush=True);traceback.print_exc();raise SystemExit(1)
