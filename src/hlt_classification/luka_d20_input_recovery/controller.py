"""Selective, journalled recovery while unrelated original jobs remain live."""
from decimal import Decimal
import os
from pathlib import Path
import re
import subprocess
import time
from . import contracts as c,runtime

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def env():return {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}


def states(jobs):
    jobs=list(dict.fromkeys(jobs))
    if not jobs:return {}
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),
        '-o','JobIDRaw,State,ExitCode'],text=True,timeout=30)
    result={}
    for line in text.splitlines():
        row=line.strip().split('|')
        if len(row)>=3 and row[0] in jobs:
            c.require(row[0] not in result,'Ambiguous Slurm accounting row')
            result[row[0]]=(row[1].split()[0],row[2])
    c.require(set(result)==set(jobs),'Missing scheduler accounting; wait before recovery')
    return result


def qualified(spec,old):
    q=old.resources.qualification(old.spec);receipt=old.worker.task_receipt(old.spec,'qualify')
    c.require(q['content_hash']==spec['qualification_hash'] and receipt['job']==spec['qualification_job'],
        'Original qualification or accepted job changed')
    c.require(states([receipt['job']])[receipt['job']]==('COMPLETED','0:0'),
        'Original qualification did not finish successfully')


def create(project,commit,root,tasks):
    project=Path(project).resolve(strict=True);root=Path(root).absolute()
    c.require(project.is_relative_to(c.BASE) and root.is_relative_to(c.BASE/'runs')
        and root.resolve()==root and not any(p.is_symlink() for p in (root,*root.parents))
        and not root.exists() and not root.is_symlink(),'Use a fresh Hydra recovery directory')
    src=runtime.source(project);c.require(src['commit']==commit,'Exact pushed repair commit required')
    parent_spec=c.ref(c.PARENT/'campaign_spec.json')
    temporary=dict(project=str(project),parent_spec=parent_spec)
    old=runtime.original(temporary)
    q=old.resources.qualification(old.spec);receipt=old.worker.task_receipt(old.spec,'qualify')
    c.require(states([receipt['job']])[receipt['job']]==('COMPLETED','0:0'),'Qualification did not finish successfully')
    accepted=old.campaign.accepted_jobs(old.spec);latest={r['task']:r for r in accepted}
    c.require(tasks and len(tasks)==len(set(tasks)) and all(t in old.contracts.GPU_TASKS for t in tasks),
        'Explicit registered recovery roots required')
    status=states(latest[t]['job'] for t in tasks)
    for task in tasks:
        state=status[latest[task]['job']]
        c.require(state[0] in TERMINAL and state!=('COMPLETED','0:0'),'Recovery root must be a failed terminal segment: '+task)
    receipts=[c.PARENT/'tasks/qualify.json',*sorted((c.PARENT/'tasks/recovery').glob('qualify-*.json'))]
    matching=[path for path in receipts if path.exists() and c.read(path)==receipt]
    c.require(len(matching)==1,'Original qualification receipt is absent or ambiguous')
    root.mkdir(parents=True)
    spec=c.artifact('SPEC',root=str(root),project=str(project),source=src,parent_spec=parent_spec,
        qualification=c.ref(c.PARENT/'qualification.json'),qualification_hash=q['content_hash'],
        qualification_task=c.ref(matching[0]),qualification_job=receipt['job'],
        requested_tasks={task:latest[task]['job'] for task in tasks},policy=c.policy())
    c.publish(root/'repair_spec.json',spec);return spec


def descendants(graph,roots):
    selected=set(roots)
    for task,deps in graph.items():
        if any(dep in selected for dep in deps):selected.add(task)
    return selected


def replacements(spec,old):
    latest={r['task']:r for r in old.campaign.accepted_jobs(old.spec)}
    c.require(set(old.contracts.tasks())<=set(latest),'Original campaign submission is incomplete')
    status=states(r['job'] for r in latest.values());closure=descendants(old.contracts.tasks(),spec['requested_tasks'])
    for task,job in spec['requested_tasks'].items():
        c.require(latest[task]['job']==job,'Recovery root already has a newer attempt; do not duplicate it')
        c.require(status[job][0] in TERMINAL and status[job]!=('COMPLETED','0:0'),
            'Requested recovery root is no longer a failed terminal job')
        fit,_,_,end=old.contracts.segment(task)
        committed=list((Path(old.spec['root'])/'fits'/fit/'epochs').glob('*.json'))
        c.require(len(committed)<=end,'A later segment already committed; preserve it for receipt-only inspection')
    tasks=[];cancel=[];preserved={}
    for task in old.contracts.tasks():
        row=latest[task];state=status[row['job']]
        if task not in closure:
            preserved[task]=row['job'];continue
        if task!='report' and state==('COMPLETED','0:0'):
            old.worker.task_receipt(old.spec,task);preserved[task]=row['job'];continue
        c.require(task=='report' or state[0] in TERMINAL or state[0]=='PENDING',
            'A descendant scientific segment is active; preserve it and inspect before recovery: '+task)
        tasks.append(task)
        if state[0]=='PENDING':cancel.append(row)
    # Retain all original edges, choosing new IDs only for the selected closure.
    rows=old.campaign.commands(old.spec,tasks,existing={t:r['job'] for t,r in latest.items() if t not in tasks})
    for row in rows:
        cut=old.campaign.script_index(row['command'])
        row['command']=row['command'][:cut]+wrapper(spec,row['task'])
    return dict(commands=rows,cancel=cancel,preserved=preserved,states={r['job']:status[r['job']] for r in latest.values()})


def wrapper(spec,task):
    return [str(Path(spec['project'])/'sbatch/run_luka_d20_input_recovery.sh'),
        spec['project'],str(Path(spec['root'])/'repair_spec.json'),task]


def bootstrap_commands(spec):
    result=[]
    for phase in ('probe','release'):
        args=['sbatch','--parsable','--no-requeue','--partition=compute','--account=default','--qos=normal',
            '--nodes=1','--ntasks=1','--cpus-per-task='+('4' if phase=='probe' else '1'),
            '--mem='+('32000M' if phase=='probe' else '4000M'),
            '--time='+('240' if phase=='probe' else '60'),'--export=NONE','--kill-on-invalid-dep=yes',
            '--job-name=lukadio_'+phase,'--comment='+spec['content_hash']+':'+phase,
            '--chdir='+spec['project'],'--output='+str(Path(spec['root'])/(phase+'-%j.out'))]
        if phase=='release':args+=['--dependency=afterok:{probe}']
        result.append(dict(task=phase,command=args+wrapper(spec,phase)))
    return result


def dry_review(rows):
    text=subprocess.check_output(['scontrol','show','config'],text=True,timeout=30)
    c.require(re.search(r'(?m)^\s*ClusterName\s*=\s*hydra\s*$',text),'Hydra required')
    for row in rows:
        command=row['command'];cut=next(i for i in range(1,len(command)) if not command[i].startswith('-'))
        script=Path(command[cut])
        c.require(script.is_absolute() and script.name=='run_luka_d20_input_recovery.sh'
            and script.read_bytes().splitlines(keepends=True)[0]==b'#!/usr/bin/env bash\n',
            'Submit the absolute Unix-shebang recovery script directly')
        review=['sbatch','--test-only',*[v for v in command[1:] if not v.startswith('--dependency=')]]
        subprocess.run(review,check=True,timeout=30,env=env())


def submit(spec,rows,folder,*,old=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);jobs={}
    maker=c.artifact if old is None else old.contracts.artifact
    parents=dict(repair=spec['content_hash']) if old is None else dict(spec=old.spec['content_hash'])
    plan=maker('PLAN',parents=parents,commands=rows,execution_recovery=spec['content_hash'])
    c.publish(folder/'plan.json',plan)
    for row in rows:
        command=list(row['command'])
        for task,job in jobs.items():command=[v.replace('{'+task+'}',job) for v in command]
        c.require(not any('{' in v or '}' in v for v in command),'Unresolved replacement dependency')
        fields=dict(parents=parents,task=row['task'],plan=plan['content_hash'],command=command,
            execution_recovery=spec['content_hash'],execution_spec=c.ref(Path(spec['root'])/'repair_spec.json'))
        c.publish(folder/(row['task']+'.intent.json'),maker('SUBMISSION_INTENT',**fields))
        result=subprocess.run(command,capture_output=True,text=True,timeout=30,env=env())
        c.publish(folder/(row['task']+'.response.json'),maker('SUBMISSION_RESPONSE',**fields,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        job=result.stdout.strip().split(';')[0]
        c.require(result.returncode==0 and re.fullmatch('[1-9][0-9]*',job),
            'Uncertain submission; inspect the durable intent/response before retry')
        c.publish(folder/(row['task']+'.submitted.json'),maker('SUBMITTED',job=job,**fields))
        jobs[row['task']]=job;print('SUBMITTED '+row['task']+': '+job,flush=True)
    return jobs


def queue(spec,*,execute=False,authorization=None):
    old=runtime.check(spec);qualified(spec,old);preview=replacements(spec,old);boot=bootstrap_commands(spec)
    print(__import__('json').dumps(dict(bootstrap=boot,recovery=preview),indent=2),flush=True)
    dry_review([*boot,*preview['commands']])
    if not execute:return preview
    c.require(authorization==c.AUTH,'Exact targeted I/O recovery authorization required')
    with old.campaign.submission_lock(old.spec['root']):
        claim=Path(spec['root'])/'submission'
        c.require(not claim.exists(),'Already queued or uncertain; inspect existing journals')
        replacements(spec,old)  # Recheck while holding the original writer lock.
        claim.mkdir();c.publish(claim/'claim.json',c.artifact('CLAIM',parents=dict(repair=spec['content_hash'])))
        # Existing jobs remain untouched until the CPU equivalence probe passes.
        return submit(spec,boot,claim/'bootstrap')


def bound(spec,phase):
    job=os.environ.get('SLURM_JOB_ID','');c.require(re.fullmatch('[1-9][0-9]*',job),'Registered Slurm allocation required')
    path=Path(spec['root'])/'submission/bootstrap'/(phase+'.submitted.json')
    for _ in range(30):
        if path.exists():break
        time.sleep(1)
    row=c.bootstrap_receipt(spec,phase)
    c.require(row['job']==job,'Bootstrap job binding differs')
    fields=dict(v.split('=',1) for v in subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30).split() if '=' in v)
    cpus,mem,minutes=(4,32000,240) if phase=='probe' else (1,4000,60)
    expected=dict(JobId=job,JobState='RUNNING',JobName='lukadio_'+phase,Comment=spec['content_hash']+':'+phase,
        WorkDir=spec['project'],Partition='compute',Account='default',QOS='normal',NumNodes='1',NumTasks='1')
    c.require(all(fields.get(k)==v for k,v in expected.items()) and fields.get('CPUs/Task')==str(cpus)
        and fields.get('NumCPUs') in (str(cpus),str(cpus*2)),'CPU recovery allocation differs')
    from hlt_classification.luka_fullstat_multiview.execution import allocation_minutes
    c.require(allocation_minutes(fields.get('TimeLimit',''))==minutes,'CPU recovery walltime differs')
    tres=dict(v.split('=',1) for v in fields.get('AllocTRES','').split(',') if '=' in v)
    value=re.fullmatch(r'([0-9.]+)([KMGT]?)',tres.get('mem',''))
    c.require(value is not None and Decimal(value[1])*{'':1,'M':1,'G':1024,'T':1024**2,'K':Decimal(1)/1024}[value[2]]==mem
        and not any(k.startswith('gres/gpu') for k in tres),'CPU recovery RAM/GPU allocation differs')
    return job


def cancel_pending(spec,old,rows):
    for row in rows:
        current=states([row['job']])[row['job']]
        if current[0] in TERMINAL:continue
        c.require(current[0]=='PENDING','Selected job became active; preserve it')
        fields=dict(v.split('=',1) for v in subprocess.check_output(['scontrol','show','job','-o',row['job']],text=True,timeout=30).split() if '=' in v)
        expected=dict(JobId=row['job'],JobState='PENDING',JobName='lukadfull_'+row['task'],
            Comment=old.spec['content_hash']+':'+row['task'],WorkDir=old.spec['project'])
        c.require(all(fields.get(k)==v for k,v in expected.items()),'Pending job binding changed; nothing cancelled')
        intent=c.artifact('CANCEL_PENDING',parents=dict(repair=spec['content_hash']),job=row['job'],task=row['task'])
        folder=Path(spec['root'])/'cancellations';c.publish(folder/(row['job']+'.intent.json'),intent)
        # The scheduler-side state filter also protects the race after inspection.
        subprocess.run(['scancel','--state=PENDING',row['job']],check=True,timeout=30)
        c.publish(folder/(row['job']+'.requested.json'),c.artifact('CANCEL_REQUESTED',parents=dict(intent=intent['content_hash'])))
        for _ in range(15):
            final=states([row['job']])[row['job']]
            if final[0] in TERMINAL:break
            c.require(final[0]=='PENDING','Selected job started; preserve it and stop recovery')
            time.sleep(1)
        c.require(final[0] in TERMINAL,'Cancellation not confirmed; stop before replacement submission')


def release(spec):
    old=runtime.check(spec);qualified(spec,old);bound(spec,'release');proof=runtime.parity(spec,old)
    c.require(states([proof['job']])[proof['job']]==('COMPLETED','0:0'),'Reader probe did not complete successfully')
    with old.campaign.submission_lock(old.spec['root']):
        claim=Path(spec['root'])/'release.intent.json'
        c.require(not claim.exists(),'Release already started or uncertain; inspect journals')
        plan=replacements(spec,old);dry_review(plan['commands'])
        c.publish(claim,c.artifact('RELEASE_INTENT',parents=dict(repair=spec['content_hash']),plan=plan))
        cancel_pending(spec,old,plan['cancel'])
        attempts=sorted((Path(old.spec['root'])/'submission').glob('attempt_*'))
        folder=Path(old.spec['root'])/'submission'/f'attempt_{len(attempts)+1:03d}'
        jobs=submit(spec,plan['commands'],folder,old=old)
        c.publish(Path(spec['root'])/'released.json',c.artifact('RELEASED',parents=dict(repair=spec['content_hash']),
            jobs=jobs,attempt=str(folder),preserved=plan['preserved']))
        return jobs
