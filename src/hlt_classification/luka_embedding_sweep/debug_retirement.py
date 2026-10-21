"""Exact-ID retirement for the two campaigns explicitly selected by the user."""
from pathlib import Path
import getpass
import re
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
from . import debug as d

SWEEP_HASH = 'a377036840fa64f54d8a1540f758b2fd600a99394ec1e96c9d04c87661cece8d'
ALTERNATING_HASH = 'd234b34ca725af6d7525b5815af750a6aa3da6dc5fd074863e8e5d208e0676d6'
AUTHORIZATION = 'CANCEL_REVIEWED_LUKA_DEBUG_TRANSFER_JOBS_V1'
TERMINAL = {'COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def scheduler_environment():
    return {k:v for k,v in d.clean_environment().items() if not k.startswith('SCANCEL_')}


def bound_rows(spec, plan, jobs, *, sweep):
    if (set(jobs) != {r['task_id'] for r in plan['commands']}
            or len(set(jobs.values())) != len(jobs) or any(not str(j).isdigit() for j in jobs.values())):
        raise ValueError('Incomplete/ambiguous exact job binding')
    rows=[]
    for row in plan['commands']:
        cmd=row['command']; opts=dict(a[2:].split('=',1) for a in cmd if a.startswith('--') and '=' in a)
        job=str(jobs[row['task_id']])
        rows.append(dict(job_id=job,task_id=row['task_id'],sweep=sweep,spec_hash=spec['content_hash'],
            job_name=opts['job-name'],partition=opts['partition'],account=opts['account'],qos=opts['qos'],
            comment=opts['comment'],workdir=opts['chdir'],command=cmd[-4],
            stdout=opts['output'].replace('%j',job)))
    return rows


def alternating(path):
    from hlt_classification.luka_alternating_gate import campaign, submission
    from hlt_classification.luka_alternating_gate.contracts import validate
    path=Path(path).resolve(strict=True); spec=load_json(path)
    validate(spec,'SPEC')
    if spec['content_hash'] != ALTERNATING_HASH or path != Path(spec['root'])/'campaign_spec.json':
        raise PermissionError('Only the explicitly selected alternating campaign may be cancelled')
    gate=load_json(spec['gate_path'])
    campaign.read_gate(gate)
    validate_source_snapshot(gate['source_snapshot'],repository=gate['project_dir'],require_clean=True)
    report=load_json(Path(gate['root'])/'preflight/receipt.json')
    campaign.validate_spec(spec,gate,report)
    plan=submission.render(spec,gate)
    folder=Path(spec['root'])/'submission'
    events,jobs=load_exact_dag_journal(folder/'submission_ledger_journal',identity=spec['content_hash'],plan=plan)
    ledger=load_json(folder/'submission_ledger.json')
    if (ledger != assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'],events=events)
            or load_json(folder/'command_plan.json') != plan):
        raise ValueError('Alternating exact ledger/plan differs')
    return bound_rows(spec,plan,jobs,sweep=False),[file_ref(p) for p in
        (path,Path(spec['gate_path']),folder/'command_plan.json',folder/'submission_ledger.json')]


def plan(spec, alternating_path):
    if spec['parents']['original'] != SWEEP_HASH:
        raise PermissionError('Only the explicitly selected pending sweep may be transferred')
    original=load_json(spec['original_path'])
    rows=bound_rows(original,spec['original_plan'],spec['original_jobs'],sweep=True)
    others,refs=alternating(alternating_path)
    if {r['job_id'] for r in rows} & {r['job_id'] for r in others}:
        raise ValueError('Campaign job IDs overlap')
    return d.artifact('RETIRE_PLAN',parents=dict(spec=spec['content_hash']),
        authorization=AUTHORIZATION,user=getpass.getuser(),rows=rows+others,alternating_refs=refs,
        alternating_path=str(Path(alternating_path).resolve()),delete_files=False,
        running_alternating_work_will_be_lost=True)


def accounting(rows):
    ids=[r['job_id'] for r in rows]
    output=subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),'-o',
        'JobIDRaw,JobName%100,User,Account,Partition,State%40,ElapsedRaw,Start'],
        check=True,capture_output=True,text=True,timeout=60,env=d.clean_environment()).stdout
    result={}
    for line in output.splitlines():
        if not line.strip(): continue
        parts=line.strip().split('|')
        if len(parts)==9 and parts[-1]=='': parts.pop()
        if len(parts)!=8 or parts[0] not in ids or parts[0] in result:
            raise ValueError('Missing/ambiguous scheduler accounting; inspect before proceeding')
        job,name,user,account,partition,state,elapsed,start=parts
        if not elapsed.isdigit(): raise ValueError('Missing elapsed accounting')
        result[job]=dict(name=name,user=user,account=account,partition=partition,
            state=state.split()[0],elapsed=int(elapsed),start=start)
    if set(result)!=set(ids): raise ValueError('Missing exact job accounting; wait and inspect')
    return result


def check_account(row, state, user, *, retired=False):
    if (state['name'],state['user'],state['account'],state['partition']) != (
            row['job_name'],user,row['account'],row['partition']):
        raise PermissionError('Job accounting identity differs: '+row['job_id'])
    if row['sweep']:
        allowed={'CANCELLED'} if retired else {'PENDING','CANCELLED'}
        if state['state'] not in allowed or state['elapsed'] != 0 or state['start'] not in ('Unknown','N/A','None',''):
            raise ValueError('Original sweep has started or is not confirmed cancelled: '+row['job_id'])
    elif state['state'] not in TERMINAL | {'PENDING','RUNNING','SUSPENDED','CONFIGURING','COMPLETING'}:
        raise ValueError('Unexpected alternating state: '+state['state'])


def check_live(row, user):
    output=subprocess.run(['scontrol','show','job','-o',row['job_id']],check=True,
        capture_output=True,text=True,timeout=60,env=d.clean_environment()).stdout
    # Command/output paths in these plans contain no whitespace. Do not parse SubmitLine as a shell command.
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(\S*)',output)
    record=dict(pairs)
    expected=dict(JobId=row['job_id'],JobName=row['job_name'],Account=row['account'],QOS=row['qos'],
        Partition=row['partition'],Comment=row['comment'],WorkDir=row['workdir'],
        Command=row['command'],StdOut=row['stdout'],Requeue='0',Restarts='0')
    if (any(record.get(k)!=v for k,v in expected.items())
            or not re.fullmatch(re.escape(user)+r'\(\d+\)',record.get('UserId',''))):
        raise PermissionError('Live job identity differs; nothing authorized for this ID: '+row['job_id'])
    state=record.get('JobState')
    if row['sweep'] and (state!='PENDING' or record.get('RunTime')!='00:00:00'):
        raise ValueError('Original sweep started during review; stop and preserve progress')
    if not row['sweep'] and state not in {'PENDING','RUNNING','SUSPENDED','CONFIGURING','COMPLETING'}:
        raise ValueError('Alternating state changed during review; inspect and rerun dry review')


def inspect(reviewed):
    states=accounting(reviewed['rows'])
    for row in reviewed['rows']:
        state=states[row['job_id']]
        check_account(row,state,reviewed['user'])
        if state['state'] not in TERMINAL: check_live(row,reviewed['user'])
        print(row['job_id'],row['job_name'],state['state'],
              'preserve' if state['state'] in TERMINAL else 'CANCEL',flush=True)
    return states


def retire(spec, alternating_path, *, execute=False, plan_hash=None, authorization=None):
    d.check_spec(spec)
    reviewed=plan(spec,alternating_path)
    folder=Path(spec['root'])/'retirement'; folder.mkdir(exist_ok=True)
    saved=folder/'plan.json'
    if execute:
        if plan_hash!=reviewed['content_hash'] or authorization!=AUTHORIZATION:
            raise PermissionError('Exact reviewed retirement hash and cancellation authorization required')
        if not saved.is_file() or load_json(saved)!=reviewed:
            raise ValueError('Saved unchanged retirement dry review required')
    write_immutable_json(saved,reviewed)
    inspect(reviewed)  # Validate every target before the first mutation.
    if not execute: return reviewed
    (folder/'live_claim').mkdir(exist_ok=False)
    intent=d.artifact('RETIRE_INTENT',parents=dict(plan=reviewed['content_hash']),automatic_retry=False)
    write_immutable_json(folder/'intent.json',intent)
    requested=[]
    # Two bounded exact-ID requests, not a polling loop or a broad name/partition cancel.
    # --ctld applies the pending-only filter at the controller for the old sweep.
    for sweep in (True,False):
        rows=[row for row in reviewed['rows'] if row['sweep'] is sweep]
        states=accounting(rows)
        ids=[]
        for row in rows:
            state=states[row['job_id']]
            check_account(row,state,reviewed['user'])
            if state['state'] in TERMINAL: continue
            check_live(row,reviewed['user'])
            ids.append(row['job_id'])
        if not ids: continue
        command=['scancel','--ctld']+(['--state=PENDING'] if sweep else [])+ids
        subprocess.run(command,check=True,capture_output=True,text=True,
                       timeout=60,env=scheduler_environment())
        requested.extend(ids)
        for job in ids:
            write_immutable_json(folder/('cancel-'+job+'.json'),
                d.artifact('CANCEL_REQUEST',parents=dict(plan=reviewed['content_hash']),job_id=job))
    result=d.artifact('RETIRE_COMPLETE',parents=dict(plan=reviewed['content_hash'],intent=intent['content_hash']),
        cancellation_requested=requested,accounting_confirmation_required=True,files_deleted=False)
    write_immutable_json(folder/'receipt.json',result)
    return result


def confirm_retired(spec):
    folder=Path(spec['root'])/'retirement'
    reviewed=load_json(folder/'plan.json')
    d.validate(reviewed,'RETIRE_PLAN',dict(spec=spec['content_hash']))
    if reviewed!=plan(spec,reviewed['alternating_path']):
        raise ValueError('Retirement scope changed')
    intent=load_json(folder/'intent.json'); receipt=load_json(folder/'receipt.json')
    d.validate(intent,'RETIRE_INTENT',dict(plan=reviewed['content_hash']))
    d.validate(receipt,'RETIRE_COMPLETE',dict(plan=reviewed['content_hash'],intent=intent['content_hash']))
    requested=receipt['cancellation_requested']
    allowed={row['job_id'] for row in reviewed['rows']}
    if (receipt['files_deleted'] is not False or receipt['accounting_confirmation_required'] is not True
            or intent['automatic_retry'] is not False or len(requested)!=len(set(requested))
            or not set(requested)<=allowed):
        raise ValueError('Retirement receipt differs')
    for job in requested:
        event=load_json(folder/('cancel-'+job+'.json'))
        d.validate(event,'CANCEL_REQUEST',dict(plan=reviewed['content_hash']))
        if event['job_id']!=job: raise ValueError('Cancellation event identity differs')
    rows=[r for r in reviewed['rows'] if r['sweep']]
    states=accounting(rows)
    for row in rows: check_account(row,states[row['job_id']],reviewed['user'],retired=True)
    d.no_original_outputs(load_json(spec['original_path']))
    return states
