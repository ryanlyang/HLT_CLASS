"""Read-only authenticated partial results and provisional epoch snapshots."""
from pathlib import Path
import re
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from .contracts import TASKS,selected_tasks,setting,validate,artifact
from . import inputs,reuse,training


def recovery(auc,lo,hi):
    if not hi>lo: raise ValueError('Invalid historical recovery anchors')
    return 100*(auc-lo)/(hi-lo)


def scheduler_states(spec):
    from .submission import accepted_jobs
    jobs=accepted_jobs(spec,'science')
    if not jobs: return {}
    run=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True)
    if run.returncode: raise RuntimeError('Accounting unavailable; do not infer job outcomes')
    by_id={}
    for line in run.stdout.splitlines():
        fields=line.split('|')
        if len(fields)>=3 and fields[0] in jobs.values(): by_id[fields[0]]=fields[1:3]
    return {t:by_id.get(j,['ACCOUNTING_PENDING','']) for t,j in jobs.items()}


def results(spec):
    from .worker import receipt
    bundle=inputs.Bundle(spec['bundle_root']); arrays={n:load_npz_arrays(bundle.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT','OFFLINE')}
    arrays['U100_SPORC']=load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    selected={}; hashes={}; incomplete=[]
    tasks=selected_tasks(spec)
    for task in tasks:
        if task in spec['reused']:
            proof=spec['reused'][task]; arrays[task]=load_npz_arrays(reuse.check(proof)); selected[task]=proof['report']['selected_pass']; hashes[task]=proof['content_hash']
        elif (Path(spec['root'])/task/'receipt.json').exists():
            receipt(spec,task); report,arrays[task]=training.validate_fit(Path(spec['root'])/task/'fit',spec,task)
            selected[task]=report['selected_pass']; hashes[task]=report['content_hash']
        else: incomplete.append(task)
    states=scheduler_states(spec); subset=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype!=np.int64 or subset.shape!=(50000,) or len(np.unique(subset))!=50000
            or subset.min()<0 or subset.max()>=spec['counts']['validation']): raise ValueError('Invalid overlapping report subset')
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],arrays['M0HLT'][k]) for k in ('identities','labels')): raise ValueError('Validation joins differ')
    rows=[]
    print({'group_B_only':'Group B','group_C_only':'Group C','direct_reconstruction_only':'Direct reconstruction'}.get(spec.get('scope'),'Group A')+': native HLT CE+U100-w5; training heads removed. Exploratory validation, no independent test.')
    for role,ix in (('validation',slice(None)),('original50k',subset)):
        metrics={n:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix]) for n,a in arrays.items()}
        lo=metrics['M0HLT']['macro_ovr_auc']; hi=metrics['OFFLINE']['macro_ovr_auc']
        print('\n'+role+' (original50k overlaps validation)' if role=='original50k' else '\n'+role)
        print(f'{"model":18} {"epoch":>5} {"AUC":>12} {"recovery%":>10} {"delta matched":>15}')
        for task,m in metrics.items():
            auc=m['macro_ovr_auc']; ctrl='C'+str(setting(task)['seed_index']) if task in tasks else None
            delta=None if ctrl not in metrics else auc-metrics[ctrl]['macro_ovr_auc']
            location=('historical SPORC' if task not in tasks else 'SPORC reused'
                if task in spec['reused'] and spec['reused'][task]['source_mode']=='transferred_original_SPORC_result' else 'Hydra')
            label=task+' [SPORC]' if location=='SPORC reused' else task
            description=''
            if spec.get('campaign_kind')=='R8_FOLLOWUP_V2' and task in tasks:
                from .reconstruction_followup import description as describe
                description='  '+describe(task)
            print(f'{label:18} {str(selected.get(task,"-")):>5} {auc:12.8f} {recovery(auc,lo,hi):10.1f} '+('--' if delta is None else f'{delta:+.8f}').rjust(15)+description)
            rows.append(dict(model=task,role=role,metrics=m,epoch=selected.get(task),recovery=recovery(auc,lo,hi),delta_matched=delta,reused=task in spec['reused'],location=location))
    for task in incomplete: print(task+': '+str(states.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**hashes),rows=rows,incomplete=incomplete,states=states)


def progress(spec):
    bundle=inputs.Bundle(spec['bundle_root']); anchors=bundle.json('origin/campaign_spec.json')['recovery_anchors']
    lo,hi=anchors['M0HLT'],anchors['OFFLINE']; rows={}; best={}
    tasks=selected_tasks(spec)
    for task in tasks:
        rows[task]=[]
        for path in sorted((Path(spec['root'])/task/'fit').glob('epoch_*.json')):
            r=load_json(path); validate(r,'EPOCH',dict(spec=spec['content_hash']))
            if r['task']!=task: raise ValueError('Epoch ownership differs')
            rows[task].append(r)
        if rows[task]: best[task]=max(rows[task],key=lambda r:r['validation']['macro_ovr_auc'])
        elif task in spec['reused']:
            report=spec['reused'][task]['report']; rows[task]=report['validation_history']; best[task]=rows[task][report['selected_pass']-1]
    print('Provisional epoch snapshots; historical anchors. No inference or writes. Original50k is report-only.')
    states=scheduler_states(spec)
    for task in tasks:
        ctrl='C'+str(setting(task)['seed_index']); control=best.get(ctrl)
        description=''
        if spec.get('campaign_kind')=='R8_FOLLOWUP_V2':
            from .reconstruction_followup import description as describe
            description=': '+describe(task)
        print('\n'+task+(' [reused]' if task in spec['reused'] else '')+description)
        if not rows[task]: print('  no published validation epoch; '+str(states.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
        for r in rows[task]:
            auc=r['validation']['macro_ovr_auc']; delta=None if control is None else auc-control['validation']['macro_ovr_auc']
            print(f'  epoch {r["pass_number"]:2d} AUC={auc:.8f} recovery={recovery(auc,lo,hi):5.1f}% delta_selected_{ctrl}='+('--' if delta is None else f'{delta:+.8f}'))
            if spec.get('scope')=='direct_reconstruction_only' and task not in spec['reused']:
                matched=next((c for c in rows.get(ctrl,[]) if c['pass_number']==r['pass_number']),None)
                print('    delta_same_epoch_'+ctrl+'='+('--' if matched is None else f'{auc-matched["validation"]["macro_ovr_auc"]:+.8f}'))
                if 'reconstruction_losses' in r: print('    reconstruction: '+str(r['reconstruction_losses']))
        if task in best:
            r=best[task]; print(f'  BEST epoch={r["pass_number"]} AUC={r["validation"]["macro_ovr_auc"]:.8f}')
        logs=list(Path(spec['root']).glob(task+'-*.out'))
        if len(logs)>1: raise ValueError('Multiple attempts: inspect exact submission journal')
        if logs:
            marker=None
            with logs[0].open(errors='replace') as f:
                for line in f:
                    if re.search(r'epoch=\d+/\d+ batch=\d+/\d+',line): marker=line.strip()
            if marker: print('  last logged marker: '+marker)
    return dict(best={t:dict(epoch=r['pass_number'],auc=r['validation']['macro_ovr_auc'],recovery=recovery(r['validation']['macro_ovr_auc'],lo,hi)) for t,r in best.items()},states=states)
