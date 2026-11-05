"""Read-only saved results and provisional HLT-only epoch comparisons."""
from pathlib import Path
import math
import re
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from .contracts import TASKS,ARMS,artifact,validate
from . import campaign,training,worker


def epochs(spec,task):
    rows={}
    for path in sorted((Path(spec['root'])/task/'fit').glob('epoch_*.json')):
        row=load_json(path); validate(row,'EPOCH',dict(spec=spec['content_hash']))
        ep=row['pass_number']; auc=row['validation']['macro_ovr_auc']
        if (row['task']!=task or type(ep) is not int or not 1<=ep<=30 or ep in rows
                or path.name!=f'epoch_{ep:02d}.json' or row['validation']['rows']!=spec['counts']['validation']
                or not math.isfinite(auc) or not 0<=auc<=1): raise ValueError('Invalid published epoch')
        rows[ep]=auc
    return rows


def progress(spec):
    campaign.check(spec,check_storage=False); b=Bundle(spec['bundle_root'])
    anchors=b.json('origin/campaign_spec.json')['recovery_anchors']; lo=anchors['M0HLT']; hi=anchors['OFFLINE']
    if not hi>lo: raise ValueError('Invalid recovery anchors')
    ref={r['pass_number']:r['validation']['macro_ovr_auc'] for r in b.json('reference/matched/report.json')['validation_history']}
    histories={t:epochs(spec,t) for t in TASKS}
    def cell(v): return '--' if v is None else f'{v:.8f} / {100*(v-lo)/(hi-lo):6.1f}%'
    print(f'HLT-only full validation. Historical anchors M0HLT={lo:.8f}, OFFLINE={hi:.8f}.')
    print('Every fit includes U100 embedding weight5; controls have no particle loss. Epoch AUC/recovery, not running best.')
    for task,rows in histories.items():
        if not rows: continue
        ctrl='control_s'+str(ARMS[task]['seed_index'])
        print('\n'+task); print(f'{"epoch":>5} {"current":>25} {ctrl:>25} {"historical U100-w5":>25}')
        for ep,auc in rows.items(): print(f'{ep:5} {cell(auc):>25} {cell(histories[ctrl].get(ep)):>25} {cell(ref.get(ep)):>25}')
    print('\nBEST SO FAR (earliest exact AUC tie)')
    for task,rows in histories.items():
        if not rows: continue
        ep=max(rows,key=lambda e:(rows[e],-e)); print(f'{task:30} {len(rows):2}/30 ep={ep:2} {cell(rows[ep])}')
        logs=list(Path(spec['root']).glob(task+'-*.out'))
        if len(logs)>1: raise ValueError('Ambiguous task logs')
        if logs:
            markers=re.findall(r'PARTICLE_TEACHER '+re.escape(task)+r' epoch=(\d+)/30 batch=(\d+)/(\d+)',logs[0].read_text(errors='replace'))
            if markers: print('  Last emitted epoch/batch/total:',*markers[-1])
    print('Provisional read-only snapshot; no independent test, inference or job changes.')


def results(spec):
    parent=campaign.check(spec,check_storage=False); b=Bundle(spec['bundle_root'])
    arrays={n:load_npz_arrays(b.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT','OFFLINE')}
    arrays['U100_W5_SPORC']=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    arrays['U100_W5_A6000_REPLAY']=load_npz_arrays(Path(parent['root'])/'setup/control_precision/hydra_validation_predictions.npz')
    pending=[]; selected={}; hashes={}
    for task in TASKS:
        root=Path(spec['root'])/task
        if not (root/'receipt.json').exists(): pending.append(task); continue
        worker.receipt(spec,task); report,a=training.validate_fit(root/'fit',spec,task)
        arrays[task]=a; selected[task]=report['selected_pass']; hashes[task]=report['content_hash']
    subset=load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype!=np.int64 or subset.shape!=(spec['original50k_rows'],) or len(np.unique(subset))!=len(subset)
            or np.any(subset<0) or np.any(subset>=spec['counts']['validation'])): raise ValueError('Invalid overlapping report subset')
    metrics={}
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],arrays['M0HLT'][k]) for k in ('identities','labels')): raise ValueError('Population mismatch')
        metrics[name]={role:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])
            for role,ix in (('validation',slice(None)),('original50k',subset))}
    print('Native HLT fits with teacher_m16 particle targets; every arm includes U100 embedding weight5.')
    print('Historical SPORC recovery anchors. Original50k overlaps validation; no independent test.')
    rows=[]
    for role in ('validation','original50k'):
        lo=metrics['M0HLT'][role]['macro_ovr_auc']; hi=metrics['OFFLINE'][role]['macro_ovr_auc']
        if not hi>lo: raise ValueError('Invalid recovery anchors')
        print('\n'+role); print(f'{"model":30} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta matched":>15}')
        for name,roles in metrics.items():
            auc=roles[role]['macro_ovr_auc']; recovery=100*(auc-lo)/(hi-lo); delta=None
            if name in ARMS:
                ctrl='control_s'+str(ARMS[name]['seed_index'])
                if ctrl in metrics: delta=auc-metrics[ctrl][role]['macro_ovr_auc']
            print(f'{name:30} {str(selected.get(name,"-")):>3} {auc:12.8f} {recovery:10.1f} '+('--' if delta is None else f'{delta:+.8f}').rjust(15))
            rows.append(dict(model=name,role=role,epoch=selected.get(name),metrics=roles[role],recovery=recovery,delta_matched=delta))
    print('Pending/incomplete: '+(', '.join(pending) or 'none'))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**hashes),rows=rows,pending=pending)
