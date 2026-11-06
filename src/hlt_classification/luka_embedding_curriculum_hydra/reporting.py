"""Read-only partial results and per-spent-epoch curriculum progress."""
from pathlib import Path
import re
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_next_supervision_hydra.reuse import check as reuse_check
from .contracts import TASKS,ORDER,artifact,validate,setting
from . import training,submission,inputs


def states(spec):
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs: return {}
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows={x.split('|')[0]:x.split('|')[1:3] for x in r.stdout.splitlines() if len(x.split('|'))>=3}
    return {t:rows.get(j,['ACCOUNTING_PENDING','']) for t,j in jobs.items()}


def references(spec):
    bundle=inputs.Bundle(spec['bundle_root'])
    arrays={n:load_npz_arrays(bundle.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT','OFFLINE')}
    arrays['U100_SPORC']=load_npz_arrays(bundle.path('reference/matched/predictions.npz'))
    arrays['E00']=load_npz_arrays(reuse_check(spec['control'],imported=spec['imported']))
    return bundle,arrays


def recovery(auc,lo,hi):
    if not hi>lo: raise ValueError('Invalid recovery anchors')
    return 100*(auc-lo)/(hi-lo)


def results(spec):
    from .worker import receipt
    bundle,arrays=references(spec); selected={}; pending=[]
    for task in ORDER:
        folder=Path(spec['root'])/task
        if (folder/'receipt.json').exists():
            receipt(spec,task); r,arrays[task]=training.validate_fit(folder/'fit',spec,task); selected[task]=r['selected_pass']
        else: pending.append(task)
    base=arrays['M0HLT']
    for a in arrays.values():
        if any(not np.array_equal(a[k],base[k]) for k in ('identities','labels')): raise ValueError('Results populations differ')
    subset=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if subset.dtype!=np.int64 or subset.shape!=(50000,) or len(np.unique(subset))!=50000 or subset.min()<0 or subset.max()>=len(base['labels']):
        raise ValueError('Invalid overlapping report subset')
    selected['E00']=spec['control']['report']['selected_pass']; rows=[]
    print('Curriculum: 20 distinct rows, one seed; E00 reused. HLT-only deployment. Exploratory validation.')
    for role,ix in (('validation',slice(None)),('original50k (overlaps validation)',subset)):
        m={t:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])['macro_ovr_auc'] for t,a in arrays.items()}
        print('\n'+role); print(f'{"model":16} {"epoch":>5} {"AUC":>12} {"recovery%":>10} {"delta E00":>13}')
        for task,auc in m.items():
            rec=recovery(auc,m['M0HLT'],m['OFFLINE']); delta=auc-m['E00']
            print(f'{task:16} {str(selected.get(task,"-")):>5} {auc:12.8f} {rec:10.1f} {delta:+13.8f}')
            rows.append(dict(task=task,role=role,auc=auc,recovery=rec,delta_control=delta))
    statuses=states(spec)
    for task in pending: print(task+': '+str(statuses.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,states=statuses)


def progress(spec):
    _,arrays=references(spec)
    m={n:evaluate_probabilities(a['labels'],a['probabilities'])['macro_ovr_auc'] for n,a in arrays.items()}
    control_history=spec['control']['report']['validation_history']; statuses=states(spec); best={}
    print(f'Full validation: M0HLT={m["M0HLT"]:.8f}, OFFLINE={m["OFFLINE"]:.8f}, selected E00={m["E00"]:.8f}')
    print('PRETRAIN classification is diagnostic; only eligible classification epochs select the deployed model.')
    for task in ORDER:
        folder=Path(spec['root'])/task/'fit'; rows=[]; print('\n'+task+' '+str(setting(task)))
        for path in sorted(folder.glob('epoch_*.json')):
            row=load_json(path); validate(row,'EPOCH',dict(spec=spec['content_hash']))
            if row['task']!=task: raise ValueError('Epoch ownership differs')
            rows.append(row); auc=row['validation']['macro_ovr_auc']; epoch=row['pass_number']
            same=next((r['validation']['macro_ovr_auc'] for r in control_history if r['pass_number']==epoch),None)
            print(f'  epoch={epoch:2} {row["phase"]:14} AUC={auc:.8f} recovery={recovery(auc,m["M0HLT"],m["OFFLINE"]):5.1f}% '
                f'delta_E00={auc-m["E00"]:+.8f} val_embed={row["embedding_validation_loss"]:.6f} '
                f'CE_w={row["ce_weight_end"]:.3f} emb_w={row["embedding_weight_end"]:.3f}'
                +('' if same is None else f' delta_same_epoch={auc-same:+.8f}'))
        eligible=[r for r in rows if r['phase']=='classification']
        if eligible:
            row=max(eligible,key=lambda r:selection_key(r['validation'],r['update'])); auc=row['validation']['macro_ovr_auc']
            best[task]=dict(epoch=row['pass_number'],auc=auc,recovery=recovery(auc,m['M0HLT'],m['OFFLINE']))
            print('  BEST eligible: '+str(best[task]))
        if not rows: print('  No published epoch; '+str(statuses.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
        if (folder/'transition.json').exists():
            tr=load_json(folder/'transition.json'); validate(tr,'TRANSITION')
            if tr['task']!=task or tr['parents']['spec']!=spec['content_hash']: raise ValueError('Transition ownership differs')
            print(f'  Transition used epoch {tr["selected_epoch"]}; val_embed={tr["embedding_validation_loss"]:.6f}')
        logs=list(Path(spec['root']).glob(task+'-*.out'))
        if len(logs)>1: raise ValueError('Multiple attempts; inspect exact submission journals')
        marker=None
        if logs:
            with logs[0].open(errors='replace') as f:
                for line in f:
                    if re.search(r'epoch=\d+/30 batch=\d+/\d+',line): marker=line.strip()
        if marker: print('  Last logged: '+marker)
    return dict(best=best,states=statuses)
