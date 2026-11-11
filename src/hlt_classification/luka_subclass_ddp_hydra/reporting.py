"""Fresh FP32-control contrasts and epoch-by-epoch recovery, without inference."""
from pathlib import Path
import math
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_selective_teacher_hydra.reporting import recovery
from hlt_classification.luka_capacity_hydra.complementarity import align
from hlt_classification.luka_subclass_hydra.reporting import references as historical
from .contracts import FITS,artifact,validate,term_names,cell
from . import inputs,submission,training


def references(spec):
    bundle,arrays=historical(spec)
    arrays['C0_BF16_HISTORICAL']=arrays.pop('C0_U100_HYDRA')
    return bundle,arrays


def states(spec):
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs:return {}
    run=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows={r.split('|')[0]:r.split('|')[1:3] for r in run.stdout.splitlines() if len(r.split('|'))>=3}
    return {task:rows.get(job,['ACCOUNTING_PENDING','']) for task,job in jobs.items()}


def results(spec):
    from .worker import receipt
    bundle,arrays=references(spec);epochs={};pending=[]
    for task in FITS:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists():pending.append(task);continue
        receipt(spec,task);r,arrays[task]=training.validate_fit(folder/'fit',spec,task)
        if r['parents']!=inputs.fit_parents(spec,task):raise ValueError('Fit target lineage differs')
        epochs[task]=r['selected_epoch']
    base=arrays['M0HLT'];arrays={k:align(base,a) for k,a in arrays.items()}
    ix=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if (len(base['labels'])!=spec['counts']['validation'] or ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],)
            or len(np.unique(ix))!=len(ix) or ix.min()<0 or ix.max()>=len(base['labels'])):raise ValueError('Evaluation population differs')
    print('Two-GTX1080Ti FP32: fresh U100-w5 C0 and ten subclass weights. Delta C0 uses this fresh control.')
    print('Historical recovery anchors; exploratory selected-checkpoint validation. Original50k overlaps; no independent test.')
    rows=[]
    for role,subset in (('validation',slice(None)),('original50k',ix)):
        metrics={k:evaluate_probabilities(a['labels'][subset],a['probabilities'][subset])['macro_ovr_auc'] for k,a in arrays.items()}
        print('\n'+role);print(f'{"model":21} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta fresh C0":>15}')
        for name,auc in metrics.items():
            delta=auc-metrics['C0'] if 'C0' in metrics else None
            rec=recovery(auc,metrics['M0HLT'],metrics['OFFLINE'])
            row=dict(model=name,role=role,epoch=epochs.get(name),auc=auc,recovery=rec,delta_C0=delta);rows.append(row)
            text='--' if delta is None else f'{delta:+.8f}'
            print(f'{name:21} {str(epochs.get(name,"-")):>3} {auc:12.8f} {rec:10.1f} {text:>15}')
    status=states(spec)
    for task in pending:print(task+': '+str(status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,states=status)


def read_epochs(spec,task):
    paths=sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json'));rows=[]
    if not paths:return rows
    parents=inputs.fit_parents(spec,task);steps=math.ceil(spec['counts']['train']/128)
    for e,path in enumerate(paths,1):
        row=load_json(path);validate(row,'EPOCH',parents)
        if (e>30 or row['task']!=task or row['epoch']!=e or row['technical'] is not False
                or row['update']!=e*steps or row['learning_rate']!=learning_rate(e)
                or row['validation']['rows']!=spec['counts']['validation'] or set(row['weighted_losses'])!=term_names(task)):
            raise ValueError('Epoch meaning differs')
        training.validate_diagnostics(row,task,steps);rows.append(row)
    return rows


def progress(spec):
    _,arrays=references(spec);base=arrays['M0HLT'];arrays={k:align(base,a) for k,a in arrays.items()}
    metrics={k:evaluate_probabilities(a['labels'],a['probabilities'])['macro_ovr_auc'] for k,a in arrays.items()}
    epochs={t:read_epochs(spec,t) for t in FITS};best={};status=states(spec)
    for task,rows in epochs.items():
        if rows:
            selected=max(rows,key=lambda r:selection_key(r['validation'],r['update']))
            best[task]=dict(epoch=selected['epoch'],auc=selected['validation']['macro_ovr_auc'])
    control={r['epoch']:r['validation']['macro_ovr_auc'] for r in epochs['C0']}
    print('Provisional FP32 validation epochs. Delta C0 is best fresh control so far; same-epoch contrast uses published C0 only.')
    for task,rows in epochs.items():
        print('\n'+task+' '+str(cell(task)))
        if not rows:print(status.get(task,['NO_PUBLISHED_EPOCH','']));continue
        for row in rows:
            auc=row['validation']['macro_ovr_auc'];e=row['epoch'];rec=recovery(auc,metrics['M0HLT'],metrics['OFFLINE'])
            delta='--' if 'C0' not in best else f'{auc-best["C0"]["auc"]:+.8f}'
            same='--' if e not in control else f'{auc-control[e]:+.8f}'
            print(f'  epoch {e:2} AUC={auc:.8f} recovery={rec:.1f}% delta_C0_so_far={delta} delta_same_epoch={same} '
                f'lambda={row["subclass_coefficient"]:g} weighted_KL={row["weighted_losses"]["weighted_subclass"]:.5f}')
        print('  BEST '+str(best[task]))
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best=best,states=status)
