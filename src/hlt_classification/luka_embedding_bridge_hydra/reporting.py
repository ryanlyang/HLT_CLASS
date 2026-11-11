"""Read-only completed and live contrasts against the fresh FP32/DDP control."""
from pathlib import Path
import math
import subprocess
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from hlt_classification.luka_subclass_hydra.reporting import references as historical
from hlt_classification.luka_capacity_hydra.complementarity import align
from .contracts import artifact,validate
from .registry import FITS,HLT_FITS,ALIASES,cell
from . import submission,training


def states(spec):
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs:return {}
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows={s.split('|')[0]:s.split('|')[1:3] for s in r.stdout.splitlines() if len(s.split('|'))>=3}
    return {t:rows.get(j,['ACCOUNTING_PENDING','']) for t,j in jobs.items()}


def references(spec):
    b,a=historical(spec);a['C0_BF16_HISTORICAL']=a.pop('C0_U100_HYDRA')
    value=spec.get('historical_selective')
    if value:
        from .campaign import check_historical
        check_historical(value,spec)
        for task,files in value['files'].items():a[task+'_BF16_HISTORICAL']=load_npz_arrays(files['validation_predictions.npz']['path'])
    return b,a


def measure_cpu(spec):
    """Real predictions and real miniature receipts, with explicit extrapolation.

    No scientific fit exists yet. Full-size metric kernels use the frozen
    reference predictions; receipt/history checks use all 30 real miniatures.
    """
    started=time.perf_counter();bundle,arrays=references(spec);base=arrays['M0HLT']
    ix=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if len(base['labels'])!=spec['counts']['validation'] or ix.shape!=(spec['original50k_rows'],):
        raise ValueError('CPU measurement population differs')
    for a in arrays.values():
        a=align(base,a)
        for sel in (slice(None),ix):evaluate_probabilities(a['labels'][sel],a['probabilities'][sel])
    reference_seconds=time.perf_counter()-started
    elapsed=[]
    for sel in (slice(None),ix):
        started=time.perf_counter()
        for _ in range(3):evaluate_probabilities(base['labels'][sel],base['probabilities'][sel])
        elapsed.append((time.perf_counter()-started)/3)
    started=time.perf_counter()
    for task in FITS:training.validate_fit(Path(spec['root'])/'setup'/('mini_'+task),spec,task,technical=True)
    mini_seconds=time.perf_counter()-started
    # validate_fit: primary plus component metrics. results: full and subset.
    full_calls=sum(2+len(training.component_names(t)) for t in FITS)
    # 2 -> 30 epoch receipt/history scaling deliberately overcounts miniature
    # metric work, rather than pretending its small validation was full-size.
    estimate=reference_seconds+15*mini_seconds+full_calls*elapsed[0]+len(FITS)*elapsed[1]
    return dict(reference_seconds=reference_seconds,miniature_validation_seconds=mini_seconds,
        full_metric_seconds=elapsed[0],subset_metric_seconds=elapsed[1],full_metric_calls=full_calls,
        subset_metric_calls=len(FITS),report_estimate_seconds=estimate,method='real_metrics_and_miniature_receipts_extrapolated',
        train_rows=spec['counts']['train'],validation_rows=spec['counts']['validation'])


def ancestor_costs(costs):
    """Include every scientific producer once; shared setup is reported separately."""
    result={}
    for task in costs:
        chain=[];current=task
        while current is not None:
            chain.append(current);current=cell(current)['parent']
        result[task]=dict(costs[task],ancestors=chain[1:],
            total_chain_gpu_hours=sum(costs[t]['gpu_hours'] for t in chain) if all(t in costs for t in chain) else None)
    return result


def results(spec):
    from .worker import receipt,fit_parents
    bundle,arrays=references(spec);base=arrays['M0HLT'];ix=load_npz_arrays(bundle.path('reference/historical/subset.npz'))['indices']
    if len(base['labels'])!=spec['counts']['validation'] or ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix) or ix.min()<0 or ix.max()>=len(base['labels']):
        raise ValueError('Comparison population differs')
    metrics={};epochs={};components={};pending=[];costs={}
    def add(name,a):
        a=align(base,a);metrics[name]={role:evaluate_probabilities(a['labels'][sel],a['probabilities'][sel]) for role,sel in (('validation',slice(None)),('original50k',ix))}
    for n,a in arrays.items():add(n,a)
    del arrays
    for task in FITS:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists():pending.append(task);continue
        completed=receipt(spec,task);r,a=training.validate_fit(folder/'fit',spec,task)
        if r['parents']!=fit_parents(spec,task):raise ValueError('Completed optimization lineage differs')
        add(task,a);epochs[task]=r['selected_epoch'];components[task]=r['components']
        costs[task]=dict(worker_seconds=completed['execution_seconds'],gpu_hours=completed['execution_seconds']*completed['allocated_gpus']/3600)
        del a,r
    print('Embedding bridges, shared responses and regional targets. R0/P0 alias fresh FP32/DDP C0.')
    print('Single-seed exploratory validation; historical anchors; original50k overlaps validation; no independent test.')
    rows=[]
    for role in ('validation','original50k'):
        print('\n'+role);print(f'{"model":22} {"ep":>3} {"AUC":>12} {"recovery%":>10} {"delta fresh C0":>15} role')
        for name,m in metrics.items():
            auc=m[role]['macro_ovr_auc'];delta=auc-metrics['C0'][role]['macro_ovr_auc'] if 'C0' in metrics and name in HLT_FITS else None
            anchors={k:metrics[k][role]['macro_ovr_auc'] for k in ('M0HLT','OFFLINE')};rec=recovery_percent(auc,anchors)
            kind=cell(name)['role'] if name in FITS else 'historical reference'
            text='--' if delta is None else f'{delta:+.8f}'
            print(f'{name:22} {str(epochs.get(name,"-")):>3} {auc:12.8f} {rec:10.1f} {text:>15} {kind}')
            rows.append(dict(model=name,role=role,epoch=epochs.get(name),metrics=m[role],auc=auc,recovery=rec,delta_C0=delta,model_role=kind))
    status=states(spec)
    costs=ancestor_costs(costs)
    if costs:
        print('\nMeasured worker GPU-hours include authentication, startup, fitting and export; queue wait is excluded.')
        for task,cost in costs.items():
            chain='--' if cost['total_chain_gpu_hours'] is None else f'{cost["total_chain_gpu_hours"]:.2f}'
            print(f'{task:12} own={cost["gpu_hours"]:.2f} chain={chain}')
    setup_cost=None
    if (Path(spec['root'])/'setup/receipt.json').is_file():
        setup=receipt(spec,'setup',full=False);setup_cost=setup['execution_seconds']*setup['allocated_gpus']/3600
        print(f'Shared setup GPU-hours: {setup_cost:.2f} (not counted again for each chain).')
    for task in pending:print(task+': '+str(status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,components=components,pending=pending,states=status,aliases=ALIASES,
        costs=costs,setup_gpu_hours=setup_cost)


def read_epochs(spec,task):
    from .worker import fit_parents
    paths=sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json'))
    if not paths:return []
    parents=fit_parents(spec,task);rows=[];steps=math.ceil(spec['counts']['train']/128)
    for e,path in enumerate(paths,1):
        row=load_json(path);validate(row,'EPOCH',parents)
        if (e>30 or row['task']!=task or row['epoch']!=e or row['technical']is not False or row['update']!=e*steps
                or row['learning_rate']!=learning_rate(e) or row['validation']['rows']!=spec['counts']['validation']
                or set(row['weighted_losses'])!={'ce','embedding','extra','weighted_embedding','weighted_extra','total'}
                or not np.isfinite(list(row['weighted_losses'].values())).all()):raise ValueError('Live epoch meaning differs')
        training.validate_diagnostics(row);rows.append(row)
    return rows


def progress(spec,*,summary=False):
    epochs={t:read_epochs(spec,t) for t in FITS};best={};status=states(spec)
    for task,rows in epochs.items():
        if rows:
            row=max(rows,key=lambda r:training.selection(task,r));auc=row['validation']['macro_ovr_auc']
            best[task]=dict(epoch=row['epoch'],auc=auc,recovery=recovery_percent(auc,spec['recovery_anchors']),epochs=len(rows),role=cell(task)['role'])
    control={r['epoch']:r['validation']['macro_ovr_auc'] for r in epochs['C0']}
    print('Provisional logged validation; matched comparison is fresh C0. Assistants/teachers are separate oracle diagnostics.')
    for task,rows in epochs.items():
        if not rows:
            print(task+': '+str(status.get(task,['NO_PUBLISHED_EPOCH',''])));continue
        if not summary:
            print('\n'+task+' '+str(cell(task)))
            for row in rows:
                auc=row['validation']['macro_ovr_auc'];e=row['epoch'];rec=recovery_percent(auc,spec['recovery_anchors'])
                same='--' if e not in control or task not in HLT_FITS else f'{auc-control[e]:+.8f}'
                delta='--' if 'C0' not in best or task not in HLT_FITS else f'{auc-best["C0"]["auc"]:+.8f}'
                print(f'  epoch {e:2} AUC={auc:.8f} recovery={rec:.1f}% delta_C0_so_far={delta} delta_same_epoch={same} '
                      f'CE={row["weighted_losses"]["ce"]:.5f} U100={row["weighted_losses"]["weighted_embedding"]:.5f} new={row["weighted_losses"]["weighted_extra"]:.5f}')
        r=best[task];delta='--' if task not in HLT_FITS or 'C0'not in best else f'{r["auc"]-best["C0"]["auc"]:+.8f}'
        print(f'{task:12} {r["epochs"]:2}/30 selected_ep={r["epoch"]:2} AUC={r["auc"]:.8f} recovery={r["recovery"]:.1f}% delta_C0={delta} {r["role"]}')
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best=best,states=status,aliases=ALIASES)
