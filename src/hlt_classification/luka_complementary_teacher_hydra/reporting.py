"""Read-only completed and provisional results; S0 is the matched control."""
from pathlib import Path
import numpy as np
from hlt_classification.data.cache_contracts import load_json,load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_teacher_sweep_hydra.reporting import recovery
from .contracts import FITS,FOLDS,TEACHERS,STUDENTS,NAMES,artifact,validate,term_names
from . import inputs,submission,training,folds


def states(spec):
    import subprocess
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs: return {}
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows={x.split('|')[0]:x.split('|')[1:3] for x in r.stdout.splitlines() if len(x.split('|'))>=3}
    return {t:rows.get(j,['ACCOUNTING_PENDING','']) for t,j in jobs.items()}


def references(spec):
    b=inputs.Bundle(spec['bundle_root'])
    return b,{t:load_npz_arrays(b.path(f'reference/{t}/predictions.npz')) for t in ('M0HLT','OFFLINE')}


def results(spec):
    from .worker import receipt,validate_teacher,fit_parents
    b,arrays=references(spec);selected={};pending=[];preparation={}
    for t in FITS:
        folder=Path(spec['root'])/t
        if not (folder/'receipt.json').exists(): pending.append(t);continue
        receipt(spec,t);r,a=validate_teacher(spec,t) if t in TEACHERS else training.validate_fit(folder/'fit',spec,t)
        if r['parents']!=fit_parents(spec,t): raise ValueError('Completed fit parents differ')
        if t in FOLDS:
            m,_=folds.read_oof(spec,t);preparation[t]=dict(last_epoch=r['selected_epoch'],trained_rows=m['trained_rows'],held_rows=m['held_rows'])
        else: arrays[t]=a;selected[t]=r['selected_epoch']
    base=arrays['M0HLT']
    for a in arrays.values():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],base[k]) for k in ('identities','labels')): raise ValueError('Comparison population differs')
    subset=load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if subset.dtype!=np.int64 or subset.shape!=(spec['original50k_rows'],) or len(np.unique(subset))!=len(subset) or subset.min()<0 or subset.max()>=len(base['labels']):
        raise ValueError('Invalid overlapping historical subset')
    print('Three OFFLINE teachers; four HLT students with total embedding weight5. No U100 objective.')
    print('Single-seed development validation; original50k overlaps validation. No independent test.')
    rows=[]
    for role,ix in (('validation',slice(None)),('original50k',subset)):
        scores={n:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])['macro_ovr_auc'] for n,a in arrays.items()}
        print('\n'+role);print(f'{"model":12} {"epoch":>5} {"AUC":>12} {"recovery%":>10} {"delta S0":>12} role / recipe')
        for n,auc in scores.items():
            delta=auc-scores['S0'] if n in STUDENTS and 'S0' in scores else None;ds='--' if delta is None else f'{delta:+.8f}'
            rec=recovery(auc,scores['M0HLT'],scores['OFFLINE']);oracle=n in (*TEACHERS,'OFFLINE')
            print(f'{n:12} {str(selected.get(n,"-")):>5} {auc:12.8f} {rec:10.1f} {ds:>12} {"OFFLINE oracle" if oracle else "HLT"} / {NAMES.get(n,"historical reference")}')
            rows.append(dict(model=n,role=role,epoch=selected.get(n),auc=auc,recovery=rec,delta_s0=delta,oracle=oracle))
    print('OOF preparation (not development-validation comparisons): '+str(preparation));print('Pending/incomplete: '+(', '.join(pending) or 'none'))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,preparation=preparation,pending=pending,states=states(spec))


def progress(spec):
    from .worker import fit_parents
    best={};status=states(spec);lo,hi=(spec['recovery_anchors'][k] for k in ('M0HLT','OFFLINE'))
    print('Provisional logged epochs; no inference or job changes. F rows are TRAIN-only OOF preparation.')
    for t in FITS:
        paths=sorted((Path(spec['root'])/t/'fit').glob('epoch-*.json'));print('\n'+t+' '+NAMES[t])
        if not paths: print(status.get(t,['NO_PUBLISHED_EPOCH','']));continue
        parents=fit_parents(spec,t);rows=[]
        for e,p in enumerate(paths,1):
            r=load_json(p);validate(r,'EPOCH',parents)
            if r['task']!=t or r['epoch']!=e or r['technical'] is not False or set(r['weighted_losses'])!=term_names(t): raise ValueError('Epoch semantics differ')
            if t in FOLDS:
                if r['validation'] is not None: raise ValueError('OOF fit exposed validation')
                print(f'  epoch {e:2}: TRAIN loss {r["weighted_losses"]}; selection fixed at30')
            else:
                if r['validation']['rows']!=spec['counts']['validation']: raise ValueError('Progress validation rows differ')
                auc=r['validation']['macro_ovr_auc'];print(f'  epoch {e:2} AUC={auc:.8f} recovery={recovery(auc,lo,hi):.1f}% losses={r["weighted_losses"]}')
            rows.append(r)
        if t not in FOLDS:
            r=max(rows,key=lambda r:selection_key(r['validation'],r['update']));auc=r['validation']['macro_ovr_auc']
            best[t]=dict(epoch=r['epoch'],auc=auc,recovery=recovery(auc,lo,hi));print('  BEST '+str(best[t]))
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best=best,states=status)
