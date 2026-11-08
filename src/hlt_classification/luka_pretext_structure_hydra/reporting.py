"""Read-only completed and provisional results; no pretext-oracle AUC fiction."""
from pathlib import Path
import math
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import learning_rate
from hlt_classification.luka_capacity_hydra.complementarity import align
from hlt_classification.luka_selective_teacher_hydra.reporting import recovery
from .contracts import FITS, TEACHERS, STUDENTS, artifact, validate, term_names
from . import inputs, submission, student_training, teacher_training


def status(spec):
    # Use this campaign's exact ledger, not the donor's source-bound plan.
    import subprocess
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs: return {}
    run=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows={s.split('|')[0]:s.split('|')[1:3] for s in run.stdout.splitlines() if len(s.split('|'))>=3}
    return {t:rows.get(j,['ACCOUNTING_PENDING','']) for t,j in jobs.items()}


def references(spec):
    b=inputs.Bundle(spec['bundle_root'])
    arrays={n:load_npz_arrays(b.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT','OFFLINE')}
    arrays['U100_SPORC']=load_npz_arrays(b.path('reference/matched/predictions.npz'))
    arrays['OLD_U100_CE_OFFLINE']=load_npz_arrays(spec['reused']['extra_control_predictions']['path'])
    return b,arrays


def teacher_results(spec):
    from .worker import validate_teacher
    rows=[]
    print('OFFLINE pretext teachers: selected by validation task loss; no jet-class CE and no classifier AUC.')
    for t in TEACHERS:
        if not (Path(spec['root'])/t/'receipt.json').exists(): continue
        r=validate_teacher(spec,t); row=dict(task=t,epoch=r['selected_epoch'],pretext_loss=r['selection_loss'])
        rows.append(row); print(f'{t:5} selected_epoch={row["epoch"]:2} pretext_loss={row["pretext_loss"]:.6f}')
    return rows


def results(spec):
    from .worker import receipt
    b,arrays=references(spec); epochs={}; pending=[]; teachers=teacher_results(spec)
    for t in STUDENTS:
        folder=Path(spec['root'])/t
        if not (folder/'receipt.json').exists(): pending.append(t); continue
        receipt(spec,t); r,arrays[t]=student_training.validate_fit(folder/'fit',spec,t)
        if r['parents']!=inputs.fit_parents(spec,t): raise ValueError('Completed fit target parents differ')
        epochs[t]=r['selected_epoch']
    base=arrays['M0HLT']; arrays={n:align(base,a) for n,a in arrays.items()}
    ix=load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if ix.dtype!=np.int64 or ix.shape!=(spec['original50k_rows'],) or len(np.unique(ix))!=len(ix) or ix.min()<0 or ix.max()>=len(base['labels']):
        raise ValueError('Invalid historical overlapping subset')
    rows=[]
    print('Native HLT students: CE + U100-w5, with registered pretext/structure supervision.')
    print('Single-seed exploratory validation. Original50k overlaps; no independent test.')
    for role,subset in (('validation',slice(None)),('original50k',ix)):
        metrics={n:evaluate_probabilities(a['labels'][subset],a['probabilities'][subset]) for n,a in arrays.items()}
        aucs={n:m['macro_ovr_auc'] for n,m in metrics.items()}
        print('\n'+role+'\nmodel                ep          AUC recovery%     delta C_U    delta C_CE')
        for n,auc in aucs.items():
            du=auc-aucs['C_U'] if 'C_U' in aucs else None; dc=auc-aucs['C_CE'] if 'C_CE' in aucs else None
            rec=recovery(auc,aucs['M0HLT'],aucs['OFFLINE']); fmt=lambda v:'--' if v is None else f'{v:+.8f}'
            print(f'{n:20} {str(epochs.get(n,"-")):>2} {auc:.8f} {rec:9.1f} {fmt(du):>13} {fmt(dc):>13}')
            rows.append(dict(model=n,role=role,epoch=epochs.get(n),auc=auc,recovery=rec,delta_C_U=du,delta_C_CE=dc,metrics=metrics[n]))
    done={r['task'] for r in teachers}; pending += [t for t in TEACHERS if t not in done]
    state=status(spec)
    for t in pending: print(t+': '+str(state.get(t,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,teachers=teachers,pending=pending,states=state)


def progress(spec):
    _,refs=references(spec); anchor={k:evaluate_probabilities(a['labels'],a['probabilities'])['macro_ovr_auc'] for k,a in refs.items()}
    best={}; print('Provisional published epochs; no inference, file writes or job changes.')
    for task in FITS:
        paths=sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json')); rows=[]
        if not paths: continue
        expected=inputs.teacher_parents(spec) if task in TEACHERS else inputs.fit_parents(spec,task)
        print(f'\n{task} ({len(paths)}/30 published epochs)')
        for e,path in enumerate(paths,1):
            row=load_json(path); validate(row,'TEACHER_EPOCH' if task in TEACHERS else 'EPOCH',expected)
            if (e>30 or row['epoch']!=e or row['task']!=task or row['technical'] is not False
                    or row['parents']['spec']!=spec['content_hash'] or row['update']!=e*math.ceil(spec['counts']['train']/128)):
                raise ValueError('Epoch meaning differs')
            if task in TEACHERS:
                if (set(row['validation_losses'])!={'primary','clean_signature'}
                        or not np.isfinite(list(row['validation_losses'].values())).all()
                        or row['selection_loss']!=sum(row['validation_losses'].values())
                        or row['learning_rate']!=learning_rate(e)):
                    raise ValueError('Teacher progress metrics differ')
            elif not np.isfinite(row['validation']['macro_ovr_auc']) or not 0<=row['validation']['macro_ovr_auc']<=1:
                raise ValueError('Invalid student progress AUC')
            rows.append(row)
            if task in TEACHERS: print(f' epoch {e:2} pretext_loss={row["selection_loss"]:.6f}')
            else:
                auc=row['validation']['macro_ovr_auc']; rec=recovery(auc,anchor['M0HLT'],anchor['OFFLINE'])
                print(f' epoch {e:2} AUC={auc:.8f} recovery={rec:.1f}% delta_U100={auc-anchor["U100_SPORC"]:+.8f}')
        if task in TEACHERS: chosen=min(rows,key=lambda r:(r['selection_loss'],r['epoch'])); metric=chosen['selection_loss']
        else: chosen=max(rows,key=lambda r:selection_key(r['validation'],r['update'])); metric=chosen['validation']['macro_ovr_auc']
        best[task]=dict(epoch=chosen['epoch'],metric=metric); print(' BEST '+str(best[task]))
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best=best,states=status(spec))
