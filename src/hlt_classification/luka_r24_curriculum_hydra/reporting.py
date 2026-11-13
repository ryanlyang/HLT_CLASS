"""Authenticated completed results and provisional validation epochs; no inference."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import FITS,artifact,cell,control
from . import submission,training


def states(spec):
    jobs=submission.accepted_jobs(spec,'science')
    if not jobs:return {}
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],
        check=True,capture_output=True,text=True)
    rows={line.split('|')[0]:line.split('|')[1:3] for line in r.stdout.splitlines() if len(line.split('|'))>=3}
    return {task:rows.get(job,['ACCOUNTING_PENDING','']) for task,job in jobs.items()}


def read_epochs(spec,task):
    rows=[]
    for e,path in enumerate(sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json')),1):
        if e>30:raise ValueError('Extra epochs')
        row=load_json(path);training.validate_epoch(spec,task,row,e,spec['counts'],4,False);rows.append(row)
    return rows


def selected(rows):return max(rows,key=lambda row:selection_key(row['validation'],row['update'])) if rows else None


def results(spec):
    from .worker import receipt
    reports={};status=states(spec)
    for task in FITS:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists():continue
        receipt(spec,task);reports[task],_=training.validate_fit(folder/'fit',spec,task)
    print('Four GTX1080Ti FP32; every fit includes native CE + U100-w5. Historical recovery anchors.')
    print('Selected exploratory validation; no independent test. Delta uses the fresh same-seed control.')
    print(f'{"task":6} {"epoch":>5} {"AUC":>12} {"recovery%":>10} {"control":>8} {"delta AUC":>12}  curriculum')
    rows=[]
    for task in FITS:
        if task not in reports:
            print(f'{task:6} {status.get(task,["NOT_SUBMITTED_OR_UNCONFIRMED",""])}');continue
        r=reports[task];auc=r['validation']['macro_ovr_auc'];match=reports.get(control(task))
        delta=None if match is None else auc-match['validation']['macro_ovr_auc'];rec=recovery_percent(auc,spec['recovery_anchors'])
        text='--' if delta is None else f'{delta:+.8f}'
        print(f'{task:6} {r["selected_epoch"]:5} {auc:12.8f} {rec:10.1f} {control(task):>8} {text:>12}  {cell(task)}')
        rows.append(dict(task=task,epoch=r['selected_epoch'],auc=auc,recovery=rec,control=control(task),delta=delta,report=r['content_hash']))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,states=status,pending=[t for t in FITS if t not in reports])


def progress(spec):
    epochs={t:read_epochs(spec,t) for t in FITS};best={t:selected(rows) for t,rows in epochs.items()};status=states(spec)
    print('Provisional saved validation; no inference or job changes. Same-seed four-GPU controls.')
    for task,rows in epochs.items():
        print('\n'+task+' '+str(cell(task))+' '+str(status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
        match=control(task);control_best=best[match];matched={r['epoch']:r for r in epochs[match]}
        for row in rows:
            auc=row['validation']['macro_ovr_auc'];rec=recovery_percent(auc,spec['recovery_anchors'])
            delta='--' if control_best is None else f'{auc-control_best["validation"]["macro_ovr_auc"]:+.8f}'
            same='--' if row['epoch'] not in matched else f'{auc-matched[row["epoch"]]["validation"]["macro_ovr_auc"]:+.8f}'
            print(f'  epoch {row["epoch"]:2} AUC={auc:.8f} recovery={rec:.1f}% delta_best_{match}={delta} same_epoch={same} '
                f'encoder_weight={row["encoder_weight"]:.4g} decoder_weight={row["decoder_weight"]:.4g}')
        if rows:print(f'  BEST epoch={best[task]["epoch"]} AUC={best[task]["validation"]["macro_ovr_auc"]:.8f}')
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best={t:None if r is None else
        dict(epoch=r['epoch'],auc=r['validation']['macro_ovr_auc'],recovery=recovery_percent(r['validation']['macro_ovr_auc'],spec['recovery_anchors']))
        for t,r in best.items()},states=status)
