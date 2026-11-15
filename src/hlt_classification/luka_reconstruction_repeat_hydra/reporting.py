"""Authenticated selected results and fixed-window repeatability, without inference."""
from pathlib import Path
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact,cell,control,RECIPES,SEED_INDICES,LATE_EPOCHS,RECOVERY_THRESHOLD
from . import submission,training,identity


def states(spec):
    jobs=submission.accepted_jobs(spec,'science_pascal')
    if not jobs:return {}
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'-o','JobIDRaw,State,ExitCode'],
        check=True,capture_output=True,text=True)
    rows={line.split('|')[0]:line.split('|')[1:3] for line in r.stdout.splitlines() if len(line.split('|'))>=3}
    return {task:rows.get(job,['ACCOUNTING_PENDING','']) for task,job in jobs.items()}


def read_epochs(spec,task):
    rows=[]
    for e,path in enumerate(sorted((Path(spec['root'])/task/'fit').glob('epoch-*.json')),1):
        if e>30 or path.name!=f'epoch-{e:02}.json':raise ValueError('Missing or extra epochs')
        row=load_json(path);training.validate_epoch(spec,task,row,e,spec['counts'],4,False);rows.append(row)
    return rows


def selected(rows):return max(rows,key=lambda row:selection_key(row['validation'],row['update'])) if rows else None


def stability(rows,anchors):
    late=[r for r in rows if r['epoch'] in LATE_EPOCHS]
    values=[recovery_percent(r['validation']['macro_ovr_auc'],anchors) for r in late]
    if not np.isfinite(values).all():raise ValueError('Nonfinite recovery')
    complete=[r['epoch'] for r in late]==list(LATE_EPOCHS)
    longest=streak=0;previous=None
    for row,value in zip(late,values):
        streak=(streak if previous==row['epoch']-1 else 0)+1 if value>=RECOVERY_THRESHOLD else 0
        longest=max(longest,streak);previous=row['epoch']
    return dict(window=list(LATE_EPOCHS),observed_epochs=len(late),complete=complete,
        last_epoch=rows[-1]['epoch'] if rows else None,
        last_recovery=recovery_percent(rows[-1]['validation']['macro_ovr_auc'],anchors) if rows else None,
        mean_recovery=float(np.mean(values)) if complete else None,
        observed_mean_recovery=float(np.mean(values)) if values else None,
        threshold=RECOVERY_THRESHOLD,epochs_at_threshold=sum(v>=RECOVERY_THRESHOLD for v in values),
        longest_streak=longest)


def summarize(rows):
    summary={}
    for name in RECIPES:
        group=[r for r in rows if cell(r['task'])['recipe']==name]
        rec=[r['recovery'] for r in group];delta=[r['delta'] for r in group if r['delta'] is not None]
        late=[r['late']['mean_recovery'] for r in group if r['late']['complete']]
        complete=len(group)==len(SEED_INDICES)
        summary[name]=dict(completed_seeds=len(group),expected_seeds=len(SEED_INDICES),complete=complete,
            seeds=[cell(r['task'])['seed_index'] for r in group],
            mean_recovery=float(np.mean(rec)) if rec else None,minimum_recovery=min(rec) if rec else None,
            selected_seeds_at_threshold=sum(v>=RECOVERY_THRESHOLD for v in rec),
            all_three_selected_at_threshold=all(v>=RECOVERY_THRESHOLD for v in rec) if complete else None,
            matched_pairs=len(delta),mean_delta=float(np.mean(delta)) if delta else None,
            minimum_delta=min(delta) if delta else None,positive_pairs=sum(v>0 for v in delta),
            late_complete_seeds=len(late),mean_late_recovery=float(np.mean(late)) if late else None)
    return summary


def results(spec):
    from .worker import receipt
    reports={};initial={};status=states(spec);reference=None
    for task in spec['tasks']:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists():continue
        receipt(spec,task);r,arrays=training.validate_fit(folder/'fit',spec,task)
        if reference is None:reference=arrays
        elif any(not np.array_equal(reference[k],arrays[k]) for k in ('identities','labels')):
            raise ValueError('Completed fits have different evaluation populations')
        reports[task]=r;initial[task]=load_json(folder/'fit/initialization.json')
    identity.validate_pairs(spec,initial)
    print('18 reconstruction repeats: 4 GTX1080Ti FP32; CE + U100-w5 in every fit.')
    print('Exploratory validation; no independent test. Historical recovery anchors. Fresh same-seed controls.')
    print('Late window is epochs 21-30; counts/streaks use 52% recovery. Partial groups remain provisional.')
    print(f'{"task":8} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta ctrl":>12} {"last%":>7} {"late mean%":>11} {">=52":>6} {"streak":>7}')
    rows=[]
    for task in spec['tasks']:
        if task not in reports:
            print(f'{task:8} {status.get(task,["NOT_SUBMITTED_OR_UNCONFIRMED",""])}');continue
        r=reports[task];auc=r['validation']['macro_ovr_auc'];match=reports.get(control(task))
        delta=None if match is None else auc-match['validation']['macro_ovr_auc']
        rec=recovery_percent(auc,spec['recovery_anchors']);late=stability(r['history'],spec['recovery_anchors'])
        dt='--' if delta is None else f'{delta:+.8f}'
        print(f'{task:8} {r["selected_epoch"]:3} {auc:12.8f} {rec:7.1f} {dt:>12} '
            f'{late["last_recovery"]:7.1f} {late["mean_recovery"]:11.1f} {late["epochs_at_threshold"]:3}/10 {late["longest_streak"]:7}')
        rows.append(dict(task=task,epoch=r['selected_epoch'],auc=auc,recovery=rec,control=control(task),
            delta=delta,late=late,report=r['content_hash']))
    summary=summarize(rows)
    print('\nrecipe    seeds   mean rec%    min rec%    mean delta  paired gains   mean late%   selected >=52')
    def fmt(value,pattern):return '--' if value is None else format(value,pattern)
    for name,s in summary.items():
        print(f'{name:8} {s["completed_seeds"]}/3 {fmt(s["mean_recovery"],".1f"):>11} '
            f'{fmt(s["minimum_recovery"],".1f"):>11} {fmt(s["mean_delta"],"+.8f"):>13} '
            f'{s["positive_pairs"]}/{s["matched_pairs"]} {fmt(s["mean_late_recovery"],".1f"):>18} '
            f'{s["selected_seeds_at_threshold"]}/{s["completed_seeds"]}')
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,summary=summary,states=status,
        pending=[t for t in spec['tasks'] if t not in reports])


def progress(spec):
    epochs={t:read_epochs(spec,t) for t in spec['tasks']};best={t:selected(rows) for t,rows in epochs.items()};status=states(spec)
    print('Provisional saved validation; no inference or job changes. Late window: epochs 21-30.')
    for task,rows in epochs.items():
        print('\n'+task+' '+str(status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
        match=control(task);control_best=best[match];matched={r['epoch']:r for r in epochs[match]}
        for row in rows:
            auc=row['validation']['macro_ovr_auc'];rec=recovery_percent(auc,spec['recovery_anchors'])
            delta='--' if control_best is None else f'{auc-control_best["validation"]["macro_ovr_auc"]:+.8f}'
            same='--' if row['epoch'] not in matched else f'{auc-matched[row["epoch"]]["validation"]["macro_ovr_auc"]:+.8f}'
            print(f'  epoch {row["epoch"]:2} AUC={auc:.8f} recovery={rec:.1f}% delta_best_{match}={delta} same_epoch={same} '
                f'encoder_weight={row["encoder_weight"]:.4g} decoder_weight={row["decoder_weight"]:.4g}')
        if rows:
            r=best[task];print(f'  BEST epoch={r["epoch"]} AUC={r["validation"]["macro_ovr_auc"]:.8f} '
                f'recovery={recovery_percent(r["validation"]["macro_ovr_auc"],spec["recovery_anchors"]):.1f}%')
            print('  Late stability: '+str(stability(rows,spec['recovery_anchors'])))
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),best={t:None if r is None else
        dict(epoch=r['epoch'],auc=r['validation']['macro_ovr_auc'],recovery=recovery_percent(r['validation']['macro_ovr_auc'],spec['recovery_anchors']))
        for t,r in best.items()},states=status,late={t:stability(rows,spec['recovery_anchors']) for t,rows in epochs.items()})
