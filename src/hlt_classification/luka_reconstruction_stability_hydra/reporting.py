"""Authenticated selected results and fixed-window stability, without inference."""
from pathlib import Path
import subprocess
import numpy as np
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.runner import selection_key
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact,cell,control,RECIPES,SEED_INDICES,LATE_EPOCHS,RECOVERY_THRESHOLD,READOUTS
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


def for_readout(history,name):
    return [dict(r,validation=r['readouts'][name]) for r in history if name in r['readouts']]


def stability(rows,anchors):
    late=[r for r in rows if r['epoch'] in LATE_EPOCHS]
    values=[recovery_percent(r['validation']['macro_ovr_auc'],anchors) for r in late]
    complete=[r['epoch'] for r in late]==list(LATE_EPOCHS)
    thresholds={}
    for threshold in (50.,52.):
        longest=streak=0;previous=None
        for row,value in zip(late,values):
            streak=(streak if previous==row['epoch']-1 else 0)+1 if value>=threshold else 0
            longest=max(longest,streak);previous=row['epoch']
        thresholds[str(int(threshold))]=dict(count=sum(v>=threshold for v in values),longest_streak=longest)
    return dict(window=list(LATE_EPOCHS),observed_epochs=len(late),complete=complete,
        last_epoch=rows[-1]['epoch'] if rows else None,
        last_recovery=recovery_percent(rows[-1]['validation']['macro_ovr_auc'],anchors) if rows else None,
        mean_recovery=float(np.mean(values)) if complete else None,
        minimum_recovery=min(values) if complete else None,
        observed_mean_recovery=float(np.mean(values)) if values else None,thresholds=thresholds)


def statistics(values):
    return dict(mean=float(np.mean(values)) if values else None,minimum=min(values) if values else None,
        sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,count=len(values))


def summarize(rows):
    result={}
    for readout in READOUTS:
        result[readout]={}
        for name in RECIPES:
            group=[r for r in rows if r['readout']==readout and cell(r['task'])['recipe']==name]
            pairs=[r['delta'] for r in group if r['delta'] is not None]
            result[readout][name]=dict(completed_seeds=len(group),expected_seeds=3,complete=len(group)==3,
                seeds=[cell(r['task'])['seed_index'] for r in group],
                auc=statistics([r['auc'] for r in group]),recovery=statistics([r['recovery'] for r in group]),
                paired_delta=statistics(pairs),positive_pairs=sum(v>0 for v in pairs),
                r24_delta=statistics([r['delta_r24'] for r in group if r['delta_r24'] is not None]),
                selected_at_50=sum(r['recovery']>=50 for r in group),selected_at_52=sum(r['recovery']>=52 for r in group),
                late_mean=statistics([r['late']['mean_recovery'] for r in group if r['late']['complete']]))
    return result


def _requested(readout):
    if readout=='all':return READOUTS
    if readout not in READOUTS:raise ValueError('Unknown readout')
    return (readout,)


def fmt(value,pattern):return '--' if value is None else format(value,pattern)


def results(spec,readout='RAW'):
    from .worker import receipt
    requested=_requested(readout);reports={};initial={};status=states(spec);reference=None
    for task in spec['tasks']:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').exists():continue
        receipt(spec,task);report,arrays=training.validate_fit(folder/'fit',spec,task)
        if reference is None:reference=arrays
        elif any(not np.array_equal(reference[k],arrays[k]) for k in ('identities','labels')):raise ValueError('Different evaluation populations')
        reports[task]=report;initial[task]=load_json(folder/'fit/initialization.json')
    identity.validate_pairs(spec,initial)
    def auc(task,name):
        return None if task not in reports else reports[task]['selected_readouts'][name]['validation']['macro_ovr_auc']
    rows=[]
    for task,report in reports.items():
        for name in READOUTS:
            selected=report['selected_readouts'][name];value=auc(task,name);match=auc(control(task),name)
            r24=auc(f'R24_s{cell(task)["seed_index"]}',name)
            rows.append(dict(task=task,readout=name,epoch=selected['epoch'],auc=value,
                recovery=recovery_percent(value,spec['recovery_anchors']),control=control(task),
                delta=None if match is None else value-match,delta_r24=None if r24 is None else value-r24,
                delta_same_fit_raw_bn=value-auc(task,'RAW_BN'),late=stability(for_readout(report['history'],name),spec['recovery_anchors']),
                report=report['content_hash']))
    summary=summarize(rows)
    print('42 fits: 4 GTX1080Ti FP32, CE+U100-w5, three fresh paired seeds. RAW is primary.')
    print('Readouts share each training trajectory. Exploratory validation; no independent test.')
    for name in requested:
        print('\n'+name+' selected results; deltas use the same readout in matched controls.')
        print(f'{"task":12} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta ctrl":>12} {"delta R24":>12} {"late mean%":>11} {">=52":>7}')
        for r in rows:
            if r['readout']!=name:continue
            print(f'{r["task"]:12} {r["epoch"]:3} {r["auc"]:12.8f} {r["recovery"]:7.1f} {fmt(r["delta"],"+.8f"):>12} '
                f'{fmt(r["delta_r24"],"+.8f"):>12} {fmt(r["late"]["mean_recovery"],".1f"):>11} {r["late"]["thresholds"]["52"]["count"]:3}/10')
        print('recipe        seeds    mean%     min%      SD%    mean delta    positive pairs')
        for config,v in summary[name].items():
            print(f'{config:12} {v["completed_seeds"]}/3 {fmt(v["recovery"]["mean"],".1f"):>9} '
                f'{fmt(v["recovery"]["minimum"],".1f"):>8} {fmt(v["recovery"]["sample_sd"],".1f"):>8} '
                f'{fmt(v["paired_delta"]["mean"],"+.8f"):>13} {v["positive_pairs"]}/{v["paired_delta"]["count"]}')
    pending=[t for t in spec['tasks'] if t not in reports]
    for task in pending:print(task,status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED','']))
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),primary='RAW',rows=rows,summary=summary,states=status,pending=pending)


def progress(spec,readout='RAW'):
    requested=_requested(readout);epochs={t:read_epochs(spec,t) for t in spec['tasks']};status=states(spec)
    best={name:{t:selected(for_readout(rows,name)) for t,rows in epochs.items()} for name in READOUTS}
    print('Provisional full validation; RAW is primary. Saved artifacts only, no inference or job changes.')
    for name in requested:
        for task,history in epochs.items():
            rows=for_readout(history,name)
            print('\n'+task+' '+name+' '+str(status.get(task,['NOT_SUBMITTED_OR_UNCONFIRMED',''])))
            match=control(task);control_best=best[name][match];paired={r['epoch']:r for r in for_readout(epochs[match],name)}
            for row in rows:
                value=row['validation']['macro_ovr_auc'];delta=None if control_best is None else value-control_best['validation']['macro_ovr_auc']
                same=None if row['epoch'] not in paired else value-paired[row['epoch']]['validation']['macro_ovr_auc']
                print(f'  epoch {row["epoch"]:2} AUC={value:.8f} recovery={recovery_percent(value,spec["recovery_anchors"]):.1f}% '
                    f'delta_selected_{match}={fmt(delta,"+.8f")} same_epoch={fmt(same,"+.8f")}')
            if rows:
                b=best[name][task];print(f'  BEST epoch={b["epoch"]} recovery={recovery_percent(b["validation"]["macro_ovr_auc"],spec["recovery_anchors"]):.1f}%')
    return artifact('PROGRESS',parents=dict(spec=spec['content_hash']),primary='RAW',best=best,states=status,
        late={name:{t:stability(for_readout(rows,name),spec['recovery_anchors']) for t,rows in epochs.items()} for name in READOUTS})
