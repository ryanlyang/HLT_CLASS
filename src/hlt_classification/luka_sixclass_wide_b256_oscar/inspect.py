"""Saved in-range progress and selected-checkpoint recovery; no inference."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from . import contracts as c, campaign, artifacts, reporting, training
from . import recovery as execution_recovery


def selected_metadata(spec,stage):
    spec=execution_recovery.origin(spec,stage)
    path=Path(spec['root'])/('train_'+stage)/'result.json'
    if not path.exists():return None
    receipt=load_json(path);c.validate(receipt,'RESULT')
    if receipt['stage']!=stage or receipt['parents']['spec']!=spec['content_hash']:
        raise ValueError('Selected reference binding differs')
    if not any(r['job']==receipt['job_id'] and r['task']=='train_'+stage for r in campaign.accepted_jobs(spec)):
        raise ValueError('Selected reference job differs')
    report=load_json(validate_file_ref(receipt['files']['report']))
    training.validate_report(report,stage,receipt['fit_parents'],c.COUNTS['validation'])
    if receipt['parents']['report']!=report['content_hash'] or receipt['selected_epoch']!=report['selected_pass']:
        raise ValueError('Selected reference report differs')
    return report


def progress(spec,stage):
    campaign.check(spec,live=False)
    spec=execution_recovery.origin(spec,stage)
    if stage not in c.STAGES:raise ValueError('Unregistered stage')
    records=sorted((r for r in campaign.accepted_jobs(spec) if c.task_stage(r['task'])==stage),key=lambda r:int(r['job']))
    jobs=[r['job'] for r in records]
    print('Stage:',stage,'; saved jobs:',', '.join(jobs) or 'not submitted')
    print(f"Same {c.COUNTS['validation']:,} six-class in-range validation jets; selection excludes outside-range diagnostics.")
    lo,hi=(selected_metadata(spec,s) for s in ('M0HLT','OFFLINE'))
    def recovery(v,key):
        n=reporting.recovery(v[key],lo['validation'][key],hi['validation'][key]) if lo and hi else None
        return f'{n:.2f}' if n is not None else '-'
    print('Recovery uses completed new balanced M0HLT/OFFLINE; unavailable anchors/gaps print a dash.')
    print(f'{"Epoch":>5} {"Accuracy":>11} {"Acc rec.%":>10} {"In AUC":>12} {"AUC rec.%":>10} {"Full AUC":>12} {"Full rec.%":>11} {"Full Hbb AUC":>13} {"Unique TRAIN":>13} {"Repeated %":>11}')
    seen=0
    for path in sorted((Path(spec['root'])/('train_'+stage)/'epochs').glob('*.json')):
        r=load_json(path);c.validate(r,'EPOCH_COMMIT')
        if r['parents']['spec']!=spec['content_hash'] or r['stage']!=stage or r['epoch']!=seen+1 or r['job_id'] not in jobs:
            raise ValueError('Saved epoch/job binding differs')
        first,last=c.epoch_window(spec,next(j['task'] for j in records if j['job']==r['job_id']))
        if not first<=r['epoch']<=last:raise ValueError('Epoch outside registered allocation window')
        seen+=1;training.check_row(r['row'],seen,stage,c.COUNTS['validation'])
        v=r['row']['validation'];s=r['row']['sampling']
        full=r['row']['full_validation'];rec=None
        if lo and hi:
            low=lo['validation_history'][lo['selected_pass']-1]['full_validation']['macro_ovr_auc']
            high=hi['validation_history'][hi['selected_pass']-1]['full_validation']['macro_ovr_auc']
            rec=reporting.recovery(full['macro_ovr_auc'],low,high)
        rec='n/a' if rec is None else f'{rec:.2f}'
        print(f'{seen:5d} {v["accuracy"]:11.6f} {recovery(v,"accuracy"):>10} {v["macro_ovr_auc"]:12.8f} {recovery(v,"macro_ovr_auc"):>10} {full["macro_ovr_auc"]:12.8f} {rec:>11} {r["row"]["full_hbb"]["auc"]:13.8f} {s["unique_jets"]:13,d} {100*s["repeated_draws"]/s["draws"]:11.1f}')
    if jobs:
        running=next((j for j in reversed(records) if (Path(spec['root'])/f'{j["task"]}-{j["job"]}.out').exists()),records[-1])
        log=Path(spec['root'])/f'{running["task"]}-{running["job"]}.out'
        if log.exists():
            lines=[x for x in log.read_text(errors='replace').splitlines() if x.startswith('BALANCED progress ')]
            if lines:print('Latest marker:',lines[-1])
    print(f'{seen}/20 epochs committed. Publication may follow; payloads/scheduler not reverified.')


def results(spec):
    campaign.check(spec,live=False);memo={}
    rows={s:completed_metadata(spec,s,memo) for s in c.STAGES
          if (Path(execution_recovery.origin(spec,s)['root'])/('train_'+s)/'result.json').exists()}
    if not rows:raise ValueError('No completed results yet; use progress')
    measurements={s:r['metrics'] for s,r in rows.items()}
    for stage,m in measurements.items():c.validate(m,'METRICS',dict(report=rows[stage]['report']['content_hash']))
    out={}
    for s,m in measurements.items():
        recovered={}
        for name,p in m['populations'].items():
            if 'M0HLT' not in measurements or 'OFFLINE' not in measurements:
                recovered[name]=None;continue
            lo=measurements['M0HLT']['populations'][name];hi=measurements['OFFLINE']['populations'][name]
            v={k:reporting.recovery(p['summary'].get(k),lo['summary'].get(k),hi['summary'].get(k)) for k in ('accuracy','macro_ovr_auc')}
            for eff in ('R50','R30'):
                v[eff]={cls:reporting.recovery(p[eff][cls].get('rejection'),lo[eff][cls].get('rejection'),hi[eff][cls].get('rejection')) for cls in c.CLASS_NAMES[1:]}
            recovered[name]=v
        out[s]=dict(selected_epoch=rows[s]['report']['selected_pass'],metrics=m['populations'],recovery_percent=recovered)
    return c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),models=out,
        anchors='new_balanced_M0HLT_OFFLINE_on_identical_subsets',checkpoint_reselection=False,
        payloads_reverified=False,
        unpublished_stages=[s for s in c.STAGES if s not in rows])


def completed_metadata(spec,stage,memo):
    if stage in memo:return memo[stage]
    spec=execution_recovery.origin(spec,stage)
    report=selected_metadata(spec,stage)
    if report is None:raise ValueError('Missing fresh same-campaign teacher')
    receipt=load_json(Path(spec['root'])/('train_'+stage)/'result.json')
    dependencies={t:completed_metadata(spec,t,memo) for t in c.TEACHERS[stage]}
    parents=artifacts.fit_parents(spec,stage,dependencies)
    if receipt['fit_parents']!=parents:raise ValueError('Teacher lineage differs')
    rows,_=artifacts.history(spec,stage,parents,states=False)
    if rows!=report['validation_history']:raise ValueError('Epoch history differs')
    metric=load_json(validate_file_ref(receipt['files']['metrics']))
    c.validate(metric,'METRICS',dict(report=report['content_hash']))
    selected=rows[report['selected_pass']-1]
    if (metric['populations']['in_range']['summary']!=selected['validation']
            or metric['populations']['full_validation']['summary']!=selected['full_validation']):
        raise ValueError('Selected metrics/history differ')
    memo[stage]=dict(receipt=receipt,report=report,metrics=metric)
    return memo[stage]


def print_results(result):
    rows=result['models']
    order=[s for s in ('M0HLT','OFFLINE','CONCAT','UE100','DE050','DE000') if s in rows]
    def fmt(v,n=2):return 'n/a' if v is None else f'{v:,.{n}f}'
    for pop,title in [('in_range','200 <= pT < 2600 GeV'),('full_validation','FULL VALIDATION'),('outside_range','OUTSIDE RANGE ONLY')]:
        count=rows[order[0]]['metrics'][pop]['summary']['rows']
        print(f'\n{title}: {count:,} jets; same in-range-selected checkpoints.')
        print(f'{"Model":<9} {"Epoch":>5} {"Accuracy":>11} {"Acc rec.%":>10} {"Macro AUC":>12} {"AUC rec.%":>10}')
        for s in order:
            r=rows[s];v=r['metrics'][pop]['summary'];rec=r['recovery_percent'][pop] or {}
            print(f'{s:<9} {r["selected_epoch"]:5d} {v["accuracy"]:11.6f} {fmt(rec.get("accuracy")):>10} {v["macro_ovr_auc"]:12.8f} {fmt(rec.get("macro_ovr_auc")):>10}')
        for efficiency in ('R50','R30'):
            print('\nQCD '+efficiency)
            print(f'{"Class":<8}'+''.join(f'{s:>12}' for s in order)+f'{"DE000 rec.%":>13}')
            for cls in c.CLASS_NAMES[1:]:
                values=[rows[s]['metrics'][pop][efficiency][cls] for s in order]
                cells=['censored' if v['status']=='censored' else fmt(v.get('rejection'),1) for v in values]
                rec=(rows.get('DE000',{}).get('recovery_percent',{}).get(pop) or {}).get(efficiency,{})
                print(f'{cls:<8}'+''.join(f'{v:>12}' for v in cells)+f'{fmt(rec.get(cls)):>13}')
    print('\nRecovery = 100*(model-M0HLT)/(OFFLINE-M0HLT); censored = zero QCD passing.')
    print('Saved metadata/report hashes and job bindings checked; binary payloads/full lineage/scheduler not reverified.')
    print('No inference, final-test access, file changes or job submission.')
