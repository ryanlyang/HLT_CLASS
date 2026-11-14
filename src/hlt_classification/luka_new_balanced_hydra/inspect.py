"""Saved in-range progress and selected-checkpoint recovery; no inference."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from . import contracts as c, campaign, artifacts, reporting


def progress(spec,stage):
    campaign.check(spec,live=False)
    if stage not in c.STAGES:raise ValueError('Unregistered stage')
    jobs=[r['job'] for r in campaign.accepted_jobs(spec) if r['task']=='train_'+stage]
    print('Stage:',stage,'; saved jobs:',', '.join(jobs) or 'not submitted')
    print('Same 500,000 in-range validation jets; selection excludes outside-range diagnostics.')
    print(f'{"Epoch":>5} {"Accuracy":>11} {"AUC":>12} {"Unique TRAIN":>13} {"Repeated %":>11}')
    seen=0
    for path in sorted((Path(spec['root'])/('train_'+stage)/'epochs').glob('*.json')):
        r=load_json(path);c.validate(r,'EPOCH_COMMIT')
        if r['parents']['spec']!=spec['content_hash'] or r['stage']!=stage or r['epoch']!=seen+1 or r['job_id'] not in jobs:
            raise ValueError('Saved epoch/job binding differs')
        seen+=1;v=r['row']['validation'];s=r['row']['sampling']
        print(f'{seen:5d} {v["accuracy"]:11.6f} {v["macro_ovr_auc"]:12.8f} {s["unique_jets"]:13,d} {100*s["repeated_draws"]/s["draws"]:11.1f}')
    if jobs:
        log=Path(spec['root'])/f'train_{stage}-{jobs[-1]}.out'
        if log.exists():
            lines=[x for x in log.read_text(errors='replace').splitlines() if x.startswith('BALANCED progress ')]
            if lines:print('Latest marker:',lines[-1])
    print(f'{seen}/20 epochs committed. Publication may follow; payloads/scheduler not reverified.')


def results(spec):
    campaign.check(spec,live=False);memo={}
    rows={s:artifacts.completed(spec,s,memo=memo) for s in c.STAGES}
    measurements={s:load_json(r['receipt']['files']['metrics']['path']) for s,r in rows.items()}
    for stage,m in measurements.items():c.validate(m,'METRICS',dict(report=rows[stage]['report']['content_hash']))
    out={}
    for s,m in measurements.items():
        recovered={}
        for name,p in m['populations'].items():
            lo=measurements['M0HLT']['populations'][name];hi=measurements['OFFLINE']['populations'][name]
            v={k:reporting.recovery(p['summary'].get(k),lo['summary'].get(k),hi['summary'].get(k)) for k in ('accuracy','macro_ovr_auc')}
            for eff in ('R50','R30'):
                v[eff]={cls:reporting.recovery(p[eff][cls].get('rejection'),lo[eff][cls].get('rejection'),hi[eff][cls].get('rejection')) for cls in c.CLASS_NAMES[1:]}
            recovered[name]=v
        out[s]=dict(selected_epoch=rows[s]['report']['selected_pass'],metrics=m['populations'],recovery_percent=recovered)
    return c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),models=out,
        anchors='new_balanced_M0HLT_OFFLINE_on_identical_subsets',checkpoint_reselection=False)
