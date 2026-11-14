"""Small saved-metadata progress view; completed comparisons use fresh anchors."""
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as meta
from . import contracts as c,submission,artifacts,reporting,campaign


def progress(spec,stage):
    campaign.check(spec,live_source=False)
    if stage not in c.STAGES:raise ValueError('Unregistered stage')
    if stage in spec['completed']:
        return progress(meta.load_json(meta.validate_file_ref(spec['completed'][stage]['spec'])),stage)
    jobs=submission.accepted_jobs(spec,partial=True);task='train_'+stage
    print('Saved job binding:',jobs.get(task,'not submitted'),'=',task)
    print('Original natural 500,000 validation jets; per-epoch selected-native metrics.')
    print(f'{"Epoch":>5} {"Accuracy":>11} {"AUC":>12} {"Unique TRAIN":>13} {"Repeated %":>11}')
    inherited=spec['continuations'].get(stage,{}).get('epochs',[])
    reports=[]
    for ref in inherited:
        receipt=meta.load_json(meta.validate_file_ref(ref));reports.append(meta.validate_file_ref(receipt['report']))
    folder=Path(spec['root'])/task
    for epoch in range(len(inherited)+1,c.EPOCHS+1):
        path=folder/'epochs'/f'{epoch:02d}'/'receipt.json'
        if not path.exists():break
        receipt=meta.load_json(path);c.validate(receipt,'EPOCH_COMMIT')
        reports.append(meta.validate_file_ref(receipt['report']))
    for path in reports:
        e=meta.load_json(path);c.validate(e,'EPOCH');row=e['row'];v=row['validation'];s=row['sampling']
        print(f'{e["epoch"]:5d} {v["accuracy"]:11.6f} {v["macro_ovr_auc"]:12.8f} {s["unique_jets"]:13,d} {100*s["repeated_draws"]/s["draws"]:11.1f}')
    log=Path(spec['root'])/(task+'-'+str(jobs.get(task,''))+'.out')
    if log.exists():
        markers=[x for x in log.read_text(errors='replace').splitlines() if x.startswith('BALANCED progress ')]
        if markers:print('Latest marker (not a live countdown):',markers[-1])
    print(f'Completed validation epochs: {len(reports)}/20. Final publication may follow.')
    print('Metadata only; no inference, test access, or file/job changes. Payloads and live scheduler state not reverified.')


def results(spec):
    campaign.check(spec,live_source=False)
    rows={s:artifacts.completed(spec,s) for s in c.STAGES}
    metrics={s:meta.load_json(r['receipt']['files']['metrics']['path']) for s,r in rows.items()}
    for s,m in metrics.items():c.validate(m,'METRICS',dict(report=rows[s]['report']['content_hash']))
    out={}
    for s,m in metrics.items():
        recovery={}
        for name,p in m['populations'].items():
            low=metrics['M0HLT']['populations'][name];high=metrics['OFFLINE']['populations'][name]
            v={k:reporting.recovery(p['summary'].get(k),low['summary'].get(k),high['summary'].get(k))
               for k in ('accuracy','macro_ovr_auc')}
            for eff in ('R50','R30'):
                v[eff]={cls:reporting.recovery(p[eff][cls].get('rejection'),low[eff][cls].get('rejection'),high[eff][cls].get('rejection'))
                        for cls in c.CLASS_NAMES[1:]}
            recovery[name]=v
        out[s]=dict(selected_epoch=rows[s]['report']['selected_pass'],metrics=m['populations'],recovery_percent=recovery)
    return c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),models=out,
        anchors='fresh_balanced_M0HLT_and_OFFLINE_per_population',negative_or_zero_anchor_gap='unresolved',
        inference=False,checkpoint_reselection=False)
