"""Fast saved metadata results; matched seed/readout anchors; no inference."""
from pathlib import Path
import statistics
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.luka_sixclass_balanced_oscar.reporting import recovery
from . import contracts as c,campaign,artifacts


def statistics_of(values):
    values=[v for v in values if v is not None]
    return dict(rows=len(values),mean=statistics.mean(values) if values else None,
        minimum=min(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None)


def results(spec):
    campaign.check(spec,live=False);models={}
    for task in c.FITS:
        if not (artifacts.fit_root(spec,task)/'result.json').exists():continue
        r=artifacts.completed(spec,task,payloads=False);models[task]={}
        for name in c.readouts(task):
            m=load_json(validate_file_ref(r['receipt']['files'][name+'_metrics']))
            c.validate(m,'METRICS',r['receipt']['fit_parents'])
            if m['selected']!=r['report']['selected'][name] or m['task']!=task or m['readout']!=name:raise ValueError('Selected metric binding differs')
            late=[row['readouts'][name]['macro_ovr_auc'] for row in r['report']['history']
                  if 16<=row['epoch']<=20 and name in row['readouts']]
            models[task][name]=dict(epoch=m['selected']['epoch'],metrics=m['measurements'],recovery=None,
                delta_to_control=None,delta_to_d20=None,late_epoch16_20=statistics_of(late))
    for task,streams in models.items():
        x=c.cell(task);lo=models.get(f'M0HLT_s{x["seed"]}');hi=models.get(f'OFFLINE_s{x["seed"]}')
        control=models.get(f'U100_W5_s{x["seed"]}' if x['reconstruction'] else f'M0HLT_s{x["seed"]}')
        for name,row in streams.items():
            if control and name in control:row['delta_to_control']=row['metrics']['summary']['macro_ovr_auc']-control[name]['metrics']['summary']['macro_ovr_auc']
            d20=models.get(f'D20_s{x["seed"]}')
            if d20 and name in d20:row['delta_to_d20']=row['metrics']['summary']['macro_ovr_auc']-d20[name]['metrics']['summary']['macro_ovr_auc']
            if not lo or not hi or name not in lo or name not in hi:continue
            a,b=lo[name]['metrics'],hi[name]['metrics'];m=row['metrics']
            recovered={key:recovery(m['summary'][key],a['summary'][key],b['summary'][key]) for key in ('accuracy','macro_ovr_auc')}
            for eff in ('R50','R30'):
                recovered[eff]={label:recovery(m[eff][label].get('rejection'),a[eff][label].get('rejection'),b[eff][label].get('rejection')) for label in c.CLASS_NAMES[1:]}
            row['recovery']=recovered
    aggregate={}
    for arm in c.ARMS:
        rows=[models[f'{arm}_s{s}']['RAW'] for s in c.seeds_for(arm) if f'{arm}_s{s}' in models]
        aggregate[arm]=dict(completed=len(rows),expected=len(c.seeds_for(arm)),
            auc=statistics_of([r['metrics']['summary']['macro_ovr_auc'] for r in rows]),
            accuracy=statistics_of([r['metrics']['summary']['accuracy'] for r in rows]),
            auc_recovery=statistics_of([r['recovery']['macro_ovr_auc'] if r['recovery'] else None for r in rows]),
            accuracy_recovery=statistics_of([r['recovery']['accuracy'] if r['recovery'] else None for r in rows]),
            paired_auc_delta=statistics_of([r['delta_to_control'] for r in rows]),
            paired_auc_delta_to_d20=statistics_of([r['delta_to_d20'] for r in rows]),
            rejections={eff:{label:dict(
                rejection=statistics_of([r['metrics'][eff][label].get('rejection') for r in rows]),
                recovery=statistics_of([r['recovery'][eff][label] if r['recovery'] else None for r in rows]))
                for label in c.CLASS_NAMES[1:]} for eff in ('R50','R30')})
    return c.artifact('RESULTS',parents=dict(spec=spec['content_hash']),models=models,primary='RAW',
        aggregate_primary=aggregate,missing=[t for t in c.FITS if t not in models],payload_verification='report_and_metrics_only')


def table(spec,readout='RAW',*,rejections=False):
    result=results(spec)
    print(f'Same {c.COUNTS["validation"]:,} natural in-range validation jets; readout={readout}; primary=RAW.')
    print('Matched same-seed/readout M0HLT/OFFLINE recovery. Metadata only; payloads/scheduler not reverified.')
    print(f'{"Model":18} {"Epoch":>5} {"Accuracy":>10} {"Acc rec.%":>10} {"AUC":>12} {"AUC rec.%":>10} {"vs control":>12}')
    for task in c.FITS:
        if task not in result['models']:continue
        if readout not in result['models'][task]:continue
        row=result['models'][task][readout];m=row['metrics']['summary'];r=row['recovery']
        a='-' if r is None or r['accuracy'] is None else f'{r["accuracy"]:.2f}'
        b='-' if r is None or r['macro_ovr_auc'] is None else f'{r["macro_ovr_auc"]:.2f}'
        delta='-' if row['delta_to_control'] is None else f'{row["delta_to_control"]:+.8f}'
        print(f'{task:18} {row["epoch"]:5d} {m["accuracy"]:10.6f} {a:>10} {m["macro_ovr_auc"]:12.8f} {b:>10} {delta:>12}')
    print('Missing:',', '.join(result['missing']) or 'none')
    if rejections:
        for eff in ('R50','R30'):
            print('\n'+eff+' at each AUC-selected checkpoint; censored = zero passing QCD.')
            print(f'{"Model":18} {"Class":8} {"Rejection":>11} {"QCD passing":>12} {"Recovery %":>11}')
            for task in c.FITS:
                if task not in result['models']:continue
                if readout not in result['models'][task]:continue
                row=result['models'][task][readout]
                for label,value in row['metrics'][eff].items():
                    number='censored' if value.get('rejection') is None else f'{value["rejection"]:.1f}'
                    r=row['recovery'][eff].get(label) if row['recovery'] else None
                    rec='-' if r is None else f'{r:.2f}'
                    print(f'{task:18} {label:8} {number:>11} {value["background_passing"]:12d} {rec:>11}')
    print('No inference, final-test access or file/job changes.')
    return result


def progress(spec,task,readout='RAW'):
    campaign.check(spec,live=False);c.cell(task)
    anchors={}
    for arm in ('M0HLT','OFFLINE'):
        t=f'{arm}_s{c.cell(task)["seed"]}'
        if (artifacts.fit_root(spec,t)/'result.json').exists():
            selected=artifacts.completed(spec,t,payloads=False)['report']['selected']
            if readout in selected:anchors[arm]=selected[readout]['validation']
    print(f'{task}: {readout}; same {c.COUNTS["validation"]:,} validation jets. Metadata only.')
    print(f'{"Epoch":>5} {"Accuracy":>10} {"AUC":>12} {"AUC rec.%":>10} {"Unique TRAIN":>13}')
    paths=sorted((artifacts.fit_root(spec,task)/'epochs').glob('*.json'))
    commits=[]
    if paths:
        first=load_json(paths[0]);parents=first['parents']
        if parents.get('spec')!=spec['content_hash'] or parents.get('population')!=spec['population']['content_hash']:
            raise ValueError('Progress population binding differs')
        _,commits=artifacts.history(spec,task,parents)
    for r in commits:
        row=r['row']
        if readout not in row['readouts']:continue
        m=row['readouts'][readout];v=None
        if len(anchors)==2:v=recovery(m['macro_ovr_auc'],anchors['M0HLT']['macro_ovr_auc'],anchors['OFFLINE']['macro_ovr_auc'])
        text='-' if v is None else f'{v:.2f}'
        print(f'{r["epoch"]:5d} {m["accuracy"]:10.6f} {m["macro_ovr_auc"]:12.8f} {text:>10} {row["sampling"]["unique_jets"]:13,d}')
    print('No inference, final-test access or file/job changes.')
