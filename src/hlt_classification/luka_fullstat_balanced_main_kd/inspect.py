"""Fast saved-metadata views with fresh balanced recovery anchors."""
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_fullstat_balanced_ladder import contracts as old,reporting
from hlt_classification.jetclass2_delphes.runner import selection_key
from . import contracts as c,campaign,submission,source,worker,training


def reference(spec,stage):
    refs=spec['reuse']['references'][stage]['files']
    report=meta.load_json(meta.validate_file_ref(refs['report']));old.validate(report,'TRAINING_REPORT')
    metrics=meta.load_json(meta.validate_file_ref(refs['metrics']));old.validate(metrics,'METRICS',dict(report=report['content_hash']))
    receipt=meta.load_json(meta.validate_file_ref(spec['reuse']['references'][stage]['receipt']))
    if (report['stage']!=stage or report['content_hash']!=receipt['parents']['report']
            or report['selected_pass']!=receipt['selected_pass']
            or metrics['populations']['natural']['summary']!=report['validation']):
        raise ValueError('Reference metrics binding differs')
    return report,metrics


def pct(value,lo,hi):
    v=reporting.recovery(value,lo,hi)
    return 'n/a' if v is None else f'{v:.2f}'


def progress(spec,stage='DE000'):
    campaign.check(spec,live_source=False)
    if stage!='DE000':raise ValueError('Only DE000 is registered')
    low,_=reference(spec,'M0HLT');high,_=reference(spec,'OFFLINE')
    lo,hi=low['validation'],high['validation']
    jobs=submission.accepted_jobs(spec,partial=True);job=jobs.get('train_DE000')
    print('Saved job binding:',job or 'not submitted','= train_DE000')
    print('Original natural 500,000 validation jets; fresh balanced anchors.')
    print(f'M0HLT AUC {lo["macro_ovr_auc"]:.8f}; OFFLINE AUC {hi["macro_ovr_auc"]:.8f}.')
    parent=source.check(spec['reuse'],spec['project_dir'],source_check=False)
    oldpath=Path(parent['root'])/'train_DE000/task_receipt.json';control={}
    if oldpath.is_file():
        receipt=meta.load_json(oldpath);old.validate(receipt,'TASK_RECEIPT')
        report=meta.load_json(meta.validate_file_ref(receipt['files']['report']));old.validate(report,'TRAINING_REPORT')
        if receipt['parents']!=dict(spec=parent['content_hash'],report=report['content_hash']) or report['stage']!='DE000':
            raise ValueError('Original DE000 reference binding differs')
        control={r['pass_number']:r['validation']['macro_ovr_auc'] for r in report['validation_history']}
    folder=Path(spec['root'])/'train_DE000';continuation=spec['continuations'].get(stage,{})
    refs=list(continuation.get('epochs',[]));eps=list(continuation.get('epoch_parents',[]))
    parents=worker.fit_parents(spec,stage,worker.dependencies_for(spec,parent))
    for epoch in range(len(refs)+1,c.EPOCHS+1):
        path=folder/'epochs'/f'{epoch:02d}'/'receipt.json'
        if not path.exists():break
        refs.append(meta.file_ref(path));eps.append(parents)
    print(f'{"Epoch":>5} {"Previous AUC":>13} {"New AUC":>12} {"AUC rec.%":>10} {"Accuracy":>10} {"Acc rec.%":>10}')
    history=[]
    for epoch,(ref,ep) in enumerate(zip(refs,eps,strict=True),1):
        receipt=meta.load_json(meta.validate_file_ref(ref));c.validate(receipt,'EPOCH_COMMIT',ep)
        report=meta.load_json(meta.validate_file_ref(receipt['report']));c.validate(report,'EPOCH',ep)
        if report['epoch']!=epoch or receipt['epoch']!=epoch:raise ValueError('Epoch binding differs')
        row=report['row'];training.check_row(row,epoch,stage,c.COUNTS['validation']);history.append(row)
        v=row['validation'];previous=f'{control[epoch]:.8f}' if epoch in control else '-'
        print(f'{epoch:5d} {previous:>13} {v["macro_ovr_auc"]:12.8f} '
              f'{pct(v["macro_ovr_auc"],lo["macro_ovr_auc"],hi["macro_ovr_auc"]):>10} '
              f'{v["accuracy"]:10.6f} {pct(v["accuracy"],lo["accuracy"],hi["accuracy"]):>10}')
    print(f'Completed validation epochs: {len(history)}/{c.EPOCHS}')
    if history:
        best=max(history,key=lambda r:selection_key(r['validation'],r['update']));v=best['validation']
        print(f'Best epoch {best["pass_number"]}: AUC {v["macro_ovr_auc"]:.8f}; recovery {pct(v["macro_ovr_auc"],lo["macro_ovr_auc"],hi["macro_ovr_auc"])}%.')
    log=Path(spec['root'])/f'train_DE000-{job}.out'
    if log.exists():
        lines=[x for x in log.read_text(errors='replace').splitlines() if x.startswith('BALANCED_MAIN_KD progress ')]
        if lines:print('Latest marker (training only; not live countdown):',lines[-1])
    print('Provisional epoch reports; selected publication may follow. Metadata/report and saved bindings checked.')
    print('Payloads/full lineage/live scheduler not replayed. No inference, test access or file/job changes.')


def results(spec):
    campaign.check(spec,live_source=False)
    folder=Path(spec['root'])/'train_DE000'
    receipt=meta.load_json(folder/'task_receipt.json');c.validate(receipt,'TASK_RECEIPT')
    report=meta.load_json(meta.validate_file_ref(receipt['files']['report']));c.validate(report,'TRAINING_REPORT')
    measured=meta.load_json(meta.validate_file_ref(receipt['files']['metrics']));c.validate(measured,'METRICS',dict(report=report['content_hash']))
    parent=source.check(spec['reuse'],spec['project_dir'],source_check=False)
    parents=worker.fit_parents(spec,'DE000',worker.dependencies_for(spec,parent))
    training.validate_report(report,'DE000',parents,c.COUNTS['validation'])
    if (receipt['parents']!=dict(spec=spec['content_hash'],report=report['content_hash'])
            or receipt['stage']!='DE000' or receipt['fit_parents']!=parents
            or receipt['selected_pass']!=report['selected_pass']
            or measured['checkpoint_reselection'] is not False
            or receipt['job_id']!=submission.accepted_jobs(spec)['train_DE000']
            or measured['populations']['natural']['summary']!=report['validation']):
        raise ValueError('Published result binding differs')
    references={stage:reference(spec,stage) for stage in c.REFERENCES};out={}
    for name,pop in measured['populations'].items():
        lo=references['M0HLT'][1]['populations'][name];hi=references['OFFLINE'][1]['populations'][name]
        recovery={k:reporting.recovery(pop['summary'].get(k),lo['summary'].get(k),hi['summary'].get(k)) for k in ('accuracy','macro_ovr_auc')}
        for eff in ('R50','R30'):
            recovery[eff]={cls:reporting.recovery(row.get('rejection'),lo[eff][cls].get('rejection'),hi[eff][cls].get('rejection')) for cls,row in pop[eff].items()}
        out[name]=dict(metrics=pop,recovery_percent=recovery)
    return c.artifact('RESULTS',parents=dict(spec=spec['content_hash'],report=report['content_hash']),
        selected_epoch=report['selected_pass'],populations=out,
        references={k:dict(epoch=r['selected_pass'],populations=m['populations']) for k,(r,m) in references.items()},
        payloads_reverified=False,inference=False)
