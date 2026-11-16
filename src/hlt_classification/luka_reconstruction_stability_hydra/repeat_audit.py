"""Read-only recovery report: preserve individual results, withhold invalid pairs."""
import importlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref,validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_u100_contrastive.training import recovery_percent
from .contracts import artifact

DONOR='15c35c428fa0548b97700d639323e593ea3aabac'


def differences(reference,actual):
    out={}
    for key in ('native','predictor'):
        if reference['components'][key]!=actual['components'][key]:
            out[key]=dict(control=reference['components'][key],fit=actual['components'][key])
    for key in ('seeds','dropout_rank_seeds','epoch1_order_hash','world_size'):
        if reference[key]!=actual[key]:out[key]=dict(control=reference[key],fit=actual[key])
    return out


def pair_status(records,task,control):
    if task not in records or control not in records:return dict(status='incomplete',differences={})
    delta=differences(records[control],records[task])
    return dict(status='mismatch' if delta else 'matched',differences=delta)


def run(path):
    spec=load_json(path)
    if spec['source_commit']!=DONOR:raise ValueError('This audit is bound to the 15c35c42 repeat campaign')
    if capture_source_snapshot(spec['project_dir'],require_clean=True)!=spec['source_snapshot']:
        raise ValueError('Repeat source snapshot differs')
    for ref in (spec['original_spec'],spec['benchmark_spec'],*spec['benchmark_outputs']):validate_file_ref(ref)
    folder=Path(spec['project_dir'])/'src/hlt_classification/luka_reconstruction_repeat_hydra';name='repeat_failure_audit_runtime'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=folder/'__init__.py':raise ValueError('Ambiguous repeat runtime')
    if name not in sys.modules:
        loader=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(loader);sys.modules[name]=module;loader.loader.exec_module(module)
    c=importlib.import_module(name+'.contracts');training=importlib.import_module(name+'.training')
    c.validate(spec,'SPEC')
    if spec['recipe']!=c.recipe() or spec['tasks']!=list(c.FITS):raise ValueError('Repeat scientific registration differs')
    worker=importlib.import_module(name+'.worker')
    reports={};initial={};reference=None;receipts={}
    for task in spec['tasks']:
        folder=Path(spec['root'])/task
        if not (folder/'receipt.json').is_file():continue
        worker.receipt(spec,task)
        report,arrays=training.validate_fit(folder/'fit',spec,task)
        if reference is not None and any(not np.array_equal(reference[k],arrays[k]) for k in ('identities','labels')):
            raise ValueError('Different validation populations')
        reference=arrays;reports[task]=report;initial[task]=load_json(folder/'fit/initialization.json')
        receipts[task]=file_ref(folder/'receipt.json')
    rows=[]
    for task,report in reports.items():
        control=c.control(task);pair=pair_status(initial,task,control);auc=report['validation']['macro_ovr_auc']
        rows.append(dict(task=task,epoch=report['selected_epoch'],auc=auc,recovery=recovery_percent(auc,spec['recovery_anchors']),
            control=control,pair=pair,delta=auc-reports[control]['validation']['macro_ovr_auc'] if pair['status']=='matched' else None))
    print('Authenticated individual repeat fits. Mismatched starts have no paired delta; no training rerun or source changes.')
    print('Exploratory validation; historical recovery anchors; no independent test.')
    print(f'{"task":12} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta":>12}  pairing')
    for row in rows:
        delta='--' if row['delta'] is None else f'{row["delta"]:+.8f}'
        print(f'{row["task"]:12} {row["epoch"]:3} {row["auc"]:12.8f} {row["recovery"]:7.1f} {delta:>12}  {row["pair"]["status"]}')
    return artifact('REPEAT_FAILURE_AUDIT',parents=dict(repeat=spec['content_hash']),source=file_ref(Path(path).resolve()),
        receipts=receipts,rows=rows,pending=[t for t in spec['tasks'] if t not in reports],
        original_pairing_failure_preserved=True,mismatch_cause='unresolved_requires_named_initial_tensors')
