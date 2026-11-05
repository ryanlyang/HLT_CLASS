"""Saved-results and provisional epochs, without inference or writes."""
from pathlib import Path
import math
import re
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from hlt_classification.luka_cross_message_hydra import contracts as donor_contracts
from hlt_classification.luka_cross_message_hydra import worker as donor_worker, training as donor_training
from .contracts import TASKS, artifact, validate, matched_task
from . import campaign, training, worker


def epochs(root, task, spec, *, parent=False):
    rows = {}; checker = donor_contracts.validate if parent else validate
    for path in sorted((Path(root)/task/'fit').glob('epoch_*.json')):
        row = load_json(path); checker(row,'EPOCH',dict(spec=spec['content_hash']))
        epoch = row['pass_number']; auc = row['validation']['macro_ovr_auc']
        if (row['task'] != task or type(epoch) is not int or not 1 <= epoch <= 30
                or path.name != f'epoch_{epoch:02d}.json' or epoch in rows
                or row['validation']['rows'] != spec['counts']['validation']
                or not math.isfinite(auc) or not 0 <= auc <= 1):
            raise ValueError('Invalid published epoch')
        rows[epoch] = auc
    return rows


def progress(spec):
    parent = campaign.check(spec,check_storage=False); b = Bundle(spec['bundle_root'])
    anchors = b.json('origin/campaign_spec.json')['recovery_anchors']; lo = anchors['M0HLT']; hi = anchors['OFFLINE']
    if not hi > lo: raise ValueError('Invalid recovery anchors')
    def cell(auc): return '--' if auc is None else f'{auc:.8f} / {100*(auc-lo)/(hi-lo):5.1f}%'
    ref = {r['pass_number']:r['validation']['macro_ovr_auc'] for r in b.json('reference/matched/report.json')['validation_history']}
    print(f'CE-only controls. Full validation; historical M0HLT={lo:.8f}, OFFLINE={hi:.8f}.')
    print('Each row is epoch AUC/recovery. Parent epochs may be provisional. U100-w5 is historical SPORC.')
    for task in TASKS:
        own = epochs(spec['root'],task,spec); matched = matched_task(task)
        other = epochs(parent['root'],matched,parent,parent=True)
        print(f'\n{task} versus parent {matched} (CE+embedding5)')
        print(f'{"Epoch":>5} {"CE only":>24} {"CE+embedding architecture":>26} {"Native U100-w5":>24}')
        for ep in sorted(set(own)|set(other)):
            print(f'{ep:5} {cell(own.get(ep)):>24} {cell(other.get(ep)):>26} {cell(ref.get(ep)):>24}')
        for label, rows in ((task,own),(matched,other)):
            if not rows:
                print(label+': no completed epoch yet'); continue
            ep = max(rows,key=lambda e:(rows[e],-e))
            print(f'{label}: {len(rows)}/30 epochs; best epoch={ep}, {cell(rows[ep])}')
        logs = list(Path(spec['root']).glob(task+'-*.out'))
        if len(logs)>1: raise ValueError('Ambiguous CE-only logs')
        if logs:
            markers = re.findall(r'CROSS_MESSAGE_CE '+re.escape(task)+r' epoch=(\d+)/30 batch=(\d+)/(\d+)',logs[0].read_text(errors='replace'))
            if markers:
                ep,batch,total = map(int,markers[-1])
                print(f'Last logged training marker: epoch{ep}/30, batch{batch}/{total} ({100*batch/total:.1f}%).')
    print('Read-only provisional snapshot; no inference, file or job changes. No independent test.')


def results(spec):
    parent = campaign.check(spec,check_storage=False); b = Bundle(spec['bundle_root'])
    arrays = {name:load_npz_arrays(b.path(f'reference/{name}/predictions.npz')) for name in ('M0HLT','OFFLINE')}
    arrays['U100_W5_SPORC'] = load_npz_arrays(b.path('reference/matched/predictions.npz'))
    arrays['U100_W5_A6000_REPLAY'] = load_npz_arrays(Path(parent['root'])/'setup/control_precision/hydra_validation_predictions.npz')
    selected = {}; pending = []; hashes = {}
    for task in TASKS:
        for name,root,s,old in ((task,Path(spec['root']),spec,False),
                                (matched_task(task),Path(parent['root']),parent,True)):
            if not (root/name/'receipt.json').exists(): pending.append(name); continue
            if old:
                donor_worker.receipt(s,name); report,a = donor_training.validate_fit(root/name/'fit',s,name)
                if report['parents'].get('embedding') != b.json('targets/u100/manifest.json')['content_hash']:
                    raise ValueError('Parent embedding target lineage differs')
            else:
                worker.receipt(s,name); report,a = training.validate_fit(root/name/'fit',s,name)
            if report['parents']['prepared'] != spec['anchor']['evidence']['prepared']:
                raise ValueError('Comparison population differs')
            arrays[name] = a; selected[name] = report['selected_pass']; hashes[name] = report['content_hash']
    subset = load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],)
            or len(np.unique(subset)) != len(subset) or np.any(subset<0) or np.any(subset>=spec['counts']['validation'])):
        raise ValueError('Invalid overlapping report subset')
    metrics = {}
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],arrays['M0HLT'][k]) for k in ('identities','labels')):
            raise ValueError('Comparison identities/labels differ')
        metrics[name] = {role:evaluate_probabilities(a['labels'][ix],a['probabilities'][ix])
                        for role,ix in (('validation',slice(None)),('original50k',subset))}
    print('Two fresh CE-only architecture controls; no privileged optimization targets.')
    print('Historical SPORC recovery anchors; original50k overlaps validation; no independent test.')
    rows = []
    for role in ('validation','original50k'):
        lo = metrics['M0HLT'][role]['macro_ovr_auc']; hi = metrics['OFFLINE'][role]['macro_ovr_auc']
        if not hi>lo: raise ValueError('Invalid recovery anchors')
        print('\n'+role); print(f'{"model":26} {"epoch":>5} {"AUC":>12} {"recovery%":>10}')
        for name,m in metrics.items():
            auc = m[role]['macro_ovr_auc']; rec = 100*(auc-lo)/(hi-lo)
            print(f'{name:26} {str(selected.get(name,"-")):>5} {auc:12.8f} {rec:10.1f}')
            rows.append(dict(model=name,role=role,epoch=selected.get(name),metrics=m[role],recovery=rec))
        for task in TASKS:
            matched = matched_task(task)
            if task in metrics and matched in metrics:
                diff = metrics[matched][role]['macro_ovr_auc']-metrics[task][role]['macro_ovr_auc']
                print(f'{matched} minus {task}: AUC={diff:+.8f}, recovery={100*diff/(hi-lo):+.2f} percentage points')
    print('Pending/incomplete: '+(', '.join(pending) or 'none'))
    print('Read-only; no inference, file or job changes.')
    return artifact('RESULTS',parents=dict(spec=spec['content_hash'],**hashes),rows=rows,pending=pending)
