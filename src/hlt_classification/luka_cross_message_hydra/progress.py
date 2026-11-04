"""Read-only published epoch preview, including unfinished fits."""
from pathlib import Path
import math
import re
from hlt_classification.data.cache_contracts import load_json
from .contracts import TASKS, validate
from . import campaign, inputs


def show(spec):
    campaign.check(spec,check_storage=False)
    b=inputs.Bundle(spec['bundle_root']); origin=b.json('origin/campaign_spec.json')
    anchors=origin['recovery_anchors']; lo=anchors['M0HLT']; hi=anchors['OFFLINE']
    if not hi>lo: raise ValueError('Invalid recovery anchors')
    control=b.json('reference/matched/report.json')
    histories={'U100_w5_SPORC':{r['pass_number']:r['validation']['macro_ovr_auc'] for r in control['validation_history']}}
    root=Path(spec['root'])
    for task in TASKS:
        rows={}
        for path in sorted((root/task/'fit').glob('epoch_*.json')):
            row=load_json(path); validate(row,'EPOCH',dict(spec=spec['content_hash']))
            epoch=row['pass_number']; auc=row['validation']['macro_ovr_auc']
            if (row['task']!=task or type(epoch) is not int or not 1<=epoch<=30
                    or path.name!=f'epoch_{epoch:02d}.json' or epoch in rows
                    or row['validation']['rows']!=spec['counts']['validation']
                    or not math.isfinite(auc) or not 0<=auc<=1):
                raise ValueError('Invalid published epoch')
            rows[epoch]=auc
        if rows: histories[task]=rows
    print('Full validation; historical SPORC recovery anchors. Teacher rows are privileged oracles.')
    print('Epoch AUC / recovery%; not running-best. Original50k is not used here.')
    # Keep wide campaigns readable: one comparison table per task.
    ref=histories['U100_w5_SPORC']; reference=max(ref.values())
    for task,rows in histories.items():
        if task=='U100_w5_SPORC': continue
        print('\n'+task+'   [PRIVILEGED ORACLE]' if task.startswith('teacher_') else '\n'+task)
        print('Epoch          U100-w5 AUC / rec%              Current AUC / rec%')
        for epoch in sorted(rows):
            print(f'{epoch:5}       {ref[epoch]:.8f} / {100*(ref[epoch]-lo)/(hi-lo):7.1f}%'
                  f'          {rows[epoch]:.8f} / {100*(rows[epoch]-lo)/(hi-lo):7.1f}%')
    print('\nBEST AUC SO FAR (earliest on exact AUC tie)')
    for task,rows in histories.items():
        epoch=max(rows,key=lambda e:(rows[e],-e)); auc=rows[epoch]
        print(f'{task:25} {len(rows):2}/30 ep={epoch:2} AUC={auc:.8f} recovery={100*(auc-lo)/(hi-lo):5.1f}% delta_U100={auc-reference:+.8f}')
    for task in TASKS:
        logs=list(root.glob(task+'-*.out'))
        if len(logs)>1: raise ValueError('Ambiguous task logs')
        if not logs: continue
        matches=re.findall(r'CROSS_MESSAGE '+re.escape(task)+r' epoch=(\d+)/(\d+) batch=(\d+)/(\d+)',logs[0].read_text(errors='replace'))
        if matches:
            epoch,total,batch,batches=map(int,matches[-1])
            if epoch<=max(histories.get(task,{0:0})):
                print(f'{task}: validation published through epoch {epoch}; no newer batch marker.')
            else: print(f'{task}: last marker epoch {epoch}/{total}, batch {batch}/{batches} ({100*batch/batches:.1f}% of training batches).')
    print('Provisional snapshot; no inference, file writes, or job changes. Progress is last logged, not instantaneous.')
