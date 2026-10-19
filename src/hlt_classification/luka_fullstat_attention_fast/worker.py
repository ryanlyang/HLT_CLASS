"""Attention-only recovery workers; no new export, controls or logit fit."""
from pathlib import Path
import os
import time
import numpy as np
import torch
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.runner import predict, _forward
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullstat_attention import worker as old, banks as old_banks
from hlt_classification.luka_fullstat_attention.contracts import REFERENCES
from . import campaign, training, checkpoints
from .banks import ResidentBank
from .contracts import artifact, validate, policy


def load_inputs(g):
    parent = g['reuse']['gate']; measured = g['reuse']['preflight']
    tr, val = old.data(parent,'train','D000'), old.data(parent,'validation','D000')
    if (old_banks.alignment(tr) != measured['audit']['train_alignment']
            or old_banks.alignment(val) != measured['audit']['validation_alignment']):
        raise ValueError('Original HLT population/slot alignment changed')
    _, parents = old.checkpoint(parent)
    # Admit before allocating the resident matrix bank. The24GiB reserve covers
    # producer/consumer padded buffers, model/optimizer, serialization and RSS slack.
    budget = min(policy()['resident_cap_bytes'],
        int(.85*policy()['memory_mb']*1024**2)-memory_snapshot(torch)['rss_bytes']-policy()['reserve_bytes'])
    bank = ResidentBank(old.bank_path(parent),tr,parents=parents,teacher='COARSE_D033',
                        require_attention=True,max_bytes=budget)
    if bank.manifest['content_hash'] != g['reuse']['export']['result']['bank_hash']:
        raise ValueError('Exported bank changed')
    return tr, val, bank


def fit(directory, g, tr, val, bank, *, device, store, resume=None, resume_ref=None):
    node = next(n for n in g['reuse']['gate']['nodes'] if n['node_id']=='ATTN_D000')
    model = new_model(node)
    report,state = training.train(model,tr,val,bank=bank,node=node,device=device,store=store,resume=resume)
    old.report_valid(report,node,bank.manifest['content_hash'],tr.foundation_sha256,len(val))
    directory = Path(directory); directory.mkdir(exist_ok=False)
    save_state(directory/'selected.pt',state)
    restored = torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
    if set(restored)!=set(state) or any(not torch.equal(restored[k],state[k]) for k in state):
        raise ValueError('Selected checkpoint roundtrip differs')
    model.load_state_dict(restored,strict=True)
    model.eval(); model.capture_attention=False
    native = DelphesParticleTransformer().to(device).eval()
    native.load_state_dict(restored,strict=True)
    raw=val.batch(np.arange(min(4,len(val))))
    with torch.inference_mode():
        torch.testing.assert_close(_forward(model,raw,device,bf16=str(device).startswith('cuda')),
                                   _forward(native,raw,device,bf16=str(device).startswith('cuda')),rtol=0,atol=0)
    del native
    arrays=dict(identities=val.identities,labels=val.labels,probabilities=predict(model,val,device=device,batch_size=128))
    if evaluate_probabilities(arrays['labels'],arrays['probabilities']) != report['validation']:
        raise ValueError('Selected validation prediction readback differs')
    atomic_publish_bytes(directory/'validation_predictions.npz',deterministic_npz_bytes(arrays))
    write_immutable_json(directory/'training_report.json',report)
    execution=artifact('FIT_EXECUTION',parents=campaign.binding(g),
        native_inference_parity=True,epoch_checkpoints=str(store.root),
        resumed_from_epoch=0 if resume is None else resume['epoch'],resume=resume_ref,policy=policy())
    write_immutable_json(directory/'execution.json',execution)
    return report


def completed(s,g,name):
    root=Path(s['root']); path=root/name/'receipt.json'
    if not path.is_file(): return None
    row=load_json(path); validate(row,'TASK',dict(spec=s['content_hash']))
    if row['task_id']!=name or row['source_commit']!=s['source_commit']:
        raise ValueError('Task/source identity differs')
    paths={ref['path'] for ref in row['outputs']}
    if len(paths)!=len(row['outputs']) or any(not p.startswith(name+'/') for p in paths):
        raise ValueError('Output scope differs')
    for ref in row['outputs']: validate_file_ref(ref,root=root)
    if name=='train_ATTN_D000':
        required={name+'/'+f for f in ('selected.pt','training_report.json','validation_predictions.npz','execution.json')}
        if not required<=paths: raise ValueError('Missing scientific outputs')
        node=next(n for n in g['reuse']['gate']['nodes'] if n['node_id']=='ATTN_D000')
        report=load_json(root/name/'training_report.json')
        old.report_valid(report,node,g['reuse']['export']['result']['bank_hash'],
            g['reuse']['gate']['anchor']['evidence']['prepared'],g['reuse']['gate']['counts']['validation'])
        p=load_npz_arrays(root/name/'validation_predictions.npz'); original=old.validation_arrays(g['reuse']['gate'])
        if (set(p)!={'identities','labels','probabilities'} or not np.array_equal(p['identities'],original['identities'])
                or not np.array_equal(p['labels'],original['labels'])
                or evaluate_probabilities(p['labels'],p['probabilities'])!=report['validation']
                or row['result']!=dict(report_hash=report['content_hash'])):
            raise ValueError('Prediction population/metrics/report differs')
        execution=load_json(root/name/'execution.json'); validate(execution,'FIT_EXECUTION',campaign.binding(g))
        if not execution['native_inference_parity'] or execution['policy']!=policy():
            raise ValueError('Deployment/execution recipe differs')
        if execution['resume'] is not None:
            value=checkpoints.load(validate_file_ref(execution['resume']),binding=campaign.binding(g))
            if value['epoch']!=execution['resumed_from_epoch']: raise ValueError('Resume epoch lineage differs')
        elif execution['resumed_from_epoch']!=0: raise ValueError('Missing resume provenance')
    elif name=='aggregate':
        if not completed(s,g,'train_ATTN_D000'): raise ValueError('Missing completed fit')
        expected=results(s,g)
        if row['result']!=expected or load_json(root/name/'results.json')!=expected or name+'/results.json' not in paths:
            raise ValueError('Comparison differs')
    else: raise ValueError('Unregistered task')
    return row


def results(s,g):
    rows=[]
    for name in REFERENCES:
        refs=g['reuse']['gate']['anchor']['evidence']['models'][name]
        report=load_json(validate_file_ref(refs['report']))
        arrays=load_npz_arrays(validate_file_ref(refs['predictions']))
        if evaluate_probabilities(arrays['labels'],arrays['probabilities'])!=report['validation']:
            raise ValueError('Reference prediction metrics differ')
        rows.append(dict(node_id=name,origin='completed_parent_reference',selected_epoch=report['selected_pass'],validation=report['validation']))
    done=completed(s,g,'train_ATTN_D000')
    if done is None: raise ValueError('Attention fit has not completed20epochs')
    report=load_json(Path(s['root'])/'train_ATTN_D000/training_report.json')
    rows.append(dict(node_id='ATTN_D000',origin='new_fast_attention_fit',selected_epoch=report['selected_pass'],validation=report['validation']))
    for row in rows:
        row['recovery']=recovery(row['validation'],rows[0]['validation'],rows[1]['validation'])
    coarse=next(r['validation'] for r in rows if r['node_id']=='COARSE_D000')
    return artifact('RESULTS',parents=dict(spec=s['content_hash']),rows=rows,
        primary_contrast='ATTN_D000 minus original COARSE_D000',new_logit_fit=False,single_seed=True,
        contrasts={k:report['validation'][k]-coarse[k] for k in ('accuracy','macro_ovr_auc','cross_entropy')})


def run(s,name,*,resume_path=None):
    g,measured=campaign.check_spec(s)
    if name not in ('train_ATTN_D000','aggregate'): raise ValueError('No export/logit/test task')
    if resume_path is not None and name!='train_ATTN_D000': raise ValueError('Only training can resume')
    existing=completed(s,g,name)
    if existing: return existing
    root=Path(s['root']); directory=root/name
    if directory.exists(): raise FileExistsError('Preserve partial publication; inspect before retrying')
    campaign.storage(g)
    if name=='train_ATTN_D000':
        job=old.execution(g['reuse']['gate'],measured)
        # OS advisory lock is released on process death, unlike a stale lock dir.
        # It prevents concurrent attempts from publishing the same science task.
        import fcntl
        with (root/'training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            attempt=root/'attempts'/job; attempt.mkdir(parents=True,exist_ok=False)
            store=checkpoints.EpochStore(attempt/'checkpoints',binding=campaign.binding(g))
            restored=None if resume_path is None else checkpoints.load(resume_path,binding=campaign.binding(g))
            log=PhaseLog(attempt,lambda:memory_snapshot(torch))
            with log.phase('load_authenticated_HLT_and_resident_targets'):
                tr,val,bank=load_inputs(g)
            with log.phase('science_identical_twenty_epoch_fit'):
                report=fit(directory,g,tr,val,bank,device='cuda',store=store,resume=restored,
                           resume_ref=None if resume_path is None else file_ref(Path(resume_path)))
                result=dict(report_hash=report['content_hash'])
                del tr,val,bank
            old.execution(g['reuse']['gate'],measured)
    else:
        job=os.environ.get('SLURM_JOB_ID')
        result=results(s,g)
        directory.mkdir(exist_ok=False)
        write_immutable_json(directory/'results.json',result)
    campaign.check_spec(s)
    row=artifact('TASK',parents=dict(spec=s['content_hash']),task_id=name,source_commit=s['source_commit'],
        job_id=job,result=result,outputs=[file_ref(p,root=root) for p in sorted(directory.iterdir()) if p.is_file()])
    write_immutable_json(directory/'receipt.json',row)
    return completed(s,g,name)
