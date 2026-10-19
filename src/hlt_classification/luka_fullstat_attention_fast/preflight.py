"""Real-bank random-access benchmark, exact kernel parity and measured admission."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import write_immutable_json
from hlt_classification.jetclass2_delphes.cache import RamCache, RamBlock
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_attention.preflight import parity, stress
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullstat_attention import worker as old, banks as old_banks, training as original
from hlt_classification.luka_fullstat_attention.contracts import artifact as original_artifact
from . import campaign, worker, training, checkpoints
from .banks import batches
from .contracts import artifact,validate,policy


def selected_cache(cache, indices):
    blocks=[]
    for index in indices:
        raw=cache.batch(np.array([index]))
        n=int(raw['mask'].sum())
        blocks.append(RamBlock(0,np.array([0,n],np.int64),raw['features'][0,:,:n].T.copy(),
            raw['vectors'][0,:,:n].T.copy(),raw['identities'].copy(),raw['labels'].copy()))
    return RamCache(blocks,role=cache.role,foundation_sha256=cache.foundation_sha256,coordinate_name=cache.coordinate_name)


class SelectedBank:
    """Ephemeral TRAIN-only technical slice; never a reusable scientific bank."""
    def __init__(self,bank,cache,indices):
        self.bank,self.indices=bank,indices
        self.probabilities=bank.probabilities[indices]
        self.identities=bank.identities[indices]
        self.manifest=original_artifact('TECHNICAL_BANK_SLICE',parents=dict(bank=bank.manifest['content_hash']),
            teacher='COARSE_D033',coordinate='D033',attention=True,alignment=old_banks.alignment(cache))

    def attention_batch(self,ix,width):
        return self.bank.attention_batch(self.indices[ix],width)


def scientific_equal(a,b):
    def clean(r):
        r=deepcopy(r)
        for key in ('content_hash','runtime_seconds'): r.pop(key)
        for row in r['validation_history']:
            row.pop('train_seconds'); row.pop('validation_seconds')
        return r
    if clean(a)!=clean(b): raise ValueError('Optimized/original training metrics or recipe differ')


def kernel_parity(tr,val,bank,node,root,device):
    # Tiny independent witnesses contain every class; scientific metrics never
    # influence gate acceptance. Original and optimized BOTH read the same bytes.
    ix=np.concatenate([np.flatnonzero(tr.labels==k)[:2] for k in range(11)])
    vi=np.concatenate([np.flatnonzero(val.labels==k)[:2] for k in range(11)])
    a,b=selected_cache(tr,ix),selected_cache(val,vi)
    targets=SelectedBank(bank,a,ix)
    with parity_backend(device):
        expected,state=original.train(new_model(node),a,b,bank=targets,node=node,device=device,technical=True)
        store=checkpoints.EpochStore(Path(root)/'parity_checkpoints',binding=dict(technical_bank=targets.manifest['content_hash']))
        actual,produced=training.train(new_model(node),a,b,bank=targets,node=node,device=device,technical=True,store=store)
        scientific_equal(expected,actual)
        for key in state: torch.testing.assert_close(state[key],produced[key],rtol=0,atol=0)
        loaded=checkpoints.load(Path(root)/'parity_checkpoints/epoch-01.json',binding=store.binding)
        if loaded['history']!=actual['validation_history'] or loaded['selected']!=actual['selected_pass']:
            raise ValueError('Epoch checkpoint readback differs')
        for key in state: torch.testing.assert_close(state[key],loaded['best_state'][key],rtol=0,atol=0)
    return dict(exact_selected_weights=True,exact_validation_and_losses=True,epoch_checkpoint_readback=True,
                train_witnesses=len(a),validation_witnesses=len(b),
                checkpoint_bytes=(Path(root)/'parity_checkpoints/epoch-01.pt').stat().st_size)


def budget(*,train_step_seconds,validation_seconds,auth_seconds,load_seconds,counts):
    values=(train_step_seconds,validation_seconds,auth_seconds,load_seconds)
    if (any(not math.isfinite(v) or v<0 for v in values)
            or train_step_seconds<=0 or validation_seconds<=0):
        raise ValueError('Invalid timing evidence')
    epoch=train_step_seconds*math.ceil(counts['train']/128)+validation_seconds
    #2x throughput/startup reserve, two extra validation readbacks,15min for
    # checkpoint IO, final authentication and publication. No19h underestimate.
    minutes=max(policy()['minimum_train_minutes'],math.ceil(2*(2*auth_seconds+load_seconds+20*epoch+2*validation_seconds+900)/60))
    if minutes>policy()['maximum_minutes']:
        raise ValueError(f'Measured conservative request {minutes}min exceeds48h; no science')
    return minutes


def validate_report(g,r):
    validate(r,'PREFLIGHT',dict(gate=g['content_hash']))
    p=policy(); parent=g['reuse']['gate']
    if (r['scientific_fit'] is not False or r['technical_checks_passed'] is not True
            or r['bank_hash']!=g['reuse']['export']['result']['bank_hash']
            or r['policy']!=p or r['counts']!=parent['counts']
            or r['environment']!=parent['anchor']['evidence']['environment']
            or r['train_minutes']!=budget(**r['budget_inputs'])
            or r['budget_inputs']['counts']!=parent['counts']
            or r['benchmark_batches']!=p['benchmark_batches']
            or r['full_epoch_training_measured'] is not False or r['full_validation_measured'] is not True
            or r['loader_benchmark']['target_bytes_equal'] is not True
            or r['resident_target_bytes'] != 4*g['reuse']['preflight']['audit']['sum_squared_counts']+8*(parent['counts']['train']+1)
            or len(r['step_blocks'])!=8 or any(not math.isfinite(x) or x<=0 for x in r['step_blocks'])
            or r['budget_inputs']['train_step_seconds']!=max(r['step_blocks'])
            or r['reserved_host_peak_bytes']>=.85*p['memory_mb']*1024**2
            or r['memory']['gpu_peak_bytes']>=.9*r['gpu']['total_memory_bytes']
            or any(r['kernel_parity'][k] is not True for k in
                   ('exact_selected_weights','exact_validation_and_losses','epoch_checkpoint_readback'))
            or len(r['native_parity'])!=2 or [q['bf16'] for q in r['native_parity']]!=[False,True]
            or any(q[k] is not True for q in r['native_parity'] for k in
                   ('logits','input_gradients','parameter_gradients','buffers','optimizer_updates','native_attention_surface'))
            or len(r['stress'])!=2 or r['stress'][0]['backward'] is not True
            or r['stress'][1]['backward'] is not False
            or any(q['coordinate']!='D000' or q['batch']!=128 or q['steps']!=3 for q in r['stress'])
            or {x['role']:x['maximum'] for x in r['stress']}!=g['reuse']['preflight']['audit']['maxima']):
        raise ValueError('Performance gate failed or evidence differs')
    if (r['estimated_epoch_train_seconds']!=max(r['step_blocks'])*math.ceil(parent['counts']['train']/128)
            or type(r['kernel_parity']['checkpoint_bytes']) is not int or r['kernel_parity']['checkpoint_bytes']<=0
            or 22*r['kernel_parity']['checkpoint_bytes']+2*2**30>p['result_headroom_bytes']):
        raise ValueError('Timing/checkpoint envelope evidence differs')


def run(g):
    started=time.monotonic(); parent=campaign.check_gate(g); auth=time.monotonic()-started
    job=old.execution(parent); campaign.storage(g)
    root=Path(g['root'])/'preflight'; root.mkdir(exist_ok=False)
    log=PhaseLog(root,lambda:memory_snapshot(torch)); torch.cuda.reset_peak_memory_stats()
    with log.phase('load_full_authenticated_bank_in_RAM_once'):
        started=time.monotonic(); tr,val,bank=worker.load_inputs(g); load=time.monotonic()-started
    node=next(n for n in parent['nodes'] if n['node_id']=='ATTN_D000')
    with log.phase('installed_native_and_original_training_parity'):
        native=[parity(tr.batch(np.arange(4)),'cuda',bf16=b) for b in (False,True)]
        checked=kernel_parity(tr,val,bank,node,root,'cuda')
    with log.phase('full_population_maximum_support_stress'):
        support=[stress(tr,node,'cuda',backward=True),stress(val,node,'cuda',backward=False)]
        gc.collect(); torch.cuda.empty_cache()
    order=np.random.default_rng(np.random.SeedSequence([node['sampler_seed'],1])).permutation(len(tr))
    with log.phase('old_vs_resident_random_target_loader'):
        #ResidentBank already authenticated the complete bank. Borrow its metadata
        #to measure the unmodified old LRU without repeating the full hash pass.
        legacy=old_banks.Bank.__new__(old_banks.Bank)
        for key in ('root','manifest','counts','ends'): setattr(legacy,key,getattr(bank,key))
        legacy._lru=OrderedDict()
        old_seconds=fast_seconds=0.
        for start in range(0,32*128,128):
            ix=order[start:start+128]; width=max(16,int(bank.counts[ix].max()))
            now=time.monotonic(); expected=legacy.attention_batch(ix,width); old_seconds+=time.monotonic()-now
            now=time.monotonic(); actual=bank.attention_batch(ix,width); fast_seconds+=time.monotonic()-now
            if not np.array_equal(expected,actual): raise ValueError('Resident target bytes differ')
        del legacy,expected,actual
    with log.phase('full_bank_shuffled_training_throughput'):
        model=new_model(node).to('cuda').train(); model.capture_attention=True
        optimizer=training.optimizer_for(model); block_times=[]; completed=0
        p=policy(); measured_order=order[:128*(p['benchmark_batches']+p['benchmark_warmup'])]
        now=time.monotonic()
        for i,(ix,raw,target) in enumerate(batches(tr,bank,measured_order),1):
            training.update(model,optimizer,raw,bank.probabilities[ix],target,device='cuda')
            del raw,target
            if i==p['benchmark_warmup']: now=time.monotonic()
            elif i>p['benchmark_warmup'] and (i-p['benchmark_warmup'])%32==0:
                elapsed=time.monotonic()-now; block_times.append(elapsed/32); completed+=32
                print(f'FAST benchmark {completed}/{p["benchmark_batches"]} updates: {elapsed/32:.5f} seconds/update',flush=True)
                now=time.monotonic()
        if completed!=p['benchmark_batches']: raise ValueError('Incomplete full-bank throughput sample')
    with log.phase('full_500k_validation_timing'):
        model.capture_attention=False
        started=time.monotonic(); probabilities=predict(model,val,device='cuda',batch_size=128)
        evaluate_probabilities(val.labels,probabilities)  #Validity only; no quality gate.
        validation_seconds=time.monotonic()-started
        del probabilities,model,optimizer
    memory=memory_snapshot(torch)
    inputs=dict(train_step_seconds=max(block_times),validation_seconds=validation_seconds,
                auth_seconds=auth,load_seconds=load,counts=parent['counts'])
    minutes=budget(**inputs)
    report=artifact('PREFLIGHT',parents=dict(gate=g['content_hash']),job_id=job,scientific_fit=False,
        technical_checks_passed=True,bank_hash=bank.manifest['content_hash'],policy=policy(),counts=parent['counts'],
        environment=installed_environment(),gpu=gpu_identity(),memory=memory,
        reserved_host_peak_bytes=memory['peak_rss_bytes']+policy()['reserve_bytes'],
        resident_target_bytes=bank.loaded_bytes,kernel_parity=checked,native_parity=native,stress=support,
        benchmark_batches=completed,step_blocks=block_times,budget_inputs=inputs,train_minutes=minutes,
        loader_benchmark=dict(batches=32,original_seconds=old_seconds,resident_seconds=fast_seconds,
                              target_bytes_equal=True),
        estimated_epoch_train_seconds=max(block_times)*math.ceil(len(tr)/128),
        full_epoch_training_measured=False,full_validation_measured=True)
    validate_report(g,report)
    if 22*checked['checkpoint_bytes']+2*2**30>policy()['result_headroom_bytes']:
        raise OSError('Measured epoch checkpoints exceed declared result envelope')
    old.execution(parent); campaign.check_gate(g)
    write_immutable_json(root/'receipt.json',report)
    print(f'FAST measured epoch estimate={report["estimated_epoch_train_seconds"]/60:.2f}min training; '
          f'full validation={validation_seconds/60:.2f}min; conservative request={minutes}min',flush=True)
    return report
