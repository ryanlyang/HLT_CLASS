"""Real all-rung measurement, native KD parity and full-support admission."""
from pathlib import Path
import gc
import math
import re
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, distillation_loss, installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.acceptance import installed_parity
from hlt_classification.jetclass2_delphes.runner import _forward
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_embedding_aux.model import new_model
from hlt_classification.luka_embedding_aux.preflight import parity
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from .contracts import artifact, validate, ARMS
from . import worker, data, training, campaign, source


def kd_parity(raw,node,device,*,bf16):
    states = []
    with parity_backend(device):
        for augmented in (False,True):
            torch.manual_seed(node['initialization_seed'])
            model = (new_model(node) if augmented else DelphesParticleTransformer()).to(device).train()
            if augmented:
                model.capture = True
            optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
            torch.manual_seed(817)
            logits = _forward(model,raw,device,bf16=bf16)
            labels = torch.from_numpy(raw['labels']).to(device)
            q = torch.full_like(logits,1/11,dtype=torch.float32)
            loss = distillation_loss(logits,labels,teacher_probabilities=q)
            if augmented:
                loss,*_ = training.objective(logits,labels,q,model.prediction,torch.zeros_like(model.prediction),0.)
            loss.backward(); optimizer.step()
            states.append({k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('predictor.')})
        for k in states[0]:
            torch.testing.assert_close(states[0][k],states[1][k],rtol=.01 if bf16 else 2e-5,atol=5e-4 if bf16 else 2e-6)
    return dict(bf16=bf16,zero_aux_native_kd=True)


def stress(cache,node,device,*,backward):
    lengths = np.concatenate([np.diff(b.offsets) for b in cache.blocks])
    raw = cache.batch(np.repeat(int(np.argmax(lengths)),128))
    model = new_model(node).to(device)
    model.train(backward); model.capture = backward
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward):
            logits = _forward(model,raw,device,bf16=str(device).startswith('cuda'))
            if backward:
                loss,*_ = training.objective(logits,torch.from_numpy(raw['labels']).to(device),
                    torch.full_like(logits,1/11,dtype=torch.float32),model.prediction,
                    torch.ones_like(model.prediction),node['embedding_coefficient'])
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite longest-support stress')
                optimizer.step()
            elif not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation stress')
    return dict(node=node['node_id'],role=cache.role,coordinate=cache.coordinate_name,maximum=int(lengths.max()),
                batch=128,steps=3,backward=backward)


def admission(gate,measurement):
    times = measurement['fits']
    expected = {n['node_id'] for arm in ARMS for n in gate['nodes'][arm]}
    if (set(times) != expected or any(not math.isfinite(measurement[k]) or measurement[k] < 0 for k in ('authentication','setup'))
            or any(set(t) != {'train_epoch','validation_epoch','other'} or any(not math.isfinite(v) or v < 0 for v in t.values())
                   or t['train_epoch'] <= 0 or t['validation_epoch'] <= 0 for t in times.values())
            or any(type(measurement[k]) is not int or measurement[k] <= 0 for k in ('peak_rss_bytes','gpu_peak_bytes'))):
        raise ValueError('Missing/invalid measured evidence')
    minutes = {arm:max(30,math.ceil(1.75*(measurement['authentication']+measurement['setup']+300+
        sum(30*(times[n['node_id']]['train_epoch']+times[n['node_id']]['validation_epoch'])+times[n['node_id']]['other']
            for n in gate['nodes'][arm]))/60)) for arm in ARMS}
    if (max(minutes.values()) > gate['science_ceiling_minutes']
            or measurement['peak_rss_bytes'] >= .85*gate['memory_mb']*1024**2
            or measurement['gpu_peak_bytes'] >= .9*measurement['gpu']['total_memory_bytes']):
        raise ValueError('Whole-ladder resource admission failed; preserve evidence, no automatic schedule change')
    return minutes


def run(gate):
    begun = time.monotonic(); campaign.check_gate(gate)
    authentication = time.monotonic()-begun
    job = worker.execution(gate)
    worker.storage(gate)
    root = Path(gate['root']); directory = root/'preflight'
    directory.mkdir(exist_ok=False)
    log = PhaseLog(directory,lambda:memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    environment,gpu = installed_environment(),gpu_identity()
    if environment != gate['anchor']['evidence']['environment']:
        raise ValueError('Parent installed environment differs')
    started = time.monotonic()
    with log.phase('all_registered_views_and_fixed_OFFLINE_targets'):
        meta = regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train,val = data.load_views(gate,'train',authenticated=meta),data.load_views(gate,'validation',authenticated=meta)
        embeddings = data.embedding_bank(gate,train['D000'])
        worker.snapshot_references(gate,val['D000'],meta)
        ref = gate['embedding_import']['predictions']; validate_file_ref(ref)
        atomic_publish_bytes(directory/'references/SINGLE_EMBED.npz',Path(ref['path']).read_bytes())
        anchor = worker.anchor_bank(gate,train['U000'],device='cuda',technical=True)
        write_immutable_json(directory/'anchor_logits.json',anchor.manifest)
    setup = time.monotonic()-started
    with log.phase('installed_native_and_auxiliary_KD_parity'):
        raw = train['D000'].batch(np.arange(4)); node = gate['nodes']['EMBED'][-1]
        with parity_backend('cuda'):
            native = installed_parity(train['D000'],device='cuda')
        parities = [parity(raw,node,'cuda',bf16=v) for v in (False,True)]
        kd_checks = [kd_parity(raw,node,'cuda',bf16=v) for v in (False,True)]
    stresses,timings = [],{}
    for arm in ARMS:
        incoming = anchor
        for node in gate['nodes'][arm]:
            co = node['coordinate']
            with log.phase('stress_'+node['node_id']):
                for role,cache in (('train',train[co]),('validation',val[co])):
                    row = stress(cache,node,'cuda',backward=role=='train')
                    if row['maximum'] != meta[1]['summaries'][role][co]['maximum']:
                        raise ValueError('Stress support differs from original prepared summary')
                    stresses.append(row)
                gc.collect(); torch.cuda.empty_cache()
            with log.phase('full_population_'+node['node_id']):
                previous = incoming.manifest
                record,incoming = worker.fit(directory/node['node_id'],gate,node,train[co],val[co],incoming,
                                              embeddings,device='cuda',technical=True)
                report,_,_ = worker.validate_step(directory/node['node_id'],gate,node,previous,embeddings.manifest['content_hash'],technical=True)
                row = report['validation_history'][0]
                timings[node['node_id']] = dict(train_epoch=row['train_seconds'],validation_epoch=row['validation_seconds'],
                    other=max(0.,record['seconds']-row['train_seconds']-row['validation_seconds']))
    memory = memory_snapshot(torch)
    measurement = dict(authentication=authentication,setup=setup,fits=timings,gpu=gpu,environment=environment,
        peak_rss_bytes=memory['peak_rss_bytes'],gpu_peak_bytes=memory['gpu_peak_bytes'])
    write_immutable_json(directory/'measurement.json',measurement)
    minutes = admission(gate,measurement)
    worker.storage(gate)
    if source.current(gate['project_dir'],gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Source changed during preflight')
    receipt = artifact('PREFLIGHT',parents=dict(gate=gate['content_hash']),job_id=job,
        embedding_hash=embeddings.manifest['content_hash'],**measurement,train_minutes=minutes,
        technical_checks_passed=True,native_parity=native,predictor_parities=parities,kd_parities=kd_checks,stress=stresses,
        outputs=[file_ref(p,root=directory) for p in sorted(directory.rglob('*')) if p.is_file()])
    write_immutable_json(directory/'receipt.json',receipt)
    validate_report(gate,receipt)
    return receipt


def validate_report(gate,report):
    validate(gate,'GATE')
    validate(report,'PREFLIGHT',dict(gate=gate['content_hash']))
    validate_file_ref(gate['embedding_import']['bank'])
    if load_json(gate['embedding_import']['bank']['path'])['content_hash'] != report['embedding_hash']:
        raise ValueError('Fixed OFFLINE bank differs')
    directory = Path(gate['root'])/'preflight'
    names = [r['path'] for r in report['outputs']]
    required = {'measurement.json','anchor_logits.json'} | {'references/'+n+'.npz' for n in ('M0HLT','OFFLINE','SINGLE_EMBED','original50k')}
    for arm in ARMS:
        required |= {n['node_id']+'/'+f for n in gate['nodes'][arm]
                     for f in ('receipt.json','training_report.json','validation_predictions.npz')}
    if len(names) != len(set(names)) or not required <= set(names):
        raise ValueError('Preflight output coverage differs')
    for ref in report['outputs']:
        validate_file_ref(ref,root=directory)
    measurement = load_json(directory/'measurement.json')
    if (any(report[k] != v for k,v in measurement.items()) or report['train_minutes'] != admission(gate,measurement)
            or report['technical_checks_passed'] is not True or report['native_parity']['passed'] is not True
            or report['environment'] != gate['anchor']['evidence']['environment']
            or not re.fullmatch('[1-9][0-9]*',report['job_id'])
            or [p['bf16'] for p in report['predictor_parities']] != [False,True]
            or any(p[k] is not True for p in report['predictor_parities'] for k in
                   ('native_forward','zero_control_update','auxiliary_backbone_gradient','auxiliary_head_gradient'))
            or report['kd_parities'] != [dict(bf16=v,zero_aux_native_kd=True) for v in (False,True)]):
        raise ValueError('Technical acceptance differs')
    expected = [(n['node_id'],r,n['coordinate'],r=='train') for arm in ARMS for n in gate['nodes'][arm] for r in ('train','validation')]
    if [(s['node'],s['role'],s['coordinate'],s['backward']) for s in report['stress']] != expected or any(
            s['batch'] != 128 or s['steps'] != 3 or type(s['maximum']) is not int or not 0 < s['maximum'] <= 512
            or s['maximum'] != gate['support'][s['role']][s['coordinate']] for s in report['stress']):
        raise ValueError('Full support stress coverage differs')
    anchor = load_json(directory/'anchor_logits.json')
    if anchor['parents']['checkpoint'] != gate['anchor']['evidence']['models']['U000']['checkpoint']['sha256']:
        raise ValueError('Wrong starting teacher')
    for arm in ARMS:
        incoming = anchor
        for node in gate['nodes'][arm]:
            _,_,incoming = worker.validate_step(directory/node['node_id'],gate,node,incoming,report['embedding_hash'],technical=True)
