"""Fresh initialization, installed parity, every cold fit, and measured admission."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from hlt_classification.luka_fullsim.regular_cache import metadata
from hlt_classification.luka_gating.preflight import parity, stress
from hlt_classification.luka_gating.contracts import execution_policy
from . import campaign, worker, training
from .contracts import artifact, validate


def budgets(measurements, authentication_seconds=0.):
    if type(authentication_seconds) not in (int,float) or not math.isfinite(authentication_seconds) or authentication_seconds<0:
        raise ValueError('Invalid authentication timing')
    result = {}
    for m in measurements:
        for key in ('setup_seconds','epoch_seconds','other_seconds'):
            if type(m[key]) not in (int,float) or not math.isfinite(m[key]) or m[key]<0:
                raise ValueError('Invalid cold-fit timing')
        if m['node_id'] in result: raise ValueError('Duplicate measured cold fit')
        minutes = math.ceil(1.75*(authentication_seconds+m['setup_seconds']+
                                  30*m['epoch_seconds']+m['other_seconds'])/60)
        if minutes>1440: raise ValueError('Measured fit exceeds registered tier3 24h bound; no science')
        result[m['node_id']] = max(30,minutes)
    if not result: raise ValueError('Missing cold measurements')
    return result


def memory_ok(gate, memory, gpu):
    if (not 0<memory['peak_rss_bytes']<.85*gate['memory_mb']*1024**2
            or not 0<memory['gpu_peak_bytes']<.9*gpu['total_memory_bytes']):
        raise MemoryError('Measured host/GPU headroom failed')


def initialization_checks(gate):
    result = []
    for node in gate['nodes']:
        first = training.new_student(node)
        digest = training.initial_digest(first)
        native = training.initial_digest(first.mod)
        # A changed previous model must not influence the fresh construction.
        with torch.no_grad():
            for p in first.parameters(): p.add_(1.)
        second = training.new_student(node)
        if digest != training.initial_digest(second):
            raise ValueError('Cold seeded initialization depends on previous model')
        result.append(dict(node_id=node['node_id'],initialization_sha256=digest,
            native_sha256=native,repeatable_after_mutation=True))
    # Gate/no-gate controls have exactly the same fresh native backbone.
    for co in ('D066','D033','D000'):
        names = [b+'_'+co for b in ('LOGIT','SELF','PRIV')]
        if len({r['native_sha256'] for r in result if r['node_id'] in names})!=1:
            raise ValueError('Cold native backbone seeds differ across branches')
    return result


def run(gate):
    begin=time.monotonic(); campaign.check_gate(gate); auth=time.monotonic()-begin
    job=worker.execution(gate)
    if installed_environment()!=gate['anchor']['evidence']['environment']:
        raise PermissionError('Installed environment differs from U100 anchor')
    root=Path(gate['root'])/'preflight'; root.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    log=PhaseLog(root,lambda:memory_snapshot(torch))
    with log.phase('fresh_initialization_and_installed_parity'):
        initial=initialization_checks(gate)
        tr=worker.data(gate,'train','D000'); val=worker.data(gate,'validation','D000')
        alignments=dict(train=worker.banks.alignment(tr),validation=worker.banks.alignment(val))
        raw=tr.batch(np.arange(min(4,len(tr))))
        parities=[parity(raw,'cuda',bf16=b) for b in (False,True)]
        del tr,val,raw
    probe=artifact('SPEC',parents=dict(gate=gate['content_hash']),root=str(root),technical=True)
    write_immutable_json(root/'probe_spec.json',probe)
    measurements=[]
    for task in gate['tasks']:
        if task['kind']=='aggregate': continue
        gc.collect(); torch.cuda.empty_cache()
        with log.phase('cold_probe_'+task['node_id']):
            receipt=worker.execute_task(probe,gate,task['task_id'],technical=True,stress_fn=stress)
            measurements.append(receipt['measurement'])
    memory=memory_snapshot(torch); gpu=gpu_identity(); memory_ok(gate,memory,gpu)
    begin=time.monotonic(); campaign.check_gate(gate); auth+=time.monotonic()-begin
    begin=time.monotonic()
    outputs=[file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()]
    auth+=2*(time.monotonic()-begin)
    report=artifact('PREFLIGHT',parents=dict(gate=gate['content_hash']),job_id=job,
        environment=installed_environment(),gpu=gpu,memory=memory,parities=parities,
        attention_execution=execution_policy(),initialization=initial,alignments=alignments,
        measurements=measurements,fit_minutes=budgets(measurements,auth),authentication_seconds=auth,
        technical_checks_passed=True,scientific_fit=False,outputs=outputs)
    validate_report(gate,report)
    write_immutable_json(root/'receipt.json',report)
    print('COLD PREFLIGHT PASSED: per-fit requested minutes='+str(report['fit_minutes']),flush=True)
    return report


def validate_report(gate,r):
    validate(r,'PREFLIGHT',dict(gate=gate['content_hash']))
    names=[n['node_id'] for n in gate['nodes']]
    if (r['technical_checks_passed'] is not True or r['scientific_fit'] is not False
            or r['environment']!=gate['anchor']['evidence']['environment']
            or r['attention_execution']!=execution_policy() or gate['attention_execution']!=execution_policy()
            or [m['node_id'] for m in r['measurements']]!=names
            or r['fit_minutes']!=budgets(r['measurements'],r['authentication_seconds'])
            or [i['node_id'] for i in r['initialization']]!=names
            or any(i['repeatable_after_mutation'] is not True for i in r['initialization'])):
        raise ValueError('Incomplete cold evidence/registration/budget')
    checks=('identity_logits','identity_gradients','identity_updates','native_attention_surface',
        'gate_gradient','context_sensitivity','endpoint_context_rejected','multiplicative_equivalence',
        'auxiliary_vector_gradient_topology')
    if (len(r['parities'])!=2 or [p['bf16'] for p in r['parities']]!=[False,True]
            or any(p.get(k) is not True for p in r['parities'] for k in checks)):
        raise ValueError('Missing installed parity')
    memory_ok(gate,r['memory'],r['gpu'])
    prepared=metadata(gate['anchor']['evidence']['prepared_root'])[1]
    for node,m,i in zip(gate['nodes'],r['measurements'],r['initialization']):
        if m['initialization_sha256']!=i['initialization_sha256']:
            raise ValueError('Observed fit was not the registered fresh initialization')
        if [s['role'] for s in m['stress']]!=['train','validation']:
            raise ValueError('Missing separate TRAIN and validation stress')
        for s in m['stress']:
            if (s['node_id']!=node['node_id'] or s['coordinate']!=node['coordinate']
                    or s['maximum']!=prepared['summaries'][s['role']][node['coordinate']]['maximum']
                    or s['batch']!=128 or s['steps']!=3 or s['backward']!=(s['role']=='train')
                    or s['mode']!=('joint' if s['role']=='train' else 'inference')
                    or s['frozen_backbone_unchanged'] is not None
                    or not math.isfinite(s['seconds']) or s['seconds']<0):
                raise ValueError('Cold stress support differs')
        s=m['storage']; required=s['factor_bank_bytes']+gate['counts']['train']*128+3*2**30
        if (s['factor_bank_bytes']<=0 or s['scratch_required_bytes']!=required
                or s['persistent_required_bytes']!=2*2**30 or s['declared_quota_gib']!=gate['available_quota_gib']
                or required>s['scratch_free_bytes']
                or 2*2**30>min(s['persistent_free_bytes'],s['declared_quota_gib']*2**30)):
            raise ValueError('Storage admission differs')
    for co in ('D066','D033','D000'):
        if len({i['native_sha256'] for i in r['initialization'] if i['node_id'] in [b+'_'+co for b in ('LOGIT','SELF','PRIV')]})!=1:
            raise ValueError('Native initialization controls differ')
    if set(r['alignments'])!={'train','validation'}: raise ValueError('Missing population binding')
    for role,a in r['alignments'].items():
        if a['role']!=role or a['rows']!=gate['counts'][role] or a['prepared']!=gate['anchor']['evidence']['prepared']:
            raise ValueError('Wrong preflight population')
    root=Path(gate['root'])/'preflight'; paths=[ref['path'] for ref in r['outputs']]
    required={'probe_spec.json'}|{t['task_id']+'/receipt.json' for t in gate['tasks'] if t['kind']=='train'}
    if len(paths)!=len(set(paths)) or not required<=set(paths): raise ValueError('Missing technical payload coverage')
    for ref in r['outputs']: validate_file_ref(ref,root=root)
    probe=load_json(root/'probe_spec.json'); validate(probe,'SPEC',dict(gate=gate['content_hash']))
    if Path(probe['root'])!=root or probe['technical'] is not True: raise ValueError('Wrong technical root')
    seen={}
    for task in gate['tasks']:
        if task['kind']!='train': continue
        receipt=worker.completed(probe,gate,task['task_id'],seen,technical=True)
        if receipt is None or receipt['measurement'] not in r['measurements']:
            raise ValueError('Missing bound technical task/measurement')
        for ref in receipt['outputs']:
            if dict(ref,path=task['task_id']+'/'+ref['path']) not in r['outputs']:
                raise ValueError('Unbound selected-output payload')
