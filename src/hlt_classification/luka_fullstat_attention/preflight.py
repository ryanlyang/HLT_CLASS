"""Oscar-only real parity, full support audit and measured miniature admission."""
from hashlib import sha256
from pathlib import Path
import gc
import math
import os
import time
import numpy as np
import torch

from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer, installed_environment
from hlt_classification.jetclass2_delphes.runner import _forward, train_kernel
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
from hlt_classification.luka_attention.model import new_model
from hlt_classification.luka_attention.preflight import parity, stress
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from . import campaign, worker, banks, training
from .contracts import artifact, validate, recipe, TEACHER


def audit_payloads(gate, meta):
    """Decode ordinary cached blocks only; compare ordered persistent slots."""
    manifest, foundation, memberships = meta
    root = gate['anchor']['evidence']['cache_root']
    pop = {r:dict(ids=[], labels=[], lengths=[]) for r in ('train','validation')}
    squared = 0
    for number, chunk in enumerate(manifest['chunks'], 1):
        role, i = chunk['role'], chunk['file_index']
        if role not in pop:
            raise PermissionError('Sealed cache role is forbidden')
        validate_file_ref(chunk['assignment'], root=Path(root))
        hlt = worker.cache.read_block(root, chunk, 'D000', foundation['files'][i], memberships[i])
        n = np.diff(hlt.offsets)
        if role == 'train':
            off = worker.cache.read_block(root, chunk, 'D033', foundation['files'][i], memberships[i])
            if any(not np.array_equal(a,b) for a,b in ((hlt.identities,off.identities),
                    (hlt.labels,off.labels),(hlt.offsets,off.offsets))):
                raise ValueError('Persistent D033/D000 ordered-slot alignment differs')
            squared += sum(int(v)**2 for v in n)
            del off
        p = pop[role]
        p['ids'].append(hlt.identities.copy()); p['labels'].append(hlt.labels.copy()); p['lengths'].append(n)
        del hlt
        if number % 100 == 0:
            print(f'FullStat attention audit: {number}/{len(manifest["chunks"])} blocks', flush=True)
    alignments, maxima = {}, {}
    for role,p in pop.items():
        ids, labels, n = [np.concatenate(p[k]) for k in ('ids','labels','lengths')]
        if len(n) != gate['counts'][role] or np.any((n < 1) | (n > 512)) or len(np.unique(ids,axis=0)) != len(ids):
            raise ValueError('Full population/support/unique identity differs')
        digest = sha256()
        for a in (ids, labels.astype('<i8'), n.astype('<i8')):
            digest.update(a.tobytes())
        alignments[role] = dict(prepared=manifest['content_hash'], role=role,
            ordered_slots_sha256=digest.hexdigest(), rows=len(n))
        maxima[role] = int(n.max())
        if role == 'validation':
            ref = worker.validation_arrays(gate)
            if not np.array_equal(ids,ref['identities']) or not np.array_equal(labels,ref['labels']):
                raise ValueError('Reference validation population differs from cache')
    return dict(train_alignment=alignments['train'], validation_alignment=alignments['validation'],
                maxima=maxima, sum_squared_counts=squared, prepared=manifest['content_hash'])


def teacher_stress(bundle, node, device):
    n = banks.counts(bundle)
    raw = bundle.batch(np.repeat(int(np.argmax(n)), 128))
    model = new_model(node).to(device).eval().requires_grad_(False)
    model.capture_attention = True
    with torch.inference_mode():
        for _ in range(3):
            logits = _forward(model, raw, device, bf16=str(device).startswith('cuda'))
            a = model.take_attention()
            if not torch.isfinite(logits).all() or not torch.isfinite(a).all():
                raise ValueError('Nonfinite maximum teacher attention')
    return dict(role='train', coordinate='D033', maximum=int(n.max()), batch=128, steps=3, backward=False)


def budgets(measurements, counts, auth_seconds, load_seconds, export_seconds):
    values = [auth_seconds, load_seconds, export_seconds]
    values += [m[k] for m in measurements for k in ('train_seconds','validation_seconds','readback_seconds')]
    if (len(measurements) != 2 or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values)
            or any(m['train_seconds'] <= 0 or m['validation_seconds'] <= 0 for m in measurements)):
        raise ValueError('Invalid timing measurements')
    trains = []
    for m in measurements:
        if any(type(m['counts'][r]) is not int or not 0 < m['counts'][r] <= counts[r] for r in ('train','validation')):
            raise ValueError('Invalid measured population')
        a = math.ceil(counts['train']/128)/math.ceil(m['counts']['train']/128)
        b = math.ceil(counts['validation']/128)/math.ceil(m['counts']['validation']/128)
        trains.append(math.ceil(2*(auth_seconds+load_seconds+export_seconds+
            20*(a*m['train_seconds']+b*m['validation_seconds'])+b*m['readback_seconds'])/60))
    train, export = max(30,max(trains)), max(15,math.ceil(2*(auth_seconds+load_seconds+export_seconds)/60))
    if max(train,export) > 2880:
        raise ValueError('Measured request exceeds48h; no scientific submission')
    return train, export


def zero_weight_check(node, tr, val, bank, device):
    """An entire miniature epoch matches the existing FullStat20 kernel."""
    torch.manual_seed(node['initialization_seed'])
    native = DelphesParticleTransformer()
    with parity_backend(device):
        expected, state = train_kernel(native, tr, val, node=node, device=device,
            teacher_probabilities=bank.probabilities, teacher_identities=bank.identities,
            acceptance_passes=1, training_recipe=recipe()['base'])
        actual, produced = training.train(new_model(node), tr, val, bank=bank,
                                          node=node, device=device, technical=True)
    if actual['validation'] != expected['validation']:
        raise ValueError('Zero-weight training metrics differ from native kernel')
    for key in state:
        torch.testing.assert_close(state[key], produced[key], rtol=0, atol=0)
    return True


def run(gate):
    start = time.monotonic()
    campaign.check_gate(gate)
    auth = time.monotonic()-start
    job = worker.execution(gate)
    meta = worker.metadata(gate)
    root = Path(gate['root'])/'preflight'
    root.mkdir(exist_ok=False)
    runtime = dict(job_id=job, environment=installed_environment(), gpu=gpu_identity(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cudnn_v8_disabled=os.environ.get('TORCH_CUDNN_V8_API_DISABLED','unset'))
    write_immutable_json(root/'runtime.json', runtime)
    log = PhaseLog(root, lambda:memory_snapshot(torch))
    torch.cuda.reset_peak_memory_stats()
    with log.phase('authenticate_full_required_cache_payloads_and_alignment'):
        audit_start = time.monotonic()
        audit = audit_payloads(gate, meta)
        audit_seconds = time.monotonic()-audit_start
        storage = campaign.storage(gate,audit['sum_squared_counts'],gate['counts']['train'])
    with log.phase('representative_miniature_views'):
        start = time.monotonic()
        teacher = worker.data(gate,'train','D033',miniature=True,authenticated=meta)
        tr = worker.data(gate,'train','D000',miniature=True,authenticated=meta)
        val = worker.data(gate,'validation','D000',miniature=True,authenticated=meta)
        if banks.alignment(teacher) != banks.alignment(tr):
            raise ValueError('Miniature slots differ')
        measured_load = time.monotonic()-start
        future = max(meta[0]['summaries']['train']['D033']['resident_bytes'],
            sum(meta[0]['summaries'][r]['D000']['resident_bytes'] for r in ('train','validation')))
        # Full runtime caches, identity/target arrays and bounded mmap working set.
        reserve = 2*future+gate['counts']['train']*128+banks.MAX_MAPPED_BYTES+2**30
        if reserve >= .75*gate['memory_mb']*1024**2:
            raise MemoryError('Full caches/target working set exceed reserved host envelope')
        load_estimate = max(audit_seconds, measured_load*future/max(1,tr.nbytes+val.nbytes))
    with log.phase('installed_native_attention_and_zero_loss_parity'):
        parities = [parity(tr.batch(np.arange(4)),'cuda',bf16=b) for b in (False,True)]
    stress_rows = []
    for label, bundle, backward in (('train',tr,True),('validation',val,False)):
        with log.phase('maximum_'+label+'_support'):
            row = stress(bundle,gate['nodes'][1],'cuda',backward=backward)
            if row['maximum'] != audit['maxima'][label]:
                raise ValueError('Miniature does not include full support witness')
            stress_rows.append(row)
            gc.collect(); torch.cuda.empty_cache()
    with log.phase('maximum_teacher_export_support'):
        stress_rows.append(teacher_stress(teacher,gate['nodes'][0],'cuda'))
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('selected_D033_miniature_target_export_and_readback'):
        path,parents = worker.checkpoint(gate)
        model = new_model(gate['nodes'][0])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.requires_grad_(False)
        start = time.monotonic()
        banks.export(root/'bank',model,teacher,parents=parents,teacher=TEACHER,attention=True,device='cuda')
        bank = banks.Bank(root/'bank',tr,parents=parents,teacher=TEACHER,require_attention=True)
        ns = banks.counts(teacher)
        ratio = max(gate['counts']['train']/len(teacher), audit['sum_squared_counts']/sum(int(n)**2 for n in ns))
        export_estimate = max(time.monotonic()-start,.001)*ratio
        del model,teacher
        gc.collect(); torch.cuda.empty_cache()
    with log.phase('lambda_zero_native_training_kernel_parity'):
        zero = zero_weight_check(gate['nodes'][0],tr,val,bank,'cuda')
        gc.collect(); torch.cuda.empty_cache()
    measurements = []
    for node in gate['nodes']:
        with log.phase('one_pass_'+node['node_id']):
            start = time.monotonic()
            report,model = worker.fit(root/('train_'+node['node_id']),gate,node,tr,val,bank,
                                      device='cuda',technical=True)
            elapsed = time.monotonic()-start
            hist = report['validation_history'][0]
            measurements.append(dict(node_id=node['node_id'],counts=dict(train=len(tr),validation=len(val)),
                train_seconds=hist['train_seconds'],validation_seconds=hist['validation_seconds'],
                readback_seconds=max(0.,elapsed-hist['train_seconds']-hist['validation_seconds']),
                report=report))
            del model
            gc.collect(); torch.cuda.empty_cache()
    snap = memory_snapshot(torch)
    reserved = snap['peak_rss_bytes']+reserve
    if reserved >= .85*gate['memory_mb']*1024**2 or snap['gpu_peak_bytes'] >= .9*runtime['gpu']['total_memory_bytes']:
        raise MemoryError('Measured GPU/host headroom failed')
    start = time.monotonic()
    campaign.check_gate(gate)
    if worker.execution(gate) != job or installed_environment() != runtime['environment'] or gpu_identity() != runtime['gpu']:
        raise PermissionError('Runtime changed during the technical preflight')
    auth += time.monotonic()-start
    outputs = [file_ref(p,root=root) for p in sorted(root.rglob('*')) if p.is_file()]
    train_minutes,export_minutes = budgets(measurements,gate['counts'],auth,load_estimate,export_estimate)
    result = artifact('PREFLIGHT',parents=dict(gate=gate['content_hash']),job_id=job,
        environment=runtime['environment'],gpu=runtime['gpu'],parities=parities,zero_weight_kernel_parity=zero,
        stress=stress_rows,audit=audit,storage=storage,measurements=measurements,
        authentication_seconds=auth,full_load_seconds_estimate=load_estimate,
        full_export_seconds_estimate=export_estimate,train_minutes=train_minutes,export_minutes=export_minutes,
        reserved_host_peak_bytes=reserved,gpu_peak_bytes=snap['gpu_peak_bytes'],outputs=outputs,
        technical_checks_passed=True,scientific_fit=False,full_population_training_measured=False)
    write_immutable_json(root/'measurement.json',result)
    validate_report(gate,result)
    write_immutable_json(root/'receipt.json',result)
    return result


def validate_report(gate, report):
    validate(report,'PREFLIGHT',dict(gate=gate['content_hash']))
    if (report['technical_checks_passed'] is not True or report['scientific_fit'] is not False
            or report['full_population_training_measured'] is not False or report['zero_weight_kernel_parity'] is not True
            or report['environment'] != gate['anchor']['evidence']['environment'] or 'L40S' not in report['gpu']['name']
            or not str(report['job_id']).isdigit() or int(report['job_id']) <= 0):
        raise ValueError('Incomplete/mismatched Oscar acceptance')
    if len(report['parities']) != 2 or [p['bf16'] for p in report['parities']] != [False,True]:
        raise ValueError('Both installed parity modes required')
    for p in report['parities']:
        if any(p.get(k) is not True for k in ('logits','input_gradients','parameter_gradients','buffers','optimizer_updates','native_attention_surface')):
            raise ValueError('Missing native parity evidence')
    if (not 0 < report['reserved_host_peak_bytes'] < .85*gate['memory_mb']*1024**2
            or not 0 < report['gpu_peak_bytes'] < .9*report['gpu']['total_memory_bytes']):
        raise ValueError('Resource headroom differs')
    meta = worker.metadata(gate)[0]
    a = report['audit']
    if a['prepared'] != meta['content_hash']:
        raise ValueError('Payload audit cache parent differs')
    for role in ('train','validation'):
        binding = a[role+'_alignment']
        if (binding.get('prepared') != meta['content_hash'] or binding.get('role') != role
                or binding.get('rows') != gate['counts'][role]
                or a['maxima'][role] != meta['summaries'][role]['D000']['maximum']):
            raise ValueError('Full-population alignment/support differs')
    expected = [('train','D000',True),('validation','D000',False),('train','D033',False)]
    if len(report['stress']) != 3:
        raise ValueError('Missing distinct support stresses')
    for row,(role,co,backward) in zip(report['stress'],expected):
        if (row['role'] != role or row['coordinate'] != co or row['backward'] is not backward
                or row['maximum'] != meta['summaries'][role][co]['maximum']
                or row['batch'] != 128 or row['steps'] != 3):
            raise ValueError('Maximum support stress differs')
    m = report['measurements']
    if [r['node_id'] for r in m] != [n['node_id'] for n in gate['nodes']] or m[0]['counts'] != m[1]['counts']:
        raise ValueError('Paired miniature fits differ')
    times = budgets(m,gate['counts'],report['authentication_seconds'],report['full_load_seconds_estimate'],report['full_export_seconds_estimate'])
    if times != (report['train_minutes'],report['export_minutes']):
        raise ValueError('Measured requests differ')
    if report['storage'] != campaign.storage(gate,a['sum_squared_counts'],gate['counts']['train'],check_space=False):
        raise ValueError('Storage declaration differs')
    root = Path(gate['root'])/'preflight'
    paths = [r['path'] for r in report['outputs']]
    required = {'runtime.json','bank/manifest.json','bank/population.npz'} | {
        'train_'+n['node_id']+'/'+f for n in gate['nodes']
        for f in ('selected.pt','training_report.json','validation_predictions.npz','costs.json')}
    if len(paths) != len(set(paths)) or not required <= set(paths):
        raise ValueError('Missing technical payloads')
    for ref in report['outputs']:
        validate_file_ref(ref,root=root)
    runtime = load_json(root/'runtime.json')
    if (any(runtime[k] != report[k] for k in ('job_id','environment','gpu'))
            or runtime['cublas_workspace_config'] != ':4096:8'
            or runtime['cudnn_v8_disabled'] != gate['anchor']['evidence']['cudnn_v8_disabled']):
        raise ValueError('Runtime evidence differs')
    bank = load_json(root/'bank/manifest.json')
    _,parents = worker.checkpoint(gate)
    validate(bank,'BANK',parents)
    if (bank['coordinate'] != 'D033' or bank['teacher'] != TEACHER or bank['attention'] is not True
            or bank['temperature'] != 2. or bank['surface'] != recipe()['surface']
            or bank['alignment']['rows'] != m[0]['counts']['train']
            or bank['alignment']['prepared'] != meta['content_hash'] or bank['alignment']['role'] != 'train'):
        raise ValueError('Miniature target parent/population differs')
    for ref in [bank['population']]+[s['payload'] for s in bank['shards']]:
        if dict(ref,path='bank/'+ref['path']) not in report['outputs']:
            raise ValueError('Unbound technical bank payload')
    for row,node in zip(m,gate['nodes']):
        saved = load_json(root/('train_'+node['node_id'])/'training_report.json')
        if saved != row['report']:
            raise ValueError('Miniature report differs')
        worker.report_valid(saved,node,bank['content_hash'],meta['content_hash'],row['counts']['validation'],technical=True)
