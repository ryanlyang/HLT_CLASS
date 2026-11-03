"""Original training kernels; relocated frozen keys and independent Hydra jobs."""
from pathlib import Path
import time
import numpy as np
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.execution import gpu_identity
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_fullsim.preflight_v2 import require_backend
from hlt_classification.luka_embedding_weight.worker import validate_predictions
from hlt_classification.luka_u100_contrastive import training, resources
from hlt_classification.luka_u100_contrastive.model import new_model
from hlt_classification.luka_u100_contrastive.worker import extended_metrics
from hlt_classification.luka_u100_pair_hydra import portability, site
from .contracts import ARMS, artifact, validate
from . import inputs, campaign, submission


def setup(spec, *, device='cuda'):
    folder = Path(spec['root'])/'setup'; folder.mkdir(exist_ok=False)
    start = time.perf_counter()
    train = inputs.load_view(spec, 'train', 'D000'); val = inputs.load_view(spec, 'validation', 'D000')
    u100 = inputs.embedding_bank(spec, train); keys = inputs.key_bank(spec, train)
    write_immutable_json(folder/'portability.json', portability.setup_record(spec, val))
    d = inputs.delta(spec)
    write_immutable_json(folder/'inputs.json', artifact('INPUTS', parents=dict(spec=spec['content_hash']),
        offline_manifest=file_ref(d.path('keys/manifest.json')), offline_payload=file_ref(d.path('keys/keys.npz')),
        u100_manifest=file_ref(inputs.Bundle(spec['bundle_root']).path('targets/u100/manifest.json')),
        target_exported=False, teachers_retrained=False))
    resources.measure(spec, train, val, u100, keys, load_seconds=time.perf_counter()-start, device=device)


def receipt(spec, task):
    folder = Path(spec['root'])/task; value = load_json(folder/'receipt.json')
    validate(value, 'TASK', dict(spec=spec['content_hash']))
    if (value['task'] != task or value['job_id'] != submission.accepted_jobs(spec,
            'setup' if task == 'setup' else 'science').get(task)
            or len({r['path'] for r in value['outputs']}) != len(value['outputs'])):
        raise ValueError('Task exact-job/output ownership differs')
    for r in value['outputs']: validate_file_ref(r, root=folder)
    return value


def completed_setup(spec):
    value = receipt(spec, 'setup'); root = Path(spec['root'])/'setup'
    required = {'inputs.json','portability.json','capacity.json','baseline_parity.json',
        'resource_diagnostic.json','acceptance.json'} | {'mini_'+s+'/receipt.json' for s in ('none','random','mix','kinmix','hard','cross')}
    if not required <= {r['path'] for r in value['outputs']}: raise ValueError('Incomplete contrastive setup')
    record = load_json(root/'inputs.json'); validate(record, 'INPUTS', dict(spec=spec['content_hash']))
    d = inputs.delta(spec)
    expected = dict(offline_manifest=file_ref(d.path('keys/manifest.json')), offline_payload=d.rows['keys/keys.npz'],
        u100_manifest=file_ref(inputs.Bundle(spec['bundle_root']).path('targets/u100/manifest.json')))
    # The payload is already authenticated in setup; zero-control workers only
    # inspect its bound file reference, never open the OFFLINE payload.
    if (record['offline_manifest'] != expected['offline_manifest'] or record['u100_manifest'] != expected['u100_manifest']
            or record['offline_payload'] != dict(expected['offline_payload'], path=str(d.root/'keys/keys.npz'))
            or record['target_exported'] is not False or record['teachers_retrained'] is not False):
        raise ValueError('Shared input reuse differs')
    portability.validate_setup(spec, load_json(root/'portability.json'))
    resources.accepted(spec)
    return value


def run(spec, task):
    if task not in ('setup','report',*spec['remaining']): raise ValueError('Unregistered or already reused task')
    campaign.check(spec); require_backend(); job = site.allocation(spec)
    stage = 'setup' if task == 'setup' else 'science'
    for _ in range(12):
        if submission.accepted_jobs(spec, stage).get(task) == job: break
        time.sleep(5)
    else: raise PermissionError('Worker is not the accepted exact job')
    claim = Path(spec['root'])/'claims'/task; claim.mkdir(parents=True, exist_ok=False)
    write_immutable_json(claim/'intent.json', artifact('RUN_CLAIM', parents=dict(spec=spec['content_hash']), task=task, job_id=job))
    if task == 'setup': setup(spec)
    else:
        completed_setup(spec); accepted = resources.accepted(spec)
        if accepted['environment'] != installed_environment() or accepted['gpu'] != gpu_identity():
            raise PermissionError('Measured software/GPU changed')
        folder = Path(spec['root'])/task; folder.mkdir(exist_ok=False)
        if task == 'report':
            value = results(spec); write_immutable_json(folder/'results.json', value); print_results(value)
        else:
            train = inputs.load_view(spec,'train','D000'); val = inputs.load_view(spec,'validation','D000')
            bank = inputs.embedding_bank(spec,train); sampler, key_hash = None, None
            if ARMS[task]['coefficient']:
                keys = inputs.key_bank(spec,train); sampler = resources.make_sampler(keys,task,'cuda')
                key_hash = keys.manifest['content_hash']
            model = new_model(spec['imported'],task)
            training.fit(folder/'fit',spec,task,model,train,val,bank,sampler,device='cuda',key_hash=key_hash)
            training.validate_fit(folder/'fit',spec,task)
    campaign.check(spec); folder = Path(spec['root'])/task
    value = artifact('TASK',parents=dict(spec=spec['content_hash']),task=task,job_id=job,
        outputs=[file_ref(p,root=folder) for p in sorted(folder.rglob('*')) if p.is_file()])
    write_immutable_json(folder/'receipt.json',value)
    return value


def results(spec):
    campaign.check(spec,check_storage=False); d = inputs.delta(spec); b = inputs.Bundle(spec['bundle_root'])
    u100_hash = b.json('targets/u100/manifest.json')['content_hash']
    key_hash = load_json(d.path('keys/manifest.json'))['content_hash']
    arrays = {n:load_npz_arrays(b.path(f'reference/{n}/predictions.npz')) for n in ('M0HLT','OFFLINE')}
    arrays['U100_W5_HYDRA_REPLAY'] = load_npz_arrays(spec['portability_evidence']['refs']['predictions']['path'])
    locations = {n:'SPORC' for n in arrays}; locations['U100_W5_HYDRA_REPLAY'] = 'Hydra replay/SPORC weights'
    selected, pending, jobs = {}, [], {}
    for arm in ARMS:
        if arm in spec['reused']:
            report = load_json(d.path(f'completed/{arm}/report.json'))
            training.validate_report(report,d.manifest['origin_spec'],arm)
            a = load_npz_arrays(d.path(f'completed/{arm}/predictions.npz'))
            validate_predictions(a,spec['counts']['validation'])
            if not portability.same_metrics(evaluate_probabilities(a['labels'],a['probabilities']),report['validation']):
                raise ValueError('Reused predictions differ from saved report')
            locations[arm] = 'SPORC'; jobs[arm] = d.manifest['original_replay']['completed'][arm]['job_id']
        else:
            folder = Path(spec['root'])/arm
            if not (folder/'receipt.json').exists(): pending.append(arm); continue
            task = receipt(spec,arm); report,a = training.validate_fit(folder/'fit',spec,arm)
            locations[arm] = 'Hydra'; jobs[arm] = task['job_id']
        expected = dict(spec=d.manifest['origin_spec']['content_hash'] if arm in spec['reused'] else spec['content_hash'],
            recipe=spec['recipe']['content_hash'],prepared=spec['anchor']['evidence']['prepared'],u100=u100_hash)
        if ARMS[arm]['coefficient']: expected['offline'] = key_hash
        if report['parents'] != expected: raise ValueError('Fit target lineage differs')
        arrays[arm] = a; selected[arm] = report['selected_pass']
    subset = load_npz_arrays(b.path('reference/historical/subset.npz'))['indices']
    if (subset.dtype != np.int64 or subset.shape != (spec['original50k_rows'],)
            or len(np.unique(subset)) != len(subset) or np.any(subset < 0) or np.any(subset >= spec['counts']['validation'])):
        raise ValueError('Invalid overlapping report-only subset')
    metrics = {}
    for name,a in arrays.items():
        validate_predictions(a,spec['counts']['validation'])
        if any(not np.array_equal(a[k],arrays['M0HLT'][k]) for k in ('identities','labels')):
            raise ValueError('Result population differs')
        metrics[name] = {role:extended_metrics(a['labels'][ix],a['probabilities'][ix])
            for role,ix in (('validation',slice(None)),('original50k',subset))}
    rows = []
    for name, values in metrics.items():
        control = 'u100_w5_s'+str(ARMS[name]['replicate']) if name in ARMS else None
        row = dict(model=name,site=locations[name],job_id=jobs.get(name),selected_epoch=selected.get(name),
            matched_control=control,matched_control_site=locations.get(control),**values)
        for role in values:
            row[role+'_recovery'] = recovery(values[role],metrics['M0HLT'][role],metrics['OFFLINE'][role])
            row[role+'_delta_matched'] = values[role]['macro_ovr_auc']-metrics[control][role]['macro_ovr_auc'] if control in metrics else None
            row[role+'_delta_hydra_replay'] = values[role]['macro_ovr_auc']-metrics['U100_W5_HYDRA_REPLAY'][role]['macro_ovr_auc']
        rows.append(row)
    return artifact('RESULTS',parents=dict(spec=spec['content_hash']),rows=rows,pending=pending,
        recovery_anchors='historical_SPORC_M0HLT_OFFLINE',regime='original_full30',independent_test=False)


def print_results(value):
    print('CE1 + U100 embedding5 + independent OFFLINE identification. Original full30; native HLT inference.')
    print('Historical SPORC recovery anchors; mixed-site development validation, not independent test.')
    for role in ('validation','original50k'):
        print('\n'+role)
        print(f'{"model":24} {"site":27} {"ep":>3} {"AUC":>12} {"rec%":>7} {"delta seed control":>19}')
        for row in value['rows']:
            rec = row[role+'_recovery']['macro_ovr_auc']; delta = row[role+'_delta_matched']
            diff = '--' if delta is None else f'{delta:+.8f}'
            print(f'{row["model"]:24} {row["site"]:27} {str(row["selected_epoch"] or "-"):>3} '
                  f'{row[role]["macro_ovr_auc"]:12.8f} {rec if rec is not None else float("nan"):7.1f} {diff:>19}')
    print('Pending/incomplete: '+(', '.join(value['pending']) or 'none'))
