"""Shared export, twenty native final fits, retained reporting and owned retirement."""
from pathlib import Path
import os
import platform
import re
import subprocess
import sys
import numpy as np
import torch
from hlt_classification.data.cache_contracts import (load_json, load_npz_arrays, write_immutable_json,
    atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.context_fusion.worker import save_state
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.jetclass2_delphes.runner import predict
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_attention.worker import execution as gpu_execution
from hlt_classification.luka_embedding_weight.worker import storage, validate_predictions
from hlt_classification.luka_trial_hybrid.worker import data, native, release
from hlt_classification.luka_fullsim import regular_cache
from hlt_classification.luka_fullsim.preflight_v2 import PhaseLog, memory_snapshot
from .contracts import artifact, validate, WEIGHTS, TEACHERS, REFERENCES
from . import campaign, source, submission, banks, training, startup, reference

FIT_FILES = ('selected.pt', 'training_report.json', 'validation_predictions.npz', 'receipt.json')


def execution(spec, task):
    job = os.environ.get('SLURM_JOB_ID', ''); site = spec['execution_site']
    if not re.fullmatch('[1-9][0-9]*', job):
        raise PermissionError('A real Slurm allocation is required')
    details = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, capture_output=True, text=True).stdout
    fields = dict(x.split('=', 1) for x in details.split() if '=' in x)
    if (fields.get('JobName') != 'lukaax_'+task or fields.get('Comment') != 'lukaax:'+spec['content_hash']+':'+task
            or fields.get('WorkDir') != spec['project_dir']):
        raise PermissionError('Exact task identity differs')
    if task != 'report':
        return gpu_execution(spec, spec['reuse']['acceptance']['preflight'])
    prefix = site['conda_base']+'/envs/'+site['conda_env']
    if (fields.get('Account') != site['account'] or fields.get('Partition') != site['partition'] or fields.get('QOS') != site['qos']
            or fields.get('NumNodes') != '1' or fields.get('NumTasks') != '1' or fields.get('NumCPUs') != str(spec['report_cpus'])
            or os.environ.get('SLURM_CLUSTER_NAME') != site['cluster'] or platform.machine() != site['architecture']
            or os.environ.get('SLURM_CPUS_PER_TASK') != str(spec['report_cpus'])
            or os.environ.get('SLURM_MEM_PER_NODE') != str(spec['report_memory_mb'])
            or sys.prefix != prefix or os.environ.get('CONDA_PREFIX') != prefix or os.environ.get('PYTHONNOUSERSITE') != '1'):
        raise PermissionError('Report allocation/environment differs')
    return job


def comparison_refs(spec):
    return dict(spec['reuse']['references'], K5_EMBED5=spec['reference']['predictions'])


def snapshot_references(folder, spec, val):
    folder.mkdir(exist_ok=False)
    for name, ref in comparison_refs(spec).items():
        validate_file_ref(ref); a = load_npz_arrays(ref['path']); validate_predictions(a, len(val))
        if not np.array_equal(a['identities'], val.identities) or not np.array_equal(a['labels'], val.labels):
            raise ValueError('Historical reference validation population differs')
        raw = Path(ref['path']).read_bytes(); validate_file_ref(ref)
        atomic_publish_bytes(folder/(name+'.npz'), raw)
    ref = spec['reuse']['subset']; validate_file_ref(ref)
    raw = Path(ref['path']).read_bytes(); validate_file_ref(ref)
    atomic_publish_bytes(folder/'original50k.npz', raw)


def output_names(task):
    if task == 'targets':
        return (['targets/references/'+n+'.npz' for n in (*REFERENCES, 'original50k')]
            + ['targets/banks/'+t+'/'+f for t in TEACHERS for f in ('manifest.json', 'targets.npz')]
            + ['targets/bundle.json', 'targets/startup.json', 'targets/acquisition.json', 'targets/costs.json'])
    if task not in WEIGHTS:
        raise ValueError('Unknown fit task')
    return [task+'/fit/'+f for f in FIT_FILES]+[task+'/costs.json']


def target_payloads():
    return ['targets/banks/'+t+'/targets.npz' for t in TEACHERS]


def fit(folder, spec, arm, tr, val, bank, device):
    folder.mkdir(exist_ok=False)
    student, report, state = training.train(tr, val, bank=bank, node=spec['nodes'][arm], device=device)
    training.validate_report(report, spec, arm, bank.manifest['content_hash'])
    expected = predict(student, val, device=device, batch_size=128)
    save_state(folder/'selected.pt', state)
    saved = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
    if saved.keys() != state.keys() or any(not torch.equal(saved[k], state[k]) for k in state):
        raise ValueError('Native checkpoint byte roundtrip differs')
    del student; release()
    restored = native(file_ref(folder/'selected.pt'), device)
    actual = predict(restored, val, device=device, batch_size=128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if evaluate_probabilities(val.labels, actual) != report['validation']:
        raise ValueError('Selected main-head metrics differ after native extraction')
    atomic_publish_bytes(folder/'validation_predictions.npz', deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=actual)))
    write_immutable_json(folder/'training_report.json', report)
    receipt = artifact('FIT', parents=dict(spec=spec['content_hash'], bundle=bank.manifest['content_hash']), arm=arm,
        checkpoint_probability_roundtrip=True, outputs=[file_ref(folder/f, root=folder) for f in FIT_FILES[:-1]])
    write_immutable_json(folder/'receipt.json', receipt)
    del restored; release()


def validate_fit(spec, arm):
    root = Path(spec['root']); folder = root/arm/'fit'; bundle = load_json(root/'targets/bundle.json')
    row = load_json(folder/'receipt.json'); validate(row, 'FIT', dict(spec=spec['content_hash'], bundle=bundle['content_hash']))
    if (row['arm'] != arm or row['checkpoint_probability_roundtrip'] is not True
            or [r['path'] for r in row['outputs']] != list(FIT_FILES[:-1])):
        raise ValueError('Fit output registration differs')
    for ref in row['outputs']:
        validate_file_ref(ref, root=folder)
    report = load_json(folder/'training_report.json'); training.validate_report(report, spec, arm, bundle['content_hash'])
    arrays = load_npz_arrays(folder/'validation_predictions.npz'); validate_predictions(arrays, spec['counts']['validation'])
    if evaluate_probabilities(arrays['labels'], arrays['probabilities']) != report['validation']:
        raise ValueError('Saved main-head prediction metrics differ')
    return report, arrays


def validate_task(spec, task, seen=None, *, retired=None):
    seen = {} if seen is None else seen
    if task in seen:
        return seen[task]
    root = Path(spec['root']); row = load_json(root/task/'receipt.json')
    deps = next(r['dependencies'] for r in spec['tasks'] if r['task_id'] == task)
    parents = dict(spec=spec['content_hash'], **{d: validate_task(spec, d, seen, retired=retired)['content_hash'] for d in deps})
    validate(row, 'TASK', parents)
    if (row['task'] != task or not re.fullmatch('[1-9][0-9]*', row['job_id'])
            or row['job_id'] != submission.accepted_jobs(spec).get(task)
            or [r['path'] for r in row['outputs']] != output_names(task)):
        raise ValueError('Task output/journal identity differs')
    removed = {} if retired is None else {r['path']: r for r in retired['payloads']}
    for ref in row['outputs']:
        if ref['path'] in removed:
            if ref != removed[ref['path']] or (root/ref['path']).exists() or (root/ref['path']).is_symlink():
                raise ValueError('Retired payload descriptor/absence differs')
        else:
            validate_file_ref(ref, root=root)
    costs = load_json(root/task/'costs.json'); validate(costs, 'COSTS', dict(spec=spec['content_hash']))
    if (costs['task'] != task or costs['job_id'] != row['job_id'] or costs['separate_preflight_performed'] is not False
            or costs['environment'] != spec['anchor']['evidence']['environment']):
        raise ValueError('Task runtime evidence differs')
    bundle = load_json(root/'targets/bundle.json')
    if task == 'targets':
        banks.validate_bundle(bundle, spec, payloads=retired is None)
        startup.check(load_json(root/'targets/startup.json'), spec, bundle['content_hash'])
        reference.check_acquisition(load_json(root/'targets/acquisition.json'), spec)
        originals = dict(comparison_refs(spec), original50k=spec['reuse']['subset'])
        saved = {ref['path']: ref for ref in row['outputs']}
        for name, ref in originals.items():
            copy = saved['targets/references/'+name+'.npz']
            if any(copy[k] != ref[k] for k in ('sha256', 'bytes')):
                raise ValueError('Reference snapshot no longer matches frozen c50 import')
    else:
        validate_fit(spec, task)
    seen[task] = row
    return row


def retired_evidence(spec):
    root = Path(spec['root'])
    if not (root/'report/retirement.json').exists():
        return None
    if not (root/'report/complete.json').is_file():
        raise ValueError('Partial target retirement; preserve evidence and request explicit recovery')
    r = load_json(root/'report/retirement.json'); validate(r, 'RETIREMENT', dict(spec=spec['content_hash']))
    done = load_json(root/'report/complete.json'); validate(done, 'COMPLETE', dict(spec=spec['content_hash'], retirement=r['content_hash']))
    if ([v['path'] for v in r['payloads']] != target_payloads() or done['job_id'] != submission.accepted_jobs(spec).get('report')
            or r['results'] != done['results'] or done['retirement'] != file_ref(root/'report/retirement.json', root=root)
            or done['tasks'] != {t: load_json(root/t/'receipt.json')['content_hash'] for t in ('targets', *WEIGHTS)}):
        raise ValueError('Retirement completion/coverage differs')
    validate_file_ref(done['results'], root=root)
    return r


def results(spec):
    campaign.validate_spec(spec, imports=False); root = Path(spec['root']); seen = {}; retired = retired_evidence(spec)
    validate_task(spec, 'targets', seen, retired=retired)
    refs = root/'targets/references'; arrays = {n: load_npz_arrays(refs/(n+'.npz')) for n in REFERENCES}
    anchor = arrays['M0HLT']; ix = load_npz_arrays(refs/'original50k.npz')['indices']
    if (ix.dtype != np.int64 or ix.shape != (spec['original50k_rows'],) or len(np.unique(ix)) != len(ix)
            or np.any(ix < 0) or np.any(ix >= len(anchor['labels']))):
        raise ValueError('Report-only original50k subset differs')
    pending, picks = [], {}
    for arm in WEIGHTS:
        if not (root/arm/'receipt.json').is_file():
            pending.append(arm); continue
        validate_task(spec, arm, seen, retired=retired)
        report, arrays[arm] = validate_fit(spec, arm); picks[arm] = report['selected_pass']
    rows = []
    for name, a in arrays.items():
        validate_predictions(a, spec['counts']['validation'])
        if not np.array_equal(a['identities'], anchor['identities']) or not np.array_equal(a['labels'], anchor['labels']):
            raise ValueError('Comparison population differs')
        rows.append(dict(model=name, selected_epoch=picks.get(name), input='ORACLE' if name == 'OFFLINE' else 'single HLT',
            validation=evaluate_probabilities(a['labels'], a['probabilities']),
            original50k=evaluate_probabilities(a['labels'][ix], a['probabilities'][ix])))
    baseline = {r['model']: r for r in rows}
    for row in rows:
        for role in ('validation', 'original50k'):
            row[role+'_recovery'] = recovery(row[role], baseline['M0HLT'][role], baseline['OFFLINE'][role])
            row[role+'_delta_c50'] = row[role]['macro_ovr_auc']-baseline['C50_DE000'][role]['macro_ovr_auc']
            row[role+'_delta_k5'] = row[role]['macro_ovr_auc']-baseline['K5_EMBED5'][role]['macro_ovr_auc']
    return artifact('RESULTS', parents=dict(spec=spec['content_hash'], **{t: r['content_hash'] for t, r in seen.items()}),
                    rows=rows, pending=pending, counts=spec['counts'], single_training_seed=True, original50k_selection=False)


def retire(spec, result, job):
    if result['pending']:
        raise ValueError('Cannot retire before all twenty consumers finish')
    root = Path(spec['root']).absolute()
    if root != root.resolve(strict=True):
        raise PermissionError('Noncanonical owned root')
    folder = root/'report'; write_immutable_json(folder/'results.json', result)
    refs, paths = [], []
    for name in target_payloads():
        path = root/name
        if path.is_symlink() or path != path.resolve(strict=True) or not path.resolve().is_relative_to(root):
            raise PermissionError('Retirement escapes exact newly owned canonical target')
        ref = file_ref(path, root=root)
        if ref['path'] != name:
            raise PermissionError('Retirement alias differs')
        validate_file_ref(ref, root=root); refs.append(ref); paths.append(path)
    intent = artifact('RETIREMENT', parents=dict(spec=spec['content_hash']), payloads=refs,
        results=file_ref(folder/'results.json', root=root), reason='twenty_consumers_complete_models_and_results_retained')
    write_immutable_json(folder/'retirement.json', intent)
    for path, ref in zip(paths, refs):
        if path.is_symlink() or path != path.resolve(strict=True):
            raise PermissionError('Retirement target changed')
        validate_file_ref(ref, root=root); path.unlink()
    done = artifact('COMPLETE', parents=dict(spec=spec['content_hash'], retirement=intent['content_hash']), job_id=job,
        results=intent['results'], retirement=file_ref(folder/'retirement.json', root=root),
        tasks={t: load_json(root/t/'receipt.json')['content_hash'] for t in ('targets', *WEIGHTS)})
    write_immutable_json(folder/'complete.json', done)
    print('Retired four newly owned TRAIN target payloads; no backup. Models, results and moments retained.', flush=True)
    return done


def run(spec, task, *, device='cuda'):
    if task not in {r['task_id'] for r in spec['tasks']}:
        raise ValueError('Unknown task')
    campaign.check(spec); job = execution(spec, task)
    if submission.accepted_jobs(spec).get(task) != job:
        raise ValueError('Allocation not accepted for exact task')
    deps = next(r['dependencies'] for r in spec['tasks'] if r['task_id'] == task)
    parents = dict(spec=spec['content_hash']); seen = {}
    for dep in deps:
        rec = validate_task(spec, dep, seen); submission.successful(spec, dep, rec['job_id'])
        parents[dep] = rec['content_hash']
    if installed_environment() != spec['anchor']['evidence']['environment']:
        raise ValueError('Installed environment changed')
    storage(spec); root = Path(spec['root']); folder = root/task; folder.mkdir(exist_ok=False)
    if task == 'report':
        result = results(spec)
        if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
            raise ValueError('Source changed before retirement')
        done = retire(spec, result, job)
        if results(spec) != result:
            raise ValueError('Post-retirement result authentication differs')
        print_results(result); return done
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    log = PhaseLog(folder, lambda: memory_snapshot(torch))
    meta = regular_cache.metadata(spec['anchor']['evidence']['prepared_root'])
    if task == 'targets':
        with log.phase('authenticate_and_acquire_shared_targets'):
            acquired = reference.acquire(spec)
            write_immutable_json(folder/'acquisition.json', acquired)
            print('Target acquisition: '+acquired['mode'], flush=True)
        if acquired['mode'] == 'export_after_authenticated_retirement':
            for teacher in TEACHERS:
                with log.phase('export_frozen_c50_'+teacher+'_TRAIN'):
                    record = banks.record(spec, teacher); model = native(record['files']['checkpoint'], device)
                    tr = data(spec, 'train', record['coordinate'], meta)
                    banks.export(spec, teacher, model, tr, device=device)
                    del model, tr; release()
        with log.phase('shared_references_and_disposable_safety_checks'):
            tr, val = (data(spec, role, 'D000', meta) for role in ('train', 'validation'))
            bank = banks.Bundle(spec, tr)
            write_immutable_json(folder/'bundle.json', bank.manifest)
            snapshot_references(folder/'references', spec, val)
            checked = startup.run(spec, tr, val, bank, device); startup.check(checked, spec, bank.manifest['content_hash'])
            write_immutable_json(folder/'startup.json', checked)
            del tr, val, bank; release()
    else:
        with log.phase('load_shared_TRAIN_targets_and_HLT_views'):
            tr, val = (data(spec, role, 'D000', meta) for role in ('train', 'validation'))
            bank = banks.Bundle(spec, tr)
            if bank.manifest != load_json(root/'targets/bundle.json'):
                raise ValueError('Shared target identity changed')
        with log.phase('fresh_fixed30_CE_only_head_science'):
            fit(folder/'fit', spec, task, tr, val, bank, device)
        del tr, val, bank; release()
    costs = artifact('COSTS', parents=dict(spec=spec['content_hash']), task=task, job_id=job,
        environment=installed_environment(), memory=memory_snapshot(torch), separate_preflight_performed=False)
    write_immutable_json(folder/'costs.json', costs); storage(spec)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during execution')
    receipt = artifact('TASK', parents=parents, task=task, job_id=job,
        outputs=[file_ref(root/p, root=root) for p in output_names(task)])
    write_immutable_json(folder/'receipt.json', receipt); validate_task(spec, task)
    return receipt


def print_results(value):
    print('Frozen c50 teachers; twenty fresh DE000 fits, main CE0.5; KD/embedding/relation grid.')
    print('Exploratory development validation; NO independent test. Original50k report-only. Native HLT inference.')
    for role in ('validation', 'original50k'):
        print('\n'+role)
        print(f'{"model":22} {"pick":>5} {"accuracy":>10} {"AUC":>10} {"AUC rec%":>10} {"delta K5/E5":>12}')
        for row in value['rows']:
            m = row[role]; rec = row[role+'_recovery']['macro_ovr_auc']; text = 'n/a' if rec is None else f'{rec:.1f}'
            print(f'{row["model"]:22} {str(row["selected_epoch"] or "-"):>5} {m["accuracy"]:10.6f} '
                  f'{m["macro_ovr_auc"]:10.6f} {text:>10} {row[role+"_delta_k5"]:+12.8f}')
    print('Pending: '+(', '.join(value['pending']) or 'none'))
