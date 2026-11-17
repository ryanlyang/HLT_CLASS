"""Preserve scientific artifacts while replacing a predictive storage admission rule."""
import importlib
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, with_content_hash, validate_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullsim.alljets_data import FLAGS

PARENT_COMMIT = '3ccc35ae709c63d8833cb0388c618b53dcde5b9e'
MESSAGE = 'OSError: Insufficient remaining stability output allowance plus one GiB reserve'
SALVAGE = {'R_L25_s9': '180544', 'R256_s9': '180546', 'C512_s8': '180557'}
RESTART = {'DEC2_s8': '180559', 'PROJ_s9': '180560', 'C_L50_s9': '180561',
           'R_L50_s9': '180562', 'C512_s9': '180563', 'R512_s9': '180564', 'DEC2_s9': '180565'}
FAILED = {**SALVAGE, **RESTART}
POLICY = 'available_space_advisory_actual_io_errors_fatal/v1'
AUTHORIZATION = 'AUTHORIZE_STABILITY_STORAGE_SEVEN_RESTARTS_THREE_SALVAGES_V1'
MODULES = ('__init__', 'core', 'submission', 'runtime', 'reporting')
REQUIRED = tuple('src/hlt_classification/luka_stability_storage_recovery/' + n + '.py' for n in MODULES) + (
    'scripts/luka_stability_storage_recovery.py', 'scripts/queue_luka_stability_storage_recovery.sh',
    'sbatch/run_luka_stability_storage_recovery.sh', 'tests/test_luka_stability_storage_recovery.py',
    'docs/LUKA_STABILITY_STORAGE_RECOVERY.md')


def artifact(kind, **fields):
    return with_content_hash(dict(contract='LUKA_STABILITY_STORAGE_' + kind + '/v1', schema_version=1, **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract='LUKA_STABILITY_STORAGE_' + kind + '/v1')
    if value.get('schema_version') != 1 or any(value.get(k) != v for k, v in FLAGS.items()):
        raise ValueError('Storage recovery scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Storage recovery lineage differs')


def source(project, commit):
    project = Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or Path(__file__).resolve() != project / REQUIRED[1]:
        raise ValueError('Use the exact pushed storage recovery checkout')
    snap = capture_source_snapshot(project, require_clean=True)
    names = subprocess.check_output(['git', '-C', str(project), 'ls-files'], text=True).splitlines()
    if snap['git_commit'] != commit or not set(REQUIRED) <= set(names):
        raise ValueError('Commit the complete storage recovery driver')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'], check=True, capture_output=True)
    return snap


def parent(ref, *, full=False):
    p = load_json(validate_file_ref(ref))
    if p['source_commit'] != PARENT_COMMIT:
        raise ValueError('Only the original 3ccc35ae stability execution is supported')
    folder = Path(p['project_dir']) / 'src/hlt_classification/luka_reconstruction_stability_hydra'
    name = 'stability_storage_parent'
    if name not in sys.modules:
        loader = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(loader); sys.modules[name] = module; loader.loader.exec_module(module)
    elif Path(sys.modules[name].__file__).resolve() != folder / '__init__.py':
        raise ValueError('Ambiguous stability parent runtime')
    runtime = lambda part: importlib.import_module(name + '.' + part)
    old = runtime('campaign').check(p, full=full, check_storage=False)
    return p, old, runtime


def accounting(jobs):
    if not jobs: return {}
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs.values()),
                             '-o', 'JobIDRaw,State,ExitCode'], check=True, capture_output=True, text=True)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 3 and fields[0] in jobs.values():
            if fields[0] in rows: raise ValueError('Duplicate Slurm accounting row')
            rows[fields[0]] = fields[1:3]
    return {task: rows.get(job, ['ACCOUNTING_PENDING', '']) for task, job in jobs.items()}


def evidence(p, runtime, *, states=None):
    """Authenticate exact old failure IDs and separate complete fits from empty starts."""
    root = Path(p['root']); jobs = runtime('submission').accepted_jobs(p, 'science_pascal')
    if any(jobs.get(t) != j for t, j in FAILED.items()):
        raise ValueError('Original failure jobs differ from the exact submission journal')
    states = accounting(FAILED) if states is None else states
    records = {}
    for task, job in FAILED.items():
        if states.get(task) != ['FAILED', '1:0']:
            raise ValueError('Original job must be terminal FAILED 1:0: ' + task)
        folder = root / task; fit = folder / 'fit'; log = root / f'{task}-{job}.out'
        text = log.read_text(errors='replace')
        errors = [re.sub(r'^\[rank\d+\]:\s*', '', line).strip() for line in text.splitlines()
                  if re.search(r'\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):', line)]
        errors = [line for line in errors if 'CalledProcessError:' not in line and 'ChildFailedError:' not in line]
        if MESSAGE not in errors or any(line != MESSAGE for line in errors):
            raise ValueError('Failure includes an unrecognized cause: ' + task)
        if (folder / 'receipt.json').exists(): raise ValueError('Old task already has a completion receipt: ' + task)
        refs = [file_ref(log)]
        if task in SALVAGE:
            report, _ = runtime('training').validate_fit(fit, p, task)
            startup, _ = runtime('training').validate_fit(folder / 'startup', p, task, technical=True)
            memory = load_json(folder / 'startup_memory.json')
            runtime('contracts').validate(memory, 'STARTUP', dict(spec=p['content_hash']))
            if memory['task'] != task or memory['miniature'] != startup['content_hash'] or memory['discarded'] is not True:
                raise ValueError('Startup lineage differs: ' + task)
            intent = load_json(root / 'claims' / task / 'intent.json')
            if intent != runtime('contracts').artifact('RUN_CLAIM', parents=dict(spec=p['content_hash']), task=task, job_id=job):
                raise ValueError('Original claim differs: ' + task)
            refs.extend(file_ref(path) for path in (fit / 'receipt.json', folder / 'startup/receipt.json',
                                                   folder / 'startup_memory.json', root / 'claims' / task / 'intent.json'))
            records[task] = dict(job=job, action='reuse_completed_fit', report=report['content_hash'], references=refs)
        else:
            # These seven failed at the first campaign check, before the claim.
            # Do not silently restart a different partially trained failure.
            if folder.exists() or (root / 'claims' / task).exists():
                raise ValueError('Restart requires an unclaimed, absent old output directory: ' + task)
            records[task] = dict(job=job, action='restart_from_scratch', references=refs)
    return records


def observe_space(root, estimate=0):
    """Advisory only: no estimate, reserve, or budget rejects valid work."""
    root = Path(root)
    available = shutil.disk_usage(root if root.exists() else root.parent).free
    print(f'STORAGE advisory: available={available / 2**30:.2f} GiB; estimated new outputs={estimate / 2**30:.2f} GiB; no predictive admission gate.', flush=True)
    return dict(available_bytes=available, estimated_new_output_bytes=estimate, policy=POLICY)


def fresh_root(root, protected):
    root = Path(root).absolute()
    if (root.exists() or not root.parent.is_dir() or root.resolve() != root
        or any(p.is_symlink() for p in (root, *root.parents))
        or any(root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh, separate recovery root')
    return root


def create(root, parent_spec, project, commit):
    ref = file_ref(Path(parent_spec).resolve()); p, old, runtime = parent(ref, full=True)
    snap = source(project, commit)
    accepted = runtime('worker').completed_setup(p, 'pascal')
    records = evidence(p, runtime)
    root = fresh_root(root, [p['root'], p['project_dir'], project, *[old[k] for k in
                      ('root', 'project_dir', 'environment_dir', 'bundle_root', 'raw_root', 'delta_root')]])
    estimate = len(RESTART) * p['storage_inventory']['estimated_fit_bytes'] + 64 * 1024**2
    space = observe_space(root, estimate)
    spec = artifact('SPEC', parents=dict(original=p['content_hash'], acceptance=accepted['content_hash'], source=snap['content_hash']),
                    parent_spec=ref, source_snapshot=snap, source_commit=commit, project_dir=str(Path(project).resolve()),
                    root=str(root), environment_dir=p['environment_dir'], storage_policy=POLICY,
                    salvaged=list(SALVAGE), tasks=list(RESTART), failures=records,
                    acceptance=file_ref(Path(p['root']) / 'setup_pascal/acceptance.json'),
                    minutes={t: accepted['minutes'][t] for t in RESTART}, storage_observation=space)
    root.mkdir(); write_immutable_json(root / 'recovery_spec.json', spec)
    return spec


def check(spec, *, full=False):
    validate(spec, 'SPEC'); p, old, runtime = parent(spec['parent_spec'], full=full)
    snap = source(spec['project_dir'], spec['source_commit'])
    accepted = load_json(validate_file_ref(spec['acceptance']))
    runtime('contracts').validate(accepted, 'ACCEPTANCE')
    if (spec['parents'] != dict(original=p['content_hash'], acceptance=accepted['content_hash'], source=snap['content_hash'])
        or spec['source_snapshot'] != snap or spec['environment_dir'] != p['environment_dir']
        or spec['storage_policy'] != POLICY or spec['tasks'] != list(RESTART) or spec['salvaged'] != list(SALVAGE)
        or spec['minutes'] != {t: accepted['minutes'][t] for t in RESTART}
        or spec['acceptance'] != file_ref(Path(p['root']) / 'setup_pascal/acceptance.json')
        or load_json(Path(spec['root']) / 'recovery_spec.json') != spec or set(spec['failures']) != set(FAILED)):
        raise ValueError('Immutable storage recovery registration differs')
    for task, value in spec['failures'].items():
        if value['job'] != FAILED[task] or value['action'] != ('reuse_completed_fit' if task in SALVAGE else 'restart_from_scratch'):
            raise ValueError('Recovery action differs')
        for ref in value['references']: validate_file_ref(ref)
    return p, old, runtime
