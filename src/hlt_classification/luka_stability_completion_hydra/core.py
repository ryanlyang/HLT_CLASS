"""Authenticate four finished fits and the measured, time-limited SOFT setup."""
import importlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, with_content_hash, validate_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.luka_fullsim.alljets_data import FLAGS

ORIGINAL = '3ccc35ae709c63d8833cb0388c618b53dcde5b9e'
STORAGE = '794b8f8016f9b47a3368bc223cecdf01d4d345f4'
SOFT = 'd28c403360f7b4662f679f5564529a048d8db5e2'
SALVAGE = {'DEC2_s7': '180553', 'PROJ_s7': '180548', 'PROJ_s8': '180554', 'R512_s8': '180558'}
TASKS = ('SOFT_s7', 'SOFT_s8', 'SOFT_s9')
SETUP_JOB = '183850'
RELEASE_JOB = '183851'
MAX_MINUTES = 7 * 24 * 60
POLICY = 'measured_soft_walltime_2x_plus_30min_bounded_7days/v1'
STORAGE_POLICY = 'available_space_advisory_actual_io_errors_fatal/v1'
AUTHORIZATION = 'AUTHORIZE_STABILITY_COMPLETION_FOUR_SALVAGES_THREE_SOFT_V1'
MODULES = ('__init__', 'core', 'audit', 'submission', 'runtime', 'reporting')
REQUIRED = tuple('src/hlt_classification/luka_stability_completion_hydra/' + n + '.py' for n in MODULES) + (
    'scripts/luka_stability_completion_hydra.py', 'scripts/queue_luka_stability_completion_hydra.sh',
    'sbatch/run_luka_stability_completion_hydra.sh', 'tests/test_luka_stability_completion_hydra.py',
    'docs/LUKA_STABILITY_COMPLETION_HYDRA.md')


def artifact(kind, **fields):
    return with_content_hash(dict(contract='LUKA_STABILITY_COMPLETION_' + kind + '/v1', schema_version=1, **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract='LUKA_STABILITY_COMPLETION_' + kind + '/v1')
    if value.get('schema_version') != 1 or any(value.get(k) != v for k, v in FLAGS.items()):
        raise ValueError('Completion evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Completion lineage differs')


def source(project, commit):
    project = Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit) or Path(__file__).resolve() != project / REQUIRED[1]:
        raise ValueError('Use the exact pushed completion checkout')
    snap = capture_source_snapshot(project, require_clean=True)
    names = subprocess.check_output(['git', '-C', str(project), 'ls-files'], text=True).splitlines()
    if snap['git_commit'] != commit or not set(REQUIRED) <= set(names):
        raise ValueError('Commit the complete recovery implementation')
    subprocess.run(['git', '-C', str(project), 'merge-base', '--is-ancestor', commit, 'origin/main'], check=True, capture_output=True)
    return snap


def runtime(spec, package, alias, commit):
    if spec['source_commit'] != commit:
        raise ValueError('Unexpected parent execution commit')
    folder = Path(spec['project_dir']) / 'src/hlt_classification' / package
    if alias not in sys.modules:
        loader = importlib.util.spec_from_file_location(alias, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(loader); sys.modules[alias] = module; loader.loader.exec_module(module)
    elif Path(sys.modules[alias].__file__).resolve() != folder / '__init__.py':
        raise ValueError('Ambiguous pinned runtime')
    return lambda part: importlib.import_module(alias + '.' + part)


def parents(spec, *, full=False):
    storage = load_json(validate_file_ref(spec['storage_spec']))
    sr = runtime(storage, 'luka_stability_storage_recovery', 'completion_storage_parent', STORAGE)
    original, old, original_runtime = sr('core').check(storage, full=full)
    soft = load_json(validate_file_ref(spec['soft_spec']))
    fr = runtime(soft, 'luka_reconstruction_stability_hydra', 'completion_soft_parent', SOFT)
    soft_old = fr('campaign').check(soft, full=full, check_storage=False)
    if (original['source_commit'] != ORIGINAL or fr('recovery').parent(soft) != original or soft_old != old
        or soft['environment_dir'] != storage['environment_dir'] or soft['tasks'] != list(TASKS)):
        raise ValueError('Recoveries do not share the exact original experiment')
    return original, storage, soft, old, original_runtime, sr, fr


def check(spec, *, full=False):
    validate(spec, 'SPEC'); values = parents(spec, full=full)
    original, storage, soft, _, _, _, _ = values
    snap = source(spec['project_dir'], spec['source_commit'])
    qualification = load_json(validate_file_ref(spec['qualification']))
    expected = dict(original=original['content_hash'], storage=storage['content_hash'], soft=soft['content_hash'],
                    source=snap['content_hash'], qualification=qualification['content_hash'])
    if (spec['parents'] != expected or spec['source_snapshot'] != snap or spec['tasks'] != list(TASKS)
        or spec['salvaged'] != SALVAGE or spec['walltime_policy'] != POLICY or spec['max_minutes'] != MAX_MINUTES
        or spec['storage_policy'] != STORAGE_POLICY or spec['environment_dir'] != soft['environment_dir']
        or spec['minutes'] != qualification['minutes'] or spec['qualification']['path'] != str(Path(spec['root']) / 'qualification.json')
        or load_json(Path(spec['root']) / 'recovery_spec.json') != spec):
        raise ValueError('Immutable completion registration differs')
    from .audit import check_record
    check_record(qualification, values)
    return values


def create(root, storage_spec, soft_spec, project, commit):
    refs = dict(storage_spec=file_ref(Path(storage_spec).resolve()), soft_spec=file_ref(Path(soft_spec).resolve()))
    values = parents(refs, full=True)
    original, storage, soft, old, _, sr, _ = values
    snap = source(project, commit)
    from .audit import qualify
    qualification = qualify(values)
    protected = [p[k] for p in (original, storage, soft) for k in ('root', 'project_dir')]
    protected += [project, *[old[k] for k in ('root', 'project_dir', 'environment_dir', 'bundle_root', 'raw_root', 'delta_root')]]
    root = sr('core').fresh_root(root, protected)
    observation = sr('core').observe_space(root, qualification['storage']['scientific_bytes'] + 64 * 1024**2)
    root.mkdir()
    write_immutable_json(root / 'qualification.json', qualification)
    value = artifact('SPEC', parents=dict(original=original['content_hash'], storage=storage['content_hash'], soft=soft['content_hash'],
        source=snap['content_hash'], qualification=qualification['content_hash']), **refs,
        root=str(root), project_dir=str(Path(project).resolve()), source_commit=commit, source_snapshot=snap,
        environment_dir=soft['environment_dir'], tasks=list(TASKS), salvaged=SALVAGE,
        qualification=file_ref(root / 'qualification.json'), minutes=qualification['minutes'],
        max_minutes=MAX_MINUTES, walltime_policy=POLICY, storage_policy=STORAGE_POLICY, storage_observation=observation)
    write_immutable_json(root / 'recovery_spec.json', value)
    return value
