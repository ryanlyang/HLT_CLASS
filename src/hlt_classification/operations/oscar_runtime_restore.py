"""Reconstruct a campaign runtime in HOME using only stdlib in the driver.

The damaged prefix is metadata-only input. Never activate it, clone its files,
solve new package versions, mutate a campaign, or submit a job here.
"""
from __future__ import annotations

import base64
import csv
from email.parser import BytesParser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
import zipfile

HOME_BASE = Path('/oscar/home/rlyang/hlt_classification')
TARGET = HOME_BASE / 'environments/new_balanced_oscar_v1'
SCRATCH = Path('/oscar/scratch/rlyang/hlt_classification/runtime_downloads')
GIB = 2**30
RUNTIME_LIMIT = 12 * GIB
RESULT_GIB = 14
MIN_AVAILABLE_GIB = 28  # 12 runtime + 14 results + 2 margin


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def hashed(value):
    body = {k: v for k, v in value.items() if k != 'content_hash'}
    return dict(body, content_hash=digest(body))


def read_hashed(path):
    value = json.loads(Path(path).read_bytes())
    require(value == hashed(value), 'Content hash differs: ' + str(path))
    return value


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    """Publish completion last, atomically, without replacing existing evidence."""
    path = Path(path)
    require(not os.path.lexists(path), 'Existing record: ' + str(path))
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(hashed(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def normalized(name):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)), 'Invalid package name')
    return re.sub(r'[-_.]+', '-', name).lower()


def version_string(value):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_!-]*', value)), 'Invalid package version')
    return value


def inventory(spec_path):
    """Read exact Conda URLs and pip METADATA even if Python cannot initialize."""
    spec = read_hashed(spec_path)
    require(spec.get('contract') == 'LUKA_NEW_BALANCED_OSCAR_SPEC/v1', 'Unexpected campaign')
    require(Path(spec['root']).resolve() == Path(spec_path).resolve().parent, 'Spec root differs')
    expected = spec['environment']
    require(expected == hashed(expected), 'Runtime fingerprint hash differs')
    require(expected['architecture'] == 'x86_64' and expected['python'].startswith('3.10.'),
            'Only this registered Linux Python 3.10 runtime is supported')
    require(expected['torch_cuda'] == '11.8' and expected['versions']['torch'].endswith('+cu118'),
            'Expected recorded CUDA 11.8 wheel')
    old = Path(spec['environment_dir'])
    conda = []
    owned = set()
    names = set()
    for path in sorted((old / 'conda-meta').glob('*.json')):
        record = json.loads(path.read_bytes())
        # Conda has legitimate private names such as _openmp_mutex.
        name = record['name']
        require(bool(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name)), 'Invalid Conda package name')
        require(name not in names, 'Duplicate Conda package: ' + name)
        names.add(name)
        url, md5 = record['url'], record['md5']
        parts = urlsplit(url)
        require(parts.scheme == 'https' and parts.netloc == 'conda.anaconda.org'
                and parts.path.startswith('/conda-forge/') and not parts.query and not parts.fragment,
                'Unapproved Conda package URL: ' + url)
        require(bool(re.fullmatch('[a-f0-9]{32}', md5)), 'Missing Conda MD5: ' + name)
        for filename in record['files']:
            for part in Path(filename).parts:
                if part.endswith('.dist-info'):
                    owned.add(part)
        conda.append(dict(name=record['name'], version=record['version'], build=record['build'],
                          url=url, md5=md5, record_sha256=sha_file(path)))
    python = [r for r in conda if r['name'] == 'python']
    require(len(python) == 1 and python[0]['version'] == expected['python'], 'Python build differs')
    require({'pip', 'setuptools', 'wheel'} <= names, 'Incomplete Conda bootstrap records')
    packages = {}
    for folder in sorted((old / 'lib/python3.10/site-packages').glob('*.dist-info')):
        if folder.name in owned:
            continue  # Conda restores these, including their missing metadata.
        path = folder / 'METADATA'
        require(path.is_file() and (folder / 'RECORD').is_file(), 'Unidentifiable distribution: ' + str(folder))
        message = BytesParser().parsebytes(path.read_bytes())
        name, version = normalized(message['Name']), version_string(message['Version'])
        require(name not in packages, 'Duplicate pip distribution: ' + name)
        require(folder.name.lower() == (name.replace('-', '_') + '-' + version + '.dist-info').lower(),
                'Distribution directory disagrees with metadata: ' + str(folder))
        packages[name] = dict(version=version, metadata_sha256=sha_file(path),
                              record_sha256=sha_file(folder / 'RECORD'))
    for name, version in expected['versions'].items():
        require(packages.get(normalized(name), {}).get('version') == version,
                'Recorded campaign package differs: ' + name)
    return hashed(dict(contract='LUKA_OSCAR_RUNTIME_RESTORE_PLAN/v1', schema_version=1,
                       parents=dict(spec=spec['content_hash'], environment=expected['content_hash']),
                       spec=str(Path(spec_path).resolve()), old_environment=str(old),
                       expected_environment=expected, conda=conda, pip=packages))


def clean_env(work, target=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'CONDA_', 'SBATCH_', 'SLURM_'))
           and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CONDA_PKGS_DIRS=str(work / 'conda-cache'), TMPDIR=str(work / 'tmp'),
               PIP_CONFIG_FILE=os.devnull, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if target is not None:
        env.update(CONDA_PREFIX=str(target), LD_LIBRARY_PATH=str(target / 'lib'))
        env['PATH'] = str(target / 'bin') + os.pathsep + env.get('PATH', '')
    return env


def pip_command(target, *args):
    return [str(target / 'bin/python'), '-I', '-B', '-m', 'pip', '--disable-pip-version-check',
            '--no-cache-dir', *map(str, args)]


def wheel_lock(wheelhouse, packages, limit=RUNTIME_LIMIT):
    """Bind downloaded wheels and reject extras/incorrect versions before installation."""
    wheels = {}
    unpacked = 0
    for path in sorted(wheelhouse.glob('*.whl')):
        with zipfile.ZipFile(path) as archive:
            metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            require(len(metadata) == 1, 'Ambiguous wheel metadata: ' + path.name)
            record = BytesParser().parsebytes(archive.read(metadata[0]))
            name, version = normalized(record['Name']), version_string(record['Version'])
            require(name in packages and name not in wheels and packages[name]['version'] == version,
                    'Wheel inventory differs: ' + path.name)
            unpacked += sum(entry.file_size for entry in archive.infolist())
        wheels[name] = dict(version=version, filename=path.name, sha256=sha_file(path))
    require(set(wheels) == set(packages), 'Wheel set is incomplete')
    require(unpacked < limit, 'Downloaded packages exceed runtime budget')
    lines = [f"{name}=={r['version']} --hash=sha256:{r['sha256']}" for name, r in sorted(wheels.items())]
    return wheels, unpacked, '\n'.join(lines) + '\n'


def verify_files(prefix, plan):
    """Check every Conda file and hashed pip RECORD payload, not just import torch."""
    records = [json.loads(p.read_bytes()) for p in (prefix / 'conda-meta').glob('*.json')]
    wanted = {(r['name'], r['version'], r['build']) for r in plan['conda']}
    require({(r['name'], r['version'], r['build']) for r in records} == wanted, 'Conda builds changed')
    for record in records:
        for name in record['files']:
            path = prefix / name
            require(path.resolve().is_relative_to(prefix.resolve()) and path.exists(),
                    'Missing or external Conda file: ' + str(path))
    site = prefix / 'lib/python3.10/site-packages'
    for name, package in plan['pip'].items():
        folder = site / (name.replace('-', '_') + '-' + package['version'] + '.dist-info')
        require((folder / 'RECORD').is_file(), 'Missing pip RECORD: ' + name)
        metadata = BytesParser().parsebytes((folder / 'METADATA').read_bytes())
        require(normalized(metadata['Name']) == name and metadata['Version'] == package['version'],
                'Installed pip version differs: ' + name)
        count = 0
        with (folder / 'RECORD').open(newline='', encoding='utf-8') as stream:
            for filename, encoded, size in csv.reader(stream):
                path = (site / filename).resolve()
                require(path.is_relative_to(prefix.resolve()), 'pip RECORD escapes new prefix')
                require(path.is_file(), 'Missing installed pip file: ' + str(path))
                if encoded:
                    algorithm, value = encoded.split('=', 1)
                    require(algorithm == 'sha256', 'Unexpected pip digest algorithm')
                    actual = base64.urlsafe_b64encode(bytes.fromhex(sha_file(path))).decode().rstrip('=')
                    require(actual == value and path.stat().st_size == int(size), 'pip file hash differs: ' + str(path))
                    count += 1
                else:
                    require(path.suffix == '.pyc' or path == folder.resolve() / 'RECORD',
                            'Unhashed pip payload: ' + str(path))
        require(count > 0, 'Empty pip RECORD: ' + name)
        print('FILES VERIFIED:', name, flush=True)


def verify_runtime(target, project, spec_path, env):
    code = """
import json, sys
from pathlib import Path
import ssl, sqlite3, bz2, lzma
import numpy, scipy, awkward, uproot, sklearn, threadpoolctl
import torch
import torch.nn.functional
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from hlt_classification.jetclass2_delphes.model import installed_environment, DelphesParticleTransformer
expected = json.loads(Path(sys.argv[2]).read_bytes())['environment']
actual = installed_environment()
if actual != expected:
    raise SystemExit('STOP: runtime differs from registered fingerprint: ' + repr(actual))
torch.set_num_threads(1)
model = DelphesParticleTransformer().eval()
features = torch.zeros(2, 17, 4)
vectors = torch.ones(2, 4, 4); vectors[:, 3] = 3
logits = model(features, vectors, torch.ones(2, 1, 4, dtype=torch.bool))
assert logits.shape == (2, 11) and torch.isfinite(logits).all()
torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('PASS: exact registered runtime/Weaver sources and synthetic CPU forward/backward.', flush=True)
"""
    subprocess.run([str(target / 'bin/python'), '-I', '-u', '-B', '-c', code, str(project), str(spec_path)],
                   env=env, check=True)


def startup_retry_evidence(spec_path):
    """Only a terminal campaign with no saved training can use the fresh-fit retry."""
    # Deliberately lazy: these imports require the newly repaired scientific stack.
    from hlt_classification.luka_new_balanced_oscar import campaign
    spec = read_hashed(spec_path)
    require(spec.get('contract') == 'LUKA_NEW_BALANCED_OSCAR_SPEC/v1', 'Unexpected retry campaign')
    require(Path(spec['root']).resolve() == Path(spec_path).resolve().parent, 'Retry root differs')
    jobs = campaign.accepted_jobs(spec)
    require(bool(jobs), 'No authenticated previous submission')
    states = {row['job']: campaign.terminal(row['job']) for row in jobs}
    root = Path(spec['root'])
    for task in spec['tasks']:
        output = root / task
        require(not (output / 'result.json').exists()
                and not list(output.rglob('*.pt'))
                and not list((output / 'epochs').glob('*.json')),
                'Saved training exists; use checkpoint recovery instead of fresh fits: ' + task)
    return dict(contract='LUKA_OSCAR_STARTUP_RETRY/v1', schema_version=1,
                parents=dict(spec=spec['content_hash']), jobs=states, no_training_states=True)


def restore(spec_path, project, conda_exe, available_gib):
    require(os.name == 'posix', 'Run this restoration on Oscar, not the local Windows host')
    require(available_gib >= MIN_AVAILABLE_GIB, 'Need 28 GiB HOME headroom including the results reservation')
    spec_path, project, conda_exe = map(lambda p: Path(p).resolve(strict=True), (spec_path, project, conda_exe))
    require(spec_path.is_relative_to(HOME_BASE / 'checkpoints'), 'Unexpected campaign location')
    require(project.is_relative_to(HOME_BASE / 'source') and conda_exe.is_file(), 'Unexpected source/Conda executable')
    require(TARGET.parent.resolve() == TARGET.parent and not os.path.lexists(TARGET),
            'New HOME environment exists or has a redirected parent; inspect it before retrying')
    receipt = Path(str(TARGET) + '.restore')
    require(not os.path.lexists(receipt), 'Restoration records already exist; inspect them before retrying')
    plan = inventory(spec_path)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    receipt.mkdir()
    publish(receipt / 'plan.json', plan)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    require(SCRATCH.resolve() == SCRATCH, 'Scratch download path is redirected')
    work = Path(tempfile.mkdtemp(prefix='new-balanced-', dir=SCRATCH))
    (work / 'tmp').mkdir(); (work / 'wheels').mkdir()
    print('Persistent runtime:', TARGET, '\nBuild records:', receipt, '\nScratch downloads:', work, flush=True)
    try:
        explicit = receipt / 'conda-explicit.txt'
        explicit.write_text('@EXPLICIT\n' + '\n'.join(r['url'] + '#' + r['md5'] for r in plan['conda']) + '\n')
        subprocess.run([str(conda_exe), 'create', '-y', '--copy', '--no-default-packages', '--prefix', str(TARGET),
                        '--file', str(explicit)], env=clean_env(work), check=True)
        env = clean_env(work, TARGET)
        ordinary = {k: v for k, v in plan['pip'].items() if k != 'torch'}
        pins = receipt / 'pip-pins.txt'
        pins.write_text('\n'.join(f"{k}=={v['version']}" for k, v in sorted(ordinary.items())) + '\n')
        subprocess.run(pip_command(TARGET, 'download', '--no-deps', '--only-binary=:all:',
                                  '--index-url', 'https://pypi.org/simple', '--dest', work / 'wheels', '-r', pins),
                       env=env, check=True)
        subprocess.run(pip_command(TARGET, 'download', '--no-deps', '--only-binary=:all:',
                                  '--index-url', 'https://download.pytorch.org/whl/cu118', '--dest', work / 'wheels',
                                  'torch==' + plan['pip']['torch']['version']), env=env, check=True)
        conda_bytes = sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())
        wheels, unpacked, lock = wheel_lock(work / 'wheels', plan['pip'], limit=RUNTIME_LIMIT - conda_bytes)
        publish(receipt / 'wheels.json', dict(contract='LUKA_OSCAR_RUNTIME_WHEELS/v1', schema_version=1,
                parents=dict(plan=plan['content_hash']), files=wheels, unpacked_bytes=unpacked))
        lock_path = receipt / 'pip-hashes.txt'; lock_path.write_text(lock)
        subprocess.run(pip_command(TARGET, 'install', '--no-deps', '--no-compile', '--no-index',
                                  '--find-links', work / 'wheels', '--require-hashes', '-r', lock_path), env=env, check=True)
        subprocess.run(pip_command(TARGET, 'check'), env=env, check=True)
        verify_files(TARGET, plan)
        verify_runtime(TARGET, project, spec_path, env)
        size = sum(p.stat().st_size for p in TARGET.rglob('*') if p.is_file())
        require(size <= RUNTIME_LIMIT, 'Installed runtime exceeds reserved budget')
        remaining = available_gib - math.ceil(size / GIB) - 2
        require(remaining >= RESULT_GIB, 'Insufficient remaining result headroom')
        publish(receipt / 'complete.json', dict(contract='LUKA_OSCAR_RUNTIME_RESTORED/v1', schema_version=1,
            parents=dict(plan=plan['content_hash'], environment=plan['expected_environment']['content_hash']),
            environment_dir=str(TARGET), installed_bytes=size, declared_remaining_result_gib=remaining,
            downloads=str(work), gpu_validation='inside_scientific_jobs', jobs_submitted=0))
        print(f'RUNTIME_RESTORED: {size / GIB:.2f} GiB in HOME; {remaining} GiB conservatively remains from the declared quota.', flush=True)
        print('No jobs submitted; old environment, campaign, models and dataset untouched.', flush=True)
    except Exception as exc:
        publish(receipt / 'failure.json', dict(contract='LUKA_OSCAR_RUNTIME_RESTORE_FAILURE/v1', schema_version=1,
            parents=dict(plan=plan['content_hash']), error=repr(exc), downloads=str(work)))
        raise
