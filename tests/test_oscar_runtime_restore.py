"""Runtime recovery must fail closed before changing science or submitting jobs."""
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
from hlt_classification.operations import oscar_runtime_restore as r


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp):
    old = tmp / 'broken'
    for name, version, files in [('python', '3.10.21', ['lib/python3.10/os.py']),
                                  ('pip', '26.2.1', ['lib/python3.10/site-packages/pip-26.2.1.dist-info/METADATA']),
                                  ('setuptools', '84.0.0', []), ('wheel', '0.48.0', []),
                                  ('_openmp_mutex', '4.5', [])]:
        write_json(old / 'conda-meta' / (name + '.json'), dict(name=name, version=version, build='build0',
                   url=f'https://conda.anaconda.org/conda-forge/linux-64/{name}-{version}-build0.conda',
                   md5='0'*32, files=files))
    packages = {'torch': '2.5.1+cu118', 'numpy': '2.2.6', 'weaver-core': '0.5.3'}
    for name, version in packages.items():
        dist = old / 'lib/python3.10/site-packages' / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.mkdir(parents=True)
        (dist / 'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
        (dist / 'RECORD').write_text('unused\n')
    # Missing METADATA is allowed only when Conda owns it (same failure as Oscar).
    (old / 'lib/python3.10/site-packages/pip-26.2.1.dist-info').mkdir()
    expected = r.hashed(dict(architecture='x86_64', python='3.10.21', torch_cuda='11.8', cudnn=90100,
                           versions=packages, weaver_python_sources={'nn/model/ParticleTransformer.py': 'e'*64}))
    root = tmp / 'checkpoints/failed'
    spec_path = root / 'campaign_spec.json'
    write_json(spec_path, r.hashed(dict(contract='LUKA_NEW_BALANCED_OSCAR_SPEC/v1', root=str(root),
                                      environment_dir=str(old), environment=expected, tasks={'train_OFFLINE': []})))
    return spec_path, old


def test_inventory_never_needs_broken_python_and_preserves_every_version(tmp_path):
    spec, old = fixture(tmp_path)
    plan = r.inventory(spec)
    assert not (old / 'bin/python').exists()
    assert len(plan['conda']) == 5 and set(plan['pip']) == {'torch', 'numpy', 'weaver-core'}
    assert plan['pip']['torch']['version'] == '2.5.1+cu118'
    assert plan['parents']['environment'] == plan['expected_environment']['content_hash']


@pytest.mark.parametrize('corruption', ['spec', 'runtime', 'metadata', 'version', 'url', 'checksum', 'unknown'])
def test_bad_records_cannot_create_a_restore_plan(tmp_path, corruption):
    spec, old = fixture(tmp_path)
    if corruption in ('spec', 'runtime'):
        obj = json.loads(spec.read_bytes()); obj['environment']['python'] = '3.10.0'
        write_json(spec, obj if corruption == 'spec' else r.hashed(obj))
    elif corruption in ('metadata', 'version'):
        path = old / 'lib/python3.10/site-packages/numpy-2.2.6.dist-info/METADATA'
        path.unlink() if corruption == 'metadata' else path.write_text('Name: numpy\nVersion: 9.0\n')
    elif corruption in ('url', 'checksum'):
        path = old / 'conda-meta/python.json'; obj = json.loads(path.read_bytes())
        obj['url' if corruption == 'url' else 'md5'] = 'http://untrusted.example/package'
        write_json(path, obj)
    else:
        (old / 'lib/python3.10/site-packages/unknown-1.0.dist-info').mkdir()
    with pytest.raises(ValueError):
        r.inventory(spec)


def make_wheel(path, name, version, payload=b'payload'):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA", f'Name: {name}\nVersion: {version}\n')
        z.writestr(name + '/payload', payload)


def test_wheel_lock_authenticates_exact_inventory_and_space(tmp_path):
    path = tmp_path / 'torch.whl'; make_wheel(path, 'torch', '2.5.1+cu118')
    packages = {'torch': {'version': '2.5.1+cu118'}}
    wheels, size, lock = r.wheel_lock(tmp_path, packages)
    assert size > 0 and r.sha_file(path) in lock
    assert wheels['torch']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='budget'):
        r.wheel_lock(tmp_path, packages, limit=1)
    make_wheel(tmp_path / 'extra.whl', 'another', '1.0')
    with pytest.raises(ValueError, match='inventory'):
        r.wheel_lock(tmp_path, packages)


def installed_fixture(prefix):
    site = prefix / 'lib/python3.10/site-packages'
    folder = site / 'example-1.0.dist-info'; folder.mkdir(parents=True)
    (folder / 'METADATA').write_text('Name: example\nVersion: 1.0\n')
    payload = site / 'example.py'; payload.write_bytes(b'x=1\n')
    with (folder / 'RECORD').open('w', newline='') as stream:
        w = csv.writer(stream)
        for p in [payload, folder / 'METADATA']:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('=')
            w.writerow([p.relative_to(site).as_posix(), 'sha256=' + encoded, p.stat().st_size])
        w.writerow(['example-1.0.dist-info/RECORD', '', ''])
    write_json(prefix / 'conda-meta/python.json', dict(name='python', version='3.10.21', build='b', files=[]))
    return dict(conda=[dict(name='python', version='3.10.21', build='b')], pip={'example': {'version': '1.0'}}), payload


@pytest.mark.parametrize('damage', ['delete', 'modify', 'escape', 'empty', 'conda_missing'])
def test_verify_files_detects_damage_not_detectable_by_import(tmp_path, damage):
    prefix = tmp_path / 'env'; plan, payload = installed_fixture(prefix)
    r.verify_files(prefix, plan)
    record = prefix / 'lib/python3.10/site-packages/example-1.0.dist-info/RECORD'
    if damage == 'delete': payload.unlink()
    elif damage == 'modify': payload.write_bytes(b'x=2\n')
    elif damage == 'escape': record.write_text('../../../../outside,,\n')
    elif damage == 'empty': record.write_text('')
    else:
        p = prefix / 'conda-meta/python.json'; obj = json.loads(p.read_bytes()); obj['files'] = ['lib/os.py']; write_json(p, obj)
    with pytest.raises(ValueError): r.verify_files(prefix, plan)


def test_environment_cleanup_prevents_pip_and_python_redirects(tmp_path, monkeypatch):
    monkeypatch.setenv('PIP_EXTRA_INDEX_URL', 'https://unexpected.invalid')
    monkeypatch.setenv('PYTHONPATH', '/broken'); monkeypatch.setenv('CONDA_PKGS_DIRS', '/broken-cache')
    monkeypatch.setenv('LD_PRELOAD', '/bad'); monkeypatch.setenv('SBATCH_MEM', '42')
    env = r.clean_env(tmp_path, tmp_path / 'env')
    assert 'PIP_EXTRA_INDEX_URL' not in env and 'PYTHONPATH' not in env and 'LD_PRELOAD' not in env
    assert 'SBATCH_MEM' not in env
    assert env['CONDA_PKGS_DIRS'] == str(tmp_path / 'conda-cache')
    assert env['LD_LIBRARY_PATH'] == str(tmp_path / 'env/lib')


def test_publication_is_atomic_immutable_and_hash_checked(tmp_path):
    path = tmp_path / 'complete.json'; r.publish(path, {'passed': True})
    assert r.read_hashed(path)['passed']
    with pytest.raises(ValueError): r.publish(path, {'passed': False})
    assert not list(tmp_path.glob('.publish-*'))


@pytest.mark.parametrize('fail', [False, True])
def test_full_restore_command_flow_and_failure_evidence(tmp_path, monkeypatch, fail):
    spec, old = fixture(tmp_path)
    project = tmp_path / 'source/project'; project.mkdir(parents=True)
    exe = tmp_path / 'base/bin/conda'; exe.parent.mkdir(parents=True); exe.write_text('fake')
    target = tmp_path / 'environments/new'
    monkeypatch.setattr(r, 'HOME_BASE', tmp_path); monkeypatch.setattr(r, 'TARGET', target)
    monkeypatch.setattr(r, 'SCRATCH', tmp_path / 'scratch')
    # Driver is Linux-only; fake its OS check without changing pathlib's platform.
    monkeypatch.setattr(r, 'os', SimpleNamespace(**dict(vars(os), name='posix')))
    calls = []
    def run(args, **kw):
        calls.append(args)
        assert kw['check'] and kw['env']['CONDA_PKGS_DIRS'].startswith(str(tmp_path / 'scratch'))
        if 'create' in args:
            assert '--copy' in args and '--no-default-packages' in args
            target.mkdir()
        elif 'download' in args:
            if fail: raise subprocess.CalledProcessError(1, args)
            dest = Path(args[args.index('--dest')+1])
            pins = Path(args[args.index('-r')+1]).read_text().splitlines() if '-r' in args else [args[-1]]
            for pin in pins:
                name, version = pin.split('=='); make_wheel(dest/(name+'.whl'), name, version)
        elif 'install' in args:
            assert all(x in args for x in ('--no-deps', '--no-index', '--require-hashes'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(r.subprocess, 'run', run)
    verified = []
    monkeypatch.setattr(r, 'verify_files', lambda *a: verified.append('files'))
    monkeypatch.setattr(r, 'verify_runtime', lambda *a: verified.append('runtime'))
    if fail:
        with pytest.raises(subprocess.CalledProcessError): r.restore(spec, project, exe, 28)
        assert (tmp_path/'environments/new.restore/failure.json').exists()
        assert not (tmp_path/'environments/new.restore/complete.json').exists()
        assert not verified
    else:
        r.restore(spec, project, exe, 28)
        complete = r.read_hashed(tmp_path/'environments/new.restore/complete.json')
        assert complete['jobs_submitted'] == 0 and complete['declared_remaining_result_gib'] >= 14
        assert verified == ['files', 'runtime']
    assert not any('sbatch' in arg for call in calls for arg in call)
    assert r.inventory(spec)['old_environment'] == str(old)
    with pytest.raises(ValueError, match='exists'): r.restore(spec, project, exe, 28)


def test_retry_rejects_pending_jobs_and_saved_epochs(tmp_path, monkeypatch):
    from hlt_classification.luka_new_balanced_oscar import campaign
    spec, _ = fixture(tmp_path)
    monkeypatch.setattr(campaign, 'accepted_jobs', lambda s: [{'job': '123'}])
    monkeypatch.setattr(campaign, 'terminal', lambda j: (_ for _ in ()).throw(ValueError('not terminal')))
    with pytest.raises(ValueError, match='terminal'): r.startup_retry_evidence(spec)
    monkeypatch.setattr(campaign, 'terminal', lambda j: ['123', 'FAILED', '1:0'])
    assert r.startup_retry_evidence(spec)['no_training_states']
    epochs = spec.parent/'train_OFFLINE/epochs'; epochs.mkdir(parents=True); (epochs/'01.json').write_text('{}')
    with pytest.raises(ValueError, match='Saved training'): r.startup_retry_evidence(spec)


def test_shells_have_valid_syntax():
    project = Path(__file__).resolve().parents[1]
    bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
    for name in ('recover_luka_new_balanced_oscar.sh', 'queue_luka_new_balanced_oscar.sh'):
        subprocess.run([bash, '-n', str(project/'scripts'/name)], check=True)


def test_runtime_probe_executes_real_cpu_model_and_rejects_drift(tmp_path, monkeypatch):
    import weaver
    from hlt_classification.jetclass2_delphes.model import installed_environment
    project = Path(__file__).resolve().parents[1]
    spec = tmp_path/'spec.json'; write_json(spec, {'environment': installed_environment()})
    actual_run = subprocess.run
    def local_run(args, **kw):
        # The Windows test interpreter uses an installed Weaver target directory.
        code = 'import sys; sys.path.insert(0, ' + repr(str(Path(weaver.__path__[0]).parent)) + ')\n' + args[5]
        return actual_run([sys.executable, *args[1:5], code, *args[6:]], **kw)
    monkeypatch.setattr(r.subprocess, 'run', local_run)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r.verify_runtime(tmp_path, project, spec, env)
    obj = json.loads(spec.read_bytes()); obj['environment']['torch_cuda'] = 'wrong'; write_json(spec, obj)
    with pytest.raises(subprocess.CalledProcessError): r.verify_runtime(tmp_path, project, spec, env)


@pytest.mark.parametrize('repair_exit', [0, 1])
def test_real_detached_recovery_only_queues_after_repair_success(tmp_path, repair_exit):
    import time
    bash = 'C:/Program Files/Git/bin/bash.exe' if sys.platform == 'win32' else 'bash'
    def sp(path):
        p = Path(path).resolve()
        return '/' + p.drive[0].lower() + p.as_posix()[2:] if sys.platform == 'win32' else str(p)
    base = tmp_path/'base'; commit = '1'*40
    (base/'environments').mkdir(parents=True)
    project = base/'source'/('HLT_Classification_new_balanced_oscar_'+commit[:8])
    (project/'scripts').mkdir(parents=True); (project/'sbatch').mkdir()
    data = tmp_path/'data'; data.mkdir(); (data/'transfer_complete.json').write_text('{}')
    spec = base/'checkpoints/luka_new_balanced_oscar_f9dfc80b_r1/campaign_spec.json'
    write_json(spec, {})
    script = project/'scripts/recover_luka_new_balanced_oscar.sh'
    original = (Path(__file__).resolve().parents[1]/'scripts'/script.name).read_text()
    # Only relocate the two site-specific roots; run the actual shell control flow.
    original = original.replace('BASE=/oscar/home/rlyang/hlt_classification', 'BASE='+sp(base))
    original = original.replace('DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1', 'DATASET='+sp(data))
    script.write_bytes(original.encode())
    (project/'sbatch/luka_new_balanced_oscar_common.sh').write_bytes(b':\n')
    (project/'scripts/launch_luka_new_balanced_oscar.sh').write_bytes(b'echo QUEUED >> "$FIXTURE_CALLS"\n')
    bindir = tmp_path/'bin'; bindir.mkdir()
    conda = tmp_path/'conda/bin/conda'; conda.parent.mkdir(parents=True); conda.write_bytes(b'#!/bin/sh\nexit 0\n'); conda.chmod(0o755)
    base_python = conda.parent/'python'
    base_python.write_bytes(b'''#!/usr/bin/env bash
if [[ "$3" == -c ]]; then echo 18; exit 0; fi
echo REPAIR >> "$FIXTURE_CALLS"
sleep 1
exit "$FIXTURE_EXIT"
'''); base_python.chmod(0o755)
    # Fake repaired interpreter needs no real prefix; common.sh and calls are fixtures.
    # Create it from base Python only when repair succeeds, as the real installer does.
    body = base_python.read_text().replace('exit "$FIXTURE_EXIT"', '''if [[ "$FIXTURE_EXIT" == 0 ]]; then
mkdir "$FIXTURE_TARGET"
mkdir "$FIXTURE_TARGET/bin"
printf '#!/usr/bin/env bash\\necho GUARDED >> "$FIXTURE_CALLS"\\n' > "$FIXTURE_TARGET/bin/python"
chmod +x "$FIXTURE_TARGET/bin/python"
fi
exit "$FIXTURE_EXIT"''')
    base_python.write_bytes(body.encode())
    for name, body in [('git', 'echo '+commit), ('tail', 'echo TAIL_ATTACHED')]:
        p = bindir/name; p.write_bytes(('#!/usr/bin/env bash\n'+body+'\n').encode()); p.chmod(0o755)
    envfile = tmp_path/'env.sh'; envfile.write_bytes(b'export PATH="${FIXTURE_BIN}:${PATH}"\n')
    calls = tmp_path/'calls'
    env = dict(os.environ, BASH_ENV=sp(envfile), FIXTURE_BIN=sp(bindir), FIXTURE_CALLS=sp(calls),
               FIXTURE_EXIT=str(repair_exit), FIXTURE_TARGET=sp(base/'environments/new_balanced_oscar_v1'))
    args = [bash, sp(script), 'detach', commit, '28', sp(conda), '--authorize', 'AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1']
    result = subprocess.run(args, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'DETACHED runtime recovery PID' in result.stdout and 'TAIL_ATTACHED' in result.stdout
    log = base/'checkpoints'/('luka_new_balanced_oscar_'+commit[:8]+'_r1.runtime-launch/launcher.log')
    deadline = time.monotonic()+12
    while (not log.exists() or 'RECOVERY_LAUNCHER_EXIT=' not in log.read_text()) and time.monotonic()<deadline:
        time.sleep(.05)
    assert f'RECOVERY_LAUNCHER_EXIT={repair_exit}' in log.read_text(), log.read_text()
    assert calls.read_text().splitlines() == (['REPAIR', 'GUARDED', 'QUEUED'] if repair_exit == 0 else ['REPAIR'])
    assert subprocess.run(args, env=env, capture_output=True, timeout=5).returncode != 0
