"""Transfer scope, streaming corruption checks and archive integrity regressions."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from hlt_classification.luka_u100_pair_transfer import bundle as b


@pytest.fixture
def origin(tmp_path, monkeypatch):
    base = tmp_path/'source'
    base.mkdir()

    def write(name, value):
        path = base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b.canonical(value) if isinstance(value, dict) else value)
        return path

    def ref(path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=b.sha256(path))

    def group(name, keys):
        result = {}
        for key in keys:
            suffix = '.pt' if key == 'checkpoint' else '.npz' if key in ('predictions', 'subset') else '.json'
            result[key] = ref(write(name+'/'+key+suffix, (name+'/'+key).encode()))
        return result

    froot = base/'foundation'
    ffiles = {name: ref(write('foundation/'+name, b.seal(dict(contract=name))))
              for name in ('inventory.json', 'split_manifest.json')}
    foundation = b.seal(dict(artifacts=ffiles))
    write('foundation/foundation.json', foundation)
    records = {}
    for role, coord in (*b.VIEWS, ('validation', 'U100'), ('train', 'OFFLINE')):
        name = f'inputs/{role}/{coord}/0000.npz'
        record = ref(write('prepared/'+name, (role+coord).encode()))
        record.update(path=name, file_index=0, rows=b.COUNTS[role])
        records.setdefault(role, {})[coord] = [record]
    assignments = {}
    for role in ('train', 'validation'):
        name = f'assignments/{role}_0000.npz'
        record = ref(write('prepared/'+name, (role+'maps').encode()))
        assignments[role] = [dict(record, path=name, file_index=0)]
    prepared = b.seal(dict(foundation_root=str(froot), parents=dict(foundation=foundation['content_hash']),
                           assignments=assignments))
    cache = b.seal(dict(blocks=records))
    write('prepared/prepared.json', prepared)
    write('prepared/cache_manifest.json', cache)
    fixed_spec = ref(write('fixed30/spec.json', b'fixed30'))
    teacher = b.seal(dict(files=group('teacher', ('report', 'receipt', 'checkpoint', 'predictions'))))
    payload = ref(write('bank/targets.npz', b'actual precomputed embeddings'))
    bank = b.seal(dict(role='train', coordinate='U100', rows=b.COUNTS['train'],
        parents=dict(prepared=prepared['content_hash'], teacher=teacher['content_hash']),
        payload=dict(payload, path='targets.npz')))
    bank_ref = ref(write('bank/manifest.json', bank))
    anchor = dict(evidence=dict(prepared_root=str(base/'prepared'), prepared=prepared['content_hash'],
        cache=cache['content_hash'], models={n: group(n, ('report', 'receipt', 'predictions', 'checkpoint'))
                                            for n in ('M0HLT', 'OFFLINE')}), spec=fixed_spec)
    imported = dict(teacher=teacher, spec=ref(write('historical/spec.json', b'old spec')),
        files=group('historical', ('receipt', 'report', 'predictions', 'subset', 'bank_manifest',
                                  'readback', 'fit_receipt', 'results', 'checkpoint')))
    control = b.seal(dict(job_id='21993056', u100_manifest=bank_ref, environment=dict(python='3.10'),
        spec=ref(write('control/spec.json', b'control spec')),
        files=group('control', ('receipt', 'fit', 'report', 'checkpoint', 'predictions',
                                'bundle', 'setup_receipt', 'acceptance'))))
    root = base/'campaign'
    spec = b.seal(dict(contract='LUKA_U100_PAIR_CORRECTION_SPEC/v1', counts=b.COUNTS,
        root=str(root), source_commit='c'*40, project_dir=str(base/'project'),
        reference=control, anchor=anchor, imported=imported))
    spec_path = write('campaign/campaign_spec.json', spec)
    proof = dict(campaign=spec['content_hash'], reference=control['content_hash'],
        prepared=prepared['content_hash'], cache=cache['content_hash'], source_commit='c'*40,
        control_job='21993056', counts=b.COUNTS, environment=control['environment'], original_source_replay=True)
    monkeypatch.setattr(b, 'authenticate', lambda path, value: deepcopy(proof))
    write('datasets/do-not-copy.root', b'raw sentinel')
    write('Fresh_check/do-not-copy.npz', b'unrelated sentinel')
    return SimpleNamespace(base=base, spec=spec, spec_path=spec_path, proof=proof)


def test_exact_scope_excludes_unused_views_checkpoints_and_directories(origin):
    plan = b.inventory(origin.spec_path)
    names = {r['path'] for r in plan['files']}
    assert {n for n in names if n.startswith('prepared/inputs/')} == {
        f'prepared/inputs/{role}/{coord}/0000.npz' for role, coord in b.VIEWS}
    assert 'prepared/assignments/train_0000.npz' in names
    assert 'prepared/assignments/validation_0000.npz' not in names
    assert {n for n in names if n.endswith('.pt')} == {'reference/matched/checkpoint.pt'}
    assert 'reference/historical/subset.npz' in names
    assert 'targets/u100/targets.npz' in names
    assert not any('Fresh_check' in r['source_path'] or '/datasets/' in r['source_path'] for r in plan['files'])
    assert len(names) == 39


def test_roundtrip_determinism_original_bytes_and_plain_python_verification(origin, tmp_path):
    before = {p: b.sha256(p) for p in origin.base.rglob('*') if p.is_file()}
    result = b.export(origin.spec_path, tmp_path/'export1')
    plan = b.verify(result['archive'], result['sha256'])
    assert plan['content_hash'] == result['manifest']
    second = b.export(origin.spec_path, tmp_path/'export2')
    assert result['sha256'] == second['sha256']
    with tarfile.open(result['archive']) as tar:
        assert tar.extractfile('origin/campaign_spec.json').read() == origin.spec_path.read_bytes()
        assert all(m.isfile() and not m.issparse() for m in tar)
    assert before == {p: b.sha256(p) for p in before}
    assert not (tmp_path/'export1/campaign-inputs.tar.partial').exists()
    assert (tmp_path/'export1/SHA256SUMS').read_text().startswith(result['sha256'])


def test_corruption_is_not_published(origin, tmp_path):
    path = origin.base/'prepared/inputs/train/D000/0000.npz'
    path.write_bytes(b'x'*path.stat().st_size)
    with pytest.raises(ValueError, match='Source bytes differ'):
        b.export(origin.spec_path, tmp_path/'bad')
    assert not (tmp_path/'bad/campaign-inputs.tar').exists()
    assert (tmp_path/'bad/campaign-inputs.tar.partial').exists()
    with pytest.raises(ValueError, match='fresh directory'):
        b.export(origin.spec_path, tmp_path/'bad')


def test_missing_payload_never_triggers_reconstruction(origin, tmp_path):
    (origin.base/'prepared/inputs/train/U100/0000.npz').unlink()
    with pytest.raises(ValueError, match='Missing'):
        b.export(origin.spec_path, tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


def test_exclusive_export_preserves_existing_output(origin, tmp_path):
    path = tmp_path/'existing'
    path.mkdir()
    sentinel = path/'campaign-inputs.tar'
    sentinel.write_bytes(b'keep')
    with pytest.raises(ValueError, match='fresh directory'):
        b.export(origin.spec_path, path)
    assert sentinel.read_bytes() == b'keep'


def test_changed_source_between_inventory_and_copy(origin, tmp_path, monkeypatch):
    plan = b.inventory(origin.spec_path)
    path = origin.base/'prepared/inputs/train/U100/0000.npz'
    path.write_bytes(b'new content')
    monkeypatch.setattr(b, 'inventory', lambda _: plan)
    with pytest.raises((ValueError, OSError)):
        b.export(origin.spec_path, tmp_path/'changed')
    assert not (tmp_path/'changed/campaign-inputs.tar').exists()


def test_reference_replay_failure_stops_before_export(origin, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('authentication rejected')
    monkeypatch.setattr(b, 'authenticate', fail)
    with pytest.raises(ValueError, match='authentication rejected'):
        b.export(origin.spec_path, tmp_path/'never')
    assert not (tmp_path/'never').exists()


def test_storage_limit_and_input_overlap(origin, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='overlaps'):
        b.export(origin.spec_path, origin.base/'prepared/export')
    monkeypatch.setattr(b.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='headroom'):
        b.export(origin.spec_path, tmp_path/'short')
    assert not (tmp_path/'short').exists()


@pytest.mark.parametrize('name', ('../escape', '/absolute', 'a/../b', 'a//b', './a',
                                'C:/data', 'a\\b', 'a\nb', ''))
def test_reject_paths(name):
    with pytest.raises(ValueError, match='path'):
        b.relative(name)


def rewrite_archive(path, plan, changes):
    with tarfile.open(path, 'w') as tar:
        raw = b.canonical(plan)
        tar.addfile(b.member('manifest.json', len(raw)), io.BytesIO(raw))
        for row in plan['files']:
            raw = Path(row['source_path']).read_bytes()
            tar.addfile(b.member(row['path'], len(raw)), io.BytesIO(raw))
        changes(tar)


@pytest.mark.parametrize('kind', ('extra', 'duplicate', 'symlink', 'traversal'))
def test_reject_extra_or_unsafe_archive_members(origin, tmp_path, kind):
    plan = b.inventory(origin.spec_path)
    def change(tar):
        name = 'extra' if kind != 'duplicate' else plan['files'][0]['path']
        if kind == 'traversal': name = '../escape'
        info = tarfile.TarInfo(name)
        if kind == 'symlink': info.type, info.linkname = tarfile.SYMTYPE, '../escape'
        tar.addfile(info)
    path = tmp_path/'malformed.tar'
    rewrite_archive(path, plan, change)
    with pytest.raises(ValueError):
        b.verify(path, b.sha256(path))
    assert not (tmp_path/'escape').exists()


def test_transport_and_internal_checksums(origin, tmp_path):
    result = b.export(origin.spec_path, tmp_path/'export')
    with pytest.raises(ValueError, match='Archive SHA256'):
        b.verify(result['archive'], '0'*64)
    plan = b.inventory(origin.spec_path)
    plan['files'][0]['sha256'] = '0'*64
    plan.pop('content_hash')
    plan = b.seal(plan)
    path = tmp_path/'bad-payload.tar'
    rewrite_archive(path, plan, lambda _: None)
    with pytest.raises(ValueError, match='member SHA256'):
        b.verify(path, b.sha256(path))


def test_missing_archive_members(origin, tmp_path):
    plan = b.inventory(origin.spec_path)
    path = tmp_path/'empty.tar'
    with tarfile.open(path, 'w') as tar:
        raw = b.canonical(plan)
        tar.addfile(b.member('manifest.json', len(raw)), io.BytesIO(raw))
    with pytest.raises(ValueError, match='Incomplete'):
        b.verify(path, b.sha256(path))


def test_origin_subprocess_uses_original_source_and_no_scheduler(monkeypatch, tmp_path):
    expected = {'proof': 'ok'}
    def run(command, **kwargs):
        assert command[1:4] == ['-B', '-s', '-c']
        assert 'source.import_control' in command[4]
        assert 'campaign.check(s,check_storage=False)' in command[4]
        assert not any(w in command[4] for w in ('sbatch', 'scancel', 'torch.load'))
        assert kwargs['env']['PYTHONPATH'] == str(tmp_path/'src')
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        return SimpleNamespace(returncode=0, stdout='PAIR_TRANSFER_ORIGIN='+json.dumps(expected), stderr='')
    monkeypatch.setattr(b.subprocess, 'run', run)
    assert b.authenticate(tmp_path/'spec', dict(project_dir=str(tmp_path))) == expected


def test_changed_proof(origin, monkeypatch):
    bad = dict(origin.proof, control_job='wrong')
    monkeypatch.setattr(b, 'authenticate', lambda *args: bad)
    with pytest.raises(ValueError, match='Origin proof'):
        b.inventory(origin.spec_path)


def test_validation_bank_cannot_be_exported(origin):
    path = origin.base/'bank/manifest.json'
    value = b.read_json(path)
    value.pop('content_hash')
    value['role'] = 'validation'
    path.write_bytes(b.canonical(b.seal(value)))
    with pytest.raises(ValueError, match='bank binding'):
        b.inventory(origin.spec_path)


def test_stdlib_only_verifier_import():
    import ast
    tree = ast.parse(Path(b.__file__).read_text())
    imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {'numpy', 'torch', 'hlt_classification'} & imports
