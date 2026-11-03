"""Portable inputs, unchanged model path, exact Hydra DAG and failure boundaries."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.luka_u100_pair_transfer import bundle as transfer
from hlt_classification.luka_u100_pair_hydra import inputs, contracts as c, campaign, submission, worker, site
from hlt_classification.data.cache_contracts import load_json, load_npz_arrays
from hlt_classification.cms_proxy_ladder.contracts import file_ref
from test_luka_u100_pair_transfer import origin
from test_luka_u100_pair_correction import dataset, prepared, all_prepared, compact, fixture, threads


def extracted(origin, tmp_path, monkeypatch):
    out = transfer.export(origin.spec_path, tmp_path/'export')
    root = tmp_path/'relocated'
    inputs.unpack(out['archive'], root, out['sha256'])
    monkeypatch.setattr(c, 'ARCHIVE_SHA256', out['sha256'])
    monkeypatch.setattr(c, 'MANIFEST_HASH', out['manifest'])
    return root, inputs.Bundle(root)


def test_extract_relocates_exact_bytes_and_never_mutates_original_json(origin, tmp_path, monkeypatch):
    root, bundle = extracted(origin, tmp_path, monkeypatch)
    bundle.verify_all()
    assert bundle.path('origin/campaign_spec.json').read_bytes() == origin.spec_path.read_bytes()
    assert not (root/'prepared/assignments/validation_0000.npz').exists()
    assert not (root/'prepared/inputs/validation/U100').exists()
    with pytest.raises(ValueError, match='new directory'):
        inputs.unpack(tmp_path/'export/campaign-inputs.tar', root, c.ARCHIVE_SHA256)
    with pytest.raises(PermissionError):
        bundle.path('prepared/inputs/validation/U100/0000.npz')
    path = bundle.path('prepared/inputs/train/D000/0000.npz')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        bundle.path('prepared/inputs/train/D000/0000.npz')


def test_extraction_failure_never_publishes_completion(origin, tmp_path, monkeypatch):
    out = transfer.export(origin.spec_path, tmp_path/'export')
    plan = transfer.verify(out['archive'], out['sha256'])
    # Simulate changed bytes between initial verification and the copy phase.
    changed = deepcopy(plan); changed['files'][0]['sha256'] = '0'*64
    monkeypatch.setattr(transfer, 'verify', lambda *a: changed)
    root = tmp_path/'incomplete'
    with pytest.raises(ValueError, match='changed'):
        inputs.unpack(out['archive'], root, out['sha256'])
    assert not (root/'extraction_receipt.json').exists()


def test_untrusted_archive_and_wrong_bundle_identity_fail(origin, tmp_path, monkeypatch):
    out = transfer.export(origin.spec_path, tmp_path/'export')
    with pytest.raises(ValueError, match='SHA256'):
        inputs.unpack(out['archive'], tmp_path/'bad', '0'*64)
    assert not (tmp_path/'bad').exists()
    inputs.unpack(out['archive'], tmp_path/'valid', out['sha256'])
    with pytest.raises(ValueError, match='registered'):
        inputs.Bundle(tmp_path/'valid')


def test_portable_real_arrays_metadata_particle_targets_without_old_path_access(all_prepared, tmp_path, monkeypatch):
    from hlt_classification.luka_fullsim import regular_cache
    gate, m, p, inv, splits = all_prepared
    root = tmp_path/'portable'; root.mkdir()
    names = ['prepared/cache_manifest.json', 'prepared/prepared.json', 'foundation/foundation.json',
             'foundation/inventory.json', 'foundation/split_manifest.json']
    for role, coord in transfer.VIEWS:
        names.extend('prepared/'+r['path'] for r in m['blocks'][role][coord])
    names.extend('prepared/'+r['path'] for r in p['assignments']['train'])
    for name in names:
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(gate['root'])/name, root/name)
    refs = {name: file_ref(root/name) for name in names}
    class Portable:
        def __init__(self, _):
            self.root = root
            self.manifest = dict(origin=dict(cache=m['content_hash'], prepared=p['content_hash']))
        def path(self, name):
            assert name in refs, name
            path = transfer.regular(root/name)
            assert file_ref(path) == refs[name]
            return path
        def json(self, name):
            return transfer.checked(load_json(self.path(name)))
    monkeypatch.setattr(inputs, 'Bundle', Portable)
    # Any accidental historical metadata/raw fallback must fail, not silently
    # succeed because the fixture still has its old files available.
    monkeypatch.setattr(regular_cache, 'metadata', lambda *a: pytest.fail('old metadata path followed'))
    monkeypatch.setattr(regular_cache, 'selected_file_rows', lambda *a: pytest.fail('raw reconstruction attempted'))
    s = dict(campaign.registration(), root=str(tmp_path), bundle_root=str(root), content_hash='f'*64)
    s['counts'] = p['counts']
    meta = inputs.metadata(s)
    assert meta[0] == m and meta[1] == p
    train = inputs.load_view(s, 'train', 'D000')
    val = inputs.load_view(s, 'validation', 'D000')
    (tmp_path/'setup').mkdir()
    particles = inputs.export(s, train, device='cpu')
    restored = inputs.particle_bank(s, train)
    np.testing.assert_array_equal(restored.vectors, particles.vectors)
    np.testing.assert_array_equal(restored.identities, train.identities)
    assert restored.manifest['rows'] == len(train)
    assert len(val) == p['counts']['validation']
    assert all(Path(r['path']).is_relative_to(root) for r in restored.manifest['assignment_refs'])
    for role, co in [('validation', 'U100'), ('final_test', 'D000'), ('train', 'OFFLINE')]:
        with pytest.raises(PermissionError): inputs.load_view(s, role, co)
    with pytest.raises(PermissionError): inputs.particle_bank(s, val)
    # Loading full-array views still verifies original byte fingerprints.
    (root/('prepared/'+m['blocks']['train']['D000'][0]['path'])).write_bytes(b'corrupt')
    with pytest.raises(AssertionError): inputs.load_view(s, 'train', 'D000')


def spec_fixture(root):
    s, *rest = fixture(root)
    s.update(campaign.registration())
    s['counts'] = dict(train=22, validation=22)
    return s


def test_hydra_exact_four_fit_two_lane_graph(tmp_path):
    s = spec_fixture(tmp_path)
    first = submission.render(s, 'setup'); envelope = submission.render(s, 'envelope')
    assert first['jobs'] == 1 and envelope['jobs'] == 5
    assert [r['task_id'] for r in envelope['commands']] == [*c.ARMS, 'report']
    assert [r['dependencies'] for r in envelope['commands'][:4]] == [[], [], ['p4_pair_u100'], ['pair_u100']]
    for row in first['commands']+envelope['commands']:
        args = row['command']
        for arg in ('--partition=gpus', '--account=default', '--qos=normal', '--gres=gpu:nvidia_l40:1'):
            assert arg in args
        assert not any('afterok:' in a or '/home/ryreu/' in a or 'preflight' in a for a in args)
        assert 'run_luka_u100_pair_hydra.sh' in args[-4]
    measured = submission.render(s, 'science', acceptance=c.artifact('ACCEPTANCE', minutes={a:180 for a in c.ARMS}))
    assert submission.within_envelope(envelope, measured)
    changed = deepcopy(measured); changed['commands'][0]['command'].append('--gres=gpu:2')
    assert not submission.within_envelope(envelope, changed)


def test_exact_dispatch_releases_once_and_failed_setup_releases_none(tmp_path, monkeypatch):
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    calls = []
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(returncode=0, stdout=str(1000+len(calls))+'\n', stderr='')
    monkeypatch.setattr(dag.subprocess, 'run', run)
    monkeypatch.setattr(campaign, 'check', lambda *a, **kw: None)
    monkeypatch.setattr(submission, 'check_site', lambda *a: None)
    monkeypatch.setattr(submission.resources, 'accepted', lambda s: c.artifact('ACCEPTANCE', minutes={a:120 for a in c.ARMS}))
    monkeypatch.setattr(worker, 'completed_setup', lambda s: None)
    monkeypatch.setattr(submission, 'accounting', lambda j: ['COMPLETED', '0:0'])
    s = spec_fixture(tmp_path); review = submission.review(s)
    submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 6 and len(submission.accepted_jobs(s, 'science')) == 5
    with pytest.raises(FileExistsError): submission.dispatch(s, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    failed_root = tmp_path/'failed'; failed_root.mkdir(); failed = spec_fixture(failed_root)
    review = submission.review(failed)
    monkeypatch.setattr(submission, 'accounting', lambda j: ['FAILED', '1:0'])
    with pytest.raises(RuntimeError, match='no scientific'):
        submission.dispatch(failed, plan_hash=review['content_hash'], authorization=c.AUTHORIZATION)
    assert len(calls) == 7


def test_site_review_checks_hydra_and_requests_without_submitting(tmp_path, monkeypatch):
    calls = []
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(stdout='ClusterName = hydra\n' if args[0] == 'scontrol' else 'test passes', stderr='', returncode=0)
    monkeypatch.setattr(site.subprocess, 'run', run)
    plan = submission.render(spec_fixture(tmp_path), 'envelope')
    site.check_site(plan)
    assert all('--test-only' in x and '--wrap=true' in x for x in calls[1:])
    monkeypatch.setattr(site.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='ClusterName = sporc\n'))
    with pytest.raises(PermissionError, match='Hydra'): site.check_site(plan)


def test_environment_mismatch_remains_strict(monkeypatch):
    from hlt_classification.jetclass2_delphes import model
    monkeypatch.setattr(model, 'installed_environment', lambda: dict(python='3.12', weaver_python_sources={'a':'x'}))
    with pytest.raises(ValueError, match='python'):
        site.environment(dict(python='3.10', weaver_python_sources={'a':'x'}))


def test_allocation_requires_exact_site_venv_and_one_l40(tmp_path, monkeypatch):
    import sys
    s = spec_fixture(tmp_path)
    s.update(environment_dir=str(Path(sys.prefix).resolve()), reference=dict(environment={}))
    for key, value in dict(SLURM_JOB_ID='123', SLURM_CLUSTER_NAME='hydra',
                           SLURM_CPUS_PER_TASK='6', SLURM_MEM_PER_NODE='120000', PYTHONNOUSERSITE='1').items():
        monkeypatch.setenv(key, value)
    details = 'Account=default Partition=gpus QOS=normal NumNodes=1 NumTasks=1 NumCPUs=6'
    monkeypatch.setattr(site.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=details))
    monkeypatch.setattr(site.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(site, 'environment', lambda expected: expected)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA L40')
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    assert site.allocation(s) == '123'
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA L40S')
    with pytest.raises(PermissionError, match='exactly one L40'): site.allocation(s)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA L40')
    monkeypatch.setenv('SLURM_CLUSTER_NAME', 'sporc')
    with pytest.raises(PermissionError, match='Hydra allocation'): site.allocation(s)


def test_source_guard_requires_complete_port_and_unchanged_scientific_donors(monkeypatch):
    project = Path(campaign.__file__).resolve().parents[3]
    snapshot = dict(content_hash='a'*64)
    monkeypatch.setattr(campaign, '_source', lambda *a: snapshot)
    changes = ['A\tsrc/hlt_classification/luka_u100_pair_hydra/site.py']
    files = set(campaign.REQUIRED)
    def run(command, **kw):
        args = command[3:]
        if args[0] == 'ls-files':
            text = '\n'.join(sorted(files))
        elif args[0] == 'diff':
            text = '\n'.join(changes)
        else:
            assert args[:2] == ['merge-base', '--is-ancestor']; text = ''
        return SimpleNamespace(stdout=text)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    assert campaign.current(project, 'f'*40) == snapshot
    changes.append('M\tsrc/hlt_classification/luka_u100_pair_correction/model.py')
    with pytest.raises(ValueError, match='preserve original'): campaign.current(project, 'f'*40)
    changes.pop(); files.remove('sbatch/run_luka_u100_pair_hydra.sh')
    with pytest.raises(ValueError, match='complete additive'): campaign.current(project, 'f'*40)


def test_control_parity_reloads_native_checkpoint_and_uses_saved_rows(tmp_path, monkeypatch):
    from hlt_classification.jetclass2_delphes.model import DelphesParticleTransformer
    from hlt_classification.jetclass2_delphes.runner import predict
    from hlt_classification.data.cache_contracts import atomic_publish_bytes, deterministic_npz_bytes
    compact(monkeypatch); s, tr, val, bank, particles = fixture(tmp_path)
    s.update(parity_tolerances=c.PARITY, bundle_root=str(tmp_path), reference=dict(environment={}))
    paths = dict()
    for name in ('checkpoint.pt', 'predictions.npz'):
        paths['reference/matched/'+name] = tmp_path/name
    model = DelphesParticleTransformer().eval()
    torch.save(model.state_dict(), paths['reference/matched/checkpoint.pt'])
    probs = predict(model, val, device='cpu', batch_size=128)
    atomic_publish_bytes(paths['reference/matched/predictions.npz'], deterministic_npz_bytes(dict(
        identities=val.identities, labels=val.labels, probabilities=probs)))
    monkeypatch.setattr(worker, 'Bundle', lambda _: SimpleNamespace(path=lambda name: paths[name]))
    monkeypatch.setattr(worker, 'installed_environment', lambda: {})
    monkeypatch.setattr(worker, 'gpu_identity', lambda: dict(name='CPU fixture'))
    value = worker.control_parity(s, val, device='cpu')
    worker.require_parity(value, s)
    assert value['max_abs'] == 0 and value['auc_abs'] == 0 and value['rms'] == 0
    bad = {k:v for k,v in value.items() if k != 'content_hash'}; bad['max_abs'] = .1
    with pytest.raises(ValueError, match='numerical portability'):
        worker.require_parity(transfer.seal(bad), s)
    val.identities = val.identities[::-1].copy()
    with pytest.raises(ValueError, match='population'):
        worker.control_parity(s, val, device='cpu')


@pytest.mark.parametrize('arm', c.ARMS)
def test_hydra_uses_original_production_fit_and_retained_reload(tmp_path, monkeypatch, arm):
    from hlt_classification.luka_u100_pair_correction import training
    compact(monkeypatch); s, tr, val, bank, particles = fixture(tmp_path)
    s.update(campaign.registration()); s['counts'] = dict(train=22, validation=22)
    net = worker.new_model(s['imported'], arm, particles.manifest['statistics'])
    report = worker.training.fit(tmp_path/'fit', s, arm, net, tr, val, bank, particles, device='cpu', technical=True)
    checked, values = training.validate_fit(tmp_path/'fit', s, arm, technical=True)
    assert report == checked and len(values['labels']) == 22
    assert worker.training.fit is training.fit
