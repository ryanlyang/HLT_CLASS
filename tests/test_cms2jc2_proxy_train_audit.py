"""Train-only paired diagnostics: synthetic physical banks, no remote writes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest

from hlt_classification.cms2jc2_proxy_audit import campaign as a, worker, diagnostics as d
from hlt_classification.cms2jc2_production import contracts as k, population as p, output
from hlt_classification.cms2jc2_response import dev_diagnostics as diag
from hlt_classification.cms2jc2_response.contracts import artifact as old_artifact
from hlt_classification.cms2jc2_response.dev_data import file_ref
from hlt_classification.cms2jc2_response import generation_benchmark_engine as engine
from hlt_classification.cms2jc2_response.readers import Pair
from test_cms2jc2_proxy_production import population, study, pairs_for
from test_cms2jc2_generation_benchmark import bundle, sample, native_threads
from test_jetclass2_delphes_split_registry import registered
from test_jetclass2_delphes import snapshot
from hlt_classification.jetclass2_delphes.split_registry import select_profile


def cms_sample():
    return [Pair(p_.identity, p_.source_group, p_.offline, p_.offline) for p_ in sample(3)]


def publish_bank(study, shard, pairs):
    rows = [(pair.identity, pair.offline, '0'*64) for pair in pairs]
    relative = f"attempts/audit_fixture/shards/{shard['shard_id']}/block_00000.npz"
    path = Path(study['root'])/relative
    k.atomic_publish_bytes(path, engine.encode(engine.pack(rows), 'stored'))
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(bytes.fromhex(engine.physical_digest(pair.identity, pair.offline)))
    receipt = k.artifact('SHARD', parents={'study': study['content_hash'],
        'population': study['population']['content_hash']}, role='train',
        shard_id=shard['shard_id'], attempt='audit_fixture', jets=len(pairs),
        particles=sum(len(pair.offline) for pair in pairs), blocks=[dict(relative=relative,
        sha256=k.sha256_file(path), bytes=path.stat().st_size, jets=len(pairs))],
        ordered_identities=shard['ordered_identities'], physical_schema_verified=True,
        environment=study['numerical_environment']['content_hash'], test_lock=None,
        source=study['source']['content_hash'], output_bytes=path.stat().st_size,
        physical_digest=digest.hexdigest())
    receipt_path = Path(study['root'])/'shards'/f"{shard['shard_id']}.json"
    k.write(receipt_path, receipt)
    return dict(shard_id=shard['shard_id'], receipt=k.ref(receipt_path))


@pytest.fixture
def audit(tmp_path, monkeypatch, study, bundle):
    for shard in study['shards']:
        if shard['role'] == 'train':
            publish_bank(study, shard, pairs_for(study, shard))
    def iterate(root, population, shard, review):
        assert shard['role'] == 'train'
        yield from pairs_for(study, shard)
    monkeypatch.setattr(p, 'iterate', iterate)
    ranges = diag.fit_ranges(cms_sample(), 'a'*64)
    hist = diag.Histograms(ranges)
    for pair in cms_sample():
        for side, value in (('offline', pair.offline), ('real', pair.hlt)):
            hist.add(side, diag.conditions(pair.offline), pair.offline, value)
    reference = a.artifact('REPORT', parents={'cms_report': study['reviewed_confirmation_hash'],
                            'ranges': ranges['content_hash']}, histograms=hist.payload(),
                            cms_jets=3, source_files=1, coverage={'included_indices': list(range(36))})
    root = tmp_path/'audit'
    refs = dict(reference=k.write(root/'cms_reference.json', reference), ranges=k.write(root/'ranges.json', ranges))
    spec = a.artifact('SPEC', parents={'study': study['content_hash'], 'source': study['source']['content_hash']},
        root=str(root), project_dir=study['project_dir'], source=study['source'],
        study=k.ref(Path(study['root'])/'study_spec.json'), train=a.train_receipts(study),
        imports=refs, resources=a.RESOURCES, jets=study['counts']['train'], role='train', replica=0)
    return spec, study, reference, ranges


def test_metadata_requires_all_train_not_other_roles(audit):
    spec, study, _, _ = audit
    assert sum(k.load_json(k.checked(r['receipt']))['jets'] for r in spec['train']) == 160
    assert not (Path(study['root'])/'dataset_manifest.json').exists()
    assert not (Path(study['root'])/'test_build_lock.json').exists()
    a.load_inputs(spec)
    # An unrelated corrupt nontrain receipt must never be opened.
    shard = next(s for s in study['shards'] if s['role'] == 'final_test')
    k.atomic_publish_bytes(Path(study['root'])/'shards'/f"{shard['shard_id']}.json", b'bad')
    assert a.train_receipts(study) == spec['train']


def test_missing_train_fails_closed(audit, monkeypatch):
    spec, study, _, _ = audit
    original = k.load_json
    missing = spec['train'][0]['receipt']['path']
    def read(path):
        if str(path) == missing:
            raise FileNotFoundError(path)
        return original(path)
    monkeypatch.setattr(k, 'load_json', read)
    with pytest.raises(FileNotFoundError): a.train_receipts(study)


def test_shard_checks_pairing_and_frozen_diagnostics(audit):
    spec, study, _, ranges = audit
    row = spec['train'][0]
    ref = worker.analyze_shard(spec, study, ranges, row)
    result = k.load_json(k.checked(ref))
    assert result['jets'] == 64
    assert result['histograms']['cells']['offline/all']['jets'] == 64
    assert sum(result['paired_count_histograms']['delta'].values()) == 64
    assert worker.analyze_shard(spec, study, ranges, row) == ref


def test_wrong_identity_fails(audit, monkeypatch):
    spec, study, _, ranges = audit
    def wrong(*args):
        pairs = pairs_for(study, study['shards'][0])
        yield from reversed(pairs)
    monkeypatch.setattr(p, 'iterate', wrong)
    with pytest.raises(ValueError, match='identities'):
        worker.analyze_shard(spec, study, ranges, spec['train'][0])


def test_forbidden_shard_rejected_before_read(audit, monkeypatch):
    spec, study, _, ranges = audit
    shard = next(s for s in study['shards'] if s['role'] == 'validation')
    monkeypatch.setattr(k, 'checked', lambda _: pytest.fail('Forbidden receipt read'))
    with pytest.raises(PermissionError):
        worker.analyze_shard(spec, study, ranges, {'shard_id': shard['shard_id']})


def test_physical_corruption_fails(audit, monkeypatch):
    spec, study, _, ranges = audit
    original = output.sha256_file
    monkeypatch.setattr(output, 'sha256_file', lambda path: '0'*64 if str(path).endswith('.npz') else original(path))
    with pytest.raises(ValueError, match='checksum'):
        worker.analyze_shard(spec, study, ranges, spec['train'][0])


def test_full_report_different_populations_and_outputs(audit):
    spec, study, reference, ranges = audit
    refs = [worker.analyze_shard(spec, study, ranges, row) for row in spec['train']]
    result = worker.report(spec, reference, ranges, refs, seconds=1.)
    assert result['jets'] == 160 and result['cms_reference']['cms_jets'] == 3
    assert set(result['histograms']['cells']) >= {s+'/all' for s in d.SIDES}
    assert not any('proxy1' in n for n in result['histograms']['cells'])
    assert worker.read_report(spec) == result
    assert (Path(spec['root'])/'distributions.pdf').read_bytes().startswith(b'%PDF')
    assert 'not uncertainty' in d.text_table(result)
    assert 'underflow_fraction' in d.csv_table(result)
    assert worker.report(spec, reference, ranges, refs, seconds=2.) == result
    with pytest.raises(ValueError, match='Incomplete'):
        worker.report(spec, reference, ranges, refs[:-1], seconds=1.)


def test_exact_count_changes_and_quantiles():
    class Particle:
        def __init__(self, categories): self.category = np.asarray(categories)
        def __len__(self): return len(self.category)
    hist = {}
    for before, after in (([0, 0], [0]), ([0], [0, 1, 2]), ([0], [0])):
        d.paired_counts(Particle(before), Particle(after), hist)
    result = d.count_summary(d.merge_counts([hist, hist]))
    assert result['delta']['jets'] == 6
    assert result['delta']['mean'] == pytest.approx(1/3)
    assert result['delta']['negative_fraction'] == pytest.approx(1/3)
    assert result['delta']['quantiles']['0.5'] == 0
    assert result['delta']['quantiles']['0.99'] == 2
    assert result['pid_0_delta']['mean'] == pytest.approx(-1/3)


def test_moments_tv_and_missing():
    a = dict(count=2, covered_jets=2, sum=4., sumsq=10., minimum=1., maximum=3., bins=[0, 1, 1, 0])
    b = {**a, 'count': 4, 'bins': [0, 2, 2, 0]}
    assert d.moments(a)['mean'] == 2 and d.moments(a)['sd'] == 1
    assert d.tv(a, b) == 0
    assert d.tv(a, None) is None
    assert d.moments(None)['mean'] is None


def test_plan_is_single_cpu_only_job(audit):
    spec, *_ = audit
    plan = a.plan(spec)
    argv = plan['argv']
    assert '--cpus-per-task=8' in argv and '--mem=64G' in argv and '--time=08:00:00' in argv
    assert '--partition=tigris' in argv and not any('gpu' in v or 'array=' in v for v in argv)
    assert argv[-2] == spec['project_dir']


def test_submission_requires_hash_and_no_duplicate(audit, monkeypatch):
    spec, *_ = audit
    k.write(Path(spec['root'])/'command_plan.json', a.plan(spec))
    monkeypatch.setattr(a, 'source_snapshot', lambda *args: spec['source'])
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(a.subprocess, 'run', run)
    assert a.submit(spec)['content_hash'] == a.plan(spec)['content_hash'] and not calls
    with pytest.raises(PermissionError): a.submit(spec, execute=True, reviewed_hash='0'*64)
    result = a.submit(spec, execute=True, reviewed_hash=a.plan(spec)['content_hash'])
    assert result['job'] == '12345' and len(calls) == 2
    with pytest.raises(FileExistsError): a.submit(spec, execute=True, reviewed_hash=a.plan(spec)['content_hash'])
    assert sum('--test-only' not in argv for argv in calls) == 1


def test_scope_change_rejected(audit):
    spec, *_ = audit
    bad = k.with_content_hash({**spec, 'role': 'final_test'})
    with pytest.raises(ValueError, match='scope'): a.load_inputs(bad)


def test_no_login_node_run(audit, monkeypatch):
    spec, *_ = audit
    monkeypatch.setattr(a, 'source_snapshot', lambda *args: spec['source'])
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(PermissionError, match='login node'): worker.run(spec)


def cms_fixture(tmp_path):
    """Real byte-authenticated reduced -> frozen -> pilot metadata chain."""
    ranges = diag.fit_ranges(cms_sample(), 'a'*64)
    hist = diag.Histograms(ranges)
    for pair in cms_sample():
        for side, output_ in (('offline', pair.offline), ('real', pair.hlt), ('proxy0', pair.hlt)):
            hist.add(side, diag.conditions(pair.offline), pair.offline, output_)
    pilot_root = tmp_path/'old'
    pilot = old_artifact('DEV_STAGE', root=str(pilot_root), name='pilot_r1', stage='pilot')
    pilot_path = pilot_root/'stages/pilot_r1/stage_spec.json'
    k.write(pilot_path, pilot)
    ranges_ref = k.write(pilot_root/'ranges.json', ranges)
    receipt = old_artifact('DEV_OUTPUTS', parents={'stage': pilot['content_hash']}, owner='prepare',
        outputs={'ranges': {'relative': 'ranges.json', 'sha256': ranges_ref['sha256']}})
    k.write(pilot_path.parent/'receipts/prepare.json', receipt)
    frozen = old_artifact('FROZEN_STAGE', stage='frozen_confirm', parent_spec=file_ref(pilot_path),
        reuse={'parents': {'ranges': ranges['content_hash']}})
    frozen_path = tmp_path/'frozen.json'; k.write(frozen_path, frozen)
    reduced = old_artifact('FROZEN_REDUCED_STAGE', stage='frozen_reduced', subject_spec=file_ref(frozen_path))
    reduced_ref = k.write(tmp_path/'reduced.json', reduced)
    report = old_artifact('FROZEN_REDUCED_REPORT', jets=3,
        by_file={'a': {'JOINT': hist.payload()}}, coverage={'included_indices': list(range(36))},
        full_population_status='inconclusive_incomplete_population', decision={'status': 'rejected'})
    root = tmp_path/'dataset'
    report_ref = k.write(root/'provenance/confirmation.json', report)
    study = dict(root=str(root), confirmation_spec=reduced_ref, reviewed_confirmation_hash=report['content_hash'],
        imports={'confirmation_report': dict(relative='provenance/confirmation.json',
                  sha256=report_ref['sha256'], bytes=report_ref['bytes'])})
    return study, ranges


def test_reduced_cms_reference_discovery_preserves_negative_status(tmp_path):
    study, ranges = cms_fixture(tmp_path)
    reference, actual = a.cms_reference(study)
    assert actual == ranges
    assert reference['cms_jets'] == 3
    assert reference['decision']['status'] == 'rejected'
    assert reference['qualification_status'] == 'inconclusive_incomplete_population'
    assert reference['coverage']['included_indices'] == list(range(36))
    assert all(k_.split('/')[0] in ('offline', 'real') for k_ in reference['histograms']['cells'])


def test_cms_reference_rejects_wrong_report_hash(tmp_path):
    study, _ = cms_fixture(tmp_path)
    study['reviewed_confirmation_hash'] = 'f'*64
    with pytest.raises(ValueError, match='report identity'): a.cms_reference(study)


def test_create_freezes_inputs_and_disjoint_location(audit, tmp_path, monkeypatch):
    spec, study, reference, ranges = audit
    project = tmp_path/'project'
    names = ('cms2jc2_production/population.py', 'cms2jc2_response/bridge.py',
             'cms2jc2_response/dev_diagnostics.py', 'cms2jc2_response/metrics.py',
             'cms2jc2_response/features.py')
    files = {'src/hlt_classification/'+n: 'a'*64 for n in names}
    fake = deepcopy(study)
    fake['source']['files'] = files
    original_load = k.load_json
    def load(path):
        return fake if Path(path) == Path(study['root'])/'study_spec.json' else original_load(path)
    monkeypatch.setattr(k, 'load_json', load)
    monkeypatch.setattr(a.production, 'validate_study', lambda _: None)
    monkeypatch.setattr(a.production, 'require_preflight', lambda _: None)
    source = a.artifact('SOURCE', commit='a'*40, files=files)
    monkeypatch.setattr(a, 'source_snapshot', lambda *args: source)
    monkeypatch.setattr(a, 'cms_reference', lambda _: (reference, ranges))
    result = a.create(dataset_root=study['root'], root=tmp_path/'new_audit', project=project, commit='a'*40)
    assert result['train'] == spec['train'] and result['jets'] == 160
    assert (tmp_path/'new_audit/command_plan.json').is_file()
    with pytest.raises(ValueError, match='separate'):
        a.create(dataset_root=study['root'], root=Path(study['root'])/'new_audit', project=project, commit='a'*40)


def test_real_root_spawned_shard_matches_serial(registered, tmp_path, monkeypatch):
    data_root, inv, _, registry = registered
    monkeypatch.setattr(p, 'PROFILE', 'TRAIN_44')
    monkeypatch.setattr(p, 'COUNTS', dict(train=44, validation=5, final_test=11))
    pop = p.build(inv, select_profile(registry, inv, 'TRAIN_44'), select_profile(registry, inv, 'TRAIN_22'))
    file = next(r for r in pop['files'] if r['role'] == 'train')
    entries = p.unpack_entries(file['entry_mask'], file['raw_entries'])
    shard = dict(shard_id='s00000', role='train', path=file['path'], start=0, stop=len(entries),
        jets=len(entries), ordered_identities=k.digest_ids(p.ids(inv['content_hash'], file, entries)))
    from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
    review = provisional_compatibility(inventory_hash='a'*64)
    source = k.artifact('SOURCE', files={}, executable=True)
    env = old_artifact('NUMERICAL_ENVIRONMENT', fixture=True)
    study = k.artifact('STUDY', root=str(tmp_path/'physical'), population=pop, shards=[shard],
        data_root=str(data_root), review=review, source=source, numerical_environment=env)
    import uproot
    original_arrays = uproot.behaviors.TBranch.HasBranches.arrays
    reads = []
    def offline_only(self, expressions=None, *args, **kwargs):
        assert expressions == list(p.BRANCHES)
        assert not any('hlt' in name or 'label' in name for name in expressions)
        reads.append(expressions)
        return original_arrays(self, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, 'arrays', offline_only)
    pairs = list(p.iterate(data_root, pop, shard, review))
    row = publish_bank(study, shard, pairs)
    ranges = diag.fit_ranges(cms_sample(), 'a'*64)
    spec = a.artifact('SPEC', root=str(tmp_path/'serial'))
    serial = k.load_json(k.checked(worker.analyze_shard(spec, study, ranges, row)))
    process_spec = a.artifact('SPEC', root=str(tmp_path/'process'))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        ref = pool.submit(worker.analyze_shard, process_spec, study, ranges, row).result(timeout=90)
    parallel = k.load_json(k.checked(ref))
    assert reads and serial['jets'] == len(pairs)
    for field in ('histograms', 'paired_count_histograms', 'physical_digest'):
        assert serial[field] == parallel[field]
