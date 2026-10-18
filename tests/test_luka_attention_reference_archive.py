"""Synthetic retained-result fixtures; no ROOT reads, training or submissions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (
    load_json, load_npz_arrays, with_content_hash, write_immutable_json,
    deterministic_npz_bytes, atomic_publish_bytes,
)
from hlt_classification.jetclass2_delphes.reporting import evaluate_probabilities, recovery
from hlt_classification.luka_attention import contracts as old
from hlt_classification.luka_fullsim import alljets_data
from hlt_classification.luka_fullsim.regular_campaign import nodes as parent_nodes
from hlt_classification.luka_attention_weight import contracts as new, source, campaign
from hlt_classification.luka_attention_weight import reference_archive as archive


def overwrite(path, record):
    """Only test fixtures are deliberately re-signed/corrupted."""
    path.write_text(json.dumps(with_content_hash(record)), encoding='utf-8')


@pytest.fixture
def saved(tmp_path, monkeypatch):
    root = tmp_path/'old/science'
    root.mkdir(parents=True)
    inventory = dict(files=[dict(path='file.root', sha256='a'*64, tree_key='tree')])
    members = [dict(file_index=0, entries_by_class=[[c, c+11] for c in range(11)])]
    splits = dict(memberships=dict(validation=members), comparison_rows=11,
                  comparison_memberships=[dict(file_index=0, entries_by_class=[[c] for c in range(11)])])
    identities, labels = alljets_data.row_arrays(inventory, members)
    rng = np.random.default_rng(37)
    p = rng.uniform(.1, 1., (22, 11)).astype(np.float32)
    p /= p.sum(1, keepdims=True)
    metrics = evaluate_probabilities(labels, p)
    original = evaluate_probabilities(labels[:11], p[:11])
    models = {}
    for name in ('M0HLT', 'OFFLINE', 'DIRECT_D000', 'COARSE_U100'):
        directory = tmp_path/'parent'/name
        for filename, data in (('report.json', dict(validation=metrics)),
                               ('diagnostics.json', dict(original_validation=original)),
                               ('receipt.json', dict(fixture=True)),
                               ('selected.pt', dict(fixture_not_real_checkpoint=True))):
            write_immutable_json(directory/filename, data)
        models[name] = {key: file_ref(directory/filename) for key, filename in
            [('report', 'report.json'), ('diagnostics', 'diagnostics.json'),
             ('receipt', 'receipt.json'), ('checkpoint', 'selected.pt')]}
    write_immutable_json(tmp_path/'parent/spec.json', dict(fixture=True))
    anchor = old.artifact('ANCHOR', spec=file_ref(tmp_path/'parent/spec.json'),
        evidence=dict(prepared_root=str(tmp_path/'prepared'), raw_container=str(tmp_path/'raw'),
                      prepared='b'*64, cache='c'*64, models=models))
    nodes = old.nodes(parent_nodes())
    gate = old.artifact('GATE', root=str(root.parent), project_dir=str(tmp_path/'old_source'),
        source_commit=archive.REFERENCE_COMMIT, anchor=anchor, nodes=nodes, tasks=old.tasks(nodes),
        counts=dict(train=22, validation=22))
    write_immutable_json(root.parent/'gate_spec.json', gate)
    preflight = old.artifact('PREFLIGHT', parents=dict(gate=gate['content_hash']),
        technical_checks_passed=True, scientific_fit=False, train_minutes=100, export_minutes=15,
        outputs=[dict(path='export_ANCHOR/bank/attention-0000000.npy', bytes=100, sha256='d'*64)])
    write_immutable_json(root.parent/'preflight/receipt.json', preflight)
    spec = old.artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=preflight['content_hash']),
        root=str(root), gate_path=str(root.parent/'gate_spec.json'), project_dir=gate['project_dir'],
        source_commit=gate['source_commit'], train_minutes=100, export_minutes=15)
    write_immutable_json(root/'campaign_spec.json', spec)
    rows = [dict(node_id=name, origin='parent_fixed30', validation=metrics, original_validation=original)
            for name in models]
    receipts = {}
    for task in gate['tasks'][:-1]:
        name = task['task_id']
        outputs = []
        if task['kind'] == 'train':
            node = next(n for n in nodes if n['node_id'] == task['node_id'])
            report = old.artifact('TRAINING_REPORT', parents=dict(recipe=old.recipe()['content_hash'],
                prepared='b'*64, bank='e'*64), node=node, scientific_fit=True, passes=30,
                selected_weights_restored=True, selected_pass=1, validation=metrics,
                validation_history=[dict(pass_number=i, update=i, validation=metrics) for i in range(1, 31)])
            write_immutable_json(root/name/'training_report.json', report)
            write_immutable_json(root/name/'costs.json', dict(parameters=10))
            atomic_publish_bytes(root/name/'validation_predictions.npz', deterministic_npz_bytes(
                dict(identities=identities, labels=labels, probabilities=p)))
            outputs = [file_ref(root/name/f, root=root) for f in
                       ('training_report.json', 'costs.json', 'validation_predictions.npz')]
            outputs.append(dict(path=name+'/selected.pt', bytes=10, sha256='f'*64))
            rows.append(dict(node_id=node['node_id'], origin='new', validation=metrics,
                             original_validation=original, costs=dict(parameters=10)))
        else:
            outputs = [dict(path=name+'/bank/manifest.json', bytes=12, sha256='f'*64)]
        receipt = old.artifact('TASK', parents=dict(spec=spec['content_hash']), task_id=name, job_id='123',
            dependencies={d: receipts[d]['content_hash'] for d in task['dependencies']}, outputs=outputs)
        receipts[name] = receipt
        # No export artifacts, even their receipts, survive this synthetic fixture.
        if task['kind'] == 'train':
            write_immutable_json(root/name/'receipt.json', receipt)
    for row in rows:
        row['recovery'] = {field: recovery(row[field], rows[0][field], rows[1][field])
                           for field in ('validation', 'original_validation')}
    result = old.artifact('RESULTS', parents=dict(spec=spec['content_hash']), rows=rows,
        primary_contrast='ATTN_D000 minus LOGIT_D000', single_seed=True,
        contrasts={field: {key: 0. for key in ('accuracy', 'macro_ovr_auc', 'cross_entropy')}
                   for field in ('validation', 'original_validation')})
    write_immutable_json(root/'aggregate/results.json', result)
    receipt = old.artifact('TASK', parents=dict(spec=spec['content_hash']), task_id='aggregate',
        dependencies={d: r['content_hash'] for d, r in receipts.items()}, job_id='124',
        outputs=[file_ref(root/'aggregate/results.json', root=root)])
    write_immutable_json(root/'aggregate/receipt.json', receipt)

    # Real archived validators/selector/metrics/row join; only source and prepared
    # inventory authentication are fixture stand-ins (covered by existing tests).
    monkeypatch.setattr(archive.campaign, 'check_gate', lambda g: None)
    monkeypatch.setattr(archive.regular_cache, 'metadata', lambda _: (
        dict(content_hash='c'*64), dict(content_hash='b'*64), inventory, splits))
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not require old payloads or open raw particles')
    monkeypatch.setattr(archive.campaign, 'check_spec', forbidden)
    monkeypatch.setattr(archive.worker, 'completed', forbidden)
    monkeypatch.setattr(alljets_data, 'ParticleReader', forbidden)
    return SimpleNamespace(root=root, spec=spec, gate=gate, rows=rows, anchor=anchor)


def test_deleted_old_preflight_banks_and_checkpoints_are_not_needed(saved):
    before = {str(p): p.read_bytes() for p in saved.root.parent.rglob('*') if p.is_file()}
    result = archive.read_archive(saved.root/'campaign_spec.json')
    assert result['rows'] == saved.rows and len(result['predictions']) == 6
    assert all(result['archive'][k] == v for k, v in archive.SCOPE.items())
    assert not (saved.root.parent/'preflight/export_ANCHOR').exists()
    assert not (saved.root/'train_ATTN_D000/selected.pt').exists()
    assert before == {str(p): p.read_bytes() for p in saved.root.parent.rglob('*') if p.is_file()}


@pytest.mark.parametrize('filename', ['receipt.json', 'training_report.json', 'costs.json', 'validation_predictions.npz'])
@pytest.mark.parametrize('corrupt', [False, True])
def test_retained_science_outputs_must_exist_and_authenticate(saved, filename, corrupt):
    path = saved.root/'train_ATTN_D000'/filename
    if corrupt:
        with path.open('ab') as stream:
            stream.write(b'corrupt')
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.read_archive(saved.root/'campaign_spec.json')


def rebind_training(saved, name):
    """Re-sign all retained direct parents so tests exercise semantic checks."""
    task = 'train_'+name
    receipt = load_json(saved.root/task/'receipt.json')
    receipt['outputs'] = [file_ref(saved.root/r['path'], root=saved.root)
                           if not r['path'].endswith('selected.pt') else r for r in receipt['outputs']]
    overwrite(saved.root/task/'receipt.json', receipt)
    aggregate = load_json(saved.root/'aggregate/receipt.json')
    aggregate['dependencies'][task] = load_json(saved.root/task/'receipt.json')['content_hash']
    overwrite(saved.root/'aggregate/receipt.json', aggregate)


@pytest.mark.parametrize('change', ['order', 'labels', 'probabilities', 'nan', 'schema'])
def test_resigned_prediction_mismatch_rejected(saved, change):
    path = saved.root/'train_ATTN_D000/validation_predictions.npz'
    p = load_npz_arrays(path)
    if change == 'order':
        p = {k: v[::-1] for k, v in p.items()}
    elif change == 'labels':
        p['labels'] = (p['labels']+1)%11
    elif change == 'probabilities':
        p['probabilities'] = p['probabilities'][:, ::-1]
    elif change == 'nan':
        p['probabilities'][0, 0] = np.nan
    else:
        p['extra'] = np.array([1])
    path.write_bytes(deterministic_npz_bytes(p))
    rebind_training(saved, 'ATTN_D000')
    with pytest.raises(ValueError):
        archive.read_archive(saved.root/'campaign_spec.json')


@pytest.mark.parametrize('change', ['selection', 'recipe', 'bank', 'node', 'prepared', 'dependency', 'escape'])
def test_resigned_history_and_lineage_mismatch_rejected(saved, change):
    directory = saved.root/'train_ATTN_D000'
    if change in ('dependency', 'escape'):
        r = load_json(directory/'receipt.json')
        if change == 'dependency':
            r['dependencies']['export_ATTN_D033'] = '0'*64
        else:
            r['outputs'][-1]['path'] = 'train_ATTN_D000/../foreign/selected.pt'
        overwrite(directory/'receipt.json', r)
    else:
        r = load_json(directory/'training_report.json')
        if change == 'selection': r['selected_pass'] = 2
        if change == 'node': r['node']['attention_coefficient'] = .3
        if change in ('recipe', 'prepared'): r['parents'][change] = '0'*64
        if change == 'bank': r['parents']['bank'] = 'not a hash'
        overwrite(directory/'training_report.json', r)
    if change == 'escape':
        aggregate = load_json(saved.root/'aggregate/receipt.json')
        aggregate['dependencies']['train_ATTN_D000'] = load_json(directory/'receipt.json')['content_hash']
        overwrite(saved.root/'aggregate/receipt.json', aggregate)
    else:
        rebind_training(saved, 'ATTN_D000')
    with pytest.raises(ValueError): archive.read_archive(saved.root/'campaign_spec.json')


def test_resigned_aggregate_metrics_rejected(saved):
    result = load_json(saved.root/'aggregate/results.json')
    result['rows'][-1]['validation']['accuracy'] += .01
    overwrite(saved.root/'aggregate/results.json', result)
    receipt = load_json(saved.root/'aggregate/receipt.json')
    receipt['outputs'] = [file_ref(saved.root/'aggregate/results.json', root=saved.root)]
    overwrite(saved.root/'aggregate/receipt.json', receipt)
    with pytest.raises(ValueError, match='aggregate differs'):
        archive.read_archive(saved.root/'campaign_spec.json')


def test_archive_wrapper_snapshots_and_rejects_different_anchor(saved, monkeypatch):
    anchor = new.artifact('ANCHOR', spec=saved.anchor['spec'], evidence=saved.anchor['evidence'])
    def replay(path, project, program, marker, *, extra):
        assert Path(extra[0]).name == 'reference_archive.py' and project == saved.spec['project_dir']
        return file_ref(path), archive.read_archive(path)
    monkeypatch.setattr(source, 'replay', replay)
    result = source.references(saved.root/'campaign_spec.json', anchor)
    source.validate_archive(result, anchor)
    bad = deepcopy(result)
    bad['archive']['legacy_preflight_reusable'] = True
    bad['archive'] = with_content_hash(bad['archive'])
    with pytest.raises(ValueError, match='scope'):
        source.validate_archive(with_content_hash(bad), anchor)
    wrong = deepcopy(anchor)
    wrong['evidence']['models']['COARSE_U100']['checkpoint']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='anchor differs'):
        source.references(saved.root/'campaign_spec.json', with_content_hash(wrong))


def test_original_source_replay_uses_pinned_pythonpath_and_explicit_reader(saved, monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout='WEIGHT_REFERENCES={"fixture":true}\n')
    monkeypatch.setattr(source.subprocess, 'run', run)
    reader = Path(archive.__file__)
    _, result = source.replay(saved.root/'campaign_spec.json', saved.spec['project_dir'],
        source.REFERENCE_CHECK, 'WEIGHT_REFERENCES=', extra=(reader,))
    assert result == dict(fixture=True)
    cmd, kwargs = calls[0]
    assert cmd[-1] == str(reader) and '-B' in cmd and '-s' in cmd
    assert kwargs['cwd'] == saved.spec['project_dir']
    assert kwargs['env']['PYTHONPATH'] == str(Path(saved.spec['project_dir'])/'src')
    assert kwargs['env']['PYTHONNOUSERSITE'] == '1'


def test_archive_never_authorizes_new_science_without_own_preflight(saved, monkeypatch):
    monkeypatch.setattr(campaign, 'check_gate', lambda g: None)
    # An old reference's accepted preflight is not the new arm's acceptance.
    gate = dict(root=str(saved.root.parent.parent/'new_arm'))
    with pytest.raises(FileNotFoundError): campaign.science(gate)
    assert not (Path(gate['root'])/'science').exists()
