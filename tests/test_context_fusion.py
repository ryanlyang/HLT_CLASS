"""Fusion graph, identity joins, native wrapper, exact DAG, and real CPU miniature.

Hardware/parity/source pinning are substituted in the synthetic workflow only;
this does not claim Oscar or installed-Weaver acceptance.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hlt_classification.context_fusion import campaign as c, inputs as i, worker as w
from hlt_classification.context_fusion.contracts import artifact, validate, load_json, write_json, file_ref, validate_file_ref
from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes import runner
from hlt_classification.jetclass2_delphes.contracts import artifact as kernel_artifact
from hlt_classification.jetclass2_delphes.model import model_contract
from test_jetclass2_dzfix_fusion_chain import make_cache, fake_native
from test_context_ladder import parent, parent_v2, final_pilot, study, relocated, consumer, dataset, released


def test_frozen_graph_schedule_seeds_and_deployability():
    nodes, tasks = c.nodes(), c.tasks()
    assert len(nodes) == 8 and len(tasks) == 16
    assert sum(t['kind'] == 'reduce' for t in tasks) == 6
    assert tasks[0]['dependencies'] == []  # Imported U000, not an expired Slurm ID.
    assert nodes[0]['teacher'] == 'U000'
    assert [(n['node_id'], n['deployable']) for n in nodes][-4:] == [
        ('FUSION_D000', False), ('FINAL_DIRECT_D000', True),
        ('FUSION_D000_D000', True), ('FINAL_BRIDGE_D000', True)]
    assert nodes[5]['initialization_seed'] == nodes[7]['initialization_seed'] == c.paired_seed('D000', 'initialization')
    assert nodes[5]['sampler_seed'] == nodes[7]['sampler_seed'] == c.paired_seed('D000', 'sampler')
    assert c.registration()['training_recipe'] == runner.recipe()
    assert c.registration()['counts'] == dict(train=100000, validation=50000)
    assert 'same_entire_50k' in c.registration()['validation_policy']
    assert c.registration()['alpha'] == 1.
    assert not any('DENSE' in t['task_id'] for t in tasks)
    seen = set()
    for task in tasks:
        assert set(task['dependencies']) <= seen
        seen.add(task['task_id'])


def test_paired_transport_restores_native_widths_and_values():
    a, b = make_cache('train'), make_cache('train', 'U000')
    # Extend the context only. A different sequence length must not be silently
    # imposed on the primary pair-embedding BN population.
    b.blocks[0].offsets = np.arange(len(b)+1)*20
    b.blocks[0].features = np.repeat(b.blocks[0].features, 5, axis=0)
    b.blocks[0].vectors = np.repeat(b.blocks[0].vectors, 5, axis=0)
    pair = i.PairedCache(a, b)
    indices = np.arange(3)
    raw = pair.batch(indices)
    result = i.unpack(*(torch.from_numpy(raw[k]) for k in ('features', 'vectors', 'mask')))
    for output, expected in zip(result, (a.batch(indices), b.batch(indices))):
        for key, tensor in zip(('features', 'vectors', 'mask'), output):
            np.testing.assert_array_equal(tensor.numpy(), expected[key])
    np.testing.assert_array_equal(raw['identities'], a.identities[:3])
    assert raw['features'].shape[1:3] == (2, 17)
    for field in ('identities', 'labels', 'role', 'foundation_sha256'):
        other = deepcopy(b)
        if field in ('identities', 'labels'):
            setattr(other, field, np.roll(getattr(other, field), 1, axis=0))
        else:
            setattr(other, field, 'bad')
        with pytest.raises(ValueError, match='join'):
            i.PairedCache(a, other)


def test_hlt_pair_alias_and_test_firewall():
    a = make_cache('train')
    pair = i.PairedCache(a, a)
    assert pair.nbytes == a.nbytes
    raw = pair.batch(np.arange(4))
    np.testing.assert_array_equal(raw['features'][:, 0], raw['features'][:, 1])
    with pytest.raises(PermissionError):
        i.cache({}, c.nodes()[0], 'final_test', workers=1, memory_mb=1)
    with pytest.raises(ValueError):
        i.unpack(torch.zeros(1, 17, 16), torch.zeros(1, 4, 16), torch.zeros(1, 1, 16))


def test_real_context_foundation_paired_loader(released, tmp_path):
    from hlt_classification.cms_proxy_ladder import data, cache
    release, release_root = released
    foundation_root = tmp_path/'foundation'
    foundation = data.build_foundation(release, release_root=release_root,
        output_root=foundation_root, capacity=512, workers=1)
    parent_spec = dict(foundation=foundation, foundation_root=str(foundation_root))
    assert foundation['schema_version'] == 3
    for role in ('train', 'validation'):
        for node in (c.nodes()[0], c.nodes()[6], c.nodes()[5]):
            paired = i.cache(parent_spec, node, role, workers=1, memory_mb=180000)
            raw = paired.batch(np.arange(3))
            if node['context_coordinate'] is None:
                views = [(raw, node['coordinate'])]
            else:
                unpacked = i.unpack(*(torch.from_numpy(raw[k]) for k in ('features','vectors','mask')))
                views = [(dict(zip(('features','vectors','mask'), tensors)), coord)
                    for tensors, coord in zip(unpacked, (node['coordinate'], node['context_coordinate']))]
                if node['context_coordinate'] == node['coordinate']:
                    assert paired.primary is paired.context
            for actual, coord in views:
                reference = cache.prepare_cache(foundation, foundation_root=foundation_root,
                    role=role, coordinate=coord, workers=1, max_ram_bytes=2**30)
                expected = reference.batch(np.arange(3))
                np.testing.assert_array_equal(paired.identities, reference.identities)
                for key in ('features','vectors','mask'):
                    np.testing.assert_array_equal(actual[key], expected[key])
    with pytest.raises(MemoryError):
        i.budgets(parent_spec, workers=1, memory_mb=1)


def test_fusion_wrapper_logits_and_gradients_match_native(fake_native):
    pair = i.PairedCache(make_cache('train'), make_cache('train', 'U000'))
    raw = pair.batch(np.arange(4))
    model = i.new_model(c.nodes()[0]).eval()
    for injection in model.injections:
        torch.nn.init.normal_(injection.residual_projection.weight, std=.01)
    data = [torch.from_numpy(raw[k]) for k in ('features', 'vectors', 'mask')]
    xs = data[0].requires_grad_(True)
    output = model(xs, *data[1:])
    output.square().sum().backward()
    gradient = xs.grad.clone()
    model.zero_grad(set_to_none=True)
    xs2 = data[0].detach().clone().requires_grad_(True)
    a, b = i.unpack(xs2, *data[1:])
    expected = model.forward_fused(*a, *b, alpha=1.).logits
    expected.square().sum().backward()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, xs2.grad, rtol=0, atol=0)
    assert gradient[:, 1].abs().sum() > 0


def evidence(spec, parent):
    stats = {name: dict(calls=3, saved_cuda_tensors=1, saved_cuda_bytes=100, restored_cuda_tensors=1)
        for name in c.PAIR_OFFLOAD_POLICY['scope']}
    parities = {}
    for kind in ('adjacent', 'endpoint'):
        for precision in ('fp32', 'bf16'):
            parities[kind+'_'+precision] = dict(passed=True, device_type='cuda', precision=precision,
                steps=3, checks=c.PARITY_CHECKS, parity_backend=c.PARITY_BACKEND,
                saved_tensor_storage=c.PAIR_OFFLOAD_POLICY, tolerance=c.PARITY_TOLERANCES[precision], offload_stats=stats)
    measurements = []
    for index in (0, 6, 5):
        node = c.nodes()[index]
        report = kernel_artifact('KERNEL_TRAINING_REPORT', node=node,
            foundation_sha256=parent['foundation']['content_hash'], recipe_sha256=c.recipe()['content_hash'],
            acceptance_only=True, scientific_fit=False, passes=1, runtime_seconds=10., final_test_accessed=False)
        measurements.append(dict(node=node, training=report, cache_seconds=1., inference_seconds=1.,
            stress_steps=3, stress_batch=256, stress_offload=stats if node['context_coordinate'] else None))
    return artifact('PREFLIGHT', parents=dict(spec=spec['content_hash']), passed=True,
        counts=spec['counts'], site=spec['execution_site'], source_commit=spec['source_commit'],
        single_parity=kernel_artifact('WEAVER_PARITY', passed=True, device='cuda', model=model_contract(),
            forward_and_feature_and_parameter_gradients=True),
        native_mask_parity=True, checkpoint_bank_roundtrip=True,
        environment=kernel_artifact('INSTALLED_ENVIRONMENT', version=2, architecture='x86_64'),
        gpu=dict(name='L40S', total_memory_bytes=48*2**30), gpu_peak_bytes=2**30, peak_rss_bytes=2**30,
        cache_bytes=2**30, measurements=measurements, offload_parity=parities, train_minutes=60, reduce_minutes=30)


def simple_spec(tmp_path):
    spec = artifact('SPEC', root=str(tmp_path/'run'), project_dir=str(tmp_path/'source'), source_commit='a'*40,
        parent_import={'foundation_sha256': 'f'*64}, **c.registration())
    return spec, dict(foundation={'content_hash': 'f'*64})


def test_preflight_rejects_scope_resource_and_parity_drift(tmp_path):
    spec, parent = simple_spec(tmp_path)
    value = evidence(spec, parent)
    assert c.validate_preflight(value, spec, parent) == value
    changes = [('counts', dict(train=1000000, validation=250000)), ('passed', False),
        ('source_commit', 'b'*40), ('train_minutes', 2881), ('gpu_peak_bytes', 48*2**30),
        ('peak_rss_bytes', 180000*1024**2), ('cache_bytes', 180000*1024**2),
        ('native_mask_parity', False), ('checkpoint_bank_roundtrip', False), ('measurements', [])]
    for key, val in changes:
        with pytest.raises(ValueError):
            c.validate_preflight(with_content_hash(dict(value, **{key: val})), spec, parent)
    for key in ('steps', 'device_type', 'checks'):
        bad = deepcopy(value)
        bad['offload_parity']['adjacent_fp32'][key] = None
        with pytest.raises(ValueError):
            c.validate_preflight(with_content_hash(bad), spec, parent)
    bad = deepcopy(value)
    bad['offload_parity']['endpoint_bf16']['offload_stats']['cross']['restored_cuda_tensors'] = 0
    with pytest.raises(ValueError):
        c.validate_preflight(with_content_hash(bad), spec, parent)


def test_gate_science_plan_and_exact_submission_guards(tmp_path, monkeypatch):
    spec, parent = simple_spec(tmp_path)
    Path(spec['root']).mkdir()
    monkeypatch.setattr(c, 'validate_spec', lambda s: parent)
    gate = c.submit(spec, mode='gate')
    assert len(gate['commands']) == 1
    assert (Path(spec['root'])/'submission_gate/dry_run_submission_ledger.json').is_file()
    with pytest.raises(FileNotFoundError):
        c.plan(spec, 'science')
    path = Path(spec['root'])/'preflight/result.json'
    write_json(path, evidence(spec, parent))
    plan = c.submit(spec, mode='science')
    assert len(plan['commands']) == 16
    assert not (Path(spec['root'])/'submission_science/submission_ledger.json').exists()
    for row in plan['commands']:
        args = row['command']
        assert '--partition=gpu' in args and '--account=default' in args and '--qos=norm-gpu' in args
        assert '--export=NONE' in args and '--no-requeue' in args
        assert '--comment=jc2cxf:'+spec['content_hash']+':'+row['task_id'] in args
        assert not any('JOB_reduce_U000' in a for a in args)
        if row['task_id'].startswith(('train_', 'reduce_')):
            assert '--gres=gpu:l40s:1' in args and '--mem=180000M' in args
        else:
            assert not any(a.startswith('--gres=') for a in args)
    assert 'CUBLAS_WORKSPACE_CONFIG' in gate['commands'][0]['command'][-1]
    assert not any('CUBLAS_WORKSPACE_CONFIG' in r['command'][-1] for r in plan['commands'])
    monkeypatch.setattr(c.context, 'check_submission_site', lambda p: pytest.fail('Do not reach live checks'))
    with pytest.raises(PermissionError):
        c.submit(spec, mode='science', execute=True, plan_hash='0'*64, authorization=c.AUTHORIZATION)
    with pytest.raises(PermissionError):
        c.submit(spec, mode='science', execute=True, plan_hash=plan['content_hash'], authorization='wrong')


def test_final_test_and_protected_output_rejected(tmp_path):
    with pytest.raises(PermissionError):
        validate(with_content_hash(dict(artifact('SPEC'), final_test_accessed=True)), 'SPEC')
    parent = dict(project_dir=str(tmp_path/'oldsource'), campaign_root=str(tmp_path/'oldrun'),
        gate_root=str(tmp_path/'gate'), foundation_root=str(tmp_path/'foundation'),
        foundation=dict(release_root=str(tmp_path/'release'), release=dict(study_root=str(tmp_path/'proxy'),
            offline_root=str(tmp_path/'offline'), request=dict(provenance_root=str(tmp_path/'provenance')))))
    for root in (tmp_path, tmp_path/'proxy/new', tmp_path/'oldrun/new', tmp_path/'source/new'):
        with pytest.raises(PermissionError):
            c.validate_location(root, parent, tmp_path/'source')
    c.validate_location(tmp_path/'newrun', parent, tmp_path/'source')


def test_parent_import_authenticates_finished_scientific_teacher(tmp_path, monkeypatch):
    parent_spec = dict(content_hash='p'*64, campaign_root=str(tmp_path),
        source_commit=c.PARENT_COMMIT, schema_version=4,
        foundation=dict(role_counts=c.COUNTS, content_hash='f'*64),
        scientific_plan=dict(recipe=c.recipe()))
    path = tmp_path/'campaign_spec.json'
    write_json(path, parent_spec)
    rows = [dict(node_id=name, state='COMPLETE') for name in ('M0HLT','OFFLINE','U000',*[f'other_{i}' for i in range(6)])]
    pointers = {}
    for row in rows:
        name = row['node_id']
        report = kernel_artifact('KERNEL_TRAINING_REPORT', scientific_fit=True, acceptance_only=False,
            recipe_sha256=c.recipe()['content_hash'], final_test_accessed=False)
        write_json(tmp_path/(name+'.json'), report)
        pointers['train_'+name] = dict(result=dict(training_report=name+'.json', training_report_sha256=report['content_hash']))
    pointers['reduce_U000'] = dict(result=dict(train_bank='bank',
        teacher_report_sha256=pointers['train_U000']['result']['training_report_sha256']))
    for name, pointer in pointers.items():
        write_json(tmp_path/'tasks'/(name+'.json'), pointer)
    calls = []
    monkeypatch.setattr(c.context, 'validate_campaign', lambda p, **kw: calls.append(kw))
    monkeypatch.setattr(c.production, 'result_rows', lambda p: rows)
    monkeypatch.setattr(c.production, 'completed_task', lambda p, task: pointers.get(task))
    _, imported = c.import_parent(path)
    assert calls == [dict(check_source=True)] and len(imported['receipts']) == 10
    validate_file_ref(imported['receipts']['reduce_U000'])
    pointers['reduce_U000']['result']['teacher_report_sha256'] = 'bad'
    with pytest.raises(ValueError, match='another teacher'):
        c.import_parent(path)
    pointers['reduce_U000']['result']['teacher_report_sha256'] = pointers['train_U000']['result']['training_report_sha256']
    rows[-1]['state'] = 'UNFINISHED'
    with pytest.raises(ValueError, match='committed'):
        c.import_parent(path)
    rows[-1]['state'] = 'COMPLETE'
    bad = load_json(tmp_path/'U000.json')
    bad['acceptance_only'] = True
    import json
    (tmp_path/'U000.json').write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises(ValueError, match='scientific fit'):
        c.import_parent(path)


def test_live_submit_once_and_interrupted_claim(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hlt_classification.scouting import hcwdl_exact_dag_submission as dag
    spec, parent_spec = simple_spec(tmp_path)
    Path(spec['root']).mkdir()
    monkeypatch.setattr(c, 'validate_spec', lambda s: parent_spec)
    monkeypatch.setattr(c.context, 'check_submission_site', lambda p: None)
    plan = c.submit(spec, mode='gate')
    calls = []
    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(dag.subprocess, 'run', sbatch)
    kwargs = dict(mode='gate', execute=True, plan_hash=plan['content_hash'], authorization=c.AUTHORIZATION)
    first = c.submit(spec, **kwargs)
    assert c.submit(spec, **kwargs) == first
    assert len(calls) == 1
    # No automatic retry if a claim exists without a final ledger.
    ledger = Path(spec['root'])/'submission_gate/submission_ledger.json'
    ledger.unlink()
    with pytest.raises(PermissionError, match='interrupted'):
        c.submit(spec, **kwargs)
    assert len(calls) == 1


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(17, 11)

    def to(self, *args, **kwargs):
        return super().to('cpu')

    def forward(self, features, vectors, mask):
        if features.ndim == 4:
            features = features.mean(1)
        return self.fc(features.mean(-1))


def test_real_kernel_paired_cold_kd_and_identity_rejection(monkeypatch):
    torch.set_num_threads(1)
    train = i.PairedCache(make_cache('train', 'U050'), make_cache('train', 'U000'))
    val = i.PairedCache(make_cache('validation', 'U050'), make_cache('validation', 'U000'))
    q = np.full((len(train), 11), 1/11, np.float32)
    # Actual shared optimizer/loss/validation/selection with a tiny CPU model.
    report, state = runner.train_kernel(Tiny(), train, val, node=c.nodes()[0], device='cpu',
        teacher_probabilities=q, teacher_identities=train.identities, acceptance_passes=1)
    assert report['node']['context_coordinate'] == 'U000' and report['passes'] == 1
    assert report['recipe_sha256'] == c.recipe()['content_hash'] and state
    assert report['acceptance_only'] and not report['scientific_fit']
    with pytest.raises(ValueError, match='identity'):
        runner.train_kernel(Tiny(), train, val, node=c.nodes()[0], device='cpu',
            teacher_probabilities=q, teacher_identities=train.identities[::-1], acceptance_passes=1)


def test_full_new_dag_worker_publication_and_corruption(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    spec, parent = simple_spec(tmp_path)
    root = Path(spec['root'])
    root.mkdir()
    # A miniature spec with authentic bank bytes. Only remote admission/data
    # construction are replaced; optimizer, banks, checkpoints, DAG and receipts
    # below are real. Run 60 small CPU passes, the actual unchanged schedule.
    metrics = runner.evaluate_probabilities(np.arange(88)%11, np.full((88,11), 1/11, np.float32))
    spec['parent_import']['rows'] = [dict(node_id=n, state='COMPLETE', validation=metrics,
        passes=60, selected_pass=1, recovery_to_pure_offline={}) for n in ('M0HLT','OFFLINE','U000')]
    parent['campaign_root'] = str(tmp_path/'parent')
    from hlt_classification.jetclass2_delphes.banks import publish_bank
    train0 = make_cache('train', 'U000')
    publish_bank(Path(parent['campaign_root'])/'bank', foundation_sha256='f'*64,
        teacher_report_sha256='b'*64, teacher_node='U000', role='train', identities=train0.identities,
        probabilities=np.full((len(train0),11), 1/11, np.float32))
    spec['parent_import']['u000_bank'] = dict(train_bank='bank', teacher_report_sha256='b'*64)
    spec = with_content_hash(spec)
    write_json(root/'preflight/result.json', evidence(spec, parent))
    monkeypatch.setattr(c, 'validate_spec', lambda s: parent)
    monkeypatch.setattr(w, 'execution', lambda *a, **kw: None)
    def cached(p, node, role, **kw):
        first = make_cache(role, node['coordinate'])
        return first if node['context_coordinate'] is None else i.PairedCache(first, make_cache(role, node['context_coordinate']))
    monkeypatch.setattr(w, 'cache', cached)
    monkeypatch.setattr(w, 'new_model', lambda n: Tiny())
    monkeypatch.setattr(w, 'train_kernel', lambda *a, **k: runner.train_kernel(*a, **dict(k, device='cpu')))
    monkeypatch.setattr(w, 'predict', lambda *a, **k: runner.predict(*a, **dict(k, device='cpu')))
    with pytest.raises(ValueError, match='Uncommitted'):
        w.run(spec, 'train_FUSION_U100')
    assert not (root/'train_FUSION_U100').exists()
    for task in c.tasks():
        row = w.run(spec, task['task_id'])
        assert row['final_test_accessed'] is False
        assert w.completed(spec, task['task_id']) == row
    assert w.run(spec, 'complete')['result']['new_fits'] == 8
    rows = w.result_rows(spec)
    assert len(rows) == 11 and all(r['state'] == 'COMPLETE' for r in rows)
    first = w.completed(spec, 'train_FUSION_U050')
    checkpoint = root/first['result']['checkpoint']
    checkpoint.write_bytes(b'corrupt test fixture')
    with pytest.raises(ValueError, match='bytes differ'):
        w.completed(spec, 'train_FINAL_BRIDGE_D000')


def test_helper_static_no_mutation_and_correct_cli():
    root = Path(__file__).resolve().parents[1]
    shell = (root/'scripts/queue_jetclass2_context_fusion.sh').read_text()
    assert 'origin/main' in shell and '--execute' in shell
    assert 'jetclass2_delphes_common.sh' in shell and 'oscar_l40s' in shell
    assert 'scancel' not in shell and 'git reset' not in shell and 'rm -' not in shell
    assert 'context_fusion.py' in shell
