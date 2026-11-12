"""Exact crop frontend equivalence, including pathological physical inputs."""
import numpy as np
import pytest
import torch
from hlt_classification.luka_embedding_bridge_hydra import regions, regional_inputs, registry
from test_luka_embedding_bridge_hydra import physical, synthetic_dataset, small


def identical(a, b):
    x, mask = a; y, other = b
    assert mask.tobytes() == other.tobytes() and mask.shape == other.shape
    assert len(x) == len(y)
    for lhs, rhs in zip(x, y):
        assert set(lhs) == set(rhs)
        for key in lhs:
            assert lhs[key].shape == rhs[key].shape
            assert lhs[key].dtype == rhs[key].dtype
            assert lhs[key].tobytes() == rhs[key].tobytes(), key


@pytest.mark.parametrize('scale', registry.SCALES)
@pytest.mark.parametrize('seed', [17, 20261112])
def test_exact_random_long_empty_neutral_and_partial_tracking_crops(scale, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (0, 1, 2, 16, 50, 298, 513):
        a = physical(n)
        a[:, :4] *= rng.lognormal(0, 3, (n, 1))
        a[:, 6:8] *= rng.normal(0, 10, (n, 2))
        a[::3, 4:6] = [0, 1]; a[::3, 6:] = 0
        a[1::3, 7] = 0; a[1::3, 9] = 0
        a[1::3, 11] = 0; a[1::3, 13] = 0
        rows.append(a)
    sources = [a[::-1].copy() for a in rows]
    for labels in (np.arange(len(rows)) % 11, np.zeros(len(rows), np.int64)):
        identical(regions.crops(rows, sources, labels, scale), regional_inputs.crops(rows, sources, labels, scale))


@pytest.mark.parametrize('column,value', [(0,np.nan),(4,.5),(5,7),(6,1.),(8,-1.),(10,.5),(3,0.)])
def test_full_physical_validation_is_retained(column, value):
    a = physical(3)
    a[0, 4:6] = [0, 1]; a[0, 6:] = 0
    a[0, column] = value
    for frontend in (regions.crops, regional_inputs.crops):
        with pytest.raises(ValueError):frontend([a], [a], [0], 'fine')


@pytest.mark.parametrize('batch', range(6))
def test_same_dropout_gradients_adamw_buffers_and_rng(monkeypatch, batch):
    from hlt_classification.luka_embedding_bridge_hydra import banks, distributed as d
    from hlt_classification.luka_subclass_hydra.benchmark_training import parity_snapshot, compare
    from hlt_classification.luka_subclass_ddp_hydra.resources import PROBE_TOLERANCES
    small(monkeypatch)
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        ds = synthetic_dataset(); original = ds.crops
        target = banks.Targets(ds.spec, 'TP', ds); ctx = d.Context(0, 1, torch.device('cpu'))
        def batched(role, indices, source, scale):
            rows = [ds.rows[role][i] for i in indices]
            return regional_inputs.crops(rows, rows, ds.view(role,'D000').labels[indices], scale)
        snapshots = []; states = []
        for frontend in (original, batched):
            ds.crops = frontend
            m, w, opt = d.build(ds.spec, 'TP', ctx)
            values, _ = d.step(m,w,opt,'TP',ds,target,np.arange(11),3,batch,3.,ctx)
            snapshots.append(parity_snapshot(m,opt,values['total'])); states.append(torch.get_rng_state())
        result = compare(*snapshots, PROBE_TOLERANCES)
        assert result['passed'], result
        assert all(result[k] == 0 for k in PROBE_TOLERANCES)
        assert torch.equal(*states)
    finally:torch.set_num_threads(old_threads)


def test_frontend_witness_uses_real_prepared_views_and_rejects_changed_bytes(tmp_path,monkeypatch):
    from test_luka_embedding_bridge_hydra import test_original_physical_preparation_and_quarter_views
    from hlt_classification.luka_embedding_bridge_hydra import inputs,contracts
    test_original_physical_preparation_and_quarter_views(tmp_path,monkeypatch)
    ds=synthetic_dataset();s=ds.spec
    s.update(root=str(tmp_path),anchor=dict(evidence=dict(prepared='c'*64)),transfer_hash='d'*64,native_hash='e'*64,
             crop_execution_policy=contracts.crop_execution_policy())
    actual=inputs.Dataset(s)
    proof=regional_inputs.qualify(s,actual)
    assert len(proof['checks'])==12 and all(row['byte_exact'] for row in proof['checks'])
    original=regional_inputs.crops
    def changed(*args):
        batches,valid=original(*args)
        batches[0]['features'][0,0,0]+=np.float32(.001)
        return batches,valid
    monkeypatch.setattr(regional_inputs,'crops',changed)
    with pytest.raises(ValueError,match='bytes differ'):regional_inputs.qualify(s,actual)
