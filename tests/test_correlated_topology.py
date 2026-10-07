"""Stronger tracking plus count loss: no real jobs or final-test inference."""
import copy
from pathlib import Path
import numpy as np
import pytest

from test_literature_proxy import particles
from test_correlated_tracking_ladder import parent, final_pilot, study, dataset, released, fake_source
from test_correlated_tracking_ladder import profile_for as old_profile_for
from test_literature_proxy_v3_campaign import parent_v2, pilot
from hlt_classification.correlated_topology import kernel as k, views as v, dataset as d
from hlt_classification.correlated_tracking import kernel as tracking
from hlt_classification.literature_proxy_v2.kernel import calibration
from hlt_classification.cms_proxy_ladder import correlated_topology as c, correlated as old
from hlt_classification.cms_proxy_ladder import release as r, data, cache, gate as g, production as p, submission as s
from hlt_classification.cms_proxy_ladder.contracts import write_json, artifact, file_ref
from hlt_classification.data.cache_contracts import with_content_hash, load_json
from hlt_classification.cms_proxy_ladder.campaign import task_graph


def rates():
    return calibration('a'*64, jets=20000, particles=818458, soft=202534, dropped=40098, pairs=74625)


def frozen(**overrides):
    return with_content_hash(dict(k.recipe(rates()), **overrides))


def test_tracking_amplitude_is_high_without_other_smearing_or_pid_changes():
    src = particles((0, 3, 4), [5., 5., 5.])
    sides, _ = tracking.generate_all(src, 'amplitude')
    out = k.generate(src, 'amplitude', frozen(), historical_mid=sides['CORR_MID']).particles
    assert k.equal(out, sides['CORR_HIGH'])
    np.testing.assert_allclose(out.tracking[:, :2]-src.tracking[:, :2],
                               2*(sides['CORR_MID'].tracking[:, :2]-src.tracking[:, :2]))
    np.testing.assert_allclose(out.tracking[:, 2:]**2-src.tracking[:, 2:]**2,
                               4*(sides['CORR_MID'].tracking[:, 2:]**2-src.tracking[:, 2:]**2))
    for name in ('p4', 'charge', 'category', 'valid'):
        np.testing.assert_array_equal(getattr(src, name), getattr(out, name))
    with pytest.raises(ValueError, match='Historical'):
        k.generate(src, 'amplitude', frozen(), historical_mid=src)


def test_eligible_drops_and_disjoint_merges_conserve_p4():
    src = particles((0, 1, 2, 3, 4, 5), [.5]*6)
    out = k.generate(src, 'drops', frozen(p_drop=1., p_merge=0.))
    assert out.counts['dropped_particles'] == 3
    assert set(out.particles.category) == {3, 4, 5}
    np.testing.assert_array_equal(out.mapping, [3, 4, 5])
    src = particles((1,)*6+(2, 2), [10.]*8)
    out = k.generate(src, 'merges', frozen(p_drop=0., p_merge=1.))
    assert out.counts['merged_pairs'] == 4 and len(out.particles) == 4
    np.testing.assert_allclose(src.p4.sum(0), out.particles.p4.sum(0))
    assert not out.particles.tracking.any() and not out.particles.valid.any()
    assert len(set(out.mapping)) == len(out.mapping)
    assert out.counts['input_particles'] - out.counts['dropped_particles'] - out.counts['merged_pairs'] == len(out.particles)
    out = k.generate(particles((1,)*4, [.5]*4), 'empty', frozen(p_drop=1.))
    assert len(out.particles) == 0 and out.counts['empty_output'] == 1


def test_historical_portability_is_bounded_and_never_weakens_new_exact_replay():
    from hlt_classification.cms2jc2_response.bridge import Particles
    src = particles((0,), [5.])
    sides, _ = tracking.generate_all(src, 'portable')
    saved = sides['CORR_MID']
    tr = saved.tracking.copy(); tr[0, 0] += 1e-14
    rounded = Particles(saved.p4, saved.charge, saved.category, tr, saved.valid, saved.keys)
    response = k.generate(src, 'portable', frozen(), historical_mid=rounded)
    assert response.historical_replay['bitwise_exact_jets'] == 0
    assert 0 < response.historical_replay['max_tolerance_fraction'] <= 1
    assert k.equal(response.particles, k.generate(src, 'portable', frozen()).particles)
    tr = saved.tracking.copy(); tr[0, 0] += 1e-5
    changed = Particles(saved.p4, saved.charge, saved.category, tr, saved.valid, saved.keys)
    with pytest.raises(ValueError, match='roundoff'):
        k.generate(src, 'portable', frozen(), historical_mid=changed)


def test_permutation_determinism_and_fixed_topology_interpolation():
    src = particles((0, 1, 1, 3, 4), [3., 4., 4., 5., 6.])
    rate = frozen(p_merge=1.)
    out = k.generate(src, 'views', rate)
    perm = src.take([4, 2, 1, 0, 3])
    reordered = k.generate(perm, 'views', rate)
    assert k.equal(out.particles, reordered.particles) and out.particles.keys == reordered.particles.keys
    mapping = v.match_particles(out.particles, src, identity='views', frozen=rate)
    assert np.array_equal(mapping, out.mapping)
    kwargs = dict(identity='views', proxy=out.particles, offline=src, mapping=mapping)
    assert v.build_view(**kwargs, coordinate='D000') is out.particles
    assert v.build_view(**kwargs, coordinate='OFFLINE') is src
    for coord, f in (('D066', 1/3), ('D033', 2/3), ('D050', .5)):
        pview = v.build_view(**kwargs, coordinate=coord)
        for name in ('p4', 'category', 'charge', 'valid'):
            np.testing.assert_array_equal(getattr(pview, name), getattr(out.particles, name))
        eligible = out.particles.valid[:, :2] & out.particles.valid[:, 2:]
        clean = src.tracking[mapping]
        np.testing.assert_allclose((pview.tracking[:, :2]-clean[:, :2])[eligible],
                                   f*(out.particles.tracking[:, :2]-clean[:, :2])[eligible])
        assert v.build_inputs(pview).features.shape[1] == 17
    assert not v.view_contract()['source_indices_model_visible']
    with pytest.raises(ValueError):
        v.build_view(**dict(kwargs, mapping=np.full(len(mapping), -1, dtype=np.int32)), coordinate='D050')


@pytest.fixture
def topology_request(released, tmp_path, monkeypatch):
    release, root = released
    monkeypatch.setattr(d, 'COUNTS', dict(old.COUNTS))
    # Metadata fake only: independent test below authenticates a genuine pilot.
    noise = tmp_path/'noise-pilot'/'study_spec.json'
    noise.parent.mkdir()
    noise.write_text('{}')
    monkeypatch.setattr(d, 'noise_calibration', lambda *a, **kw: rates())
    return d.request(root/'release.json', noise)


@pytest.fixture
def topology(topology_request, tmp_path):
    root = tmp_path/'topology'
    return d.build_release(topology_request, output_root=root, workers=2), root


def test_materialized_release_replay_population_and_train_diagnostics(topology, topology_request):
    request = topology_request
    release, root = topology
    assert r.validate_release(release, root=root)
    assert release['schema_version'] == 7 and release['exact_process_replay']
    original = load_json(request['original_release']['path'])
    assert release['identity_sha256'] == original['identity_sha256']
    for role in ('train', 'validation'):
        rows = list(data.iter_paired(release, release_root=root, role=role))
        assert len(rows) == d.COUNTS[role]
        for row in rows:
            assert k.equal(row.proxy, k.generate(row.offline, row.identity, request['recipe']).particles)
    report = load_json(root/'diagnostics.json')
    assert report['jets'] == d.COUNTS['train'] and report['role'] == 'train'
    assert report['statistics']['CORR_HIGH_TOPO/jet/multiplicity']['count'] == d.COUNTS['train']
    with pytest.raises(PermissionError):
        list(d.iter_paired({}, release_root='does-not-exist', role='final_test'))
    with pytest.raises(FileExistsError):
        d.build_release(request, output_root=root, workers=2)


def test_foundation_caches_and_six_fit_plan(topology, tmp_path):
    release, root = topology
    froot = tmp_path/'foundation'
    f = data.build_foundation(release, release_root=root, output_root=froot, workers=1, capacity=512)
    assert f['schema_version'] == 7 and f['inputs'] == v.input_contract()
    plan = c.scientific_plan(f, foundation_root=froot)
    assert plan['schema_version'] == 9 and plan['fresh_fit_count'] == 6
    assert len(task_graph(plan)) == 11 and plan['probability_publication_count'] == 3
    assert set(plan['branches']) == {'DIRECT', 'COARSE'}
    assert not plan['ascending_views_are_duplicates']
    for coordinate in ('OFFLINE', 'D000', 'D066', 'D033'):
        batch = cache.prepare_cache(f, foundation_root=froot, role='train', coordinate=coordinate,
                                    workers=1, max_ram_bytes=2**30)
        rows = list(data.iter_paired(release, release_root=root, role='train'))
        assert [bytes(i).hex() for i in batch.identities] == [row.identity for row in rows]
        if coordinate in ('OFFLINE', 'D000'):
            expected = np.concatenate([v.build_inputs(row.offline if coordinate == 'OFFLINE' else row.proxy).features for row in rows])
            np.testing.assert_array_equal(np.concatenate([b.features for b in batch.blocks]), expected)
    # Real CPU CE -> teacher probabilities -> KD; tiny network, production kernel.
    import torch
    from test_jetclass2_delphes_training import TinyModel
    from hlt_classification.jetclass2_delphes.runner import train_kernel, predict
    torch.set_num_threads(1)
    teacher = TinyModel()
    def cached(role, coordinate):
        return cache.prepare_cache(f, foundation_root=froot, role=role, coordinate=coordinate,
                                   workers=1, max_ram_bytes=2**30)
    train, val = cached('train', 'OFFLINE'), cached('validation', 'OFFLINE')
    node = next(n for n in plan['nodes'] if n['node_id'] == 'OFFLINE')
    report, _ = train_kernel(teacher, train, val, node=node, device='cpu', acceptance_passes=1)
    probabilities = predict(teacher, train, device='cpu', temperature=2.)
    for branch in ('DIRECT', 'COARSE'):
        node = next(n for n in plan['nodes'] if n['branch'] == branch)
        student = cached('train', node['coordinate'])
        report, _ = train_kernel(TinyModel(), student, cached('validation', node['coordinate']),
            node=node, device='cpu', acceptance_passes=1, teacher_probabilities=probabilities,
            teacher_identities=train.identities)
        assert report['acceptance_only'] and np.isfinite(report['validation']['accuracy'])


def test_mutation_rejected_and_original_preserved(topology, topology_request):
    request = topology_request
    release, root = topology
    original = Path(request['original_release']['path']).read_bytes()
    bad = with_content_hash(dict(release, exact_process_replay=False))
    with pytest.raises(ValueError):
        r.validate_release(bad, root=root)
    block = root/release['proxy_blocks'][0]['path']
    block.write_bytes(b'corruption')
    with pytest.raises(ValueError):
        list(data.iter_paired(release, release_root=root, role='train'))
    assert Path(request['original_release']['path']).read_bytes() == original


def profile_for(gate, foundation):
    # Reuse the registered common profile fields, replace only schema + node.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(old, 'scientific_plan', lambda f: c_plan(f))
        profile = old_profile_for(gate, foundation)
    return with_content_hash(dict(profile, schema_version=12,
        contract='JETCLASS2_CMS_PROXY_LADDER_RUNTIME_PROFILE/v12'))


_old_plan = old.scientific_plan
def c_plan(f):
    return _old_plan(f, _version=9, _foundation_version=7, _prefix='CORRHT')


@pytest.fixture
def gated(topology_request, tmp_path, monkeypatch):
    request = topology_request
    monkeypatch.setattr(c, 'source_lock', fake_source)
    project = tmp_path/'code'; project.mkdir()
    gate = c.create_gate(original_release=request['original_release']['path'], noise_spec=request['noise_spec']['path'],
        gate_root=tmp_path/'gate', project_dir=project, source_commit='e'*40,
        available_quota_gib=10, persistent=True)
    g.run_gate_task(gate, 'authenticate_release')
    original = g.build_foundation
    monkeypatch.setattr(g, 'build_foundation', lambda *a, **kw: original(*a, **dict(kw, workers=1)))
    g.run_gate_task(gate, 'build_foundation')
    monkeypatch.setattr(g, '_measure_preflight', lambda spec, foundation, **kw: profile_for(spec, foundation))
    g.run_gate_task(gate, 'preflight')
    return gate


def test_gate_measured_profile_and_exact_dry_submission(gated, tmp_path, monkeypatch):
    campaign = c.create_campaign(gate_root=gated['gate_root'], campaign_root=tmp_path/'science')
    assert p.validate_campaign(campaign, check_source=True)
    assert len(s.gate_plan(gated)['commands']) == 3
    plan = s.science_plan(campaign)
    assert len(plan['commands']) == 11
    for row in plan['commands']:
        assert '--partition=debug' in row['command'] and '--account=reu-aisocial' in row['command']
        assert not any('DENSE' in a for a in row['command'])
    s.submit(subject=gated, mode='gate', execute=False, authorization_phrase=None)
    s.submit(subject=campaign, mode='science', execute=False, authorization_phrase=None)
    assert not (tmp_path/'science/submission_ledger.json').exists()
    with pytest.raises(PermissionError):
        s.submit(subject=campaign, mode='science', execute=True, authorization_phrase='wrong')
    bad = copy.deepcopy(campaign['runtime_profile'])
    bad['passed'] = False
    with pytest.raises(ValueError):
        c.validate_profile(with_content_hash(bad), foundation=campaign['foundation'], spec=gated)
    # Execute the real preflight coordinator, substituting hardware/model probes.
    import test_correlated_tracking_ladder as inherited
    monkeypatch.setattr(inherited, 'profile_for', profile_for)
    inherited.test_preflight_exercises_installed_parity_ce_kd_reducer_and_sporc(gated, monkeypatch)


def test_real_noise_pilot_authentication_and_no_recalibration(pilot, monkeypatch):
    from hlt_classification.literature_proxy_v3 import worker
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    worker.run(pilot, workers=2)
    path = Path(pilot['root'])/'study_spec.json'
    assert d.noise_calibration(path, full=True) == pilot['calibration']
    assert k.recipe(d.noise_calibration(path))['p_drop'] == pilot['calibration']['p_drop']
    report = Path(pilot['root'])/'report.json'
    report.write_bytes(report.read_bytes()+b' ')
    with pytest.raises(ValueError):
        d.noise_calibration(path)


def test_missing_tracking_untouched_and_semantic_recipe_rejected():
    from hlt_classification.cms2jc2_response.bridge import Particles
    src = particles((0, 3, 4), [5., 5., 5.])
    tr, valid = src.tracking.copy(), src.valid.copy()
    tr[0, 2:] = 0.; valid[0, 2:] = False
    src = Particles(src.p4, src.charge, src.category, tr, valid, src.keys)
    out = k.generate(src, 'missing', frozen())
    np.testing.assert_array_equal(out.particles.tracking[0], src.tracking[0])
    view = v.build_view(identity='missing', proxy=out.particles, offline=src,
                        coordinate='D033', mapping=out.mapping)
    np.testing.assert_array_equal(view.tracking[0], src.tracking[0])
    for key, value in (('pid_changes', True), ('redraw_per_epoch', True), ('merge', 'all_charges'), ('p_drop', 1.1)):
        with pytest.raises(ValueError):
            k.generate(src, 'bad', frozen(**{key: value}))


def test_gate_storage_freshness_and_source_fail_closed(topology_request, tmp_path, monkeypatch):
    project = tmp_path/'code'; project.mkdir()
    monkeypatch.setattr(c, 'source_lock', fake_source)
    args = dict(original_release=topology_request['original_release']['path'],
        noise_spec=topology_request['noise_spec']['path'], gate_root=tmp_path/'gate',
        project_dir=project, source_commit='e'*40, available_quota_gib=10, persistent=True)
    with pytest.raises(PermissionError):
        c.create_gate(**dict(args, available_quota_gib=9))
    with pytest.raises(PermissionError):
        c.create_gate(**dict(args, gate_root=Path(topology_request['offline_root'])/'bad'))
    gate = c.create_gate(**args)
    with pytest.raises(FileNotFoundError):
        c.create_campaign(gate_root=gate['gate_root'], campaign_root=tmp_path/'premature')
    assert not (tmp_path/'premature').exists()
    with pytest.raises(FileExistsError):
        c.create_gate(**args)
    monkeypatch.setattr(c, 'source_lock', lambda *a: {})
    with pytest.raises(ValueError, match='source drift'):
        c.validate_gate(gate, check_source=True)
