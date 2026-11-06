"""Versioned Group A semantics and the complete, interleaved research registry."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import recipe as embedding_recipe, COUNTS

BASE = 'ddcdd5fa22ce2455b12a62d61a926caa07969159'
ORDER = tuple('P1 G1 J1 Q1 V1 D1 S1 C0 C1 P2 G2 J2 Q2 V2 D2 S2 P11 G8 J6 Q5 V6 D5 S5 P3 G5 J5 Q3 V7 D3 P5 G3 J3 Q6 P6 G4 J4 P7 P8 P9 P4'.split())
ARMS = {}
for name in ORDER:
    if name[0] not in 'PGJC':
        continue
    ARMS[name] = dict(family=name[0], source='OFFLINE', coefficient=.3, seed_index=0,
                      law='M', group_size=2, independent=False, dot=False)
for name in ('P2', 'G2', 'J2', 'C1'): ARMS[name]['seed_index'] = 1
for name in ('P3', 'G3', 'J3'): ARMS[name]['coefficient'] = .1
for name in ('P4', 'J4'): ARMS[name]['coefficient'] = 1.
for name, law in zip(('P5','P6','P7','P8','P9'), ('G','S','K','F','Hsemi')): ARMS[name]['law'] = law
for name in ('P11','G8','J6'): ARMS[name]['source'] = 'HLT'
ARMS['G4']['group_size'] = 3
ARMS['G5']['independent'] = True
ARMS['J5']['dot'] = True
for name in ('C0','C1'): ARMS[name]['coefficient'] = 0.
TASKS = tuple(n for n in ORDER if n in ARMS)
AUTHORIZATION = 'AUTHORIZE_NEXT_SUPERVISION_GROUP_A_V2'
FEATURES = ('log_pt_fraction','log_energy_fraction','deta','sin_dphi','cos_dphi','charge',
            'pid0','pid1','pid2','pid3','pid4','pid_unknown',
            'd0','dz','d0err','dzerr','d0sig','dzsig','valid_d0','valid_dz','valid_d0err','valid_dzerr')
GROUP_PROPOSALS=4096


def artifact(kind, *, parents=None, **fields):
    version=2 if kind in ('SPEC','RESOURCE_CHECK','ACCEPTANCE','DIRECT_RECONSTRUCTION_SPEC') else 1
    return with_content_hash(dict(contract=f'LUKA_NEXT_SUPERVISION_{kind}/v{version}', schema_version=version,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    version=2 if kind in ('SPEC','RESOURCE_CHECK','ACCEPTANCE','DIRECT_RECONSTRUCTION_SPEC') else 1
    validate_content_hash(value, expected_contract=f'LUKA_NEXT_SUPERVISION_{kind}/v{version}', expected_schema_version=version)
    if any(value.get(k) != v for k,v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Supervision scope or parent lineage differs')


def recipe():
    return artifact('RECIPE', embedding=embedding_recipe(), arms=ARMS, global_order=list(ORDER),
        implemented_group='A', deferred_groups=['B','C'], epochs=30, batch_size=128, embedding_weight=5.,
        seed_offset=1000003, auxiliary_seed=2026110507, sampler_seed=2026110508, ramp_epochs=3,
        particle_pool=500, positives=32, parent_cap=16, group_pool=128, group_positives=16,
        group_parent_cap=8, mining_pool=4096, mining_ranks=[257,1024],
        group_feature_proposals=GROUP_PROPOSALS,
        features=list(FEATURES), coordinate='own_parent_axis', length_unit='mm',
        normalization='TRAIN_population_mean_std_floor1e-3_continuous_only',
        pt_edges=[-8.,-6.,-4.,-2.,0.], radius_edges=[.02,.05,.1,.2,.4,.8],
        feature_expansion='exact_category_charge_validity_then_Linf_numeric_radius_then_discrete_Hamming',
        kinematic_bins=[20,10], quotas='equal_G_S_K_F_remainder_in_that_order',
        capacity='strict_no_replacement_no_anchor_drops_no_quota_relaxation',
        jet_candidates='original_OFFLINE_mix_1024_including_HLT_key_control',
        loss='FP32_per_jet_half_mean_positive_BCE_half_mean_negative_BCE_or_jet_CE',
        deployment='native_HLT_only_auxiliary_heads_removed', logit_kd=False)


def setting(task):
    if task.startswith('R'):
        from .reconstruction_contracts import setting as reconstruction
        return reconstruction(task)
    if task not in ARMS:
        from .structured_contracts import ARMS as structured
        if task in structured: return structured[task]
        from .probabilistic_contracts import ARMS as probabilistic
        if task in probabilistic: return probabilistic[task]
        raise NotImplementedError('Unknown or unselected supervision task')
    return ARMS[task]


def authorization(spec):
    if spec.get('scope')=='direct_reconstruction_only':
        from .reconstruction_contracts import AUTHORIZATION as reconstruction
        return reconstruction
    if spec.get('scope')=='group_C_only':
        from .probabilistic_contracts import AUTHORIZATION as probabilistic
        return probabilistic
    if spec.get('scope')=='group_B_only':
        from .structured_contracts import AUTHORIZATION as structured
        return structured
    return AUTHORIZATION


def selected_tasks(spec): return tuple(spec.get('tasks',TASKS))


def seed(imported, cfg, kind):
    return imported['node'][kind] + 1000003*cfg['seed_index']


def coefficient(cfg, position):
    if not math.isfinite(position) or position < 0: raise ValueError('Invalid schedule position')
    return cfg['coefficient']*min(position/3.,1.)
