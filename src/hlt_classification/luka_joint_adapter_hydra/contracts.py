"""Two jointly trained oracles, twelve embedding students, no quality gates."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from hlt_classification.luka_u100_pair_hydra.contracts import SITE, MANIFEST_HASH
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE = 'ceb2146149d2b1508a36b2cf91b274e0b58a26e6'
AUTHORIZATION = 'AUTHORIZE_JOINT_ADAPTER_HYDRA_V1'
TEACHERS = ('teacher_bias', 'teacher_edge')
ARMS = {f'{site}_{mode}_l{tag}': dict(site=site, mode=mode, embedding=5, correction_weight=weight)
        for tag, weight in (('0p3', .3), ('1', 1.))
        for mode in ('both', 'final_aux', 'aux') for site in ('bias', 'edge')}
# Injection and final supervision for bias first, then edge, then placement controls.
ARMS = {name: ARMS[name] for tag in ('0p3', '1') for name in (
    f'bias_both_l{tag}', f'bias_final_aux_l{tag}', f'edge_both_l{tag}',
    f'edge_final_aux_l{tag}', f'bias_aux_l{tag}', f'edge_aux_l{tag}')}
TASKS = (*TEACHERS, *ARMS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_JOINT_ADAPTER_HYDRA_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_JOINT_ADAPTER_HYDRA_{kind}/v1')
    if (value.get('schema_version') != 1 or any(value.get(k) != v for k, v in FLAGS.items())
            or (parents is not None and value['parents'] != parents)):
        raise ValueError('Joint-adapter scope or parents differ')
    return value


def recipe():
    return artifact('RECIPE', base=base_recipe(), epochs=30, batch_size=128, block_index=6,
        token_hidden=32, pair_hidden=16, rank=4, edge_channels=4, query_chunk=32,
        bound=2., geometry_scale=10., rms_floor=.05, statistics_rows=4096,
        ce=1., correction_weights=[.3, 1.], correction_ramp_epochs=3,
        predictor_seed=2026110301, alignment_seed=2026110302, arms=ARMS,
        teacher_base='scratch_joint_CE_all_backbone_parameters_and_buffers_train', teacher_view='U100',
        teacher_schedule='same_batch128_30epoch_schedule_as_students',
        final_aux_tap='eighth_particle_block_output_before_class_attention',
        student_init='scratch_matched_M0HLT', correction_target='pre_alignment',
        edge_attention='native_probabilities_without_dropout_parallel_branch',
        validation_selection='full_validation_AUC', final_test_access=False)


def settings(task):
    if task in TEACHERS: return dict(site=task.removeprefix('teacher_'), mode='teacher', embedding=0, correction_weight=0.)
    return ARMS[task]


def dependencies(task):
    if task == 'report': return list(TASKS)
    if task in ARMS and ARMS[task]['mode'] != 'ce': return ['teacher_'+ARMS[task]['site']]
    if task not in (*TASKS, 'setup'): raise ValueError('Unknown task')
    return []
