"""Fixed first experiment: two oracles, twelve students, no quality gates."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from hlt_classification.luka_u100_pair_hydra.contracts import SITE, MANIFEST_HASH
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE = '54473f108c4b1b435f799ca2e41a9c69a4ebe332'
AUTHORIZATION = 'AUTHORIZE_FROZEN_ADAPTER_HYDRA_V1'
TEACHERS = ('teacher_bias', 'teacher_edge')
ARMS = {f'{site}_{mode}_e{weight}': dict(site=site, mode=mode, embedding=weight)
        for weight in (5, 0) for mode in ('both', 'aux', 'ce') for site in ('bias', 'edge')}
TASKS = (*TEACHERS, *ARMS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FROZEN_ADAPTER_HYDRA_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FROZEN_ADAPTER_HYDRA_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Frozen-adapter scope or parents differ')
    return value


def recipe():
    return artifact('RECIPE', base=base_recipe(), epochs=30, batch_size=128, block_index=6,
        token_hidden=32, pair_hidden=16, rank=4, edge_channels=4, query_chunk=32,
        bound=2., geometry_scale=10., rms_floor=.05, statistics_rows=4096,
        ce=1., correction=1., correction_ramp_epochs=3, teacher_lr=.0003,
        predictor_seed=2026110301, alignment_seed=2026110302, arms=ARMS,
        teacher_base='frozen_M0HLT_eval_parameters_and_buffers', teacher_view='U100',
        student_init='scratch_matched_M0HLT', correction_target='pre_alignment',
        edge_attention='native_probabilities_without_dropout_parallel_branch',
        validation_selection='full_validation_AUC', final_test_access=False)


def settings(task):
    if task in TEACHERS: return dict(site=task.removeprefix('teacher_'), mode='teacher', embedding=0)
    return ARMS[task]


def dependencies(task):
    if task == 'report': return list(TASKS)
    if task in ARMS and ARMS[task]['mode'] != 'ce': return ['teacher_'+ARMS[task]['site']]
    if task not in (*TASKS, 'setup'): raise ValueError('Unknown task')
    return []
