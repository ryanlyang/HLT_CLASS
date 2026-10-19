"""New FullStat scientific identities; never inherit trial holdout semantics."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat.recipe import recipe as base_recipe
from hlt_classification.luka_fullstat.contracts import COUNTS
from hlt_classification.luka_attention.contracts import SURFACE

FLAGS = dict(final_test_accessed=False, final_test_sealed=True,
             event_independence_verified=False, old_trial_overlap_verified=False,
             evaluation_scope='development_validation')
REFERENCES = ('M0HLT', 'OFFLINE', 'DIRECT_D000', 'COARSE_D033', 'COARSE_D000')
TEACHER = 'COARSE_D033'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_LAST_ATTENTION_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value,
        expected_contract=f'LUKA_FULLSTAT_LAST_ATTENTION_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat evaluation scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('FullStat attention parent hashes differ')
    return digest


def recipe():
    return artifact('RECIPE', base=base_recipe(128), surface=dict(SURFACE), epochs=20,
                    coefficients=dict(LOGIT=0., ATTN=.1), teacher=TEACHER,
                    coordinate='D000', initialization='fresh_coordinate_paired')


def nodes(parent_nodes):
    old = next(n for n in parent_nodes if n['node_id'] == 'COARSE_D000')
    if old['coordinate'] != 'D000' or old['context_coordinate'] is not None or not old['deployable']:
        raise ValueError('Expected regular single-HLT D000 parent')
    return [dict(old, node_id=arm+'_D000', teacher=TEACHER, branch=arm,
                 attention_coefficient=coefficient)
            for arm, coefficient in recipe()['coefficients'].items()]


def tasks():
    return [dict(task_id='export_D033', kind='export', dependencies=[]),
            dict(task_id='train_LOGIT_D000', kind='train', node_id='LOGIT_D000', dependencies=['export_D033']),
            dict(task_id='train_ATTN_D000', kind='train', node_id='ATTN_D000', dependencies=['export_D033']),
            dict(task_id='aggregate', kind='aggregate',
                 dependencies=['export_D033', 'train_LOGIT_D000', 'train_ATTN_D000'])]
