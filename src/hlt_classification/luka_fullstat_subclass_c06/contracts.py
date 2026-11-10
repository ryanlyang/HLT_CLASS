"""Contrastive0.6 ablation; the completed subclass0.5 experiment stays immutable."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_subclass import contracts as previous

TASK = previous.TASK
TASKS = {TASK: []}
COUNTS = previous.COUNTS
TEACHERS = previous.TEACHERS
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SUBCLASS_C06_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_SUBCLASS_C06_AUTOSUBMIT_V1'


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_SUBCLASS_C06_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **previous.FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_SUBCLASS_C06_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in previous.FLAGS.items()):
        raise PermissionError('Scientific scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Parent hashes differ')
    return value


def recipe():
    old = previous.recipe()
    fields = {k:v for k,v in old.items() if k not in
              ('contract','schema_version','parents','content_hash',*previous.FLAGS)}
    fields.update(matching_coefficient=.6, historical_control='completed_subclass_c03',
                  subclass_targets='byte_identical_completed_subclass_c03')
    return artifact('RECIPE', parents=dict(previous=old['content_hash']), **fields)


def node(parent):
    old = parent['nodes'][TASK]
    if (parent['recipe'] != previous.recipe() or old['coordinate'] != 'D000'
            or old['context_coordinate'] is not None or old['embedding_targets'] != list(TEACHERS)
            or old['total_embedding_weight'] != 10. or old['matching_coefficient'] != .3):
        raise ValueError('Expected completed contrastive0.3/subclass0.5 HLT node')
    return dict(old, branch='FULLSTAT_SUBCLASS_C06', matching_coefficient=.6)


def coefficient(update, steps):
    return previous.ramp(.6, update, steps)


def subclass_coefficient(update, steps):
    # Preserve the old floating-point arithmetic exactly, independently of the
    # new contrastive coefficient. No matching coefficient is an input here.
    return .5 * (previous.coefficient(update, steps) / .3)
