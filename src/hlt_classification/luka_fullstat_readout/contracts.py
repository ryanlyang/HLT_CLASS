"""FullStat adaptation: fresh HLT student, one frozen U100 readout, 20 epochs."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_u100_embedding.contracts import recipe as original_recipe

TASKS = {'train_readout': []}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_READOUT_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_READOUT_AUTOSUBMIT_V1'
EPOCHS = 20


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_READOUT_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_READOUT_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat sealed-role flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return value


def recipe():
    base = original_recipe()
    return artifact('RECIPE', parents=dict(original=base['content_hash']),
        original=base, base=base['base'], epochs=EPOCHS, batch_size=128,
        ce_weight=1., embedding_weight=5., readout_ce_weight=1., ramp_passes=3.,
        readout='exact_frozen_U100_native_linear_on_destandardized_prediction_FP32',
        ramp='min(completed_updates/ceil(TRAIN_rows/128)/3,1)',
        logit_kd=False, warm_start=False, relational=False, attention=False,
        teacher='COARSE_U100', teacher_role='TRAIN_target_and_frozen_readout',
        inference='native_HLT_only_predictor_and_readout_removed',
        save_every_completed_epoch=True, automatic_resume=False)


def node(base):
    if (base['node_id'] != 'U100_EMBED_W5' or base['branch'] != 'u100_w5'
            or base['coordinate'] != 'D000' or base.get('teacher') is not None
            or base.get('context_coordinate') is not None or base['deployable'] is not True
            or base['auxiliary_weight'] != 5.):
        raise ValueError('Expected completed original HLT-only U100-w5 node')
    return dict(base, node_id='U100_READOUT_CE', branch='u100_readout_ce')
