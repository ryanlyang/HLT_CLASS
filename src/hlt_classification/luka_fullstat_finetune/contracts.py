"""New identities: warm-start refinement is not an exact optimizer resume."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_attention.contracts import FLAGS, COUNTS
from hlt_classification.luka_fullstat_fine_tail.contracts import recipe as parent_recipe

TASKS = {'finetune': []}
AUTHORIZATION = 'AUTHORIZE_FULLSTAT_FINETUNE_V1'
LAUNCH_AUTHORIZATION = 'AUTHORIZE_FULLSTAT_FINETUNE_AUTOSUBMIT_V1'
EPOCHS = 10
CALIBRATION_EPOCHS = 3


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_FINETUNE_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_FINETUNE_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('FullStat sealed-role flags differ')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact parents differ')
    return value


def recipe():
    old = parent_recipe('DE000')
    return artifact('RECIPE', parents=dict(objective=old['content_hash']),
        objective=old, epochs=EPOCHS, batch_size=128, calibration_epochs=CALIBRATION_EPOCHS,
        calibration_lr=3e-4, calibration_weight_decay=0.,
        calibration='TRAIN_only_cached_frozen_eval_native_features_all_fourteen_heads',
        optimizer='fresh_AdamW_original_groups_betas_eps_weight_decay',
        initialization='selected_fine_tail_DE000_native_then_frozen_native_head_calibration',
        schedule='cosine_1.5e-5_to_1.5e-6_over_ten_additional_epochs_no_warmup',
        peak_lr=1.5e-5, floor_lr=1.5e-6, sampler='original_seed_absolute_epochs_21_to_30',
        all_native_trainable=True, exact_resume=False, native_inference_unchanged=True,
        selection='original_validation_key_including_unchanged_initializer_epoch0',
        save_every_completed_epoch=True, automatic_resume=False)


def learning_rate(position):
    if not math.isfinite(position) or not 0 <= position <= EPOCHS:
        raise ValueError('Continuation position must be in [0,10]')
    return 1.5e-6 + (1.5e-5-1.5e-6)*.5*(1+math.cos(math.pi*position/EPOCHS))
