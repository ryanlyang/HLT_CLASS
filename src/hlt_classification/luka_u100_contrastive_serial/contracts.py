"""New execution/scientific identity; historical fixed30 contracts stay unchanged."""
import math
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_contrastive.contracts import ARMS, recipe as original_recipe

AUTHORIZATION = 'AUTHORIZE_LUKA_CONTRASTIVE_SERIAL_DEBUG_V1'
ORDER = (
    'mix_l0p1_s0', 'hard_l0p1_s0', 'cross_l0p3_s0', 'mix_l0p3_s1', 'u100_w5_s1',
    'random_l0p1_s0', 'hard_l0p3_s0', 'cross_l0p1_s0', 'mix_l0p03_s0',
    'random_l1_s0', 'cross_l1_s0', 'hard_l1_s0', 'random_l3_s0', 'mix_l3_s0', 'mix_l10_s0',
    # Only attempted again if not successfully completed when the transition settles.
    'kinmix_l0p3_s0', 'mix_l0p3_s0', 'u100_w5_s0', 'random_l0p3_s0', 'mix_l1_s0',
)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_U100_CONTRASTIVE_SERIAL_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    result = validate_content_hash(value, expected_contract=f'LUKA_U100_CONTRASTIVE_SERIAL_{kind}/v1')
    if any(value.get(k) != v for k, v in FLAGS.items()) or (parents is not None and value['parents'] != parents):
        raise ValueError('Serial campaign flags or parent lineage differs')
    return result


def policy():
    return artifact('STOP_POLICY', metric='full_validation_macro_ovr_auc', min_epochs=15,
        max_epochs=30, patience=5, min_delta=1e-5, monitor_from_epoch=1,
        checkpoint_selection='original_selection_key_every_actual_improvement',
        schedule='unchanged_warmup3_cosine30', terminal_outcome='successful_early_stop')


def recipe():
    old = original_recipe()
    return artifact('RECIPE', parents=dict(original=old['content_hash'], stopping=policy()['content_hash']),
        original=old, stopping=policy(), all_twenty_arms_retained=True,
        zero_control='same_U100_updates_with_common_epoch_boundary_stopping')


def stopping(history, *, technical=False):
    """Replay the first stop, independently of exact best-checkpoint selection."""
    if not history:
        raise ValueError('Empty training history')
    p = policy(); reference = None; last = 0
    for epoch, row in enumerate(history, 1):
        auc = row['validation']['macro_ovr_auc']
        if row['pass_number'] != epoch or not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError('Invalid ordered validation history')
        if reference is None or auc >= reference + p['min_delta']:
            reference, last = auc, epoch
        reason = ('technical' if technical else 'patience' if epoch >= p['min_epochs']
                  and epoch-last >= p['patience'] else 'max_epochs' if epoch == p['max_epochs'] else None)
        if reason is not None:
            return dict(reason=reason, stop_epoch=epoch, last_meaningful_epoch=last,
                        monitor_best_auc=reference, stale_epochs=epoch-last)
    return None
