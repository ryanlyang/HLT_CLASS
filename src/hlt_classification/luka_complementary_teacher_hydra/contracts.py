"""Fixed ten-fit registration; no U100 target in this experiment."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS
from hlt_classification.jetclass2_delphes.batch128_30e_recipe import recipe as base_recipe

BASE = '54ff594df4fb7dddc56c0e99ffb13b9448a6a6c9'
AUTHORIZATION = 'AUTHORIZE_LUKA_COMPLEMENTARY_TEACHER_HYDRA_V1'
TEACHERS = ('T0', 'TE', 'TF')
FOLDS = ('F0', 'F1', 'F2')
STUDENTS = dict(S0=('T0',), SE=('T0', 'TE'), SF=('T0', 'TF'), SEF=('T0', 'TE', 'TF'))
FITS = ('T0', 'F0', 'F1', 'F2', 'S0', 'TE', 'TF', 'SE', 'SF', 'SEF')
DEPENDENCIES = {t: (() if t in ('T0', *FOLDS) else ('T0',) if t == 'TE'
                    else FOLDS if t == 'TF' else STUDENTS[t]) for t in FITS}
NAMES = dict(T0='ordinary OFFLINE', TE='explanation-diverse OFFLINE', TF='OOF error-focused OFFLINE',
             **{t: 'OOF reference fold '+t[1] for t in FOLDS},
             **{t: '+'.join(v)+' total embedding weight5' for t, v in STUDENTS.items()})


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_COMPLEMENTARY_TEACHER_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_COMPLEMENTARY_TEACHER_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()): raise PermissionError('Population flags differ')
    if parents is not None and value['parents'] != parents: raise ValueError('Artifact parents differ')
    return digest


def weights(task):
    if task not in STUDENTS: raise ValueError('Unknown student')
    return {t: 5./len(STUDENTS[task]) for t in STUDENTS[task]}


def recipe():
    return artifact('RECIPE', optimizer_schedule_source=base_recipe(), classification_weight=1., embedding_beta=1.,
        epochs=30, batch_size=128, teachers=list(TEACHERS),
        folds=list(FOLDS), students={k: list(v) for k,v in STUDENTS.items()}, order=list(FITS),
        predictor_seeds=dict(T0=2026110900, TE=2026110901, TF=2026110902), teacher_seed_offset=200000,
        student_weights={t: weights(t) for t in STUDENTS}, train_targets_only=True, u100_objective=False,
        target_width=128, normalization='TRAIN_FP64_population_std_floor1e-3',
        export='clean_native_preclassifier', selection='native_full_validation_original_ties',
        fold_seed=2026110903, fold_selection='last_epoch_no_heldout_access_during_fit',
        fold_assignment='class_stratified_sha256_identity_seed_round_robin',
        difficulty='1+2*(1-p_true)_divided_by_TRAIN_class_mean',
        explanation=dict(score='true_logit_minus_logsumexp_others', coefficient=1., threshold=.67,
            denominator_floor=1e-4, scale_floor=.001, microbatch=8, perturbation=.05,
            taps=['native17_before_input_bn','deterministic_pair4_before_input_bn'],
            scale='frozen_T0_sqrt_running_var_plus_bn_eps', precision='FP32_math_SDPA_eval',
            reduction='masked_energy_div_reference_masked_energy_mean_jets'),
        native_hlt_inference=True)


def term_names(task):
    if task not in FITS: raise ValueError('Unknown fit')
    return {'ce', *STUDENTS[task]} if task in STUDENTS else {'ce', 'explanation'} if task == 'TE' else {'ce'}
