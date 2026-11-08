"""Immutable thirty-fit scientific registry. No accuracy admission gates."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_embedding.contracts import COUNTS, recipe as u100_recipe
from hlt_classification.jetclass2_delphes.model import model_config

BASE = 'a6da7a02b773d2397ff0d301bb4b5b400fc7528c'
AUTHORIZATION = 'AUTHORIZE_LUKA_PRETEXT_STRUCTURE_HYDRA_V1'
TEACHERS = tuple(f'T{i:02}' for i in range(1, 9))
SINGLES = {f'E{i:02}': (f'T{i:02}',) for i in range(1, 9)}
COMBINATIONS = dict(M01=('T01', 'T05'), M02=('T05', 'T06', 'T07'),
                    M03=('T01', 'T04', 'T05', 'T06', 'T07'), M04=TEACHERS)
STRUCTURES = dict(S01=('C2',), S02=('C3',), S03=('C2', 'C3'), S04=('B',),
                  S05=('C2', 'C3', 'B'), S06=('C2', 'C3', 'B'),
                  S07=('C2', 'C3', 'B', 'R'), S08=('C2', 'C3', 'B', 'R'))
STUDENTS = (*SINGLES, *COMBINATIONS, *STRUCTURES, 'C_U', 'C_CE')
FITS = ('C_U', 'S05', 'T01', 'E01', 'T05', 'E05', 'S07', 'T07', 'E07', 'C_CE',
        'T06', 'E06', 'S06', 'T04', 'E04', 'T08', 'E08', 'S03', 'T03', 'E03',
        'T02', 'E02', 'S04', 'S01', 'S02', 'S08', 'M04', 'M03', 'M02', 'M01')


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_PRETEXT_STRUCTURE_HYDRA_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_PRETEXT_STRUCTURE_HYDRA_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Population disclosure differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Artifact lineage differs')
    return digest


def dependencies(task):
    if task not in FITS:
        raise ValueError('Unknown scientific task')
    return SINGLES.get(task, COMBINATIONS.get(task, ()))


def weights(task):
    if task not in STUDENTS:
        raise ValueError('Not a student')
    teachers = dependencies(task)
    extra = {t: 5./len(teachers) for t in teachers}
    if task == 'C_CE':
        extra['ce_teacher'] = 5.
    return dict(u100=5., **extra)


def term_names(task):
    return {'ce', *weights(task)} | ({'structure'} if task in STRUCTURES else set())


def cell(task):
    if task not in STUDENTS:
        raise ValueError('Not a student')
    return dict(width=128, objective=task, seed=0)


def ramp(position):
    if position < 0:
        raise ValueError('Negative fractional epoch')
    return min(1., position/3.)


def recipe():
    return artifact('RECIPE', parents=dict(u100=u100_recipe()['content_hash']), base=u100_recipe(),
        epochs=30, batch_size=128, model=model_config(), order=list(FITS),
        teachers=dict(zip(TEACHERS, ('random_removed_set', 'hard_patch_set', 'soft_removed_set',
            'patch_map', 'missing_pair_histogram', 'clean_correlations', 'clean_branches', 'denoising_set'))),
        combinations={k: list(v) for k, v in COMBINATIONS.items()},
        structures={k: list(v) for k, v in STRUCTURES.items()}, structure_source='S08_HLT_else_OFFLINE',
        structure_weights={k: .3 if k == 'S06' else 1. for k in STRUCTURES},
        weights={k: weights(k) for k in STUDENTS}, extra_ramp_epochs=3, clean_signature_weight=.25,
        teacher_seed_offset=200000, head_seed=2026110800, minimum_scale=.001,
        corruption_statistics_draws=2, validation_draws=2, teacher_class_labels_used=False,
        teacher_selection='lowest_full_validation_pretext_objective_earliest_tie',
        student_selection='native_full_validation_original_ties', structure_definition='physical_v1',
        decoder=dict(blocks=2, width=128, heads=4, feedforward=256, dropout=.1,
                     pooled_bottleneck_only=True, fixed_slots='maximum_train_validation_metadata'),
        set_loss=dict(assignment='detached_FP32_exact_Hungarian', target_average=[.5,.5],
                      presence_weight=.25, count_weight=.25),
        normalization='TRAIN_FP64_population_std_floor_0.001_constant_mask',
        teacher_export='clean_OFFLINE_TRAIN_pooled_128_FP32', deployable='native_mod_only_HLT',
        poor_accuracy_gate=False, automatic_retry=False)
