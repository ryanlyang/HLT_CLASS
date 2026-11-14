"""New population/evaluation identity; reuse the frozen Oscar optimization recipe."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullstat_balanced_ladder import contracts as donor
from hlt_classification.luka_new_fullstat_hydra.contracts import GPUS

PREFIX = 'LUKA_NEW_BALANCED_HYDRA'
FLAGS = dict(donor.FLAGS)
STAGES, COORDINATES, TEACHERS = donor.STAGES, donor.COORDINATES, donor.TEACHERS
CLASS_NAMES, EDGES = donor.CLASS_NAMES, donor.EDGES
ELIGIBLE, DRAWS, EPOCHS, BATCH, STEPS = donor.ELIGIBLE, donor.DRAWS, donor.EPOCHS, donor.BATCH, donor.STEPS
node, coefficient = donor.node, donor.coefficient
COUNTS = dict(train=2425536, validation=500000, final_test=1200000)
ADDITIONS = dict(train=279164, validation=136976, final_test=113361)
OUTSIDE = dict(train=853628, validation=136976, final_test=413361, unused=205760)
PARENT_HASH = '2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02'
POPULATION_HASH = '7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939'
EXTENSION_SEED = 2026111305
AUTH = 'AUTHORIZE_NEW_BALANCED_HYDRA_SIX_V1'
RESULT_GIB = 14
BASELINE_RESERVE_GIB = 17
BLOCK = 2048


def artifact(kind, *, parents=None, **fields):
    if set(fields).intersection(('contract', 'schema_version', 'parents', *FLAGS)):
        raise ValueError('Reserved artifact fields')
    return with_content_hash(dict(contract=f'{PREFIX}_{kind}/v1', schema_version=1,
                                  parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'{PREFIX}_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k,v in FLAGS.items()):
        raise ValueError('Scientific scope differs')
    if parents is not None and value.get('parents') != parents:
        raise ValueError('Parent bindings differ')
    return value


def population_policy():
    return dict(parent_role_preservation=True, unused_only_extension=True, seed=EXTENSION_SEED,
        order='SHA256_block_order_source_stratum_class_pt_bin_v1', block_rows=BLOCK,
        quota='sequential_largest_remainder_proportional_unused_capacity',
        additions=ADDITIONS, counts=COUNTS, pt_edges_GeV=list(EDGES), boundary='[200,2000)',
        excluded_label=206, final_test_sealed=True, in_range_unused_remaining=689)


def recipe():
    original=donor.recipe()
    fields={k:v for k,v in original.items() if k not in ('content_hash','contract','schema_version','parents',*FLAGS)}
    fields.update(original_counts=dict(train=3000000,validation=500000,final_test=1500000,unused=735950),
        population_policy=population_policy(), selection_population='new_in_range_500000_validation',
        outside_validation='original_outside_validation_diagnostic_selected_checkpoint_only',
        native_and_auxiliary_seed_domain='unchanged_Oscar_balanced_v1',
        final_test_primary_rows=1200000, final_test_outside_rows=413361)
    return artifact('RECIPE', **fields)


def tasks():
    return donor.tasks()
