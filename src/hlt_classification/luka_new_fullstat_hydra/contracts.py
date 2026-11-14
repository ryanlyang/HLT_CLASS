"""Separate scientific identity; old FullStat artifacts remain unchanged."""
from pathlib import Path
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash

SOURCES = {
    '/eos/user/l/llambrec/jetclass/output_20261028_fullsim_fullstat_qcdrun2com/': 'qcd',
    '/eos/user/l/llambrec/jetclass/output_20261111_fullsim_fullstat_higgs2p_sb/': 'nonqcd',
}
MANIFEST_SHA = '80f5bc70db305272632ebd22537a2bd3d286a6e552bde620d2f796ee70e21097'
COUNTS = dict(train=3000000, validation=500000, final_test=1500000, unused=735950)
ROLES = tuple(COUNTS)
SCALARS = ('jet_label', 'hlt_matched', 'jet_nparticles', 'hlt_jet_nparticles', 'jet_pt')
SEED, BLOCK = 20261016, 2048
FLAGS = dict(final_test_accessed=False, event_independence_verified=False,
             old_dataset_overlap_verified=False, evaluation_scope='development_validation')
GPUS = {'l40s': ('nvidia_l40s', 'NVIDIA L40S'), 'l40': ('nvidia_l40', 'NVIDIA L40'),
        'a6000': ('nvidia_rtx_a6000', 'NVIDIA RTX A6000')}


def artifact(kind, **fields):
    return with_content_hash(dict(contract=f'LUKA_NEW_FULLSTAT_HYDRA_{kind}/v1',
                                  schema_version=1, **FLAGS, **fields))


def validate(value, kind):
    result = validate_content_hash(value, expected_contract=f'LUKA_NEW_FULLSTAT_HYDRA_{kind}/v1',
                                   expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise ValueError('Dataset scope/independence flags differ')
    return result


def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Invalid relative path')
    parts = relative.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ValueError('Unsafe relative path')
    root = Path(root).resolve()
    for i in range(1, len(parts)+1):
        if root.joinpath(*parts[:i]).is_symlink():
            raise ValueError('Symlink alias forbidden')
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Escaping path')
    return path


def policy():
    return dict(seed=SEED, block_rows=BLOCK,
        algorithm='shared_block_hash_order_stratum_exact_quotas_v1_new_dataset_domain',
        strata='signal_class_or_QCD_source_bin', additional_pt_cut=None,
        quarantine_raw_labels=[206], recognized_labels='unchanged_old_eleven_class_map',
        producer_label_confirmation=False, natural_sampling=True)


def recipe():
    from hlt_classification.luka_fullstat.recipe import recipe as original
    return artifact('RECIPE', base=original(128), loss='CE_only', ce_weight=1.,
                    epochs=20, batch_size=128, sampling='natural_without_replacement',
                    epoch_resume='full_model_optimizer_rng_completion_last')


def node():
    from hlt_classification.luka_fullsim.regular_campaign import nodes
    return next(n for n in nodes() if n['node_id']=='OFFLINE')
