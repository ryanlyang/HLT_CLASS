"""FullStat identity, paths and immutable artifact namespace."""
from pathlib import Path
import re

from hlt_classification.data.cache_contracts import canonical_sha256, validate_content_hash, with_content_hash
from hlt_classification.jetclass2_delphes.contracts import relative_file

SOURCE = 'ryang@lxplus.cern.ch:/eos/user/l/llambrec/jetclass/output_20261015_fullsim_fullstat/merged'
COUNTS = dict(train=3000000, validation=500000, final_test=1500000, unused=105325)
ROLES = tuple(COUNTS)
ORDINARY = ('train', 'validation')
SEED = 20261016
BLOCK_ROWS = 2048
COORDINATES = ('OFFLINE', 'U000', 'U050', 'U100', 'D066', 'D033', 'D000')


def artifact(kind, **fields):
    return with_content_hash(dict(contract=f'LUKA_FULLSTAT_{kind}/v1', schema_version=1, **fields))


def validate(value, kind, parents=None):
    digest = validate_content_hash(value, expected_contract=f'LUKA_FULLSTAT_{kind}/v1', expected_schema_version=1)
    if parents is not None and value.get('parents') != parents:
        raise ValueError('FullStat artifact parents differ')
    return digest


def source_path(raw, relative):
    path = relative_file(Path(raw), relative)
    parts = relative.split('/')
    if (len(parts) != 4 or parts[2] != 'merged'
            or re.fullmatch(r'ntuple_merged_[0-9]+\.root', parts[3]) is None):
        raise ValueError('Unexpected FullStat ROOT path')
    signal = parts[:2] == ['jetclass2', 'train_higgs2p']
    qcd = parts[0] == 'jetclass2sb' and re.fullmatch(r'QCD_PT-[A-Za-z0-9_-]+', parts[1])
    if not (signal or qcd):
        raise ValueError('Unexpected FullStat source category')
    if any((Path(raw)/Path(*parts[:i])).is_symlink() for i in range(1, 5)):
        raise ValueError('Symlink source alias forbidden')
    return path, 'signal' if signal else parts[1]


def identity(record, entry):
    return canonical_sha256(dict(domain='LUKA_FULLSTAT_ROW/v1', path=record['path'],
        sha256=record['sha256'], tree=record['tree_key'], entry=int(entry)))


def split_policy():
    return dict(seed=SEED, block_rows=BLOCK_ROWS,
        algorithm='shared_block_hash_order_stratum_exact_quotas_v1',
        strata='signal_class_or_QCD_source_bin', boundary_blocks_may_split=True,
        event_independence_verified=False, old_trial_overlap_verified=False,
        additional_pt_cut=None, final_test_accessed=False)
