"""Source-bound all-eligible development campaign; never a legacy split override."""
from pathlib import Path
import subprocess

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from . import alljets_data as data, regular_campaign as regular

COUNTS = dict(train=299234, validation=74895)
SPLITS = '4b76251f98b1670c5378da96b0d13d25dd71323c8cd71f35df09076cf3d8e3a7'
AUTHORIZATION = 'AUTHORIZE LUKA ALLJETS EXPLORATORY EXACT PLAN'
JOB_PREFIX = 'lukaa'
WORKER = 'run_luka_fullsim_alljets.sh'
REQUIRED = [f'src/hlt_classification/luka_fullsim/alljets_{name}.py' for name in ('data', 'campaign', 'reporting')]
REQUIRED += ['scripts/luka_fullsim_alljets.py', 'scripts/queue_luka_fullsim_alljets.sh',
             'sbatch/'+WORKER, 'tests/test_luka_fullsim_alljets.py',
             'docs/plans/LUKA_FULLSIM_ALLJETS_PLAN.md', 'docs/contracts/LUKA_FULLSIM_ALLJETS.md',
             'docs/LUKA_FULLSIM_ALLJETS_RUNBOOK.md']
make_artifact, validate = data.artifact, data.validate
load_foundation, load_prepared, data_artifact = data.load_foundation, data.load_prepared, data.artifact
nodes, tasks, recipe, conventions = regular.nodes, regular.tasks, regular.recipe, regular.conventions
evidence, gate_ledger, gate_jobs = regular.evidence, regular.gate_ledger, regular.gate_jobs
create_science, validate_spec = regular.create_science, regular.validate_spec
prepared_root, preparation_gate = regular.prepared_root, regular.preparation_gate
preflight_accounting = regular.preflight_accounting


def is_gate(value):
    return value.get('contract') == data.PREFIX+'GATE/v1'


def is_recovery(value):
    return False


def source(project, commit):
    result = regular.source(project, commit)
    tracked = subprocess.run(['git', '-C', str(project), 'ls-files'], check=True, capture_output=True, text=True).stdout.splitlines()
    if not set(REQUIRED) <= set(tracked):
        raise ValueError('Commit the complete all-jets implementation before queueing')
    if Path(__file__).resolve() != Path(project).resolve()/'src/hlt_classification/luka_fullsim/alljets_campaign.py':
        raise ValueError('All-jets import escapes pinned project')
    return result


def check_population(inventory, splits):
    data.validate_splits(splits, inventory)
    if (inventory['content_hash'] != regular.INVENTORY
            or splits['content_hash'] != SPLITS
            or splits['parents']['original_splits'] != regular.SPLITS
            or splits['selected_counts'] != {**COUNTS, 'final_test': 0}
            or splits['comparison_rows'] != 50000):
        raise ValueError('Wrong all-jets population or original validation reference')


def registration():
    return {**regular.registration(), 'counts': dict(COUNTS),
            'class_balance': 'natural_all_eligible_file_role_proportions_no_rebalancing',
            'validation_policy': 'entire_74895_checkpoint_selection_original_50000_secondary_reporting_only',
            'historical_test_repurposed_for_training': True, 'independent_test_available': False,
            'evaluation_scope': 'exploratory_development_validation', 'comparison_rows': 50000,
            'preflight_cache_policy': 'one_coordinate_train_validation_at_a_time',
            'validation_predictions_saved': True}


def gate_policy():
    return {**regular.gate_policy(), 'counts': dict(COUNTS),
            'split_sha256': SPLITS, 'original_split_sha256': regular.SPLITS,
            'split_algorithm': 'original_validation_files_held_out_all_other_files_train_v1',
            'preflight_cache_policy': 'one_coordinate_train_validation_at_a_time',
            'storage_policy': 'authenticated_scalar_ragged_upper_bound_with_reserve'}


def create_gate(*, root, container, project, commit):
    pinned = source(project, commit)
    root, container, project = Path(root).resolve(), Path(container).resolve(strict=True), Path(project).resolve()
    for protected in (container, project):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise PermissionError('All-jets output overlaps protected source/raw input')
    gate = make_artifact('GATE', parents=dict(source=pinned['content_hash']), root=str(root),
        input_container=str(container), project_dir=str(project), source_commit=commit,
        source_snapshot=pinned, **gate_policy())
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def validate_gate(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash']))
    if (any(gate.get(k) != v for k, v in gate_policy().items())
            or source(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('All-jets gate source/policy differs')
    for protected in (gate['input_container'], gate['project_dir']):
        a, b = Path(gate['root']).resolve(), Path(protected).resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise PermissionError('All-jets output overlaps protected data/source')
    return gate
