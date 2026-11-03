"""Review first; publish a fresh execution only after old allocations are terminal."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from .contracts import ARMS, ORDER, artifact, validate, recipe, policy
from . import source


def storage(root, budget=8*2**30):
    root = Path(root)
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    if used >= budget or shutil.disk_usage(root if root.exists() else root.parent).free < budget-used+2**30:
        raise OSError('Eight GiB allowance plus one GiB headroom unavailable; no automatic deletion')


def request(root, parent, project, commit, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    if root.resolve() != root or root.is_symlink() or type(quota_gib) is not int or quota_gib < 8:
        raise ValueError('Fresh unredirected root and at least eight personally available GiB required')
    snapshot = source.current(project, commit)
    proof = source.inspect_parent(parent); old = load_json(proof['spec']['path'])
    protected = [project, Path(old['root']), Path(old['project_dir']),
                 Path(old['anchor']['evidence']['prepared_root']), Path(old['anchor']['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Output overlaps protected source/data/results')
    from .submission import accounting, validate_completed
    # A fit may complete during the read-only replay. Finalization authenticates
    # every newly completed receipt before allowing reuse or submission.
    states = accounting(proof['jobs'] | proof['setup_jobs']); validate_completed(proof, states, require_all_receipts=False)
    value = artifact('REQUEST', parents=dict(source=snapshot['content_hash'], parent=proof['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        parent=proof, reviewed_states=states, order=list(ORDER), stopping_policy=policy(), recipe=recipe(),
        storage_budget_bytes=8*2**30, available_quota_gib=quota_gib,
        transition='cancel_pending_exact_parent_jobs_only_wait_running_then_reuse_complete',
        partial_outputs='preserve_original_restart_in_new_root_if_no_completed_receipt')
    storage(root); root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'request.json', value)
    return value


def check_request(value):
    validate(value, 'REQUEST', dict(source=value['source_snapshot']['content_hash'], parent=value['parent']['content_hash']))
    if (value['order'] != list(ORDER) or value['stopping_policy'] != policy() or value['recipe'] != recipe()
            or value['storage_budget_bytes'] != 8*2**30 or value['available_quota_gib'] < 8
            or load_json(Path(value['root'])/'request.json') != value
            or source.current(value['project_dir'], value['source_commit']) != value['source_snapshot']):
        raise ValueError('Request/source registration differs')
    return source.validate_parent(value['parent'])


def finalize(request, proof, states):
    check_request(request)
    from .submission import validate_completed, terminal
    old = source.validate_parent(proof); validate_completed(proof, states)
    if proof['spec'] != request['parent']['spec'] or proof['jobs'] != request['parent']['jobs']:
        raise ValueError('Parent job graph changed during transition')
    if not all(terminal(states.get(j)) for j in proof['jobs'].values()):
        raise ValueError('Original campaign still has live or unresolved allocations')
    if not request['parent']['completed'].keys() <= proof['completed'].keys():
        raise ValueError('Previously completed fit disappeared')
    if any(proof['completed'][a] != row for a, row in request['parent']['completed'].items()):
        raise ValueError('Previously completed fit changed')
    remaining = [a for a in ORDER if a not in proof['completed']]
    spec = artifact('SPEC', parents=dict(request=request['content_hash'], parent=proof['content_hash']),
        request=request, parent=proof, terminal_states=states, remaining=remaining, reused=list(proof['completed']),
        root=request['root'], project_dir=request['project_dir'], source_commit=request['source_commit'],
        source_snapshot=request['source_snapshot'], recipe=recipe(), stopping_policy=policy(),
        arms=ARMS, imported=old['imported'], anchor=old['anchor'], counts=old['counts'],
        original50k_rows=old['original50k_rows'], recovery_anchors=old['recovery_anchors'],
        execution_site=old['execution_site'], cpus=old['cpus'], memory_mb=old['memory_mb'],
        storage_budget_bytes=request['storage_budget_bytes'], max_concurrent_fits=1,
        preflight=False, new_setup=False, targets_reexported=False,
        resource_policy='reuse_original_full30_measured_upper_bounds_unchanged_batch_updates')
    write_immutable_json(Path(spec['root'])/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    validate(spec, 'SPEC', dict(request=spec['request']['content_hash'], parent=spec['parent']['content_hash']))
    old = check_request(spec['request']); source.validate_parent(spec['parent'])
    if (spec['parent']['spec'] != spec['request']['parent']['spec'] or spec['recipe'] != recipe()
            or spec['stopping_policy'] != policy() or spec['arms'] != ARMS
            or spec['remaining'] != [a for a in ORDER if a not in spec['parent']['completed']]
            or spec['reused'] != list(spec['parent']['completed']) or spec['max_concurrent_fits'] != 1
            or any(spec[k] is not False for k in ('preflight','new_setup','targets_reexported'))
            or spec['resource_policy'] != 'reuse_original_full30_measured_upper_bounds_unchanged_batch_updates'
            or any(spec[k] != old[k] for k in ('imported','anchor','counts','execution_site','cpus','memory_mb',
                                              'original50k_rows','recovery_anchors'))
            or any(spec[k] != spec['request'][k] for k in ('root','project_dir','source_commit','source_snapshot',
                                                         'storage_budget_bytes'))
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Serial campaign registration differs')
    from .submission import validate_completed, terminal
    validate_completed(spec['parent'], spec['terminal_states'])
    if not all(terminal(spec['terminal_states'].get(j)) for j in spec['parent']['jobs'].values()):
        raise ValueError('Unresolved old allocations in finalized spec')
    if check_storage:
        storage(spec['root'], spec['storage_budget_bytes'])
    return old
