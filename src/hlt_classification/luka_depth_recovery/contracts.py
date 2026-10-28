"""Two execution lanes, distinct from actual scientific dependencies."""
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_supervision_depth.contracts import ARMS, TEACHER_TASKS, recipe

AUTHORIZATION = 'AUTHORIZE_DEPTH_PENDING_ONLY_TWO_LANES_DEBUG_V1'
# Round-robin submission gives CE diversity and its capacity control early places.
# All previous same-lane jobs are included: premature cancellation of a dependent
# job cannot accidentally open a third slot while an earlier job is still running.
LANES = (
    ('teacher_ce', 'ce_diverse', 'set8_p3', 'set8_p10', 'set48_p10',
     'aligned8_p10', 'set248_p10', 'baseline_seed1', 'set8_p1', 'set8_p30'),
    ('duplicate_offline', 'teacher_supcon', 'supcon_diverse', 'set48_p3',
     'set248_p3', 'set48_p10_seed1', 'supcon_low', 'supcon_replace',
     'supcon_set8', 'supcon_set48', 'set48_p30'),
)
ALL_FITS = set(ARMS) | set(TEACHER_TASKS)


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_DEPTH_RECOVERY_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_DEPTH_RECOVERY_{kind}/v1', expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope changed')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Recovery parents changed')


def scientific_dependencies(task):
    if task in TEACHER_TASKS:
        return ['targets']
    if task not in ARMS:
        raise ValueError('Unknown scientific task')
    extra = set(recipe(task)['target_names'].values()) & set(TEACHER_TASKS)
    return ['targets', *sorted(extra)]


def schedule(reused):
    if not {'targets', 'baseline'} <= set(reused) or not set(reused) <= ALL_FITS | {'targets'}:
        raise ValueError('Completed shared targets and baseline required; unknown reuse forbidden')
    lanes = [[t for t in lane if t not in reused] for lane in LANES]
    rows = []
    for i in range(max(map(len, lanes))):
        for lane, names in enumerate(lanes):
            if i >= len(names):
                continue
            task = names[i]
            science = scientific_dependencies(task)
            success = [t for t in science if t not in reused]
            slots = [t for t in names[:i] if t not in success]
            rows.append(dict(task_id=task, lane=lane, scientific_dependencies=science,
                success_dependencies=success, slot_dependencies=slots,
                dependencies=list(dict.fromkeys(success + slots))))
    expected = ALL_FITS - set(reused)
    if {r['task_id'] for r in rows} != expected:
        raise ValueError('Schedule would skip or repeat a registered fit')
    prior = set()
    for row in rows:
        if not set(row['dependencies']) <= prior:
            raise ValueError('Not a topological submission order')
        prior.add(row['task_id'])
    rows.append(dict(task_id='report', lane=None, scientific_dependencies=[],
        success_dependencies=[], slot_dependencies=[r['task_id'] for r in rows],
        dependencies=[r['task_id'] for r in rows]))
    return rows
