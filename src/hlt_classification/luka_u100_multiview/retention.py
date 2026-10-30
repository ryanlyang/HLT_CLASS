"""Optional exact, receipt-bound retirement of newly owned target payloads only."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from .contracts import artifact, validate, TASKS


def allowed():
    return [f'accept/bank_{n}/targets.npz' for n in ('U100', 'CONCAT', 'FUSION')] + [
        'teacher_CONCAT/bank/targets.npz', 'teacher_FUSION/bank/targets.npz']


def validate_output(spec, ref, *, root):
    path = Path(root)/ref['path']
    if path.exists():
        return validate_file_ref(ref, root=root)
    study = Path(spec['root']); intent = study/'retirement_intent.json'
    if not intent.is_file():
        return validate_file_ref(ref, root=root)  # preserve ordinary missing-file failure
    value = load_json(intent)
    validate(value, 'RETIREMENT', dict(spec=spec['content_hash']))
    relative = path.relative_to(study).as_posix()
    expected = dict(ref, path=relative)
    if (relative not in allowed() or expected not in value['outputs']
            or {r['path'] for r in value['outputs']} != set(allowed())):
        raise ValueError('Missing output is not an authorized owned-bank retirement')
    return path


def retire(spec, *, execute=False, plan_hash=None):
    from . import campaign, worker
    campaign.check(spec)
    root = Path(spec['root'])
    if (root/'retirement_intent.json').exists():
        raise FileExistsError('Retirement already claimed; preserve inventory and inspect completion')
    receipts = {}
    for task in TASKS:
        worker.completed(spec, task)
        receipts[task] = file_ref(root/task/'receipt.json')
    refs = []
    for relative in allowed():
        path = root/relative
        if path.is_symlink() or path.resolve() != path.absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError('Owned bank path redirects outside its recorded location')
        refs.append(file_ref(path, root=root))
    plan = artifact('RETIREMENT', parents=dict(spec=spec['content_hash']), outputs=refs,
                    receipts=receipts, bytes=sum(r['bytes'] for r in refs), permanent=True)
    if not execute: return plan
    if plan_hash != plan['content_hash']:
        raise PermissionError('Exact reviewed retirement hash required')
    write_immutable_json(root/'retirement_intent.json', plan)
    for ref in refs:
        path = validate_file_ref(ref, root=root)
        path.unlink()
    write_immutable_json(root/'retirement_complete.json', artifact('RETIREMENT_COMPLETE',
        parents=dict(spec=spec['content_hash'], intent=plan['content_hash']), deleted_bytes=plan['bytes']))
    return plan
