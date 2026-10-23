"""One immutable eight-job recovery registration, no silent replacement."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_multiteacher_trial import campaign as donor
from .contracts import artifact, validate, tasks, REPLACEMENTS
from . import source


def policy(parent, resources, archive):
    value = donor.policy(parent, resources)
    prior = archive['prior_request']
    if any(type(prior[k]) is not int or prior[k] < 1 for k in ('train_minutes','export_minutes')):
        raise ValueError('Invalid historical teacher requests')
    minutes = max(360, prior['train_minutes']+prior['export_minutes'])
    if minutes > 1440:
        raise ValueError('Combined teacher/export request exceeds registered 24h ceiling')
    value.update(tasks=tasks(), teacher_nodes=archive['nodes'], teacher_minutes=minutes,
        scientific_fits=7, teacher_fits=2, student_fits=5, storage_budget_bytes=5*2**30,
        teacher_policy='new_LOGIT_replacements_not_restored_historical_weights',
        checkpoint_retention='retain_all_selected_teachers_and_students',
        teacher_resource_estimate='sum_archived_train_and_export_requests_min6h_no_new_measurement')
    return value


def parents(spec):
    return {k:spec[k]['content_hash'] for k in ('source_snapshot','anchor','archive','resources','reference')} | {
        'survivors':artifact('SURVIVORS', teachers=spec['survivors'])['content_hash']}


def create(root, parent_path, attention_path, reference_path, project, commit, *, available_quota_gib):
    root, project = Path(root).resolve(), Path(project).resolve()
    snapshot = source.current(project, commit)
    anchor, parent = source.donor.authenticate(parent_path)
    survivors = source.survivors(anchor)
    archive = source.archived(attention_path, anchor, parent)
    ref = source.donor.reference(reference_path, anchor, source.donor.donor_node(parent))
    resources = source.donor.resources(ref, anchor, source.donor.donor_node(parent))
    if type(available_quota_gib) is not int or available_quota_gib < 5:
        raise ValueError('Reserve at least5GiB for recovery and all five students')
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container']),
        Path(attention_path).resolve().parent.parent, Path(reference_path).resolve().parent.parent]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('Recovery root overlaps existing source, data or campaign')
    fields = dict(root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        anchor=anchor, survivors=survivors, archive=archive, resources=resources, reference=ref,
        u100_reference=None, available_quota_gib=available_quota_gib, **policy(parent, resources, archive))
    spec = artifact('SPEC', parents=parents(fields), **fields)
    source.imports(spec)
    source.inputs(spec)
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    storage(spec)
    return spec


def validate_spec(spec):
    validate(spec, 'SPEC', parents(spec))
    parent = load_json(spec['anchor']['spec']['path'])
    if (any(spec.get(k) != v for k,v in policy(parent, spec['resources'], spec['archive']).items())
            or spec.get('u100_reference') is not None
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 5
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Recovery registration differs')


def check(spec):
    validate_spec(spec)
    if source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Pinned recovery source changed')
    source.imports(spec)


def storage(spec):
    # Same enforced remaining-budget semantics as the donor, now5GiB.
    return donor.storage(spec)
