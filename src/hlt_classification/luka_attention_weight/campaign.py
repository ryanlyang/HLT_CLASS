"""Fresh split-partition registrations; no implicit source or acceptance reuse."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from . import source
from .contracts import artifact, validate, nodes, recipe, COUNTS, ARMS, AUTHORIZATION


def registration(parent, arm):
    coefficient, site = ARMS[arm]
    return dict(arm=arm, coefficient=coefficient, counts=COUNTS, nodes=nodes(parent['nodes'], arm),
        recipe=recipe(coefficient), execution_site=execution_site(site), cpus=6, memory_mb=128000,
        gate_minutes=480, science_ceiling_minutes=1440, automatic_followup=False,
        automatic_retry=False, retention='verified_owned_scratch_payloads_only_v1')


def create(root, parent_path, reference_path, project, commit, *, arm, available_quota_gib, scratch_base):
    project, root = Path(project).resolve(), Path(root).resolve()
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    imported = source.references(reference_path, anchor)
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
        Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container']),
        Path(reference_path).resolve().parent.parent]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New root overlaps existing campaign/source/data')
    # This is a remote Linux execution path, even when registering on Windows.
    scratch = str(scratch_base)
    if scratch not in ('/tmp', '/local/scratch'):
        raise PermissionError('Register /tmp or /local/scratch; no persistent storage fallback')
    if type(available_quota_gib) is not int or available_quota_gib < 1:
        raise ValueError('Declare personally available persistent storage GiB')
    root.mkdir(parents=True, exist_ok=False)
    predictions = {}
    for name, ref in imported['predictions'].items():
        if name not in {a+'_'+c for a in ('LOGIT','ATTN') for c in ('D066','D033','D000')}:
            raise ValueError('Unexpected reference prediction')
        path = validate_file_ref(ref)
        atomic_publish_bytes(root/'references'/(name+'.npz'), path.read_bytes())
        predictions[name] = file_ref(root/'references'/(name+'.npz'))
        if predictions[name]['sha256'] != ref['sha256']:
            raise ValueError('Reference copy differs')
    saved = artifact('REFERENCES', parents=dict(anchor=anchor['content_hash']),
        imported_hash=imported['content_hash'], rows=imported['rows'], predictions=predictions,
        spec=imported['spec'])
    write_immutable_json(root/'references.json', saved)
    gate = artifact('GATE', parents=dict(source=active['content_hash'], anchor=anchor['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        anchor=anchor, reference_import=imported, references=file_ref(root/'references.json'),
        available_quota_gib=available_quota_gib, scratch_base=scratch, **registration(parent, arm))
    write_immutable_json(root/'gate_spec.json', gate)
    return gate


def read_gate(gate):
    validate(gate, 'GATE', dict(source=gate['source_snapshot']['content_hash'], anchor=gate['anchor']['content_hash']))
    validate(gate['anchor'], 'ANCHOR')
    validate(gate['reference_import'], 'REFERENCES', dict(anchor=gate['anchor']['content_hash']))
    if load_json(Path(gate['root'])/'gate_spec.json') != gate:
        raise ValueError('Gate record differs')
    validate_file_ref(gate['references'])
    refs = load_json(gate['references']['path'])
    validate(refs, 'REFERENCES', dict(anchor=gate['anchor']['content_hash']))
    if refs['rows'] != gate['reference_import']['rows'] or refs['imported_hash'] != gate['reference_import']['content_hash']:
        raise ValueError('Reference snapshot differs')
    if refs['predictions'].keys() != gate['reference_import']['predictions'].keys():
        raise ValueError('Reference prediction coverage differs')
    for name, ref in refs['predictions'].items():
        if ref['sha256'] != gate['reference_import']['predictions'][name]['sha256']:
            raise ValueError('Reference prediction hash differs')
        validate_file_ref(ref)
    return refs


def check_gate(gate):
    read_gate(gate)
    if source.current(gate['project_dir'], gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Execution source differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'])
    if anchor != gate['anchor'] or any(gate.get(k) != v for k,v in registration(parent, gate['arm']).items()):
        raise ValueError('Anchor or scientific registration differs')
    return parent


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate) != {'preflight':report['job_id']}:
        raise ValueError('Preflight job differs from exact submission journal')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'], ladder_minutes=report['ladder_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science' or spec['project_dir'] != gate['project_dir']
            or spec['source_commit'] != gate['source_commit'] or spec['ladder_minutes'] != report['ladder_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Science registration differs')
    return gate, report
