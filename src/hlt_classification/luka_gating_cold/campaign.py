"""Fresh tier3 gate; no mutation or acceptance reuse of warm campaigns."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from hlt_classification.luka_gating.contracts import execution_policy
from .contracts import artifact, validate, recipe, nodes, tasks, COUNTS, AUTHORIZATION
from . import source


def registration(parent):
    models = nodes(parent['nodes'])
    return dict(counts=COUNTS, nodes=models, tasks=tasks(models), recipe=recipe(),
        attention_execution=execution_policy(), execution_site=execution_site('sporc_a100'),
        cpus=6, memory_mb=128000, gate_minutes=480, science_ceiling_minutes=1440,
        automatic_followup=False, automatic_retry=False)


def create(root, parent_path, project, commit, *, available_quota_gib, scratch_base='/tmp'):
    project, root = Path(project).resolve(), Path(root).resolve()
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path)
    protected = [project, Path(parent['project_dir']), Path(parent['root']).parent,
                 Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise PermissionError('New root overlaps existing campaign/source/data')
    if str(scratch_base) not in ('/tmp','/local/scratch'):
        raise PermissionError('Explicit node-local scratch required; no home fallback')
    if type(available_quota_gib) is not int or available_quota_gib < 2:
        raise ValueError('Declare at least 2 personally available persistent GiB')
    gate = artifact('GATE', parents=dict(source=active['content_hash'],anchor=anchor['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        anchor=anchor, scratch_base=str(scratch_base), available_quota_gib=available_quota_gib,
        **registration(parent))
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'gate_spec.json',gate)
    return gate


def read_gate(gate):
    validate(gate,'GATE',dict(source=gate['source_snapshot']['content_hash'],anchor=gate['anchor']['content_hash']))
    validate(gate['anchor'],'ANCHOR')
    if (gate['execution_site'] != execution_site('sporc_a100') or gate['recipe'] != recipe()
            or gate['attention_execution'] != execution_policy()
            or gate['scratch_base'] not in ('/tmp','/local/scratch')
            or type(gate['available_quota_gib']) is not int or gate['available_quota_gib'] < 2
            or load_json(Path(gate['root'])/'gate_spec.json') != gate):
        raise ValueError('Cold gate source/site/recipe/record differs')


def check_gate(gate):
    read_gate(gate)
    if source.current(gate['project_dir'],gate['source_commit']) != gate['source_snapshot']:
        raise ValueError('Execution source differs')
    anchor, parent = source.authenticate(gate['anchor']['spec']['path'])
    if anchor != gate['anchor'] or any(gate.get(k) != v for k,v in registration(parent).items()):
        raise ValueError('Scientific registration or anchor changed')
    return parent


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate,report)
    if accepted_jobs(gate) != {'preflight':report['job_id']}:
        raise ValueError('Accepted gate journal differs')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC',parents=dict(gate=gate['content_hash'],preflight=report['content_hash']),
        root=str(root),gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'],source_commit=gate['source_commit'],fit_minutes=report['fit_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def validate_spec(spec, gate, report):
    validate(spec,'SPEC',dict(gate=gate['content_hash'],preflight=report['content_hash']))
    if (Path(spec['root']) != Path(gate['root'])/'science' or spec['project_dir'] != gate['project_dir']
            or spec['source_commit'] != gate['source_commit'] or spec['fit_minutes'] != report['fit_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Science registration differs')


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate_spec(spec,gate,report)
    return gate, report
