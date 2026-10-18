"""Separate gate/science registrations; exact source, admission and quota checks."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullsim.science_campaign import preflight_accounting
from . import source
from .contracts import artifact, validate, recipe, nodes, tasks, COUNTS, execution_policy

AUTHORIZATION = 'AUTHORIZE LUKA PRIVILEGED GATE EXACT PLAN'


def registration(parent):
    models = nodes(parent['nodes'])
    return dict(counts=COUNTS, nodes=models, tasks=tasks(models), recipe=recipe(),
                attention_execution=execution_policy(),
                execution_site=execution_site('sporc_a100_debug'), cpus=6, memory_mb=128000,
                gate_minutes=480, automatic_followup=False, automatic_retry=False)


def create(root, parent_path, project, commit, *, available_quota_gib):
    project, root = Path(project).resolve(), Path(root).resolve()
    active = source.current(project, commit)
    anchor, parent = source.authenticate(parent_path, project)
    protect = [project, Path(parent['project_dir']), Path(parent['root']).parent,
               Path(anchor['evidence']['prepared_root']), Path(anchor['evidence']['raw_container'])]
    for path in protect:
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise PermissionError('New root overlaps source or parent artifacts')
    if type(available_quota_gib) is not int or available_quota_gib<=0:
        raise ValueError('Declare personally available quota GiB')
    gate = artifact('GATE', parents=dict(source=active['content_hash'], anchor=anchor['content_hash']),
                    root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
                    anchor=anchor, available_quota_gib=available_quota_gib, **registration(parent))
    root.mkdir(parents=True,exist_ok=False)
    write_immutable_json(root/'gate_spec.json',gate)
    return gate


def check_gate(gate):
    validate(gate,'GATE',dict(source=gate['source_snapshot']['content_hash'],anchor=gate['anchor']['content_hash']))
    if source.current(gate['project_dir'],gate['source_commit'])!=gate['source_snapshot']:
        raise ValueError('Active source differs')
    imported,parent = source.authenticate(gate['anchor']['spec']['path'],gate['project_dir'])
    if (imported!=gate['anchor'] or any(gate.get(k)!=v for k,v in registration(parent).items())
            or load_json(Path(gate['root'])/'gate_spec.json')!=gate):
        raise ValueError('Gate/anchor registration changed')
    return parent


def storage(gate, lengths):
    # One factor bank each for science and technical PRIV_D000, a temporary
    # publication allowance, small probability banks, checkpoints and reserve.
    payload = 4*9*sum(int(n) for n in lengths)
    required = 3*payload + 5*2**30 + 24*len(lengths)*11*4
    free = shutil.disk_usage(gate['root']).free
    if required>min(free, gate['available_quota_gib']*2**30):
        raise OSError(f'Gate banks need {required/2**30:.2f} GiB including technical exports and reserve')
    return dict(factor_bank_bytes=payload, required_bytes=required,
                filesystem_free_bytes=free, declared_quota_gib=gate['available_quota_gib'])


def evidence(gate):
    check_gate(gate)
    from .submission import accepted_jobs
    from .preflight import validate_report
    report = load_json(Path(gate['root'])/'preflight/receipt.json')
    validate_report(gate, report)
    if accepted_jobs(gate)!={'preflight':report['job_id']}:
        raise ValueError('Preflight job differs from exact journal')
    preflight_accounting(report['job_id'])
    return report


def science(gate):
    report = evidence(gate)
    root = Path(gate['root'])/'science'
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash']),
        root=str(root), gate_path=str(Path(gate['root'])/'gate_spec.json'),
        project_dir=gate['project_dir'], source_commit=gate['source_commit'],
        train_minutes=report['train_minutes'], export_minutes=report['export_minutes'])
    root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check_spec(spec):
    gate = load_json(spec['gate_path'])
    report = evidence(gate)
    validate(spec,'SPEC',dict(gate=gate['content_hash'],preflight=report['content_hash']))
    if (Path(spec['root'])!=Path(gate['root'])/'science' or spec['project_dir']!=gate['project_dir']
            or spec['source_commit']!=gate['source_commit'] or spec['train_minutes']!=report['train_minutes']
            or spec['export_minutes']!=report['export_minutes']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Science registration differs')
    return gate, report
