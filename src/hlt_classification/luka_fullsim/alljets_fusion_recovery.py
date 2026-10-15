"""Explicit dual-source v2 recovery from the authenticated A100 diagnostic."""
from pathlib import Path
import json
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.jetclass2_delphes import dzfix_fusion_model as fusion
from . import alljets_fusion_campaign as c, alljets_fusion_source as sources
from . import alljets_fusion_diagnostic as diagnostic
from .preflight_reuse import _files
from .regular_recovery import failed_accounting

DIAGNOSTIC_COMMIT = '658cafa5f4f59b7be907cbfed245a2c37cb0dcd6'
ORIGINAL_COMMIT = 'e9b0b560e60b60bf4e779cd69756e1e3163a7fb6'
REPORT_HASH = '60abf23c62b03b2b627d8afc5823b69f65cedbf3564ab1edb104f7b0a48bea8b'
ADDITIONS = {'src/hlt_classification/luka_fullsim/alljets_fusion_'+n+'.py' for n in ('sync', 'recovery')}
CHANGES = {'src/hlt_classification/luka_fullsim/alljets_fusion_'+n+'.py'
           for n in ('source', 'campaign', 'worker', 'preflight', 'submission')}

ORIGINAL_CHECK = '''
import json, sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_fullsim import alljets_fusion_diagnostic as d
from hlt_classification.scouting.hcwdl_exact_dag_submission import load_exact_dag_journal
from hlt_classification.scouting.hcwdl_recovery import assemble_submission_ledger
spec = load_json(sys.argv[1])
gate, runtime = d.validate(spec, original=True)
root = Path(spec['root'])/'submission'
events, jobs = load_exact_dag_journal(root/'submission_ledger_journal', identity=spec['content_hash'], plan=d.plan(spec))
if load_json(root/'submission_ledger.json') != assemble_submission_ledger(campaign_spec_sha256=spec['content_hash'], events=events):
    raise ValueError('Diagnostic submission ledger differs from exact journal')
if set(jobs) != {'diagnostic'}:
    raise ValueError('Diagnostic job binding missing')
print('LUKA_SYNC_AUTH='+json.dumps(dict(gate=gate['content_hash'], diagnostic=spec['content_hash'], jobs=jobs)))
'''


def compatible_source(spec, active, project):
    before_root, after_root = Path(spec['project_dir']), Path(project)
    validate_source_snapshot(spec['source_snapshot'], repository=before_root, require_clean=True)
    validate_source_snapshot(active, repository=after_root, require_clean=True)
    before, after = _files(before_root), _files(after_root)
    if not before or not before <= after or after-before != ADDITIONS:
        raise ValueError('Unregistered synchronous recovery source additions/deletions')
    records = {}
    for name in sorted(before):
        old, new = sha256_file(before_root/name), sha256_file(after_root/name)
        if old != new and name not in CHANGES:
            raise ValueError('Existing scientific source changed: '+name)
        records[name] = dict(original=old, execution=new)
    return dict(modules=records, additions=sorted(ADDITIONS), allowed_changes=sorted(CHANGES))


def check_report(spec, report, gate, runtime, job):
    """Re-read every bound diagnostic output, not presence-only reuse."""
    root = Path(spec['root'])
    c.validate(report, 'CUDA_DIAGNOSTIC_REPORT', dict(spec=spec['content_hash']))
    if (report['content_hash'] != REPORT_HASH or report.get('science_admissible') is not False
            or report.get('all_cases_completed') is not False
            or [row['name'] for row in report['cases']] != [row[0] for row in diagnostic.CASES]):
        raise ValueError('Expected the reviewed four-case A100 diagnostic, not science admission')
    for row, (name, node, candidate, blocking) in zip(report['cases'], diagnostic.CASES):
        refs = row['outputs']
        paths = [ref['path'] for ref in refs]
        expected_result = f'cases/{name}/'+('result.json' if candidate else 'failure.json')
        if (row['completed'] is not candidate or row['returncode'] != (0 if candidate else 1)
                or len(paths) != len(set(paths)) or expected_result not in paths
                or any(not path.startswith(f'cases/{name}/') for path in paths)
                or row['log']['path'] != name+'.out'):
            raise ValueError('Original/candidate outcomes or output scopes differ')
        for ref in [row['log'], *refs]:
            validate_file_ref(ref, root=root)
        if f'cases/{name}/runtime.json' not in paths or f'cases/{name}/witness.json' not in paths:
            raise ValueError('Missing diagnostic runtime/witness binding')
        case_runtime = load_json(root/f'cases/{name}/runtime.json')
        if (case_runtime.get('job_id') != job or case_runtime.get('environment') != runtime['environment']
                or case_runtime.get('gpu') != runtime['gpu']
                or case_runtime.get('cuda_launch_blocking') != ('1' if blocking else '0')):
            raise ValueError('Diagnostic case environment/job differs')
        result = load_json(root/expected_result)
        c.validate(result, 'CUDA_DIAGNOSTIC_CASE' if candidate else 'CUDA_DIAGNOSTIC_FAILURE',
                   dict(spec=spec['content_hash']))
        if (result.get('name') != name or result.get('completed') is not candidate
                or result.get('science_admissible') is not False):
            raise ValueError('Diagnostic case facts differ')
        if not candidate:
            if not any(s in result['error'] for s in ('CUDA', 'cuDNN')):
                raise ValueError('Original failure was not the diagnosed CUDA path')
            continue
        witness = load_json(root/f'cases/{name}/witness.json')
        expected_node = next(n for n in c.nodes() if n['node_id'] == node)
        if (result['witness'] != witness or witness['node'] != expected_node
                or witness['prepared_sha256'] != gate['prepared_sha256']
                or witness['train_rows'] != c.COUNTS['train'] or witness['batch_size'] != 256
                or witness['candidate'] is not True or result['synchronous_restore']['cuda_restores'] <= 0):
            raise ValueError('Candidate stress witness differs')
        fusion.validate_offload_stats(result['offload'], calls=3)
        if set(result['parities']) != {'False', 'True'}:
            raise ValueError('Both precision parities required')
        for key, precision in [('False', 'fp32'), ('True', 'bf16')]:
            evidence = result['parities'][key]
            parity = evidence['report']
            expected = dict(passed=True, device_type='cuda', precision=precision, steps=3,
                checks=fusion.PARITY_CHECKS, parity_backend=fusion.PARITY_BACKEND,
                saved_tensor_storage=fusion.PAIR_OFFLOAD_POLICY, tolerance=fusion.PARITY_TOLERANCES[precision])
            if (evidence['candidate'] != diagnostic.POLICY['candidate'] or evidence['cuda_restores'] <= 0
                    or any(parity.get(k) != v for k,v in expected.items())):
                raise ValueError('Candidate parity incomplete or weakened')
            fusion.validate_offload_stats(parity['offload_stats'], calls=3)


def authenticate(diagnostic_path, active, project):
    path = Path(diagnostic_path).resolve(strict=True)
    spec = load_json(path)
    c.validate(spec, 'CUDA_DIAGNOSTIC_SPEC')
    if (spec['source_commit'] != DIAGNOSTIC_COMMIT
            or spec['source_snapshot']['git_commit'] != DIAGNOSTIC_COMMIT
            or path != Path(spec['root'])/'diagnostic_spec.json' or spec['policy'] != diagnostic.POLICY):
        raise ValueError('Recovery requires the original 658cafa5 diagnostic')
    source_reuse = compatible_source(spec, active, project)
    env = dict(os.environ, PYTHONPATH=str(Path(spec['project_dir'])/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    print('Authenticating diagnostic and failed gate in their ORIGINAL pinned sources...', flush=True)
    run = subprocess.run([sys.executable, '-u', '-s', '-c', ORIGINAL_CHECK, str(path)],
        cwd=spec['project_dir'], env=env, capture_output=True, text=True)
    if run.returncode:
        raise ValueError('Original diagnostic authentication failed:\n'+run.stderr)
    records = [json.loads(line.removeprefix('LUKA_SYNC_AUTH=')) for line in run.stdout.splitlines()
               if line.startswith('LUKA_SYNC_AUTH=')]
    for ref in (spec['original_gate'], spec['original_runtime']):
        validate_file_ref(ref)
    gate = load_json(spec['original_gate']['path'])
    runtime = load_json(spec['original_runtime']['path'])
    c.validate(gate, 'GATE')
    if (gate['source_commit'] != ORIGINAL_COMMIT or gate['schema_version'] != 1
            or any(gate.get(k) != v for k,v in c.registration().items())
            or len(records) != 1 or records[0]['gate'] != gate['content_hash']
            or records[0]['diagnostic'] != spec['content_hash'] or set(records[0]['jobs']) != {'diagnostic'}):
        raise ValueError('Original gate/scientific registration/journal differs')
    job = records[0]['jobs']['diagnostic']
    report_path = Path(spec['root'])/'diagnostic_report.json'
    report = load_json(report_path)
    check_report(spec, report, gate, runtime, job)
    reuse = dict(diagnostic=file_ref(path), report=file_ref(report_path), original_gate=spec['original_gate'],
        original_runtime=spec['original_runtime'], source_reuse=source_reuse,
        accounting=dict(original=dict(job_id=str(runtime['job_id']), state='FAILED', exit_code='1:0'),
                        diagnostic=dict(job_id=job, state='FAILED', exit_code='1:0')))
    return gate, spec, reuse


def location(root, gate, diagnostic_spec, project):
    root = Path(root).resolve()
    parent = load_json(gate['parent_import']['spec']['path'])
    c.protect(root, parent, project)
    for name in (gate['root'], gate['project_dir'], diagnostic_spec['root'], diagnostic_spec['project_dir']):
        other = Path(name).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise PermissionError('Recovery overlaps original artifacts/source')
    return parent


def create(*, diagnostic_path, root, project, commit):
    root, project = Path(root).resolve(), Path(project).resolve()
    active = sources.source(project, commit)
    gate, diagnostic_spec, reuse = authenticate(diagnostic_path, active, project)
    location(root, gate, diagnostic_spec, project)
    for recorded in reuse['accounting'].values():
        if failed_accounting(recorded['job_id']) != recorded:
            raise ValueError('Terminal accounting differs')
    value = c.artifact('GATE', version=2,
        parents=dict(source=active['content_hash'], original_gate=gate['content_hash'], diagnostic=REPORT_HASH),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        parent_import=gate['parent_import'], prepared_sha256=gate['prepared_sha256'], recovery=reuse,
        **c.registration(2))
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'gate_spec.json', value)
    return value


def validate_gate(value):
    c.validate(value, 'GATE')
    if (value['schema_version'] != 2 or any(value.get(k) != v for k,v in c.registration(2).items())
            or sources.source(value['project_dir'], value['source_commit']) != value['source_snapshot']
            or load_json(Path(value['root'])/'gate_spec.json') != value):
        raise ValueError('Synchronous recovery source/registration differs')
    for name in ('diagnostic', 'report', 'original_gate', 'original_runtime'):
        validate_file_ref(value['recovery'][name])
    gate, diagnostic_spec, reuse = authenticate(value['recovery']['diagnostic']['path'],
        value['source_snapshot'], value['project_dir'])
    parent = location(value['root'], gate, diagnostic_spec, value['project_dir'])
    if (value['recovery'] != reuse or value['parent_import'] != gate['parent_import']
            or value['prepared_sha256'] != gate['prepared_sha256']
            or value['parents'] != dict(source=value['source_snapshot']['content_hash'],
                original_gate=gate['content_hash'], diagnostic=REPORT_HASH)):
        raise ValueError('Recovered cache/source/diagnostic bindings differ')
    return parent
