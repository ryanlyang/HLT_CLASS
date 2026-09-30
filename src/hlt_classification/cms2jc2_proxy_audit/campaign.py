"""Freeze a separate diagnostic scope without changing production admission."""
from pathlib import Path
import os
import re
import shutil
import subprocess

from hlt_classification.cms2jc2_production import campaign as production, contracts as k
from hlt_classification.cms2jc2_response import contracts as old
from hlt_classification.cms2jc2_response.dev_data import checked_file
from hlt_classification.cms2jc2_response.dev_diagnostics import merge_payloads

EXTRA = ('scripts/cms2jc2_proxy_train_audit.py', 'sbatch/run_cms2jc2_proxy_train_audit.sh',
         'scripts/queue_cms2jc2_proxy_train_audit.sh',
         'docs/plans/CMS2JC2_PROXY_TRAIN_AUDIT_PLAN.md',
         'docs/contracts/CMS2JC2_PROXY_TRAIN_AUDIT.md')
RESOURCES = dict(cpus=8, memory_gib=64, hours=8, partition='tigris', account='reu-aisocial', gpus=0)


def artifact(kind, **fields):
    return k.artifact('AUDIT_'+kind, validation_accessed=False, native_hlt_access=False, **fields)


def validate(value, kind, **kwargs):
    k.validate(value, 'AUDIT_'+kind, test=False, **kwargs)
    if value['validation_accessed'] is not False or value['native_hlt_access'] is not False:
        raise ValueError('Audit data access scope differs')


def source_snapshot(project, commit):
    project = Path(project)
    source = production.source_snapshot(project, commit)
    files = dict(source['files'])
    extra = [*EXTRA, *(p.relative_to(project).as_posix() for p in
                       (project/'src/hlt_classification/cms2jc2_proxy_audit').glob('*.py'))]
    for name in extra:
        subprocess.run(['git', '-C', str(project), 'ls-files', '--error-unmatch', name],
                       check=True, capture_output=True)
        files[name] = k.sha256_file(project/name)
    return artifact('SOURCE', commit=commit, files=files)


def train_receipts(study):
    """Metadata only; never glob receipts belonging to other roles."""
    rows = [s for s in study['shards'] if s['role'] == 'train']
    if (not rows or len({r['shard_id'] for r in rows}) != len(rows)
            or sum(s['jets'] for s in rows) != study['counts']['train']):
        raise ValueError('Registered training population differs')
    result = []
    for shard in rows:
        path = k.safe(study['root'], f"shards/{shard['shard_id']}.json")
        receipt = k.load_json(path)
        kind = 'SHARD_EXECUTION_REPAIR' if 'execution_repair' in receipt else 'SHARD'
        k.validate(receipt, kind, test=False, parents={'study': study['content_hash'],
                    'population': study['population']['content_hash']})
        if any(receipt[n] != shard[n] for n in ('shard_id', 'role', 'jets', 'ordered_identities')):
            raise ValueError('Training receipt identity differs')
        result.append(dict(shard_id=shard['shard_id'], receipt=k.ref(path)))
    return result


def cms_reference(study):
    """Reuse frozen summary data; resolve ranges through authenticated metadata."""
    report = production.imported(study, 'confirmation_report')
    k.validate_content_hash(report, expected_contract=report['contract'])
    if report['content_hash'] != study['reviewed_confirmation_hash']:
        raise ValueError('CMS report identity differs')
    stage = k.load_json(k.checked(study['confirmation_spec']))
    if stage['stage'] == 'frozen_reduced':
        stage = k.load_json(checked_file(stage['subject_spec']))
    if stage['stage'] != 'frozen_confirm':
        raise ValueError('Expected frozen CMS confirmation')
    ranges_hash = stage['reuse']['parents']['ranges']
    visited = set()
    while stage['stage'] != 'pilot':
        k.validate_content_hash(stage, expected_contract=stage['contract'])
        if stage['content_hash'] in visited:
            raise ValueError('Cyclic reference lineage')
        visited.add(stage['content_hash'])
        stage = k.load_json(checked_file(stage['parent_spec']))
    k.validate_content_hash(stage, expected_contract=stage['contract'])
    receipt = k.load_json(Path(stage['root'])/'stages'/stage['name']/'receipts/prepare.json')
    old.validate(receipt, 'DEV_OUTPUTS', parents={'stage': stage['content_hash']})
    record = receipt['outputs']['ranges']
    path = k.safe(stage['root'], record['relative'])
    if k.sha256_file(path) != record['sha256']:
        raise ValueError('CMS diagnostic ranges bytes differ')
    ranges = k.load_json(path)
    old.validate(ranges, 'DEV_RANGES')
    if ranges['content_hash'] != ranges_hash:
        raise ValueError('CMS diagnostic range lineage differs')
    payloads = []
    for group in report['by_file'].values():
        payload = group['JOINT']
        if payload['cells']['offline/all']['jets'] != payload['cells']['real/all']['jets']:
            raise ValueError('CMS paired population differs')
        payloads.append(dict(cells={n: v for n, v in payload['cells'].items()
                                   if n.split('/')[0] in ('offline', 'real')},
                             correlations={n: v for n, v in payload['correlations'].items()
                                           if n in ('offline', 'real')}))
    hist = merge_payloads(payloads)
    if hist['cells']['real/all']['jets'] != report['jets']:
        raise ValueError('CMS reference population differs')
    reference = artifact('REPORT', parents={'cms_report': report['content_hash'], 'ranges': ranges_hash},
        histograms=hist, cms_jets=report['jets'], source_files=len(report['by_file']),
        coverage=report.get('coverage'), decision=report.get('decision'),
        qualification_status=report.get('full_population_status', report.get('status')),
        reference_only=True, raw_cms_particles_accessed=False, physics_production_qualified=False)
    return reference, ranges


def create(*, dataset_root, root, project, commit):
    root, dataset_root, project = map(lambda p: Path(p).resolve(), (root, dataset_root, project))
    study = k.load_json(dataset_root/'study_spec.json')
    production.validate_study(study)
    production.require_preflight(study)
    if Path(study['root']).resolve() != dataset_root:
        raise ValueError('Dataset location differs')
    if (root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root)
                            for p in (dataset_root, Path(study['data_root']).resolve()))):
        raise ValueError('Use a new, separate audit output directory')
    if shutil.disk_usage(root.parent).free < 2*k.GIB:
        raise OSError('Train audit needs at least 2 GiB free for diagnostics')
    source = source_snapshot(project, commit)
    # Observable and offline reader semantics must agree with the frozen producer.
    for name in ('cms2jc2_production/population.py', 'cms2jc2_response/bridge.py',
                 'cms2jc2_response/dev_diagnostics.py', 'cms2jc2_response/metrics.py',
                 'cms2jc2_response/features.py'):
        name = 'src/hlt_classification/'+name
        if source['files'][name] != study['source']['files'][name]:
            raise ValueError('Frozen diagnostic/reader semantics changed: '+name)
    print('C2JPA prepare: binding training receipts and saved CMS summaries (no particle reads)', flush=True)
    receipts = train_receipts(study)
    reference, ranges = cms_reference(study)
    root.mkdir(parents=True)
    refs = dict(reference=k.write(root/'cms_reference.json', reference),
                ranges=k.write(root/'ranges.json', ranges))
    spec = artifact('SPEC', parents={'study': study['content_hash'], 'source': source['content_hash']},
        root=str(root), project_dir=str(project), source=source, study=k.ref(dataset_root/'study_spec.json'),
        train=receipts, imports=refs, resources=RESOURCES, jets=study['counts']['train'],
        role='train', replica=0, physics_production_qualified=False)
    k.write(root/'audit_spec.json', spec)
    k.write(root/'command_plan.json', plan(spec))
    return spec


def load_inputs(spec, *, source=False):
    validate(spec, 'SPEC')
    study = k.load_json(k.checked(spec['study']))
    production.validate_study(study)
    production.require_preflight(study)
    if (spec['parents'] != {'study': study['content_hash'], 'source': spec['source']['content_hash']}
            or spec['resources'] != RESOURCES or spec['role'] != 'train' or spec['replica'] != 0
            or spec['jets'] != study['counts']['train'] or spec['train'] != train_receipts(study)):
        raise ValueError('Audit frozen scope differs')
    if source and source_snapshot(spec['project_dir'], spec['source']['commit']) != spec['source']:
        raise ValueError('Audit executable source differs')
    reference, ranges = [k.load_json(k.checked(spec['imports'][n])) for n in ('reference', 'ranges')]
    validate(reference, 'REPORT')
    old.validate(ranges, 'DEV_RANGES')
    if reference['parents'] != {'cms_report': study['reviewed_confirmation_hash'], 'ranges': ranges['content_hash']}:
        raise ValueError('Audit reference lineage differs')
    return study, reference, ranges


def plan(spec):
    r, root, project = spec['resources'], Path(spec['root']), spec['project_dir']
    argv = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
            f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_gib']}G", f"--time={r['hours']:02d}:00:00",
            f"--partition={r['partition']}", f"--account={r['account']}", '--job-name=c2jp_train_audit',
            f"--comment=c2jpa:{spec['content_hash']}", f'--chdir={project}',
            f'--output={root}/slurm-%j.out', f'{project}/sbatch/run_cms2jc2_proxy_train_audit.sh',
            project, str(root/'audit_spec.json')]
    return artifact('PLAN', parents={'spec': spec['content_hash']}, argv=argv, resources=r)


def submit(spec, *, execute=False, reviewed_hash=None):
    load_inputs(spec, source=True)
    expected = plan(spec)
    saved = k.load_json(Path(spec['root'])/'command_plan.json')
    if saved != expected:
        raise ValueError('Command plan differs')
    if not execute:
        return expected
    if reviewed_hash != expected['content_hash']:
        raise PermissionError('Live submission requires the exact reviewed plan hash')
    env = {n: v for n, v in os.environ.items() if not n.startswith(('SBATCH_', 'SLURM_'))}
    subprocess.run([expected['argv'][0], '--test-only', *expected['argv'][1:]],
                   env=env, check=True, capture_output=True, text=True)
    root = Path(spec['root'])
    # Exclusive intent: a disconnect/ambiguous response never silently duplicates work.
    with (root/'submission_intent.json').open('x') as stream:
        import json
        json.dump(expected, stream, sort_keys=True)
    result = subprocess.run(expected['argv'], env=env, capture_output=True, text=True)
    k.atomic_publish_bytes(root/'submission_response.txt', (result.stdout+'\n'+result.stderr).encode())
    if result.returncode or not re.fullmatch(r'\d+(?:;[\w.-]+)?\s*', result.stdout):
        raise RuntimeError('Submission outcome requires inspection; do not automatically retry')
    row = artifact('LEDGER', parents={'plan': expected['content_hash']}, job=result.stdout.strip().split(';')[0])
    k.write(root/'submission_ledger.json', row)
    return row
