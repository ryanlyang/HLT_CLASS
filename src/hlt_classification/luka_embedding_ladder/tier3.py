"""Explicit same-resource tier3 science from an untouched accepted debug gate."""
from pathlib import Path
import subprocess
import torch

from hlt_classification.data.cache_contracts import (
    load_json, write_immutable_json, with_content_hash, validate_content_hash)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.cms_proxy_ladder.submission import _walltime
from hlt_classification.jetclass2_delphes.execution import (
    execution_site, DEBUG_PROFILE_TRANSFER)
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_fullsim.regular_submission import check_site, clean_environment
from hlt_classification.scouting.hcwdl_exact_dag_submission import submit_exact_dag
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from . import contracts, source, worker

BASE_COMMIT = '0342951d7a11b959c3baa1611f9e581f6004c9f6'
MODULE = 'src/hlt_classification/luka_embedding_ladder/tier3.py'
REQUIRED = {
    MODULE, 'scripts/luka_embedding_ladder_tier3.py',
    'scripts/queue_luka_embedding_ladder_tier3.sh',
    'sbatch/run_luka_embedding_ladder_tier3.sh',
    'tests/test_luka_embedding_ladder_tier3.py',
    'docs/plans/LUKA_EMBEDDING_LADDER_TIER3_PLAN.md',
    'docs/contracts/LUKA_EMBEDDING_LADDER_TIER3.md',
    'docs/LUKA_EMBEDDING_LADDER_TIER3_RUNBOOK.md',
}
AUTHORIZATION = 'AUTHORIZE_LUKA_EMBEDDING_LADDER_TIER3_V1'
REPLAY = '''
import json,sys
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.luka_embedding_ladder.campaign import evidence
print('LADDER_TIER3_ACCEPTED='+json.dumps(evidence(load_json(sys.argv[1])),
    sort_keys=True,allow_nan=False))
'''


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_EMBEDDING_LADDER_TIER3_{kind}/v1',
        schema_version=1, parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_EMBEDDING_LADDER_TIER3_{kind}/v1',
                          expected_schema_version=1)
    if any(value.get(k) != v for k, v in FLAGS.items()):
        raise PermissionError('Exploratory scope differs')
    if parents is not None and value['parents'] != parents:
        raise ValueError('Transfer parents differ')


def git(project, *args):
    return subprocess.run(['git', '-C', str(project), *args], check=True,
                          capture_output=True, text=True).stdout.splitlines()


def current(project, commit):
    project = Path(project).resolve()
    snapshot = _source(project, commit)
    if (Path(__file__).resolve() != project/MODULE
            or not REQUIRED <= set(git(project, 'ls-files'))
            or not set(git(project, 'diff', '--name-only', BASE_COMMIT, '--', 'src')) <= {MODULE}):
        raise ValueError('Commit complete additive transfer; preserve measured scientific runtime')
    return snapshot


def matched_runtime(project, measured_commit):
    # Changes since the original ladder release may only be these additions.
    # Existing scientific files cannot change, even if an old-source replay passes.
    for row in git(project, 'diff', '--name-status', measured_commit, '--', 'src'):
        status, path = row.split('\t', 1)
        if status != 'A' or not (path == MODULE or path.startswith((
                'src/hlt_classification/luka_fullstat_embedding/',
                'src/hlt_classification/luka_embedding_weight/'))):
            raise ValueError('Runtime differs from the measured original ladder: '+row)


def authenticate(path, project):
    path = Path(path).resolve(strict=True)
    gate = load_json(path)
    contracts.validate(gate, 'GATE')
    if path != Path(gate['root'])/'gate_spec.json':
        raise ValueError('Original gate location differs')
    validate_source_snapshot(gate['source_snapshot'], repository=gate['project_dir'], require_clean=True)
    matched_runtime(project, gate['source_commit'])
    if (gate['execution_site'] != execution_site('sporc_a100_debug')
            or (gate['cpus'], gate['memory_mb'], gate['science_ceiling_minutes']) != (6, 90000, 1440)):
        raise ValueError('Only the registered same-resource SPORC debug transfer is allowed')
    print('Authenticating completed debug ladder preflight in its original source...', flush=True)
    ref, report = source.replay(path, gate['project_dir'], REPLAY, 'LADDER_TIER3_ACCEPTED=')
    report_path = Path(gate['root'])/'preflight/receipt.json'
    if load_json(report_path) != report:
        raise ValueError('Accepted preflight changed during replay')
    return gate, report, ref, file_ref(report_path)


def fields(gate, report, arm, project, commit, snapshot, gate_ref, report_ref):
    if arm not in contracts.ARMS:
        raise ValueError('Choose EMBED or LOGIT explicitly')
    return dict(root=str(Path(gate['root'])/'science_tier3'/arm),
        gate_path=str(Path(gate['root'])/'gate_spec.json'), gate_ref=gate_ref, preflight_ref=report_ref,
        arm=arm, project_dir=str(Path(project).resolve()), source_commit=commit, source_snapshot=snapshot,
        transfer_policy=DEBUG_PROFILE_TRANSFER, execution_site=execution_site('sporc_a100'),
        cpus=gate['cpus'], memory_mb=gate['memory_mb'], train_minutes=report['train_minutes'][arm])


def debug_conflict(gate, arm):
    path = Path(gate['root'])/'science'/arm
    if path.exists() or path.is_symlink():
        raise FileExistsError('Original debug science already prepared for this arm; inspect it before transferring')


def canonical_root(root):
    root = Path(root)
    if root.is_symlink() or root.resolve() != root:
        raise PermissionError('Tier3 output must not be redirected through a symlink')


def create(gate_path, arm, project, commit):
    snapshot = current(project, commit)
    gate, report, ref, report_ref = authenticate(gate_path, project)
    debug_conflict(gate, arm)
    spec = artifact('SPEC', parents=dict(gate=gate['content_hash'], preflight=report['content_hash'],
        source=snapshot['content_hash']), **fields(gate, report, arm, project, commit, snapshot, ref, report_ref))
    root = Path(spec['root'])
    canonical_root(root)
    if root.exists():
        if load_json(root/'campaign_spec.json') != spec:
            raise FileExistsError('Existing tier3 arm differs; preserve it')
        validate_spec(spec, gate, report)
        return spec
    root.mkdir(parents=True, exist_ok=False)
    write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def validate_spec(spec, gate, report):
    validate(spec, 'SPEC', dict(gate=gate['content_hash'], preflight=report['content_hash'],
                               source=spec['source_snapshot']['content_hash']))
    for ref in (spec['gate_ref'], spec['preflight_ref']):
        validate_file_ref(ref)
    expected = fields(gate, report, spec['arm'], spec['project_dir'], spec['source_commit'],
        spec['source_snapshot'], file_ref(Path(gate['root'])/'gate_spec.json'),
        file_ref(Path(gate['root'])/'preflight/receipt.json'))
    if (any(spec.get(k) != v for k, v in expected.items())
            or spec['source_snapshot']['git_commit'] != spec['source_commit']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec):
        raise ValueError('Tier3 science registration differs')
    if type(spec['train_minutes']) is not int or not 1 <= spec['train_minutes'] <= 1440:
        raise ValueError('Measured walltime exceeds unchanged 24h ceiling')
    canonical_root(spec['root'])
    debug_conflict(gate, spec['arm'])


def check_spec(spec):
    if current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Transfer execution source changed')
    gate, report, _, _ = authenticate(spec['gate_path'], spec['project_dir'])
    validate_spec(spec, gate, report)
    return gate, report


def render(spec):
    validate(spec, 'SPEC')
    site, root = spec['execution_site'], Path(spec['root'])
    if site != execution_site('sporc_a100'):
        raise ValueError('Transfer plan must target tier3')
    minutes = spec['train_minutes']
    if type(minutes) is not int or not 1 <= minutes <= 1440:
        raise ValueError('Measured walltime exceeds unchanged 24h ceiling')
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--export=NONE',
        '--partition='+site['partition'], '--account='+site['account'], '--qos='+site['qos'],
        '--cpus-per-task='+str(spec['cpus']), '--mem='+str(spec['memory_mb'])+'M',
        '--time='+_walltime(minutes), '--gres='+site['gres'],
        '--job-name=lukaelt_'+spec['arm'].lower()+'_ladder',
        '--comment=lukaelt:'+spec['content_hash']+':ladder', '--chdir='+spec['project_dir'],
        '--output='+str(root/'ladder-%j.out'),
        spec['project_dir']+'/sbatch/run_luka_embedding_ladder_tier3.sh', spec['project_dir'],
        str(root/'campaign_spec.json'), 'ladder']
    return artifact('PLAN', parents=dict(spec=spec['content_hash']), jobs=1, mode='science',
        authorization=AUTHORIZATION, commands=[dict(task_id='ladder', dependencies=[], command=command)])


def submit(spec, *, execute=False, plan_hash=None, authorization=None):
    check_spec(spec)
    plan = render(spec)
    if execute and (plan_hash != plan['content_hash'] or authorization != AUTHORIZATION):
        raise PermissionError('Exact reviewed tier3 plan hash and authorization required')
    root = Path(spec['root'])/'submission'
    root.mkdir(exist_ok=True)
    saved, dry, ledger = root/'command_plan.json', root/'dry_run_submission_ledger.json', root/'submission_ledger.json'
    if execute and (not saved.is_file() or load_json(saved) != plan):
        raise ValueError('Unchanged saved tier3 dry review required')
    write_immutable_json(saved, plan)
    check_site(plan)
    if execute:
        expected = build_submission_ledger(campaign_spec_sha256=spec['content_hash'],
            jobs={'ladder': '1'}, commands={'ladder': plan['commands'][0]['command']}, dry_run=True)
        if not dry.is_file() or load_json(dry) != expected:
            raise ValueError('Canonical dry ledger required')
        if not ledger.is_file():
            (root/'live_claim').mkdir(exist_ok=False)
            write_immutable_json(root/'live_claim/intent.json', artifact('SUBMISSION_CLAIM',
                parents=dict(spec=spec['content_hash'], plan=plan['content_hash'])))
    result = submit_exact_dag(identity=spec['content_hash'], plan=plan, output=ledger if execute else dry,
        canonical_dry_run=dry, execute=execute, environment=clean_environment())
    return result if execute else plan


def run(spec):
    gate, measured = check_spec(spec)
    # This allocation-only mapping is not a new gate artifact and is never saved.
    # All scientific receipts continue to bind the untouched measured gate.
    job = worker.execution(dict(gate, execution_site=spec['execution_site']), measured)
    root = Path(spec['root'])
    if (root/'ladder').exists() or (root/'receipt.json').exists():
        raise FileExistsError('Preserve partial/completed ladder; no automatic restart')
    worker.storage(gate)
    log = worker.PhaseLog(root, lambda:worker.memory_snapshot(torch))
    with log.phase('load_views_and_fixed_OFFLINE_targets'):
        meta = worker.regular_cache.metadata(gate['anchor']['evidence']['prepared_root'])
        train = worker.data.load_views(gate, 'train', authenticated=meta)
        val = worker.data.load_views(gate, 'validation', authenticated=meta)
        embeddings = worker.data.embedding_bank(gate, train['D000'])
        if embeddings.manifest['content_hash'] != measured['embedding_hash']:
            raise ValueError('Measured embedding target changed')
        incoming = worker.anchor_bank(gate, train['U000'], device='cuda', technical=False)
        write_immutable_json(root/'anchor_logits.json', incoming.manifest)
    arrays = {}
    for node in gate['nodes'][spec['arm']]:
        co = node['coordinate']
        with log.phase('fit_'+node['node_id']):
            directory = root/'ladder'/node['node_id']
            previous = incoming.manifest
            _, incoming = worker.fit(directory, gate, node, train[co], val[co], incoming,
                                     embeddings, device='cuda', technical=False)
            _, a, _ = worker.validate_step(directory, gate, node, previous,
                                           embeddings.manifest['content_hash'], technical=False)
            arrays[node['node_id']] = a
            worker.storage(gate)
    value = artifact('RESULTS', parents=dict(spec=spec['content_hash']),
                     rows=worker.comparison(gate, arrays), counts=gate['counts'])
    write_immutable_json(root/'results.json', value)
    if current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Source changed during tier3 science')
    validate_source_snapshot(gate['source_snapshot'], repository=gate['project_dir'], require_clean=True)
    outputs = [file_ref(p, root=root) for p in sorted(root.rglob('*')) if p.is_file()
        and 'submission' not in p.relative_to(root).parts and not p.name.endswith('.out') and p.name != 'campaign_spec.json']
    write_immutable_json(root/'receipt.json', artifact('SCIENCE',
        parents=dict(spec=spec['content_hash']), job_id=job, outputs=outputs))
    return value


def results(spec):
    gate = load_json(spec['gate_path'])
    measured = load_json(Path(gate['root'])/'preflight/receipt.json')
    from .preflight import validate_report
    validate_report(gate, measured)
    validate_spec(spec, gate, measured)
    root = Path(spec['root'])
    receipt = load_json(root/'receipt.json')
    validate(receipt, 'SCIENCE', dict(spec=spec['content_hash']))
    expected = {'results.json', 'anchor_logits.json'}
    for node in gate['nodes'][spec['arm']]:
        expected |= {'ladder/'+node['node_id']+'/'+f for f in
                     ('receipt.json', 'training_report.json', 'validation_predictions.npz')}
    names = [r['path'] for r in receipt['outputs']]
    if len(names) != len(set(names)) or not expected <= set(names):
        raise ValueError('Incomplete tier3 science receipt coverage')
    for ref in receipt['outputs']:
        validate_file_ref(ref, root=root)
    incoming = load_json(root/'anchor_logits.json')
    if incoming['parents']['checkpoint'] != gate['anchor']['evidence']['models']['U000']['checkpoint']['sha256']:
        raise ValueError('Starting U000 checkpoint differs')
    arrays = {}
    for node in gate['nodes'][spec['arm']]:
        _, a, incoming = worker.validate_step(root/'ladder'/node['node_id'], gate, node, incoming,
                                             measured['embedding_hash'], technical=False)
        arrays[node['node_id']] = a
    value = load_json(root/'results.json')
    validate(value, 'RESULTS', dict(spec=spec['content_hash']))
    if value['counts'] != gate['counts'] or value['rows'] != worker.comparison(gate, arrays):
        raise ValueError('Tier3 comparison replay differs')
    return value
