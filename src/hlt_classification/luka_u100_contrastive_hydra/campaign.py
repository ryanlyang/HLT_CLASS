"""Fresh Hydra execution; original results and source remain immutable."""
from pathlib import Path
import shutil
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_pair_hydra import portability, site
from .contracts import artifact, validate, BASE, ARMS, COUNTS, SITE, recipe, MANIFEST_HASH
from . import inputs

MODULES = ('__init__', 'contracts', 'transfer', 'inputs', 'campaign', 'submission', 'worker')
RUNTIME = {'src/hlt_classification/luka_u100_contrastive_hydra/'+n+'.py' for n in MODULES} | {
    'scripts/luka_u100_contrastive_hydra.py', 'scripts/queue_luka_u100_contrastive_hydra.sh',
    'sbatch/run_luka_u100_contrastive_hydra.sh'}
REQUIRED = RUNTIME | {'tests/test_luka_u100_contrastive_hydra.py',
    'docs/plans/LUKA_U100_CONTRASTIVE_HYDRA_PLAN.md', 'docs/contracts/LUKA_U100_CONTRASTIVE_HYDRA.md',
    'docs/LUKA_U100_CONTRASTIVE_HYDRA_RUNBOOK.md'}


def current(project, commit):
    project = Path(project).resolve(); snapshot = _source(project, commit)
    def git(*args):
        return subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True, text=True).stdout.splitlines()
    git('merge-base', '--is-ancestor', BASE, commit)
    changed = git('diff', '--name-status', '--no-renames', BASE, commit, '--', 'src', 'scripts', 'sbatch')
    if (not REQUIRED <= set(git('ls-files')) or any(r[:2] not in ('A\t', 'M\t') or r[2:] not in RUNTIME for r in changed)
            or Path(__file__).resolve() != project/'src/hlt_classification/luka_u100_contrastive_hydra/campaign.py'):
        raise ValueError('Commit complete additive Hydra adapter; preserve existing science')
    return snapshot


def registration():
    return dict(recipe=recipe(), arms=ARMS, counts=COUNTS, original50k_rows=50000,
        execution_site=SITE, cpus=6, memory_mb=120000, storage_budget_bytes=8*2**30,
        setup_minutes=720, fit_max_minutes=1440, report_minutes=120,
        student_dependencies='none', max_concurrent_fits=None, automatic_retry=False,
        reference_retrained=False, standalone_preflight=False, stopping_policy='original_full30')


def storage(spec):
    root = Path(spec['root']); used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(root if root.exists() else root.parent).free < remaining+2**30:
        raise OSError('Need remaining eight-GiB output allowance plus one GiB free headroom')


def create(root, bundle_root, delta_root, delta_hash, precision_spec, project, commit, quota_gib):
    root = Path(root).absolute(); project = Path(project).resolve()
    b = inputs.Bundle(bundle_root); d = inputs.Delta(delta_root, delta_hash); d.verify_all()
    old = d.manifest['origin_spec']; pair_origin = b.json('origin/campaign_spec.json')
    if type(quota_gib) is not int or quota_gib < 8: raise ValueError('Reserve at least eight GiB of output space')
    protected = (b.root, d.root, project, Path(sys.prefix).resolve(), Path(precision_spec).resolve().parent)
    if (root.exists() or root.resolve() != root or not root.parent.is_dir()
            or any(p.is_symlink() for p in (root, *root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate campaign directory')
    if 'u100_w5_s0' not in d.manifest['completed']:
        raise ValueError('Transfer must include the completed matched seed0 control')
    reference = pair_origin['reference']; site.environment(reference['environment'])
    snapshot = current(project, commit)
    evidence = portability.authenticate(precision_spec, b.root, reference, COUNTS)
    reused = list(d.manifest['completed']); remaining = [a for a in ARMS if a not in reused]
    if not remaining: raise ValueError('All twenty original fits are already complete')
    spec = artifact('SPEC', parents=dict(source=snapshot['content_hash'], delta=d.manifest['content_hash'],
        transfer=MANIFEST_HASH, portability=evidence['content_hash']), **registration(),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=snapshot,
        environment_dir=str(Path(sys.prefix).resolve()), bundle_root=str(b.root), delta_root=str(d.root),
        delta_hash=delta_hash, reference=reference, imported=old['imported'], anchor=old['anchor'],
        recovery_anchors=old['recovery_anchors'], portability_evidence=evidence,
        available_quota_gib=quota_gib, reused=reused, remaining=remaining)
    inputs.bind(spec, d); storage(spec)
    root.mkdir(); write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec, *, check_storage=True):
    d = inputs.delta(spec)
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], delta=d.manifest['content_hash'],
        transfer=MANIFEST_HASH, portability=spec['portability_evidence']['content_hash']))
    old = d.manifest['origin_spec']
    if (any(spec.get(k) != v for k, v in registration().items()) or spec['available_quota_gib'] < 8
            or spec['reused'] != d.manifest['completed']
            or spec['remaining'] != [a for a in ARMS if a not in spec['reused']]
            or spec['recovery_anchors'] != old['recovery_anchors']
            or load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or Path(sys.prefix).resolve() != Path(spec['environment_dir'])):
        raise ValueError('Immutable contrastive Hydra registration/source differs')
    b = inputs.Bundle(spec['bundle_root'])
    if spec['reference'] != b.json('origin/campaign_spec.json')['reference']:
        raise ValueError('Matched reference differs')
    inputs.bind(spec, d); site.environment(spec['reference']['environment'])
    portability.validate_evidence(spec['portability_evidence'], b.root, spec['reference'], COUNTS)
    if check_storage: storage(spec)
    return spec
