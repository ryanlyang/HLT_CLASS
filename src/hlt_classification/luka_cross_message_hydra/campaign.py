"""Additive source pin and immutable local Hydra campaign registration."""
from pathlib import Path
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullsim.foundation import _source
from hlt_classification.luka_u100_pair_hydra import portability, site
from hlt_classification.luka_u100_contrastive_hydra.campaign import storage
from .contracts import BASE, ARMS, TEACHERS, COUNTS, SITE, MANIFEST_HASH, recipe, artifact, validate
from . import inputs

MODULES=('__init__','contracts','model','inputs','training','campaign','worker','resources','submission','progress')
RUNTIME={'src/hlt_classification/luka_cross_message_hydra/'+n+'.py' for n in MODULES} | {
    'scripts/luka_cross_message_hydra.py','scripts/queue_luka_cross_message_hydra.sh',
    'sbatch/run_luka_cross_message_hydra.sh'}
REQUIRED=RUNTIME | {'tests/test_luka_cross_message_hydra.py',
    'docs/plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md','docs/contracts/LUKA_CROSS_MESSAGE_HYDRA.md',
    'docs/LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md'}


def current(project,commit):
    project=Path(project).resolve(); snapshot=_source(project,commit)
    def git(*args):
        return subprocess.run(['git','-C',str(project),*args],check=True,capture_output=True,text=True).stdout.splitlines()
    git('merge-base','--is-ancestor',BASE,commit)
    changed=git('diff','--name-status','--no-renames',BASE,commit,'--','src','scripts','sbatch')
    if (not REQUIRED <= set(git('ls-files')) or any(r[:2] not in ('A\t','M\t') or r[2:] not in RUNTIME for r in changed)
            or Path(__file__).resolve()!=project/'src/hlt_classification/luka_cross_message_hydra/campaign.py'):
        raise ValueError('Commit the complete additive message implementation; preserve donor runtime')
    return snapshot


def registration():
    return dict(recipe=recipe(),arms=ARMS,teachers=list(TEACHERS),counts=COUNTS,original50k_rows=50000,
        execution_site=SITE,cpus=6,memory_mb=120000,storage_budget_bytes=8*2**30,
        setup_minutes=720,fit_max_minutes=1440,report_minutes=120,standalone_preflight=False,
        max_concurrent_fits=None,automatic_retry=False,reference_retrained=False)


def create(root,bundle_root,delta_root,delta_hash,precision_spec,project,commit,quota_gib,encoder_root,encoder_hash):
    root=Path(root).absolute(); project=Path(project).resolve()
    b=inputs.Bundle(bundle_root); d=inputs.Delta(delta_root,delta_hash,b); d.verify_all()
    e=inputs.EncoderDelta(encoder_root,encoder_hash,b); e.verify_all()
    old=b.json('origin/campaign_spec.json')
    if type(quota_gib) is not int or quota_gib<8: raise ValueError('Reserve at least eight GiB')
    protected=(b.root,d.root,project,Path(sys.prefix).resolve(),Path(precision_spec).resolve().parent,
        e.root)
    if (root.exists() or root.resolve()!=root or not root.parent.is_dir()
            or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate output directory')
    site.environment(old['reference']['environment']); snapshot=current(project,commit)
    evidence=portability.authenticate(precision_spec,b.root,old['reference'],COUNTS)
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],transfer=MANIFEST_HASH,
        delta=delta_hash,portability=evidence['content_hash'],encoder=encoder_hash),**registration(),root=str(root),project_dir=str(project),
        source_commit=commit,source_snapshot=snapshot,environment_dir=str(Path(sys.prefix).resolve()),
        bundle_root=str(b.root),delta_root=str(d.root),delta_hash=delta_hash,available_quota_gib=quota_gib,
        reference=old['reference'],imported=old['imported'],anchor=old['anchor'],portability_evidence=evidence,
        encoder_root=str(e.root),encoder_hash=encoder_hash)
    storage(spec); root.mkdir(); write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,check_storage=True):
    d=inputs.delta(spec); b=inputs.Bundle(spec['bundle_root']); old=b.json('origin/campaign_spec.json')
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],transfer=MANIFEST_HASH,
        delta=d.manifest['content_hash'],portability=spec['portability_evidence']['content_hash'],
        encoder=inputs.encoder_delta(spec).manifest['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration().items()) or spec['available_quota_gib']<8
            or any(spec[k]!=old[k] for k in ('reference','imported','anchor'))
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec
            or current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])):
        raise ValueError('Source or immutable registration differs')
    site.environment(spec['reference']['environment'])
    portability.validate_evidence(spec['portability_evidence'],b.root,spec['reference'],COUNTS)
    if check_storage: storage(spec)
    return spec
