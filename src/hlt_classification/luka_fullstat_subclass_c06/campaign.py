"""One new student, exact completed-parent evidence, no standalone preflight."""
from pathlib import Path
import os
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from hlt_classification.luka_fullstat_multiview.campaign import storage
from hlt_classification.luka_fullstat_readout.source import RUNTIME
from . import source
from .contracts import artifact, validate, recipe, node, TASK, TASKS, COUNTS


def registration(parent,reuse):
    if parent['counts'] != COUNTS or set(reuse['banks']) != {'U100','CONCAT','FUSION','OFFLINE'}:
        raise ValueError('Wrong parent population or bank inventory')
    return dict(counts=dict(COUNTS),recipe=recipe(),nodes={TASK:node(parent)},tasks=TASKS,
        anchor=parent['anchor'],references={**parent['references'], 'SUBCLASS_C03':reuse['reference']},
        execution_site=parent['execution_site'],runtime_prefix=RUNTIME,cpus=6,memory_mb=192000,
        minutes={TASK:2880},storage_budget_bytes=6*2**30,filesystem_headroom_bytes=2*2**30,
        preflight_jobs=0,setup_jobs=0,scientific_fits=1,teacher_fits=0,automatic_retry=False)


def validate_spec(spec):
    validate(spec,'SPEC',dict(source=spec['source_snapshot']['content_hash'],reuse=spec['reuse']['content_hash']))
    parent=load_json(spec['reuse']['parent_spec']['path']);validate(spec['reuse'],'REUSE',dict(spec=parent['content_hash']))
    if (any(spec.get(k)!=v for k,v in registration(parent,spec['reuse']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib']<8):
        raise ValueError('Campaign science/resources differ')
    root=Path(spec['root'])
    if root.resolve()!=root or root.is_symlink() or not root.name.startswith('luka_fullstat_subclass_c06_'):
        raise PermissionError('Dedicated unredirected result root required')
    protected(root,dict(anchor=spec['anchor'],protected_roots=[parent['root'],parent['project_dir'],
        *spec['reuse']['u100']['protected_roots']]),spec['project_dir'])
    from . import recovery
    recovery.check(spec)
    if spec.get('resume') is not None:
        previous=Path(spec['resume']['spec']['path']).parent.resolve()
        if root.is_relative_to(previous) or previous.is_relative_to(root):raise PermissionError('Recovery roots overlap')
    return parent


def check(spec):
    validate_spec(spec)
    if (source.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Pinned source/spec changed')
    source.check(spec['reuse']);storage(spec)
    return spec


def create(root,parent_path,project,commit,*,quota_gib,resume_spec=None):
    root,project=Path(root).absolute(),Path(project).resolve()
    if os.path.lexists(root):raise FileExistsError('Use a fresh study; preserve existing claims')
    snapshot=source.current(project,commit);reuse=source.authenticate(parent_path,project)
    parent=load_json(reuse['parent_spec']['path'])
    from . import recovery
    resume=None if resume_spec is None else recovery.inspect(resume_spec,project,commit)
    spec=artifact('SPEC',parents=dict(source=snapshot['content_hash'],reuse=reuse['content_hash']),
        root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snapshot,
        reuse=reuse,resume=resume,available_quota_gib=quota_gib,**registration(parent,reuse))
    validate_spec(spec);storage(spec);root.mkdir(exist_ok=False)
    write_immutable_json(root/'campaign_spec.json',spec)
    return spec
