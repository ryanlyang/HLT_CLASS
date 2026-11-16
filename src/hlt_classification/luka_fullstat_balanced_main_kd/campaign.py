"""Register one fresh DE000, preserving the original sampling/teacher population."""
from pathlib import Path
import shutil
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.luka_fullstat_balanced_ladder.campaign import root_check
from hlt_classification.jetclass2_delphes.execution import execution_site
from . import contracts as c,source

DEFAULT_PARENT='/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_balanced_ff22b20f_r1/campaign_spec.json'
MIN_QUOTA_GIB=8


def resources():
    return dict(cpus=6,memory_mb=192000,minutes={'train_DE000':2880},preparation_workers=4,
                candidate_workers=3,result_budget_bytes=6*2**30)


def create(root,parent,project,commit,quota_gib):
    if type(quota_gib)!=int or quota_gib<MIN_QUOTA_GIB:raise ValueError('At least 8 GiB available HOME quota required')
    src=source.current(project,commit);reuse=source.authenticate(parent,project)
    old=source.check(reuse,project,source_check=False)
    protected=[old['root'],old['project_dir'],old['runtime_prefix'],project,old['population_root'],
               *(old['inputs']['evidence'][k] for k in ('raw_container','cache_root','foundation_root'))]
    root=root_check(root,protected)
    if shutil.disk_usage(root.parent).free<MIN_QUOTA_GIB*2**30:raise OSError('Insufficient free result storage')
    spec=c.artifact('SPEC',root=str(root),project_dir=str(Path(project).resolve()),source_commit=commit,source=src,
        reuse=reuse,inputs=old['inputs'],population=old['population'],population_root=old['population_root'],
        recipe=c.recipe(),nodes={'DE000':c.node('DE000')},counts=dict(c.COUNTS),eligible_train_rows=c.ELIGIBLE,
        execution_site=execution_site('oscar_l40s'),runtime_prefix=source.RUNTIME,
        available_result_bytes=quota_gib*2**30,**resources(),tasks=c.tasks(),completed={},continuations={},recovery=None)
    check(spec);root.mkdir(parents=True,exist_ok=False);meta.write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,live_source=True):
    c.validate(spec,'SPEC')
    old=source.check(spec['reuse'],spec['project_dir'],source_check=live_source)
    if (spec['recipe']!=c.recipe() or spec['nodes']!={'DE000':c.node('DE000')}
            or spec['counts']!=c.COUNTS or spec['eligible_train_rows']!=c.ELIGIBLE
            or spec['execution_site']!=execution_site('oscar_l40s') or spec['runtime_prefix']!=source.RUNTIME
            or spec['available_result_bytes']<MIN_QUOTA_GIB*2**30
            or any(spec[k]!=v for k,v in resources().items()) or spec['tasks']!=c.tasks()
            or spec['completed'] or set(spec['continuations'])-{'DE000'}):
        raise ValueError('Main-head KD registration differs')
    for key in ('inputs','population','population_root'):
        if spec[key]!=old[key]:raise ValueError('Changed balanced population: '+key)
    if live_source and source.current(spec['project_dir'],spec['source_commit'])!=spec['source']:
        raise ValueError('Registered source differs')
    if spec['recovery'] is None:
        if spec['continuations']:raise ValueError('Unregistered continuation')
    else:
        from .recovery import validate_registration
        validate_registration(spec)
    return spec
