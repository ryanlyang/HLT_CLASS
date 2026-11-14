"""Registration of six fresh fits sharing one frozen balanced population."""
from pathlib import Path
from hlt_classification.operations import fullstat_pt_metadata as meta
from hlt_classification.jetclass2_delphes.execution import execution_site
from . import contracts as c, source, population

DEFAULT_PARENT='/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json'
MIN_QUOTA_GIB=36


def resources():
    return dict(cpus=6,memory_mb=192000,minutes={f'train_{s}':2880 for s in c.STAGES},
                preparation_workers=4,candidate_workers=3,result_budget_bytes=32*2**30)


def root_check(root,protected):
    root=Path(root).absolute()
    if root!=root.resolve() or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('Output path must be canonical and not a symlink')
    for path in protected:
        p=Path(path).resolve()
        if root.is_relative_to(p) or p.is_relative_to(root):raise ValueError('Output overlaps protected data/source/result')
    if root.exists():raise FileExistsError('Campaign root exists; use explicit recovery, never relaunch')
    return root


def create(root,parent,project,commit,quota_gib):
    if type(quota_gib)!=int or quota_gib<MIN_QUOTA_GIB:raise ValueError('At least 36 GiB available HOME quota required')
    src=source.current(project,commit);inputs,prepared=source.authenticate(parent)
    protected=[project,str(Path(parent).parent),*(inputs['evidence'][k] for k in ('raw_container','cache_root','foundation_root'))]
    root=root_check(root,protected);root.mkdir(parents=True,exist_ok=False)
    pop=population.prepare(root/'population',prepared,input_hash=inputs['content_hash'])
    spec=c.artifact('SPEC',root=str(root),project_dir=str(Path(project).resolve()),source_commit=commit,source=src,
        inputs=inputs,population=pop,population_root=str(root/'population'),recipe=c.recipe(),
        nodes={s:c.node(s) for s in c.STAGES},counts=dict(c.COUNTS),eligible_train_rows=c.ELIGIBLE,
        execution_site=execution_site('oscar_l40s'),runtime_prefix=source.RUNTIME,
        available_result_bytes=quota_gib*2**30,**resources(),
        tasks=c.tasks(),completed={},continuations={},recovery=None)
    check(spec);meta.write_immutable_json(root/'campaign_spec.json',spec)
    return spec


def check(spec,*,live_source=True):
    c.validate(spec,'SPEC');c.validate(spec['population'],'POPULATION');c.validate(spec['inputs'],'INPUTS')
    if (spec['recipe']!=c.recipe() or spec['nodes']!={s:c.node(s) for s in c.STAGES}
            or spec['counts']!=c.COUNTS or spec['eligible_train_rows']!=c.ELIGIBLE
            or spec['execution_site']!=execution_site('oscar_l40s') or spec['runtime_prefix']!=source.RUNTIME
            or spec['available_result_bytes']<MIN_QUOTA_GIB*2**30
            or any(spec[k]!=v for k,v in resources().items())):
        raise ValueError('Balanced scientific or resource registration differs')
    if spec['population']['parents']['inputs']!=spec['inputs']['content_hash']:
        raise ValueError('Population input binding differs')
    if live_source and source.current(spec['project_dir'],spec['source_commit'])!=spec['source']:
        raise ValueError('Registered source differs')
    completed=set(spec['completed'])
    if not completed<=set(c.STAGES) or not set(spec['continuations'])<=set(c.STAGES)-completed:
        raise ValueError('Recovery stage inventory differs')
    expected={k:[d for d in deps if d.removeprefix('train_') not in completed]
              for k,deps in c.tasks().items() if k.removeprefix('train_') not in completed}
    if spec['tasks']!=expected:raise ValueError('Exact remaining DAG differs')
    if spec['recovery'] is None and (completed or spec['continuations']):raise ValueError('Unregistered historical reuse')
    if spec['recovery'] is not None:
        from .recovery import validate_registration
        validate_registration(spec)
    return spec
