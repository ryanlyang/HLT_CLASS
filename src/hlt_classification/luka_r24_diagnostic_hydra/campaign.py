"""Pin both the completed R24 science and its four-card benchmark without editing either."""
import importlib
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import capture_source_snapshot
from hlt_classification.jetclass2_delphes.model import installed_environment
from hlt_classification.luka_next_supervision_hydra import campaign as original_campaign
from hlt_classification.luka_u100_pair_hydra.inputs import Bundle
from .contracts import DONOR, BENCHMARK, FITS, LANES, artifact, validate, recipe

MODULES = ('__init__', 'contracts', 'campaign', 'kernel', 'training', 'resources', 'worker', 'submission', 'reporting', 'identity', 'original')
REQUIRED = tuple(f'src/hlt_classification/luka_r24_diagnostic_hydra/{n}.py' for n in MODULES) + (
    'scripts/luka_r24_diagnostic_hydra.py', 'scripts/queue_luka_r24_diagnostic_hydra.sh',
    'sbatch/run_luka_r24_diagnostic_hydra.sh', 'tests/test_luka_r24_diagnostic_hydra.py',
    'docs/plans/LUKA_R24_DIAGNOSTIC_HYDRA_PLAN.md', 'docs/contracts/LUKA_R24_DIAGNOSTIC_HYDRA.md',
    'docs/LUKA_R24_DIAGNOSTIC_HYDRA_RUNBOOK.md', 'docs/LUKA_R24_DIAGNOSTIC_HYDRA_VALIDATION.md')


def hardware(lane):
    if lane not in LANES: raise ValueError('Unknown lane')
    pascal=lane=='pascal'
    return dict(world_size=4 if pascal else 1,cpus=12 if pascal else 6,memory_mb=64000 if pascal else 32000,
        execution_site=dict(cluster='hydra',partition='gpus',account='default',qos='normal',
            gres='gpu:geforce_gtx_1080_ti:4' if pascal else 'gpu:nvidia_rtx_a6000:1'))


def profile(spec,lane): return dict(spec,**hardware(lane))


def registration():
    return dict(recipe=recipe(),tasks=list(FITS),global_batch=128,hardware={l:hardware(l) for l in LANES},
        cpu_site=dict(partition='compute',account='default',qos='normal',cpus=1,memory_mb=4000),
        setup_minutes=360,fit_max_minutes=2880,release_minutes=60,report_minutes=60,
        output_budget_bytes=9*2**30,automatic_retry=False,science_dependencies='none',
        resource_policy=dict(safety_multiplier=2.,fixed_overhead_seconds=1800.,host_fraction=.85,gpu_fraction=.88))


def source(project, commit):
    project=Path(project).resolve()
    if not re.fullmatch('[0-9a-f]{40}',commit) or Path(__file__).resolve()!=project/REQUIRED[2]:
        raise ValueError('Use the diagnostic driver from its exact pushed worktree')
    snap=capture_source_snapshot(project,require_clean=True)
    names=subprocess.check_output(['git','-C',str(project),'ls-files'],text=True).splitlines()
    if snap['git_commit']!=commit or not set(REQUIRED)<=set(names): raise ValueError('Commit the complete diagnostic implementation')
    subprocess.run(['git','-C',str(project),'merge-base','--is-ancestor',commit,'origin/main'],check=True,capture_output=True)
    return snap


def benchmark(path, *, full):
    b=load_json(path)
    if b.get('source_commit')!=BENCHMARK: raise ValueError('Use the completed 1094dea4 benchmark')
    name='r24_diagnostic_benchmark_evidence'
    folder=Path(b['project_dir'])/'src/hlt_classification/luka_reconstruction_benchmark'
    if name not in sys.modules:
        loader=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(loader);sys.modules[name]=module;loader.loader.exec_module(module)
    bc=importlib.import_module(name+'.campaign');old=bc.check(b,full=full)
    if full: importlib.import_module(name+'.runner').results(b)
    report=load_json(Path(b['root'])/'benchmark_report.json');bc.validate(report,'REPORT',dict(spec=b['content_hash']))
    row=report['measurements'].get('4')
    if (not row or row['qualified'] is not True or report['reference_phase']['exit_code']!=0
            or not all(p['passed'] is True for p in row['parity'].values())
            or set(row['stress'])!={'partial11','tail','longest_hlt','largest_offline'}
            or not all(p['passed'] is True for p in row['stress'].values())):
        raise ValueError('Four-card benchmark qualification missing')
    return b,old,report


def storage(spec):
    root=Path(spec['root']);used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining=max(0,spec['output_budget_bytes']-used)
    if used>spec['output_budget_bytes'] or shutil.disk_usage(root if root.exists() else root.parent).free<remaining+2**30:
        raise OSError('Insufficient remaining diagnostic output allowance plus one GiB reserve')


def create(root,benchmark_spec,project,commit,quota_gib):
    b,old,report=benchmark(benchmark_spec,full=True);snap=source(project,commit)
    root=Path(root).absolute();project=Path(project).resolve()
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','raw_root','delta_root')]
    protected += [project,Path(b['root']).resolve(),Path(b['project_dir']).resolve()]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root
            or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate diagnostic output root')
    if type(quota_gib) is not int or quota_gib<10: raise ValueError('Reserve at least ten personally available GiB')
    refs=[file_ref(Path(b['root'])/n) for n in ('benchmark_report.json','receipt.json')]
    anchors=Bundle(old['bundle_root']).json('origin/campaign_spec.json')['recovery_anchors']
    spec=artifact('SPEC',parents=dict(original=old['content_hash'],benchmark=b['content_hash'],
        benchmark_report=report['content_hash'],source=snap['content_hash'],recipe=recipe()['content_hash']),
        **registration(),root=str(root),project_dir=str(project),source_commit=commit,source_snapshot=snap,
        benchmark_spec=file_ref(Path(benchmark_spec).resolve()),benchmark_outputs=refs,
        original_spec=b['original_spec'],environment_dir=old['environment_dir'],environment=old['reference']['environment'],
        counts=old['counts'],maximum=old['target_inventory']['maximum'],imported=old['imported'],
        quota_gib=quota_gib,recovery_anchors=anchors,reference=report['measurements']['4']['projection'],
        embedding_hash=Bundle(old['bundle_root']).json('targets/u100/manifest.json')['content_hash'],
        foundation_hash=old['anchor']['evidence']['prepared'])
    storage(spec);root.mkdir();write_immutable_json(root/'campaign_spec.json',spec);return spec


def check(spec, *, full=False):
    validate(spec,'SPEC')
    for ref in (spec['original_spec'],spec['benchmark_spec'],*spec['benchmark_outputs']):validate_file_ref(ref)
    old=load_json(spec['original_spec']['path']);b=load_json(spec['benchmark_spec']['path'])
    snap=source(spec['project_dir'],spec['source_commit'])
    report=load_json(spec['benchmark_outputs'][0]['path'])
    expected=dict(original=old['content_hash'],benchmark=b['content_hash'],benchmark_report=report['content_hash'],
                  source=snap['content_hash'],recipe=recipe()['content_hash'])
    if (old.get('source_commit')!=DONOR or b.get('source_commit')!=BENCHMARK or b['original_spec']!=spec['original_spec']
            or Path(original_campaign.__file__).resolve()!=Path(old['project_dir'])/'src/hlt_classification/luka_next_supervision_hydra/campaign.py'
            or capture_source_snapshot(old['project_dir'],require_clean=True)!=old['source_snapshot']
            or capture_source_snapshot(b['project_dir'],require_clean=True)!=b['source_snapshot']
            or any(spec.get(k)!=v for k,v in registration().items()) or spec['source_snapshot']!=snap or spec['parents']!=expected
            or spec['environment']!=old['reference']['environment'] or installed_environment()!=spec['environment']
            or spec['counts']!=old['counts'] or spec['maximum']!=old['target_inventory']['maximum']
            or spec['imported']!=old['imported'] or spec['environment_dir']!=old['environment_dir']
            or spec['benchmark_outputs']!=[file_ref(Path(b['root'])/n) for n in ('benchmark_report.json','receipt.json')]
            or spec['reference']!=report['measurements']['4']['projection']
            or spec['embedding_hash']!=Bundle(old['bundle_root']).json('targets/u100/manifest.json')['content_hash']
            or spec['foundation_hash']!=old['anchor']['evidence']['prepared'] or spec['quota_gib']<10
            or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or spec['recovery_anchors']!=Bundle(old['bundle_root']).json('origin/campaign_spec.json')['recovery_anchors']
            or load_json(Path(spec['root'])/'campaign_spec.json')!=spec):
        raise ValueError('Immutable diagnostic/source/environment identity differs')
    if full:benchmark(spec['benchmark_spec']['path'],full=True)
    storage(spec);return old
