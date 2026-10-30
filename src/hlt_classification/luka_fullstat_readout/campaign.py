"""One fresh fit reusing verified original U100-w5 artifacts read-only."""
from pathlib import Path
import shutil
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.luka_fullstat_multiteacher.campaign import protected
from . import source
from .contracts import artifact, validate, recipe, node, COUNTS, teacher_replay


def registration(reuse):
    validate(reuse, 'REUSE')
    teacher = reuse['teacher_node']
    if (reuse['counts'] != COUNTS or reuse['source_commit'] != source.DONOR
            or teacher['node_id'] != 'COARSE_U100' or teacher['coordinate'] != 'U100'
            or teacher.get('context_coordinate') is not None
            or reuse['execution_site'] != execution_site('oscar_l40s')
            or 'L40S' not in reuse['gpu']['name']
            or set(reuse['jobs']) != {'export_U100', 'train_w5'}):
        raise ValueError('Wrong FullStat U100 teacher, population or Oscar evidence')
    # Includes a temporary teacher TRAIN view during integrated parity checks.
    summaries = reuse['summaries']
    rows = sum(summaries[r][c]['resident_bytes'] for r,c in
               (('train','D000'),('validation','D000'),('train','U100')))
    if 2*rows + COUNTS['train']*128*4*2 + 16*2**30 >= .85*180000*1024**2:
        raise MemoryError('Readout input/bank staging exceeds inherited memory envelope')
    references = {n: reuse['anchor']['evidence']['models'][n] for n in ('M0HLT','OFFLINE')}
    references['U100_W5'] = reuse['reference']
    return dict(counts=dict(COUNTS), node=node(reuse['base_node']), recipe=recipe(),
        teacher_replay=teacher_replay(),
        anchor=reuse['anchor'], references=references, bank_manifest=reuse['bank_manifest'],
        teacher_refs=reuse['teacher_refs'], runtime_prefix=source.RUNTIME,
        execution_site=reuse['execution_site'], cpus=6, memory_mb=180000,
        minutes={'train_readout':1440}, storage_budget_bytes=4*2**30,
        preflight_jobs=0, setup_jobs=0, scientific_fits=1, automatic_retry=False,
        historical_control=True)


def storage(spec):
    root = Path(spec['root']); existing = root if root.exists() else root.parent
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    remaining = spec['storage_budget_bytes']-used
    if remaining <= 0 or shutil.disk_usage(existing).free < remaining:
        raise OSError('Registered 4 GiB result allowance unavailable; nothing deleted')


def validate_spec(spec):
    validate(spec, 'SPEC', dict(source=spec['source_snapshot']['content_hash'], reuse=spec['reuse']['content_hash']))
    if (any(spec.get(k) != v for k,v in registration(spec['reuse']).items())
            or type(spec['available_quota_gib']) is not int or spec['available_quota_gib'] < 4):
        raise ValueError('Registered readout science/resources differ')
    root = Path(spec['root'])
    if root.resolve() != root or root.is_symlink() or not root.name.startswith('luka_fullstat_readout_'):
        raise PermissionError('Dedicated unredirected readout result root required')
    protected(root, spec['reuse'], spec['project_dir'])


def create(root, previous_spec, project, commit, *, quota_gib):
    root, project = Path(root).absolute(), Path(project).resolve()
    active = source.current(project, commit); reuse = source.authenticate(previous_spec, project)
    spec = artifact('SPEC', parents=dict(source=active['content_hash'], reuse=reuse['content_hash']),
        root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=active,
        reuse=reuse, available_quota_gib=quota_gib, **registration(reuse))
    validate_spec(spec); storage(spec)
    root.mkdir(exist_ok=False); write_immutable_json(root/'campaign_spec.json', spec)
    return spec


def check(spec):
    validate_spec(spec)
    if (load_json(Path(spec['root'])/'campaign_spec.json') != spec
            or source.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']
            or source.authenticate(spec['reuse']['parent_spec']['path'], spec['project_dir']) != spec['reuse']):
        raise ValueError('Pinned source, parent or registration changed')
    validate_file_ref(spec['bank_manifest']); storage(spec)
    return spec
