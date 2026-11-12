"""Reuse completed preparation after the v2 frozen-U100 replay failure.

Only immutable array payloads are hard-linked. Manifests receive the new spec
lineage; no old qualification result, claim, or acceptance is carried forward.
"""
from pathlib import Path
import json
import math
import os
import subprocess
import sys

from hlt_classification.data.cache_contracts import load_json, validate_content_hash, write_immutable_json
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.provenance import validate_source_snapshot
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from .contracts import artifact, validate, recipe
from .registry import COORDINATES
from .storage import regular
from . import inputs

SOURCE = '9239773651179dc37f071156b30340d3e5bc3fba'


def layout():
    return [(f'{role}/{coord}', 'VIEW') for role in ('train','validation') for coord in COORDINATES] + [
        ('train/physical','PHYSICAL'),('validation/physical','PHYSICAL'),('u100_train','U100_BANK')]


def compatible(old,new):
    keys=('counts','recipe','nodes','bundle_root','transfer_root','transfer_hash','native_root','native_hash',
          'parent_spec','embedding_hash','anchor','reference','imported','environment_dir','native_input_policy')
    if any(old.get(k)!=new.get(k) for k in keys):
        raise ValueError('Preparation reuse has different scientific inputs or environment')


def source_spec(path):
    path=Path(path).resolve(strict=True);old=load_json(path)
    validate_content_hash(old,expected_contract='LUKA_EMBEDDING_BRIDGE_HYDRA_SPEC/v2',expected_schema_version=2)
    if (old['source_commit']!=SOURCE or path!=Path(old['root'])/'campaign_spec.json'
            or old['recipe']!=recipe() or Path(sys.prefix).resolve()!=Path(old['environment_dir'])):
        raise ValueError('Only the pinned v2 preparation is recoverable in its original environment')
    return old


def terminal_jobs(old, *, source=None):
    """Old runtime authenticates its own immutable plan/journal under old schemas."""
    if source is None:source=SOURCE
    if old['source_commit']!=source or old['source_snapshot']['git_commit']!=source:
        raise ValueError('Original preparation source changed')
    # Authenticate a foreign checkout's bytes without claiming this process
    # imported its modules. The old CLI below runs from that checkout and then
    # enforces its own module location, published ancestry and campaign guards.
    validate_source_snapshot(old['source_snapshot'],repository=old['project_dir'],require_clean=True)
    cli=Path(old['project_dir'])/'scripts/luka_embedding_bridge_hydra.py'
    result=subprocess.run([sys.executable,'-B','-s',str(cli),'inspect-submission','--spec',
        str(Path(old['root'])/'campaign_spec.json')],check=True,capture_output=True,text=True)
    jobs=json.loads(result.stdout)
    if jobs['science'] or set(jobs['bootstrap'])!={'setup','release'}:
        raise ValueError('Reuse requires setup-only execution with no scientific submissions')
    job=jobs['bootstrap']['setup']
    if not job.isdigit() or not jobs['bootstrap']['release'].isdigit():
        raise ValueError('Exact numeric original jobs required')
    result=subprocess.run(['sacct','-X','-n','-P','-j',job,'-S','2026-11-01',
        '-o','JobIDRaw,State,ExitCode'],check=True,capture_output=True,text=True)
    rows=[r.split('|')[1:3] for r in result.stdout.splitlines() if r.split('|')[0]==job]
    if rows!=[['FAILED','1:0']]:
        raise ValueError('Original setup is not confirmed FAILED with exit 1:0: '+str(rows))
    active=subprocess.run(['squeue','--me','--noheader','--format=%i'],check=True,capture_output=True,text=True)
    if set(active.stdout.split()) & set(jobs['bootstrap'].values()):
        raise ValueError('Original setup/release still active; inspect the exact job IDs before recovery')
    return job


def inspect(path):
    """Full hash authentication, no rewriting of source preparation."""
    print('EMBEDDING_BRIDGE recovery: authenticating original source and terminal jobs',flush=True)
    old=source_spec(path);job=terminal_jobs(old);root=Path(old['root'])/'setup'
    if any((root/name).exists() for name in ('receipt.json','acceptance.json','parity.json','data/u100_validation')) or list(root.glob('mini_*')):
        raise ValueError('Only the pre-training U100 replay failure is recoverable')
    preparation=load_json(root/'preparation.json');validate(preparation,'PREPARATION',dict(spec=old['content_hash']))
    failed=load_json(root/'u100_replay.json')
    validate_content_hash(failed,expected_contract='LUKA_EMBEDDING_BRIDGE_HYDRA_U100_REPLAY/v1',expected_schema_version=1)
    transfer=inputs.transferred(old)
    if (failed['parents']!=dict(spec=old['content_hash'],teacher=transfer.manifest['teacher']['sha256'],original=old['embedding_hash'])
            or failed['passed']is not False or failed['rows']!=4096 or (failed['atol'],failed['rtol'])!=(1e-4,1e-4)
            or not math.isfinite(failed['maximum_error']) or failed['maximum_error']<=0
            or not math.isfinite(preparation['seconds']) or preparation['seconds']<=0
            or not 0<preparation['host_peak_bytes']<old['memory_mb']*2**20*.85):
        raise ValueError('Original failure/preparation evidence differs')
    ds=inputs.Dataset(old)
    if preparation['data']!=ds.manifest['content_hash']:
        raise ValueError('Preparation completion does not bind these data')
    refs=[];manifests=[]
    for folder,kind in layout():
        if kind=='VIEW':
            role,coord=folder.split('/');store=ds.view(role,coord).store
            if coord in inputs.native.HISTORICAL and store.manifest['proof']!=transfer.manifest['summaries'][role][coord]:
                raise ValueError('Historical population proof differs')
        elif kind=='PHYSICAL':store=ds.physical(folder.split('/')[0])
        else:store=ds.u100('train')
        manifests.append(file_ref(root/'data'/folder/'manifest.json',root=root/'data'))
        refs.extend(dict(info['file'],path=folder+'/'+info['file']['path']) for info in store.manifest['arrays'].values())
        print('EMBEDDING_BRIDGE recovery verified '+folder,flush=True)
    # Validate U100 arrays against the original bank, not just the new containers.
    import numpy as np
    from hlt_classification.data.cache_contracts import load_npz_arrays
    original=load_npz_arrays(inputs.Bundle(old['bundle_root']).path('targets/u100/targets.npz'))
    if any(not np.array_equal(ds.u100('train')[k],v) for k,v in original.items()):
        raise ValueError('Copied U100 TRAIN targets/moments differ from the original bank')
    expected={r['path'] for r in refs+manifests}|{'manifest.json'}
    entries=list((root/'data').rglob('*'))
    if any(p.is_symlink() for p in entries):raise ValueError('Redirected preparation member')
    actual={p.relative_to(root/'data').as_posix() for p in entries if p.is_file()}
    if actual!=expected:
        raise ValueError('Preparation inventory has extra/missing/partial files')
    for name in expected:regular(root/'data',name)
    return artifact('PREPARATION_REUSE',parents=dict(original_spec=old['content_hash'],data=ds.manifest['content_hash']),
        source_spec=file_ref(Path(path).resolve()),source_commit=SOURCE,failed_setup_job=job,
        preparation=file_ref(root/'preparation.json'),failed_replay=file_ref(root/'u100_replay.json'),
        data=file_ref(root/'data/manifest.json'),manifests=manifests,arrays=refs,
        payload_bytes=sum(r['bytes'] for r in refs),method='immutable_array_hardlinks_new_manifests',
        old_qualification_reused=False)


def check(value,spec):
    """Cheap binding checks; Store readers still hash every consumed payload."""
    validate(value,'PREPARATION_REUSE');old=source_spec(validate_file_ref(value['source_spec']))
    compatible(old,spec)
    if (value['source_commit']!=SOURCE or value['parents']['original_spec']!=old['content_hash']
            or value['method']!='immutable_array_hardlinks_new_manifests' or value['old_qualification_reused']is not False
            or value['payload_bytes']!=sum(r['bytes'] for r in value['arrays'])
            or len({r['path'] for r in value['arrays']})!=len(value['arrays'])):
        raise ValueError('Preparation reuse binding differs')
    root=Path(old['root'])/'setup'
    for key,name in (('preparation','preparation.json'),('failed_replay','u100_replay.json'),('data','data/manifest.json')):
        if value[key]['path']!=str(root/name):raise ValueError('Preparation evidence path differs')
        validate_file_ref(value[key])
    if load_json(root/'data/manifest.json')['content_hash']!=value['parents']['data']:
        raise ValueError('Original prepared data changed')
    return old


def materialize(spec):
    value=spec['preparation_reuse'];old=check(value,spec)
    # Recheck the original inventory and terminal accounting immediately before
    # linking. A previously valid receipt cannot authorize changed source files.
    if inspect(value['source_spec']['path'])!=value:
        raise ValueError('Preparation recovery evidence changed')
    source=Path(old['root'])/'setup/data';dest=Path(spec['root'])/'setup/data'
    dest.mkdir(exist_ok=False);views={r:{} for r in spec['counts']}
    for folder,kind in layout():
        m=load_json(source/folder/'manifest.json');out=dest/folder;out.mkdir(parents=True)
        for info in m['arrays'].values():
            src=regular(source/folder,info['file']['path'])
            os.link(src,out/info['file']['path'])
        fields={k:v for k,v in m.items() if k not in {'content_hash','contract','schema_version','parents',*FLAGS}}
        parents=inputs.data_parents(spec)
        if kind=='U100_BANK':parents=dict(**parents,original=spec['embedding_hash'])
        rebuilt=artifact(kind,parents=parents,**fields)
        write_immutable_json(out/'manifest.json',rebuilt)
        if kind!='U100_BANK':
            role,name=folder.split('/');views[role][name]=rebuilt['content_hash']
    rebuilt=artifact('DATA',parents=inputs.data_parents(spec),views=views,counts=spec['counts'])
    write_immutable_json(dest/'manifest.json',rebuilt)
    write_immutable_json(dest.parent/'preparation_reuse.json',value)
    print(f"EMBEDDING_BRIDGE reused {len(value['arrays'])} authenticated arrays; no particle reconstruction",flush=True)
    return rebuilt


def credit(spec):
    """Only absent link destinations need a reservation credit; never double count."""
    value=spec.get('preparation_reuse')
    if value is None:return 0
    check(value,spec)
    dest=Path(spec['root'])/'setup/data'
    return sum(r['bytes'] for r in value['arrays'] if not (dest/r['path']).exists())
