"""Bounded TRAIN benchmark, distinct from scientific campaign qualification."""
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from hlt_classification.data.cache_contracts import (with_content_hash, validate_content_hash,
    load_json, load_npz_arrays, write_immutable_json, atomic_publish_bytes, deterministic_npz_bytes)
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullsim.alljets_data import FLAGS
from hlt_classification.luka_u100_multiview.views import lengths
from . import campaign, contracts, inputs, targets, reuse


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f'LUKA_SUBCLASS_GPU_BENCHMARK_{kind}/v1', schema_version=1,
        parents=dict(parents or {}), **FLAGS, **fields))


def validate(value, kind, parents=None):
    validate_content_hash(value, expected_contract=f'LUKA_SUBCLASS_GPU_BENCHMARK_{kind}/v1', expected_schema_version=1)
    if any(value.get(k)!=v for k,v in FLAGS.items()) or (parents is not None and value['parents']!=parents):
        raise ValueError('Benchmark disclosure or lineage differs')


def registration():
    return dict(benchmark_only=True, science_release=False, gpu_profile='1080ti_fp32',
        gpus=2, cpus=6, memory_mb=32000, minutes=120,
        execution_site=dict(cluster='hydra',partition='gpus',account='default',qos='normal',
            gres='gpu:geforce_gtx_1080_ti:2',gpu_name='NVIDIA GeForce GTX 1080 Ti',compute_capability=[6,1]),
        global_batch=128, precision='fp32', tasks=['S0p5','S100'], warmup_batches=3, measured_batches=12,
        validation_batches=8, parity_batches=[16,15], position=3.,
        synchronization='DDP_with_SyncBatchNorm_common_global_padding', automatic_retry=False,
        parity_tolerances=dict(loss_abs=1e-3, gradient_relative_l2=5e-4, buffer_abs=5e-5),
        counts=contracts.COUNTS, recipe=contracts.recipe(), storage_budget_bytes=2*2**30)


def create(root, parent, project, commit):
    old=reuse.original(parent); root=Path(root).absolute(); project=Path(project).resolve()
    protected=[Path(old[k]).resolve() for k in ('root','project_dir','environment_dir','bundle_root','offline_root')]+[project]
    if (root.exists() or not root.parent.is_dir() or root.resolve()!=root or any(p.is_symlink() for p in (root,*root.parents))
            or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected)):
        raise ValueError('Use a fresh separate benchmark root')
    source=campaign.current(project,commit); campaign.site.environment(old['reference']['environment'])
    spec=artifact('SPEC',parents=dict(source=source['content_hash'],teacher_sweep=old['content_hash']),
        **registration(), root=str(root), project_dir=str(project), source_commit=commit, source_snapshot=source,
        environment_dir=str(Path(sys.prefix).resolve()),teacher_sweep_spec=file_ref(Path(parent).resolve()),
        teacher_refs=reuse.references(old), nodes={t:dict(old['imported']['node'],node_id=t) for t in contracts.FITS},
        **{k:old[k] for k in campaign.COPIED})
    inputs.validate_sources(spec)
    import shutil
    if shutil.disk_usage(root.parent).free < 3*2**30: raise OSError('Benchmark needs 3 GiB free output space')
    root.mkdir(); write_immutable_json(root/'benchmark_spec.json',spec)
    return spec


def check(spec, *, authenticate_parent=True):
    validate(spec,'SPEC'); validate_file_ref(spec['teacher_sweep_spec'])
    if (any(spec.get(k)!=v for k,v in registration().items())
            or Path(sys.prefix).resolve()!=Path(spec['environment_dir'])
            or campaign.current(spec['project_dir'],spec['source_commit'])!=spec['source_snapshot']
            or load_json(Path(spec['root'])/'benchmark_spec.json')!=spec):
        raise ValueError('Immutable benchmark registration differs')
    old=reuse.original(spec['teacher_sweep_spec']['path']) if authenticate_parent else load_json(spec['teacher_sweep_spec']['path'])
    if (spec['parents']!=dict(source=spec['source_snapshot']['content_hash'],teacher_sweep=old['content_hash'])
            or spec['teacher_refs']!=reuse.references(old) or any(spec[k]!=old[k] for k in campaign.COPIED)
            or spec['nodes']!={t:dict(old['imported']['node'],node_id=t) for t in contracts.FITS}):
        raise ValueError('Benchmark original TRAIN sources differ')
    campaign.site.environment(spec['reference']['environment']); inputs.validate_sources(spec)


def prepare(spec):
    """Build real targets once, then publish the same padded batches for both modes."""
    started=time.perf_counter(); root=Path(spec['root']); (root/'setup').mkdir()
    train,val=inputs.views(spec,'S0p5'); offline=inputs.offline_bank(spec,train)
    targets.build(spec,train,offline); del offline
    banks=inputs.banks(spec,'S0p5',train); inputs.join(train,banks)
    return publish_batches(spec,train,val,banks,preparation_started=started)


def publish_batches(spec,train,val,banks,*,preparation_started):
    root=Path(spec['root'])/'batches'; root.mkdir(); rows={}
    seed=spec['nodes']['S0p5']['sampler_seed']
    order=np.random.default_rng(np.random.SeedSequence([seed,1])).permutation(len(train))
    if len(train)<128*(spec['warmup_batches']+spec['measured_batches']) or len(val)<128*spec['validation_batches']:
        raise ValueError('Insufficient benchmark population')
    def publish(name,cache,ix):
        raw=cache.batch(ix)
        arrays={k:raw[k] for k in ('features','vectors','mask','labels')}
        arrays.update(identities=cache.identities[ix],indices=ix.astype(np.int64))
        if cache.role=='train': arrays.update(u100=banks['u100'].targets[ix],subclass=banks['subclass'].targets[ix])
        path=root/(name+'.npz'); atomic_publish_bytes(path,deterministic_npz_bytes(arrays))
        rows[name]=dict(file=file_ref(path,root=root),role=cache.role,rows=len(ix))
    for b in range(spec['warmup_batches']+spec['measured_batches']):
        publish(f'train_{b:02}',train,order[b*128:(b+1)*128])
    publish('longest',train,np.argsort(lengths(train),kind='stable')[-128:])
    publish('partial11',train,order[:11])
    tail=len(train)%128
    publish('tail',train,order[-(tail or 128):])
    for n in spec['parity_batches']: publish(f'parity{n}',train,order[:n])
    for b in range(spec['validation_batches']): publish(f'val_{b:02}',val,np.arange(b*128,(b+1)*128))
    total=sum(p.stat().st_size for p in root.glob('*.npz'))
    if total>spec['storage_budget_bytes']: raise OSError('Benchmark payload exceeded output budget')
    value=artifact('BATCHES',parents=dict(spec=spec['content_hash'],subclass=banks['subclass'].manifest['content_hash'],
        u100=spec['embedding_hash']),batches=rows,global_padding_before_sharding=True,
        preparation_seconds=time.perf_counter()-preparation_started,payload_bytes=total)
    write_immutable_json(root/'manifest.json',value); return value


def read_batches(spec):
    root=Path(spec['root'])/'batches'; m=load_json(root/'manifest.json'); validate(m,'BATCHES')
    if m['parents']!=dict(spec=spec['content_hash'],subclass=targets.check_manifest(spec)['content_hash'],u100=spec['embedding_hash']):
        raise ValueError('Prepared benchmark targets changed')
    expected={f'train_{i:02}':128 for i in range(15)}|{f'val_{i:02}':128 for i in range(8)}
    expected.update(longest=128,partial11=11,tail=spec['counts']['train']%128 or 128)
    expected.update({f'parity{n}':n for n in spec['parity_batches']})
    if set(m['batches'])!=set(expected) or m['global_padding_before_sharding'] is not True:
        raise ValueError('Benchmark batch coverage differs')
    arrays={}
    for name,row in m['batches'].items():
        role='validation' if name.startswith('val_') else 'train'
        if row['role']!=role or row['rows']!=expected[name] or row['file']['path']!=name+'.npz':
            raise ValueError('Benchmark batch role/ownership differs')
        arrays[name]=load_npz_arrays(validate_file_ref(row['file'],root=root))
        check_batch(arrays[name],expected[name],role)
    return m,arrays


def check_batch(raw,n,role):
    required={'features','vectors','mask','labels','identities','indices'}|({'u100','subclass'} if role=='train' else set())
    if set(raw)!=required: raise ValueError('Unexpected benchmark fields')
    length=raw['features'].shape[-1]
    shapes=dict(features=(n,17,length),vectors=(n,4,length),mask=(n,1,length),labels=(n,),identities=(n,32),indices=(n,))
    if role=='train': shapes.update(u100=(n,128),subclass=(n,10))
    if (any(raw[k].shape!=shape for k,shape in shapes.items()) or length<1
            or any(not np.isfinite(a).all() for a in raw.values())
            or raw['labels'].dtype!=np.int64 or not np.isin(raw['labels'],np.arange(11)).all()
            or raw['identities'].dtype!=np.uint8 or len(np.unique(raw['identities'],axis=0))!=n
            or raw['indices'].dtype!=np.int64 or len(np.unique(raw['indices']))!=n or (raw['indices']<0).any()
            or not np.isin(raw['mask'],[0,1]).all()
            or any(raw[k].dtype!=np.float32 for k in ('features','vectors'))):
        raise ValueError('Invalid benchmark inputs')
    if role=='train':
        q=raw['subclass']
        if (raw['u100'].dtype!=np.float32 or q.dtype!=np.float32 or (q<0).any()
                or not np.allclose(q.sum(1),1,atol=2e-6,rtol=0) or np.any(q[raw['labels']!=0,5:]!=0)):
            raise ValueError('Invalid benchmark target')


def run(spec):
    from . import benchmark_site as site, benchmark_submission as submission
    check(spec); site.allocation(spec); job=submission.require_job(spec)
    root=Path(spec['root']); (root/'run_claim').mkdir()
    write_immutable_json(root/'run_claim/intent.json',artifact('CLAIM',parents=dict(spec=spec['content_hash']),job_id=job))
    cli=str(Path(spec['project_dir'])/'scripts/luka_subclass_gpu_benchmark.py')
    base=[sys.executable,'-u','-B','-s',cli]
    subprocess.run(base+['prepare','--spec',str(root/'benchmark_spec.json')],check=True)
    outcomes={}
    for world in (1,2):
        check(spec,authenticate_parent=False)
        # Each phase is a separate process group. A failed/OOM phase cannot leave
        # stale CUDA allocations in the next comparison.
        cmd=[sys.executable,'-u','-B','-s','-m','torch.distributed.run','--standalone',
             '--nnodes=1','--nproc-per-node='+str(world),cli,'rank','--spec',str(root/'benchmark_spec.json')]
        with (root/f'world{world}.log').open('x',encoding='utf-8') as handle:
            result=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT)
        outcomes[str(world)]=result.returncode
        print(f'BENCHMARK world={world} exit={result.returncode}; log={root}/world{world}.log',flush=True)
    check(spec,authenticate_parent=False)
    value=report(spec,outcomes)
    write_immutable_json(root/'report.json',value)
    print_summary(value)
    if any(outcomes.values()): raise RuntimeError('Benchmark phase failed; inspect report and world logs; no science released')
    return value


def report(spec,outcomes):
    root=Path(spec['root']); phases={}
    for world in (1,2):
        path=root/f'world{world}.json'
        if outcomes[str(world)]==0:
            r=load_json(path); validate(r,'PHASE',dict(spec=spec['content_hash']))
            if (r['world_size']!=world or r['precision']!='fp32' or r['global_batch']!=128
                    or set(r['tasks'])!=set(spec['tasks'])): raise ValueError('Phase identity differs')
            for row in r['tasks'].values():
                if not np.isfinite(row['mean_step_seconds']) or row['mean_step_seconds']<=0:
                    raise ValueError('Invalid benchmark timing')
            phases[str(world)]=r
    comparisons={}
    if len(phases)==2:
        for task in spec['tasks']:
            a,b=[phases[str(w)]['tasks'][task] for w in (1,2)]
            speedup=a['mean_step_seconds']/b['mean_step_seconds']
            comparisons[task]=dict(two_gpu_speedup=speedup,
                two_independent_one_gpu_runs_relative_throughput=2/speedup,
                single_full_batch_memory_pass=a['longest_support_pass'],double_full_batch_memory_pass=b['longest_support_pass'])
    return artifact('REPORT',parents=dict(spec=spec['content_hash']),phase_exit_codes=outcomes,phases=phases,
        comparisons=comparisons,science_qualified=False,science_submitted=False,
        note='Bounded FP32 benchmark only; no scientific AUC, no automatic campaign release. Projections exclude full validation, startup and diagnostics. Independent-run throughput is idealized and assumes no contention.')


def print_summary(value):
    print(value['note'])
    for world,phase in value['phases'].items():
        for task,row in phase['tasks'].items():
            print(f'{world} GPU {task}: {row["mean_step_seconds"]:.3f}s/update; '
                  f'{row["train_only_30_epoch_hours"]:.2f}h projected training; '
                  f'longest-support pass={row["longest_support_pass"]}; '
                  f'peak GPU GiB={row["peak_gpu_gib"]}')
    print('Comparisons:',value['comparisons'])
    if '2' in value['phases']: print('Two-GPU deterministic parity:',value['phases']['2']['parity'])
