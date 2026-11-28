"""Actual authorized-row equality gate before reusing GPU qualification."""
from pathlib import Path
import sys
import numpy as np
from . import contracts as c,decode,runtime

BAD_FILE=c.BAD_FILE


def baseline(batch):
    module=sys.modules['hlt_classification.luka_d20_full_hydra.basket_reader']
    inputs=sys.modules['hlt_classification.luka_d20_full_hydra.inputs']
    module.close_reader();native=module.Reader;module.Reader=module._d20_original_reader
    try:return inputs.work_batch(batch)
    finally:module.close_reader();module.Reader=native


def samples(old):
    p,parent,arrays=old.population.load(old.spec['dataset'],Path(old.spec['root'])/'population')
    c.require(p==old.spec['population'],'Original membership replay differs')
    selected=[];signal=False;bad=False
    for index,record,a,codes,keep in old.population.selections(parent,arrays):
        is_bad=record['path']==BAD_FILE
        is_signal=not signal and not record['path'].startswith('raw/qcd/')
        if not (is_bad or is_signal):continue
        if is_bad:bad=True
        if is_signal:signal=True
        c.require(c.digest(Path(old.spec['dataset'])/record['path'])==record['sha256'],'Probe source hash differs')
        for role in (0,1):
            entries=np.flatnonzero(codes==role)
            c.require(len(entries)>0,'Probe file lacks an authorized role')
            lo=int(entries[0]);hi=lo+1
            while hi<len(codes) and codes[hi]==role and hi-lo<4:hi+=1
            for coordinate,targets in [('D000',role==0),('OFFLINE',False),('U100',False)]:
                selected.append((old.spec['dataset'],record,index,lo,hi,role,keep[lo:hi],coordinate,targets,
                    parent[0]['content_hash'],dict(hlt=a['hlt_jet_nparticles'][lo:hi],offline=a['jet_nparticles'][lo:hi])))
    c.require(bad and signal,'Probe must cover the timed-out QCD file and a signal file')
    return selected


def exact(left,right):
    lrows,lt=left;rrows,rt=right
    c.require(len(lrows)==len(rrows) and len(lt)==len(rt),'Reader row/target support differs')
    for (lid,lv),(rid,rv) in zip(lrows,rrows,strict=True):
        c.require(lid==rid,'Reader canonical identity differs')
        for name in ('features','vectors'):
            a,b=getattr(lv,name),getattr(rv,name)
            c.require(a.dtype==b.dtype and np.array_equal(a,b),'Reader numeric bytes differ: '+name)
    for a,b in zip(lt,rt,strict=True):
        c.require(len(a)==len(b),'Target fields differ')
        for x,y in zip(a,b,strict=True):
            c.require(x.dtype==y.dtype and np.array_equal(x,y),'Reader target/frame/aggregate differs')


def run(spec,job):
    old=runtime.install(spec);args=samples(old);path=str(Path(spec['root'])/'repair_spec.json')
    # Separate bounded pools prevent a stalled baseline read from trapping a
    # login process or consuming a GPU allocation. Results remain small.
    base=list(decode.decoded(args,workers=2,function=baseline,repair_path=path,max_restarts=0))
    repaired=list(decode.decoded(args,workers=2,repair_path=path,max_restarts=0))
    c.require(len(base)==len(repaired)==len(args),'Probe span count differs')
    records=[]
    for arg,left,right in zip(args,base,repaired,strict=True):
        exact(left[1],right[1])
        records.append(dict(path=arg[1]['path'],sha256=arg[1]['sha256'],role=arg[5],
            start=arg[3],stop=arg[4],coordinate=arg[7],targets=arg[8],byte_equal=True))
    for record in {a[1]['path']:a[1] for a in args}.values():
        c.require(c.digest(Path(old.spec['dataset'])/record['path'])==record['sha256'],'Probe source changed')
    proof=c.artifact('READER_PARITY',parents=dict(repair=spec['content_hash'],qualification=spec['qualification_hash']),
        job=job,cases=len(records),records=records,passed=True,models_updated=False)
    c.publish(Path(spec['root'])/'reader_parity.json',proof);return proof
