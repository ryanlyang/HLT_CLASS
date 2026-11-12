"""Copy authenticated completed exports; never run a teacher forward."""
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json, atomic_publish_bytes
from hlt_classification.cms_proxy_ladder.contracts import file_ref, validate_file_ref
from hlt_classification.luka_fullstat_six_teacher import kd_banks as previous

NAMES, FILES = previous.NAMES, previous.FILES
folder = previous.folder


def donor(spec):
    old = load_json(validate_file_ref(spec['baseline']['spec']))
    if (old['reuse'] != spec['reuse'] or old['anchor'] != spec['anchor']
            or old['counts'] != spec['counts']):
        raise ValueError('Native KD supervision or population differs from control')
    return old


def manifest(spec,name,*,verify_payload=True):
    old = donor(spec)
    # Validate original contract and exporter lineage at the copied location.
    value=previous.manifest(dict(old,root=spec['root']),name,verify_payload=verify_payload)
    refs=spec['baseline']['exports'][name]
    for filename,ref in zip(FILES,refs):
        if filename=='manifest.json' or verify_payload:
            if file_ref(folder(spec,name)/filename)['sha256'] != ref['sha256']:
                raise ValueError('Copied native KD bytes differ')
    return value


def arrays(spec,name,cache):
    manifest(spec,name)
    return previous.arrays(dict(donor(spec),root=spec['root']),name,cache)


def prepare(spec,device,started):
    old=donor(spec)
    for name in NAMES:
        previous.manifest(old,name)
        refs=spec['baseline']['exports'][name]
        expected=[file_ref(folder(old,name)/n) for n in FILES]
        if refs != expected:raise ValueError('Completed export inventory changed')
        dest=folder(spec,name);dest.mkdir(parents=True,exist_ok=False)
        for n,ref in zip(FILES,refs):
            atomic_publish_bytes(dest/n,Path(validate_file_ref(ref)).read_bytes())
        manifest(spec,name)
        print('SIX TEACHER S0 reused frozen native KD bank: '+name,flush=True)
