"""Reuse the authenticated full-wide scalar membership, never historical models."""
from pathlib import Path
import numpy as np
from hlt_classification.luka_sixclass_wide_b256_oscar import population as donor
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref
from hlt_classification.data.cache_contracts import load_npz_arrays
from . import contracts as c


def build(dataset,folder):return donor.build(dataset,folder)


def load(dataset,folder):
    # Scalar replay remains authoritative before any privileged particle read.
    p,f,members,roles,arrays,_=donor.load(dataset,folder)
    if p['content_hash']!=c.POPULATION_HASH:raise ValueError('Wrong full population')
    return p,(p,f,members,roles,arrays),arrays


def selections(parent,arrays):
    _,f,members,roles,_=parent
    for index,(record,a,rs) in enumerate(zip(f['files'],members,roles,strict=True)):
        masks=donor.masks(a,rs);codes=np.full(len(rs),255,np.uint8);keep=np.zeros(len(rs),bool)
        for code,role in enumerate(('train','validation')):
            codes[masks[role]]=code;keep[masks[role]]=True
        if keep.any():yield index,record,a,codes,keep


def rows(spec):return load_npz_arrays(validate_file_ref(spec['population']['rows'],root=Path(spec['root'])/'population'))
