"""Held-out residual quantiles and a variance-preserving shared-jet copula."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from .contracts import QUANTILES, artifact, validate
from .rng import normals


def weighted_quantiles(values, weights, probabilities):
    v,w = np.asarray(values,np.float64),np.asarray(weights,np.float64)
    q = np.asarray(probabilities,np.float64)
    if v.ndim != 1 or w.shape != v.shape or not len(v) or np.any(w <= 0):
        raise ValueError("Invalid weighted quantile observations")
    if not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError("Invalid weighted quantile values")
    order = np.argsort(v,kind="stable"); v,w = v[order],w[order]
    unique,starts = np.unique(v,return_index=True)
    sums = np.add.reduceat(w,starts)
    cdf = (np.cumsum(sums)-.5*sums)/sums.sum()
    return np.interp(q,cdf,unique,left=unique[0],right=unique[-1])


def _weighted_latent(values,weights):
    result = np.empty_like(values)
    for j in range(values.shape[1]):
        unique,inverse = np.unique(values[:,j],return_inverse=True)
        weights_by_value = np.bincount(inverse,weights=weights,minlength=len(unique))
        cdf = (np.cumsum(weights_by_value)-.5*weights_by_value)/weights_by_value.sum()
        result[:,j] = ndtri(np.clip(cdf[inverse],.001,.999))
    return result


def _sqrt(matrix, *, inverse=False):
    eig,vectors = np.linalg.eigh((matrix+matrix.T)/2)
    eig = np.maximum(eig,1e-10 if inverse else 0.)
    return (vectors*(1/np.sqrt(eig) if inverse else np.sqrt(eig)))@vectors.T


def fit_cell(residuals,weights,jet_ids) -> dict:
    r,w,ids = np.asarray(residuals,np.float64),np.asarray(weights,np.float64),np.asarray(jet_ids)
    if r.ndim != 2 or len(r) == 0 or w.shape != (len(r),) or ids.shape != (len(r),):
        raise ValueError("Residual-cell shape differs")
    if not np.isfinite(r).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Invalid residual calibration")
    dim = r.shape[1]; unique,inverse = np.unique(ids,return_inverse=True); jets = len(unique)
    tables = np.column_stack([weighted_quantiles(r[:,j],w,QUANTILES) for j in range(dim)])
    latent = _weighted_latent(r,w)
    latent -= np.average(latent,axis=0,weights=w)
    variance = np.average(latent**2,axis=0,weights=w)
    latent /= np.sqrt(np.maximum(variance,1e-10))
    raw = (latent*w[:,None]).T@latent/w.sum()
    # Constant coordinates draw a degenerate marginal, but keep a well-defined
    # latent correlation matrix; their synthetic output still has zero variance.
    np.fill_diagonal(raw,1.)
    shrinkage = 1000/(1000+jets)
    correlation = (1-shrinkage)*raw+shrinkage*np.eye(dim)
    eigenvalues,vectors = np.linalg.eigh(correlation)
    corrected = (vectors*np.maximum(eigenvalues,1e-9))@vectors.T
    d = np.sqrt(np.diag(corrected)); corrected /= d[:,None]*d[None,:]
    correction = float(np.linalg.norm(corrected-correlation))
    shared_sum = np.zeros((dim,dim)); usable = 0
    order = np.argsort(inverse,kind="stable"); split = np.flatnonzero(np.diff(inverse[order]))+1
    for indexes in np.split(order,split):
        if len(indexes) < 2:
            continue
        a,b = latent[indexes],w[indexes]
        denominator = b.sum()**2-np.square(b).sum()
        if denominator <= 0:
            continue
        summed = (a*b[:,None]).sum(axis=0)
        shared_sum += (np.outer(summed,summed)-(a*b[:,None]).T@(a*b[:,None]))/denominator
        usable += 1
    shared = shared_sum/max(1,usable)*(1-shrinkage)
    root,whitener = _sqrt(corrected),_sqrt(corrected,inverse=True)
    whitened = whitener@shared@whitener
    e,v = np.linalg.eigh((whitened+whitened.T)/2)
    shared = root@((v*np.clip(e,0,1))@v.T)@root
    independent = corrected-shared
    return dict(jets=jets,records=len(r),supported=jets>=1000,shared_estimable_jets=usable,
                quantiles=tables.tolist(), correlation=corrected.tolist(),
                shared_covariance=shared.tolist(), independent_covariance=independent.tolist(),
                shared_factor=_sqrt(shared).tolist(), independent_factor=_sqrt(independent).tolist(),
                shrinkage=shrinkage,positive_definite_correction=correction)


def fit_backend(residuals,weights,jet_ids,conditioning,*,location_membership_hash:str,
                residual_membership_hash:str,with_crowding:bool,edges:dict,coordinates:list[str]) -> dict:
    if location_membership_hash == residual_membership_hash:
        raise PermissionError("Residuals must be calibrated out of location-fit sample")
    residuals = np.asarray(residuals,np.float64)
    conditioning = np.asarray(conditioning,np.float64)
    if conditioning.shape != (len(residuals),4) or residuals.shape[1] != len(coordinates):
        raise ValueError("Expected category/log-pT/abs-eta/crowding residual conditioning")
    cells = {}
    for i,row in enumerate(conditioning):
        category = int(row[0]); bins = [int(np.searchsorted(edges[n],row[j]))
                                       for n,j in (("pt",1),("eta",2),("crowding",3))]
        keys = [(category,*bins)] if with_crowding else []
        keys += [(category,*bins[:2]),(category,bins[0]),(category,)]
        for key in keys:
            cells.setdefault(key,[]).append(i)
    result = []
    weights,jet_ids = np.asarray(weights),np.asarray(jet_ids)
    for key,indexes in sorted(cells.items()):
        cell = fit_cell(residuals[indexes],weights[indexes],jet_ids[indexes])
        # Always retain category-only fallback, even if unsupported. Never take
        # another category's tracking distribution to disguise sparse support.
        if cell["supported"] or len(key) == 1:
            result.append(dict(key=list(key),**cell))
    return artifact("RESIDUAL_BACKEND",parents={"fit_location_membership":location_membership_hash,
                    "fit_residual_membership":residual_membership_hash}, cells=result,
                    edges=edges, with_crowding=with_crowding, coordinates=coordinates,
                    probabilities=list(QUANTILES), tail_policy="clamp_report_v1")


def sample(backend:dict,condition,*,jet:str,replica:int,object_key:str,module:str,validate_model:bool=True):
    if validate_model:
        validate(backend,"RESIDUAL_BACKEND")
    category = int(condition[0]); edges = backend["edges"]
    bins = [int(np.searchsorted(edges[n],condition[j])) for n,j in (("pt",1),("eta",2),("crowding",3))]
    candidates = [(category,*bins)] if backend["with_crowding"] else []
    candidates += [(category,*bins[:2]),(category,bins[0]),(category,)]
    lookup = {tuple(c["key"]):c for c in backend["cells"]}
    key = next((key for key in candidates if key in lookup),None)
    if key is None:
        # Generation must expose an unseen-state policy, not invent calibrated
        # residuals. This is a typed support result, not a fitting exception.
        return None,dict(unseen_category=True,supported=False,backoff=True,tail_clamped=False)
    cell = lookup[key]; dim = len(backend["coordinates"])
    common = normals(jet,replica,"shared_jet",module,dim)
    individual = normals(jet,replica,"kinematics",module+":"+object_key,dim)
    z = np.asarray(cell["shared_factor"])@common + np.asarray(cell["independent_factor"])@individual
    u = ndtr(z); q = np.asarray(cell["quantiles"])
    result = np.array([np.interp(u[j],backend["probabilities"],q[:,j]) for j in range(dim)])
    return result,dict(unseen_category=False,supported=cell["supported"],backoff=key!=candidates[0],
                       tail_clamped=bool(np.any(u<QUANTILES[0]) or np.any(u>QUANTILES[-1])))
