"""Portable table, spline and bounded-tree conditional response estimators.

No pickle, automatic validation split, adaptive candidate search or early stop.
Array inputs are fit_location only; held-out calibration is a separate call.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from .contracts import artifact, candidates, validate
from .features import FEATURE_NAMES

INTERACTIONS = ((2,3),(2,18),(13,23),(5,7),(6,8))


def _candidate(candidate_id):
    return next(r for r in candidates()["candidates"] if r["id"] == candidate_id)


def _data(x, y, weights, task, classes):
    x = np.asarray(x, np.float64); y = np.asarray(y); w = np.asarray(weights, np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) == 0 or w.shape != (len(x),):
        raise ValueError("Invalid conditional-fit dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Invalid/nonfinite fit records")
    if task == "categorical":
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or classes is None or classes < 2:
            raise ValueError("Categorical target contract differs")
        if np.any(y < 0) or np.any(y >= classes):
            raise ValueError("Unknown categorical target state")
        target = np.eye(classes)[y]
    elif task == "continuous":
        target = y.reshape(len(x), -1).astype(np.float64, copy=True)
    else:
        raise ValueError("Unknown response fitting task")
    return x, target, w


def _standardizer(x):
    q = np.quantile(x,[.001,.25,.5,.75,.999],axis=0)
    scale = q[3]-q[1]; scale[scale == 0] = 1
    return dict(center=q[2].tolist(), scale=scale.tolist(), low=q[0].tolist(), high=q[4].tolist())


def _scale(x, standardizer):
    x = np.clip(x, standardizer["low"],standardizer["high"])
    return (x-standardizer["center"])/standardizer["scale"]


def _table_fit(x,y,w,candidate):
    edges = [np.unique(np.quantile(x[:,col],np.linspace(0,1,bins+1)[1:-1])).tolist()
             for col,bins in ((2,candidate["pt_bins"]),(3,candidate["eta_bins"]))]
    # Missing-value/identity states are explicit cells, not numerical fake values.
    def key(row, detailed):
        cat = int(row[0]); state = tuple(int(v) for v in row[9:13])
        prefix = (cat,int(row[1]),*state,*(int(v) for v in row[28:]))
        return prefix + (int(np.searchsorted(edges[0],row[2])),int(np.searchsorted(edges[1],row[3]))) if detailed else prefix
    parents, bins = {}, {}
    for i,row in enumerate(x):
        for mapping,detailed in ((parents,False),(bins,True)):
            k = key(row,detailed)
            if k not in mapping:
                mapping[k] = [0.,np.zeros(y.shape[1])]
            mapping[k][0] += w[i]; mapping[k][1] += w[i]*y[i]
    global_mean = np.average(y,axis=0,weights=w)
    parent_rows = [{"key":list(k),"mean":(v[1]/v[0]).tolist()} for k,v in sorted(parents.items())]
    rows = []
    for k,(weight,total) in sorted(bins.items()):
        parent = parents[k[:-2]]; prior = parent[1]/parent[0]
        mean = (total + candidate["pseudocount"]*prior)/(weight+candidate["pseudocount"])
        rows.append(dict(key=list(k),mean=mean.tolist()))
    return dict(edges=edges, parents=parent_rows, bins=rows, global_mean=global_mean.tolist())


def _table_predict(x,model):
    parents = model.get("_parents")
    bins = model.get("_bins")
    if parents is None:
        parents = {tuple(r["key"]):r["mean"] for r in model["parents"]}
        bins = {tuple(r["key"]):r["mean"] for r in model["bins"]}
    output = []
    for row in x:
        k = (int(row[0]),int(row[1]),*(int(v) for v in row[9:13]),*(int(v) for v in row[28:]))
        cell = k + (int(np.searchsorted(model["edges"][0],row[2])),int(np.searchsorted(model["edges"][1],row[3])))
        output.append(bins.get(cell,parents.get(k,model["global_mean"])))
    return np.asarray(output, np.float64).reshape(len(x),len(model["global_mean"]))


def _spline_knots(x,interior):
    knots = []
    # Category and charge are encoded separately; continuous and valid-state predictors follow.
    for j in range(2,x.shape[1]):
        values = np.unique(np.quantile(x[:,j],np.linspace(0,1,interior+2)))
        if len(values) < 2:
            knots.append([])
        else:
            knots.append([*([float(values[0])]*4),*map(float,values[1:-1]),*([float(values[-1])]*4)])
    return knots


def _basis(x, raw, knots):
    columns = [np.ones((len(x),1)), np.eye(6)[raw[:,0].astype(int)][:,1:],raw[:,1:2]]
    bases = {}
    for j,k in enumerate(knots,2):
        if not k:
            bases[j] = np.zeros((len(x),1))
        else:
            coordinate = np.clip(x[:,j],k[3],k[-4])
            bases[j] = BSpline.design_matrix(coordinate,k,3,extrapolate=False).toarray()[:,1:]
        columns.append(bases[j])
    for a,b in INTERACTIONS:
        columns.append((bases[a][:,:,None]*bases[b][:,None,:]).reshape(len(x),-1))
    return np.column_stack(columns)


def _smooth_fit(x,y,w,candidate,task):
    standardizer = _standardizer(x); z = _scale(x,standardizer)
    knots = _spline_knots(z,candidate["interior_knots"])
    dim = _basis(z[:1],x[:1],knots).shape[1]; outdim = y.shape[1]
    penalty = np.full(dim,candidate["ridge"]); penalty[0] = 0
    chunk = 4096
    if task == "continuous":
        lhs = np.diag(penalty); rhs = np.zeros((dim,outdim))
        for start in range(0,len(x),chunk):
            b = _basis(z[start:start+chunk],x[start:start+chunk],knots)
            weighted = b*w[start:start+chunk,None]
            lhs += b.T@weighted; rhs += weighted.T@y[start:start+chunk]
        coef = np.linalg.lstsq(lhs,rhs,rcond=1e-12)[0]
        diagnostic = dict(method="chunked_weighted_ridge", convergence=True)
    else:
        def objective(flat):
            coef = flat.reshape(dim,outdim)
            loss = .5*np.sum(penalty[:,None]*coef**2)
            grad = penalty[:,None]*coef
            for start in range(0,len(x),chunk):
                b = _basis(z[start:start+chunk],x[start:start+chunk],knots)
                logits = b@coef; target = y[start:start+chunk]; weight = w[start:start+chunk]
                loss += np.sum(weight*(logsumexp(logits,axis=1)-(target*logits).sum(axis=1)))
                grad += b.T@((softmax(logits,axis=1)-target)*weight[:,None])
            return loss/w.sum(), (grad/w.sum()).ravel()
        result = minimize(objective,np.zeros(dim*outdim),jac=True,method="L-BFGS-B",
                          options={"maxiter":1000,"ftol":1e-10,"gtol":1e-7})
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise ValueError("Nonfinite smooth fit")
        coef = result.x.reshape(dim,outdim)
        diagnostic = dict(method="chunked_penalized_multinomial", convergence=bool(result.success),
                          iterations=int(result.nit), message=str(result.message))
    return dict(standardizer=standardizer,knots=knots,coefficients=coef.tolist(),diagnostic=diagnostic)


def _tree_predict(x,tree):
    x = np.asarray(x,np.float32)  # Match sklearn's split-search feature precision.
    left,right = np.asarray(tree["left"]),np.asarray(tree["right"])
    feature,threshold = np.asarray(tree["feature"]),np.asarray(tree["threshold"])
    positions = np.zeros(len(x), np.int64)
    while len(positions) and np.any(left[positions] >= 0):
        rows = np.flatnonzero(left[positions] >= 0); nodes = positions[rows]
        positions[rows] = np.where(x[rows,feature[nodes]] <= threshold[nodes],left[nodes],right[nodes])
    return np.asarray(tree["value"])[positions]


def _tree_fit(x,y,w,candidate,task):
    from sklearn.tree import DecisionTreeRegressor
    standardizer = _standardizer(x); z = np.asarray(_scale(x,standardizer),np.float32)
    base = np.average(y,axis=0,weights=w)
    if task == "categorical":
        base = np.log(np.maximum(base,1e-12)); base -= base.mean()
    prediction = np.tile(base,(len(x),1)); trees = []
    for iteration in range(candidate["trees"]):
        residual = y - (softmax(prediction,axis=1) if task == "categorical" else prediction)
        estimator = DecisionTreeRegressor(max_depth=candidate["depth"],
                     min_samples_leaf=candidate["min_effective_leaf"], random_state=20260917+iteration)
        estimator.fit(z,residual,sample_weight=w)
        native = estimator.tree_
        left,right = native.children_left.copy(),native.children_right.copy()
        feature,threshold = native.feature.copy(),native.threshold.copy()
        values = np.zeros((native.node_count,y.shape[1]))
        memberships = {0:np.arange(len(z))}
        for node in range(native.node_count):
            indexes = memberships[node]; weights = w[indexes]
            values[node] = (weights[:,None]*residual[indexes]).sum(axis=0)/(weights.sum()+candidate["l2"])
            if left[node] >= 0:
                goes_left = z[indexes,feature[node]] <= threshold[node]
                memberships[int(left[node])] = indexes[goes_left]
                memberships[int(right[node])] = indexes[~goes_left]
        # Pruning happens BEFORE updating boosting scores. Otherwise later trees
        # would be fitted against a different ensemble than the portable artifact.
        for node in reversed(range(native.node_count)):
            if left[node] < 0:
                continue
            child_effective = []
            for child in (left[node],right[node]):
                weights = w[memberships[int(child)]]
                child_effective.append(weights.sum()**2/np.square(weights).sum() if len(weights) else 0.)
            if min(child_effective) < candidate["min_effective_leaf"]:
                left[node] = right[node] = -1
        tree = dict(left=left.tolist(),right=right.tolist(),feature=feature.tolist(),
                    threshold=threshold.tolist(),value=values.tolist())
        prediction += candidate["learning_rate"]*_tree_predict(z,tree)
        trees.append(tree)
    return dict(standardizer=standardizer,base=base.tolist(),trees=trees,
                interpretation="one_shared_multioutput_weak_tree_per_stage",learning_rate=candidate["learning_rate"])


def fit_conditional(x,y,weights,*,candidate_id:str, task:str, membership_hash:str,
                    classes:int|None=None, tracking_linear: bool=False,
                    tracking_outputs: tuple[int, ...] | None=None) -> dict:
    candidate = _candidate(candidate_id)
    x,target,w = _data(x,y,weights,task,classes)
    if tracking_linear and task != "continuous":
        raise ValueError("Tracking linear response only applies to continuous targets")
    linear = None
    if candidate["family"] == "table" and tracking_linear:
        # Preserve input displacement/error dependence in the simple family.
        standardizer = _standardizer(x)
        standardizer.update(low=x.min(axis=0).tolist(), high=x.max(axis=0).tolist())
        b = np.column_stack((np.ones(len(x)),_scale(x, standardizer)[:,5:9]))
        linear = {}
        for category in np.unique(x[:,0]):
            keep = x[:,0] == category
            coef = np.linalg.lstsq(b[keep]*np.sqrt(w[keep,None]),target[keep]*np.sqrt(w[keep,None]),rcond=1e-12)[0]
            # Tracking response is not an extrapolating momentum predictor.
            if tracking_outputs is not None:
                coef[:, [j for j in range(target.shape[1]) if j not in tracking_outputs]] = 0.
            linear[str(int(category))] = coef.tolist()
            target[keep] -= b[keep]@coef
    if candidate["family"] == "table":
        model = _table_fit(x,target,w,candidate)
        model["standardizer"] = _standardizer(x)
        if linear is not None:
            model["tracking_standardizer"] = standardizer
    elif candidate["family"] == "smooth":
        model = _smooth_fit(x,target,w,candidate,task)
    else:
        model = _tree_fit(x,target,w,candidate,task)
    return artifact("CONDITIONAL_MODEL",parents={"fit_location_membership":membership_hash},
                    candidate=candidate, task=task, classes=classes, model=model,
                    tracking_linear=linear, temperature=1., temperature_calibrated=False,
                    feature_names=list(FEATURE_NAMES), output_dim=target.shape[1])


def predict(x, fitted:dict, *, probabilities:bool=True, validate_model:bool=True) -> np.ndarray:
    if validate_model:
        validate(fitted,"CONDITIONAL_MODEL")
    if fitted["candidate"] != _candidate(fitted["candidate"]["id"]):
        raise ValueError("Conditional candidate differs from registry")
    x = np.asarray(x,np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid prediction fields")
    if len(x) == 0:
        return np.empty((0,fitted["output_dim"]),np.float64)
    model, family = fitted["model"],fitted["candidate"]["family"]
    if family == "table":
        result = _table_predict(x,model)
        if fitted["tracking_linear"] is not None:
            b = np.column_stack((np.ones(len(x)),_scale(x, model["tracking_standardizer"])[:,5:9]))
            for category,coef in fitted["tracking_linear"].items():
                keep = x[:,0] == int(category)
                result[keep] += b[keep]@np.asarray(coef)
        if fitted["task"] == "categorical":
            result = np.log(np.maximum(result,1e-12))
    elif family == "smooth":
        z = _scale(x,model["standardizer"])
        result = _basis(z,x,model["knots"])@np.asarray(model["coefficients"])
    else:
        z = _scale(x,model["standardizer"]); result = np.tile(model["base"],(len(x),1))
        for tree in model["trees"]:
            result += model["learning_rate"]*_tree_predict(z,tree)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite conditional prediction")
    return softmax(result/fitted["temperature"],axis=1) if fitted["task"] == "categorical" and probabilities else result


class Predictor:
    """Validated RAM-only inference preparation, with unchanged fitted parameters.

    Tree traversal is vectorized across the frozen trees, but accumulation stays
    in tree order using cumsum (not a reassociated reduction). Chunk size bounds
    temporary memory independently of caller batch size. No cached state is
    inserted into, or exported with, a scientific artifact.
    """

    def __init__(self, fitted):
        from copy import deepcopy
        validate(fitted, "CONDITIONAL_MODEL")
        if fitted["candidate"] != _candidate(fitted["candidate"]["id"]):
            raise ValueError("Conditional candidate differs from registry")
        self.fitted = deepcopy(fitted)
        self.model = self.fitted["model"]
        self.family = self.fitted["candidate"]["family"]
        if self.family == "table":
            self.model["_parents"] = {tuple(r["key"]): r["mean"] for r in self.model["parents"]}
            self.model["_bins"] = {tuple(r["key"]): r["mean"] for r in self.model["bins"]}
        elif self.family == "tree":
            trees = self.model["trees"]
            size, dims = max(len(t["left"]) for t in trees), self.fitted["output_dim"]
            self.left = np.full((len(trees), size), -1, np.int64)
            self.right = self.left.copy()
            self.feature = np.zeros_like(self.left)
            self.threshold = np.zeros(self.left.shape)
            self.values = np.zeros((*self.left.shape, dims))
            for i, tree in enumerate(trees):
                n = len(tree["left"])
                for name in ("left", "right", "feature", "threshold"):
                    getattr(self, name)[i, :n] = tree[name]
                self.values[i, :n] = tree["value"]
        else:
            self.coef = np.asarray(self.model["coefficients"])

    def __call__(self, x, *, probabilities=True):
        fitted, model = self.fitted, self.model
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Invalid prepared prediction fields")
        if not len(x):
            return np.empty((0, fitted["output_dim"]), np.float64)
        if self.family == "table":
            result = _table_predict(x, model)
            if fitted["tracking_linear"] is not None:
                b = np.column_stack((np.ones(len(x)), _scale(x, model["tracking_standardizer"])[:, 5:9]))
                for category, coef in fitted["tracking_linear"].items():
                    keep = x[:, 0] == int(category)
                    result[keep] += b[keep]@np.asarray(coef)
            if fitted["task"] == "categorical":
                result = np.log(np.maximum(result, 1e-12))
        elif self.family == "smooth":
            z = _scale(x, model["standardizer"])
            result = _basis(z, x, model["knots"])@self.coef
        else:
            z = np.asarray(_scale(x, model["standardizer"]), np.float32)
            result = np.empty((len(x), fitted["output_dim"]))
            tree = np.arange(len(self.left))[:, None]
            for start in range(0, len(x), 16):
                chunk = z[start:start+16]
                positions = np.zeros((len(self.left), len(chunk)), np.int64)
                for _ in range(fitted["candidate"]["depth"]):
                    left = self.left[tree, positions]
                    active = left >= 0
                    feature = self.feature[tree, positions]
                    # Leaf feature sentinels are irrelevant; use column zero.
                    values = chunk[np.arange(len(chunk))[None, :], np.maximum(feature, 0)]
                    nxt = np.where(values <= self.threshold[tree, positions], left, self.right[tree, positions])
                    positions = np.where(active, nxt, positions)
                values = model["learning_rate"]*self.values[tree, positions]
                base = np.tile(model["base"], (1, len(chunk), 1))
                result[start:start+len(chunk)] = np.cumsum(np.concatenate([base, values], axis=0), axis=0)[-1]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prepared conditional prediction")
        return softmax(result/fitted["temperature"], axis=1) if fitted["task"] == "categorical" and probabilities else result


def calibrate_temperature(fitted:dict,x,y,weights,*,residual_membership_hash:str) -> dict:
    from .contracts import with_content_hash
    if fitted["task"] != "categorical" or residual_membership_hash == fitted["parents"]["fit_location_membership"]:
        raise PermissionError("Temperature calibration requires disjoint fit-residual membership")
    _,target,w = _data(x,y,weights,"categorical",fitted["classes"])
    logits = predict(x,fitted,probabilities=False)
    def loss(log_temperature):
        z = logits/np.exp(log_temperature)
        return float(np.sum(w*(logsumexp(z,axis=1)-(target*z).sum(axis=1)))/w.sum())
    optimized = minimize_scalar(loss,bounds=(-4.6,4.6),method="bounded")
    if not optimized.success or not np.isfinite(optimized.fun):
        raise ValueError("Probability calibration failed")
    value = dict(fitted,parents={**fitted["parents"],"fit_residual_membership":residual_membership_hash},
                 temperature=float(np.exp(optimized.x)),temperature_calibrated=True)
    return with_content_hash(value)
