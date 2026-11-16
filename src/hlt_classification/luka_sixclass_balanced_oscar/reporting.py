"""Primary in-range selection; diagnostic subsets share the selected model."""
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from hlt_classification.luka_fullstat_balanced_ladder.reporting import recovery
from . import contracts as c


def summary(y,p):
    complete=set(np.unique(y))==set(range(6))
    measured=evaluate_probabilities(y,p) if complete else dict(rows=len(y),status='missing_class')
    return dict(summary=measured,R50=rejection(y,p,.5),R30=rejection(y,p,.3))


def metrics(val,probabilities,outside,out_prob,arrays,parents):
    pt=arrays['validation_pt']
    if len(pt)!=len(val) or not np.all((pt>=600)&(pt<2600)):
        raise ValueError('Primary validation includes outside-range jets')
    out_pt=arrays['outside_validation_pt']
    if len(out_pt)!=len(outside) or not np.isfinite(out_pt).all() or np.any((out_pt>=600)&(out_pt<2600)):
        raise ValueError('Diagnostic validation includes in-range jets')
    values=dict(in_range=summary(val.labels,probabilities),outside_range=summary(outside.labels,out_prob))
    for low in range(600,2600,100):
        mask=(pt>=low)&(pt<low+100)
        values[f'{low}_{low+100}']=summary(val.labels[mask],probabilities[mask])
    return c.artifact('METRICS',parents=parents,populations=values,checkpoint_reselection=False,
        selection_population='in_range',diagnostic_population='outside_range')


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels, p = np.asarray(labels), np.asarray(probabilities, np.float64)
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or p.shape != (len(labels), len(c.CLASS_NAMES))
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, atol=2e-6, rtol=0)
            or set(np.unique(labels)) != set(range(len(c.CLASS_NAMES)))):
        raise ValueError("Metrics need finite normalized probabilities and every registered class")
    qcd = labels == 0
    per_class = {}
    for cls in range(1, len(c.CLASS_NAMES)):
        denominator = p[:, cls] + p[:, 0]
        score = np.divide(p[:, cls], denominator, out=np.zeros(len(p)), where=denominator > 0)
        signal_scores = np.sort(score[labels == cls])[::-1]
        index = math.ceil(.5 * len(signal_scores)) - 1
        threshold = float(signal_scores[index])
        count = int(np.count_nonzero(score[qcd] >= threshold))
        rejection = int(qcd.sum()) / count if count else None
        per_class[c.CLASS_NAMES[cls]] = dict(
            rejection_at_50pct=rejection, threshold=threshold, background_count=int(qcd.sum()),
            background_passing=count, achieved_signal_efficiency=float(np.mean(signal_scores >= threshold)),
            zero_fpr_censored=count == 0,
            one_event_resolution_rejection=int(qcd.sum()),
        )
    rejections = [r["rejection_at_50pct"] for r in per_class.values()]
    log_r50 = float(np.mean(np.log(rejections))) if all(v is not None for v in rejections) else None
    return dict(
        rows=len(labels), accuracy=float(np.mean(p.argmax(axis=1) == labels)),
        macro_ovr_auc=float(roc_auc_score(labels, p / p.sum(axis=1, keepdims=True), multi_class="ovr", average="macro")),
        cross_entropy=float(-np.mean(np.log(np.maximum(p[np.arange(len(p)), labels], 1e-30)))),
        macro_mean_log_qcd_rejection_at_50pct_signal=log_r50,
        macro_r50=math.exp(log_r50) if log_r50 is not None else None,
        per_class=per_class,
    )



def rejection(labels,probabilities,efficiency):
    if efficiency not in (.3,.5):raise ValueError('Unregistered rejection efficiency')
    labels=np.asarray(labels);p=np.asarray(probabilities,np.float64)
    if (labels.ndim!=1 or labels.dtype.kind not in 'iu' or p.shape!=(len(labels),6)
            or np.any((labels<0)|(labels>=6)) or not np.isfinite(p).all() or np.any(p<0)
            or not np.allclose(p.sum(1),1.,atol=2e-6,rtol=0)):
        raise ValueError('Invalid six-class rejection inputs')
    qcd=labels==0;out={}
    for cls in range(1,6):
        signal=labels==cls;den=p[:,cls]+p[:,0];score=np.divide(p[:,cls],den,out=np.zeros(len(p)),where=den>0)
        n=int(qcd.sum());ns=int(signal.sum())
        if not n or not ns:
            out[c.CLASS_NAMES[cls]]=dict(status='missing_class',background_count=n,signal_count=ns);continue
        cut=float(np.sort(score[signal])[::-1][math.ceil(efficiency*ns)-1]);passing=int(np.count_nonzero(score[qcd]>=cut))
        out[c.CLASS_NAMES[cls]]=dict(status='censored' if passing==0 else 'measured',rejection=n/passing if passing else None,
            threshold=cut,background_count=n,background_passing=passing,signal_count=ns,
            signal_passing=int(np.count_nonzero(score[signal]>=cut)),one_event_resolution=n)
    return out

