from __future__ import annotations
import numpy as np
from scipy import stats

def classification_metrics(probs, labels, bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == labels
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()
    brier = (np.square(probs).sum(1) - 2*probs[np.arange(len(labels)),labels] + 1).mean()
    ece, reliability = 0., []
    assignment = np.minimum((conf * bins).astype(int), bins-1)
    for b in range(bins):
        ix = assignment == b
        n = int(ix.sum())
        acc = float(correct[ix].mean()) if n else None
        confidence = float(conf[ix].mean()) if n else None
        if n:
            ece += n / len(labels) * abs(acc - confidence)
        reliability.append({"bin":b, "lower":b/bins, "upper":(b+1)/bins,
                            "count":n, "accuracy":acc, "confidence":confidence})
    topk = np.argsort(probs, axis=1)[:, -min(5,probs.shape[1]):]
    return {"accuracy":float(correct.mean()*100), "top5":float((topk == labels[:,None]).any(1).mean()*100),
            "nll":float(nll), "ece":float(ece), "brier":float(brier)}, reliability

def mean_interval(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 1:
        return {"n":0, "mean":None, "sd":None, "ci_low":None, "ci_high":None}
    mean = float(x.mean())
    if n < 2:
        return {"n":n, "mean":mean, "sd":None, "ci_low":None, "ci_high":None}
    sd = float(x.std(ddof=1))
    width = float(stats.t.ppf(.975, n-1)*sd/np.sqrt(n))
    return {"n":n, "mean":mean, "sd":sd, "ci_low":mean-width, "ci_high":mean+width}

def paired_test(differences):
    d = np.asarray(differences, dtype=float)
    result = mean_interval(d)
    if len(d) < 2:
        p = None
    elif np.allclose(d, 0, rtol=0, atol=1e-12):
        p = 1.
    elif np.std(d, ddof=1) < 1e-12:
        # Zero estimated standard error cannot support a finite t statistic.
        p = None
    else:
        p = float(stats.ttest_1samp(d, 0).pvalue)
    result["p_raw"] = p
    return result

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    prev = 0.
    for rank, index in enumerate(order):
        prev = min(1., max(prev, (len(p)-rank)*p[index]))
        out[index] = prev
    return out
