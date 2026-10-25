"""Independent authenticity check (does not use train.py).
1. Recomputes features from the downloaded EEG and runs subject-level 5-fold CV x10 with the shipped classifier.
2. Same procedure with shuffled labels: must fall to ~50% if nothing leaks.
3. Sanity check of labels: AD subjects should have lower MMSE scores than healthy subjects.
Run:  python verify_independent.py
"""
import glob, os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import neuro

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = {"power": "theta/alpha", "coh": "theta"}   # the shipped feature set

subs = []
for f in sorted(glob.glob(os.path.join(HERE, "data", "alz", "*.npz"))):
    d = np.load(f)
    ep = d["epochs"][:, [list(d["ch"]).index(c) for c in neuro.AD_CH]].astype(float)
    ep = ep[neuro.clean(ep)]
    subs.append((int(d["label"]), neuro.ad_features(ep, SPEC), float(d["mmse"])))
y = np.array([s[0] for s in subs])
X = [s[1] for s in subs]


def cv(labels, seed):
    score = np.zeros(len(labels))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(labels, labels):
        m = Pipeline([("scale", StandardScaler()), ("c", neuro.CentroidClassifier("cosine"))])
        m.fit(np.vstack([X[i] for i in tr]), np.concatenate([[labels[i]] * len(X[i]) for i in tr]))
        for i in te:                      # test subjects were never seen in training
            score[i] = m.predict(X[i]).mean()
    pred = score > 0.5
    sens = (pred & (labels == 1)).sum() / (labels == 1).sum()
    spec = (~pred & (labels == 0)).sum() / (labels == 0).sum()
    return (sens + spec) / 2


real = [cv(y, r) for r in range(10)]
rng = np.random.default_rng(0)
shuffled = [cv(rng.permutation(y), r) for r in range(10)]
mmse = np.array([s[2] for s in subs])
print(f"subjects: {len(y)} ({y.sum()} AD, {len(y) - y.sum()} healthy)")
print(f"real labels     balanced accuracy {np.mean(real):.3f} ± {np.std(real):.3f}")
print(f"shuffled labels balanced accuracy {np.mean(shuffled):.3f} ± {np.std(shuffled):.3f}   (should be ~0.50)")
print(f"mean MMSE: AD {np.nanmean(mmse[y == 1]):.1f}, healthy {np.nanmean(mmse[y == 0]):.1f}   (AD should be lower)")
assert np.mean(real) > 0.75 and abs(np.mean(shuffled) - 0.5) < 0.12
