"""
Experiments for the revised objectives (run after train.py). Results are added to models/metrics.json.

Objective 2: computational cost (time, space, implementation complexity) of the nearest-centroid
             classifier vs algorithms used in the literature (k-NN, SVM, Random Forest, Logistic
             Regression, MLP neural network, compact CNN / EEGNet), at their measured accuracy.
Objective 3: distance-measure comparison (Euclidean, Manhattan, Cosine, Minkowski p=3, Hamming,
             Jaccard) for the nearest-centroid and k-NN classifiers.

All accuracy figures use subject-level stratified 5-fold CV with epoch voting (no subject is ever in
both training and test). The feature set is the one selected by train.py.
Usage:  python experiments.py
"""
import io, json, os, pickle, time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import neuro, train

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "static", "img", "results")
REPEATS = 10
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ distance-aware centroid classifier
CentroidClassifier = neuro.CentroidClassifier


METRICS = [("Euclidean", "euclidean"), ("Manhattan", "manhattan"), ("Cosine", "cosine"),
           ("Minkowski (p=3)", "minkowski"), ("Hamming", "hamming"), ("Jaccard", "jaccard")]


def knn(metric):
    if metric in CentroidClassifier.BINARY:
        # binarise standardised features, then use the boolean distance
        from sklearn.preprocessing import Binarizer
        return make_pipeline(StandardScaler(), Binarizer(threshold=0.0),
                             KNeighborsClassifier(15, metric=metric, algorithm="brute"))
    kw = {"p": 3} if metric == "minkowski" else {}
    return make_pipeline(StandardScaler(), KNeighborsClassifier(15, metric=metric, algorithm="brute", metric_params=None, **kw))


# ------------------------------------------------------------------ CV helper (subject-level, epoch voting)
def cv(subs, X, make, repeats=REPEATS, dtype=None):
    y = np.array([s["y"] for s in subs]); reps = []
    for r in range(repeats):
        oof = np.zeros(len(subs))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(y, y):
            m = make()
            m.fit(np.vstack([X[subs[i]["id"]] for i in tr]),
                  np.concatenate([[subs[i]["y"]] * len(X[subs[i]["id"]]) for i in tr]))
            for i in te:
                oof[i] = m.predict(X[subs[i]["id"]]).mean()
        reps.append(train.metrics(y, oof))
    return {k: [float(np.mean([q[k] for q in reps])), float(np.std([q[k] for q in reps]))]
            for k in ("balanced_accuracy", "sensitivity", "specificity", "auc")}


# ------------------------------------------------------------------ compact CNN (EEGNet-style), torch
def make_cnn():
    import torch, torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self, ch=3):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (ch, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
                nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
                nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(0.25), nn.Flatten())
            with torch.no_grad():
                n = self.f(torch.zeros(1, 1, ch, neuro.FS * neuro.EPOCH_S)).shape[1]
            self.out = nn.Linear(n, 2)

        def forward(self, x):
            return self.out(self.f(x))

    class CNNClassifier:
        def __init__(self, epochs=12):
            self.epochs = epochs

        def fit(self, X, y):
            torch.manual_seed(0)
            self.net = EEGNet(X.shape[1])
            opt = torch.optim.Adam(self.net.parameters(), 1e-3)
            Xt = torch.tensor(X[:, None] / 50.0, dtype=torch.float32); yt = torch.tensor(y)
            self.net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xt))
                for b in range(0, len(Xt), 64):
                    idx = perm[b:b + 64]
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(self.net(Xt[idx]), yt[idx])
                    loss.backward(); opt.step()
            return self

        def predict(self, X):
            self.net.eval()
            with torch.no_grad():
                return self.net(torch.tensor(X[:, None] / 50.0, dtype=torch.float32)).argmax(1).numpy()

        def n_params(self):
            return sum(p.numel() for p in self.net.parameters())

        def size_bytes(self):
            buf = io.BytesIO(); torch.save(self.net.state_dict(), buf); return buf.tell()

    return CNNClassifier


# ------------------------------------------------------------------ cost measurement
def n_params(m):
    clf = m.steps[-1][1] if hasattr(m, "steps") else m
    sc = m.steps[0][1] if hasattr(m, "steps") else None
    base = 2 * sc.n_features_in_ if isinstance(sc, StandardScaler) else 0
    if isinstance(clf, (NearestCentroid, CentroidClassifier)):
        return base + clf.centroids_.size
    if isinstance(clf, KNeighborsClassifier):
        return base + clf._fit_X.size                       # stores every training epoch
    if isinstance(clf, SVC):
        return base + clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size
    if isinstance(clf, RandomForestClassifier):
        return sum(t.tree_.node_count * 3 for t in clf.estimators_)  # feature, threshold, value per node
    if isinstance(clf, LogisticRegression):
        return base + clf.coef_.size + clf.intercept_.size
    if isinstance(clf, MLPClassifier):
        return base + sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    return 0


def timed(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    out = json.load(open(os.path.join(HERE, "models", "metrics.json")))
    spec = joblib.load(os.path.join(HERE, "models", "ad_model.joblib"))["spec"]
    subs = train.load("alz", neuro.AD_CH)
    X = {s["id"]: neuro.ad_features(s["ep"], spec) for s in subs}
    Xall = np.vstack([X[s["id"]] for s in subs])
    yall = np.concatenate([[s["y"]] * len(X[s["id"]]) for s in subs])
    one_rec = X[subs[0]["id"]]                              # ~60-100 epochs = one recording
    print(f"{len(subs)} subjects, {len(Xall)} epochs, {Xall.shape[1]} features")

    # ======================= Objective 3: distance measures =======================
    dist = {"nearest_centroid": {}, "knn": {}}
    for name, metric in METRICS:
        dist["nearest_centroid"][name] = cv(subs, X, lambda m=metric: make_pipeline(StandardScaler(), CentroidClassifier(m)))
        dist["knn"][name] = cv(subs, X, lambda m=metric: knn(m), repeats=3)
        print(f"  {name:16s} NC bal.acc {dist['nearest_centroid'][name]['balanced_accuracy'][0]:.3f}"
              f"  kNN bal.acc {dist['knn'][name]['balanced_accuracy'][0]:.3f}")
    best = max(dist["nearest_centroid"], key=lambda k: (round(dist["nearest_centroid"][k]["balanced_accuracy"][0], 3),
                                                          dist["nearest_centroid"][k]["auc"][0]))
    dist["best_nearest_centroid"] = best
    dist["note"] = ("Hamming and Jaccard are defined for binary vectors, so each standardised feature is binarised "
                    "as above/below the training mean before those distances are used. k-NN uses k=15, 3 repeats.")
    out["distance_metrics"] = dist
    print("  best metric for nearest centroid:", best)

    # ======================= Objective 2: computational cost =======================
    models = {
        "Nearest Centroid, cosine (ours)": (lambda: make_pipeline(StandardScaler(), CentroidClassifier("cosine")), "O(n·d)", "O(c·d)", "O(c·d)", 0),
        "k-NN (k=15)": (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)), "O(1) (stores data)", "O(n·d)", "O(n·d)", 1),
        "Logistic Regression": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), "O(i·n·d)", "O(d)", "O(d)", 1),
        "SVM (RBF)": (lambda: make_pipeline(StandardScaler(), SVC()), "O(n²·d) to O(n³)", "O(s·d)", "O(s·d)", 2),
        "Random Forest (200 trees)": (lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1, random_state=0),
                                      "O(t·d·n·log n)", "O(t·log n)", "O(t·nodes)", 3),
        "MLP neural network (64-32)": (lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, random_state=0)),
                                       "O(i·n·w)", "O(w)", "O(w)", 4),
    }
    cost = {}
    for name, (make, t_train, t_pred, space, hp) in models.items():
        acc = cv(subs, X, make, repeats=3 if "MLP" in name else REPEATS)
        m = make().fit(Xall, yall)
        cost[name] = {
            "accuracy": acc,
            "train_s": timed(lambda: make().fit(Xall, yall), reps=3),
            "predict_ms_per_recording": timed(lambda: m.predict(one_rec), reps=20) * 1000,
            "model_bytes": len(pickle.dumps(m)),
            "parameters": int(n_params(m)),
            "big_o_train": t_train, "big_o_predict": t_pred, "big_o_space": space, "hyperparameters": hp,
            "input": "4 features / epoch",
        }
        c = cost[name]
        print(f"  {name:28s} bal.acc {acc['balanced_accuracy'][0]:.3f}  train {c['train_s']:.3f}s  "
              f"predict {c['predict_ms_per_recording']:.2f}ms  size {c['model_bytes']/1024:.1f}KB  params {c['parameters']}")

    # compact CNN on raw epochs (as used in the CNN literature)
    try:
        CNN = make_cnn()
        raw = {s["id"]: s["ep"] for s in subs}
        acc = cv(subs, raw, lambda: CNN(), repeats=1)
        Rall = np.vstack([raw[s["id"]] for s in subs])
        t0 = time.perf_counter(); cnn = CNN().fit(Rall, yall); t_train = time.perf_counter() - t0
        cost["Compact CNN (EEGNet-style)"] = {
            "accuracy": acc, "train_s": t_train,
            "predict_ms_per_recording": timed(lambda: cnn.predict(raw[subs[0]["id"]]), reps=5) * 1000,
            "model_bytes": cnn.size_bytes(), "parameters": int(cnn.n_params()),
            "big_o_train": "O(i·n·L·K)", "big_o_predict": "O(L·K)", "big_o_space": "O(w)", "hyperparameters": 8,
            "input": "raw 3 x 2500 samples / epoch",
        }
        c = cost["Compact CNN (EEGNet-style)"]
        print(f"  CNN bal.acc {acc['balanced_accuracy'][0]:.3f} train {c['train_s']:.1f}s predict {c['predict_ms_per_recording']:.1f}ms "
              f"size {c['model_bytes']/1024:.1f}KB params {c['parameters']}")
    except ImportError:
        print("  torch not installed: CNN skipped")

    # feature-extraction cost: 3-electrode vs full 19-channel montage, one recording
    full = {s["id"]: s for s in train.load("alz", neuro.ALL_1020)}
    ep3, ep19 = subs[0]["ep"], full[subs[0]["id"]]["ep"]
    out["feature_cost"] = {
        "three_electrode_ms": timed(lambda: neuro.ad_features(ep3, spec), reps=5) * 1000,
        "full_cap_ms": timed(lambda: train.fullcap(ep19), reps=5) * 1000,
        "three_electrode_features": 4, "full_cap_features": 95, "epochs": int(len(ep3)),
    }
    out["complexity"] = cost
    out["complexity_note"] = ("n = training epochs, d = features, c = classes, s = support vectors, t = trees, "
                              "i = iterations, w = network weights, L = samples per epoch, K = kernel size. "
                              "Times measured on this PC (CPU); accuracy is subject-level CV with epoch voting.")
    with open(os.path.join(HERE, "models", "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    figures(out)
    print("saved metrics + figures")


def figures(out):
    with plt.rc_context(train.DARK):
        d = out["distance_metrics"]
        names = [n for n, _ in METRICS]
        nc = [d["nearest_centroid"][n]["balanced_accuracy"][0] for n in names]
        kn = [d["knn"][n]["balanced_accuracy"][0] for n in names]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        x = np.arange(len(names)); w = 0.38
        b1 = ax.bar(x - w / 2, nc, w, color="#7A2E6B", label="Nearest centroid")
        b2 = ax.bar(x + w / 2, kn, w, color="#9AA5B1", label="k-NN (k=15)")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, names, rotation=15); ax.set_ylim(0.4, 0.95); ax.axhline(0.5, ls="--", color="#9AA5B1", lw=1)
        ax.set_ylabel("balanced accuracy"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Distance measure comparison (subject-level CV)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "distance_metrics.png"), dpi=130); plt.close(fig)

        c = out["complexity"]; names = list(c)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, key, lab, log in [(axes[0], "train_s", "training time (s)", True),
                                  (axes[1], "model_bytes", "model size (bytes)", True),
                                  (axes[2], "accuracy", "balanced accuracy", False)]:
            v = [c[n][key]["balanced_accuracy"][0] if key == "accuracy" else c[n][key] for n in names]
            cols = ["#7A2E6B" if "ours" in n else "#B9C6BE" for n in names]
            ax.barh(range(len(names)), v, color=cols)
            ax.set_yticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=7)
            ax.invert_yaxis(); ax.set_title(lab, fontsize=9)
            if log:
                ax.set_xscale("log")
            else:
                ax.set_xlim(0.4, 0.95)
            for i, val in enumerate(v):
                ax.text(val, i, f" {val:.2f}" if key == "accuracy" else (f" {val:.3g}"), va="center", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "complexity.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
