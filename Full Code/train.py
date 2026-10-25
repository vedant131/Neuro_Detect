"""
Train + validate both NeuroDetect models. Everything reported on the Analysis page comes from here.

Validation is subject-level (no epoch of a test subject is ever seen in training) with
10 x repeated stratified 5-fold CV. Band selection (Objective 2) happens INSIDE each training
fold (nested CV), so the reported numbers are not inflated by the search.

Outputs: models/ad_model.joblib, models/dep_model.joblib, models/metrics.json,
         static/img/results/*.png
Usage:   python train.py
"""
import glob, json, os, itertools
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import neuro

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "static", "img", "results")
REPEATS, FOLDS = 10, 5
rng = np.random.default_rng(0)


METRIC = "cosine"   # distance for the Alzheimer's model, chosen in experiments.py (Objective 3)


def model():
    clf = neuro.CentroidClassifier(METRIC) if METRIC != "euclidean" else NearestCentroid()
    return Pipeline([("scale", StandardScaler()), ("classifier", clf)])


# ------------------------------------------------------------------ data
def load(kind, ch):
    subs = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", kind, "*.npz"))):
        d = np.load(f)
        idx = [list(d["ch"]).index(c) for c in ch]
        ep = d["epochs"][:, idx].astype(np.float64)
        ok = neuro.clean(ep)
        if ok.sum() < 3:
            continue
        subs.append({"id": os.path.basename(f)[:-4], "y": int(d["label"]), "ep": ep[ok],
                     "all_ep": ep, "meta": {k: float(d[k]) for k in d.files if k in ("mmse", "bdi", "age")}})
    return subs


# ------------------------------------------------------------------ legacy features (old app.py), for comparison
def legacy_ad(ep):
    n = ep.shape[-1]
    fr = np.fft.rfftfreq(n, 1 / neuro.FS)
    p = np.abs(np.fft.rfft(ep[:, 1:3], axis=-1)) ** 2
    alpha = p[..., (fr >= 8) & (fr <= 12)].mean(-1)
    coh = []
    for i in (1, 2):
        f, c = signal.coherence(ep[:, 0], ep[:, i], neuro.FS, axis=-1)
        coh.append(c[:, (f >= 13) & (f <= 20)].mean(-1) / c.mean(-1))
    return np.column_stack([alpha, np.stack(coh, 1)])


def fullcap(ep):
    f, p = neuro.spectra(ep)
    return np.hstack([neuro.rel_power(f, p, b) for b in neuro.BANDS])


# ------------------------------------------------------------------ CV engine
def subject_votes(m, subs, X):
    return np.array([m.predict(X[s["id"]]).mean() for s in subs])


def metrics(y, score):
    pred = (score > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2, "precision": prec,
            "f1": 2 * prec * sens / max(prec + sens, 1e-9),
            "auc": roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"),
            "cm": [[tn, fp], [fn, tp]]}


def fit(subs, X):
    m = model()
    m.fit(np.vstack([X[s["id"]] for s in subs]), np.concatenate([[s["y"]] * len(X[s["id"]]) for s in subs]))
    return m


def select(subs, tables, specs, seed):
    """Inner CV: pick the spec with best subject-level balanced accuracy (ties -> higher AUC)."""
    y = np.array([s["y"] for s in subs])
    best, scores = None, {}
    for spec in specs:
        X = {s["id"]: tables[s["id"]](spec) for s in subs}
        sc = np.zeros(len(subs))
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(y, y):
            m = fit([subs[i] for i in tr], X)
            sc[te] = subject_votes(m, [subs[i] for i in te], X)
        r = metrics(y, sc)
        scores[spec] = (r["balanced_accuracy"], r["auc"])
        if best is None or scores[spec] > scores[best]:
            best = spec
    return best, scores


def nested_cv(subs, tables, specs, label):
    """Returns per-repeat metrics, pooled out-of-fold scores (repeat 0), spec choice counts."""
    y = np.array([s["y"] for s in subs])
    reps, chosen, oof0, folds0 = [], {}, None, []
    for r in range(REPEATS):
        oof = np.zeros(len(subs))
        for k, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=r).split(y, y)):
            trs = [subs[i] for i in tr]
            spec = specs[0] if len(specs) == 1 else select(trs, tables, specs, seed=100 + r)[0]
            chosen[spec] = chosen.get(spec, 0) + 1
            X = {s["id"]: tables[s["id"]](spec) for s in subs}
            m = fit(trs, X)
            oof[te] = subject_votes(m, [subs[i] for i in te], X)
            if r == 0:
                folds0.append((tr, te, spec))
        reps.append(metrics(y, oof))
        if r == 0:
            oof0 = oof
    summary = {k: [float(np.mean([m[k] for m in reps])), float(np.std([m[k] for m in reps]))]
               for k in reps[0] if k != "cm"}
    cm = np.sum([m["cm"] for m in reps], axis=0).tolist()   # summed over repeats
    print(f"  {label:<38} bal.acc {summary['balanced_accuracy'][0]:.3f}±{summary['balanced_accuracy'][1]:.3f}"
          f"  sens {summary['sensitivity'][0]:.3f}  spec {summary['specificity'][0]:.3f}  AUC {summary['auc'][0]:.3f}")
    return {"summary": summary, "cm_summed": cm, "n_evaluations": len(reps) * len(subs),
            "chosen": {str(k): v for k, v in chosen.items()}}, oof0, folds0


# ------------------------------------------------------------------ Objective 3: artifact robustness
def corrupt(ep, frac):
    """Inject realistic artifacts into a fraction of epochs: movement step/drift + EMG burst."""
    ep = ep.copy()
    n = ep.shape[-1]; t = np.arange(n)
    for i in rng.choice(len(ep), max(1, int(frac * len(ep))), replace=False):
        drift = 150 * np.sin(2 * np.pi * 0.7 * t / neuro.FS + rng.uniform(0, 6))
        emg = 25 * rng.standard_normal((ep.shape[1], n))
        emg = signal.sosfiltfilt(signal.butter(4, [20, 45], "bandpass", fs=neuro.FS, output="sos"), emg)
        ep[i] += drift + emg
    return ep


def robustness(subs, folds, tables, feat, frac=0.3):
    """Same outer folds; test subjects get artifacts in `frac` of epochs. Compare:
       epoch voting + rejection (NeuroDetect) / epoch voting only / whole-signal averaging."""
    y = np.array([s["y"] for s in subs])
    res = {"vote+reject": np.zeros(len(subs)), "vote": np.zeros(len(subs)), "whole-signal": np.zeros(len(subs))}
    for tr, te, spec in folds:
        X = {s["id"]: tables[s["id"]](spec) for s in subs}
        m = fit([subs[i] for i in tr], X)
        for i in te:
            bad = corrupt(subs[i]["ep"], frac)
            Xb = feat(bad, spec)
            res["vote"][i] = m.predict(Xb).mean()
            res["vote+reject"][i] = m.predict(Xb[neuro.clean(bad)]).mean() if neuro.clean(bad).any() else 0.5
            res["whole-signal"][i] = m.predict(Xb.mean(0, keepdims=True))[0]
    clean_whole = np.zeros(len(subs))
    for tr, te, spec in folds:
        X = {s["id"]: tables[s["id"]](spec) for s in subs}
        m = fit([subs[i] for i in tr], X)
        for i in te:
            clean_whole[i] = m.predict(X[subs[i]["id"]].mean(0, keepdims=True))[0]
    out = {k: metrics(y, v)["balanced_accuracy"] for k, v in res.items()}
    out["whole-signal (clean data)"] = metrics(y, clean_whole)["balanced_accuracy"]
    return out


# ------------------------------------------------------------------ figures
DARK = {"figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#C9D6CE",   # light "chart paper" theme
        "axes.labelcolor": "#15233B", "xtick.color": "#4A5568", "ytick.color": "#4A5568",
        "text.color": "#15233B", "savefig.facecolor": "#FFFFFF", "font.size": 10}


def fig_cm(cm, labels, title, path):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        cm = np.array(cm); pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Greens", vmin=0, vmax=1)
        for i, j in itertools.product(range(2), range(2)):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "#15233B", fontsize=11)
        ax.set_xticks([0, 1], [f"pred {labels[0]}", f"pred {labels[1]}"])
        ax.set_yticks([0, 1], [f"true {labels[0]}", f"true {labels[1]}"])
        ax.set_title(title, fontsize=10)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_roc(curves, path):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        for name, (y, s, color) in curves.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {roc_auc_score(y, s):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#9AA5B1")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8, facecolor="#FFFFFF")
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_grid(scores, rows, cols, xlabel, ylabel, title, path):
    g = np.array([[scores[(r, c)][0] for c in cols] for r in rows])
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        im = ax.imshow(g, cmap="PuRd", vmin=max(0.4, g.min()), vmax=g.max())
        for i, j in itertools.product(range(len(rows)), range(len(cols))):
            ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if g[i, j] > (g.min() + g.max()) / 2 else "#15233B")
        ax.set_xticks(range(len(cols)), cols, rotation=30); ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_bars(values, title, path, color="#7A2E6B"):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        names = list(values); v = [values[k] for k in names]
        ax.barh(names, v, color=color)
        for i, x in enumerate(v):
            ax.text(x + 0.01, i, f"{x:.2f}", va="center", fontsize=9)
        ax.axvline(0.5, ls="--", color="#9AA5B1"); ax.set_xlim(0.3, 1.05)
        ax.set_xlabel("balanced accuracy (subject level)"); ax.set_title(title, fontsize=10)
        ax.invert_yaxis(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global METRIC
    METRIC = "cosine"
    os.makedirs(FIG, exist_ok=True); os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    out = {}

    # ============ Alzheimer's vs healthy (ds004504) ============
    ad = load("alz", neuro.AD_CH)
    full = {s["id"]: s for s in load("alz", neuro.ALL_1020)}
    print(f"AD dataset: {sum(s['y'] for s in ad)} AD, {sum(1 - s['y'] for s in ad)} healthy")
    tab = {s["id"]: neuro.ad_feature_table(s["ep"]) for s in ad}
    ad_specs = [(p, c) for p in neuro.POWER_FEATS for c in neuro.BANDS]
    ad_tables = {s["id"]: (lambda sid: lambda sp: np.hstack([tab[sid][("pow", sp[0])], tab[sid][("coh", sp[1])]]))(s["id"]) for s in ad}
    legacy = {s["id"]: legacy_ad(s["ep"]) for s in ad}
    fc = {s["id"]: fullcap(full[s["id"]]["ep"]) for s in ad if s["id"] in full}
    ad_full = [s for s in ad if s["id"] in fc]

    print("Alzheimer's module (nested, subject-level CV):")
    res_opt, oof_ad, folds_ad = nested_cv(ad, ad_tables, ad_specs, "optimised 3-electrode (nested search)")
    res_leg, oof_leg, _ = nested_cv(ad, {k: (lambda v: lambda sp: v)(v) for k, v in legacy.items()}, ["legacy"], "old app features (raw alpha + coh ratio)")
    res_alpha, _, _ = nested_cv(ad, ad_tables, [("alpha", "beta")], "fixed alpha power + beta coherence")
    res_full, oof_full, _ = nested_cv(ad_full, {k: (lambda v: lambda sp: v)(v) for k, v in fc.items()}, ["full"], "full 19-channel cap, 5 bands (95 feats)")

    spec, grid = select(ad, ad_tables, ad_specs, seed=999)
    final = fit(ad, {s["id"]: ad_tables[s["id"]](spec) for s in ad})
    ad_spec = {"power": spec[0], "coh": spec[1]}
    joblib.dump({"model": final, "spec": ad_spec, "features": neuro.AD_FEATURE_NAMES(ad_spec)},
                os.path.join(HERE, "models", "ad_model.joblib"))
    print(f"  final AD features: {neuro.AD_FEATURE_NAMES(ad_spec)}")

    feat = lambda ep, sp: neuro.ad_features(ep, {"power": sp[0], "coh": sp[1]})
    rob = robustness(ad, folds_ad, ad_tables, feat)
    print("  artifact robustness (30% of test epochs corrupted):", {k: round(v, 3) for k, v in rob.items()})

    c = final.named_steps["scale"].inverse_transform(final.named_steps["classifier"].centroids_)
    y_ad = np.array([s["y"] for s in ad])
    out["alzheimers"] = {
        "dataset": "OpenNeuro ds004504 (eyes-closed, 19-ch, 500 Hz, preprocessed derivatives)",
        "n_subjects": {"AD": int(y_ad.sum()), "healthy": int(len(y_ad) - y_ad.sum())},
        "n_epochs": int(sum(len(s["ep"]) for s in ad)),
        "electrodes": neuro.AD_CH, "selected": ad_spec, "feature_names": neuro.AD_FEATURE_NAMES(ad_spec),
        "centroids": {"healthy": c[0].tolist(), "alzheimers": c[1].tolist()},
        "validation": f"{REPEATS}x repeated stratified {FOLDS}-fold cross-validation with nested band search", "repeats": REPEATS,
        "optimised": res_opt, "legacy": res_leg, "fixed_alpha_beta": res_alpha, "full_cap": res_full,
        "robustness_bal_acc": rob,
        "oof_votes_repeat0": {s["id"]: {"y": s["y"], "vote": float(v), "mmse": s["meta"].get("mmse")} for s, v in zip(ad, oof_ad)},
        "search_grid": {f"{p}|{c}": v[0] for (p, c), v in grid.items()},
    }
    fig_cm(res_opt["cm_summed"], ["Healthy", "AD"], f"Alzheimer's: out-of-fold, {REPEATS} repeats", os.path.join(FIG, "ad_confusion.png"))
    fig_grid(grid, neuro.POWER_FEATS, list(neuro.BANDS), "coherence band (Fz-O1/O2)", "occipital power feature",
             "Band search: CV balanced accuracy", os.path.join(FIG, "ad_band_search.png"))
    fig_bars({"Old app features": res_leg["summary"]["balanced_accuracy"][0],
              "Fixed alpha + beta coh": res_alpha["summary"]["balanced_accuracy"][0],
              "Optimised 3-electrode": res_opt["summary"]["balanced_accuracy"][0],
              "Full 19-channel cap": res_full["summary"]["balanced_accuracy"][0]},
             "Feature refinement vs baselines", os.path.join(FIG, "ad_comparison.png"))
    fig_bars({"Whole-signal, clean data": rob["whole-signal (clean data)"],
              "Whole-signal + artifacts": rob["whole-signal"],
              "Epoch voting + artifacts": rob["vote"],
              "Voting + rejection + artifacts": rob["vote+reject"]},
             "Artifact robustness (30% epochs corrupted)", os.path.join(FIG, "robustness.png"), color="#15233B")

    # ============ Depression high vs low BDI (ds003478) ============
    METRIC = "euclidean"   # depression model has a single feature, where cosine distance is meaningless
    dep = load("dep", neuro.DEP_CH)
    print(f"Depression dataset: {sum(s['y'] for s in dep)} high-BDI, {sum(1 - s['y'] for s in dep)} low-BDI")
    dtab = {s["id"]: neuro.dep_feature_table(s["ep"]) for s in dep}
    dep_specs = [(a, p) for a in neuro.ASYM_FEATS for p in neuro.POWER_FEATS]
    dep_tables = {s["id"]: (lambda sid: lambda sp: np.hstack([dtab[sid][("asym", sp[0])], dtab[sid][("pow", sp[1])]]))(s["id"]) for s in dep}
    print("Depression module (nested, subject-level CV):")
    dres, oof_dep, _ = nested_cv(dep, dep_tables, dep_specs, "optimised FAA + frontal power (nested)")
    dres_faa, oof_faa, _ = nested_cv(dep, {s["id"]: (lambda sid: lambda sp: dtab[sid][("asym", "frontal alpha")])(s["id"]) for s in dep}, ["faa"], "classic alpha FAA only")

    # old formula: O1/O2 asymmetry of first epoch -> 50 + 200*asym, "depressed" if > 50
    ydep = np.array([s["y"] for s in dep])
    occ = load("dep", ["O1", "O2"])
    occ_score = []
    for s in occ:
        f, p = neuro.spectra(s["ep"][:1])
        a = p[..., (f >= 8) & (f <= 12)].sum(-1)[0]
        occ_score.append(float(np.clip(50 + 200 * (a[1] - a[0]) / (a[1] + a[0]), 0, 100)) / 100)
    yo = np.array([s["y"] for s in occ])
    old_dep = metrics(yo, np.array(occ_score))

    _, dgrid = select(dep, dep_tables, dep_specs, seed=999)
    # The nested search did not beat the literature-standard frontal alpha asymmetry (FAA) on held-out
    # subjects, i.e. the search only overfits here. Ship the simpler, theory-driven FAA model.
    use_faa = dres_faa["summary"]["balanced_accuracy"][0] >= dres["summary"]["balanced_accuracy"][0]
    dep_spec = {"asym": "frontal alpha", "power": None} if use_faa else None
    if not use_faa:
        dspec, _ = select(dep, dep_tables, dep_specs, seed=999)
        dep_spec = {"asym": dspec[0], "power": dspec[1]}
    oof_dep = oof_faa if use_faa else oof_dep
    dfinal = fit(dep, {s["id"]: neuro.dep_features(s["ep"], dep_spec) for s in dep})
    joblib.dump({"model": dfinal, "spec": dep_spec, "features": neuro.DEP_FEATURE_NAMES(dep_spec)},
                os.path.join(HERE, "models", "dep_model.joblib"))
    print(f"  final depression features: {neuro.DEP_FEATURE_NAMES(dep_spec)}")
    print(f"  old app formula: bal.acc {old_dep['balanced_accuracy']:.3f} AUC {old_dep['auc']:.3f}")
    dc = dfinal.named_steps["scale"].inverse_transform(dfinal.named_steps["classifier"].centroids_)
    out["depression"] = {
        "dataset": "OpenNeuro ds003478 (Cavanagh depression rest EEG, eyes-closed minutes, re-referenced to mastoids)",
        "label": "high BDI (>= 13) vs low BDI (<= 7)",
        "n_subjects": {"high_BDI": int(ydep.sum()), "low_BDI": int(len(ydep) - ydep.sum())},
        "n_epochs": int(sum(len(s["ep"]) for s in dep)),
        "electrodes": neuro.DEP_CH, "selected": dep_spec, "feature_names": neuro.DEP_FEATURE_NAMES(dep_spec),
        "centroids": {"low_BDI": dc[0].tolist(), "high_BDI": dc[1].tolist()},
        "optimised": dres_faa if use_faa else dres, "searched": dres, "classic_faa": dres_faa,
        "shipped": "classic frontal alpha asymmetry" if use_faa else "nested band search",
        "old_formula": {k: v for k, v in old_dep.items()},
        "oof_votes_repeat0": {s["id"]: {"y": s["y"], "vote": float(v), "bdi": s["meta"].get("bdi")} for s, v in zip(dep, oof_dep)},
        "search_grid": {f"{a}|{p}": v[0] for (a, p), v in dgrid.items()},
    }
    fig_cm((dres_faa if use_faa else dres)["cm_summed"], ["Low BDI", "High BDI"], f"Depression: out-of-fold, {REPEATS} repeats", os.path.join(FIG, "dep_confusion.png"))
    fig_grid(dgrid, neuro.ASYM_FEATS, neuro.POWER_FEATS, "frontal power feature", "asymmetry (right-left)",
             "Band search: CV balanced accuracy", os.path.join(FIG, "dep_band_search.png"))
    fig_roc({"Alzheimer's (3 electrodes)": (y_ad, oof_ad, "#7A2E6B"),
             "Old AD features": (y_ad, oof_leg, "#9AA5B1"),
             "Depression module": (ydep, oof_dep, "#B86E0C")}, os.path.join(FIG, "roc.png"))

    with open(os.path.join(HERE, "models", "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved models/ + static/img/results/")


if __name__ == "__main__":
    main()
