"""
NeuroDetect signal-processing core. Shared by prepare_data.py, train.py and app.py so the web
app computes features exactly the way the models were trained.

Pipeline: pick channels -> 0.5-45 Hz band-pass -> 250 Hz -> 10 s epochs (uV)
          -> amplitude-based artifact rejection -> per-epoch features -> per-epoch vote.
All features are relative powers, log-ratios or coherences, so they do not depend on
amplifier gain / units (the old raw-FFT alpha power did).
"""
import numpy as np
import mne
from scipy import signal

FS = 250
EPOCH_S = 10
REJECT_UV = 500.0   # hard peak-to-peak ceiling (uV) per epoch
FLAT_UV = 0.5

AD_CH = ["Fz", "O1", "O2"]          # Alzheimer's module montage
DEP_CH = ["F3", "F4", "O1", "O2"]   # depression module (frontal + posterior asymmetry)
MONTAGE = ["Fz", "O1", "O2", "F3", "F4"]   # 5 of the 19 standard 10-20 electrodes
ALL_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
# power candidates for the band search: single relative bands + two classic slowing ratios
POWER_FEATS = list(BANDS) + ["theta/alpha", "slowing"]


# ---------------------------------------------------------------- loading / epoching
def load_raw(path):
    return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")


def _match(raw, wanted):
    lut = {c.upper(): c for c in raw.ch_names}
    missing = [w for w in wanted if w.upper() not in lut]
    if missing:
        raise ValueError(f"EEG file is missing required channel(s): {', '.join(missing)}")
    return [lut[w.upper()] for w in wanted]


def prepare(raw, channels=MONTAGE, segments=None):
    """Return epochs array (n_epochs, len(channels), FS*EPOCH_S) in microvolts.
    segments: optional [(t_start, t_end), ...] in seconds to epoch within (e.g. eyes-closed spans)."""
    raw = raw.copy().pick(_match(raw, channels))
    raw.filter(0.5, 45.0, verbose="ERROR")
    raw.resample(FS, verbose="ERROR")
    data = raw.get_data() * 1e6
    n = FS * EPOCH_S
    segments = segments or [(0.0, data.shape[1] / FS)]
    out = []
    for t0, t1 in segments:
        a, b = int(t0 * FS), min(int(t1 * FS), data.shape[1])
        out += [data[:, i:i + n] for i in range(a, b - n + 1, n)]
    return np.array(out).reshape(-1, len(channels), n)


def clean(epochs):
    """Boolean mask of epochs without blinks/movement or dead channels.
    Adaptive: an epoch is bad if any channel's peak-to-peak is > 3x that channel's median
    (AD recordings are legitimately high-amplitude, so a fixed uV limit would reject them),
    above a hard REJECT_UV ceiling, or flat."""
    ptp = np.ptp(epochs, axis=2)
    med = np.median(ptp, axis=0)
    return ((ptp < 3 * med).all(1) & (ptp.max(1) < REJECT_UV) & (ptp.min(1) > FLAT_UV))


# ---------------------------------------------------------------- features
def spectra(x):
    """Welch PSD with 0.5 Hz resolution along the last axis."""
    return signal.welch(x, FS, nperseg=2 * FS, axis=-1)


def _band(f, lo, hi):
    return (f >= lo) & (f < hi)


def rel_power(f, p, name):
    """log relative band power (or log ratio) along the last axis."""
    tot = p[..., _band(f, 1, 45)].sum(-1)
    bp = {k: p[..., _band(f, *v)].sum(-1) / tot for k, v in BANDS.items()}
    if name == "theta/alpha":
        return np.log(bp["theta"] / bp["alpha"])
    if name == "slowing":
        return np.log((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))
    return np.log(bp[name])


def coherence(x, y, band):
    """Mean magnitude-squared coherence (0..1) in a band, per epoch."""
    f, c = signal.coherence(x, y, FS, nperseg=2 * FS, axis=-1)
    return c[..., _band(f, *BANDS[band])].mean(-1)


def ad_feature_table(ep):
    """All candidate Alzheimer's features for epochs ordered like AD_CH (Fz, O1, O2).
    Returns {("pow", name): (n, 2) for O1,O2; ("coh", band): (n, 2) for Fz-O1, Fz-O2}."""
    f, p = spectra(ep[:, 1:3])
    t = {("pow", k): rel_power(f, p, k) for k in POWER_FEATS}
    for b in BANDS:
        t[("coh", b)] = np.stack([coherence(ep[:, 0], ep[:, i], b) for i in (1, 2)], axis=1)
    return t


def ad_features(ep, spec):
    """Only the selected features (same values as ad_feature_table, ~5x less work): one Welch spectrum per
    channel and one cross-spectrum per Fz-O pair, reused for both relative power and coherence."""
    f, p = spectra(ep)                                               # (n, 3, freqs): Fz, O1, O2
    _, pxy = signal.csd(ep[:, :1], ep[:, 1:3], FS, nperseg=2 * FS, axis=-1)
    coh = np.abs(pxy) ** 2 / (p[:, :1] * p[:, 1:3])
    return np.hstack([rel_power(f, p[:, 1:3], spec["power"]), coh[..., _band(f, *BANDS[spec["coh"]])].mean(-1)])


AD_FEATURE_NAMES = lambda s: [f"{s['power']} power O1", f"{s['power']} power O2",
                              f"{s['coh']} coherence Fz-O1", f"{s['coh']} coherence Fz-O2"]


def dep_feature_table(ep):
    """Candidate depression features for epochs ordered like DEP_CH (F3, F4, O1, O2).
    ("asym", "<region> <band>"): ln(right) - ln(left) band power, region frontal (F4-F3; the classic
    frontal alpha asymmetry when band=alpha) or posterior (O2-O1);
    ("pow", name): mean frontal (F3, F4) log relative power."""
    f, p = spectra(ep)
    t = {}
    for b in BANDS:
        bp = np.log(p[..., _band(f, *BANDS[b])].sum(-1))
        t[("asym", f"frontal {b}")] = (bp[:, 1] - bp[:, 0])[:, None]
        t[("asym", f"posterior {b}")] = (bp[:, 3] - bp[:, 2])[:, None]
    for k in POWER_FEATS:
        t[("pow", k)] = rel_power(f, p[:, :2], k).mean(1, keepdims=True)
    return t


ASYM_FEATS = [f"{r} {b}" for r in ("frontal", "posterior") for b in BANDS]


def dep_features(ep, spec):
    t = dep_feature_table(ep)
    cols = [t[("asym", spec["asym"])]] + ([t[("pow", spec["power"])]] if spec.get("power") else [])
    return np.hstack(cols)


DEP_FEATURE_NAMES = lambda s: [f"{s['asym']} asymmetry (right-left)"] + ([f"frontal {s['power']} power"] if s.get("power") else [])


# ---------------------------------------------------------------- inference
class CentroidClassifier:
    """Nearest-centroid classifier with a selectable distance (scikit-learn compatible).
    Centroid = mean (euclidean, cosine, minkowski), median (manhattan, the L1-optimal centre) or majority bit
    (hamming, jaccard: standardised features binarised as above/below the training mean).
    The deployed Alzheimer's model uses cosine distance, chosen in experiments.py (Objective 3)."""
    BINARY = ("hamming", "jaccard")

    def __init__(self, metric="cosine", p=3):
        self.metric, self.p = metric, p

    def get_params(self, deep=True):
        return {"metric": self.metric, "p": self.p}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        return X > 0 if self.metric in self.BINARY else X

    def fit(self, X, y):
        X, y = self._prep(X), np.asarray(y)
        self.classes_ = np.unique(y)
        centre = {"manhattan": lambda a: np.median(a, 0)}.get(self.metric, lambda a: a.mean(0))
        self.centroids_ = np.array([(X[y == c].mean(0) >= 0.5) if self.metric in self.BINARY else centre(X[y == c])
                                    for c in self.classes_])
        return self

    def distances(self, X):
        from scipy.spatial.distance import cdist
        name = "cityblock" if self.metric == "manhattan" else self.metric
        kw = {"p": self.p} if self.metric == "minkowski" else {}
        return cdist(self._prep(X).astype(float), self.centroids_.astype(float), metric=name, **kw)

    def predict(self, X):
        return self.classes_[np.argmin(self.distances(X), axis=1)]


def explain(model, x):
    """Distance of a feature vector to each class centroid (in the model's own metric) + per-feature pull.
    Positive pull = that feature is closer to the 'condition' centroid (standardised units)."""
    sc, clf = model.named_steps["scale"], model.named_steps["classifier"]
    z = sc.transform(x.reshape(1, -1))[0]
    c0, c1 = clf.centroids_
    if hasattr(clf, "distances"):
        d0, d1 = clf.distances(z[None])[0]
    else:
        d0, d1 = np.linalg.norm(z - c0), np.linalg.norm(z - c1)
    return {"dist_control": float(d0), "dist_condition": float(d1),
            "pull": (np.abs(z - c0) - np.abs(z - c1)).tolist()}


def vote(model, X):
    """Per-epoch predictions -> % of clean epochs voting for the condition."""
    y = model.predict(X)
    return float(y.mean() * 100), y.tolist()
