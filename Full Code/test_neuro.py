"""Signal-processing sanity checks on synthetic EEG.  Run:  python -m pytest test_neuro.py  (or python test_neuro.py)"""
import numpy as np
import neuro

FS, N = neuro.FS, neuro.FS * neuro.EPOCH_S
t = np.arange(N) / FS
rng = np.random.default_rng(1)


def sine(f, amp=20.0):
    return amp * np.sin(2 * np.pi * f * t)


def noise(amp=5.0):
    return amp * rng.standard_normal(N)


def epochs(*chans, n=4):
    return np.array([[c + noise(1) for c in chans] for _ in range(n)])


def test_alpha_dominant_signal_has_high_relative_alpha():
    f, p = neuro.spectra(epochs(sine(10))[:, 0])
    assert np.exp(neuro.rel_power(f, p, "alpha")).mean() > 0.8
    assert np.exp(neuro.rel_power(f, p, "beta")).mean() < 0.1


def test_slow_signal_increases_slowing_ratio():
    f, p1 = neuro.spectra(epochs(sine(3))[:, 0])
    f, p2 = neuro.spectra(epochs(sine(10))[:, 0])
    assert neuro.rel_power(f, p1, "slowing").mean() > neuro.rel_power(f, p2, "slowing").mean()


def test_coherence_identical_high_independent_low():
    x = noise(10)
    same = np.array([[x, x + noise(0.5)]] * 3)
    indep = np.array([[noise(10), noise(10)] for _ in range(3)])
    assert neuro.coherence(same[:, 0], same[:, 1], "alpha").mean() > 0.9
    assert neuro.coherence(indep[:, 0], indep[:, 1], "alpha").mean() < 0.2


def test_features_are_gain_invariant():
    """The old app used raw power, so a 2x amplifier gain flipped diagnoses. Features must not change."""
    ep = epochs(noise(10) + sine(9), noise(10) + sine(10), noise(10) + sine(11))
    spec = {"power": "theta/alpha", "coh": "alpha"}
    assert np.allclose(neuro.ad_features(ep, spec), neuro.ad_features(ep * 2, spec))
    ep4 = np.concatenate([ep, ep[:, :1]], axis=1)
    dspec = {"asym": "frontal alpha", "power": "alpha"}
    assert np.allclose(neuro.dep_features(ep4, dspec), neuro.dep_features(ep4 * 2, dspec))


def test_faa_sign_more_right_alpha_is_positive():
    """ln(F4) - ln(F3): positive when RIGHT alpha is larger (= relatively more LEFT activity = lower depression risk)."""
    ep = epochs(sine(10, 10), sine(10, 30), sine(10, 30), sine(10, 10))   # F3<F4, O1>O2
    t = neuro.dep_feature_table(ep)
    assert t[("asym", "frontal alpha")].mean() > 0 and t[("asym", "posterior alpha")].mean() < 0


def test_artifact_rejection():
    ep = epochs(noise(10), noise(10), noise(10), n=10)
    ep[3, 1, 100:200] += 400                     # blink/movement spike
    ep[6, 2] = 0                                 # dead channel
    ok = neuro.clean(ep)
    assert not ok[3] and not ok[6] and ok.sum() == 8


def test_prepare_epochs_and_missing_channel():
    import mne
    info = mne.create_info(["FZ", "o1", "O2", "F3", "F4"], 500.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((5, 500 * 35)) * 1e-5, info, verbose="ERROR")
    ep = neuro.prepare(raw)
    assert ep.shape == (3, 5, N)                 # 35 s -> 3 full 10 s epochs, case-insensitive names
    try:
        neuro.prepare(raw.copy().drop_channels(["F4"]))
        assert False, "should have raised"
    except ValueError as e:
        assert "F4" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
