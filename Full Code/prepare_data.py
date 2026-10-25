"""
Download + preprocess the two public OpenNeuro datasets used for training/validation.

  ds004504  Alzheimer's (A) vs healthy controls (C), 19-ch eyes-closed, preprocessed derivatives
  ds003478  Depression rest EEG, 64-ch, labelled by Beck Depression Inventory (BDI)

Each recording is reduced to 10 s eyes-closed epochs at 250 Hz (see neuro.prepare) and saved
as data/<dataset>/<subject>.npz; the large raw download is deleted afterwards.
Usage:  python prepare_data.py            (safe to re-run, skips finished subjects)
"""
import csv, io, os, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mne
import neuro

S3 = "https://s3.amazonaws.com/openneuro.org"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
mne.set_log_level("ERROR")


def fetch(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def tsv(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))


def save(path, epochs, **meta):
    np.savez_compressed(path, epochs=epochs.astype(np.float32), **meta)


def do_alz(row):
    sid, group = row["participant_id"], row["Group"]
    out = os.path.join(OUT, "alz", sid + ".npz")
    if os.path.exists(out):
        return sid, "skip"
    name = f"{sid}_task-eyesclosed_eeg.set"
    local = os.path.join(HERE, "Dataset", name)
    tmp = None
    if not os.path.exists(local):
        tmp = local = os.path.join(tempfile.gettempdir(), name)
        fetch(f"{S3}/ds004504/derivatives/{sid}/eeg/{name}", local)
    try:
        ep = neuro.prepare(neuro.load_raw(local), channels=neuro.ALL_1020)
        save(out, ep, ch=neuro.ALL_1020, label=int(group == "A"), mmse=float(row["MMSE"]), age=float(row["Age"]))
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return sid, f"{len(ep)} epochs"


def closed_segments(events):
    """ds003478 marks eyes-closed minutes with repeated 'Eyes Closed' triggers. Some subjects only
    carry the raw trigger codes (trial_type 'STATUS'); codes 11/13/15 are the eyes-closed 2 s ticks."""
    t = sorted(float(e["onset"]) for e in events
               if e["trial_type"].startswith("Eyes Closed") or e["value"] in ("11", "13", "15"))
    if not t:
        raise ValueError("no eyes-closed markers in events.tsv - subject skipped")
    segs, start = [], t[0]
    for a, b in zip(t, t[1:] + [None]):
        if b is None or b - a > 3.0:
            segs.append((start, a + 2.0))
            start = b
    return segs


def do_dep(row):
    sid = row["participant_id"]
    out = os.path.join(OUT, "dep", sid + ".npz")
    if os.path.exists(out):
        return sid, "skip"
    base = f"{S3}/ds003478/{sid}/eeg/{sid}_task-Rest_run-01_"
    tmpdir = tempfile.mkdtemp()
    try:
        # EEGLAB .set points at its .fdt by filename, so both must sit side by side
        for ext in ("eeg.set", "eeg.fdt"):
            fetch(base + ext, os.path.join(tmpdir, f"{sid}_task-Rest_run-01_{ext}"))
        raw = neuro.load_raw(os.path.join(tmpdir, f"{sid}_task-Rest_run-01_eeg.set"))
        # re-reference to linked mastoids to match ds004504 (A1-A2)
        raw.set_eeg_reference([c for c in raw.ch_names if c.upper() in ("M1", "M2")])
        ep = neuro.prepare(raw, segments=closed_segments(tsv(base + "events.tsv")))
        save(out, ep, ch=neuro.MONTAGE, bdi=float(row["BDI"]), label=int(float(row["BDI"]) >= 13), age=float(row["age"]))
    finally:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
    return sid, f"{len(ep)} epochs"


def run(fn, rows):
    with ThreadPoolExecutor(4) as pool:
        for sid, msg in pool.map(lambda r: _safe(fn, r), rows):
            print(sid, msg, flush=True)


def _safe(fn, row):
    try:
        return fn(row)
    except Exception as e:  # one bad subject must not kill a 1-hour download
        return row["participant_id"], f"FAILED: {e!r}"


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "alz"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "dep"), exist_ok=True)
    which = sys.argv[1:] or ["alz", "dep"]
    if "alz" in which:
        rows = [r for r in tsv(f"{S3}/ds004504/participants.tsv") if r["Group"] in ("A", "C")]
        run(do_alz, rows)
    if "dep" in which:
        # BDI <= 7 = low (control), >= 13 = high (depressed); in-between and missing excluded.
        # Original_ID 544 excluded by the dataset author (unstable BDI).
        rows = [r for r in tsv(f"{S3}/ds003478/participants.tsv")
                if r["BDI"] not in ("NaN", "n/a", "") and r["Original_ID"] != "544"
                and (float(r["BDI"]) <= 7 or float(r["BDI"]) >= 13)]
        run(do_dep, rows)
