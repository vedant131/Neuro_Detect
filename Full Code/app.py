"""
NeuroDetect web portal.
One EEG upload -> signal QC + plot -> Alzheimer's module (Fz, O1, O2) + depression module (F3, F4, O1, O2)
-> combined differential-screening report. Models and metrics come from train.py.
Run:  python app.py      (then open http://127.0.0.1:5000)
"""
import base64, io, json, os, secrets, shutil, sqlite3, uuid
from functools import wraps

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import neuro

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(HERE, "uploads")
DB_PATH = os.path.join(HERE, "instance", "users.db")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("NEURODETECT_SECRET") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024


PLAIN = {  # everyday names for the biomarkers shown in reports
    "theta/alpha power O1": "Brain-wave slowing, back left",
    "theta/alpha power O2": "Brain-wave slowing, back right",
    "theta coherence Fz-O1": "Front-to-back teamwork, left",
    "theta coherence Fz-O2": "Front-to-back teamwork, right",
    "frontal alpha asymmetry (right-left)": "Left-right balance, front of brain",
}
app.jinja_env.filters["plain"] = lambda name: PLAIN.get(name, name)


def _load(name):
    path = os.path.join(HERE, "models", name)
    return joblib.load(path) if os.path.exists(path) else None


AD_MODEL, DEP_MODEL = _load("ad_model.joblib"), _load("dep_model.joblib")
try:
    with open(os.path.join(HERE, "models", "metrics.json")) as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = None


# ------------------------------------------------------------------ auth
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, email TEXT NOT NULL, pw TEXT NOT NULL)")
    return con


def login_required(view):
    @wraps(view)
    def wrapper(*a, **kw):
        if not session.get("user"):
            flash("Please log in to use the EEG workspace.")
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapper


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        pw = request.form.get("password", "")
        if not (3 <= len(user) <= 40 and all(c.isalnum() or c in "_.-" for c in user)):
            flash("Username must be 3-40 letters, digits, _ . or -")
        elif "@" not in email or len(pw) < 8:
            flash("Enter a valid email and a password of at least 8 characters.")
        else:
            try:
                with db() as con:
                    con.execute("INSERT INTO users VALUES (?, ?, ?)", (user, email, generate_password_hash(pw)))
                flash("Account created. Please log in.", "ok")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is taken.")
    return render_template("registration.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        with db() as con:
            row = con.execute("SELECT pw FROM users WHERE username = ?", (user,)).fetchone()
        if row and check_password_hash(row[0], request.form.get("password", "")):
            session.clear()
            session["user"] = user
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("eeg_interface"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ------------------------------------------------------------------ analysis
def plot_signal(raw):
    picks = [c for c in raw.ch_names if c.upper() in {m.upper() for m in neuro.MONTAGE}] or raw.ch_names[:5]
    seg = raw.copy().pick(picks).filter(0.5, 45, verbose="ERROR")
    t0 = 2.0 if raw.times[-1] > 12 else 0.0   # skip zero-padded start of many recordings
    seg.crop(t0, min(t0 + 10, raw.times[-1]))
    d = seg.get_data() * 1e6
    fig, ax = plt.subplots(figsize=(11, 4))
    step = max(np.ptp(d, axis=1).max(), 1)
    for i, (name, x) in enumerate(zip(seg.ch_names, d)):
        ax.plot(seg.times, x - x.mean() - i * step, lw=0.8,
                color="#7A2E6B" if name.upper() in {c.upper() for c in neuro.AD_CH} else "#B86E0C")
        ax.text(-0.15, -i * step, name, ha="right", va="center", color="#15233B", fontsize=11, fontweight="bold")
    ax.set_yticks([]); ax.set_xlabel("seconds", color="#4A5568")
    ax.grid(True, color="#F3D6D6", lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors="#4A5568")
    for s in ax.spines.values():
        s.set_color("#C9D6CE")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def run_module(raw, bundle, channels, featfn):
    """Epoch -> reject artifacts -> features -> per-epoch votes + centroid explanation."""
    if bundle is None:
        return {"error": "Model not trained yet - run  python train.py"}
    try:
        ep = neuro.prepare(raw, channels)
    except ValueError as e:
        return {"error": str(e)}
    ok = neuro.clean(ep)
    if ok.sum() < 3:
        return {"error": f"Only {int(ok.sum())} artifact-free 10 s epochs found (need at least 3)."}
    X = featfn(ep[ok], bundle["spec"])
    pct, votes = neuro.vote(bundle["model"], X)
    exp = neuro.explain(bundle["model"], X.mean(0))
    c = bundle["model"].named_steps["scale"].inverse_transform(bundle["model"].named_steps["classifier"].centroids_)
    timeline, j = [], 0
    for good in ok:
        timeline.append(votes[j] if good else None)
        j += bool(good)
    return {"pct": pct, "n_epochs": int(len(ok)), "n_clean": int(ok.sum()), "timeline": timeline,
            "features": [{"name": n, "value": float(v), "control": float(c0), "condition": float(c1), "pull": float(p)}
                         for n, v, c0, c1, p in zip(bundle["features"], X.mean(0), c[0], c[1], exp["pull"])],
            "dist_control": exp["dist_control"], "dist_condition": exp["dist_condition"]}


def interpret(ad, dep):
    """Objective 1: combine both checks into one plain-language summary (title, text, colour)."""
    if "error" in ad:
        return None
    a = ad["pct"] > 50
    bacc = METRICS["depression"]["optimised"]["summary"]["balanced_accuracy"][0] * 100 if METRICS else 50
    mood_note = ("The mood check is not reliable enough on its own (in testing it was right only about %d%% of the time), so "
                 "please also use a short mood questionnaire such as GDS-15 or PHQ-9. If it shows depression, low mood may be "
                 "part of the memory problems, and depression is treatable." % round(bacc))
    found = ("Brain-wave slowing typical of Alzheimer's was found",
             "Most of this recording shows the slower rhythm and weaker front-to-back teamwork seen in Alzheimer's disease. "
             "This is not a diagnosis: please see a neurologist (brain specialist) for a full assessment.", "#7A2E6B")
    clear = ("No Alzheimer's-type slowing was found",
             "Most of this recording looks like a healthy resting brain. Some people with Alzheimer's still show this, so if "
             "memory problems continue, a doctor should still check them.", "#2F7D5B")
    base = found if a else clear
    if "error" in dep:
        return base[0], base[1] + " The mood check could not run on this recording (it needs the F3 and F4 sensors).", base[2]
    if dep_auc() < 0.65:
        # a near-chance indicator must not drive the dementia-vs-depression call
        return base[0], base[1] + " " + mood_note, base[2]
    d = dep["pct"] > 50
    if a and d:
        return ("Slowing found, and signs of low mood",
                "The recording shows Alzheimer's-type slowing and the brain-wave pattern linked to depression. Depression should be "
                "checked and treated before all memory problems are blamed on dementia.", "#B86E0C")
    if d:
        return ("No slowing, but signs of low mood",
                "There is no Alzheimer's-type slowing, but the pattern linked to depression is present. Memory complaints may be "
                "caused by low mood (sometimes called pseudodementia), which is treatable. A mental-health check is recommended.", "#B86E0C")
    return base


def dep_auc():
    return METRICS["depression"]["optimised"]["summary"]["auc"][0] if METRICS else 0.0


def analyze(path):
    raw = neuro.load_raw(path)
    ad = run_module(raw, AD_MODEL, neuro.AD_CH, neuro.ad_features)
    dep = run_module(raw, DEP_MODEL, neuro.DEP_CH, neuro.dep_features)
    return {"file": os.path.basename(path), "duration_s": float(raw.times[-1]), "sfreq": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names), "plot": plot_signal(raw), "ad": ad, "dep": dep,
            "summary": interpret(ad, dep)}


def current_job():
    job = session.get("job")
    path = os.path.join(UPLOADS, job, "result.json") if job else None
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ pages
@app.route("/")
def home():
    return render_template("home.html", m=METRICS)


@app.route("/about")
def about():
    return render_template("about.html", m=METRICS)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html", m=METRICS)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def eeg_interface():
    if request.method == "POST":
        files = [f for f in request.files.getlist("file") if f and f.filename]
        names = [secure_filename(f.filename) for f in files]
        sets = [n for n in names if n.lower().endswith(".set")]
        if len(sets) != 1 or any(not n.lower().endswith((".set", ".fdt")) for n in names):
            flash("Select exactly one EEGLAB .set file (plus its .fdt data file if it has one).")
            return redirect(url_for("eeg_interface"))
        old = session.get("job")
        if old:
            shutil.rmtree(os.path.join(UPLOADS, old), ignore_errors=True)
        job = uuid.uuid4().hex
        folder = os.path.join(UPLOADS, job)
        os.makedirs(folder)
        for f, n in zip(files, names):
            f.save(os.path.join(folder, n))
        try:
            result = analyze(os.path.join(folder, sets[0]))
        except Exception as e:  # corrupt / unsupported file: tell the user instead of a 500 page
            shutil.rmtree(folder, ignore_errors=True)
            flash(f"Could not read this EEG file: {e}")
            return redirect(url_for("eeg_interface"))
        with open(os.path.join(folder, "result.json"), "w") as f:
            json.dump(result, f)
        session["job"] = job
        return redirect(url_for("eeg_interface"))
    return render_template("eeg.html", r=current_job())


@app.route("/report")
@login_required
def report():
    r = current_job()
    if not r:
        flash("Upload an EEG recording first.")
        return redirect(url_for("eeg_interface"))
    return render_template("report.html", r=r, m=METRICS)


@app.errorhandler(413)
def too_big(_):
    flash("File too large (limit 300 MB).")
    return redirect(url_for("eeg_interface"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
