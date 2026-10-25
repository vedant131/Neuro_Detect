# NeuroDetect

**A lightweight, explainable EEG screening framework for Alzheimer's disease, with a depression marker.**

Capstone Project 2026–27 · Dept. of Electrical & Electronics Engineering, MIT World Peace University, Pune
Vedant Shinde · Rhythm Tyagi · Guide: Dr. Anagha Deshpande · Group 2627S704

> NeuroDetect is a research prototype and a screening aid. It is **not** a diagnostic device.

---

## What it does

NeuroDetect reads one short, resting, eyes-closed EEG recording and looks for the brain-wave slowing that Alzheimer's disease causes. It uses only **5 of the usual 19 electrodes**:

| Electrodes | Used for |
|---|---|
| **Fz, O1, O2** | Alzheimer's module: θ/α power at the back of the head + θ coherence front-to-back |
| **F3, F4** | Mood marker: frontal alpha asymmetry, `ln α(F4) − ln α(F3)` (preliminary, indicative only) |

The web app takes an EEGLAB `.set` file and returns a plain-language verdict. It also shows the vote of every 10-second slice, the measurements behind the verdict, and a PDF report.

## Pipeline

1. Band-pass 0.5–45 Hz and resample to 250 Hz.
2. Cut the recording into 10-second slices and reject slices spoiled by blinks or movement.
3. Turn each clean slice into a few biomarkers (relative powers and coherences, which do not depend on amplifier gain).
4. Compare each slice with a typical healthy and a typical Alzheimer's pattern: a nearest-centroid classifier with **cosine distance**.
5. Each slice votes, and the result is the share of slices that look like the condition.

## Results (Alzheimer's module)

Evaluated on OpenNeuro ds004504: 65 people (36 Alzheimer's, 29 healthy), with 10× repeated, subject-level 5-fold cross-validation.

| Metric | Value |
|---|---|
| Balanced accuracy | **82.6% ± 2.7** |
| Sensitivity | 71.1% |
| Specificity | 94.1% |
| AUC | 0.87 |
| 3 electrodes vs full 19-channel cap | 82.6% vs 73.8% |
| Model size | 16 learned parameters, ≈ 1 KB, runs on a laptop CPU (no GPU) |

The depression marker was tested on a separate dataset (OpenNeuro ds003478, 121 people, BDI scores). It reached **54.9%** balanced accuracy, close to chance, so the app reports it as *indicative only*. The two modules use separate datasets; there is no combined 3-class model.

All numbers come from `Full Code/models/metrics.json`.

## Repository layout

| Path | Contents |
|---|---|
| `Full Code/` | Source code, trained models, web app and sample recordings (see [`Full Code/README.md`](Full%20Code/README.md)) |
| `Full Code/Test Sample/` | 8 sample recordings from ds004504 (6 Alzheimer's, 2 healthy) to try the app |
| `Documents/` | End-term report, research paper, presentation, literature survey, video script and posters |
| `Documents/poster_light/` | Editable source (HTML) of the light-theme poster |
| `videos/neurodetect-film/` | 3-minute project film (HyperFrames project: scenes, narration, music, captions) |

## Quick start

```bash
cd "Full Code"
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:5000  (or double-click run.bat)
```

Register an account, log in, and upload a file from `Test Sample/`.

Reproduce training and validation (downloads the public datasets, about 1 GB kept on disk):

```bash
python prepare_data.py
python train.py
python experiments.py
python test_neuro.py
```

## Render the project film

```bash
cd videos/neurodetect-film
npm install
npx hyperframes preview      # edit and preview in the browser
npx hyperframes render       # writes renders/<name>.mp4
```

## Data

- **ds004504**: Miltiadous et al., *Data* 2023, [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095), <https://openneuro.org/datasets/ds004504>
- **ds003478**: Cavanagh et al., depression rest EEG, <https://openneuro.org/datasets/ds003478>

The sample recordings were part of the training data. For held-out performance, quote the cross-validated numbers above.
