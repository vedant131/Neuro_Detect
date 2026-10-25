# NeuroDetect

Alzheimer's + depression screening from one resting-state, eyes-closed EEG, using 5 of the 19 standard electrodes.
The Flask portal uploads an EEGLAB `.set` file and shows a combined, explainable report.

## Quick start

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000  (or double-click run.bat)
```

Register an account, log in, and upload a file from `Test Sample/`. The trained models are in `models/`.

## Reproduce training and validation

```bash
python prepare_data.py   # downloads OpenNeuro ds004504 + ds003478 (~10 GB streamed, ~1 GB kept in data/)
python train.py          # nested, subject-level CV -> models/*.joblib, models/metrics.json, static/img/results/*.png
python experiments.py    # Objective 2 (time/space/complexity vs other algorithms) + Objective 3 (6 distance measures)
python test_neuro.py     # signal-processing sanity checks
```

## Files

| File | Purpose |
|---|---|
| `neuro.py` | Shared signal processing: filtering, epoching, artifact rejection, features, voting, explanation |
| `prepare_data.py` | Downloads the public datasets and reduces them to 10 s epochs |
| `train.py` | Band search, nested CV, baselines, artifact-robustness test, figures, metrics |
| `experiments.py` | Cost comparison with SVM, k-NN, Random Forest, logistic regression, MLP and a compact CNN; distance-measure comparison |
| `app.py` | Web portal: auth (SQLite + hashed passwords), upload, analysis, report |
| `templates/` | `base.html` layout, workspace (`eeg.html`), `report.html`, `analysis.html` (live metrics) |

## Method

1. Pick Fz, O1, O2, F3, F4. Band-pass 0.5–45 Hz, resample to 250 Hz, cut into 10 s epochs.
2. Reject an epoch if any channel's peak-to-peak is more than 3× its median, over 500 µV, or flat.
3. **Alzheimer's module (Fz, O1, O2):**
   - Features: log relative power at O1/O2, plus Fz–O1/O2 magnitude-squared coherence.
   - The power feature and coherence band are chosen by nested CV from 35 combinations.
4. **Depression module (F3, F4, O1, O2):**
   - Features: right−left log-power asymmetry (frontal or posterior), plus frontal relative power.
   - Chosen by nested CV from 70 combinations.
5. Each clean epoch is classified by `StandardScaler` + a nearest-centroid classifier with **cosine distance** (selected from Euclidean, Manhattan, Cosine, Minkowski, Hamming and Jaccard in `experiments.py`). The result is the % of epochs voting for the condition. The single-feature depression model uses Euclidean distance.
6. The two module results are combined into one differential statement:
   - neurodegeneration
   - possible comorbid depression
   - possible depressive pseudodementia
   - no abnormal pattern

All features are relative, so they don't depend on amplifier gain. The old app used raw power, and a 2× gain change flipped its diagnoses.

## Data

| Dataset | Link | Used for |
|---|---|---|
| OpenNeuro **ds004504**: Alzheimer's, FTD and healthy EEG (Miltiadous et al., *Data* 2023, [doi:10.3390/data8060095](https://doi.org/10.3390/data8060095)) | https://openneuro.org/datasets/ds004504 | 36 AD + 29 healthy, eyes closed, 19-ch, preprocessed derivatives |
| OpenNeuro **ds003478**: Depression rest EEG (Cavanagh et al.) | https://openneuro.org/datasets/ds003478 | 121 young adults, high BDI (>= 13) vs low BDI (<= 7), eyes-closed minutes, re-referenced to mastoids |

The results are on the **How accurate is it?** page of the portal, which reads `models/metrics.json`.

## Limitations

- Each dataset comes from a single site.
- The depression model was trained on young adults.
- The depression indicator's accuracy is close to chance on held-out subjects; the report says so and asks for a clinical scale.
- Vote percentages are not calibrated probabilities.
- Research prototype, not a medical device.
