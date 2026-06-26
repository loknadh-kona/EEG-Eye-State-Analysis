# Analysis Report — EEG Eye-State Validation

**A healthcare-analytics case study in keeping a performance metric honest.**

Author: Loknadh Kona · Independent data-analysis portfolio project

---

## Executive summary

A Random Forest trained to classify *eyes open vs. eyes closed* from EEG initially reported **~81% accuracy**. After auditing how that number was produced, the **leakage-free** accuracy was **~51%** — statistically indistinguishable from always guessing the majority class. Three further checks (an expanded feature set, a strict chronological hold-out, and a label-permutation test) all confirmed the same conclusion.

The point of the project is not the classifier. It is a worked demonstration of the discipline that matters most in clinical and health-data analytics: **a metric is only as trustworthy as the validation behind it.** The same failure mode that inflated this score — information leaking from the test set into training — is exactly what makes a readmission-risk model, a quality dashboard, or a screening tool look better than it really is.

> *Note on scope: this uses a public EEG dataset as a methods demonstration, not clinical or patient data. The transferable value is the validation methodology.*

---

## Why this is relevant to a healthcare / clinical analyst role

| Skill demonstrated here | Where it shows up in healthcare analytics |
|---|---|
| **Leakage-aware validation** (grouping correlated records) | Preventing future or duplicate patient data from leaking into a model's training set |
| **Baseline-anchored benchmarking** (vs. a majority-class dummy) | Asking *"does this beat current practice / standard of care?"* before celebrating a metric |
| **Effect sizes & significance testing** (Cohen's d, permutation test) | Distinguishing a real, clinically meaningful difference from noise |
| **Handling non-stationarity** (chronological hold-out) | Knowing a model validated on the past can still fail on a new time period or population |
| **Honest communication of a null result** | Reporting what the data actually supports, to technical and non-technical stakeholders alike |

---

## Methods

**Data.** UCI *EEG Eye State* dataset (Emotiv EPOC, 14 channels, 128 Hz, single subject, ~117 s, 14,980 samples; target `eyeDetection`: 0 = open, 1 = closed).

<p align="center">
  <img src="figures/eeg_electrode_placements.png" width="70%" alt="Emotiv 14-channel EEG electrode placements"><br>
  <sub><i>The 14-channel Emotiv layout (10–20 system), coloured by cortical lobe. Occipital sites O1/O2 sit over the visual cortex.</i></sub>
</p>

**Quality control & cleaning.** No missing values; mild class imbalance (55.1% open / 44.9% closed). A sensor-range check flagged physiologically impossible spikes (one channel jumping while its neighbour stayed calm); rows with any channel outside 2,000–10,000 µV were treated as electrode-contact glitches and removed — **4 rows (0.03%)**, leaving 14,976 clean samples.

**Preprocessing.** Each channel band-pass filtered to **1–40 Hz** (5th-order Butterworth, zero-phase via `filtfilt`). The 1 Hz high-pass removes slow drift and the DC jump at eye-state transitions; the 40 Hz low-pass suppresses line noise and high-frequency EMG.

**Feature extraction.** Welch's method over sliding **2-second windows** (256 samples, step 32).
- *Baseline matrix:* mean Alpha (8–12 Hz) and Beta (12–30 Hz) power per channel → **28 features**.
- *Expanded matrix:* + Delta/Theta/Gamma power, Alpha/Beta ratio, per-window variance, Hjorth mobility & complexity → **126 features**.

<p align="center">
  <img src="figures/brainwave_bands.jpg" width="74%" alt="EEG frequency bands"><br>
  <sub><i>The EEG frequency bands the features are built from — chiefly Alpha (8–12 Hz) and Beta (12–30 Hz).</i></sub>
</p>

**Validation.**
1. **Leakage-safe Stratified Group K-Fold** — windows grouped into 1,024-sample time-blocks so overlapping near-duplicates never straddle the train/test split.
2. **Chronological 70/30 hold-out** — train on the past, test on the future, against a majority-class `DummyClassifier`.
3. **Expanded-feature re-run** under the same leakage-safe CV.
4. **Cohen's d** effect sizes, a **label-permutation test** (N = 200), and a confusion matrix.

---

## Results

| Validation method | Accuracy | Baseline | Reading |
|---|---|---|---|
| Naive overlapping-window CV | **80.9%** | — | Inflated by leakage |
| Leakage-safe Group CV (28 feat.) | **51.5% ± 2.6%** | ~55% | At chance |
| Leakage-safe Group CV (126 feat.) | **49.7% ± 4.5%** | ~55% | No gain — signal ceiling, not feature poverty |
| Chronological 70/30 hold-out | **36.7%** | 22.3% | +14.4 pts, but exposes non-stationarity |
| Label-permutation test (N=200) | **p = 0.50** | — | Indistinguishable from chance |

Supporting detail:
- **Cleaning:** 14,980 → 14,976 samples; 0 missing.
- **Filtering:** processed signal re-centres on ~0 µV; the artificial baseline jump at eye closure disappears.
- **Condition comparison:** Alpha/Beta band-power distributions overlap heavily — no clean separation.
- **Berger effect:** eyes-closed alpha only mildly boosted (closed/open 1.0–1.2×) and *not* dominated by occipital O1/O2 (1.00–1.05×), consistent with the consumer headset's limited posterior coverage.
- **Effect sizes:** largest alpha-power Cohen's d = 0.422; most channels negligible (|d| < 0.2); occipital near 0.
- **Confusion matrix:** the leakage-safe model defaults to predicting "open".

![Validation comparison](figures/validation_comparison.png)

---

## Interpretation

As the validation gets stricter, the apparent skill disappears (naive ~80.9% → leakage-safe ~51.5% → chronological 36.7% over a 22.3% baseline). Richer features don't rescue it, effect sizes are small, and a permutation test can't separate the model from chance (p ≈ 0.50). On this single-subject, consumer-grade recording, Alpha/Beta band-power features carry only weak, non-stationary information about eye state — and the high accuracies this dataset is famous for are largely artifacts of loose validation.

**The takeaway for analytics work:** the deliverable is a reliable *process* for keeping a performance number honest — the same discipline that separates a deployable clinical model from one that silently fails on new patients.

---

## Limitations

- **Single subject** — may not generalise to other people or sessions.
- **Feature scope** — spectral/complexity features only; no connectivity or learned temporal dynamics.
- **Window overlap** — heavy overlap pads the row count with correlated samples; grouping handles the leakage but leaves only ~15 genuinely independent time-blocks.
- **Non-stationarity** — eye state arrives in long blocks, so chronological splits get very different train/test label mixes.
- **Artifacts** — ocular/EMG contamination only partly suppressed by a 1–40 Hz filter (no ICA).
- **Crude outlier rule** — a fixed amplitude threshold can miss subtle artifacts or discard valid extremes.

---

## Future work

- Report subject-/block-wise CV **alongside** a chronological hold-out, a dummy baseline, and a permutation test — as standard practice.
- Apply **ICA-based artifact removal** to separate true neural alpha from ocular/EMG components.
- Add **learned temporal models** (temporal CNNs / LSTMs) and connectivity features.
- **Validate on multi-subject data** to test cross-person generalisation.

---

## Reproducibility

See the [`notebook/`](../notebook/eeg_eye_state_analysis.ipynb) for the full, runnable analysis. Dependencies are listed in [`requirements.txt`](../requirements.txt). The dataset lives in [`data/`](../data/) (or download it from the UCI link above).

**Citation:** Roesler, O. (2013). *EEG Eye State* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/264/eeg+eye+state
