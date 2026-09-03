# Lab Assignment 3 — AI in Healthcare (CSET343)

Data acquisition, cleaning, preprocessing and analysis across four medical data
modalities: **tabular, textual, image, and signal**.

---

## What the assignment asks for

The brief sets one agenda applied four times over. For each modality you must:

1. **Acquire and inspect** a real-world medical dataset.
2. **Clean it using modality-specific techniques**, with three healthcare-specific
   concerns called out explicitly:
   - implausible clinical values (tabular),
   - PHI removal (text),
   - imaging metadata anonymisation (image).
3. **Preprocess into a model-ready format** — scaled tensors, token sequences,
   image tensors, signal windows.
4. **Run EDA** appropriate to that modality.
5. **Apply feature engineering, feature selection & extraction, noise removal,
   and augmentation** where relevant.
6. **For the tabular part specifically**: hypothesis test, chi-square test, ANOVA.
7. **Explain why healthcare data needs extra rigour** than generic datasets —
   missingness disguised as valid values, class imbalance, privacy constraints,
   signal noise.

Point 7 is the actual thesis of the assignment. Points 1–6 are the evidence.
Each script closes with a summary table tying its findings back to it.

| Part | Modality | Dataset |
|---|---|---|
| A | Tabular | Pima Indians Diabetes |
| B | Textual | MTSamples Medical Transcriptions |
| C | Image | Chest X-Ray (Pneumonia) |
| D | Signal | MIT-BIH Arrhythmia Database (ECG) |

---

## Why healthcare data needs extra rigour — the four demonstrations

Each part exists to prove one clause of that final objective.

**A — missingness encoded as valid-looking values.** In the Pima dataset a
missing glucose reading is stored as `0`. `df.isna()` returns nothing. But a
living patient cannot have a blood glucose of zero, so those are disguised nulls
— roughly 49% of Insulin and 30% of SkinThickness. Train without catching this
and the model learns from impossible physiology. The script recodes them, then
retains *the fact that a value was missing* as its own feature, because in
clinical data an untaken test is itself signal: a clinician decided not to order
it.

**B — privacy constraints.** Clinical notes contain names, dates, MRNs and
addresses. HIPAA Safe Harbor requires 18 identifier categories to be removed
before the text can be used for research. This is a legal precondition, not a
modelling preference. The script implements the pattern set, audits what it
found, and verifies zero residual. A second clinical wrinkle: standard NLP
stopword lists delete "no" and "denies", which inverts the meaning of
"no evidence of malignancy". Negation terms are deliberately preserved.

**C — identity hides in metadata, not just pixels.** A DICOM header carries
PatientName, birth date and institution in plain text; some scanners burn the
patient name directly into the pixel border. Stripping neither leaks PHI as
surely as printing the name. The script handles EXIF removal, DICOM tag clearing
with UID regeneration, and border masking. It also refuses horizontal flip
augmentation — mirroring a chest X-ray relocates the heart and manufactures
dextrocardia, a pathology that does not occur in the source population.

**D — signal noise and patient-level leakage.** Raw ECG carries baseline wander
from breathing, 50/60 Hz powerline interference, and muscle noise, each needing a
different filter. All filtering is zero-phase (`filtfilt`), because a phase shift
would displace the R peak and corrupt every downstream timing measurement. The
train/test split is done **by record, not by beat** — beats from one patient are
highly correlated, so a random beat-level split leaks patient morphology into the
test set and inflates accuracy dramatically.

**Class imbalance** appears in all four and is handled with class weights rather
than naive oversampling wherever possible, so no synthetic patients are invented.

---

## Setup

```bash
pip install -r requirements.txt
```

Then download the datasets into `data/`:

| Part | Expected path |
|---|---|
| A | `data/diabetes.csv` |
| B | `data/mtsamples.csv` |
| C | `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg` |
| D | `data/mitdb/` — run `download_mitdb()` inside Part D, or `wfdb.dl_database('mitdb', 'data/mitdb')` |

Parts A, B and C use Kaggle downloads. Part D pulls from PhysioNet
automatically via the `wfdb` package.

## Running

```bash
python partA_tabular_pima.py
python partB_text_mtsamples.py
python partC_image_chest_xray.py
python partD_signal_mitbih_ecg.py
```

Each writes plots and a `*_model_ready.npz` file into `outputs/partX/`.

The scripts use `# %%` cell markers, so they open directly as notebooks in
VS Code or Jupyter (`jupytext --to notebook partA_tabular_pima.py`) if your
submission needs `.ipynb`.

## Runtime knobs

Parts C and D process a subset by default so a first run finishes quickly:

- **Part C** — `LIMIT = 600` controls how many images are loaded into tensors.
  Set `LIMIT = None` for the full 5,863 images.
- **Part D** — `ACTIVE_RECORDS = RECORDS[:12]` uses 12 of 48 records. Change to
  `RECORDS` for the whole database (expect a much longer run).

---

## What each part produces

**Part A** — missingness heatmap, distribution and correlation plots, PCA scree,
feature importance chart; Welch t-test with Shapiro/Levene assumption checks and
Cohen's d, chi-square with Cramér's V and expected-count warnings, one-way ANOVA
with Kruskal-Wallis backup and Tukey HSD post-hoc; Bonferroni correction across
the multiple tests; scaled train/test tensors plus a PCA variant.

**Part B** — PHI audit counts before and after de-identification, length and
specialty distributions, vocabulary statistics, LDA topics; TF-IDF with bigrams
→ chi-square selection → LSA; padded integer sequences (N × 300).

**Part C** — EXIF audit, resolution and intensity EDA, four-stage preprocessing
figure, CLAHE histogram comparison, augmentation grid; (N, 224, 224, 1) float32
tensors and class weights.

**Part D** — four-stage denoising figure, power spectrum before/after, per-record
summary table, mean beat morphology per AAMI class; 252-sample beat windows plus
19 handcrafted time/frequency/morphology features.

---

## Notes on choices worth defending in a viva

- **Split before impute.** Fitting the imputer on the full dataset leaks test
  statistics into training. Every transformer here is fitted on train only.
- **Welch over Student's t.** Levene's test shows unequal variance between the
  diabetic and non-diabetic groups, so the equal-variance assumption fails.
- **Bonferroni.** Five simultaneous t-tests inflate the family-wise false
  positive rate; α is adjusted to 0.01.
- **Median over mean imputation.** Insulin and SkinThickness are heavily right-
  skewed, so the mean is dragged by outliers.
- **CLAHE over global histogram equalisation.** Lung consolidation sits in a
  narrow intensity band; global equalisation over-amplifies noise.
- **Median-filter baseline removal over a high-pass filter.** An aggressive
  high-pass distorts the ST segment, which is the clinical marker of myocardial
  infarction.
- **Per-beat z-score normalisation.** Removes amplitude differences caused by
  electrode placement and body habitus, so morphology is compared rather than
  gain.
