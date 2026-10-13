# 🎓 Student Performance Prediction System

A machine-learning internship project that predicts a student's **final mathematics
grade (G3, scale 0–20)** from demographic, family-background, and study-habit
information — using the official [UCI Student Performance dataset][uci].

> **Disclaimer:** Predictions are statistical estimates based on historical patterns.
> They are **not** guarantees of a student's actual achievement or potential.
> Do not use this tool to limit any student's opportunities.

---

## 📁 Project Structure

```
Student-Performance-Predictor/
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   └── student-mat.csv     # UCI dataset (auto-downloaded if absent)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading, ColumnTransformer preprocessing
│   └── model.py            # CV comparison, training, evaluation & prediction
└── tests/
    └── test_pipeline.py    # 61-check test suite
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9 or higher**
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

> The first run **automatically downloads** the UCI dataset (~57 KB) and runs
> cross-validation to select the best model.  Subsequent runs use Streamlit's
> `cache_resource` and are near-instant.
>
> If the download fails the app displays a clear error with manual setup steps.

### Run the Tests

```bash
python tests/test_pipeline.py
```

Expected output: **61/61 checks passed**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI ML Repository #320][uci] |
| File | `student-mat.csv` (Math course) |
| Students | 395 |
| Raw columns | 33 |
| Features used by model | 30 (G1 and G2 excluded — see below) |
| Target | G3 — final period grade (0–20) |
| Missing values | 0 |

### Why G1 and G2 are excluded

G1 (first period) and G2 (second period) grades would allow the model to
trivially predict G3 because they are highly correlated with the final grade.
Excluding them forces the model to learn from genuine study-habit and background
signals, making predictions meaningful **before any grades are recorded**.

---

## 🔬 Methodology

| Step | Detail |
|------|--------|
| Dataset | UCI Student Performance (Math) — 395 students |
| Preprocessing | scikit-learn `ColumnTransformer`: `OrdinalEncoder` for 17 categorical columns + `StandardScaler` for 13 numerical columns |
| Leakage prevention | Preprocessor is fitted **on the training set only**; the test set is transformed with the same fitted object |
| Unknown categories | `OrdinalEncoder` maps any unseen category to `−1` (no silent fallback to 0) |
| Train / Test split | 80 % train (316 students) / 20 % test (79 students), `random_state=42` |
| Model selection | 5-fold cross-validation on the **training set** across three candidate algorithms |
| Candidate algorithms | `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor` |
| CV scoring metric | Mean Absolute Error (lower is better) |
| Selected algorithm | **RandomForestRegressor** (lowest mean CV MAE) |
| Final evaluation | The selected model is evaluated **exactly once** on the held-out test set |

### Why OrdinalEncoder rather than OneHotEncoder

All three candidate models are decision-tree ensembles.  For tree-based models,
`OrdinalEncoder` is the more appropriate choice for these categorical features:

- **Representational equivalence:** a tree can recover any grouping of ordinal
  codes with multiple binary splits, so `OneHotEncoder` adds no representational
  advantage.
- **Dimensionality:** `OneHotEncoder` would expand the 17 categorical columns
  into ~45 binary columns, raising total features from 30 to ~58 on a 395-row
  dataset — extra sparsity unlikely to improve and potentially harmful.
- **Unknown-category handling:** `OneHotEncoder(handle_unknown="ignore")` maps
  an unseen value to an all-zero vector, which looks identical to the reference
  (first) category.  `OrdinalEncoder(unknown_value=-1)` produces a distinct
  sentinel that the tree routes down its own branch — the behaviour is explicit
  and auditable.
- **Feature importance readability:** 30 features (one per original column)
  keeps the importance chart directly interpretable; 58 OHE columns would
  fragment importance across dummy variables.

---

## 📈 Model Comparison — Cross-Validation Results

Evaluated with 5-fold CV on the training set (`random_state=42`).
The test set was kept completely separate during this step.

| Algorithm | CV MAE (mean ± std) | CV RMSE (mean ± std) | Selected |
|-----------|---------------------|----------------------|----------|
| **RandomForestRegressor** | **2.9135 ± 0.2197** | **3.8533 ± 0.2780** | ✅ |
| GradientBoostingRegressor | 3.0786 ± 0.2475 | 4.0135 ± 0.3282 | |
| ExtraTreesRegressor | 3.1270 ± 0.2033 | 4.1517 ± 0.3565 | |

---

## 📊 Final Test-Set Metrics (held-out, 79 students)

These are the **actual measured values** on unseen data — not targets or approximations.

| Metric | Baseline (original code) | Final (new pipeline) | Change |
|--------|--------------------------|----------------------|--------|
| MAE    | 2.9489 grade points | 2.9698 grade points | +0.021 |
| RMSE   | 3.6945 grade points | 3.7146 grade points | +0.020 |
| R²     | 0.3344 | 0.3271 | −0.007 |

The difference between baseline and final is within normal variance for this
dataset size (~79 test samples).  The new pipeline's primary improvements are
correctness (no preprocessing leakage, no silent fallback encoding) rather
than raw accuracy, which is appropriate given the deliberate exclusion of G1/G2.

> **Why R² is modest:** Predicting final grades from study habits and demographics
> alone — without any prior-period grades — is an inherently noisy task.  R² ≈ 0.33
> means the model captures about a third of the variance.  This is expected and
> is reported honestly here.

---

## 🖥️ Dashboard Features

| Tab | Content |
|-----|---------|
| **🔮 Predict** | Enter student profile → get grade estimate with ±MAE error range |
| **📊 Model Metrics** | MAE, RMSE, R², CV comparison table, actual-vs-predicted chart, residuals, feature importance |
| **📂 Dataset** | Grade distribution, raw data preview, summary statistics |
| **ℹ️ About** | Full methodology, actual measured metrics, data source, disclaimer |

---

## 🛠️ Manual Dataset Setup (offline)

If the automatic download fails:

1. Visit [https://archive.ics.uci.edu/dataset/320/student+performance][uci]
2. Download `student+performance.zip`
3. Extract → open `student.zip` inside → extract `student-mat.csv`
4. Place `student-mat.csv` in the `data/` folder

---

## 🔧 Development Notes

### Running on Windows (console encoding)

The test file uses ASCII-only output to avoid Windows CP1252 encoding issues.
If you encounter encoding errors with other scripts, prefix the command with:

```powershell
$env:PYTHONIOENCODING="utf-8"; python ...
```

### Reproducibility

All random processes use `random_state=42`.  Given the same dataset file,
the split, preprocessor fit, model training, and evaluation produce identical
results on every run (verified by the reproducibility test in section 5 of
`tests/test_pipeline.py`).

---

## 📚 Citation

> P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student
> Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Future Business
> Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, April 2008.
>
> UCI ML Repository: https://archive.ics.uci.edu/dataset/320/student+performance

---

## ⚠️ Limitations

1. **Dataset scope:** 395 students from two Portuguese schools (2005–2006).
   Results may not generalise to other countries, school systems, or time periods.

2. **Excluded features:** G1 and G2 are intentionally excluded.  If prior grades
   were included, accuracy would be substantially higher — but predictions would
   only be useful after term grades exist.

3. **Modest R²:** The model explains roughly 33% of grade variance.  Study habits
   and demographics capture real signals but many other factors (teacher quality,
   exam difficulty, effort changes) are not in the data.

4. **No causal claims:** Feature importances reflect correlation patterns, not
   causal mechanisms.  High importance for a feature does not mean changing that
   feature will change a student's grade.

5. **Not for high-stakes decisions:** Never use this tool to label, rank, or
   limit opportunities for individual students.

---

## ⚙️ Technology Stack

- **Python 3.9+** · **pandas** · **NumPy**
- **scikit-learn** — `ColumnTransformer`, `OrdinalEncoder`, `StandardScaler`,
  `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`,
  `cross_validate`
- **Streamlit** — interactive dashboard
- **Matplotlib** — charts

[uci]: https://archive.ics.uci.edu/dataset/320/student+performance
