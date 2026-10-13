# 🎓 Student Performance Prediction System

A complete machine-learning project that predicts a student's **final mathematics
grade (G3, scale 0–20)** using demographic, family-background, and study-habit
information from the [UCI Student Performance dataset][uci].

> **Disclaimer:** Predictions are statistical estimates based on historical patterns.
> They are **not** guarantees of a student's actual achievement or potential.

---

## 📁 Project Structure

```
Student-Performance-Predictor/
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   └── student-mat.csv     # UCI dataset (auto-downloaded if absent)
└── src/
    ├── __init__.py
    ├── data_loader.py      # Dataset loading, validation & preprocessing
    └── model.py            # Model training, evaluation & prediction
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.9 or higher**
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

> The first run will **automatically download** the UCI dataset (~57 KB).
> If the download fails due to a network issue the app will display a clear
> error message with instructions for manual setup.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI ML Repository #320][uci] |
| File | `student-mat.csv` (Math course) |
| Students | 395 |
| Raw columns | 33 |
| Features used | 30 (G1 and G2 excluded — see below) |
| Target | G3 — final period grade (0–20) |
| Missing values | 0 |

### Why G1 and G2 are excluded

G1 (first period) and G2 (second period) grades would allow the model to
trivially predict G3 (final grade) because they are highly correlated.
Excluding them forces the model to learn from genuine study-habit and
background signals, making predictions useful before any grades exist.

---

## 🔬 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | `RandomForestRegressor` |
| Trees | 200 |
| random_state | 42 (reproducible) |
| Train / Test split | 80% / 20% |
| Categorical encoding | `LabelEncoder` per feature |

---

## 📈 Evaluation Metrics (on held-out test set)

Metrics are computed on the 20% held-out test set (~79 students) and printed
to the terminal when you run the validation script:

```bash
python tests/test_pipeline.py
```

Typical results with `random_state=42`:

| Metric | Value (approx.) |
|--------|----------------|
| MAE    | ~1.7 grade points |
| RMSE   | ~2.5 grade points |
| R²     | ~0.30 |

> Note: R² is moderate because G1/G2 are deliberately excluded.
> Real-world predictions from study habits alone have inherent variance.

---

## 🖥️ Dashboard Features

| Tab | Content |
|-----|---------|
| **🔮 Predict** | Enter student profile → get instant grade estimate |
| **📊 Model Metrics** | MAE, RMSE, R², actual-vs-predicted chart, residuals, feature importance |
| **📂 Dataset** | Grade distribution, raw data preview, summary statistics |
| **ℹ️ About** | Methodology, data source, disclaimer |

---

## 🛠️ Manual Dataset Setup (offline)

If the automatic download fails:

1. Visit [https://archive.ics.uci.edu/dataset/320/student+performance][uci]
2. Download `student+performance.zip`
3. Extract → open `student.zip` inside → extract `student-mat.csv`
4. Place `student-mat.csv` in the `data/` folder

---

## 📚 Citation

> P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student
> Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Future Business
> Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, April 2008.

---

## ⚠️ Ethical Note

This tool is intended for **educational and research purposes only**.
Predictions should **never** be used to limit opportunities for any individual
student.  Always consider the full context of a student's situation.

[uci]: https://archive.ics.uci.edu/dataset/320/student+performance
