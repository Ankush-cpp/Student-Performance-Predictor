"""
data_loader.py
--------------
Handles downloading, caching, and preprocessing the UCI Student Performance
dataset (student-mat.csv).  G1 and G2 are intentionally excluded from
features so the model cannot "cheat" by using earlier-term grades.

The public API returns a fitted scikit-learn ColumnTransformer (the
"preprocessor") alongside the split data, so that:
  • Training and inference use an identical, reproducible transformation.
  • Categorical inputs at inference time are encoded with the same fitted
    OrdinalEncoder — unknown categories map to -1 instead of silently
    collapsing to an existing category.

Encoding choice — OrdinalEncoder vs OneHotEncoder
--------------------------------------------------
All three candidate models (RandomForest, ExtraTrees, GradientBoosting)
are decision-tree ensembles.  For tree-based models:

1. Representational equivalence: a tree can recover any grouping of
   ordinal-coded categories with multiple binary splits, so OHE adds
   no representational advantage over OrdinalEncoder.

2. Dimensionality: OHE would expand the 17 categorical columns into
   ~45 binary columns (each unique category becomes its own column),
   raising the total feature count from 30 to ~58 on a dataset of only
   395 rows.  The extra sparsity is more likely to introduce noise than
   signal for this problem size.

3. Unknown-category handling: OHE(handle_unknown="ignore") maps an
   unseen value to an all-zero row — identical to the reference (first)
   category, which silently misrepresents the input.
   OrdinalEncoder(unknown_value=-1) produces a distinct sentinel that
   the tree can route down a dedicated branch, making the behaviour
   observable and auditable.

4. Feature importance readability: keeping 30 features (one per raw
   column) makes the importance chart directly interpretable; 58 OHE
   columns would fragment importance across dummy variables.

Conclusion: OrdinalEncoder is the appropriate choice for this dataset
and these model families.  The decision is revisited in README.md under
the Methodology section.
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "student-mat.csv"

DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

# ── column metadata ─────────────────────────────────────────────────────────
TARGET_COL = "G3"

# G1 and G2 are excluded — they are interim grades, not study-habit predictors
EXCLUDE_COLS = ["G1", "G2", TARGET_COL]

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERICAL_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

# Human-readable labels for the Streamlit UI
FEATURE_LABELS = {
    "school": "School (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
    "sex": "Sex",
    "age": "Age",
    "address": "Home address type (U = urban, R = rural)",
    "famsize": "Family size (LE3 = ≤3, GT3 = >3)",
    "Pstatus": "Parent cohabitation status (T = together, A = apart)",
    "Medu": "Mother's education (0–4)",
    "Fedu": "Father's education (0–4)",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason to choose this school",
    "guardian": "Student's guardian",
    "traveltime": "Home–school travel time (1–4)",
    "studytime": "Weekly study time (1–4)",
    "failures": "Number of past class failures (0–3)",
    "schoolsup": "Extra educational support",
    "famsup": "Family educational support",
    "paid": "Extra paid classes",
    "activities": "Extra-curricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Quality of family relationships (1–5)",
    "freetime": "Free time after school (1–5)",
    "goout": "Going out with friends (1–5)",
    "Dalc": "Workday alcohol consumption (1–5)",
    "Walc": "Weekend alcohol consumption (1–5)",
    "health": "Current health status (1–5)",
    "absences": "Number of school absences (0–93)",
}


# ── download helpers ────────────────────────────────────────────────────────

def _download_dataset() -> None:
    """Download the UCI zip and extract student-mat.csv into DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as resp:
            outer_bytes = resp.read()
    except Exception as exc:
        raise ConnectionError(
            f"Could not download the dataset from UCI:\n  {DATASET_URL}\n"
            f"Error: {exc}\n\n"
            "Please check your internet connection or manually place "
            "'student-mat.csv' inside the 'data/' folder."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer_zip:
            with zipfile.ZipFile(io.BytesIO(outer_zip.read("student.zip"))) as inner_zip:
                with inner_zip.open("student-mat.csv") as csv_file:
                    CSV_PATH.write_bytes(csv_file.read())
    except (KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(
            "The downloaded archive has an unexpected structure. "
            "Please manually place 'student-mat.csv' inside the 'data/' folder."
        ) from exc


# ── public API ───────────────────────────────────────────────────────────────

def load_raw_dataframe() -> pd.DataFrame:
    """
    Return the raw dataframe from CSV.
    Downloads the file first if it is not present.
    """
    if not CSV_PATH.exists():
        _download_dataset()

    df = pd.read_csv(CSV_PATH, sep=";")

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    required = set(CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL])
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}. "
            "Make sure 'data/student-mat.csv' is the correct UCI file."
        )

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer on *df*.

    Categorical columns  → OrdinalEncoder  (handle_unknown='use_encoded_value',
                                             unknown_value=-1)
    Numerical columns    → StandardScaler  (zero mean, unit variance)

    Why OrdinalEncoder rather than OneHotEncoder
    ---------------------------------------------
    See the module-level docstring for the full rationale.  In brief:
    all three candidate models are tree ensembles that handle ordinal
    codes natively; OHE would triple the categorical feature count on
    a 395-row dataset without adding representational power, and its
    unknown-category behaviour (all-zeros ≡ reference category) is
    less transparent than the −1 sentinel used here.
    """
    # Collect the unique sorted categories from the training data so the
    # encoder is fully deterministic.
    categories = [
        sorted(df[col].astype(str).unique().tolist())
        for col in CATEGORICAL_COLS
    ]

    cat_transformer = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_COLS),
            ("num", num_transformer, NUMERICAL_COLS),
        ],
        remainder="drop",
    )
    preprocessor.fit(df[CATEGORICAL_COLS + NUMERICAL_COLS])
    return preprocessor


def preprocess(df: pd.DataFrame):
    """
    Build feature matrix X (numpy array) and target vector y using a
    fitted ColumnTransformer.

    Returns
    -------
    X            : np.ndarray  — transformed feature matrix (no G1/G2/G3)
    y            : np.ndarray  — target (G3)
    feature_names: list[str]   — CATEGORICAL_COLS + NUMERICAL_COLS
    preprocessor : ColumnTransformer  — fitted transformer (use for inference)

    Backward-compatibility note
    ---------------------------
    Previous callers expected a 4th return value of ``encoders`` (a dict of
    LabelEncoders).  The new signature returns the ColumnTransformer instead.
    The Streamlit app and tests have been updated accordingly.
    """
    df = df.copy()

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET_COL])
    if df.empty:
        raise ValueError("No valid rows remain after dropping rows with missing G3.")

    feature_names = CATEGORICAL_COLS + NUMERICAL_COLS
    preprocessor = build_preprocessor(df)

    X = preprocessor.transform(df[feature_names])
    y = df[TARGET_COL].to_numpy()

    return X, y, feature_names, preprocessor


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: load → split (raw) → fit preprocessor on train only
    → transform both sets.

    Fitting the preprocessor **only on the training portion** prevents any
    data leakage from the test set into the scaler / encoder statistics.

    Returns
    -------
    X_train      : np.ndarray
    X_test       : np.ndarray
    y_train      : np.ndarray
    y_test       : np.ndarray
    feature_names: list[str]
    preprocessor : ColumnTransformer  — fitted on train set only
    raw_df       : pd.DataFrame       — the full un-split raw frame
    """
    raw_df = load_raw_dataframe()

    # Drop rows with missing target before splitting
    clean_df = raw_df.dropna(subset=[TARGET_COL]).copy()

    feature_names = CATEGORICAL_COLS + NUMERICAL_COLS
    X_raw = clean_df[feature_names]
    y = clean_df[TARGET_COL].to_numpy()

    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )

    # Fit the preprocessor on training data only — no leakage
    preprocessor = build_preprocessor(X_raw_train)

    X_train = preprocessor.transform(X_raw_train)
    X_test = preprocessor.transform(X_raw_test)

    return X_train, X_test, y_train, y_test, feature_names, preprocessor, raw_df
