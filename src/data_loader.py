"""
data_loader.py
--------------
Handles downloading, caching, and preprocessing the UCI Student Performance
dataset (student-mat.csv).  G1 and G2 are intentionally excluded from
features so the model cannot "cheat" by using earlier-term grades.
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, build feature matrix X and target vector y.

    Returns
    -------
    X : pd.DataFrame   feature matrix (no G1/G2/G3)
    y : pd.Series      target (G3)
    feature_names : list[str]
    encoders : dict[str, LabelEncoder]
    """
    df = df.copy()

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET_COL])
    if df.empty:
        raise ValueError("No valid rows remain after dropping rows with missing G3.")

    # Encode categorical columns
    encoders: dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    return X, y, feature_cols, encoders


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """
    Full pipeline: load → preprocess → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, encoders, raw_df
    """
    raw_df = load_raw_dataframe()
    X, y, feature_names, encoders = preprocess(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names, encoders, raw_df
