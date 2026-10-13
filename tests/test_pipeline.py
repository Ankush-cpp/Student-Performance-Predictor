"""
tests/test_pipeline.py
----------------------
Validates the full ML pipeline: data loading, preprocessing, model training,
evaluation, and single prediction.

Run with:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_loader import (
    load_raw_dataframe,
    preprocess,
    get_train_test_split,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
)
from src.model import train_model, evaluate_model, get_feature_importance, predict_single


PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"{status}  {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append(condition)


# ── 1. Raw dataset ────────────────────────────────────────────────────────────
print("\n-- 1. Raw Dataset --------------------------------------------------")
try:
    raw_df = load_raw_dataframe()
    check("Load raw dataframe", True, f"Shape: {raw_df.shape}")
    check("Has 395 rows",  len(raw_df) == 395,  f"Got {len(raw_df)}")
    check("Has 33 columns", len(raw_df.columns) == 33, f"Got {len(raw_df.columns)}")
    check("No missing values", raw_df.isnull().sum().sum() == 0)
    check("G3 range [0, 20]",
          raw_df[TARGET_COL].between(0, 20).all(),
          f"min={raw_df[TARGET_COL].min()}, max={raw_df[TARGET_COL].max()}")
except Exception as e:
    check("Load raw dataframe", False, str(e))


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("\n-- 2. Preprocessing ------------------------------------------------")
try:
    X, y, feature_names, encoders = preprocess(raw_df)
    check("Returns DataFrame X", isinstance(X, pd.DataFrame))
    check("Returns Series y",    isinstance(y, pd.Series))
    check("G1 not in features",  "G1" not in feature_names)
    check("G2 not in features",  "G2" not in feature_names)
    check("G3 not in features",  "G3" not in feature_names)
    check("X and y same length", len(X) == len(y))
    check("All X values numeric", X.select_dtypes(include=[object]).empty,
          "Categorical columns should be label-encoded")
    check("Encoders created for categoricals",
          set(CATEGORICAL_COLS).issubset(set(encoders.keys())))
except Exception as e:
    check("Preprocessing", False, str(e))


# ── 3. Train/test split ───────────────────────────────────────────────────────
print("\n-- 3. Train / Test Split -------------------------------------------")
try:
    X_train, X_test, y_train, y_test, fnames, encs, _ = get_train_test_split(
        test_size=0.2, random_state=42
    )
    total = len(X_train) + len(X_test)
    check("Total rows preserved", total == len(raw_df), f"{len(X_train)} + {len(X_test)} = {total}")
    check("Test set ~20%",
          abs(len(X_test) / total - 0.2) < 0.02,
          f"Test fraction: {len(X_test)/total:.3f}")
    check("Feature names consistent", fnames == feature_names)
except Exception as e:
    check("Train/test split", False, str(e))


# ── 4. Model training ─────────────────────────────────────────────────────────
print("\n-- 4. Model Training -----------------------------------------------")
try:
    model = train_model(X_train, y_train, random_state=42)
    check("Model trained", model is not None)
    check("Model has feature importances",
          len(model.feature_importances_) == len(feature_names),
          f"Importances: {len(model.feature_importances_)}, features: {len(feature_names)}")
except Exception as e:
    check("Model training", False, str(e))


# ── 5. Evaluation ─────────────────────────────────────────────────────────────
print("\n-- 5. Evaluation ---------------------------------------------------")
try:
    metrics = evaluate_model(model, X_test, y_test)
    check("Returns MAE",  "mae"    in metrics)
    check("Returns RMSE", "rmse"   in metrics)
    check("Returns R²",   "r2"     in metrics)
    check("Returns y_pred", "y_pred" in metrics)

    mae  = metrics["mae"]
    rmse = metrics["rmse"]
    r2   = metrics["r2"]

    check("MAE > 0",        mae > 0,         f"MAE  = {mae:.4f}")
    check("RMSE >= MAE",    rmse >= mae,      f"RMSE = {rmse:.4f}")
    check("R² in (-inf,1]", r2 <= 1.0,       f"R²   = {r2:.4f}")
    check("Predictions clipped [0,20]",
          np.all((metrics["y_pred"] >= 0) & (metrics["y_pred"] <= 20)))

    print(f"\n  📊 Actual metrics on test set:")
    print(f"      MAE  = {mae:.4f}")
    print(f"      RMSE = {rmse:.4f}")
    print(f"      R²   = {r2:.4f}")
except Exception as e:
    check("Evaluation", False, str(e))


# ── 6. Feature importance ─────────────────────────────────────────────────────
print("\n-- 6. Feature Importance -------------------------------------------")
try:
    importance = get_feature_importance(model, feature_names)
    check("Returns dict",        isinstance(importance, dict))
    check("All features present", len(importance) == len(feature_names))
    check("Importances sum ≈ 1",
          abs(sum(importance.values()) - 1.0) < 1e-6,
          f"Sum = {sum(importance.values()):.6f}")
    top_feat = list(importance.keys())[0]
    print(f"  🏆 Top feature: {top_feat} ({importance[top_feat]:.4f})")
except Exception as e:
    check("Feature importance", False, str(e))


# ── 7. Single prediction ──────────────────────────────────────────────────────
print("\n-- 7. Single Prediction --------------------------------------------")
try:
    # Build a sample input using mean/mode values
    sample = {}
    for col in feature_names:
        if col in encs:
            sample[col] = 0   # first encoded label
        else:
            sample[col] = int(raw_df[col].median())
    input_df = pd.DataFrame([sample])[feature_names]
    grade = predict_single(model, input_df)
    check("Returns a float",      isinstance(grade, float))
    check("Grade in [0, 20]",     0 <= grade <= 20, f"Grade = {grade:.2f}")
    print(f"  Sample prediction: G3 = {grade:.2f}")
except Exception as e:
    check("Single prediction", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'=' * 60}")
print(f"  Results: {passed}/{total} checks passed")
print(f"{'=' * 60}\n")

if passed < total:
    sys.exit(1)
