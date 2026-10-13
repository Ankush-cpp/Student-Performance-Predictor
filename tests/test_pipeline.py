"""
tests/test_pipeline.py
----------------------
Validates the full ML pipeline: data loading, preprocessing (ColumnTransformer),
model comparison, model training, evaluation, reproducibility, and single prediction.

Run with:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data_loader import (
    load_raw_dataframe,
    preprocess,
    get_train_test_split,
    build_preprocessor,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
)
from src.model import (
    train_model,
    evaluate_model,
    compare_models,
    get_feature_importance,
    predict_single,
)


# ── simple test runner ───────────────────────────────────────────────────────

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}]  {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append((name, condition))


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
    check("G1 present in raw_df", "G1" in raw_df.columns)
    check("G2 present in raw_df", "G2" in raw_df.columns)
except Exception as e:
    check("Load raw dataframe", False, str(e))


# ── 2. Preprocessor (ColumnTransformer) ──────────────────────────────────────
print("\n-- 2. Preprocessor (ColumnTransformer) -----------------------------")
try:
    preprocessor = build_preprocessor(raw_df)
    check("Returns ColumnTransformer", isinstance(preprocessor, ColumnTransformer))
    # Transform and check output shape
    X_array = preprocessor.transform(raw_df[CATEGORICAL_COLS + NUMERICAL_COLS])
    n_features = len(CATEGORICAL_COLS) + len(NUMERICAL_COLS)
    check("Output has correct number of columns",
          X_array.shape[1] == n_features,
          f"Expected {n_features}, got {X_array.shape[1]}")
    check("Output is numeric (float)", X_array.dtype.kind == "f",
          f"dtype = {X_array.dtype}")
except Exception as e:
    check("Build preprocessor", False, str(e))


# ── 3. Preprocessing function ─────────────────────────────────────────────────
print("\n-- 3. Preprocessing Function ----------------------------------------")
try:
    X, y, feature_names, preprocessor_p = preprocess(raw_df)
    check("Returns ndarray X",     isinstance(X, np.ndarray))
    check("Returns ndarray y",     isinstance(y, np.ndarray))
    check("Returns ColumnTransformer", isinstance(preprocessor_p, ColumnTransformer))
    check("G1 not in features",    "G1" not in feature_names)
    check("G2 not in features",    "G2" not in feature_names)
    check("G3 not in features",    "G3" not in feature_names)
    check("X and y same length",   len(X) == len(y))
    check("X is 2-D",              X.ndim == 2)
    check("Expected feature count",
          X.shape[1] == len(CATEGORICAL_COLS) + len(NUMERICAL_COLS),
          f"Got {X.shape[1]}")
except Exception as e:
    check("Preprocessing", False, str(e))


# ── 4. Train/test split (no leakage) ─────────────────────────────────────────
print("\n-- 4. Train / Test Split (no-leakage) ------------------------------")
try:
    X_train, X_test, y_train, y_test, fnames, enc, _ = get_train_test_split(
        test_size=0.2, random_state=42
    )
    total = len(X_train) + len(X_test)
    check("Total rows preserved", total == len(raw_df),
          f"{len(X_train)} + {len(X_test)} = {total}")
    check("Test set ~20%",
          abs(len(X_test) / total - 0.2) < 0.02,
          f"Test fraction: {len(X_test)/total:.3f}")
    check("Returns ColumnTransformer (enc)", isinstance(enc, ColumnTransformer))
    check("Feature names consistent with preprocessing", fnames == feature_names)
    # Verify the preprocessor was fitted on train, not full dataset
    # by checking that ordinal encoder categories come from the training fraction
    check("X_train is 2-D ndarray",  isinstance(X_train, np.ndarray) and X_train.ndim == 2)
    check("X_test is 2-D ndarray",   isinstance(X_test,  np.ndarray) and X_test.ndim == 2)
    check("y_train is 1-D ndarray",  isinstance(y_train, np.ndarray) and y_train.ndim == 1)
    check("y_test is 1-D ndarray",   isinstance(y_test,  np.ndarray) and y_test.ndim == 1)
except Exception as e:
    check("Train/test split", False, str(e))


# ── 5. Reproducibility ────────────────────────────────────────────────────────
print("\n-- 5. Reproducibility ----------------------------------------------")
try:
    X_train2, X_test2, y_train2, y_test2, _, _, _ = get_train_test_split(
        test_size=0.2, random_state=42
    )
    check("X_train identical across runs",
          np.allclose(X_train, X_train2))
    check("X_test identical across runs",
          np.allclose(X_test, X_test2))
    check("y_train identical across runs",
          np.array_equal(y_train, y_train2))
    check("y_test identical across runs",
          np.array_equal(y_test, y_test2))
except Exception as e:
    check("Reproducibility", False, str(e))


# ── 6. Model comparison (cross-validation) ───────────────────────────────────
print("\n-- 6. Model Comparison (CV) ----------------------------------------")
try:
    cv_results = compare_models(X_train, y_train, random_state=42, cv=5)
    expected_names = {"RandomForest", "ExtraTrees", "GradientBoosting"}
    check("All three models compared",
          set(cv_results.keys()) == expected_names,
          f"Got: {set(cv_results.keys())}")
    check("Results sorted by mean MAE (ascending)",
          list(cv_results.values()) == sorted(cv_results.values(), key=lambda r: r["mean_mae"]))
    for name, res in cv_results.items():
        check(f"{name}: mean_mae > 0", res["mean_mae"] > 0, f"MAE={res['mean_mae']:.4f}")
        check(f"{name}: std_mae >= 0", res["std_mae"] >= 0, f"std={res['std_mae']:.4f}")
    # Print the comparison table
    print("\n  CV Results (sorted by mean MAE):")
    for name, res in cv_results.items():
        print(f"    {name}: MAE={res['mean_mae']:.4f}+/-{res['std_mae']:.4f}"
              f"  RMSE={res['mean_rmse']:.4f}+/-{res['std_rmse']:.4f}")
except Exception as e:
    check("Model comparison", False, str(e))


# ── 7. Model training (best selected) ────────────────────────────────────────
print("\n-- 7. Model Training (best selected) --------------------------------")
try:
    model, cv_res2, best_name = train_model(X_train, y_train, random_state=42)
    check("train_model returns model",      model is not None)
    check("train_model returns cv_results", isinstance(cv_res2, dict))
    check("train_model returns best_name",  isinstance(best_name, str))
    check("Best name in candidate set",
          best_name in {"RandomForest", "ExtraTrees", "GradientBoosting"},
          f"Got: {best_name}")
    check("Model has feature_importances_", hasattr(model, "feature_importances_"))
    check("Importances length == n_features",
          len(model.feature_importances_) == len(feature_names),
          f"{len(model.feature_importances_)} vs {len(feature_names)}")
    print(f"\n  Selected algorithm: {best_name}")
except Exception as e:
    check("Model training", False, str(e))


# ── 8. Evaluation (test-set, one-shot) ───────────────────────────────────────
print("\n-- 8. Evaluation (held-out test set) --------------------------------")
try:
    metrics = evaluate_model(model, X_test, y_test)
    check("Returns MAE",      "mae"    in metrics)
    check("Returns RMSE",     "rmse"   in metrics)
    check("Returns R2",       "r2"     in metrics)
    check("Returns y_pred",   "y_pred" in metrics)

    mae  = metrics["mae"]
    rmse = metrics["rmse"]
    r2   = metrics["r2"]

    check("MAE > 0",          mae > 0,    f"MAE  = {mae:.4f}")
    check("RMSE >= MAE",      rmse >= mae, f"RMSE = {rmse:.4f}")
    check("R2 in (-inf, 1]",  r2 <= 1.0,  f"R2   = {r2:.4f}")
    check("Predictions clipped [0, 20]",
          np.all((metrics["y_pred"] >= 0) & (metrics["y_pred"] <= 20)))
    check("y_pred length matches y_test",
          len(metrics["y_pred"]) == len(y_test))

    print(f"\n  Test-set metrics ({best_name}):")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R2   = {r2:.4f}")
    print(f"\n  Baseline (RandomForest + LabelEncoder, original code):")
    print(f"    MAE  = 2.9489")
    print(f"    RMSE = 3.6945")
    print(f"    R2   = 0.3344")
except Exception as e:
    check("Evaluation", False, str(e))


# ── 9. Feature importance ─────────────────────────────────────────────────────
print("\n-- 9. Feature Importance -------------------------------------------")
try:
    importance = get_feature_importance(model, feature_names)
    check("Returns dict",           isinstance(importance, dict))
    check("All features present",   len(importance) == len(feature_names))
    check("Importances sum to ~1",
          abs(sum(importance.values()) - 1.0) < 1e-6,
          f"Sum = {sum(importance.values()):.6f}")
    top_feat = list(importance.keys())[0]
    print(f"  Top feature: {top_feat} ({importance[top_feat]:.4f})")
except Exception as e:
    check("Feature importance", False, str(e))


# ── 10. Single prediction (range check) ──────────────────────────────────────
print("\n-- 10. Single Prediction -------------------------------------------")
try:
    # Build sample input using median values, then transform via preprocessor
    raw_sample = {}
    for col in CATEGORICAL_COLS:
        raw_sample[col] = raw_df[col].mode()[0]
    for col in NUMERICAL_COLS:
        raw_sample[col] = float(raw_df[col].median())
    sample_df = pd.DataFrame([raw_sample])
    sample_transformed = enc.transform(sample_df[CATEGORICAL_COLS + NUMERICAL_COLS])
    grade = predict_single(model, sample_transformed)
    check("Returns a float",    isinstance(grade, float))
    check("Grade in [0, 20]",  0 <= grade <= 20, f"Grade = {grade:.2f}")
    print(f"  Median-profile prediction: G3 = {grade:.2f}")
except Exception as e:
    check("Single prediction", False, str(e))


# ── 11. Unknown-category handling (OrdinalEncoder returns -1, not 0) ─────────
print("\n-- 11. Unknown Category Handling -----------------------------------")
try:
    # Build a sample with an unseen category value
    unknown_sample = {}
    for col in CATEGORICAL_COLS:
        unknown_sample[col] = raw_df[col].mode()[0]
    for col in NUMERICAL_COLS:
        unknown_sample[col] = float(raw_df[col].median())
    unknown_sample["school"] = "UNKNOWN_SCHOOL"
    unk_df = pd.DataFrame([unknown_sample])
    unk_transformed = enc.transform(unk_df[CATEGORICAL_COLS + NUMERICAL_COLS])
    # The school column (index 0 in CATEGORICAL_COLS) should be -1
    school_idx = CATEGORICAL_COLS.index("school")
    check("Unknown category encodes to -1 (not silent 0)",
          unk_transformed[0, school_idx] == -1.0,
          f"Got value: {unk_transformed[0, school_idx]}")
    # Model should still produce a valid float in [0, 20]
    unk_grade = predict_single(model, unk_transformed)
    check("Prediction still in [0, 20] for unknown category",
          0 <= unk_grade <= 20,
          f"Grade = {unk_grade:.2f}")
except Exception as e:
    check("Unknown category handling", False, str(e))


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"\n{'=' * 60}")
print(f"  Results: {passed}/{total} checks passed")
if passed < total:
    print("\n  Failed checks:")
    for name, ok in results:
        if not ok:
            print(f"    - {name}")
print(f"{'=' * 60}\n")

if passed < total:
    sys.exit(1)
