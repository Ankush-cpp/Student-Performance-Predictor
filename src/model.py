"""
model.py
--------
Model training, cross-validation comparison, evaluation, and inference.

Three regressors are compared with 5-fold cross-validation on the training set:
  • RandomForestRegressor
  • ExtraTreesRegressor
  • GradientBoostingRegressor

The best candidate (by mean CV MAE) is selected and evaluated once on the
held-out test set.  The test set is never touched during model selection.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

# ── candidate configurations ─────────────────────────────────────────────────

def _candidate_models(random_state: int = 42) -> dict:
    """Return the three candidate regressors used for comparison."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        ),
    }


# ── cross-validation comparison ───────────────────────────────────────────────

def compare_models(X_train, y_train, random_state: int = 42, cv: int = 5) -> dict:
    """
    Run k-fold cross-validation on all candidate models using identical folds.

    Parameters
    ----------
    X_train      : array-like  — training features (preprocessed)
    y_train      : array-like  — training target
    random_state : int         — passed to each model for reproducibility
    cv           : int         — number of CV folds (default 5)

    Returns
    -------
    results : dict  — {model_name: {"mean_mae": float, "std_mae": float,
                                    "mean_rmse": float, "std_rmse": float}}
              Sorted by mean_mae ascending (best first).
    """
    candidates = _candidate_models(random_state)
    results = {}

    for name, model in candidates.items():
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error",
            },
            n_jobs=-1,
            return_train_score=False,
        )
        mean_mae  = -cv_scores["test_neg_mae"].mean()
        std_mae   =  cv_scores["test_neg_mae"].std()
        mean_rmse = -cv_scores["test_neg_rmse"].mean()
        std_rmse  =  cv_scores["test_neg_rmse"].std()

        results[name] = {
            "mean_mae":  round(mean_mae,  4),
            "std_mae":   round(std_mae,   4),
            "mean_rmse": round(mean_rmse, 4),
            "std_rmse":  round(std_rmse,  4),
        }

    # Sort by mean CV MAE ascending
    results = dict(sorted(results.items(), key=lambda kv: kv[1]["mean_mae"]))
    return results


# ── training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, random_state: int = 42):
    """
    Select the best candidate via cross-validation and refit it on the full
    training set.

    Returns
    -------
    model         : fitted estimator
    cv_results    : dict from compare_models (for display / logging)
    best_name     : str  — name of the winning algorithm
    """
    cv_results = compare_models(X_train, y_train, random_state=random_state)
    best_name = next(iter(cv_results))          # first key = lowest mean MAE
    candidates = _candidate_models(random_state)
    model = candidates[best_name]
    model.fit(X_train, y_train)
    return model, cv_results, best_name


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute MAE, RMSE, and R² on the held-out test set.

    The model is evaluated exactly once; grades are clipped to [0, 20].

    Returns
    -------
    dict with keys: mae, rmse, r2, y_pred
    """
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 20)   # grades are in [0, 20]

    mae  = mean_absolute_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    r2   = r2_score(y_test, y_pred_clipped)

    return {
        "mae":    round(mae,  4),
        "rmse":   round(rmse, 4),
        "r2":     round(r2,   4),
        "y_pred": y_pred_clipped,
    }


# ── feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> dict:
    """
    Return feature importances sorted in descending order.

    Works for any sklearn estimator that exposes ``feature_importances_``.

    Returns
    -------
    dict: {feature_name: importance_value, ...}
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not expose feature_importances_."
        )
    importances = model.feature_importances_
    paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: imp for name, imp in paired}


# ── single prediction ─────────────────────────────────────────────────────────

def predict_single(model, input_df) -> float:
    """
    Predict the G3 grade for a single student.

    Parameters
    ----------
    model    : fitted estimator
    input_df : 2-D array-like (one row), already preprocessed

    Returns
    -------
    float in [0, 20]
    """
    raw = model.predict(input_df)[0]
    return float(np.clip(raw, 0, 20))
