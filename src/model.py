"""
model.py
--------
Trains a Random Forest Regressor on the preprocessed student data and
computes evaluation metrics (MAE, RMSE, R²).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    random_state : int  — ensures reproducibility

    Returns
    -------
    Fitted RandomForestRegressor
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model: RandomForestRegressor, X_test, y_test) -> dict:
    """
    Compute MAE, RMSE, and R² on the test set.

    Returns
    -------
    dict with keys: mae, rmse, r2, y_pred
    """
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 20)   # grades are in [0, 20]

    mae = mean_absolute_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    r2 = r2_score(y_test, y_pred_clipped)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred_clipped,
    }


# ── feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model: RandomForestRegressor, feature_names: list) -> dict:
    """
    Return feature importances sorted in descending order.

    Returns
    -------
    dict: {feature_name: importance_value, ...}
    """
    importances = model.feature_importances_
    paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: imp for name, imp in paired}


# ── single prediction ─────────────────────────────────────────────────────────

def predict_single(model: RandomForestRegressor, input_df) -> float:
    """
    Predict the G3 grade for a single student represented as a one-row DataFrame.
    The result is clipped to the valid [0, 20] range.
    """
    raw = model.predict(input_df)[0]
    return float(np.clip(raw, 0, 20))
