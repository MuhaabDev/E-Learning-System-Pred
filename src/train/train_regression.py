from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_preprocessing import build_regression_dataset, load_tables


def _load_or_new_scalers() -> Dict[str, Any]:
    if config.SCALER_PATH.is_file():
        d = joblib.load(config.SCALER_PATH)
        if not isinstance(d, dict):
            return {"regression": d}
        return d
    return {}


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tab = load_tables("regression")
    X, y, enc_bundle = build_regression_dataset(tab)
    feat_cols = list(X.columns)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    reg_scaler = StandardScaler()
    X_train_s = pd.DataFrame(reg_scaler.fit_transform(X_train), columns=feat_cols)
    X_val_s = pd.DataFrame(reg_scaler.transform(X_val), columns=feat_cols)
    X_test_s = pd.DataFrame(reg_scaler.transform(X_test), columns=feat_cols)

    ridge = Ridge(alpha=10.0)
    lasso = Lasso(alpha=1.0, max_iter=5000)
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=25, random_state=42)

    model_names = ["Ridge", "Lasso", "Gradient Boosting", "Random Forest", "Decision Tree"]
    models = [ridge, lasso, gb, rf, dt]

    val_preds = []
    fitted: Dict[str, Any] = {}
    for name, model in zip(model_names, models):
        m = clone(model)
        m.fit(X_train_s, y_train)
        vp = m.predict(X_val_s)
        val_preds.append(vp)
        fitted[name] = m
        print(f"  {name:22s} Val R² = {r2_score(y_val, vp):.4f}")

    best_name = max(zip(model_names, val_preds), key=lambda t: r2_score(y_val, t[1]))[0]
    best_model = fitted[best_name]
    joblib.dump(best_model, config.REGRESSION_MODEL_PATH)

    scalers = _load_or_new_scalers()
    scalers["regression"] = reg_scaler
    joblib.dump(scalers, config.SCALER_PATH)
    joblib.dump(enc_bundle, config.REGRESSION_ENCODERS_PATH)

    meta_path = config.METADATA_PATH
    meta = joblib.load(meta_path) if meta_path.is_file() else {}

    preds_test = best_model.predict(X_test_s)
    defaults = X_train.median(numeric_only=True).to_dict()
    mae = float(mean_absolute_error(y_test, preds_test))
    mse = float(mean_squared_error(y_test, preds_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds_test))

    meta["regression"] = {
        "feature_columns": feat_cols,
        "best_model_val": best_name,
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_mse": mse,
        "defaults_numeric": defaults,
        "ordinal_columns": list(enc_bundle.get("ordinal_columns", [])),
        "nominal_columns": list(enc_bundle.get("nominal_columns", [])),
    }
    meta["paths"] = {"PROJECT_ROOT": str(config.PROJECT_ROOT)}
    joblib.dump(meta, meta_path)

    # ── save eval data for analytics dashboard ──────────────────────────────
    eval_path = config.MODELS_DIR / "eval_data.pkl"
    eval_data: Dict[str, Any] = joblib.load(eval_path) if eval_path.is_file() else {}

    # feature importances (if tree-based)
    fi = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    elif hasattr(best_model, "coef_"):
        fi = pd.Series(np.abs(best_model.coef_), index=feat_cols).sort_values(ascending=False)

    eval_data["regression"] = {
        "y_test": y_test.values,
        "y_pred": preds_test,
        "feature_importance": fi,
        "feature_columns": feat_cols,
        "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2},
        "best_model_name": best_name,
        "target_distribution": y.values,
    }
    joblib.dump(eval_data, eval_path)

    print(f"\nSaved regression ({best_name}) test R²={r2:.4f}")
    print(f"  -> {config.REGRESSION_MODEL_PATH}")


if __name__ == "__main__":
    main()
