from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config
from src.data_preprocessing import (
    build_classification_dataset,
    classification_preprocess_before_learning,
    merge_classification_master,
    load_tables,
)


def _load_or_new_scalers() -> Dict[str, Any]:
    if config.SCALER_PATH.is_file():
        d = joblib.load(config.SCALER_PATH)
        if not isinstance(d, dict):
            raise ValueError("scaler.pkl must be a dict with 'regression' and/or 'classification' keys")
        return d
    return {}


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tab = load_tables("classification")
    X, y, label_encoder, cat_cols_observed, inference_stats = build_classification_dataset(tab)

    feat_cols = list(X.columns)
    ohe_cols: List[str] = [
        c for c in X.columns if any(c.startswith(f"{cat}_") for cat in cat_cols_observed)
    ]
    numeric_cols = [c for c in X.columns if c not in ohe_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = clf_scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = clf_scaler.transform(X_test[numeric_cols])

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)

    acc = float(accuracy_score(y_test, y_hat))
    macro_f1 = float(f1_score(y_test, y_hat, average="macro"))
    macro_prec = float(precision_score(y_test, y_hat, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_hat, average="macro", zero_division=0))

    print("\nTest metrics — RandomForestClassifier")
    print(f"  accuracy={acc:.4f}")
    print(f"  macro-F1={macro_f1:.4f}")
    print(classification_report(y_test, y_hat, target_names=label_encoder.classes_))

    joblib.dump(rf, config.CLASSIFICATION_MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    scalers = _load_or_new_scalers()
    scalers["classification"] = clf_scaler
    joblib.dump(scalers, config.SCALER_PATH)

    meta_path = config.METADATA_PATH
    meta = joblib.load(meta_path) if meta_path.is_file() else {}
    pre_le = classification_preprocess_before_learning(merge_classification_master(tab))
    categorical_vocab: Dict[str, List[str]] = {}
    for col in pre_le.select_dtypes(include="object").columns:
        if col == "assessmentClass":
            continue
        categorical_vocab[col] = sorted(pre_le[col].dropna().astype(str).unique().tolist())

    clean_rows = pre_le.dropna(axis=0, how="any")
    seed_ix = max(1, len(clean_rows) // 8)
    seed_ix = min(seed_ix, len(clean_rows) - 1)
    default_input_row = clean_rows.iloc[seed_ix].drop(labels=["assessmentClass"], errors="ignore").to_dict()

    meta["classification"] = {
        "feature_columns": feat_cols,
        "numeric_cols_scale": numeric_cols,
        "ohe_columns_prefix": ohe_cols,
        "cat_cols_observed": cat_cols_observed,
        "inference_stats": inference_stats,
        "accuracy_test": acc,
        "macro_f1_test": macro_f1,
        "categorical_vocab": categorical_vocab,
        "default_input_row": default_input_row,
    }
    meta["paths"] = {"PROJECT_ROOT": str(config.PROJECT_ROOT)}
    joblib.dump(meta, meta_path)

    # ── save eval data for analytics dashboard ──────────────────────────────
    eval_path = config.MODELS_DIR / "eval_data.pkl"
    eval_data: Dict[str, Any] = joblib.load(eval_path) if eval_path.is_file() else {}

    cm = confusion_matrix(y_test, y_hat)
    fi = None
    if hasattr(rf, "feature_importances_"):
        fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    # class distribution in full dataset
    target_counts = pd.Series(label_encoder.inverse_transform(y.values.astype(int))).value_counts()

    # per-class report dict
    report_dict = classification_report(
        y_test, y_hat, target_names=label_encoder.classes_, output_dict=True
    )

    # ROC (one-vs-rest, macro, if proba available)
    roc_auc = None
    try:
        y_prob = rf.predict_proba(X_test)
        if len(label_encoder.classes_) == 2:
            roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass

    eval_data["classification"] = {
        "y_test": y_test.values,
        "y_pred": y_hat,
        "confusion_matrix": cm,
        "class_names": label_encoder.classes_,
        "feature_importance": fi,
        "feature_columns": feat_cols,
        "target_distribution": target_counts,
        "classification_report": report_dict,
        "roc_auc": roc_auc,
        "metrics": {
            "Accuracy": acc,
            "Precision (macro)": macro_prec,
            "Recall (macro)": macro_rec,
            "F1 Score (macro)": macro_f1,
        },
    }
    joblib.dump(eval_data, eval_path)

    print(f"\nSaved classification RF -> {config.CLASSIFICATION_MODEL_PATH}")


if __name__ == "__main__":
    main()
