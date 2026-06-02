"""
Inference helpers — expect models + scalers loaded via model_loader.
Supports single-row and batch (DataFrame) predictions for both tasks.
"""
from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

from src.data_preprocessing import transform_classification_inference_row
from src.model_loader import ModelArtifacts


# ─────────────────────────────── Regression ───────────────────────────────

def predict_regression(artifacts: ModelArtifacts, X: pd.DataFrame) -> np.ndarray:
    """X: column order matches training regression features."""
    scaler = artifacts.scalers["regression"]
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return artifacts.regression_model.predict(Xs)


def predict_regression_batch(artifacts: ModelArtifacts, df: pd.DataFrame) -> np.ndarray:
    """
    Batch regression prediction from a raw uploaded CSV.
    Aligns columns to training feature columns, fills missing with 0.
    """
    meta = artifacts.metadata["regression"]
    feat_cols = meta["feature_columns"]
    X = df.reindex(columns=feat_cols, fill_value=0)
    # coerce to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return predict_regression(artifacts, X)


# ─────────────────────────────── Classification ───────────────────────────

def predict_classification(
    artifacts: ModelArtifacts,
    *,
    raw_row: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-row classification prediction from raw merged-row fields.
    Returns (encoded_labels, decoded_class_names).
    """
    meta = artifacts.metadata["classification"]
    X_prep = transform_classification_inference_row(
        raw_row,
        training_columns=meta["feature_columns"],
        numeric_cols_scale=meta["numeric_cols_scale"],
        inference_stats=meta["inference_stats"],
    )

    scaler = artifacts.scalers["classification"]
    num_cols = list(meta["numeric_cols_scale"])
    Xs = X_prep.copy()
    if num_cols:
        Xs[num_cols] = scaler.transform(Xs[num_cols])

    y_hat = artifacts.classification_model.predict(Xs)
    decoded = artifacts.label_encoder.inverse_transform(y_hat.astype(int))
    return y_hat, decoded


def predict_classification_batch(
    artifacts: ModelArtifacts, df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch classification prediction: each row goes through the same inference
    pipeline used during training (transform_classification_inference_row).
    Returns (encoded_array, decoded_labels_array).
    """
    meta = artifacts.metadata["classification"]
    results_encoded = []
    results_decoded = []

    for _, row in df.iterrows():
        try:
            enc, dec = predict_classification(artifacts, raw_row=row.to_dict())
            results_encoded.append(enc[0])
            results_decoded.append(dec[0])
        except Exception:
            results_encoded.append(np.nan)
            results_decoded.append("ERROR")

    return np.array(results_encoded), np.array(results_decoded)
