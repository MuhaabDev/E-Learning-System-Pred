"""
Load persisted estimators + preprocessors exactly once per process (use with Streamlit cache).
Extended to carry optional evaluation data for the analytics dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.preprocessing import LabelEncoder

from src import config


@dataclass(frozen=True)
class ModelArtifacts:
    regression_model: Any
    classification_model: Any
    scalers: Dict[str, Any]
    label_encoder: LabelEncoder
    regression_encoders: Dict[str, Any]
    metadata: Dict[str, Any]
    # Optional: stored at train-time for analytics (may be empty dict if model predates this)
    eval_data: Dict[str, Any] = field(default_factory=dict)


def _assert_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model artifact: {path}. "
            "Run src/train_regression.py and src/train_classification.py first."
        )


def load_all(models_dir: Optional[Path] = None) -> ModelArtifacts:
    root = models_dir or config.MODELS_DIR
    paths = {
        "reg": root / "regression_model.pkl",
        "clf": root / "classification_model.pkl",
        "scalers": root / "scaler.pkl",
        "le": root / "label_encoder.pkl",
        "re": root / "regression_encoders.pkl",
        "meta": root / "metadata.pkl",
    }
    for p in paths.values():
        _assert_exists(p)

    eval_path = root / "eval_data.pkl"
    eval_data: Dict[str, Any] = joblib.load(eval_path) if eval_path.is_file() else {}

    return ModelArtifacts(
        regression_model=joblib.load(paths["reg"]),
        classification_model=joblib.load(paths["clf"]),
        scalers=joblib.load(paths["scalers"]),
        label_encoder=joblib.load(paths["le"]),
        regression_encoders=joblib.load(paths["re"]),
        metadata=joblib.load(paths["meta"]),
        eval_data=eval_data,
    )
