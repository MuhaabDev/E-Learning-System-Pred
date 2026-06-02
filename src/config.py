from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_REGRESSION = PROJECT_ROOT / "data" / "regression"
DATA_CLASSIFICATION = PROJECT_ROOT / "data" / "classification"
MODELS_DIR = PROJECT_ROOT / "models"

REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.pkl"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
REGRESSION_ENCODERS_PATH = MODELS_DIR / "regression_encoders.pkl"
METADATA_PATH = MODELS_DIR / "metadata.pkl"
