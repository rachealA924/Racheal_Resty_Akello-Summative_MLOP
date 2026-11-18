"""
config.py: one place for every path and setting used by the pipeline.

Paths can be overridden with environment variables, which is how the Docker
container points the app at its own data and model folders.
"""

import os
from pathlib import Path

ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"          # data/train/<class>/*.jpg (sample of the Kaggle data)
TEST_DIR = DATA_DIR / "test"            # data/test/<class>/*.jpg  (fixed held-out set)
UPLOAD_DIR = DATA_DIR / "uploads"       # data/uploads/<class>/*.jpg (new data from users)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "cassava.db"))

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "cassava_mobilenetv2.keras"   # the model in production
METADATA_PATH = MODELS_DIR / "model_metadata.json"      # version, metrics, history
ARCHIVE_DIR = MODELS_DIR / "archive"                    # older versions (for rollback)

# The 5 classes from the Kaggle "Cassava Leaf Disease Classification" dataset.
# The folder names in data/train and data/test must match these exactly.
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
CLASS_DESCRIPTIONS = {
    "CBB": "Cassava Bacterial Blight",
    "CBSD": "Cassava Brown Streak Disease",
    "CGM": "Cassava Green Mottle",
    "CMD": "Cassava Mosaic Disease",
    "Healthy": "Healthy leaf",
}

IMG_SIZE = 224          # MobileNetV2 input size
BATCH_SIZE = 32
SEED = 42

# Retraining
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 25))  # auto trigger after this many new images
REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", 30))    # old images mixed in so the model does not forget
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 10))
RETRAIN_LR = 1e-4                                            # small learning rate: we fine tune, not start over
PROMOTION_TOLERANCE = 0.02  # new model is promoted unless its macro F1 drops more than this

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
