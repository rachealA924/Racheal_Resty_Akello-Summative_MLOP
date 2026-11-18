"""
model.py: build, train, evaluate, save and load the cassava classifier.

Architecture (the best model, E11, from the Intro to ML summative):

    image 224x224x3 (0..255)
      -> Rescaling to -1..1            (MobileNetV2's expected input range)
      -> MobileNetV2, ImageNet weights, FROZEN   (pretrained feature extractor)
      -> GlobalAveragePooling2D        -> 1,280 numbers per image ("features")
      -> Dense 256, ReLU, L2 regularisation
      -> Dropout 0.3
      -> Dense 5, softmax              -> one probability per class

Only the small "head" after the pooling layer is trained. That is what makes
retraining fast enough to run on a CPU in the cloud: each image goes through
the frozen backbone once to get its 1,280 features, then only the head learns.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")  # hide noisy retracing warnings
from keras import layers, regularizers  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from src.config import (  # noqa: E402
    ARCHIVE_DIR, BATCH_SIZE, CLASS_NAMES, IMG_SIZE, METADATA_PATH, MODEL_PATH, SEED,
)

N_CLASSES = len(CLASS_NAMES)


#  BUILD
def build_model(weights="imagenet", units=256, dropout=0.3, l2=1e-4, lr=1e-3):
    """Create the model. weights=None gives random weights (only used for offline tests)."""
    keras.utils.set_random_seed(SEED)
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 127.5, offset=-1, name="scale_to_minus1_1")(inputs)
    base = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="features")(x)
    x = layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="head_dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="head_output")(x)
    model = keras.Model(inputs, outputs, name="cassava_classifier")
    compile_model(model, lr)
    return model


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def split_model(model):
    """
    Split a trained model into
      feature_extractor: image -> 1,280 features (frozen backbone)
      head:              features -> class probabilities (the trainable part)
    The head REUSES the model's own layers, so training the head updates the
    full model in place. This is how a saved model is used as the pretrained
    starting point for retraining.
    """
    pool_idx = next(i for i, l in enumerate(model.layers)
                    if isinstance(l, layers.GlobalAveragePooling2D))
    feature_extractor = keras.Model(model.input, model.layers[pool_idx].output)
    head_in = keras.Input(model.layers[pool_idx].output.shape[1:], name="features_in")
    x = head_in
    for layer in model.layers[pool_idx + 1:]:
        x = layer(x)
    head = keras.Model(head_in, x, name="head")
    return feature_extractor, head


def extract_features(feature_extractor, images, batch_size=BATCH_SIZE):
    """Run images (n, 224, 224, 3) through the frozen backbone. Returns (n, 1280)."""
    if len(images) == 0:
        return np.zeros((0, feature_extractor.output.shape[-1]), dtype=np.float32)
    return feature_extractor.predict(images, batch_size=batch_size, verbose=0)


def soft_class_weights(y):
    """
    Square root of 'balanced' class weights (the setting that worked best in the
    summative, E11). Rare classes count a bit more, without over-correcting.
    Classes missing from y are left out.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    present = counts > 0
    w = np.zeros(N_CLASSES)
    w[present] = (len(y) / (present.sum() * counts[present])) ** 0.5
    return {i: float(w[i]) for i in range(N_CLASSES) if present[i]}


def train_head(head, X_train, y_train, X_val=None, y_val=None, epochs=10, lr=1e-3,
               patience=3, class_weight=None):
    """
    Train only the head on pre-extracted features.
    Optimisation: Adam optimiser, L2 regularisation and dropout (in the layers),
    early stopping on validation loss (keeps the best epoch), and reduce the
    learning rate when validation loss stops improving.
    """
    compile_model(head, lr)
    monitor = "val_loss" if X_val is not None and len(X_val) else "loss"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=2,
                                          min_lr=1e-6),
    ]
    val = (X_val, y_val) if monitor == "val_loss" else None
    hist = head.fit(X_train, y_train, validation_data=val, epochs=epochs,
                    batch_size=BATCH_SIZE, class_weight=class_weight,
                    callbacks=callbacks, verbose=0)
    return {k: [float(v) for v in vals] for k, vals in hist.history.items()}


#  EVALUATE
def compute_metrics(y_true, proba):
    """
    All the evaluation metrics in one dict:
      accuracy, loss (cross-entropy), macro precision, macro recall, macro F1,
      weighted F1, macro AUC (one-vs-rest), per-class report and confusion matrix.
    'Macro' means every class counts equally, which matters here because CMD is
    about 61% of the data: a model could look accurate while missing rare classes.
    """
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-7, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    labels = list(range(N_CLASSES))
    m = {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "loss": log_loss(y_true, proba, labels=labels),
        "precision_macro": precision_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0),
    }
    try:
        present = sorted(set(y_true.tolist()))
        if len(present) == N_CLASSES:
            m["auc_macro"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except ValueError:
        pass
    m = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()}
    m["per_class"] = classification_report(y_true, pred, labels=labels, target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0)
    m["confusion_matrix"] = confusion_matrix(y_true, pred, labels=labels).tolist()
    return m


def evaluate_model(model, images, labels):
    proba = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    return compute_metrics(labels, proba)


#  SAVE / LOAD WITH SIMPLE VERSIONING
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {"version": 0, "history": []}


def save_model(model, metrics, source, extra=None, path=MODEL_PATH):
    """
    Save the model as the new production version.
    The previous file is copied to models/archive/ first (for rollback), the new
    file is written to a temp file and then moved into place, so a running API
    never reads a half-written model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    if path.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ARCHIVE_DIR / f"cassava_mobilenetv2_v{meta.get('version', 0)}.keras")
    fd, tmp = tempfile.mkstemp(suffix=".keras", dir=path.parent)
    os.close(fd)
    model.save(tmp)
    os.replace(tmp, path)

    version = int(meta.get("version", 0)) + 1
    entry = {
        "version": version,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_metrics": {k: v for k, v in metrics.items() if k not in ("per_class",)},
        **(extra or {}),
    }
    meta.update({k: v for k, v in entry.items()})
    meta["test_metrics_full"] = metrics
    meta["class_names"] = CLASS_NAMES
    meta.setdefault("history", []).append(entry)
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    return version


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one with the notebook first.")
    return keras.models.load_model(path, compile=False)
