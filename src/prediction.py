"""
prediction.py: predict the class of one leaf image.

The model is loaded once and kept in memory. If the model file changes on disk
(after retraining), the next prediction reloads it automatically, so the API
always serves the newest promoted model without a restart.

Command line:
    python -m src.prediction path/to/leaf.jpg
"""

import sys
import threading
import time

import numpy as np

from src.config import CLASS_DESCRIPTIONS, CLASS_NAMES, MODEL_PATH
from src.model import load_metadata, load_model
from src.preprocessing import load_image


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._mtime = None
        self._version = None
        self._lock = threading.Lock()
        # One model call at a time per container. Image decoding still runs in parallel.
        # Running many TensorFlow calls at once on a small CPU makes every one of them slower.
        self._infer_lock = threading.Lock()

    def _ensure_loaded(self):
        mtime = self.model_path.stat().st_mtime if self.model_path.exists() else None
        if self._model is not None and mtime == self._mtime:
            return
        with self._lock:
            if self._model is not None and mtime == self._mtime:
                return
            model = load_model(self.model_path)
            model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)  # warm up
            self._model, self._mtime = model, mtime
            self._version = load_metadata().get("version")

    @property
    def loaded(self):
        """True once a model is in memory (no disk access, safe to call anywhere)."""
        return self._model is not None

    @property
    def loaded_version(self):
        return self._version

    @property
    def model_version(self):
        self._ensure_loaded()
        return self._version

    def is_ready(self):
        try:
            self._ensure_loaded()
            return True
        except FileNotFoundError:
            return False

    def predict(self, source):
        """source: file path, bytes or file object. Returns a dict with the prediction."""
        start = time.perf_counter()
        self._ensure_loaded()
        x = load_image(source)[None, ...]
        with self._infer_lock:
            proba = np.asarray(self._model(x, training=False))[0]
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        return {
            "label": CLASS_NAMES[best],
            "description": CLASS_DESCRIPTIONS[CLASS_NAMES[best]],
            "confidence": round(float(proba[best]), 4),
            "probabilities": {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(len(CLASS_NAMES))},
            "model_version": self._version,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


_default = None


def get_predictor():
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_image(source):
    return get_predictor().predict(source)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.prediction path/to/leaf.jpg")
    r = predict_image(sys.argv[1])
    print(f"{r['label']} ({r['description']}), confidence {r['confidence']:.1%}")
    for name, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<8} {p:.1%}")
