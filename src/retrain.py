"""
retrain.py: the retraining pipeline.

Triggers
  manual : the "Retrain model" button in the UI (POST /retrain)
  auto   : runs by itself once RETRAIN_THRESHOLD new images are waiting

Steps
  1. Data:          read the uploaded images that have not been used yet (from the
                    database), plus a "replay" sample of old training images so the
                    model does not forget what it already knew
  2. Preprocessing: the same steps as everywhere else (RGB, 224x224), then each image
                    goes through the frozen MobileNetV2 backbone to get its features
  3. Retraining:    load the CURRENT production model and fine tune its head on the
                    new data (small learning rate, early stopping, class weights).
                    The saved model is the pretrained starting point; nothing is
                    trained from scratch.
  4. Evaluation:    score the old and the new model on the same fixed test set
  5. Promotion:     the new model replaces the old one unless its macro F1 drops by
                    more than PROMOTION_TOLERANCE. The old file is archived either way.

Command line:
    python -m src.retrain            # retrain now with all pending uploads
"""

import shutil
import threading
import traceback

import numpy as np

from src import database as db
from src.config import (
    CLASS_NAMES, PROMOTION_TOLERANCE, REPLAY_PER_CLASS, RETRAIN_EPOCHS, RETRAIN_LR,
    RETRAIN_THRESHOLD, SEED, TEST_DIR, TRAIN_DIR,
)
from src.model import (
    compute_metrics, extract_features, load_model, save_model, soft_class_weights,
    split_model, train_head,
)
from src.preprocessing import list_labelled_images, load_batch

_lock = threading.Lock()
STATUS = {"running": False, "run_id": None, "step": "idle", "progress": 0, "last_result": None}


def _step(text, progress):
    STATUS.update(step=text, progress=progress)
    print(f"[retrain] {progress:>3}% {text}", flush=True)


def is_running():
    return STATUS["running"]


def _replay_sample(rng):
    """A few old training images per class, so the model keeps its old knowledge."""
    paths, labels = list_labelled_images(TRAIN_DIR)
    chosen_p, chosen_y = [], []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(REPLAY_PER_CLASS, len(idx)), replace=False)
        chosen_p += [paths[i] for i in pick]
        chosen_y += [int(labels[i]) for i in pick]
    return chosen_p, np.array(chosen_y, dtype=np.int64)


def _train_val_split(n, y, rng, val_share=0.2):
    """Simple stratified split: about 20% of each class goes to validation."""
    val = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * val_share)) if len(idx) >= 5 else 0
        val += idx[:k].tolist()
    val = np.array(sorted(val), dtype=int)
    train = np.setdiff1d(np.arange(n), val)
    return train, val


def run_retraining(trigger="manual"):
    """Run the full pipeline once. Returns a summary dict. Only one run at a time."""
    if not _lock.acquire(blocking=False):
        return {"status": "busy", "message": "A retraining run is already in progress."}
    run_id = None
    try:
        STATUS.update(running=True, progress=0, step="starting")
        pending = db.pending_uploads()
        if not pending:
            return {"status": "skipped", "message": "No new uploaded images to train on."}
        run_id = db.start_run(trigger)
        STATUS["run_id"] = run_id
        rng = np.random.default_rng(SEED + run_id)

        # 1. DATA
        _step(f"Loading {len(pending)} uploaded images", 10)
        new_paths = [p["path"] for p in pending]
        new_y = np.array([CLASS_NAMES.index(p["label"]) for p in pending], dtype=np.int64)
        new_x, kept = load_batch(new_paths)
        keep_mask = np.array([p in kept for p in new_paths])
        new_y = new_y[keep_mask]

        replay_paths, replay_y = _replay_sample(rng)
        _step(f"Loading {len(replay_paths)} replay images from data/train", 20)
        replay_x, replay_kept = load_batch(replay_paths)
        replay_y = replay_y[[p in replay_kept for p in replay_paths]]

        test_paths, test_y = list_labelled_images(TEST_DIR)
        test_x, test_kept = load_batch(test_paths)
        test_y = test_y[[p in test_kept for p in test_paths]]
        if len(test_x) == 0:
            raise RuntimeError("data/test is empty, so the new model cannot be evaluated.")

        # 2. PREPROCESSING: FEATURES FROM THE FROZEN BACKBONE
        _step("Loading the current production model (pretrained starting point)", 30)
        model = load_model()
        feature_extractor, head = split_model(model)

        _step("Extracting MobileNetV2 features", 40)
        X = extract_features(feature_extractor, np.concatenate([new_x, replay_x]))
        y = np.concatenate([new_y, replay_y])
        X_test = extract_features(feature_extractor, test_x)

        _step("Evaluating the current model on the test set", 55)
        before = compute_metrics(test_y, head.predict(X_test, verbose=0))

        # 3. RETRAINING
        tr, va = _train_val_split(len(y), y, rng)
        _step(f"Fine tuning the head on {len(tr)} images ({len(va)} for validation)", 65)
        history = train_head(head, X[tr], y[tr], X[va] if len(va) else None,
                             y[va] if len(va) else None, epochs=RETRAIN_EPOCHS,
                             lr=RETRAIN_LR, class_weight=soft_class_weights(y[tr]))

        # 4. EVALUATION
        _step("Evaluating the retrained model on the test set", 85)
        after = compute_metrics(test_y, head.predict(X_test, verbose=0))

        # 5. PROMOTION
        promoted = after["f1_macro"] >= before["f1_macro"] - PROMOTION_TOLERANCE
        version = None
        if promoted:
            _step("Saving the new model version", 92)
            version = save_model(model, after, source=f"retrain ({trigger})",
                                 extra={"training_run_id": run_id,
                                        "n_new_images": int(len(new_y)),
                                        "epochs_run": len(history.get("loss", []))})
        # uploaded images become part of the training data for future runs
        moved = []
        for p in pending:
            dest = TRAIN_DIR / p["label"] / f"upload_{p['id']}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(p["path"], dest)
                moved.append(dest)
            except FileNotFoundError:
                moved.append(p["path"])
        db.mark_uploads_used([p["id"] for p in pending], run_id, moved)

        slim = lambda m: {k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}  # noqa: E731
        msg = (f"Promoted as version {version}." if promoted else
               "Kept the old model: the retrained one was worse on the test set.")
        db.finish_run(run_id, "success", msg, n_new_images=int(len(new_y)),
                      n_replay_images=int(len(replay_y)), metrics_before=slim(before),
                      metrics_after=slim(after), promoted=int(promoted), model_version=version)
        result = {"status": "success", "run_id": run_id, "trigger": trigger, "promoted": promoted,
                  "model_version": version, "message": msg, "n_new_images": int(len(new_y)),
                  "n_replay_images": int(len(replay_y)), "metrics_before": slim(before),
                  "metrics_after": slim(after), "history": history}
        _step("Done", 100)
        STATUS["last_result"] = result
        return result
    except Exception as e:  # report the error instead of crashing the API
        traceback.print_exc()
        if run_id:
            db.finish_run(run_id, "failed", f"{type(e).__name__}: {e}")
        result = {"status": "failed", "run_id": run_id, "message": f"{type(e).__name__}: {e}"}
        STATUS["last_result"] = result
        return result
    finally:
        STATUS["running"] = False
        _lock.release()


def start_in_background(trigger="manual"):
    """Start a run in a background thread so the API answers immediately."""
    if is_running():
        return False
    threading.Thread(target=run_retraining, args=(trigger,), daemon=True).start()
    return True


def maybe_auto_retrain():
    """The automatic trigger: start a run once enough new images are waiting."""
    pending = db.upload_counts()["pending_total"]
    if pending >= RETRAIN_THRESHOLD and not is_running():
        return start_in_background("auto")
    return False


if __name__ == "__main__":
    db.init_db()
    out = run_retraining("manual")
    print({k: v for k, v in out.items() if k != "history"})
