"""
main.py: the FastAPI app. It serves the web UI and these endpoints:

  GET  /                  the web UI
  GET  /health            uptime, model version, requests served (used by the UI and Docker)
  GET  /model             model version, test metrics and version history
  POST /predict           one image in, predicted class out
  POST /upload            bulk upload of labelled images (many files, or a .zip of class folders)
  POST /retrain           start a retraining run (manual trigger)
  GET  /retrain/status    progress of the current run and the list of past runs
  GET  /insights          numbers and interpretations for the data visualisations
  GET  /samples           test images with known labels, for trying predictions
  GET  /stats             prediction counts and upload counts

Run locally:
    uvicorn api.main:app --reload --port 8000
Interactive API docs (Swagger UI) are at /docs.
"""

import io
import os
import socket
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src import database as db
from src import retrain
from src.config import (
    ALLOWED_EXTENSIONS, CLASS_DESCRIPTIONS, CLASS_NAMES, RETRAIN_THRESHOLD, TEST_DIR, UPLOAD_DIR,
)
from src.insights import compute_insights
from src.model import load_metadata
from src.prediction import get_predictor
from src.preprocessing import InvalidImageError, list_labelled_images, save_upload

ROOT = Path(__file__).resolve().parents[1]
UI_FILE = ROOT / "ui" / "index.html"
MAX_UPLOAD_MB = 200
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "1") == "1"
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "1") == "1"

app = FastAPI(title="Cassava Leaf Disease Classifier",
              description="Predict, upload new data and retrain a MobileNetV2 cassava leaf model.",
              version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STARTED = time.time()
COUNTERS = {"requests": 0, "predictions": 0, "errors": 0, "predict_ms_total": 0.0}


@app.on_event("startup")
def startup():
    db.init_db()
    get_predictor().is_ready()   # load the model now, so the first user does not wait


@app.middleware("http")
async def count_requests(request: Request, call_next):
    COUNTERS["requests"] += 1
    try:
        response = await call_next(request)
    except Exception:
        COUNTERS["errors"] += 1
        raise
    if response.status_code >= 500:
        COUNTERS["errors"] += 1
    return response


#  UI
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return UI_FILE.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    """Model up-time and basic monitoring numbers. Async, so it answers even while predictions are busy."""
    ready = get_predictor().loaded
    up = time.time() - STARTED
    n = COUNTERS["predictions"]
    return {
        "status": "ok" if ready else "no_model",
        "model_loaded": ready,
        "model_version": get_predictor().loaded_version if ready else None,
        "started_at": datetime.fromtimestamp(STARTED, timezone.utc).isoformat(timespec="seconds"),
        "uptime_seconds": round(up, 1),
        "container": socket.gethostname(),   # shows which container answered (load balancing)
        "requests_served": COUNTERS["requests"],
        "predictions_served": n,
        "avg_prediction_ms": round(COUNTERS["predict_ms_total"] / n, 1) if n else None,
        "errors": COUNTERS["errors"],
        "retraining": retrain.is_running(),
    }


@app.get("/model")
def model_info():
    meta = load_metadata()
    if not meta.get("version"):
        raise HTTPException(404, "No model has been trained yet.")
    return meta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the disease class of one leaf photo (JPG or PNG)."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        # run the model in a worker thread so the server keeps answering other requests
        result = await run_in_threadpool(get_predictor().predict, data)
    except InvalidImageError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    COUNTERS["predictions"] += 1
    COUNTERS["predict_ms_total"] += result["latency_ms"]
    result["container"] = socket.gethostname()
    if LOG_PREDICTIONS:
        try:
            await run_in_threadpool(db.log_prediction, result["label"], result["confidence"],
                                    result["latency_ms"], result["model_version"])
        except Exception:
            pass  # monitoring must never break a prediction
    return result


def _iter_zip(data):
    """Yield (label, filename, bytes) from a zip laid out as <class>/<image>."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if any(p.startswith("__MACOSX") or p.startswith(".") for p in parts):
                continue
            if Path(info.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            label = next((p for p in reversed(parts[:-1]) if p in CLASS_NAMES), None)
            yield label, parts[-1], z.read(info)


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), label: str = Form("")):
    """
    Bulk upload for retraining.
    Either: several images plus one label for all of them,
    or:     a .zip with one folder per class (CBB/, CBSD/, CGM/, CMD/, Healthy/).
    Every image is validated, re-saved as a clean JPEG and recorded in the database.
    """
    if label and label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, rejected = [], []
    total_bytes = 0
    for f in files:
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_MB} MB.")
        items = []
        if f.filename.lower().endswith(".zip"):
            try:
                items = [(lab or label, name, b) for lab, name, b in _iter_zip(data)]
            except zipfile.BadZipFile:
                rejected.append({"file": f.filename, "reason": "not a valid zip file"})
                continue
        else:
            items = [(label, f.filename, data)]
        for lab, name, b in items:
            if not lab:
                rejected.append({"file": name, "reason": "no label (choose one, or use class folders in the zip)"})
                continue
            try:
                path = save_upload(b, lab, UPLOAD_DIR, name)
            except (InvalidImageError, ValueError) as e:
                rejected.append({"file": name, "reason": str(e)})
                continue
            db.add_upload(name, lab, path)
            saved.append({"file": name, "label": lab})
    if not saved and not rejected:
        raise HTTPException(400, "No images found in the upload.")
    auto_started = retrain.maybe_auto_retrain() if AUTO_RETRAIN and saved else False
    counts = db.upload_counts()
    return {"saved": len(saved), "rejected": rejected, "saved_by_label": _count(saved),
            "pending_total": counts["pending_total"], "retrain_threshold": RETRAIN_THRESHOLD,
            "auto_retrain_started": auto_started}


def _count(items):
    out = {}
    for it in items:
        out[it["label"]] = out.get(it["label"], 0) + 1
    return out


@app.post("/retrain", status_code=202)
def trigger_retrain():
    """Manual trigger: retrain on all uploaded images that have not been used yet."""
    if retrain.is_running():
        raise HTTPException(409, "A retraining run is already in progress.")
    if db.upload_counts()["pending_total"] == 0:
        raise HTTPException(400, "Upload some labelled images first.")
    retrain.start_in_background("manual")
    return {"message": "Retraining started.", "status_url": "/retrain/status"}


@app.get("/retrain/status")
def retrain_status():
    return {**retrain.STATUS, "uploads": db.upload_counts(),
            "retrain_threshold": RETRAIN_THRESHOLD, "auto_retrain": AUTO_RETRAIN,
            "runs": db.list_runs()}


@app.get("/insights")
def insights():
    return compute_insights()


@app.get("/samples")
def samples(per_class: int = 4):
    """Test images with their true label, so a prediction can be checked against the truth."""
    paths, labels = list_labelled_images(TEST_DIR)
    out, seen = [], {}
    for p, y in zip(paths, labels):
        c = CLASS_NAMES[y]
        if seen.get(c, 0) >= per_class:
            continue
        seen[c] = seen.get(c, 0) + 1
        out.append({"label": c, "description": CLASS_DESCRIPTIONS[c], "name": p.name,
                    "url": f"/samples/{c}/{p.name}"})
    return out


@app.get("/samples/{label}/{name}", include_in_schema=False)
def sample_file(label: str, name: str):
    if label not in CLASS_NAMES or "/" in name or ".." in name:
        raise HTTPException(404)
    path = TEST_DIR / label / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/stats")
def stats():
    return {"predictions": db.prediction_summary(), "uploads": db.upload_counts()}
