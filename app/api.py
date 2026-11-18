"""FastAPI backend (runs on Laptop 1).

    uvicorn app.api:app --host 0.0.0.0 --port 8000

GET  /trips            ids of the trips available in the evaluation folder
POST /predict          {"trip_id": "..."}  ->  GPS points, predicted route, accuracy metrics
Ground truth is read only inside score_trip(), on the server, and is never put in a response.
Every request is logged to the terminal (show this to the evaluator).
"""
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import ROOT, STEST_DIR
from pipeline import predict_trip, score_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

_cfg = json.load(open(Path(__file__).parent / "config.json"))
_eval = os.environ.get("PATHSNAP_EVAL_DIR") or _cfg.get("eval_dir", "auto")
EVAL_DIR = STEST_DIR if _eval == "auto" else (Path(_eval) if Path(_eval).is_absolute() else ROOT / _eval)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir()) if EVAL_DIR.exists() else []


@app.on_event("startup")
def startup():
    from pipeline import get_hmm_config, get_model
    get_model(), get_hmm_config()          # fail fast and load once
    log.info(f"PathSnap API ready. evaluation folder = {EVAL_DIR}  ({len(_trip_ids())} trips)")


@app.get("/trips")
def trips():
    ids = _trip_ids()
    log.info(f"GET /trips -> {len(ids)} trips")
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    ids = _trip_ids()
    if req.trip_id not in ids:                       # also blocks path traversal
        log.info(f"POST /predict trip={req.trip_id!r} -> 404 unknown trip")
        raise HTTPException(404, f"unknown trip {req.trip_id!r}")
    t0 = time.perf_counter()
    trip_dir = EVAL_DIR / req.trip_id
    log.info(f"POST /predict trip={req.trip_id} received")
    result = predict_trip(trip_dir)
    metrics = score_trip(trip_dir, result)
    total = time.perf_counter() - t0
    log.info(f"POST /predict trip={req.trip_id} done | obs={result['n_obs']} | "
             f"HMM top1={metrics.get('top1_accuracy', float('nan')):.4f} "
             f"(classifier only {metrics.get('classifier_only_top1', float('nan')):.4f}) | "
             f"continuity={metrics.get('route_continuity', float('nan')):.4f} | {total:.2f}s")
    return {"trip_id": req.trip_id, "n_obs": result["n_obs"], "gps": result["gps"],
            "pred_polylines": result["pred_polylines"], "metrics": metrics,
            "seconds": round(total, 3)}
