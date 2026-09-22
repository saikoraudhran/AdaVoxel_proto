#!/usr/bin/env python3
"""
FastAPI backend for the LiDAR "video" viewer.

Purpose (Phase F1/F2, no ML yet): serve the .npz frames produced by
preprocess_semantickitti.py so the frontend can play them back like a
video. Each frame is streamed as compact raw binary rather than JSON,
since a frame can easily be 50k-150k points -- JSON would be enormous
and slow to parse.

Binary frame format (little-endian):
    [0:4]   uint32   N = number of points
    [4:4+12N]        N * (float32 x, float32 y, float32 z)
    [4+12N:4+13N]    N * uint8 class label

Run:
    pip install -r requirements.txt
    set FRAMES_DIR=C:\path\to\Prep_Point_Clouds      (Windows / PowerShell: $env:FRAMES_DIR=...)
    export FRAMES_DIR=/path/to/Prep_Point_Clouds     (Linux/Mac)
    python main.py
    (or: uvicorn main:app --reload --port 8000)

Endpoints:
    GET /frames            -> {"count": N, "frames": ["000000", "000001", ...]}
    GET /frames/{frame_id} -> binary point+label data for one frame
"""

import os
import struct
import glob

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

FRAMES_DIR = os.environ.get("FRAMES_DIR", "./frames")

app = FastAPI(title="LiDAR Viewer Backend")

# Frontend is a plain local HTML file (opened via file:// or a local
# static server), so allow any origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def list_frame_ids():
    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.npz")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


@app.get("/frames")
def get_frame_list():
    ids = list_frame_ids()
    if not ids:
        raise HTTPException(
            status_code=404,
            detail=f"No .npz files found in FRAMES_DIR='{FRAMES_DIR}'. "
                   f"Set the FRAMES_DIR environment variable to your "
                   f"preprocessed output folder before starting the server.",
        )
    return {"count": len(ids), "frames": ids}


@app.get("/frames/{frame_id}")
def get_frame(frame_id: str):
    path = os.path.join(FRAMES_DIR, f"{frame_id}.npz")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"frame not found: {frame_id}")

    data = np.load(path)
    points = data["points"][:, :3].astype(np.float32)

    if "sem_label" in data:
        labels = data["sem_label"].astype(np.uint8)
    else:
        labels = np.zeros(points.shape[0], dtype=np.uint8)

    n = points.shape[0]
    header = struct.pack("<I", n)
    payload = header + points.tobytes(order="C") + labels.tobytes()

    return Response(content=payload, media_type="application/octet-stream")


@app.get("/health")
def health():
    return {"status": "ok", "frames_dir": FRAMES_DIR, "frame_count": len(list_frame_ids())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
