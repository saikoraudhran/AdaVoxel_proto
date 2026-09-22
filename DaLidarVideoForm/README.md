# ADAVOXEL - LiDAR Sequence Viewer (Phase F1/F2)

No ML yet. This just proves the pipeline: your preprocessed `.npz` frames
(from `preprocess_semantickitti.py`) go in, a video-like playback comes out
in the browser. Once this works, the model plugs into `backend/main.py`
without the frontend needing to change (see "Next step" below).

## 1. Backend

```powershell
cd backend
pip install -r requirements.txt

# point this at the folder of .npz files you already produced
$env:FRAMES_DIR = "C:\Users\dhanu\SIH\Prep_Point_Clouds"

python main.py
```

This starts the API at `http://localhost:8000`. Check it's working by
opening `http://localhost:8000/frames` in a browser -- you should see a
JSON list of frame ids.

## 2. Frontend

Just open `frontend/index.html` directly in your browser (double-click it,
or drag it into a browser window). It talks to `http://localhost:8000` by
default -- edit the `BACKEND_URL` constant near the top of the `<script>`
if you run the backend somewhere else.

You'll get a black 3D viewport with your point cloud sequence, colored by
class (gray/green/yellow/red/blue -- see the sidebar legend), with
play/pause, a seek bar, and a playback speed selector, plus point count
and per-frame load time in the sidebar.

## How it works

- `backend/main.py` reads each `<frame_id>.npz` (produced by
  `preprocess_semantickitti.py`) and serves it as **raw binary**, not
  JSON -- a frame's points as float32 triples, followed by one uint8 class
  label per point. This matters because JSON-encoding 100k+ points per
  frame would be huge and slow; raw binary keeps it fast.
- `frontend/index.html` is a single self-contained file using Three.js
  (loaded from a CDN) to render each frame as a colored point cloud, and
  a simple timer loop to advance frames like video playback. It prefetches
  the next couple of frames in the background so scrubbing feels smooth.

## Next step: plugging in the model

When your model is ready, you don't need to touch the frontend at all --
just change what `main.py`'s `/frames/{frame_id}` endpoint returns:
instead of loading a pre-made `.npz`, it would run inference on that
frame's raw points and build the same binary response (points + predicted
class per point) on the fly. The frontend only ever consumes that same
binary contract, so the "video" experience doesn't change when real
inference replaces the pre-processed stand-in data.
