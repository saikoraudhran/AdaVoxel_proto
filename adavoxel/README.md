# ADAVOXEL - V1 frontend prototype

Single-file demo (no build step). Open `index.html` in a browser.

- Flow: Upload -> Processing -> Viewer (frame slider, playback, modes, L0/L1/L2, zoom/pan, click-to-inspect)
- Data: generated in the browser by `frame(f)` as a stand-in for `GET /api/sequences/{job_id}/frames/{frame_id}`
- To connect a backend: replace `frame(f)` with a fetch to that endpoint (same stats shape as the plan, section 20)
- Fonts load from Google Fonts; falls back to system fonts offline
