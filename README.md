# Traffic Control Room

Vehicle detection, tracking and traffic analytics on road video, using YOLO and ByteTrack.
Built on a fork of [Ultralytics](https://github.com/ultralytics/ultralytics).

- **Detection**: finds cars, motorcycles, buses and trucks in every processed frame.
- **Tracking**: gives each vehicle an ID that stays the same across frames (ByteTrack).
- **Analytics**: counts vehicles crossing a line across the middle of the frame, splits the count by
  type, and rates traffic density.
- **Reports**: shows the latest analytics result and downloads it as CSV.

## How it works

1. **Detect.** Each frame goes through the pretrained `yolo26n.pt` model (trained on COCO). Only four
   COCO classes are kept: car, motorcycle, bus and truck. Only detections with confidence of at least
   0.50 are kept, which cuts false positives; very small or distant vehicles may be missed. Change
   `CONF_THRESHOLD` in `vehicle_processing.py` to adjust this.
2. **Track.** ByteTrack links detections between frames and gives each vehicle a stable ID.
3. **Count.** A horizontal line is drawn at half the frame height. When a tracked vehicle's box centre
   moves from one side of the line to the other, it is counted once; its ID is remembered so it is
   never counted twice.
4. **Rate density.** Based on how many vehicles are visible in the current frame:
   up to 8 is **Low**, up to 20 is **Medium**, more is **High**.

For speed, the video is processed at YOLO image size 640, every 2nd frame is used, frames wider than
960 px are scaled down, and a GPU is used when one is available. These settings are at the top of
`traffic_monitoring/vehicle_processing.py`.

## Project layout

| Path | What it does |
| --- | --- |
| `traffic_monitoring/vehicle_processing.py` | Detection, tracking and line-crossing counting |
| `traffic_monitoring/congestion.py` | Turns the visible vehicle count into Low / Medium / High |
| `traffic_monitoring/server.py` | Local FastAPI server: runs the model, streams results and serves the dashboard |
| `frontend/` | The dashboard, in plain HTML, CSS and JavaScript (no build step, no framework) |
| `traffic_monitoring/requirements.txt` | Python packages |
| `traffic_monitoring/videos/trafficv.mp4` | Demo traffic video |
| `yolo26n.pt` | YOLO26 nano model weights |
| `ultralytics/` | Ultralytics YOLO library (upstream, unchanged) |

The model and demo video are included in the repo, so it runs straight after cloning.

## Run it locally

Everything runs on your own computer: one Python server does the YOLO processing and also serves
the JavaScript dashboard.

From the repo root:

```powershell
pip install -r traffic_monitoring/requirements.txt
cd traffic_monitoring
uvicorn server:app --port 8000
```

Then open <http://localhost:8000>. Choose a video under **Video source** in the sidebar, pick a module and press **Start**. The annotated
video streams into the page while the numbers update live; **Traffic Analytics** saves a report you
can view and download as CSV under **Reports**.

```
Browser (frontend/: HTML + CSS + JS)
   │  fetch /start, /stats, /report  ·  <img src="/stream">
   ▼
localhost:8000  FastAPI (server.py) ──► vehicle_processing.py ──► YOLO + ByteTrack
```

**Using your own video:** in the left panel under **Video source**, press **Upload custom video** (or drop a file on that box) and pick an `.mp4`, `.avi`,
`.mov` or `.mkv` file. It is saved into `traffic_monitoring/videos/` and selected in the **Video**
list. Every module then runs on the selected video. Uploaded videos stay on your computer and are not committed (`*.mp4` is git-ignored). To remove
one, delete it from that folder.

If the page says "Backend offline", the server isn't running; start it with the command above.

## Backend API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend status and model settings |
| `GET /videos` | Videos available in `traffic_monitoring/videos/` |
| `POST /videos?name=<file>` | Upload a video (the request body is the file) |
| `POST /start/{detect\|track\|count}?video=<file>` | Start a run on a video, default the demo (stops any run already going) |
| `POST /stop` | Stop the current run |
| `GET /stream` | Live annotated video as an MJPEG stream (used as an `<img>` source) |
| `GET /stats` | Live numbers for the current or last run |
| `GET /report` | Last finished traffic report from a `count` run |

| `GET /` | The dashboard (static files from `frontend/`) |

One run happens at a time and every open tab sees the same one.

## What this fork changed from Ultralytics

- **Added** the `traffic_monitoring/` module, the `frontend/` JavaScript dashboard, the model weights
  and a demo video.
- **Removed** upstream material not needed here: most of `docs/`, `tests/`, `docker/`, GitHub
  workflows and `examples/`.
- **Unchanged**: the `ultralytics/` library itself.

## Limitations

- Density uses only the number of visible vehicles, not road area, lanes or speed.
- There is one fixed counting line at mid-frame, and it does not record direction of travel.
- The model is pretrained on COCO and not tuned to local traffic, so unusual vehicles may be
  mislabelled or missed.
- Using every 2nd frame speeds things up but gives the tracker less to work with, which can affect
  counts for fast vehicles.
