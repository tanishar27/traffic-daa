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
   COCO classes are kept: car, motorcycle, bus and truck. Confidence threshold is 0.20 so small and
   distant vehicles are still found.
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
| `traffic_monitoring/vehicle_processing.py` | Detection, tracking and line-crossing counting. Shared by both apps below |
| `traffic_monitoring/congestion.py` | Turns the visible vehicle count into Low / Medium / High |
| `traffic_monitoring/app.py` | **Option A**: Streamlit dashboard (runs locally or on Streamlit Cloud) |
| `traffic_monitoring/server.py` | **Option B**: FastAPI backend that streams results to the web frontend |
| `frontend/` | **Option B**: static web dashboard (HTML/CSS/JS, no build step) for Vercel |
| `traffic_monitoring/requirements.txt` | Python packages for the backend and Streamlit Cloud |
| `traffic_monitoring/videos/trafficv.mp4` | Demo traffic video |
| `yolo26n.pt` | YOLO26 nano model weights |
| `ultralytics/` | Ultralytics YOLO library (upstream, unchanged) |

The model and demo video are included in the repo, so it runs straight after cloning.

## Option A: Streamlit dashboard

One Python app, easiest to run.

**Locally**, from the repo root:

```powershell
pip install -r traffic_monitoring/requirements.txt streamlit
cd traffic_monitoring
streamlit run app.py
```

It opens at <http://localhost:8501>.

**On Streamlit Cloud**: create an app from this repo with main file `traffic_monitoring/app.py`.
Streamlit Cloud installs `streamlit` itself and the rest from `traffic_monitoring/requirements.txt`.
It runs on a shared CPU, so playback is slow.

## Option B: Vercel frontend + backend on a laptop

The website is hosted on Vercel; the YOLO processing runs on your own laptop and is reached through a
tunnel. Use this when you want a public link but the model running on your own hardware.

```
Browser ──► Vercel (frontend/)
   │
   └──► https://<random>.trycloudflare.com ──► Cloudflare tunnel ──► laptop: FastAPI + YOLO (port 8000)
```

**1. Start the backend on the laptop**, from the repo root:

```powershell
pip install -r traffic_monitoring/requirements.txt
cd traffic_monitoring
uvicorn server:app --host 0.0.0.0 --port 8000
```

Check it at <http://localhost:8000/health>.

**2. Open a tunnel** in a second terminal. Install
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
first (`winget install Cloudflare.cloudflared` on Windows):

```powershell
cloudflared tunnel --url http://localhost:8000
```

It prints an address like `https://something-random.trycloudflare.com`. No account is needed, but the
address changes every time you start the tunnel.

**3. Deploy the frontend to Vercel.** Add New → Project, import this repo, set **Root Directory** to
`frontend`, leave the framework as **Other**, and deploy.

Open the site and paste the tunnel address into **Backend URL** in the sidebar (the browser remembers
it). You can also share a link that fills it in:
`https://your-site.vercel.app/?api=https://something-random.trycloudflare.com`.

The laptop must keep the backend and the tunnel running while people use the site; otherwise the page
shows "Backend offline".

**Without Vercel**, run the backend as in step 1, then serve the frontend locally and open
<http://localhost:5500>:

```powershell
cd frontend
python -m http.server 5500
```

### Backend API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend status and model settings |
| `POST /start/{detect\|track\|count}` | Start a run (stops any run already going) |
| `POST /stop` | Stop the current run |
| `GET /stream` | Live annotated video as an MJPEG stream (used as an `<img>` source) |
| `GET /stats` | Live numbers for the current or last run |
| `GET /report` | Last finished traffic report from a `count` run |

One run happens at a time and every viewer sees the same one. CORS is open to any origin because this
is a demo project.

## What this fork changed from Ultralytics

- **Added** the `traffic_monitoring/` module, the `frontend/` web dashboard, the model weights and a
  demo video.
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
