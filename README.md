# Traffic Control Room

Vehicle detection, tracking and traffic analytics with YOLO and ByteTrack, built on top of
[Ultralytics](https://github.com/ultralytics/ultralytics).

- **Detection**: finds cars, motorcycles, buses and trucks in every frame.
- **Tracking**: gives each vehicle an ID with ByteTrack.
- **Analytics**: counts vehicles crossing a line across the middle of the frame, splits the count by type,
  and rates traffic density from the number of visible vehicles (≤ 8 Low, ≤ 20 Medium, more High).
- **Reports**: shows the latest analytics result and downloads it as CSV.

## How it fits together

```
Browser ──► Vercel (frontend/: static HTML/CSS/JS)
   │
   └──► https://<random>.trycloudflare.com ──► Cloudflare tunnel ──► laptop
                                                               FastAPI + YOLO (port 8000)
```

The heavy part (YOLO, PyTorch, OpenCV) runs on the laptop. The website on Vercel is static: it calls
the laptop through a tunnel, shows the annotated video as an MJPEG stream, and polls for live numbers.
The laptop must be running the backend and the tunnel while the site is being used.

| Folder | What's in it |
| --- | --- |
| `traffic_monitoring/server.py` | FastAPI backend (`/health`, `/start/{detect,track,count}`, `/stop`, `/stream`, `/stats`, `/report`) |
| `traffic_monitoring/vehicle_processing.py` | Detection, tracking and line-crossing counting |
| `traffic_monitoring/congestion.py` | Density rating |
| `frontend/` | The dashboard (no build step) |
| `ultralytics/` | Ultralytics YOLO library (upstream, unchanged) |

## 1. Run the backend on the laptop

Put these two files in place (both are gitignored, so they stay on the laptop):

- `yolo26n.pt` in the repo root
- `traffic_monitoring/videos/trafficv.mp4`

Then, from the repo root:

```powershell
pip install -r traffic_monitoring/requirements.txt
cd traffic_monitoring
uvicorn server:app --host 0.0.0.0 --port 8000
```

Check it at <http://localhost:8000/health>.

## 2. Open a tunnel

Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
(`winget install Cloudflare.cloudflared` on Windows), then in a second terminal:

```powershell
cloudflared tunnel --url http://localhost:8000
```

It prints an address like `https://something-random.trycloudflare.com`. No account is needed, but the
address changes every time you start the tunnel.

## 3. Deploy the frontend to Vercel

In Vercel: **Add New → Project**, import this repo, set **Root Directory** to `frontend`,
leave the framework as **Other**, and deploy. There is no build step.

Open the site and paste the tunnel address into **Backend URL** in the sidebar (it is remembered in the
browser). You can also share a link that fills it in: `https://your-site.vercel.app/?api=https://something-random.trycloudflare.com`.

## Running everything locally (no Vercel, no tunnel)

Start the backend as in step 1, then serve the frontend:

```powershell
cd frontend
python -m http.server 5500
```

Open <http://localhost:5500>. The backend URL defaults to `http://localhost:8000`.

## Notes

- One job runs at a time and every viewer sees the same one. Starting a new mode stops the current one.
- Speed depends on the laptop. On CPU, YOLO at image size 960 runs at a few frames per second; lower
  `IMAGE_SIZE` in `vehicle_processing.py` to 640 for a faster demo.
- CORS is open to any origin because this is a demo project.
