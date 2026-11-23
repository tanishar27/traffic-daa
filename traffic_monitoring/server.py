"""Local server: runs YOLO on this machine and serves the JavaScript dashboard.

Run from this folder, then open http://localhost:8000:
    uvicorn server:app --port 8000

Endpoints
    GET  /health          backend status + model settings
    GET  /videos          videos available to process (traffic_monitoring/videos)
    POST /videos?name=    upload a video (raw file bytes as the request body)
    POST /start/{mode}    start a job: mode = detect | track | count, optional ?video=name
    POST /stop            stop the running job
    GET  /stream          MJPEG stream of annotated frames (use as <img src>)
    GET  /stats           live numbers for the running / last job
    GET  /report          last finished traffic report (from "count" mode)
    GET  /                the dashboard (files in ../frontend)
"""

import asyncio
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from vehicle_processing import (
    CLASS_NAMES, CONF_THRESHOLD, DEFAULT_VIDEO, IMAGE_SIZE, VIDEO_DIR, VIDEO_TYPES,
    count_vehicles, detect_vehicles, list_videos, track_vehicles,
)

SOURCES = {"detect": detect_vehicles, "track": track_vehicles, "count": count_vehicles}
JPEG_QUALITY = 75
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Traffic Control Room API")
# Allow any origin so the dashboard also works when opened from another local port or as a file.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Job:
    """The one job this laptop runs at a time, shared by every viewer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.mode = None
        self.video = None
        self.running = False
        self.error = None
        self.frame = None      # latest JPEG bytes
        self.frame_id = 0
        self.frames = 0
        self.started = None
        self.stats = {}
        self.report = None     # last finished "count" report

    def snapshot(self):
        with self.lock:
            elapsed = time.time() - self.started if self.started else 0
            return {
                "mode": self.mode, "video": self.video, "running": self.running, "error": self.error,
                "stopped": self.stop_event.is_set(),
                "frames": self.frames, "fps": round(self.frames / elapsed, 1) if elapsed else 0,
                "stats": dict(self.stats),
            }


job = Job()


def _run(mode, video):
    gen = SOURCES[mode](video)
    try:
        for frame, value in gen:
            if job.stop_event.is_set():
                break
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with job.lock:
                if ok:
                    job.frame = buf.tobytes()
                    job.frame_id += 1
                job.frames += 1
                job.stats = dict(value) if isinstance(value, dict) else {"count": value}
        if mode == "count" and job.stats and not job.stop_event.is_set():
            with job.lock:
                job.report = {**job.stats, "video": video, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:  # surface model/video errors to the frontend
        with job.lock:
            job.error = str(e)
    finally:
        gen.close()
        with job.lock:
            job.running = False


def _stop_current():
    if job.thread and job.thread.is_alive():
        job.stop_event.set()
        job.thread.join(timeout=10)


@app.get("/health")
def health():
    return {"ok": True, "conf": CONF_THRESHOLD, "imgsz": IMAGE_SIZE,
            "classes": list(CLASS_NAMES.values()), "tracker": "ByteTrack"}


@app.get("/videos")
def videos():
    return {"videos": list_videos(), "default": DEFAULT_VIDEO}


@app.post("/videos")
async def upload_video(request: Request, name: str):
    """Save the request body as a new video. The file is streamed to disk, so large videos are fine."""
    path = Path(Path(name).name)
    if path.suffix.lower() not in VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported file type. Use one of: {', '.join(sorted(VIDEO_TYPES))}")
    VIDEO_DIR.mkdir(exist_ok=True)
    target, n = VIDEO_DIR / path.name, 1
    while target.exists():  # never overwrite an existing video
        target, n = VIDEO_DIR / f"{path.stem}-{n}{path.suffix}", n + 1
    size = 0
    with open(target, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    if not size:
        target.unlink()
        raise HTTPException(400, "The uploaded file is empty.")
    cap = cv2.VideoCapture(str(target))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        target.unlink()
        raise HTTPException(400, "OpenCV could not read that file as a video.")
    return {"name": target.name, "videos": list_videos()}


@app.post("/start/{mode}")
def start(mode: str, video: str | None = None):
    if mode not in SOURCES:
        raise HTTPException(404, f"Unknown mode '{mode}'. Use one of: {', '.join(SOURCES)}")
    video = video or DEFAULT_VIDEO
    if video not in list_videos():
        raise HTTPException(404, f"Video '{video}' not found in traffic_monitoring/videos.")
    _stop_current()
    with job.lock:
        job.stop_event = threading.Event()
        job.mode, job.video, job.running, job.error = mode, video, True, None
        job.frame, job.frames, job.stats, job.started = None, 0, {}, time.time()
        job.thread = threading.Thread(target=_run, args=(mode, video), daemon=True)
        job.thread.start()
    return job.snapshot()


@app.post("/stop")
def stop():
    _stop_current()
    return job.snapshot()


@app.get("/stats")
def stats():
    return job.snapshot()


@app.get("/report")
def report():
    return {"report": job.report}


@app.get("/stream")
async def stream():
    async def frames():
        last = -1
        idle = 0
        while True:
            with job.lock:
                fid, data, running = job.frame_id, job.frame, job.running
            if data is not None and fid != last:
                last, idle = fid, 0
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            elif not running:
                idle += 1
                if idle > 20:  # job finished and last frame already sent
                    return
            await asyncio.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


# Mounted last so the API routes above take priority.
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
