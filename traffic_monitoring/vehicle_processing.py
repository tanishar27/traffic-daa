"""Vehicle detection, tracking and counting in one module.

app.py imports:
    from vehicle_processing import detect_vehicles, track_vehicles, count_vehicles
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

from congestion import classify_density


# ---------------- SETTINGS ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolo26n.pt"
VIDEO_PATH = BASE_DIR / "videos" / "trafficv.mp4"

VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

CONF_THRESHOLD = 0.20   # lower = detects more (small/distant) vehicles
IMAGE_SIZE = 960        # larger input = better for small objects


# ---------------- SHARED HELPER ----------------

def _open_video_and_model():
    """Validate paths, then return (model, cap)."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open traffic video: {VIDEO_PATH}")

    return model, cap


# ---------------- 1. VEHICLE DETECTION ----------------

def detect_vehicles():
    """Yield (BGR frame, vehicle_count) for each video frame."""

    model, cap = _open_video_and_model()

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model(
                frame,
                classes=VEHICLE_CLASSES,
                conf=CONF_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            vehicle_count = 0

            for box in results[0].boxes:

                class_id = int(box.cls[0])
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    CLASS_NAMES[class_id],
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Vehicles: {vehicle_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Frame stays BGR: app.py uses channels="BGR"
            yield frame, vehicle_count

    finally:
        cap.release()


# ---------------- 2. VEHICLE TRACKING ----------------

def track_vehicles():
    """Yield (annotated BGR frame, tracked_count) for each video frame."""

    model, cap = _open_video_and_model()

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=CONF_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()   # BGR

            tracked_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                tracked_count = len(result.boxes.id)

            yield annotated_frame, tracked_count

    finally:
        cap.release()


# ---------------- 3. VEHICLE COUNTING ----------------

def count_vehicles():
    """Yield (BGR frame, stats dict) while counting line crossings."""

    model, cap = _open_video_and_model()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Counting line at the middle of the video
    line_y = int(height * 0.50)

    previous_positions = {}
    counted_ids = set()

    total_count = 0
    vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=CONF_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False
            )

            result = results[0]
            visible_count = 0

            # Draw counting line
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)

            cv2.putText(
                frame,
                "COUNTING LINE",
                (20, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            if result.boxes is not None and len(result.boxes) > 0:

                visible_count = len(result.boxes)

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.int().cpu().tolist()

                # IDs are None on frames where the tracker has no tracks yet
                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                else:
                    ids = [None] * len(classes)

                for box, track_id, class_id in zip(boxes, ids, classes):

                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    vehicle_name = CLASS_NAMES[class_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                    label = (
                        f"{vehicle_name} ID:{track_id}"
                        if track_id is not None
                        else vehicle_name
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    # Line-crossing check only for tracked vehicles
                    if track_id is None:
                        continue

                    if track_id in previous_positions:

                        previous_y = previous_positions[track_id]

                        crossed = (
                            previous_y < line_y <= center_y
                            or
                            previous_y > line_y >= center_y
                        )

                        if crossed and track_id not in counted_ids:

                            total_count += 1
                            counted_ids.add(track_id)
                            vehicle_counts[vehicle_name] += 1

                    previous_positions[track_id] = center_y

            # Traffic density
            density = classify_density(visible_count)

            density_color = {
                "Low": (0, 255, 0),
                "Medium": (0, 165, 255),
                "High": (0, 0, 255)
            }.get(density, (255, 255, 255))

            cv2.putText(
                frame,
                f"Total Count: {total_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Visible Vehicles: {visible_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Traffic Density: {density}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                density_color,
                2
            )

            stats = {
                "total": total_count,
                "visible": visible_count,
                "density": density,
                **vehicle_counts
            }

            yield frame, stats

    finally:
        cap.release()