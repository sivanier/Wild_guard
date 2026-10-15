"""
Animal Intrusion Detection & Crop Protection System
Phase 1 + Phase 2 combined Flask app.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000/  on the detection laptop
and  http://<laptop-local-ip>:5000/farmer  on the farmer's phone/other laptop
(both devices must be on the SAME Wi-Fi network).
"""

import os
import sqlite3
import time
import base64
import json
import uuid
import threading
from datetime import datetime
from collections import Counter
from flask_socketio import SocketIO

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, Response, send_from_directory
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "detections.db")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt"))
FALLBACK_MODEL = "yolov8n.pt"  # generic COCO-pretrained model, used only if best.pt is missing

# NOTE: COCO's pretrained classes include "elephant" but NOT "wild boar" or "monkey".
# This fallback is only here so the app runs out of the box for testing.
# Replace models/best.pt with YOUR custom-trained YOLOv8 weights for real results.

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key-before-any-real-use")
socketio = SocketIO(app, cors_allowed_origins=[], async_mode="threading")

# Hardcoded demo farmer credentials -- change before using for anything real.
FARMER_USERNAME = "farmer"
FARMER_PASSWORD = "farmer123"

# --------------------------------------------------------------------------
# Load YOLO model (once, at startup)
# --------------------------------------------------------------------------
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Loaded custom model: {MODEL_PATH}")
else:
    model = YOLO(FALLBACK_MODEL)
    print(f"[WARNING] No custom model found at {MODEL_PATH}.")
    print(f"[WARNING] Falling back to '{FALLBACK_MODEL}' (COCO classes only, "
          f"no wild boar / monkey). Replace with your trained weights.")

# --------------------------------------------------------------------------
# Crop damage "Linear Regression" model
# --------------------------------------------------------------------------
# Honesty note (read this): we do NOT have real, measured crop-damage data.
# Farmers don't hand out labeled ₹-loss-per-animal-visit datasets.
# So this trains a real sklearn LinearRegression model, but on a small
# SYNTHETIC dataset built from a simple assumed severity table below.
# Treat its output as an illustrative estimate, not a validated prediction,
# and say so clearly in your report / viva.

ANIMAL_SEVERITY = {
    "elephant": 500.0,
    "wild boar": 150.0,
    "monkey": 20.0,
    "deer": 80.0,
}
DEFAULT_SEVERITY = 50.0


def _severity(animal_name: str) -> float:
    return ANIMAL_SEVERITY.get(animal_name.lower(), DEFAULT_SEVERITY)


def train_damage_model():
    rng = np.random.default_rng(42)
    X, y = [], []
    animals = list(ANIMAL_SEVERITY.keys()) + ["other"]
    for animal in animals:
        base = _severity(animal if animal != "other" else "unknown")
        for _ in range(40):
            confidence = rng.uniform(0.4, 0.99)
            noise = rng.normal(0, base * 0.15)
            damage = max(0.0, base * confidence + noise)
            X.append([base, confidence])
            y.append(damage)
    X = np.array(X)
    y = np.array(y)
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


damage_model = train_damage_model()


def estimate_damage(animal_name: str, confidence: float) -> float:
    base = _severity(animal_name)
    pred = damage_model.predict([[base, confidence]])[0]
    return round(max(0.0, float(pred)), 2)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal TEXT NOT NULL,
            confidence REAL NOT NULL,
            est_damage REAL NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_group ON detections(animal, source, timestamp)")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
    if "bbox" not in columns:
        conn.execute("ALTER TABLE detections ADD COLUMN bbox TEXT")
    if "group_id" not in columns:
        conn.execute("ALTER TABLE detections ADD COLUMN group_id TEXT")
    conn.commit()
    conn.close()


DETECTION_COOLDOWN_SECONDS = 8


def log_detection(animal, confidence, source, bbox=None, group_id=None):
    """Persist and broadcast a detection, grouping repeated video/webcam frames."""
    damage = estimate_damage(animal, confidence)
    timestamp = datetime.now().isoformat(timespec="seconds")
    group_id = group_id or source
    conn = get_db()
    if source in ("video_upload", "webcam", "webcam_live"):
        recent = conn.execute(
            "SELECT timestamp FROM detections WHERE animal = ? AND group_id = ? ORDER BY id DESC LIMIT 1",
            (animal, group_id),
        ).fetchone()
        if recent:
            try:
                elapsed = (datetime.fromisoformat(timestamp) - datetime.fromisoformat(recent["timestamp"])).total_seconds()
                if 0 <= elapsed < DETECTION_COOLDOWN_SECONDS:
                    conn.close()
                    return None
            except ValueError:
                pass
    conn.execute(
        "INSERT INTO detections (animal, confidence, est_damage, source, timestamp, bbox, group_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (animal, confidence, damage, source, timestamp, json.dumps(bbox) if bbox else None, group_id),
    )
    event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    recent_count = conn.execute(
        "SELECT COUNT(*) FROM (SELECT id FROM detections ORDER BY timestamp DESC, id DESC LIMIT 20)"
    ).fetchone()[0]
    conn.commit()
    conn.close()
    event = {
        "id": event_id, "animal": animal, "confidence": round(float(confidence), 3),
        "est_damage": damage, "source": source, "timestamp": timestamp, "bbox": bbox,
    }
    event["risk_severity"] = compute_risk_level(recent_count).lower()
    for namespace in ("/field", "/farmer"):
        socketio.emit("detection", event, namespace=namespace)
    return damage


init_db()

# --------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------
def run_detection_on_image(img_bgr, source_label="image", group_id=None):
    """Runs YOLO on a single BGR image (numpy array). Returns annotated image + list of detections."""
    results = model(img_bgr, verbose=False)[0]
    annotated = results.plot()  # BGR numpy array with boxes drawn

    detections = []
    names = results.names
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            animal = names.get(cls_id, str(cls_id))
            bbox = [round(float(value), 1) for value in box.xyxy[0].tolist()]
            detections.append({"animal": animal, "confidence": round(conf, 3), "bbox": bbox})
            log_detection(animal, conf, source_label, bbox, group_id)

    return annotated, detections


def decode_base64_image(data_url: str):
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image_base64(img_bgr) -> str:
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def login_required(view):
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    wrapped.__name__ = view.__name__
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == FARMER_USERNAME and p == FARMER_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("farmer"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Main detection page (runs on the laptop watching the field)
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect_image", methods=["POST"])
def api_detect_image():
    """Handles a single uploaded image OR a base64 webcam frame."""
    if "image" in request.files:
        file = request.files["image"]
        arr = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        source = "image_upload"
    else:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("frame")
        if not data_url:
            return jsonify({"error": "no image provided"}), 400
        img = decode_base64_image(data_url)
        source = "webcam"

    if img is None:
        return jsonify({"error": "could not decode image"}), 400

    annotated, detections = run_detection_on_image(img, source_label=source)
    return jsonify({
        "image": encode_image_base64(annotated),
        "detections": detections,
    })


@app.route("/api/detect_video", methods=["POST"])
def api_detect_video():
    """Handles an uploaded video file. Samples frames, runs detection, writes an annotated output video."""
    if "video" not in request.files:
        return jsonify({"error": "no video provided"}), 400

    file = request.files["video"]
    filename = secure_filename(file.filename)
    in_path = os.path.join(UPLOAD_DIR, filename)
    file.save(in_path)

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_name = f"annotated_{filename}"
    if not out_name.lower().endswith(".mp4"):
        out_name += ".mp4"
    out_path = os.path.join(UPLOAD_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Sample every Nth frame to keep processing time reasonable for a demo.
    sample_every = max(1, total_frames // 150) if total_frames > 0 else 5
    video_group_id = f"video:{uuid.uuid4().hex}"

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            annotated, detections = run_detection_on_image(frame, source_label="video_upload", group_id=video_group_id)
            all_detections.extend(detections)
            writer.write(annotated)
        else:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    counts = Counter(d["animal"] for d in all_detections)
    return jsonify({
        "output_video": f"/uploads/{out_name}",
        "total_detections": len(all_detections),
        "counts": dict(counts),
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# --------------------------------------------------------------------------
# Live webcam stream (MJPEG) -- server-side webcam (the detection laptop's own camera)
# --------------------------------------------------------------------------
_stream_lock = threading.Lock()
_stream_active = {"on": False}


def gen_frames():
    cap = cv2.VideoCapture(0)
    try:
        while _stream_active["on"]:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = run_detection_on_image(frame, source_label="webcam_live")
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/webcam/<action>", methods=["POST"])
def webcam_control(action):
    with _stream_lock:
        _stream_active["on"] = (action == "start")
    return jsonify({"active": _stream_active["on"]})


# --------------------------------------------------------------------------
# Farmer dashboard (Phase 2) -- viewable from another device on the same Wi-Fi
# --------------------------------------------------------------------------
@app.route("/farmer")
@login_required
def farmer():
    return render_template("farmer.html")


@socketio.on("connect", namespace="/farmer")
def farmer_socket_connect(auth=None):
    if not session.get("logged_in"):
        return False


@socketio.on("connect", namespace="/field")
def field_socket_connect(auth=None):
    # The local detection station is intentionally available without farmer login.
    return True


def compute_risk_level(recent_count: int) -> str:
    if recent_count >= 5:
        return "High"
    if recent_count >= 2:
        return "Medium"
    return "Low"


@app.route("/api/stats")
@login_required
def api_stats():
    conn = get_db()
    rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC").fetchall()
    conn.close()

    total = len(rows)
    animal_counts = Counter(r["animal"] for r in rows)
    hour_counts = Counter(datetime.fromisoformat(r["timestamp"]).hour for r in rows)
    total_damage = sum(r["est_damage"] for r in rows)

    most_active_animal = animal_counts.most_common(1)[0][0] if animal_counts else None
    most_active_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

    # "recent" = last 20 logged events, used only as a simple demo proxy for risk level
    # (a real system would use a real time window, e.g. last 24 hours)
    recent = rows[:20]
    risk = compute_risk_level(len(recent))

    return jsonify({
        "total_detections": total,
        "animal_counts": dict(animal_counts),
        "most_active_animal": most_active_animal,
        "most_active_hour": most_active_hour,
        "estimated_total_damage": round(total_damage, 2),
        "risk_level": risk,
    })


@app.route("/api/detections")
@login_required
def api_detections():
    limit = int(request.args.get("limit", 25))
    conn = get_db()
    rows = conn.execute(
        "SELECT id, animal, confidence, est_damage, source, timestamp, bbox FROM detections "
        "ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return jsonify([{**dict(r), "bbox": json.loads(r["bbox"]) if r["bbox"] else None} for r in rows])


if __name__ == "__main__":
    # Bind to the LAN by default; override with HOST/PORT for local-only use.
    socketio.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                 port=int(os.environ.get("PORT", "5000")), debug=False, allow_unsafe_werkzeug=True)
