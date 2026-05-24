from flask import Flask, request, jsonify, send_from_directory
import cv2
import os
import base64
import threading
import uuid

from multi import MultiTrackerRuntime, TEMPLATE_DIR

app = Flask(__name__, static_folder="static", template_folder="static")

UPLOAD_FOLDER = "uploads"
DEFAULT_VIDEO = os.path.join("videos", "vid.mp4")
PREVIEW_EVERY_N_FRAMES = 5
JPEG_QUALITY = 45

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

runtime = None
runtime_lock = threading.Lock()


def get_runtime():
    global runtime
    with runtime_lock:
        if runtime is None:
            runtime = MultiTrackerRuntime(template_dir=TEMPLATE_DIR)
    return runtime


processing_state = {
    "status": "idle",
    "progress": 0,
    "current_frame_b64": None,
    "fps": 0.0,
    "message": "",
}
state_lock = threading.Lock()

current_worker = None
worker_lock = threading.Lock()


def update_state(**kwargs):
    with state_lock:
        processing_state.update(kwargs)


def reset_state(message=""):
    update_state(
        status="processing",
        progress=0,
        current_frame_b64=None,
        fps=0.0,
        message=message,
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/templates")
def api_templates():
    try:
        rt = get_runtime()
        return jsonify({"count": len(rt.templates)})
    except Exception as e:
        return jsonify({"count": 0, "error": str(e)})


@app.route("/api/status")
def get_status():
    with state_lock:
        data = {k: v for k, v in processing_state.items() if k != "current_frame_b64"}
    return jsonify(data)


@app.route("/api/current_frame")
def get_current_frame():
    with state_lock:
        frame_b64 = processing_state.get("current_frame_b64")
    return jsonify({"frame": frame_b64})


@app.route("/api/upload_and_process", methods=["POST"])
def upload_and_process():
    global current_worker

    if "video" not in request.files:
        return jsonify({"success": False, "error": "Видео файл байхгүй"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"success": False, "error": "Файл сонгоогүй"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return jsonify({"success": False, "error": "Дэмжигдэхгүй файл"}), 400

    with worker_lock:
        if current_worker is not None and current_worker.is_alive():
            return (
                jsonify(
                    {"success": False, "error": "Одоогоор өөр видео боловсруулж байна"}
                ),
                409,
            )

        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        f.save(filepath)

        reset_state("")

        current_worker = threading.Thread(
            target=process_video_thread, args=(filepath, False), daemon=True
        )
        current_worker.start()

    return jsonify({"success": True, "message": "Боловсруулалт эхэллээ"})


def process_video_thread(filepath, is_default_video=False):
    global current_worker

    try:
        rt = MultiTrackerRuntime(template_dir=TEMPLATE_DIR)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            update_state(status="error", message="Видео нээгдсэнгүй")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1

        frame_idx = 0

        update_state(message="")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = rt.process_frame(frame)
            display = result.get("frame", frame)
            fps = float(result.get("fps", 0.0))

            progress = min(99, int(((frame_idx + 1) / total) * 100))

            frame_b64 = None
            if frame_idx % PREVIEW_EVERY_N_FRAMES == 0:
                ok, buf = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(
                        buf
                    ).decode("utf-8")

            if frame_b64 is not None:
                update_state(
                    status="processing",
                    progress=progress,
                    current_frame_b64=frame_b64,
                    fps=round(fps, 1),
                    message="",
                )
            else:
                update_state(
                    status="processing",
                    progress=progress,
                    fps=round(fps, 1),
                    message="",
                )

            frame_idx += 1

        cap.release()

        if not is_default_video:
            try:
                os.remove(filepath)
            except Exception:
                pass

        update_state(status="done", progress=100, message="")

    except Exception as e:
        update_state(status="error", message=f"Алдаа: {str(e)}")

    finally:
        with worker_lock:
            current_worker = None


def start_default_video_if_exists():
    global current_worker

    if not os.path.exists(DEFAULT_VIDEO):
        update_state(status="idle", message="")
        return

    with worker_lock:
        if current_worker is not None and current_worker.is_alive():
            return

        reset_state("")

        current_worker = threading.Thread(
            target=process_video_thread, args=(DEFAULT_VIDEO, True), daemon=True
        )
        current_worker.start()


if __name__ == "__main__":
    print("=" * 60)
    print("  Замын тэмдэг таних систем")
    print("=" * 60)

    try:
        rt = get_runtime()
        print(f"  ✓ Templates ачаалсан: {len(rt.templates)} ширхэг")
    except Exception as e:
        print(f"  ⚠ Template ачаалах алдаа: {e}")

    if os.path.exists(DEFAULT_VIDEO):
        print(f"  ✓ Default video: {DEFAULT_VIDEO}")
    else:
        print(f"  ⚠ Default video олдсонгүй: {DEFAULT_VIDEO}")

    start_default_video_if_exists()

    print("=" * 60)
    print("  Браузерт нээх: http://localhost:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
