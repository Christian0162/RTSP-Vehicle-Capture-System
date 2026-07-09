from ultralytics import YOLO
import cv2
import os
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Force RTSP over TCP for more stable stream
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = ["car", "truck", "motorcycle"]

RTSP_USERNAME = os.getenv("RTSP_USERNAME")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_IP = os.getenv("RTSP_IP")
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "101")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_CODEC = os.getenv("RTSP_CODEC", "h265").lower()
PROCESS_EVERY_N_FRAMES = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "1")))
ROI_START_RATIO = float(os.getenv("ROI_START_RATIO", "0.33"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

STREAM_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"

os.makedirs("screenshots", exist_ok=True)

window_name = "YOLO Vehicle Auto Screenshot"

cooldown_seconds = 3
last_screenshot_time = 0
reconnect_delay_seconds = 0.25
max_reconnect_attempts = 5
PREFERRED_BACKENDS = ("gstreamer", "ffmpeg")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)


def normalize_codec(codec_name):
    if codec_name in {"h264", "264"}:
        return "h264"
    return "h265"


def get_codec_attempts():
    preferred = normalize_codec(RTSP_CODEC)
    fallback = "h264" if preferred == "h265" else "h265"
    return (preferred, fallback)


def build_gstreamer_pipeline(codec_name):
    # Low-latency pipeline that drops old frames and keeps only the freshest frame
    # available for YOLO. The codec-specific decoder must match the camera stream.
    codec_name = normalize_codec(codec_name)
    if codec_name == "h264":
        depay = "rtph264depay"
        parser = "h264parse"
        decoder = "avdec_h264"
    else:
        depay = "rtph265depay"
        parser = "h265parse"
        decoder = "avdec_h265"

    return (
        f'rtspsrc location="{STREAM_URL}" protocols=tcp latency=100 drop-on-latency=true '
        f"! {depay} ! {parser} ! {decoder} "
        f"! videoconvert ! appsink sync=false drop=true max-buffers=1"
    )


def open_capture(backend_name, codec_name):
    if backend_name == "gstreamer":
        return cv2.VideoCapture(build_gstreamer_pipeline(codec_name), cv2.CAP_GSTREAMER)

    return cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)


def connect_camera():
    print("Connecting to camera...")
    print(f"Using codec hint: {normalize_codec(RTSP_CODEC)}")

    timeout_open_ms = 3000
    timeout_read_ms = 3000

    for codec_name in get_codec_attempts():
        for backend_name in PREFERRED_BACKENDS:
            for attempt in range(1, max_reconnect_attempts + 1):
                cap = open_capture(backend_name, codec_name)

                # Keep the buffer small so stale frames do not delay reconnect handling.
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Ask OpenCV/FFmpeg to fail fast when the stream is unavailable.
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_open_ms)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_read_ms)

                if cap.isOpened():
                    print(
                        f"Camera connected with {backend_name} using {codec_name} on attempt {attempt}."
                    )
                    return cap

                cap.release()
                print(
                    f"Camera connection failed with {backend_name} using {codec_name} "
                    f"on attempt {attempt}. Retrying..."
                )
                time.sleep(reconnect_delay_seconds)

    print("Camera connection failed after multiple attempts.")
    return None


class FrameGrabber:
    def __init__(self):
        self._frame = None
        self._frame_time = 0.0
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame.copy(), self._frame_time, self._frame_id

    def _store_frame(self, frame):
        with self._lock:
            self._frame = frame
            self._frame_time = time.time()
            self._frame_id += 1
        self._ready_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            cap = connect_camera()
            if cap is None:
                time.sleep(reconnect_delay_seconds)
                continue

            while not self._stop_event.is_set():
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Frame read failed. Reconnecting camera...")
                    break

                self._store_frame(frame)

            cap.release()
            time.sleep(reconnect_delay_seconds)


grabber = FrameGrabber()
grabber.start()

if not grabber._ready_event.wait(timeout=15):
    grabber.stop()
    raise RuntimeError("Unable to connect to RTSP camera.")

print("Starting vehicle detection...")
print("Press Q to stop.")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s).")

last_processed_frame_id = 0

try:
    while True:
        frame, frame_time, frame_id = grabber.read()

        if frame is None:
            time.sleep(0.01)
            continue

        height, width, _ = frame.shape

        # Trigger area: lower-left / left 66%
        ROI_X1 = 0
        ROI_Y1 = int(height * ROI_START_RATIO)
        ROI_X2 = int(width * 2 / 3)
        ROI_Y2 = height

        vehicle_detected = False
        detected_vehicle = None

        should_process = (
            frame_id != last_processed_frame_id and
            frame_id % PROCESS_EVERY_N_FRAMES == 0
        )

        if should_process:
            results = model(frame, verbose=False)
            last_processed_frame_id = frame_id

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    if class_name in VEHICLE_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        inside_roi = (
                            ROI_X1 <= center_x <= ROI_X2 and
                            ROI_Y1 <= center_y <= ROI_Y2
                        )

                        if inside_roi:
                            vehicle_detected = True
                            detected_vehicle = class_name
                            break

                if vehicle_detected:
                    break

        current_time = time.time()

        if vehicle_detected and current_time - last_screenshot_time >= cooldown_seconds:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshots/{detected_vehicle}_{timestamp}.jpg"

            cv2.imwrite(filename, frame)
            print(f"Vehicle detected. Screenshot saved: {filename}")

            last_screenshot_time = current_time

        age_seconds = current_time - frame_time
        if age_seconds > 2:
            cv2.putText(
                frame,
                f"Stream delay: {age_seconds:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    grabber.stop()
    cv2.destroyAllWindows()
