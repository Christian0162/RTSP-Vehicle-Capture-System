from pathlib import Path
import os
import time

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO


load_dotenv()

# Force RTSP over TCP for a more stable camera stream.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

MODEL_PATH = Path("runs/detect/train-2/weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.10"))
PROCESS_EVERY_N_FRAMES = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "1")))

RTSP_USERNAME = os.getenv("RTSP_USERNAME")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_IP = os.getenv("RTSP_IP")
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "101")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_CODEC = os.getenv("RTSP_CODEC", "h265").lower()

STREAM_URL = (
    f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}"
    f"/Streaming/Channels/{RTSP_CHANNEL}"
)

WINDOW_NAME = "RTSP Object Detection"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
RECONNECT_DELAY_SECONDS = 0.25
MAX_RECONNECT_ATTEMPTS = 5
PREFERRED_BACKENDS = ("gstreamer", "ffmpeg")


def normalize_codec(codec_name):
    if codec_name in {"h264", "264"}:
        return "h264"
    return "h265"


def get_codec_attempts():
    preferred = normalize_codec(RTSP_CODEC)
    fallback = "h264" if preferred == "h265" else "h265"
    return (preferred, fallback)


def build_gstreamer_pipeline(codec_name):
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
            for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                cap = open_capture(backend_name, codec_name)

                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_open_ms)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_read_ms)

                if cap.isOpened():
                    print(
                        f"Camera connected with {backend_name} using {codec_name} "
                        f"on attempt {attempt}."
                    )
                    return cap

                cap.release()
                print(
                    f"Camera connection failed with {backend_name} using {codec_name} "
                    f"on attempt {attempt}. Retrying..."
                )
                time.sleep(RECONNECT_DELAY_SECONDS)

    return None


def fit_frame_to_window(frame):
    frame_height, frame_width = frame.shape[:2]
    scale = min(DISPLAY_WIDTH / frame_width, DISPLAY_HEIGHT / frame_height)
    scaled_width = int(frame_width * scale)
    scaled_height = int(frame_height * scale)

    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    top = (DISPLAY_HEIGHT - scaled_height) // 2
    bottom = DISPLAY_HEIGHT - scaled_height - top
    left = (DISPLAY_WIDTH - scaled_width) // 2
    right = DISPLAY_WIDTH - scaled_width - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def print_detected_objects(result, model):
    if len(result.boxes) == 0:
        print("Detected objects: none")
        return

    names = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        names.append(f"{class_name} ({confidence:.0%})")

    print("Detected objects: " + ", ".join(names))


def validate_settings():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    missing = [
        name
        for name, value in {
            "RTSP_USERNAME": RTSP_USERNAME,
            "RTSP_PASSWORD": RTSP_PASSWORD,
            "RTSP_IP": RTSP_IP,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(f"Missing required .env value(s): {', '.join(missing)}")


def main():
    validate_settings()

    model = YOLO(str(MODEL_PATH))

    print("Available classes:")
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    cap = connect_camera()
    if cap is None:
        raise RuntimeError("Unable to connect to RTSP camera.")

    print("Starting live detection...")
    print("Press Q to stop.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Frame read failed. Reconnecting camera...")
                cap.release()
                cap = connect_camera()
                if cap is None:
                    time.sleep(RECONNECT_DELAY_SECONDS)
                continue

            frame_count += 1
            display_frame = frame

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
                result = results[0]
                display_frame = result.plot()
                print_detected_objects(result, model)

            cv2.imshow(WINDOW_NAME, fit_frame_to_window(display_frame))

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
