from ultralytics import YOLO
import cv2
import os
import time
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

STREAM_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"

os.makedirs("screenshots", exist_ok=True)

window_name = "YOLO Vehicle Auto Screenshot"

cooldown_seconds = 5
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


cap = connect_camera()
if cap is None:
    raise RuntimeError("Unable to connect to RTSP camera.")

print("Starting vehicle detection...")
print("Press Q to stop.")

failed_reads = 0
max_failed_reads = 1

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        failed_reads += 1
        print(f"Cannot read video stream. Failed count: {failed_reads}")

        if failed_reads >= max_failed_reads:
            print("Reconnecting camera...")

            cap.release()
            time.sleep(reconnect_delay_seconds)

            cap = connect_camera()
            if cap is None:
                raise RuntimeError("Unable to reconnect RTSP camera.")
            failed_reads = 0
        else:
            # Short pause to avoid a hot loop while the camera is dropping frames.
            time.sleep(reconnect_delay_seconds)

        continue

    failed_reads = 0

    height, width, _ = frame.shape

    # Trigger area: lower-left / left 66%
    ROI_X1 = 0
    ROI_Y1 = int(height * 0.33)
    ROI_X2 = int(width * 2 / 3)
    ROI_Y2 = height

    results = model(frame, verbose=False)

    vehicle_detected = False
    detected_vehicle = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name in VEHICLE_CLASSES and confidence >= 0.50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {confidence:.2f}"

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                inside_roi = (
                    ROI_X1 <= center_x <= ROI_X2 and
                    ROI_Y1 <= center_y <= ROI_Y2
                )

                # Draw green detection box
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # cv2.putText(
                #     frame,
                #     label,
                #     (x1, y1 - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2
                # )

                # # Draw center point
                # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                if inside_roi:
                    vehicle_detected = True
                    detected_vehicle = class_name

    # Draw trigger area
    # cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

    # cv2.putText(
    #     frame,
    #     "SCREENSHOT TRIGGER AREA",
    #     (ROI_X1 + 20, ROI_Y1 + 40),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2
    # )

    current_time = time.time()

    if vehicle_detected and current_time - last_screenshot_time >= cooldown_seconds:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshots/{detected_vehicle}_{timestamp}.jpg"

        cv2.imwrite(filename, frame)
        print(f"Vehicle detected. Screenshot saved: {filename}")

        last_screenshot_time = current_time

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
