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

VEHICLE_CLASSES = ["car", "truck"]

RTSP_USERNAME = os.getenv("RTSP_USERNAME")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_IP = os.getenv("RTSP_IP")
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "101")
RTSP_PORT = os.getenv("RTSP_PORT", "554")

STREAM_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"

os.makedirs("screenshots", exist_ok=True)

window_name = "YOLO Vehicle Auto Screenshot"

cooldown_seconds = 20
last_screenshot_time = 0

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)


def connect_camera():
    print("Connecting to camera...")

    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    if cap.isOpened():
        print("Camera connected.")
    else:
        print("Camera connection failed.")

    return cap


cap = connect_camera()

print("Starting vehicle detection...")
print("Press Q to stop.")

failed_reads = 0
max_failed_reads = 3

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        failed_reads += 1
        print(f"Cannot read video stream. Failed count: {failed_reads}")

        if failed_reads >= max_failed_reads:
            print("Reconnecting camera...")

            cap.release()
            time.sleep(2)

            cap = connect_camera()
            failed_reads = 0

        time.sleep(0.5)
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