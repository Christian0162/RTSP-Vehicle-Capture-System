# RTSP Vehicle Capture System

This project connects to an RTSP camera feed, runs YOLOv8 detection on each frame, and saves a screenshot when a vehicle is detected inside the trigger area.

## Features

- RTSP camera capture
- YOLOv8 vehicle detection
- Screenshot capture on trigger
- Cooldown to prevent repeated screenshots
- Reconnect handling for dropped streams
- GStreamer-first capture attempt with FFmpeg fallback

## How It Works

1. The app connects to the RTSP stream from your camera or NVR.
2. Each frame is passed into YOLOv8.
3. If a detected vehicle is inside the region of interest, the frame is saved as a screenshot.
4. If the stream drops, the app tries to reconnect.

## Requirements

Install the Python packages listed in `requirements.txt`.

You will also need:

- Python 3.10 or newer
- A working RTSP camera or NVR stream
- `yolov8n.pt` available in the project folder or downloadable by Ultralytics

If you want to use the GStreamer backend, you also need GStreamer installed on your system.

## Setup

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your camera credentials.

## `.env` Example

```env
RTSP_USERNAME=admin
RTSP_PASSWORD=your_password
RTSP_IP=192.168.1.100
RTSP_CHANNEL=1
RTSP_PORT=554
RTSP_CODEC=h265
```

### Environment Variables

- `RTSP_USERNAME` - camera username
- `RTSP_PASSWORD` - camera password
- `RTSP_IP` - camera or NVR IP address
- `RTSP_CHANNEL` - RTSP channel number, such as `1`
- `RTSP_PORT` - usually `554`
- `RTSP_CODEC` - preferred codec for the stream, `h265` by default, `h264` if your stream works better with H.264

## Run

```bash
python main.py
```

Press `Q` to quit the application.

## Screenshot Logic

Screenshots are saved only when:

- YOLO detects a vehicle class
- the detection is inside the configured region of interest
- the cooldown period has expired

The current trigger classes are:

- `car`
- `truck`
- `motorcycle`

## Notes About Vehicle Labels

YOLO does not usually label `SUV` or `sedan` separately.

Those are typically detected as:

- `car`

So if your camera sees an SUV or sedan, the app will usually treat it as a car and save a screenshot if it is in the trigger area.

## Output

Saved screenshots are written to the `screenshots/` folder.

## Troubleshooting

### Stream keeps disconnecting

- Use a wired connection instead of Wi-Fi
- Check camera/NVR power and network stability
- Try the main stream if the substream is unstable
- Lower bitrate, FPS, or resolution
- Switch between `h265` and `h264` in `.env`

### Decoder errors in the terminal

If you see errors like:

- `Could not find ref with POC`
- `cu_qp_delta ... outside the valid range`

that usually means the RTSP stream or codec is unstable, especially with H.265.

Try:

- setting `RTSP_CODEC=h264`
- using a cleaner camera stream
- lowering bitrate or FPS

### GStreamer is not working

If your OpenCV build does not support GStreamer, the app will fall back to FFmpeg.

## Project Files

- `main.py` - main application
- `requirements.txt` - Python dependencies
- `.env` - camera configuration
- `screenshots/` - captured images