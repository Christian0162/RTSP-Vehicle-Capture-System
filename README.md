# RTSP Vehicle Capture System

This project connects to an RTSP camera feed, runs a YOLO model on each frame, and saves screenshots when the trained object is detected inside the region of interest.

The project also includes simple training and testing scripts so you can train your own YOLO model before using it with the live RTSP capture app.

## Features

- RTSP camera capture
- YOLO detection with Ultralytics
- Custom model training with `train.py`
- Image prediction test with `test.py`
- Screenshot capture on detection
- Cooldown to prevent repeated screenshots
- Reconnect handling for dropped streams
- GStreamer-first capture attempt with FFmpeg fallback

## Requirements

Install the Python packages listed in `requirements.txt`.

You will also need:

- Python 3.10 or newer
- A working RTSP camera or NVR stream
- A YOLO dataset with a `data.yaml` file
- `yolo11n.pt` in the project folder, or downloadable by Ultralytics

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
PROCESS_EVERY_N_FRAMES=1
ROI_START_RATIO=0.33
CONFIDENCE_THRESHOLD=0.50
```

### Environment Variables

- `RTSP_USERNAME` - camera username
- `RTSP_PASSWORD` - camera password
- `RTSP_IP` - camera or NVR IP address
- `RTSP_CHANNEL` - RTSP channel number, such as `1`
- `RTSP_PORT` - usually `554`
- `RTSP_CODEC` - preferred codec for the stream, `h265` by default, `h264` if your stream works better with H.264
- `PROCESS_EVERY_N_FRAMES` - process every Nth frame to reduce load, `1` means every frame
- `ROI_START_RATIO` - top boundary of the detection area as a fraction of frame height
- `CONFIDENCE_THRESHOLD` - minimum confidence needed before a detection counts

## Training A Model

Training is handled by `train.py`.

Open `train.py` and update the dataset path:

```python
data=r"C:\Users\Big Blue Logistics\Desktop\New folder\data.yaml"
```

That file should point to your YOLO dataset configuration. The dataset should already be labeled and split into training and validation images.

Then run:

```bash
python train.py
```

The current training script starts from `yolo11n.pt`:

```python
model = YOLO("yolo11n.pt")
```

It trains for 100 epochs at image size 640:

```python
model.train(
    data=r"C:\Users\Big Blue Logistics\Desktop\New folder\data.yaml",
    epochs=100,
    imgsz=640,
)
```

When training finishes, Ultralytics saves the trained model here:

```text
runs/detect/train/weights/best.pt
```

It also saves the last checkpoint here:

```text
runs/detect/train/weights/last.pt
```

## Resuming Training

If training stops before it finishes, use the resume block in `train.py`.

Comment out the normal training section and uncomment:

```python
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

Then run:

```bash
python train.py
```

## Testing The Trained Model

After training, run `test.py` to check the trained weights against a test image:

```bash
python test.py
```

The test script loads:

```text
runs/detect/train/weights/best.pt
```

and runs prediction on:

```text
test1.jpg
```

It prints the number of detected boxes and each detection name with its confidence. Ultralytics also saves the predicted image output under the `runs/` folder.

To test a different image, edit this line in `test.py`:

```python
source="test1.jpg"
```

## Using The Trained Model In The RTSP App

The live capture app is `main.py`.

Run it with:

```bash
python main.py
```

Press `Q` to quit the application.

After training your custom model, update the YOLO model path in `main.py` so it uses your trained weights:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

Then make sure the detection name in `main.py` matches the class name from your dataset:

```python
CLASSES = ["box"]
```

Use the class name that appears in your training dataset and in the `test.py` output.

## Output

Saved screenshots are written to the `screenshots/` folder.

Training and prediction results are written to the `runs/` folder.

## Troubleshooting

### Training cannot find the dataset

- Check that the `data.yaml` path in `train.py` is correct
- Make sure the path points to the YAML file, not just the dataset folder
- Confirm that the image and label paths inside `data.yaml` are valid

### The model detects nothing in `test.py`

- Try lowering the confidence value in `test.py`
- Make sure the test image contains the object you trained on
- Check that the dataset labels are correct
- Train for more epochs if the model has not learned enough yet

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

- `main.py` - live RTSP detection and screenshot capture app
- `train.py` - trains a custom YOLO model
- `test.py` - tests the trained model on a still image
- `requirements.txt` - Python dependencies
- `.env` - camera configuration
- `runs/` - training and prediction outputs
- `screenshots/` - captured images
