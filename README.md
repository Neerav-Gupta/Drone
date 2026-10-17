# Autonomous Following Drone

`tracker.py` detects and tracks a face from a USB camera, estimates how far away it is, and prints a movement command (`left`, `up`, `forward`, `hold`, ...) that a drone control loop can consume. It is tuned for low latency on a Raspberry Pi 4B.

## How it works

1. **YuNet detection**: `cv2.FaceDetectorYN` runs a small DNN face detector (`face_detection_yunet_2023mar.onnx`). When a face is already being tracked, it only searches a region of interest around the last known box (`--roi-expand`), which keeps it fast. With no target, it scans the whole frame.
2. **MOSSE tracking**: between detections, a lightweight MOSSE tracker follows the face frame to frame.
3. **Drift check**: MOSSE can slide onto background without reporting failure. A template-correlation check against a reference patch (`--similarity-threshold`) catches this and forces re-detection.
4. **Kalman filter**: a constant-velocity Kalman filter smooths the box and exposes velocity. If tracking drops for a few frames (up to 5), it coasts on the motion model before declaring the face lost.
5. **Distance estimate**: distance comes from the face width in pixels using similar triangles, after a one-time focal-length calibration.
6. **Motion command**: the face offset from frame center (`dx`, `dy`) and the distance error are compared against deadbands to produce a command such as `left+up+forward`, or `hold`.

The camera is read on a separate thread that keeps only the newest frame (single-frame queue), so latency doesn't build up. The camera is opened through V4L2 and requests MJPEG.

## Files

| File | Purpose |
| --- | --- |
| `tracker.py` | The tracker (detection, tracking, distance, motion commands) |
| `face_detection_yunet_2023mar.onnx` | YuNet face detection model |
| `requirements.txt` | Python dependencies |

## Install

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

MOSSE lives in OpenCV's contrib modules, and `cv2.FaceDetectorYN` needs OpenCV 4.5.4 or newer. If either is missing, install `opencv-contrib-python`.

The model file is included in the repo. If it's missing, download it next to `tracker.py`:

```bash
curl -L -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Usage

### 1. Calibrate distance (once per camera and resolution)

Stand a known distance from the camera, face it directly, and run:

```bash
python3 tracker.py --calibrate-distance-cm 80
```

It samples your face width for about 3 seconds and prints a `--focal-length-px` value. Use that value in later runs. Recalibrate if you change the camera, `--width`, or `--height`.

### 2. Run the tracker

```bash
python3 tracker.py --focal-length-px 401.5 --target-distance-cm 80 --distance-deadband-cm 10 --preview
```

Without `--focal-length-px`, tracking still works but distance is `n/a` and no `forward`/`backward` commands are produced.

Press `q` or `Esc` in the preview window, or `Ctrl+C`, to quit.

### Headless mode

If there's no desktop session, the script switches to headless mode automatically. Preview is off unless you pass `--preview`. For maximum speed, add `--headless`:

```bash
python3 tracker.py --focal-length-px 401.5 --headless
```

### Output

The script prints state changes and periodic status lines:

```
Face Detected, Following
  dx=+12 dy=-5 dist=82cm -> right+up
Face Lost, Searching
```

- `dx`, `dy`: face center minus frame center, in pixels (positive is right and down)
- `dist`: estimated distance to the face
- command: `left`/`right`, `up`/`down`, `forward`/`backward`, joined with `+`, or `hold`

Use these values to drive yaw/pitch/throttle PID loops or any follow-person logic.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--camera-id` | `0` | Camera index |
| `--device` | | Explicit device path, e.g. `/dev/video0` |
| `--width`, `--height`, `--fps` | `320`, `180`, `30` | Capture settings |
| `--model-path` | bundled `.onnx` | YuNet model path |
| `--score-threshold` | `0.7` | Minimum detection confidence |
| `--nms-threshold` | `0.3` | Detector non-max suppression threshold |
| `--detect-every` | `5` | Re-confirm with the detector every N frames while tracking |
| `--roi-expand` | `2.5` | Detection ROI size as a multiple of the last face box |
| `--similarity-threshold` | `0.35` | Minimum correlation to trust MOSSE before re-detecting |
| `--known-face-width-cm` | `16.0` | Real face width used for distance estimation |
| `--focal-length-px` | none | Camera focal length from calibration |
| `--calibrate-distance-cm` | none | Run calibration at this distance and exit |
| `--target-distance-cm` | `150.0` | Distance to maintain from the subject |
| `--distance-deadband-cm` | `25.0` | Tolerance before `forward`/`backward` |
| `--pixel-deadband` | `20` | Pixel tolerance before `left`/`right`/`up`/`down` |
| `--status-interval` | `0.5` | Seconds between status lines |
| `--cv-threads` | `2` | OpenCV threads (leaves a core for the camera thread) |
| `--headless` | off | Disable display |
| `--preview` | off | Show a small preview window if a display exists |
| `--display-scale` | `0.4` | Preview scale |

## Camera troubleshooting

- If the default camera doesn't open, the script tries `/dev/video{camera-id}`, camera `1`, `/dev/video0`, and `/dev/video1`. You can also pass one explicitly: `--device /dev/video1`.
- Keep `--width`/`--height` modest for speed. A USB Arducam works well at the default 320x180.
- Raise `--detect-every` for more FPS, or lower it for a more stable lock.

## Performance tips

- Use Raspberry Pi OS 64-bit.
- Keep the CPU cool, since thermal throttling hurts FPS.
- Close desktop apps while running, and use `--headless` when feeding a control loop.
