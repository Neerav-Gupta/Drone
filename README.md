# Super-Fast Person Tracking for Raspberry Pi 4B

This implementation is optimized for **high FPS + low latency** on Raspberry Pi 4B and is intended for a drone that follows a person.

## Why this is fast

- Threaded camera reader with a **single-frame queue** (no lag buildup)
- Person detection only every `N` frames
- MOSSE tracker updates in-between detections (very lightweight)
- Detection on downscaled grayscale image
- The shown image is also scaled down to reduce GUI work
- Tuned defaults for Pi 4B

## 1) Install

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 2) Run

```bash
python3 fast_face_tracking_pi.py
```

If the Pi has no desktop session, the script automatically switches to headless mode so OpenCV does not try to open a Qt window.

Preview is off by default; add `--preview` only if you want a small GUI window.

Press `q` to exit.

If the camera does not open with the default index, use the device path explicitly:

```bash
python3 fast_face_tracking_pi.py --device /dev/video0
```

The default mode is `auto`, which prefers person detectors so it can keep tracking even when the face is not visible.

## 3) Fastest mode (no GUI)

For maximum FPS, disable display:

```bash
python3 fast_face_tracking_pi.py --headless
```

## 4) Recommended performance presets

### Balanced (default)

```bash
python3 fast_face_tracking_pi.py --width 320 --height 180 --fps 30 --detect-every 15 --process-scale 0.25 --display-scale 0.4 --preview
```

### Maximum FPS (lower accuracy)

```bash
python3 fast_face_tracking_pi.py --width 320 --height 180 --fps 30 --detect-every 20 --process-scale 0.2 --min-face 28 --headless
```

### Better stability (more detector refresh)

```bash
python3 fast_face_tracking_pi.py --detect-every 8 --process-scale 0.3 --preview
```

## 5) Important Pi camera notes

- For a USB Arducam, start with `--camera-id 0` and keep `--width` / `--height` modest for speed.
- If `--camera-id 0` fails, use `--device /dev/video0` or `--device /dev/video1`.
- If the camera is not the first video device, try `--camera-id 1`.
- This script opens the camera through V4L2 and prefers MJPEG, which is usually the fastest path for USB cameras on the Pi.
- If you want maximum speed, keep the preview small or run `--headless` and feed the offsets directly to your drone control loop.

## 6) Hardware and OS tips for best speed

- Use Raspberry Pi OS 64-bit
- Keep CPU cool (thermal throttling hurts FPS)
- Close desktop apps while running tracking
- Use `--headless` when feeding tracking output to robot control logic

## Output to robot control

The script computes target center offset from frame center and overlays:

- `offset x=... y=...`

Use those offsets to drive pan/tilt PID, yaw/pitch control, or any follow-person logic.
