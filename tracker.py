# My usage:
# python3 tracker.py --focal-length-px 401.5 --target-distance-cm 80 --distance-deadband-cm 10 --preview

import argparse
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx"
)


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def expanded(self, factor: float, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clipped (x1, y1, x2, y2) ROI centered on this box, `factor`x its size."""
        frame_h, frame_w = frame_shape
        cx, cy = self.center
        half_w = max(self.w, 1) * factor / 2.0
        half_h = max(self.h, 1) * factor / 2.0
        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h))
        x2 = min(frame_w, int(cx + half_w))
        y2 = min(frame_h, int(cy + half_h))
        return x1, y1, x2, y2


class ThreadedCamera:

    @staticmethod
    def _try_open(source: int | str) -> Optional[cv2.VideoCapture]:
        candidates = [
            (cv2.CAP_V4L2, source),
            (cv2.CAP_ANY, source),
        ]
        for backend, candidate in candidates:
            cap = cv2.VideoCapture(candidate, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def __init__(self, camera_source: int | str, width: int, height: int, fps: int):
        self.cap = self._try_open(camera_source)
        if self.cap is None and isinstance(camera_source, int):
            self.cap = self._try_open(f"/dev/video{camera_source}")
        if self.cap is None:
            raise RuntimeError(f"Cannot open camera source {camera_source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(frame)

    def read(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class KalmanBoxTracker:

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self._initialized = False
        self._last_t = time.perf_counter()

    def _set_dt(self, dt: float) -> None:
        m = np.eye(8, dtype=np.float32)
        for i in range(4):
            m[i, i + 4] = dt
        self.kf.transitionMatrix = m

    def init(self, bbox: BBox) -> None:
        cx, cy = bbox.center
        self.kf.statePost = np.array(
            [cx, cy, bbox.w, bbox.h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ).reshape(8, 1)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self._initialized = True
        self._last_t = time.perf_counter()

    def forget(self) -> None:
        self._initialized = False

    def predict(self) -> Optional[BBox]:
        if not self._initialized:
            return None
        now = time.perf_counter()
        self._set_dt(max(1e-3, now - self._last_t))
        self._last_t = now
        return self._to_bbox(self.kf.predict())

    def correct(self, bbox: BBox) -> BBox:
        if not self._initialized:
            self.init(bbox)
        cx, cy = bbox.center
        measurement = np.array([cx, cy, bbox.w, bbox.h], dtype=np.float32).reshape(4, 1)
        return self._to_bbox(self.kf.correct(measurement))

    @property
    def velocity(self) -> Tuple[float, float]:
        if not self._initialized:
            return (0.0, 0.0)
        s = self.kf.statePost
        return float(s[4, 0]), float(s[5, 0])

    @staticmethod
    def _to_bbox(state: np.ndarray) -> BBox:
        cx, cy, w, h = (float(state[i, 0]) for i in range(4))
        w = max(1.0, w)
        h = max(1.0, h)
        return BBox(
            x=int(round(cx - w / 2.0)),
            y=int(round(cy - h / 2.0)),
            w=int(round(w)),
            h=int(round(h)),
        )


class FaceTracker:
    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        roi_expand: float,
        similarity_threshold: float,
    ):
        if not os.path.isfile(model_path):
            raise RuntimeError(
                f"YuNet model not found at {model_path}.\n"
                "Download it once with:\n"
                "  curl -L -o face_detection_yunet_2023mar.onnx "
                "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        self.detector = self._create_yunet(model_path, (320, 320), score_threshold, nms_threshold)
        self.roi_expand = roi_expand
        self.similarity_threshold = similarity_threshold

        self.tracker: Optional[cv2.Tracker] = None
        self.kalman = KalmanBoxTracker()
        self.bbox: Optional[BBox] = None
        self._ref_patch: Optional[np.ndarray] = None

    @staticmethod
    def _create_yunet(model_path: str, input_size, score_threshold: float, nms_threshold: float):
        if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
            return cv2.FaceDetectorYN.create(model_path, "", input_size, score_threshold, nms_threshold, 5000)
        if hasattr(cv2, "FaceDetectorYN_create"):
            return cv2.FaceDetectorYN_create(model_path, "", input_size, score_threshold, nms_threshold, 5000)
        raise RuntimeError(
            "cv2.FaceDetectorYN not available -- need OpenCV >= 4.5.4 with the dnn module "
            "(pip install --upgrade opencv-contrib-python)."
        )

    @staticmethod
    def _create_mosse_tracker() -> cv2.Tracker:
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()
        raise RuntimeError("MOSSE tracker not available. Install opencv-contrib-python package.")

    def detect(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[BBox, float]]:
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            crop = frame
            ox, oy = 0, 0

        h, w = crop.shape[:2]
        if h < 16 or w < 16:
            return None

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(crop)
        if faces is None or len(faces) == 0:
            return None

        frame_shape = frame.shape[:2]
        best: Optional[BBox] = None
        best_score = -1.0
        for f in faces:
            x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[14]
            if fw <= 0 or fh <= 0:
                continue
            bbox = BBox(x=int(x + ox), y=int(y + oy), w=int(fw), h=int(fh))
            if score <= best_score or not self._is_plausible_face(bbox, frame_shape):
                continue
            best, best_score = bbox, float(score)

        if best is None:
            return None
        return best, best_score

    @staticmethod
    def _is_plausible_face(bbox: BBox, frame_shape: Tuple[int, int]) -> bool:
        frame_h, frame_w = frame_shape
        if bbox.w <= 0 or bbox.h <= 0:
            return False

        x1, y1 = max(bbox.x, 0), max(bbox.y, 0)
        x2, y2 = min(bbox.x + bbox.w, frame_w), min(bbox.y + bbox.h, frame_h)
        visible_area = max(0, x2 - x1) * max(0, y2 - y1)
        if visible_area < 0.6 * bbox.w * bbox.h:
            return False

        area_frac = (bbox.w * bbox.h) / float(frame_w * frame_h)
        if not (0.01 <= area_frac <= 0.6):
            return False

        aspect = bbox.w / float(bbox.h)
        if not (0.6 <= aspect <= 1.6):
            return False

        return True

    def reacquire(self, frame: np.ndarray) -> Optional[Tuple[BBox, float]]:
        return self.detect(frame, roi=None)

    def search_roi(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        if self.bbox is None:
            return None
        return self.bbox.expanded(self.roi_expand, frame_shape)

    def reset_tracker(self, frame: np.ndarray, bbox: BBox) -> None:
        self.tracker = self._create_mosse_tracker()
        self.tracker.init(frame, (bbox.x, bbox.y, bbox.w, bbox.h))
        self.bbox = bbox
        self.kalman.init(bbox)
        self._ref_patch = self._extract_patch(frame, bbox)

    def update_tracker(self, frame: np.ndarray) -> Optional[Tuple[BBox, float]]:
        if self.tracker is None:
            return None
        ok, rect = self.tracker.update(frame)
        if not ok:
            return None
        x, y, w, h = rect
        bbox = BBox(int(x), int(y), int(w), int(h))
        return bbox, self._similarity(frame, bbox)

    @staticmethod
    def _extract_patch(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1 = max(0, bbox.x), max(0, bbox.y)
        x2 = min(gray.shape[1], bbox.x + bbox.w)
        y2 = min(gray.shape[0], bbox.y + bbox.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.resize(gray[y1:y2, x1:x2], (48, 48))

    def _similarity(self, frame: np.ndarray, bbox: BBox) -> float:
        if self._ref_patch is None:
            return 1.0
        patch = self._extract_patch(frame, bbox)
        if patch is None:
            return 0.0
        result = cv2.matchTemplate(patch, self._ref_patch, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])


def estimate_distance_cm(face_width_px: float, known_width_cm: float, focal_length_px: float) -> Optional[float]:
    if face_width_px <= 0 or focal_length_px <= 0:
        return None
    return (known_width_cm * focal_length_px) / face_width_px


def focal_length_from_calibration(face_width_px: float, known_distance_cm: float, known_width_cm: float) -> float:
    return (face_width_px * known_distance_cm) / known_width_cm


def classify_motion(
    dx: int,
    dy: int,
    distance_cm: Optional[float],
    target_distance_cm: float,
    pixel_deadband: int,
    distance_deadband_cm: float,
) -> str:
    moves = []
    if dy < -pixel_deadband:
        moves.append("up")
    elif dy > pixel_deadband:
        moves.append("down")
    if dx < -pixel_deadband:
        moves.append("left")
    elif dx > pixel_deadband:
        moves.append("right")
    if distance_cm is not None:
        if distance_cm > target_distance_cm + distance_deadband_cm:
            moves.append("forward")
        elif distance_cm < target_distance_cm - distance_deadband_cm:
            moves.append("backward")
    return "+".join(moves) if moves else "hold"


def draw_overlay(frame: np.ndarray, bbox: Optional[BBox], fps: float, distance_cm: Optional[float], move_cmd: str) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(frame, (cx, cy), (0, 180, 255), cv2.MARKER_CROSS, 16, 1)

    if bbox is not None:
        cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (0, 255, 0), 2)
        fx, fy = bbox.center
        cv2.circle(frame, (fx, fy), 4, (0, 255, 0), -1)
        dx, dy = fx - cx, fy - cy
        dist_str = f"{distance_cm:.0f}cm" if distance_cm is not None else "n/a"
        cv2.putText(
            frame,
            f"dx={dx:+d} dy={dy:+d} dist={dist_str} move={move_cmd}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast face tracking + distance estimation for Raspberry Pi 4B")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="", help="Explicit camera device path, e.g. /dev/video0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the YuNet ONNX face-detection model")
    parser.add_argument("--score-threshold", type=float, default=0.7, help="Minimum YuNet confidence to accept a detection")
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--detect-every", type=int, default=5, help="Re-confirm with the detector every N frames while tracking")
    parser.add_argument("--roi-expand", type=float, default=2.5, help="Detection ROI size as a multiple of the last known face box")
    parser.add_argument("--similarity-threshold", type=float, default=0.35, help="Min template-correlation to trust the MOSSE output before forcing reacquisition")

    parser.add_argument("--known-face-width-cm", type=float, default=16.0, help="Real-world face width (ear to ear, ~16cm for an adult) used for distance estimation")
    parser.add_argument("--focal-length-px", type=float, default=None, help="Camera focal length in pixels; obtain it once via --calibrate-distance-cm")
    parser.add_argument("--calibrate-distance-cm", type=float, default=None, help="Calibration mode: stand exactly this many cm from the camera; the script prints the --focal-length-px to use afterwards")
    parser.add_argument("--target-distance-cm", type=float, default=150.0, help="Distance the drone should try to maintain from the subject")
    parser.add_argument("--distance-deadband-cm", type=float, default=25.0, help="Tolerance band around the target distance before issuing forward/backward")
    parser.add_argument("--pixel-deadband", type=int, default=20, help="Tolerance in pixels around frame center before issuing left/right/up/down")
    parser.add_argument("--status-interval", type=float, default=0.5, help="Seconds between printed status lines")

    parser.add_argument("--cv-threads", type=int, default=2, help="OpenCV internal thread count (leave a core free for the camera-reader thread)")
    parser.add_argument("--headless", action="store_true", help="Disable display for max speed")
    parser.add_argument("--preview", action="store_true", help="Show a small preview window when a GUI display is available")
    parser.add_argument("--display-scale", type=float, default=0.4, help="Scale the shown image down to reduce GUI cost")
    return parser.parse_args()


def _run_calibration(cam: ThreadedCamera, tracker: FaceTracker, args: argparse.Namespace) -> None:
    print(
        f"Calibration: stand exactly {args.calibrate_distance_cm:.0f} cm from the camera, "
        "facing it directly. Sampling face width for ~3 seconds..."
    )
    samples = []
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline:
        frame = cam.read(timeout=0.25)
        if frame is None:
            continue
        found = tracker.reacquire(frame)
        if found is not None:
            bbox, _ = found
            samples.append(bbox.w)

    if not samples:
        print("No face detected during calibration -- check lighting/distance and retry.")
        return

    avg_width_px = float(np.mean(samples))
    focal_length_px = focal_length_from_calibration(avg_width_px, args.calibrate_distance_cm, args.known_face_width_cm)
    print(f"Average detected face width: {avg_width_px:.1f}px over {len(samples)} samples")
    print(f"Focal length: {focal_length_px:.1f}px")
    print(f"Re-run with:  --focal-length-px {focal_length_px:.1f}")


def main() -> None:
    args = parse_args()
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(args.cv_threads)
    except Exception:
        pass

    gui_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show_window = gui_available and args.preview and not args.headless

    if not gui_available and not args.headless:
        print("No display detected; running in headless mode to avoid Qt/xcb errors.")

    stop = False

    def _sigint_handler(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    camera_sources: list[int | str]
    if args.device:
        camera_sources = [args.device]
    else:
        camera_sources = [args.camera_id, f"/dev/video{args.camera_id}", 1, "/dev/video0", "/dev/video1"]

    cam = None
    last_error: Optional[Exception] = None
    for source in camera_sources:
        try:
            cam = ThreadedCamera(source, args.width, args.height, args.fps)
            break
        except Exception as exc:
            last_error = exc
            continue

    if cam is None:
        raise RuntimeError(
            "Unable to open any camera source. Try --device /dev/video0 or /dev/video1."
        ) from last_error

    tracker = FaceTracker(
        model_path=args.model_path,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        roi_expand=args.roi_expand,
        similarity_threshold=args.similarity_threshold,
    )

    if args.calibrate_distance_cm is not None:
        try:
            _run_calibration(cam, tracker, args)
        finally:
            cam.release()
        return

    frame_idx = 0
    last_time = time.perf_counter()
    last_status_print = 0.0
    fps_ema = 0.0
    tracking_active = False
    coast_frames = 0
    MAX_COAST_FRAMES = 5

    try:
        while not stop:
            frame = cam.read(timeout=0.25)
            if frame is None:
                continue

            frame_idx += 1
            frame_shape = frame.shape[:2]
            measured: Optional[BBox] = None

            if tracker.bbox is None:
                found = tracker.reacquire(frame)
                if found is not None:
                    measured, _ = found
                    tracker.reset_tracker(frame, measured)
            else:
                if (frame_idx % max(1, args.detect_every)) == 0:
                    roi = tracker.search_roi(frame_shape)
                    found = tracker.detect(frame, roi=roi)
                    if found is not None:
                        measured, _ = found
                        tracker.reset_tracker(frame, measured)

                if measured is None:
                    tracked = tracker.update_tracker(frame)
                    if tracked is not None:
                        cand_bbox, similarity = tracked
                        if similarity >= args.similarity_threshold:
                            measured = cand_bbox
                        else:
                            tracker.tracker = None  # drifted -- force reacquisition

            bbox: Optional[BBox]
            if measured is not None:
                bbox = tracker.kalman.correct(measured)
                tracker.bbox = bbox
                coast_frames = 0
            elif tracker.bbox is not None and coast_frames < MAX_COAST_FRAMES:
                # Brief coast on the motion model rather than snapping straight
                # to "lost" on a single bad frame.
                bbox = tracker.kalman.predict()
                tracker.bbox = bbox
                coast_frames += 1
            else:
                bbox = None
                tracker.bbox = None
                tracker.tracker = None
                tracker.kalman.forget()
                coast_frames = 0

            distance_cm: Optional[float] = None
            move_cmd = "hold"
            dx = dy = 0
            if bbox is not None:
                fh, fw = frame_shape
                cx, cy = bbox.center
                dx, dy = cx - fw // 2, cy - fh // 2
                if args.focal_length_px:
                    distance_cm = estimate_distance_cm(bbox.w, args.known_face_width_cm, args.focal_length_px)
                move_cmd = classify_motion(dx, dy, distance_cm, args.target_distance_cm, args.pixel_deadband, args.distance_deadband_cm)

            if bbox is not None and not tracking_active:
                print("Face Detected, Following")
                tracking_active = True
            elif bbox is None and tracking_active:
                print("Face Lost, Searching")
                tracking_active = False

            now = time.perf_counter()
            dt = max(1e-6, now - last_time)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.1 * inst_fps + 0.9 * fps_ema)
            last_time = now

            if bbox is not None and (now - last_status_print) >= args.status_interval:
                dist_str = f"{distance_cm:.0f}cm" if distance_cm is not None else "n/a (run --calibrate-distance-cm)"
                print(f"  dx={dx:+d} dy={dy:+d} dist={dist_str} -> {move_cmd}")
                last_status_print = now

            if show_window:
                display = frame
                if args.display_scale != 1.0:
                    display = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
                shown_bbox = None
                if bbox is not None:
                    shown_bbox = BBox(
                        x=int(bbox.x * args.display_scale),
                        y=int(bbox.y * args.display_scale),
                        w=int(bbox.w * args.display_scale),
                        h=int(bbox.h * args.display_scale),
                    )
                draw_overlay(display, shown_bbox, fps_ema, distance_cm, move_cmd)
                cv2.imshow("Fast Face Tracking (Pi 4B)", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        cam.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
