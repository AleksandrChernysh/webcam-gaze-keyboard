from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

import camera
from tracker import GazeTracker
from calibration import CalibrationSession
from config import (
    WINDOW_WIDTH_PX,
    WINDOW_HEIGHT_PX,
    SCREEN_WIDTH_MM,
    SCREEN_HEIGHT_MM,
    VIEWING_DISTANCE_MM,
    SCREEN_GEOMETRY_NOTE,
    CALIBRATION_X_LEVELS,
    CALIBRATION_Y_LEVELS,
    CALIBRATION_ROUNDS,
    CALIBRATION_SETTLE_SEC,
    CALIBRATION_RECORD_SEC,
    CALIBRATION_MIN_VALID_SAMPLES,
    HEAD_REFERENCE_FRAMES,
    HEAD_STABLE_SEC,
    HEAD_PITCH_TOLERANCE_DEG,
    HEAD_YAW_TOLERANCE_DEG,
    HEAD_ROLL_TOLERANCE_DEG,
    HEAD_CENTER_TOLERANCE,
    HEAD_WIDTH_TOLERANCE_FRACTION,
    KEYBOARD_DWELL_SEC,
    KEYBOARD_REARM_DISTANCE_PX,
    SMOOTHING_HISTORY,
)
from gaze_model import FEATURES, load_gaze_model, train_gaze_model
from keyboard import VirtualKeyboard
from logger import FrameLogger
from metrics import ScreenGeometry
from verification import run_verification


WIDTH = WINDOW_WIDTH_PX
HEIGHT = WINDOW_HEIGHT_PX
HEAD_FEATURES = (
    "head_pose_pitch_deg",
    "head_pose_yaw_deg",
    "head_pose_roll_deg",
    "head_center_x",
    "head_center_y",
    "head_width",
)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _tracking_is_valid(result) -> bool:
    return bool(
        result.valid
        and result.features.get("eye_height_valid", False)
        and result.features.get("appearance_both_valid", False)
        and all(
            name in result.features
            and _finite_float(result.features[name]) is not None
            for name in FEATURES
        )
    )


def _draw_preview(canvas, frame, width=290):
    frame_height, frame_width = frame.shape[:2]
    scale = width / frame_width
    preview = cv2.resize(
        frame,
        (int(frame_width * scale), int(frame_height * scale)),
    )
    x1 = canvas.shape[1] - preview.shape[1] - 18
    y1 = 15
    canvas[y1:y1 + preview.shape[0], x1:x1 + preview.shape[1]] = preview
    cv2.rectangle(
        canvas,
        (x1, y1),
        (x1 + preview.shape[1], y1 + preview.shape[0]),
        (0, 0, 0),
        2,
    )


def _start_screen(model_exists: bool) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "SIMPLE WEBCAM GAZE TRACKER",
        (430, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.45,
        (0, 0, 0),
        3,
    )
    lines = [
        "SPACE - 15-point calibration -> verification -> keyboard",
        (
            "V - load model -> verification -> keyboard"
            if model_exists
            else "V - saved model not found"
        ),
        (
            "L - load model -> keyboard"
            if model_exists
            else "L - saved model not found"
        ),
        "Q - quit",
    ]
    for index, line in enumerate(lines):
        color = (0, 0, 0)
        if "not found" in line:
            color = (150, 150, 150)
        cv2.putText(
            canvas,
            line,
            (355, 365 + index * 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            color,
            2,
        )
    cv2.putText(
        canvas,
        "ML: one StandardScaler + multi-output Ridge Regression pipeline",
        (340, 725),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (40, 40, 40),
        2,
    )
    return canvas


def capture_head_reference(cam, tracker) -> dict[str, float]:
    samples: list[dict[str, float]] = []

    while len(samples) < HEAD_REFERENCE_FRAMES:
        ok, frame = camera.read_frame(cam)
        if not ok:
            raise RuntimeError("Could not read camera during head alignment.")
        result = tracker.process(frame)
        annotated = result.annotated_frame if result.annotated_frame is not None else frame

        if result.valid and result.features.get("eye_height_valid", False):
            values = {
                name: _finite_float(result.features.get(name))
                for name in HEAD_FEATURES
            }
            if all(value is not None for value in values.values()):
                samples.append({name: float(value) for name, value in values.items()})

        canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        cv2.putText(
            canvas,
            "ALIGN YOUR HEAD",
            (535, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            canvas,
            "Face the camera and hold still",
            (470, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            (30, 30, 30),
            2,
        )
        cv2.putText(
            canvas,
            f"Valid frames: {len(samples)}/{HEAD_REFERENCE_FRAMES}",
            (520, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88,
            (0, 140, 0),
            2,
        )
        _draw_preview(canvas, annotated, width=340)
        cv2.imshow("Gaze Tracker", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    return {
        name: float(np.median([sample[name] for sample in samples]))
        for name in HEAD_FEATURES
    }


class HeadGate:
    """Simple non-ML check that keeps calibration head position consistent."""

    def __init__(self, reference: dict[str, float]):
        self.reference = reference
        self.stable_since: float | None = None

    def reset(self):
        self.stable_since = None

    def evaluate(self, features: dict) -> tuple[bool, str]:
        values = {name: _finite_float(features.get(name)) for name in HEAD_FEATURES}
        if any(value is None for value in values.values()):
            self.reset()
            return False, "FACE NOT TRACKED"

        pitch = values["head_pose_pitch_deg"] - self.reference["head_pose_pitch_deg"]
        yaw = values["head_pose_yaw_deg"] - self.reference["head_pose_yaw_deg"]
        roll = values["head_pose_roll_deg"] - self.reference["head_pose_roll_deg"]
        dx = values["head_center_x"] - self.reference["head_center_x"]
        dy = values["head_center_y"] - self.reference["head_center_y"]
        width_ratio = values["head_width"] / max(self.reference["head_width"], 1e-6)

        message = None
        if abs(dx) > HEAD_CENTER_TOLERANCE:
            message = "MOVE HEAD LEFT" if dx > 0 else "MOVE HEAD RIGHT"
        elif abs(dy) > HEAD_CENTER_TOLERANCE:
            message = "MOVE HEAD UP" if dy > 0 else "MOVE HEAD DOWN"
        elif width_ratio > 1 + HEAD_WIDTH_TOLERANCE_FRACTION:
            message = "MOVE FARTHER"
        elif width_ratio < 1 - HEAD_WIDTH_TOLERANCE_FRACTION:
            message = "MOVE CLOSER"
        elif abs(yaw) > HEAD_YAW_TOLERANCE_DEG:
            message = "FACE CAMERA - YAW"
        elif abs(pitch) > HEAD_PITCH_TOLERANCE_DEG:
            message = "FACE CAMERA - PITCH"
        elif abs(roll) > HEAD_ROLL_TOLERANCE_DEG:
            message = "LEVEL YOUR HEAD"

        if message is not None:
            self.reset()
            return False, message

        now = time.perf_counter()
        if self.stable_since is None:
            self.stable_since = now
        remaining = HEAD_STABLE_SEC - (now - self.stable_since)
        if remaining > 0:
            return False, f"HOLD STILL: {remaining:.1f}s"
        return True, "HEAD OK"


def run_calibration(cam, tracker, model_path: Path):
    reference = capture_head_reference(cam, tracker)
    gate = HeadGate(reference)
    session = CalibrationSession(
        width=WIDTH,
        height=HEIGHT,
        x_levels=CALIBRATION_X_LEVELS,
        y_levels=CALIBRATION_Y_LEVELS,
        rounds=CALIBRATION_ROUNDS,
        settle_sec=CALIBRATION_SETTLE_SEC,
        record_sec=CALIBRATION_RECORD_SEC,
        min_valid_samples=CALIBRATION_MIN_VALID_SAMPLES,
    )
    logger = FrameLogger()

    while not session.is_finished():
        if session.is_paused():
            camera.read_frame(cam)
            cv2.imshow("Gaze Tracker", session.draw())
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                gate.reset()
                session.resume()
            elif key == ord("q"):
                raise KeyboardInterrupt
            continue

        ok, frame = camera.read_frame(cam)
        if not ok:
            raise RuntimeError("Could not read camera during calibration.")
        result = tracker.process(frame)

        if _tracking_is_valid(result):
            quality_ok, status = gate.evaluate(result.features)
        else:
            gate.reset()
            quality_ok, status = False, "FACE OR EYES NOT DETECTED"

        target = session.current_target()
        should_log = bool(quality_ok and session.should_record())
        logger.log_result(result, target=target, should_log=should_log)
        if should_log:
            session.record_valid_sample()

        cv2.imshow("Gaze Tracker", session.draw(quality_ok, status))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        session.update(quality_ok)

    csv_path = logger.save()
    print(f"Calibration saved to: {csv_path}")
    return train_gaze_model(
        csv_path=csv_path,
        screen_width=WIDTH,
        screen_height=HEIGHT,
        model_path=model_path,
    )


def _predict(model, feature_names, features):
    values = []
    for name in feature_names:
        value = _finite_float(features.get(name))
        if value is None:
            return None
        values.append(value)
    frame = pd.DataFrame([values], columns=feature_names)
    prediction = np.asarray(model.predict(frame)[0], dtype=float)
    if prediction.shape != (2,) or not np.all(np.isfinite(prediction)):
        return None
    prediction[0] = np.clip(prediction[0], 0, WIDTH - 1)
    prediction[1] = np.clip(prediction[1], 0, HEIGHT - 1)
    return prediction


def _adaptive_alpha(distance_px: float) -> float:
    if distance_px < 80:
        return 0.18
    if distance_px < 180:
        return 0.38
    return 0.65


def run_keyboard(cam, tracker, model, feature_names, model_name):
    cv2.destroyAllWindows()
    cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Virtual Keyboard",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    keyboard = VirtualKeyboard(
        width=WIDTH,
        height=HEIGHT,
        dwell_seconds=KEYBOARD_DWELL_SEC,
        rearm_distance_px=KEYBOARD_REARM_DISTANCE_PX,
    )
    history = deque(maxlen=SMOOTHING_HISTORY)
    smoothed = None
    show_preview = False

    try:
        while True:
            ok, frame = camera.read_frame(cam)
            if not ok:
                break
            result = tracker.process(frame)
            annotated = result.annotated_frame if result.annotated_frame is not None else frame

            gaze_point = None
            tracking_valid = False
            status = "No valid gaze prediction"
            if _tracking_is_valid(result):
                prediction = _predict(model, feature_names, result.features)
                if prediction is not None:
                    history.append(prediction)
                    median = np.median(np.stack(history), axis=0)
                    if smoothed is None:
                        smoothed = median.copy()
                    else:
                        movement = float(np.linalg.norm(median - smoothed))
                        alpha = _adaptive_alpha(movement)
                        smoothed = alpha * median + (1 - alpha) * smoothed
                    gaze_point = (float(smoothed[0]), float(smoothed[1]))
                    tracking_valid = True
                    status = f"Tracking OK | {model_name}"
            else:
                history.clear()
                smoothed = None

            keyboard.update(gaze_point=gaze_point, tracking_valid=tracking_valid)
            canvas = keyboard.draw(
                gaze_point=gaze_point,
                tracking_valid=tracking_valid,
                status_text=status,
            )
            if show_preview:
                _draw_preview(canvas, annotated)
            cv2.imshow("Virtual Keyboard", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_preview = not show_preview
    finally:
        path = keyboard.save_log()
        if path is not None:
            print(f"Keyboard log saved to: {path}")
        print(f"Final typed text: {keyboard.text}")


def main():
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "gaze_model.joblib"

    cam = camera.open_camera()
    if cam is None:
        raise SystemExit(1)
    tracker = GazeTracker()

    try:
        cv2.namedWindow("Gaze Tracker", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Gaze Tracker",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

        mode = None
        while mode is None:
            cv2.imshow("Gaze Tracker", _start_screen(model_path.exists()))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                mode = "calibrate"
            elif key == ord("v") and model_path.exists():
                mode = "verify"
            elif key == ord("l") and model_path.exists():
                mode = "load"
            elif key == ord("q"):
                raise KeyboardInterrupt

        if mode == "calibrate":
            model, feature_names, validation, model_name = run_calibration(
                cam, tracker, model_path
            )
        else:
            model, feature_names, validation, model_name = load_gaze_model(
                model_path,
                expected_width=WIDTH,
                expected_height=HEIGHT,
            )
            print(f"Loaded model: {model_path}")

        print(
            "Calibration validation median error: "
            f"{validation.get('median_error_px', float('nan')):.1f} px"
        )

        if mode in {"calibrate", "verify"}:
            geometry = ScreenGeometry(
                width_px=WIDTH,
                height_px=HEIGHT,
                width_mm=SCREEN_WIDTH_MM,
                height_mm=SCREEN_HEIGHT_MM,
                viewing_distance_mm=VIEWING_DISTANCE_MM,
            )
            run_verification(
                cam=cam,
                backend=tracker,
                gaze_model=model,
                feature_names=feature_names,
                model_name=model_name,
                geometry=geometry,
                geometry_note=SCREEN_GEOMETRY_NOTE,
            )

        run_keyboard(cam, tracker, model, feature_names, model_name)

    except KeyboardInterrupt:
        pass
    finally:
        camera.close_camera(cam)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
