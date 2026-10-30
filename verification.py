from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd

from metrics import ScreenGeometry, calculate_verification_metrics


@dataclass(frozen=True)
class VerificationTarget:
    name: str
    x: int
    y: int
    repetition: int
    trial_index: int

    @property
    def trial_id(self) -> str:
        return f"r{self.repetition}_{self.trial_index:02d}_{self.name}"


class VerificationSession:
    """Two-repeat nine-point test that never changes the trained model."""

    def __init__(
        self,
        window_width: int,
        window_height: int,
        target_duration: float = 1.9,
        settle_time: float = 0.55,
        edge_margin: float = 0.15,
    ):
        if not 0.0 < edge_margin < 0.5:
            raise ValueError("edge_margin must be between 0 and 0.5.")
        if not 0.0 <= settle_time < target_duration:
            raise ValueError(
                "settle_time must be non-negative and smaller than "
                "target_duration."
            )

        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.target_duration = float(target_duration)
        self.settle_time = float(settle_time)

        x_left = int(window_width * edge_margin)
        x_center = window_width // 2
        x_right = int(window_width * (1.0 - edge_margin))
        y_top = int(window_height * edge_margin)
        y_middle = window_height // 2
        y_bottom = int(window_height * (1.0 - edge_margin))

        positions = {
            "top_left": (x_left, y_top),
            "top_center": (x_center, y_top),
            "top_right": (x_right, y_top),
            "middle_left": (x_left, y_middle),
            "center": (x_center, y_middle),
            "middle_right": (x_right, y_middle),
            "bottom_left": (x_left, y_bottom),
            "bottom_center": (x_center, y_bottom),
            "bottom_right": (x_right, y_bottom),
        }

        # Fixed permutations make repeated runs comparable while avoiding a
        # predictable left-to-right scan through the grid.
        orders = [
            [
                "center",
                "top_left",
                "bottom_right",
                "middle_left",
                "top_right",
                "bottom_center",
                "top_center",
                "bottom_left",
                "middle_right",
            ],
            [
                "bottom_left",
                "top_center",
                "middle_right",
                "top_left",
                "center",
                "bottom_right",
                "middle_left",
                "top_right",
                "bottom_center",
            ],
        ]

        self.targets: list[VerificationTarget] = []
        trial_index = 1
        for repetition, order in enumerate(orders, start=1):
            for name in order:
                x, y = positions[name]
                self.targets.append(
                    VerificationTarget(
                        name=name,
                        x=x,
                        y=y,
                        repetition=repetition,
                        trial_index=trial_index,
                    )
                )
                trial_index += 1

        self.current_index = 0
        self.target_start_time = time.perf_counter()

    def current_target(self) -> VerificationTarget | None:
        if self.is_finished():
            return None
        return self.targets[self.current_index]

    def is_finished(self) -> bool:
        return self.current_index >= len(self.targets)

    def elapsed(self) -> float:
        if self.is_finished():
            return 0.0
        return time.perf_counter() - self.target_start_time

    def should_record(self) -> bool:
        return (
            not self.is_finished()
            and self.settle_time <= self.elapsed() < self.target_duration
        )

    def update(self) -> None:
        if self.is_finished() or self.elapsed() < self.target_duration:
            return
        self.current_index += 1
        self.target_start_time = time.perf_counter()

    def draw(self, prediction_valid: bool | None = None) -> np.ndarray:
        canvas = np.full(
            (self.window_height, self.window_width, 3),
            255,
            dtype=np.uint8,
        )
        target = self.current_target()
        if target is None:
            return canvas

        elapsed = self.elapsed()
        recording = self.should_record()
        outer_color = (0, 145, 255) if recording else (170, 170, 170)

        cv2.circle(canvas, (target.x, target.y), 30, outer_color, 4)
        cv2.circle(canvas, (target.x, target.y), 8, (0, 0, 220), -1)

        phase = "RECORDING" if recording else "MOVE AND SETTLE"
        cv2.putText(
            canvas,
            phase,
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 40, 40),
            2,
        )
        cv2.putText(
            canvas,
            f"Verification {self.current_index + 1}/{len(self.targets)}",
            (25, self.window_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 60, 60),
            2,
        )

        if prediction_valid is False and recording:
            cv2.putText(
                canvas,
                "Tracking invalid - keep face visible and head steady",
                (390, self.window_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 180),
                2,
            )

        progress = min(elapsed / self.target_duration, 1.0)
        bar_left = 25
        bar_top = 60
        bar_width = 280
        cv2.rectangle(
            canvas,
            (bar_left, bar_top),
            (bar_left + bar_width, bar_top + 12),
            (190, 190, 190),
            1,
        )
        cv2.rectangle(
            canvas,
            (bar_left, bar_top),
            (bar_left + int(bar_width * progress), bar_top + 12),
            outer_color,
            -1,
        )
        return canvas


def _format_metric(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}{suffix}"


def _draw_intro(
    width: int,
    height: int,
    viewing_distance_mm: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    lines = [
        ("INDEPENDENT GAZE VERIFICATION", 1.25, 350),
        ("Look at each red point and keep your head steady.", 0.85, 440),
        ("The model will not train or change during this test.", 0.85, 490),
        (
            f"Configured eye-to-screen distance: {viewing_distance_mm:.0f} mm",
            0.78,
            555,
        ),
        ("SPACE - start verification", 0.95, 650),
        ("Q - quit", 0.75, 710),
    ]
    for text, scale, y in lines:
        size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
        )[0]
        x = max((width - size[0]) // 2, 20)
        cv2.putText(
            canvas,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            2,
        )
    return canvas


def _draw_results(
    width: int,
    height: int,
    summary: dict[str, Any],
    geometry_note: str,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        "VERIFICATION RESULTS",
        (470, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (0, 0, 0),
        3,
    )

    lines = [
        (
            "Mean spatial accuracy",
            f"{_format_metric(summary.get('mean_accuracy_px'), 1, ' px')}  |  "
            f"{_format_metric(summary.get('mean_accuracy_deg'), 2, ' deg')}",
        ),
        (
            "Median spatial accuracy",
            f"{_format_metric(summary.get('median_accuracy_px'), 1, ' px')}  |  "
            f"{_format_metric(summary.get('median_accuracy_deg'), 2, ' deg')}",
        ),
        (
            "Spatial precision RMS-S2S",
            _format_metric(
                summary.get("mean_precision_rms_s2s_deg"),
                2,
                " deg",
            ),
        ),
        (
            "Radial RMS precision",
            _format_metric(
                summary.get("mean_precision_radial_rms_deg"),
                2,
                " deg",
            ),
        ),
        (
            "Valid prediction frames",
            _format_metric(summary.get("valid_frame_rate_pct"), 1, "%"),
        ),
        (
            "Mean backend latency",
            _format_metric(summary.get("mean_backend_latency_ms"), 1, " ms"),
        ),
        (
            "P90 sample error",
            _format_metric(summary.get("p90_sample_error_px"), 1, " px"),
        ),
    ]

    y = 220
    for label, value in lines:
        cv2.putText(
            canvas,
            label,
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (50, 50, 50),
            2,
        )
        cv2.putText(
            canvas,
            value,
            (840, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
        )
        y += 62

    cv2.putText(
        canvas,
        geometry_note[:105],
        (90, 700),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (90, 90, 90),
        1,
    )
    cv2.putText(
        canvas,
        "SPACE - start keyboard    R - repeat verification    Q - quit",
        (300, 805),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 0, 0),
        2,
    )
    return canvas


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _collect_once(
    cam,
    backend,
    gaze_model,
    feature_names: list[str],
    model_name: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    session = VerificationSession(width, height)
    rows: list[dict[str, Any]] = []
    frame_id = 0

    while not session.is_finished():
        session.update()
        if session.is_finished():
            break

        ret, frame = cam.read()
        if not ret:
            raise RuntimeError("Could not read a camera frame during verification.")

        tracking_result = backend.process(frame)
        target = session.current_target()
        prediction_valid = False
        predicted_x = np.nan
        predicted_y = np.nan
        prediction_latency_ms = np.nan
        invalid_reason = None

        if tracking_result.valid and target is not None:
            features = tracking_result.features
            eye_height_valid = bool(features.get("eye_height_valid", False))
            all_features_available = all(
                name in features
                and features[name] is not None
                and np.isfinite(float(features[name]))
                for name in feature_names
            )

            if not eye_height_valid:
                invalid_reason = "eye_height_invalid"
            elif not all_features_available:
                invalid_reason = "missing_model_feature"
            else:
                pose_reason = None
                if hasattr(gaze_model, "pose_gate_reason"):
                    pose_reason = gaze_model.pose_gate_reason(features)
                elif hasattr(gaze_model, "pose_is_allowed"):
                    if not gaze_model.pose_is_allowed(features):
                        pose_reason = "outside_calibrated_range"

                if pose_reason is not None:
                    invalid_reason = f"head_gate:{pose_reason}"
                else:
                    X_live = pd.DataFrame(
                        [[features[name] for name in feature_names]],
                        columns=feature_names,
                    )
                    start = time.perf_counter()
                    prediction = np.asarray(
                        gaze_model.predict(X_live)[0],
                        dtype=float,
                    )
                    prediction_latency_ms = (
                        time.perf_counter() - start
                    ) * 1000.0

                    if np.all(np.isfinite(prediction)):
                        predicted_x = float(
                            np.clip(prediction[0], 0, width - 1)
                        )
                        predicted_y = float(
                            np.clip(prediction[1], 0, height - 1)
                        )
                        prediction_valid = True
                    else:
                        invalid_reason = "non_finite_prediction"
        else:
            invalid_reason = tracking_result.error or "backend_invalid"

        if session.should_record() and target is not None:
            sample_error_px = (
                float(
                    np.hypot(
                        predicted_x - target.x,
                        predicted_y - target.y,
                    )
                )
                if prediction_valid
                else np.nan
            )
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_perf_counter": time.perf_counter(),
                    "trial_id": target.trial_id,
                    "trial_index": target.trial_index,
                    "repetition": target.repetition,
                    "target_name": target.name,
                    "target_x": target.x,
                    "target_y": target.y,
                    "model_name": model_name,
                    "backend_method": tracking_result.method_name,
                    "backend_valid": bool(tracking_result.valid),
                    "prediction_valid": bool(prediction_valid),
                    "invalid_reason": invalid_reason,
                    "predicted_x": predicted_x,
                    "predicted_y": predicted_y,
                    "sample_error_px": sample_error_px,
                    "backend_latency_ms": tracking_result.latency_ms,
                    "prediction_latency_ms": prediction_latency_ms,
                    "appearance_both_valid": tracking_result.features.get(
                        "appearance_both_valid"
                    ),
                }
            )
            frame_id += 1

        cv2.imshow("Verification", session.draw(prediction_valid))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt

    return rows


def run_verification(
    cam,
    backend,
    gaze_model,
    feature_names: list[str],
    model_name: str,
    geometry: ScreenGeometry,
    geometry_note: str,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Run independent verification, save raw data, and show results."""
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        cv2.destroyWindow("Calibration")
    except cv2.error:
        pass

    cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Verification",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    while True:
        while True:
            cv2.imshow(
                "Verification",
                _draw_intro(
                    geometry.width_px,
                    geometry.height_px,
                    geometry.viewing_distance_mm,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                raise KeyboardInterrupt

        rows = _collect_once(
            cam=cam,
            backend=backend,
            gaze_model=gaze_model,
            feature_names=feature_names,
            model_name=model_name,
            width=geometry.width_px,
            height=geometry.height_px,
        )

        samples = pd.DataFrame(rows)
        summary, trial_metrics = calculate_verification_metrics(
            samples,
            geometry,
        )
        summary["model_name"] = model_name
        summary["geometry_note"] = geometry_note

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        samples_path = output_path / f"verification_samples_{timestamp}.csv"
        trials_path = output_path / f"verification_trials_{timestamp}.csv"
        summary_path = output_path / f"verification_summary_{timestamp}.json"

        samples.to_csv(samples_path, index=False)
        trial_metrics.to_csv(trials_path, index=False)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(summary), file, indent=2, ensure_ascii=False)

        print()
        print("Independent verification results:")
        print(
            "  Mean spatial accuracy: "
            f"{_format_metric(summary.get('mean_accuracy_px'), 1, ' px')} | "
            f"{_format_metric(summary.get('mean_accuracy_deg'), 2, ' deg')}"
        )
        print(
            "  Median spatial accuracy: "
            f"{_format_metric(summary.get('median_accuracy_px'), 1, ' px')} | "
            f"{_format_metric(summary.get('median_accuracy_deg'), 2, ' deg')}"
        )
        print(
            "  Mean spatial precision RMS-S2S: "
            f"{_format_metric(summary.get('mean_precision_rms_s2s_deg'), 2, ' deg')}"
        )
        print(
            "  Mean radial RMS precision: "
            f"{_format_metric(summary.get('mean_precision_radial_rms_deg'), 2, ' deg')}"
        )
        print(
            "  Valid prediction frames: "
            f"{_format_metric(summary.get('valid_frame_rate_pct'), 1, '%')}"
        )
        print(f"  Samples: {samples_path}")
        print(f"  Trial metrics: {trials_path}")
        print(f"  Summary: {summary_path}")

        while True:
            cv2.imshow(
                "Verification",
                _draw_results(
                    geometry.width_px,
                    geometry.height_px,
                    summary,
                    geometry_note,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cv2.destroyWindow("Verification")
                return summary
            if key == ord("r"):
                break
            if key == ord("q"):
                raise KeyboardInterrupt

        # R repeats the whole test and overwrites only the in-memory summary;
        # each run still receives a new timestamped output file.
