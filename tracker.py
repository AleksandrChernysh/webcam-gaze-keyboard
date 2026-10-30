"""Low-level tracking: MediaPipe geometry plus a compact eye-image grid."""

from __future__ import annotations

import time

import cv2
import numpy as np

from backends.base import TrackingResult
from backends.mediapipe_backend import MediaPipeBackend
from backends.eye_appearance import EyeAppearanceExtractor


class GazeTracker:
    """Extract the fixed feature set used by the Ridge gaze model.

    MediaPipe supplies facial/eye geometry. OpenCV normalizes each eye image and
    reduces it to a small 3 x 6 brightness grid. This class does not predict gaze;
    it only produces measurements for the ML model.
    """

    def __init__(self):
        self.mediapipe = MediaPipeBackend()
        self.eye_analyzer = EyeAppearanceExtractor(
            canonical_width=64,
            canonical_height=32,
            grid_rows=3,
            grid_cols=6,
        )

    @staticmethod
    def _pixel_points(face_landmarks, frame_width: int, frame_height: int):
        return {
            index: (
                float(landmark.x * frame_width),
                float(landmark.y * frame_height),
            )
            for index, landmark in enumerate(face_landmarks)
        }

    @staticmethod
    def _select(points, indices):
        return [points[index] for index in indices]

    @staticmethod
    def _mean(values, default=0.0):
        finite = [float(value) for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float(default)

    def process(self, frame) -> TrackingResult:
        start = time.perf_counter()
        result = self.mediapipe.process(frame)

        if not result.valid or not result.landmarks:
            result.method_name = "mediapipe_with_eye_appearance"
            result.latency_ms = (time.perf_counter() - start) * 1000.0
            return result

        try:
            face_landmarks = result.landmarks[0]
            frame_height, frame_width = frame.shape[:2]
            points = self._pixel_points(face_landmarks, frame_width, frame_height)

            left = self.eye_analyzer.extract(
                frame=frame,
                contour_points=self._select(
                    points, MediaPipeBackend.LEFT_EYE_CONTOUR_INDICES
                ),
                corner_points=self._select(
                    points, MediaPipeBackend.LEFT_EYE_CORNERS
                ),
            )
            right = self.eye_analyzer.extract(
                frame=frame,
                contour_points=self._select(
                    points, MediaPipeBackend.RIGHT_EYE_CONTOUR_INDICES
                ),
                corner_points=self._select(
                    points, MediaPipeBackend.RIGHT_EYE_CORNERS
                ),
            )

            valid_grids = []
            if left.valid:
                valid_grids.append(left.grid)
            if right.valid:
                valid_grids.append(right.grid)

            if valid_grids:
                mean_grid = np.mean(np.stack(valid_grids), axis=0)
            else:
                mean_grid = np.full(18, 0.5, dtype=np.float32)

            left_right_mae = 0.0
            if left.valid and right.valid:
                left_right_mae = float(
                    np.mean(np.abs(left.grid - right.grid))
                )

            appearance_features = {
                "appearance_mean_intensity": self._mean(
                    [left.mean, right.mean]
                ),
                "appearance_std_intensity": self._mean(
                    [left.std, right.std]
                ),
                "appearance_dark_fraction": self._mean(
                    [left.dark_fraction, right.dark_fraction]
                ),
                "appearance_gradient_x": self._mean(
                    [left.gradient_x, right.gradient_x]
                ),
                "appearance_gradient_y": self._mean(
                    [left.gradient_y, right.gradient_y]
                ),
                "appearance_left_right_mae": left_right_mae,
                "appearance_both_valid": bool(
                    left.valid and right.valid
                ),
            }
            for index, value in enumerate(mean_grid):
                appearance_features[f"appearance_grid_{index:02d}"] = float(value)

            result.features.update(appearance_features)
            result.method_name = "mediapipe_with_eye_appearance"
            result.error = None

            annotated = result.annotated_frame
            if annotated is not None:
                cv2.putText(
                    annotated,
                    "Eye appearance grid: ON",
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                )

        except (KeyError, ValueError, cv2.error, np.linalg.LinAlgError) as error:
            result.valid = False
            result.error = f"Eye appearance extraction failed: {error}"

        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result
