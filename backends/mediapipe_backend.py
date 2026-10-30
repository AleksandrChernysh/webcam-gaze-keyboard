from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backends.base import TrackerBackend, TrackingResult


class MediaPipeBackend(TrackerBackend):
    """Extract eye-gaze and head-pose features with MediaPipe."""

    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    LEFT_EYE_CONTOUR_INDICES = [
        246, 161, 160, 159, 158, 157, 173,
        33, 7, 163, 144, 145, 153, 154, 155, 133,
    ]
    RIGHT_EYE_CONTOUR_INDICES = [
        466, 388, 387, 386, 385, 384, 398,
        263, 249, 390, 373, 374, 380, 381, 382, 362,
    ]

    LEFT_UPPER_EYELID_INDICES = [161, 160, 159, 158, 157]
    LEFT_LOWER_EYELID_INDICES = [163, 144, 145, 153, 154, 155]
    RIGHT_UPPER_EYELID_INDICES = [388, 387, 386, 385, 384]
    RIGHT_LOWER_EYELID_INDICES = [390, 373, 374, 380, 381, 382]

    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (263, 362)
    NOSE_TIP_INDEX = 1

    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model was not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

    @staticmethod
    def _mean_point(points, indices):
        count = len(indices)
        return (
            sum(points[index][0] for index in indices) / count,
            sum(points[index][1] for index in indices) / count,
        )

    @staticmethod
    def _distance(point_a, point_b):
        return math.hypot(
            point_b[0] - point_a[0],
            point_b[1] - point_a[1],
        )

    @staticmethod
    def _extract_head_pose(mp_result):
        matrices = getattr(
            mp_result,
            "facial_transformation_matrixes",
            None,
        )

        if not matrices:
            raise ValueError(
                "MediaPipe did not return a facial transformation matrix."
            )

        matrix = np.asarray(matrices[0], dtype=np.float64)

        if matrix.size != 16:
            raise ValueError(
                f"Unexpected face transformation matrix shape: "
                f"{matrix.shape}"
            )

        matrix = matrix.reshape(4, 4)

        # The 3x3 block can contain a small scale component. SVD projects it
        # onto the nearest proper rotation matrix before Euler decomposition.
        rotation_with_scale = matrix[:3, :3]
        u, _, vt = np.linalg.svd(rotation_with_scale)
        rotation_matrix = u @ vt

        if np.linalg.det(rotation_matrix) < 0:
            u[:, -1] *= -1
            rotation_matrix = u @ vt

        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        pitch_deg, yaw_deg, roll_deg = (
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
        )

        translation = matrix[:3, 3]

        return {
            "head_pose_pitch_deg": pitch_deg,
            "head_pose_yaw_deg": yaw_deg,
            "head_pose_roll_deg": roll_deg,
            "head_pose_tx": float(translation[0]),
            "head_pose_ty": float(translation[1]),
            "head_pose_tz": float(translation[2]),
        }

    @classmethod
    def _eye_geometry(
        cls,
        pixel_points,
        iris_indices,
        corner_indices,
        upper_eyelid_indices,
        lower_eyelid_indices,
    ):
        iris_center = cls._mean_point(pixel_points, iris_indices)

        corner_a = pixel_points[corner_indices[0]]
        corner_b = pixel_points[corner_indices[1]]

        image_left_corner, image_right_corner = sorted(
            (corner_a, corner_b),
            key=lambda point: point[0],
        )

        eye_width = cls._distance(
            image_left_corner,
            image_right_corner,
        )

        if eye_width <= 1e-6:
            raise ValueError("Eye width is zero or too small.")

        horizontal_x = (
            image_right_corner[0] - image_left_corner[0]
        ) / eye_width
        horizontal_y = (
            image_right_corner[1] - image_left_corner[1]
        ) / eye_width

        vertical_x = -horizontal_y
        vertical_y = horizontal_x

        if vertical_y < 0:
            vertical_x *= -1
            vertical_y *= -1

        def project_horizontal(point):
            relative_x = point[0] - image_left_corner[0]
            relative_y = point[1] - image_left_corner[1]
            return (
                relative_x * horizontal_x
                + relative_y * horizontal_y
            )

        def project_vertical(point):
            relative_x = point[0] - image_left_corner[0]
            relative_y = point[1] - image_left_corner[1]
            return (
                relative_x * vertical_x
                + relative_y * vertical_y
            )

        upper_center = cls._mean_point(
            pixel_points,
            upper_eyelid_indices,
        )
        lower_center = cls._mean_point(
            pixel_points,
            lower_eyelid_indices,
        )

        upper_position = project_vertical(upper_center)
        lower_position = project_vertical(lower_center)

        eye_top = min(upper_position, lower_position)
        eye_bottom = max(upper_position, lower_position)
        eye_height = eye_bottom - eye_top

        if eye_height <= 1e-6:
            raise ValueError("Eye height is zero or too small.")

        iris_horizontal = project_horizontal(iris_center)
        iris_vertical = project_vertical(iris_center)

        iris_relative_x = iris_horizontal / eye_width
        iris_relative_y = (
            iris_vertical - eye_top
        ) / eye_height

        return {
            "iris_center_x": iris_center[0],
            "iris_center_y": iris_center[1],
            "eye_width": eye_width,
            "eye_height": eye_height,
            "iris_relative_x": iris_relative_x,
            "iris_relative_y": iris_relative_y,
            "eye_aspect": eye_height / eye_width,
        }

    def process(self, frame) -> TrackingResult:
        start_time = time.perf_counter()
        annotated_frame = frame.copy()
        face_landmarks_result = None
        features = {}
        error = None
        valid = False

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            mp_result = self.landmarker.detect(mp_image)
            face_landmarks_result = mp_result.face_landmarks

            if not face_landmarks_result:
                raise ValueError("No face detected.")

            face_landmarks = face_landmarks_result[0]
            frame_height, frame_width, _ = annotated_frame.shape

            normalized_points = {
                index: (landmark.x, landmark.y)
                for index, landmark in enumerate(face_landmarks)
            }
            pixel_points = {
                index: (
                    landmark.x * frame_width,
                    landmark.y * frame_height,
                )
                for index, landmark in enumerate(face_landmarks)
            }

            selected_indices = set(
                self.LEFT_IRIS_INDICES
                + self.RIGHT_IRIS_INDICES
                + self.LEFT_EYE_CONTOUR_INDICES
                + self.RIGHT_EYE_CONTOUR_INDICES
            )

            for index in selected_indices:
                draw_x = int(round(pixel_points[index][0]))
                draw_y = int(round(pixel_points[index][1]))
                cv2.circle(
                    annotated_frame,
                    (draw_x, draw_y),
                    1,
                    (0, 255, 0),
                    -1,
                )

            left_eye = self._eye_geometry(
                pixel_points=pixel_points,
                iris_indices=self.LEFT_IRIS_INDICES,
                corner_indices=self.LEFT_EYE_CORNERS,
                upper_eyelid_indices=self.LEFT_UPPER_EYELID_INDICES,
                lower_eyelid_indices=self.LEFT_LOWER_EYELID_INDICES,
            )
            right_eye = self._eye_geometry(
                pixel_points=pixel_points,
                iris_indices=self.RIGHT_IRIS_INDICES,
                corner_indices=self.RIGHT_EYE_CORNERS,
                upper_eyelid_indices=self.RIGHT_UPPER_EYELID_INDICES,
                lower_eyelid_indices=self.RIGHT_LOWER_EYELID_INDICES,
            )

            mean_iris_relative_x = (
                left_eye["iris_relative_x"]
                + right_eye["iris_relative_x"]
            ) / 2
            mean_iris_relative_y = (
                left_eye["iris_relative_y"]
                + right_eye["iris_relative_y"]
            ) / 2
            mean_eye_height = (
                left_eye["eye_height"]
                + right_eye["eye_height"]
            ) / 2
            mean_eye_aspect = (
                left_eye["eye_aspect"]
                + right_eye["eye_aspect"]
            ) / 2

            all_x = [point[0] for point in normalized_points.values()]
            all_y = [point[1] for point in normalized_points.values()]

            face_min_x = min(all_x)
            face_max_x = max(all_x)
            face_min_y = min(all_y)
            face_max_y = max(all_y)

            head_center_x = (face_min_x + face_max_x) / 2
            head_center_y = (face_min_y + face_max_y) / 2
            head_width = face_max_x - face_min_x
            head_height = face_max_y - face_min_y

            left_eye_center_norm = self._mean_point(
                normalized_points,
                self.LEFT_EYE_CORNERS,
            )
            right_eye_center_norm = self._mean_point(
                normalized_points,
                self.RIGHT_EYE_CORNERS,
            )

            image_left_eye, image_right_eye = sorted(
                (left_eye_center_norm, right_eye_center_norm),
                key=lambda point: point[0],
            )

            inter_eye_distance = self._distance(
                image_left_eye,
                image_right_eye,
            )

            if inter_eye_distance <= 1e-6:
                raise ValueError(
                    "Inter-eye distance is zero or too small."
                )

            eye_midpoint = (
                (image_left_eye[0] + image_right_eye[0]) / 2,
                (image_left_eye[1] + image_right_eye[1]) / 2,
            )
            nose_tip = normalized_points[self.NOSE_TIP_INDEX]

            nose_relative_x = (
                nose_tip[0] - eye_midpoint[0]
            ) / inter_eye_distance
            nose_relative_y = (
                nose_tip[1] - eye_midpoint[1]
            ) / inter_eye_distance

            head_roll_2d = math.atan2(
                image_right_eye[1] - image_left_eye[1],
                image_right_eye[0] - image_left_eye[0],
            )

            head_pose = self._extract_head_pose(mp_result)

            features = {
                "left_iris_center_x": left_eye["iris_center_x"],
                "left_iris_center_y": left_eye["iris_center_y"],
                "right_iris_center_x": right_eye["iris_center_x"],
                "right_iris_center_y": right_eye["iris_center_y"],
                "left_eye_width": left_eye["eye_width"],
                "right_eye_width": right_eye["eye_width"],
                "left_iris_relative_x": left_eye["iris_relative_x"],
                "right_iris_relative_x": right_eye["iris_relative_x"],
                "mean_iris_relative_x": mean_iris_relative_x,
                "left_eye_height": left_eye["eye_height"],
                "right_eye_height": right_eye["eye_height"],
                "left_iris_relative_y": left_eye["iris_relative_y"],
                "right_iris_relative_y": right_eye["iris_relative_y"],
                "mean_iris_relative_y": mean_iris_relative_y,
                "mean_eye_height": mean_eye_height,
                "eye_height_valid": mean_eye_height >= 6.0,
                "left_eye_aspect": left_eye["eye_aspect"],
                "right_eye_aspect": right_eye["eye_aspect"],
                "mean_eye_aspect": mean_eye_aspect,
                "head_center_x": head_center_x,
                "head_center_y": head_center_y,
                "head_width": head_width,
                "head_height": head_height,
                "inter_eye_distance": inter_eye_distance,
                "nose_relative_x": nose_relative_x,
                "nose_relative_y": nose_relative_y,
                "head_roll": head_roll_2d,
                **head_pose,
            }

            cv2.putText(
                annotated_frame,
                (
                    f"P:{head_pose['head_pose_pitch_deg']:.1f} "
                    f"Y:{head_pose['head_pose_yaw_deg']:.1f} "
                    f"R:{head_pose['head_pose_roll_deg']:.1f}"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            valid = True

        except Exception as exception:
            error = str(exception)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return TrackingResult(
            method_name="mediapipe",
            valid=valid,
            features=features,
            annotated_frame=annotated_frame,
            landmarks=face_landmarks_result,
            confidence=None,
            latency_ms=latency_ms,
            error=error,
        )
