from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class EyeAppearance:
    valid: bool
    grid: np.ndarray
    mean: float
    std: float
    dark_fraction: float
    gradient_x: float
    gradient_y: float


class EyeAppearanceExtractor:
    """Normalize an eye image and reduce it to a small brightness descriptor."""

    def __init__(
        self,
        canonical_width: int = 64,
        canonical_height: int = 32,
        grid_rows: int = 3,
        grid_cols: int = 6,
    ):
        self.width = int(canonical_width)
        self.height = int(canonical_height)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    @property
    def grid_size(self) -> int:
        return self.grid_rows * self.grid_cols

    def _empty(self) -> EyeAppearance:
        return EyeAppearance(
            valid=False,
            grid=np.full(self.grid_size, 0.5, dtype=np.float32),
            mean=0.0,
            std=0.0,
            dark_fraction=0.0,
            gradient_x=0.0,
            gradient_y=0.0,
        )

    @staticmethod
    def _points(values: Iterable[tuple[float, float]]) -> np.ndarray:
        points = np.asarray(list(values), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Eye points must have shape (n, 2).")
        return points

    def _normalize_eye(self, frame, contour_values, corner_values):
        contour = self._points(contour_values)
        corners = self._points(corner_values)
        if len(contour) < 6 or len(corners) != 2:
            return None

        left_corner, right_corner = sorted(corners, key=lambda point: point[0])
        left_corner = np.asarray(left_corner, dtype=np.float32)
        right_corner = np.asarray(right_corner, dtype=np.float32)

        horizontal = right_corner - left_corner
        eye_width = float(np.linalg.norm(horizontal))
        if eye_width < 8.0:
            return None
        horizontal /= eye_width

        vertical = np.array([-horizontal[1], horizontal[0]], dtype=np.float32)
        if vertical[1] < 0:
            vertical *= -1.0

        vertical_positions = (contour - left_corner) @ vertical
        top = float(np.min(vertical_positions))
        bottom = float(np.max(vertical_positions))
        eye_height = bottom - top
        if eye_height < 3.0:
            return None

        horizontal_pad = 0.08 * eye_width
        vertical_pad = max(1.5, 0.32 * eye_height)
        source = np.asarray(
            [
                left_corner - horizontal * horizontal_pad + vertical * (top - vertical_pad),
                right_corner + horizontal * horizontal_pad + vertical * (top - vertical_pad),
                right_corner + horizontal * horizontal_pad + vertical * (bottom + vertical_pad),
                left_corner - horizontal * horizontal_pad + vertical * (bottom + vertical_pad),
            ],
            dtype=np.float32,
        )
        destination = np.asarray(
            [
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(source, destination)
        eye_image = cv2.warpPerspective(
            frame,
            transform,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        transformed_contour = cv2.perspectiveTransform(
            contour.reshape(1, -1, 2), transform
        )[0]
        transformed_contour = np.round(transformed_contour).astype(np.int32)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [transformed_contour], 255)
        if int(np.count_nonzero(mask)) < 80:
            return None

        return eye_image, mask

    def extract(self, frame, contour_points, corner_points) -> EyeAppearance:
        normalized = self._normalize_eye(frame, contour_points, corner_points)
        if normalized is None:
            return self._empty()

        eye_bgr, mask = normalized
        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        gray[mask == 0] = 255
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        pixels = gray[mask > 0].astype(np.float32)
        if pixels.size < 80:
            return self._empty()

        mean = float(np.mean(pixels) / 255.0)
        std = float(np.std(pixels) / 255.0)
        dark_threshold = float(np.percentile(pixels, 25))
        dark_fraction = float(np.mean(pixels <= dark_threshold))

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_x = float(np.mean(np.abs(sobel_x[mask > 0])) / 255.0)
        gradient_y = float(np.mean(np.abs(sobel_y[mask > 0])) / 255.0)

        pixel_mean = float(np.mean(pixels))
        pixel_std = max(float(np.std(pixels)), 1.0)
        normalized_pixels = (gray.astype(np.float32) - pixel_mean) / pixel_std
        normalized_pixels = np.clip(normalized_pixels, -2.5, 2.5)
        normalized_pixels = (normalized_pixels + 2.5) / 5.0
        normalized_pixels[mask == 0] = 0.5

        grid = cv2.resize(
            normalized_pixels,
            (self.grid_cols, self.grid_rows),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1)

        return EyeAppearance(
            valid=True,
            grid=grid.astype(np.float32),
            mean=mean,
            std=std,
            dark_fraction=dark_fraction,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )
