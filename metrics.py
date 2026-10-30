from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenGeometry:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    viewing_distance_mm: float

    def validate(self) -> None:
        values = {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "viewing_distance_mm": self.viewing_distance_mm,
        }
        invalid = [name for name, value in values.items() if float(value) <= 0]
        if invalid:
            raise ValueError(
                "Screen geometry values must be positive: "
                + ", ".join(invalid)
            )


def _as_point_array(point: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(point), dtype=float)
    if result.shape != (2,):
        raise ValueError("A screen point must contain exactly x and y.")
    return result


def pixel_point_to_view_vector(
    point_px: Iterable[float],
    geometry: ScreenGeometry,
) -> np.ndarray:
    """Convert a screen pixel coordinate to a 3-D eye-to-screen vector."""
    geometry.validate()
    x_px, y_px = _as_point_array(point_px)

    x_centered_px = x_px - geometry.width_px / 2.0
    y_centered_px = y_px - geometry.height_px / 2.0

    x_mm = x_centered_px * geometry.width_mm / geometry.width_px
    y_mm = y_centered_px * geometry.height_mm / geometry.height_px

    return np.array(
        [x_mm, y_mm, geometry.viewing_distance_mm],
        dtype=float,
    )


def angular_distance_deg(
    first_point_px: Iterable[float],
    second_point_px: Iterable[float],
    geometry: ScreenGeometry,
) -> float:
    """Angle between two eye-to-screen direction vectors, in degrees."""
    first = pixel_point_to_view_vector(first_point_px, geometry)
    second = pixel_point_to_view_vector(second_point_px, geometry)

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return float("nan")

    cosine = float(np.dot(first, second) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.mean(array))


def _finite_median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.median(array))


def _finite_max(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.max(array))


def _finite_percentile(
    values: Iterable[float],
    percentile: float,
) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(np.percentile(array, percentile))


def calculate_trial_metrics(
    trial_rows: pd.DataFrame,
    geometry: ScreenGeometry,
) -> dict[str, Any]:
    """Calculate accuracy and raw-signal precision for one target trial.

    Accuracy is the distance from the mean raw prediction to the target.
    Precision is calculated before temporal smoothing. RMS-S2S only uses truly
    adjacent valid frames, so a data-loss gap is never treated as eye motion.
    """
    if trial_rows.empty:
        raise ValueError("Cannot calculate metrics for an empty trial.")

    trial_rows = trial_rows.sort_values("frame_id").copy()
    valid_mask = trial_rows["prediction_valid"].fillna(False).astype(bool)
    valid_rows = trial_rows.loc[valid_mask].copy()

    first_row = trial_rows.iloc[0]
    target = np.array(
        [float(first_row["target_x"]), float(first_row["target_y"])],
        dtype=float,
    )

    total_frames = int(len(trial_rows))
    valid_frames = int(len(valid_rows))
    valid_frame_rate = (
        100.0 * valid_frames / total_frames if total_frames else 0.0
    )

    result: dict[str, Any] = {
        "trial_id": str(first_row["trial_id"]),
        "trial_index": int(first_row["trial_index"]),
        "repetition": int(first_row["repetition"]),
        "target_name": str(first_row["target_name"]),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "valid_frame_rate_pct": float(valid_frame_rate),
        "accuracy_px": None,
        "accuracy_deg": None,
        "mean_sample_error_px": None,
        "median_sample_error_px": None,
        "p90_sample_error_px": None,
        "precision_rms_s2s_px": None,
        "precision_rms_s2s_deg": None,
        "precision_radial_rms_px": None,
        "precision_radial_rms_deg": None,
        "mean_backend_latency_ms": _finite_mean(
            trial_rows["backend_latency_ms"]
        ),
        "mean_prediction_latency_ms": _finite_mean(
            trial_rows["prediction_latency_ms"]
        ),
    }

    if valid_frames == 0:
        return result

    predictions = valid_rows[["predicted_x", "predicted_y"]].to_numpy(
        dtype=float
    )
    centroid = np.mean(predictions, axis=0)

    accuracy_px = float(np.linalg.norm(centroid - target))
    accuracy_deg = angular_distance_deg(centroid, target, geometry)

    sample_errors_px = np.linalg.norm(predictions - target, axis=1)

    radial_px = np.linalg.norm(predictions - centroid, axis=1)
    radial_deg = np.array(
        [
            angular_distance_deg(point, centroid, geometry)
            for point in predictions
        ],
        dtype=float,
    )

    s2s_px: list[float] = []
    s2s_deg: list[float] = []
    valid_frame_ids = valid_rows["frame_id"].to_numpy(dtype=int)

    for index in range(1, len(predictions)):
        if valid_frame_ids[index] - valid_frame_ids[index - 1] != 1:
            continue
        first = predictions[index - 1]
        second = predictions[index]
        s2s_px.append(float(np.linalg.norm(second - first)))
        s2s_deg.append(angular_distance_deg(first, second, geometry))

    result.update(
        {
            "mean_predicted_x": float(centroid[0]),
            "mean_predicted_y": float(centroid[1]),
            "accuracy_px": accuracy_px,
            "accuracy_deg": accuracy_deg,
            "mean_sample_error_px": float(np.mean(sample_errors_px)),
            "median_sample_error_px": float(np.median(sample_errors_px)),
            "p90_sample_error_px": float(
                np.percentile(sample_errors_px, 90)
            ),
            "precision_radial_rms_px": float(
                np.sqrt(np.mean(np.square(radial_px)))
            ),
            "precision_radial_rms_deg": float(
                np.sqrt(np.mean(np.square(radial_deg)))
            ),
        }
    )

    if s2s_px:
        result["precision_rms_s2s_px"] = float(
            np.sqrt(np.mean(np.square(s2s_px)))
        )
        result["precision_rms_s2s_deg"] = float(
            np.sqrt(np.mean(np.square(s2s_deg)))
        )

    return result


def _aggregate_target_rows(trial_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    per_target: list[dict[str, Any]] = []
    for target_name, group in trial_metrics.groupby("target_name", sort=False):
        per_target.append(
            {
                "target_name": str(target_name),
                "repetitions": int(len(group)),
                "mean_accuracy_px": _finite_mean(group["accuracy_px"]),
                "mean_accuracy_deg": _finite_mean(group["accuracy_deg"]),
                "mean_precision_rms_s2s_deg": _finite_mean(
                    group["precision_rms_s2s_deg"]
                ),
                "mean_precision_radial_rms_deg": _finite_mean(
                    group["precision_radial_rms_deg"]
                ),
                "mean_valid_frame_rate_pct": _finite_mean(
                    group["valid_frame_rate_pct"]
                ),
            }
        )
    return per_target


def calculate_verification_metrics(
    rows: pd.DataFrame | list[dict[str, Any]],
    geometry: ScreenGeometry,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a JSON-ready summary and one row per verification trial."""
    samples = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("Verification produced no recorded samples.")

    required_columns = {
        "trial_id",
        "trial_index",
        "repetition",
        "target_name",
        "target_x",
        "target_y",
        "frame_id",
        "prediction_valid",
        "predicted_x",
        "predicted_y",
        "backend_latency_ms",
        "prediction_latency_ms",
    }
    missing = required_columns.difference(samples.columns)
    if missing:
        raise ValueError(
            "Verification rows are missing columns: "
            + ", ".join(sorted(missing))
        )

    trial_records = [
        calculate_trial_metrics(group, geometry)
        for _, group in samples.groupby("trial_id", sort=False)
    ]
    trial_metrics = pd.DataFrame(trial_records)

    prediction_valid = samples["prediction_valid"].fillna(False).astype(bool)
    total_frames = int(len(samples))
    valid_frames = int(prediction_valid.sum())

    sample_errors = samples.loc[
        prediction_valid,
        "sample_error_px",
    ].to_numpy(dtype=float)

    summary: dict[str, Any] = {
        "number_of_trials": int(len(trial_metrics)),
        "number_of_targets": int(trial_metrics["target_name"].nunique()),
        "total_recorded_frames": total_frames,
        "valid_prediction_frames": valid_frames,
        "valid_frame_rate_pct": (
            100.0 * valid_frames / total_frames if total_frames else 0.0
        ),
        "mean_accuracy_px": _finite_mean(trial_metrics["accuracy_px"]),
        "median_accuracy_px": _finite_median(trial_metrics["accuracy_px"]),
        "mean_accuracy_deg": _finite_mean(trial_metrics["accuracy_deg"]),
        "median_accuracy_deg": _finite_median(trial_metrics["accuracy_deg"]),
        "worst_trial_accuracy_px": _finite_max(
            trial_metrics["accuracy_px"]
        ),
        "worst_trial_accuracy_deg": _finite_max(
            trial_metrics["accuracy_deg"]
        ),
        "mean_precision_rms_s2s_px": _finite_mean(
            trial_metrics["precision_rms_s2s_px"]
        ),
        "mean_precision_rms_s2s_deg": _finite_mean(
            trial_metrics["precision_rms_s2s_deg"]
        ),
        "mean_precision_radial_rms_px": _finite_mean(
            trial_metrics["precision_radial_rms_px"]
        ),
        "mean_precision_radial_rms_deg": _finite_mean(
            trial_metrics["precision_radial_rms_deg"]
        ),
        "mean_sample_error_px": _finite_mean(sample_errors),
        "median_sample_error_px": _finite_median(sample_errors),
        "p90_sample_error_px": _finite_percentile(sample_errors, 90),
        "mean_backend_latency_ms": _finite_mean(
            samples["backend_latency_ms"]
        ),
        "mean_prediction_latency_ms": _finite_mean(
            samples["prediction_latency_ms"]
        ),
        "per_target": _aggregate_target_rows(trial_metrics),
        "geometry": {
            "width_px": int(geometry.width_px),
            "height_px": int(geometry.height_px),
            "width_mm": float(geometry.width_mm),
            "height_mm": float(geometry.height_mm),
            "viewing_distance_mm": float(geometry.viewing_distance_mm),
        },
    }

    return summary, trial_metrics
