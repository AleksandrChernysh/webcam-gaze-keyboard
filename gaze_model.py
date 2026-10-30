"""One simple ML model for the whole project: multi-output Ridge regression."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RIDGE_ALPHA


# Fixed feature set. The model receives one row with these measurements and
# predicts two values at once: [screen_x, screen_y].
FEATURES = [
    # Eye geometry from MediaPipe.
    "mean_iris_relative_x",
    "mean_iris_relative_y",
    "left_iris_relative_x",
    "right_iris_relative_x",
    "left_iris_relative_y",
    "right_iris_relative_y",
    "mean_eye_aspect",
    # Head position and orientation.
    "head_center_x",
    "head_center_y",
    "head_width",
    "head_height",
    "nose_relative_x",
    "nose_relative_y",
    "head_pose_pitch_deg",
    "head_pose_yaw_deg",
    "head_pose_roll_deg",
    # Compact appearance description of the normalized eye images.
    "appearance_mean_intensity",
    "appearance_std_intensity",
    "appearance_dark_fraction",
    "appearance_gradient_x",
    "appearance_gradient_y",
    "appearance_left_right_mae",
    *[f"appearance_grid_{index:02d}" for index in range(18)],
]

OUTLIER_FEATURES = [
    "mean_iris_relative_x",
    "mean_iris_relative_y",
    "left_iris_relative_x",
    "right_iris_relative_x",
    "left_iris_relative_y",
    "right_iris_relative_y",
    "mean_eye_aspect",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _remove_trial_outliers(rows: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Remove frames far from the median of the same fixation trial."""
    kept_groups = []
    group_columns = ["calibration_round", "target_name"]

    for _, group in rows.groupby(group_columns, sort=False):
        keep = np.ones(len(group), dtype=bool)
        for feature in OUTLIER_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 1e-9:
                robust_z = 0.6745 * np.abs(values - median) / mad
                keep &= robust_z <= threshold
        kept_groups.append(group.loc[keep])

    if not kept_groups:
        return rows.iloc[0:0].copy()
    return pd.concat(kept_groups, ignore_index=True)


def _prepare_rows(csv_path: str | Path) -> pd.DataFrame:
    rows = pd.read_csv(csv_path)

    required = {
        "valid",
        "should_log",
        "eye_height_valid",
        "target_x",
        "target_y",
        "target_name",
        "calibration_round",
        *FEATURES,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError("Calibration CSV is missing columns: " + ", ".join(missing))

    mask = (
        _as_bool(rows["valid"])
        & _as_bool(rows["should_log"])
        & _as_bool(rows["eye_height_valid"])
    )
    rows = rows.loc[mask].copy()

    numeric_columns = FEATURES + ["target_x", "target_y", "calibration_round"]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
    rows = _remove_trial_outliers(rows)

    if len(rows) < 100:
        raise ValueError("Too few valid calibration frames after filtering.")
    if rows["target_name"].nunique() < 9:
        raise ValueError("Calibration does not cover enough screen targets.")

    return rows


def _new_model():
    # StandardScaler only normalizes feature scales. Ridge is the single ML model.
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )


def _spatial_metrics(true_xy: np.ndarray, predicted_xy: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(predicted_xy - true_xy, axis=1)
    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "sample_count": int(len(errors)),
    }


def train_gaze_model(
    csv_path: str | Path,
    screen_width: int,
    screen_height: int,
    model_path: str | Path | None = None,
):
    """Validate on round 2, then train the same Ridge model on both rounds."""
    rows = _prepare_rows(csv_path)
    round_numbers = sorted(int(value) for value in rows["calibration_round"].unique())
    if len(round_numbers) < 2:
        raise ValueError("The simplified final version requires two calibration rounds.")

    train_round = round_numbers[0]
    validation_round = round_numbers[1]
    train_rows = rows[rows["calibration_round"] == train_round]
    validation_rows = rows[rows["calibration_round"] == validation_round]

    validation_model = _new_model()
    validation_model.fit(
        train_rows[FEATURES],
        train_rows[["target_x", "target_y"]],
    )
    validation_predictions = validation_model.predict(validation_rows[FEATURES])
    validation_metrics = _spatial_metrics(
        validation_rows[["target_x", "target_y"]].to_numpy(dtype=float),
        np.asarray(validation_predictions, dtype=float),
    )

    # Final model uses all calibration data after the validation check.
    final_model = _new_model()
    final_model.fit(rows[FEATURES], rows[["target_x", "target_y"]])

    if model_path is None:
        model_path = Path(__file__).resolve().parent / "models" / "gaze_model.joblib"
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_model,
        "features": FEATURES,
        "model_name": "StandardScaler + multi-output Ridge Regression",
        "validation_metrics": validation_metrics,
        "screen_width": int(screen_width),
        "screen_height": int(screen_height),
        "ridge_alpha": float(RIDGE_ALPHA),
    }
    joblib.dump(artifact, model_path)

    print("Simple Ridge validation on calibration round 2:")
    print(f"  mean:   {validation_metrics['mean_error_px']:.1f} px")
    print(f"  median: {validation_metrics['median_error_px']:.1f} px")
    print(f"  p90:    {validation_metrics['p90_error_px']:.1f} px")
    print(f"Saved model to: {model_path}")

    return (
        final_model,
        FEATURES.copy(),
        validation_metrics,
        artifact["model_name"],
    )


def load_gaze_model(
    model_path: str | Path,
    expected_width: int,
    expected_height: int,
):
    artifact = joblib.load(model_path)
    required = {"model", "features", "model_name", "validation_metrics"}
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError("Saved model is missing fields: " + ", ".join(missing))

    saved_width = artifact.get("screen_width")
    saved_height = artifact.get("screen_height")
    if saved_width != expected_width or saved_height != expected_height:
        raise ValueError(
            "Saved model screen size does not match config.py: "
            f"saved={saved_width}x{saved_height}, "
            f"current={expected_width}x{expected_height}."
        )

    return (
        artifact["model"],
        list(artifact["features"]),
        dict(artifact["validation_metrics"]),
        str(artifact["model_name"]),
    )
