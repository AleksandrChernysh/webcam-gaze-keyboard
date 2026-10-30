from pathlib import Path
from datetime import datetime
import time

import pandas as pd

from backends.base import TrackingResult


class FrameLogger:
    def __init__(self, output_dir="data"):
        project_dir = Path(__file__).resolve().parent
        self.output_dir = project_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rows = []
        self.start_time = time.perf_counter()
        self.frame_id = 0

    def log_result(
        self,
        tracking_result: TrackingResult,
        target=None,
        should_log=True,
    ):
        timestamp_sec = time.perf_counter() - self.start_time

        num_faces = 0
        if tracking_result.landmarks is not None:
            num_faces = len(tracking_result.landmarks)

        row = {
            "frame_id": self.frame_id,
            "timestamp_sec": timestamp_sec,
            "method_name": tracking_result.method_name,
            "valid": bool(tracking_result.valid),
            "num_faces": num_faces,
            "latency_ms": tracking_result.latency_ms,
            "error": tracking_result.error,
            "target_name": target.name if target is not None else None,
            "target_x": target.x if target is not None else None,
            "target_y": target.y if target is not None else None,
            "calibration_round": (
                target.calibration_round
                if target is not None
                else None
            ),
            "trial_index": (
                target.trial_index
                if target is not None
                else None
            ),
            "should_log": bool(should_log),
        }

        for feature_name, feature_value in tracking_result.features.items():
            row[feature_name] = feature_value

        self.rows.append(row)
        self.frame_id += 1

    def save(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_log_{timestamp}.csv"

        output_path = self.output_dir / filename

        df = pd.DataFrame(self.rows)
        df.to_csv(output_path, index=False)

        return output_path
