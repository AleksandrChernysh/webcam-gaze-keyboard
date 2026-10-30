from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationTarget:
    name: str
    x: int
    y: int
    calibration_round: int
    trial_index: int


class CalibrationSession:
    """15-point calibration repeated a fixed number of times.

    The timer advances only while tracking quality is acceptable. A target is
    repeated automatically when too few valid samples were recorded.
    """

    def __init__(
        self,
        width: int,
        height: int,
        x_levels: tuple[float, ...],
        y_levels: tuple[float, ...],
        rounds: int,
        settle_sec: float,
        record_sec: float,
        min_valid_samples: int,
    ) -> None:
        if len(x_levels) != 3 or len(y_levels) != 5:
            raise ValueError("Calibration requires 3 X levels and 5 Y levels.")
        if rounds < 1:
            raise ValueError("Calibration must contain at least one round.")

        self.width = int(width)
        self.height = int(height)
        self.rounds = int(rounds)
        self.settle_sec = float(settle_sec)
        self.record_sec = float(record_sec)
        self.min_valid_samples = int(min_valid_samples)

        positions = self._build_positions(x_levels, y_levels)
        orders = self._build_orders(list(positions), self.rounds)

        self.targets: list[CalibrationTarget] = []
        for round_number, order in enumerate(orders, start=1):
            for trial_index, name in enumerate(order, start=1):
                x, y = positions[name]
                self.targets.append(
                    CalibrationTarget(
                        name=name,
                        x=x,
                        y=y,
                        calibration_round=round_number,
                        trial_index=trial_index,
                    )
                )

        self.targets_per_round = len(positions)
        self.current_index = 0
        self.active_elapsed = 0.0
        self.valid_sample_count = 0
        self.last_update_time = time.perf_counter()
        self.paused = False
        self.completed_round: int | None = None
        self.repeat_message = ""
        self.repeat_message_until = 0.0

    def _build_positions(
        self,
        x_levels: tuple[float, ...],
        y_levels: tuple[float, ...],
    ) -> dict[str, tuple[int, int]]:
        row_names = ("top", "upper", "middle", "lower", "bottom")
        col_names = ("left", "center", "right")
        positions: dict[str, tuple[int, int]] = {}

        for row_name, y_fraction in zip(row_names, y_levels):
            for col_name, x_fraction in zip(col_names, x_levels):
                if not 0.0 < x_fraction < 1.0 or not 0.0 < y_fraction < 1.0:
                    raise ValueError("Calibration fractions must be between 0 and 1.")
                name = (
                    "center"
                    if row_name == "middle" and col_name == "center"
                    else f"{row_name}_{col_name}"
                )
                positions[name] = (
                    int(round(self.width * x_fraction)),
                    int(round(self.height * y_fraction)),
                )
        return positions

    @staticmethod
    def _build_orders(names: list[str], rounds: int) -> list[list[str]]:
        orders: list[list[str]] = []
        for round_index in range(rounds):
            order = names.copy()
            np.random.default_rng(21 * (round_index + 1)).shuffle(order)
            if order[0] == "center":
                order[0], order[1] = order[1], order[0]
            orders.append(order)
        return orders

    def current_target(self) -> CalibrationTarget | None:
        if self.is_finished():
            return None
        return self.targets[self.current_index]

    def is_finished(self) -> bool:
        return self.current_index >= len(self.targets)

    def is_paused(self) -> bool:
        return self.paused

    def should_record(self) -> bool:
        return (
            not self.paused
            and not self.is_finished()
            and self.active_elapsed >= self.settle_sec
        )

    def record_valid_sample(self) -> None:
        if self.should_record():
            self.valid_sample_count += 1

    def _reset_target(self) -> None:
        self.active_elapsed = 0.0
        self.valid_sample_count = 0
        self.last_update_time = time.perf_counter()

    def resume(self) -> None:
        if self.paused:
            self.paused = False
            self.completed_round = None
            self._reset_target()

    def update(self, quality_ok: bool) -> None:
        now = time.perf_counter()
        delta = max(0.0, min(now - self.last_update_time, 0.25))
        self.last_update_time = now

        if self.paused or self.is_finished():
            return
        if quality_ok:
            self.active_elapsed += delta

        if self.active_elapsed < self.settle_sec + self.record_sec:
            return

        if self.valid_sample_count < self.min_valid_samples:
            self.repeat_message = (
                f"Only {self.valid_sample_count}/{self.min_valid_samples} valid samples"
            )
            self.repeat_message_until = time.perf_counter() + 1.2
            self._reset_target()
            return

        previous = self.current_target()
        self.current_index += 1
        if self.is_finished():
            return

        current = self.current_target()
        if (
            previous is not None
            and current is not None
            and previous.calibration_round != current.calibration_round
        ):
            self.paused = True
            self.completed_round = previous.calibration_round
        else:
            self._reset_target()

    def draw(self, quality_ok: bool = True, status_message: str = "") -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        if self.paused:
            cv2.putText(
                canvas,
                f"Round {self.completed_round} finished",
                (485, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                canvas,
                "Blink normally, then press SPACE",
                (405, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
            return canvas

        target = self.current_target()
        if target is None:
            return canvas

        cv2.circle(canvas, (target.x, target.y), 15, (0, 0, 255), -1)
        cv2.circle(canvas, (target.x, target.y), 30, (0, 0, 0), 2)

        total_trial = self.current_index + 1
        total_trials = len(self.targets)
        cv2.putText(
            canvas,
            (
                f"Round {target.calibration_round}/{self.rounds}  "
                f"Target {target.trial_index}/{self.targets_per_round}  "
                f"Total {total_trial}/{total_trials}"
            ),
            (30, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            "Keep your head still and look at the red point",
            (30, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 40, 40),
            2,
        )

        now = time.perf_counter()
        if now < self.repeat_message_until:
            text = f"REPEATING: {self.repeat_message}"
            color = (0, 0, 255)
        elif not quality_ok:
            text = status_message or "Face or eyes are not stable"
            color = (0, 0, 255)
        elif not self.should_record():
            remaining = max(0.0, self.settle_sec - self.active_elapsed)
            text = f"Move eyes to the point: {remaining:.1f}s"
            color = (0, 140, 220)
        else:
            remaining = max(
                0.0,
                self.settle_sec + self.record_sec - self.active_elapsed,
            )
            text = f"Recording: {remaining:.1f}s"
            color = (0, 150, 0)

        cv2.putText(
            canvas,
            text,
            (30, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
        )
        cv2.putText(
            canvas,
            f"Valid samples: {self.valid_sample_count}/{self.min_valid_samples}",
            (30, 174),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (50, 50, 50),
            2,
        )
        return canvas
