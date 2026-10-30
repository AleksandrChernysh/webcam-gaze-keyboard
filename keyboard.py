from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import time
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class KeyboardKey:
    key_id: str
    label: str
    action: str
    value: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class VirtualKeyboard:
    """Large two-stage gaze keyboard with dwell-time selection.

    Stage 1 selects a group such as ABC. Stage 2 selects one character
    from that group. Large targets are intentionally used because a webcam
    gaze estimator is not precise enough for a dense QWERTY layout.
    """

    GROUPS = [
        ("ABC", "ABC"),
        ("DEF", "DEF"),
        ("GHI", "GHI"),
        ("JKL", "JKL"),
        ("MNO", "MNO"),
        ("PQR", "PQR"),
        ("STU", "STU"),
        ("VWX", "VWX"),
        ("YZ+", "YZ+"),
    ]

    def __init__(
        self,
        width: int,
        height: int,
        dwell_seconds: float = 1.0,
        rearm_distance_px: float = 75.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.dwell_seconds = float(dwell_seconds)
        self.rearm_distance_px = float(rearm_distance_px)

        self.mode = "groups"
        self.active_group: str | None = None
        self.text = ""
        self.last_submitted_text = ""

        self.hovered_key_id: str | None = None
        self.hover_started_at: float | None = None
        self.hover_progress = 0.0

        self.awaiting_rearm = False
        self.rearm_origin: np.ndarray | None = None

        self.flash_message = ""
        self.flash_until = 0.0
        self.events: list[dict[str, object]] = []

    def _grid_rectangles(
        self,
        rows: int,
        columns: int,
        top: int = 250,
        bottom: int | None = None,
        left: int = 150,
        right: int | None = None,
        gap: int = 18,
    ) -> list[tuple[int, int, int, int]]:
        if bottom is None:
            bottom = self.height - 80
        if right is None:
            right = self.width - 150

        total_width = right - left
        total_height = bottom - top
        cell_width = (total_width - gap * (columns - 1)) / columns
        cell_height = (total_height - gap * (rows - 1)) / rows

        rectangles: list[tuple[int, int, int, int]] = []
        for row in range(rows):
            for column in range(columns):
                x1 = int(left + column * (cell_width + gap))
                y1 = int(top + row * (cell_height + gap))
                x2 = int(x1 + cell_width)
                y2 = int(y1 + cell_height)
                rectangles.append((x1, y1, x2, y2))
        return rectangles

    def _group_keys(self) -> list[KeyboardKey]:
        rectangles = self._grid_rectangles(rows=3, columns=3)
        keys: list[KeyboardKey] = []

        for (label, value), rect in zip(self.GROUPS, rectangles):
            x1, y1, x2, y2 = rect
            keys.append(
                KeyboardKey(
                    key_id=f"group:{value}",
                    label=label,
                    action="group",
                    value=value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return keys

    def _letter_keys(self) -> list[KeyboardKey]:
        if self.active_group is None:
            return []

        if self.active_group == "YZ+":
            items = [
                ("Y", "character", "Y"),
                ("Z", "character", "Z"),
                ("SPACE", "space", " "),
                ("BACKSPACE", "backspace", ""),
                ("CLEAR", "clear", ""),
                ("ENTER", "enter", ""),
                ("BACK", "back", ""),
            ]
            rectangles = self._grid_rectangles(rows=3, columns=3)
        else:
            items = [
                (character, "character", character)
                for character in self.active_group
            ]
            # Three huge character buttons plus one wide BACK button.
            top_rectangles = self._grid_rectangles(
                rows=1,
                columns=3,
                top=270,
                bottom=610,
            )
            back_rectangle = self._grid_rectangles(
                rows=1,
                columns=1,
                top=645,
                bottom=self.height - 80,
            )[0]
            rectangles = top_rectangles + [back_rectangle]
            items.append(("BACK", "back", ""))

        keys: list[KeyboardKey] = []
        for index, (label, action, value) in enumerate(items):
            x1, y1, x2, y2 = rectangles[index]
            keys.append(
                KeyboardKey(
                    key_id=f"{action}:{label}",
                    label=label,
                    action=action,
                    value=value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return keys

    def current_keys(self) -> list[KeyboardKey]:
        if self.mode == "groups":
            return self._group_keys()
        return self._letter_keys()

    def _key_at(self, gaze_point: tuple[float, float]) -> KeyboardKey | None:
        gaze_x, gaze_y = gaze_point
        for key in self.current_keys():
            if key.contains(gaze_x, gaze_y):
                return key
        return None

    def _reset_hover(self) -> None:
        self.hovered_key_id = None
        self.hover_started_at = None
        self.hover_progress = 0.0

    def _set_flash(self, message: str, duration: float = 0.9) -> None:
        self.flash_message = message
        self.flash_until = time.perf_counter() + duration

    def _log_selection(
        self,
        key: KeyboardKey,
        gaze_point: tuple[float, float],
    ) -> None:
        self.events.append(
            {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "mode": self.mode,
                "key_id": key.key_id,
                "label": key.label,
                "action": key.action,
                "gaze_x": round(float(gaze_point[0]), 1),
                "gaze_y": round(float(gaze_point[1]), 1),
                "current_text": self.text,
            }
        )

    def _activate_key(
        self,
        key: KeyboardKey,
        gaze_point: tuple[float, float],
    ) -> None:
        if key.action == "group":
            self.mode = "letters"
            self.active_group = key.value
            self._set_flash(f"Group {key.label}")
        elif key.action == "character":
            self.text += key.value
            self.mode = "groups"
            self.active_group = None
            self._set_flash(f"Typed {key.label}")
        elif key.action == "space":
            self.text += " "
            self.mode = "groups"
            self.active_group = None
            self._set_flash("SPACE")
        elif key.action == "backspace":
            self.text = self.text[:-1]
            self.mode = "groups"
            self.active_group = None
            self._set_flash("BACKSPACE")
        elif key.action == "clear":
            self.text = ""
            self.mode = "groups"
            self.active_group = None
            self._set_flash("CLEARED")
        elif key.action == "enter":
            self.last_submitted_text = self.text
            print(f"Keyboard submitted text: {self.text}")
            self.mode = "groups"
            self.active_group = None
            self._set_flash("ENTER")
        elif key.action == "back":
            self.mode = "groups"
            self.active_group = None
            self._set_flash("Back to groups")

        self._log_selection(key, gaze_point)
        self.awaiting_rearm = True
        self.rearm_origin = np.asarray(gaze_point, dtype=float)
        self._reset_hover()

    def update(
        self,
        gaze_point: tuple[float, float] | None,
        tracking_valid: bool,
        now: float | None = None,
    ) -> KeyboardKey | None:
        if now is None:
            now = time.perf_counter()

        if not tracking_valid or gaze_point is None:
            self._reset_hover()
            return None

        current_point = np.asarray(gaze_point, dtype=float)
        key = self._key_at(gaze_point)

        if self.awaiting_rearm:
            moved_far_enough = False
            if self.rearm_origin is not None:
                moved_far_enough = (
                    float(np.linalg.norm(current_point - self.rearm_origin))
                    >= self.rearm_distance_px
                )

            if key is None or moved_far_enough:
                self.awaiting_rearm = False
                self.rearm_origin = None
            else:
                self._reset_hover()
                return None

        if key is None:
            self._reset_hover()
            return None

        if key.key_id != self.hovered_key_id:
            self.hovered_key_id = key.key_id
            self.hover_started_at = now
            self.hover_progress = 0.0
            return None

        if self.hover_started_at is None:
            self.hover_started_at = now

        elapsed = max(0.0, now - self.hover_started_at)
        self.hover_progress = min(1.0, elapsed / self.dwell_seconds)

        if self.hover_progress >= 1.0:
            self._activate_key(key, gaze_point)
            return key

        return None

    @staticmethod
    def _fit_text(
        text: str,
        max_characters: int,
    ) -> str:
        if len(text) <= max_characters:
            return text
        return "..." + text[-(max_characters - 3):]

    def draw(
        self,
        gaze_point: tuple[float, float] | None,
        tracking_valid: bool,
        status_text: str = "",
    ) -> np.ndarray:
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 245

        cv2.rectangle(
            canvas,
            (80, 70),
            (self.width - 80, 205),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (80, 70),
            (self.width - 80, 205),
            (30, 30, 30),
            3,
        )

        visible_text = self._fit_text(self.text, max_characters=48)
        cv2.putText(
            canvas,
            f"TEXT: {visible_text}",
            (110, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (20, 20, 20),
            3,
        )

        mode_label = (
            "Look at a letter group"
            if self.mode == "groups"
            else f"Choose from {self.active_group}"
        )
        cv2.putText(
            canvas,
            f"{mode_label} - hold gaze for {self.dwell_seconds:.1f}s",
            (110, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (50, 50, 50),
            2,
        )

        if status_text:
            cv2.putText(
                canvas,
                status_text,
                (90, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                2,
            )

        keys = self.current_keys()
        for key in keys:
            hovered = key.key_id == self.hovered_key_id
            fill_color = (225, 225, 225)
            border_color = (40, 40, 40)

            if hovered and tracking_valid:
                fill_color = (190, 230, 255)
                border_color = (0, 110, 220)

            cv2.rectangle(
                canvas,
                (key.x1, key.y1),
                (key.x2, key.y2),
                fill_color,
                -1,
            )
            cv2.rectangle(
                canvas,
                (key.x1, key.y1),
                (key.x2, key.y2),
                border_color,
                4,
            )

            font_scale = 1.65 if len(key.label) <= 4 else 0.95
            thickness = 4 if len(key.label) <= 4 else 3
            text_size, _ = cv2.getTextSize(
                key.label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            text_x = key.x1 + (key.x2 - key.x1 - text_size[0]) // 2
            text_y = key.y1 + (key.y2 - key.y1 + text_size[1]) // 2
            cv2.putText(
                canvas,
                key.label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 20, 20),
                thickness,
            )

            if hovered and tracking_valid:
                bar_margin = 14
                bar_height = 18
                bar_x1 = key.x1 + bar_margin
                bar_x2 = key.x2 - bar_margin
                bar_y2 = key.y2 - bar_margin
                bar_y1 = bar_y2 - bar_height
                cv2.rectangle(
                    canvas,
                    (bar_x1, bar_y1),
                    (bar_x2, bar_y2),
                    (210, 210, 210),
                    -1,
                )
                progress_x = int(
                    bar_x1 + (bar_x2 - bar_x1) * self.hover_progress
                )
                cv2.rectangle(
                    canvas,
                    (bar_x1, bar_y1),
                    (progress_x, bar_y2),
                    (0, 140, 255),
                    -1,
                )

        now = time.perf_counter()
        if self.flash_message and now <= self.flash_until:
            message_size, _ = cv2.getTextSize(
                self.flash_message,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2,
            )
            box_width = message_size[0] + 50
            box_x1 = (self.width - box_width) // 2
            cv2.rectangle(
                canvas,
                (box_x1, 210),
                (box_x1 + box_width, 255),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                canvas,
                self.flash_message,
                (box_x1 + 25, 242),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 100, 0),
                2,
            )

        if gaze_point is not None and tracking_valid:
            gaze_x = int(np.clip(gaze_point[0], 0, self.width - 1))
            gaze_y = int(np.clip(gaze_point[1], 0, self.height - 1))
            cv2.circle(canvas, (gaze_x, gaze_y), 16, (0, 0, 255), -1)
            cv2.circle(canvas, (gaze_x, gaze_y), 27, (0, 0, 0), 2)
        else:
            cv2.putText(
                canvas,
                "GAZE PAUSED - return your head and eyes to the calibrated range",
                (300, self.height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 180),
                2,
            )

        cv2.putText(
            canvas,
            "Q: quit   D: camera preview",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (60, 60, 60),
            1,
        )

        return canvas

    def save_log(self, output_dir: str | Path = "outputs") -> Path | None:
        if not self.events:
            return None

        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"keyboard_interactions_{timestamp}.csv"

        fieldnames = list(self.events[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)

        return output_path
