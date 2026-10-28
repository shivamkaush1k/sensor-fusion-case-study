import csv
from datetime import datetime
from pathlib import Path


class AlertManager:
    FIELDS = [
        "timestamp", "frame", "event", "risk", "people_count", "visual_score",
        "audio_score", "fusion_score", "kalman_gain", "discrepancy", "audio_event", "clip",
    ]

    def __init__(self, log_file="logs/events.csv"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.log_file.exists() and self.log_file.stat().st_size:
            with self.log_file.open(newline="", encoding="utf-8") as file:
                existing_fields = next(csv.reader(file), [])
            if existing_fields != self.FIELDS:
                self.log_file.unlink()
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.FIELDS).writeheader()

    def save_event(self, frame, event, risk, people_count, visual_score, audio_score, fusion_score,
                   kalman_gain, discrepancy, audio_event, clip_path=""):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"), "frame": frame,
            "event": event, "risk": risk, "people_count": people_count,
            "visual_score": self._number(visual_score), "audio_score": self._number(audio_score),
            "fusion_score": self._number(fusion_score), "kalman_gain": self._number(kalman_gain),
            "discrepancy": self._number(discrepancy), "audio_event": audio_event, "clip": clip_path,
        }
        with self.log_file.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.FIELDS).writerow(row)

    @staticmethod
    def _number(value):
        return "" if value is None else round(float(value), 3)

