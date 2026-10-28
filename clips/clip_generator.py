from datetime import datetime
from pathlib import Path

import cv2


class ClipGenerator:
    def __init__(self, output_dir="data/clips", fps=20.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

    def save_clip(self, frames):
        if not frames:
            return ""
        height, width = frames[0].shape[:2]
        path = self.output_dir / f"event_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create clip: {path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        return str(path)

