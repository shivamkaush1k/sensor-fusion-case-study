from pathlib import Path

import cv2


class VideoProcessor:
    """Read frames from a webcam index or a video path."""

    def __init__(self, source=0):
        self.source = int(source) if str(source).isdigit() else str(Path(source))
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0

    def get_frame(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self):
        self.capture.release()

