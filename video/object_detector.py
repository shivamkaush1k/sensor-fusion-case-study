class ObjectDetector:
    """YOLO person detector returning boxes for visual tracking."""

    def __init__(self, model_name="yolov8n.pt", confidence=0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for people detection. Run: python -m pip install ultralytics"
            ) from exc
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame):
        result = self.model(frame, classes=[0], conf=self.confidence, verbose=False)[0]
        people_count = len(result.boxes)
        detections = []
        confidences = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            confidences.append(confidence)
            detections.append({"box": [x1, y1, x2, y2], "confidence": confidence})
        crowd_component = min(1.0, people_count / 6.0)
        confidence_component = sum(confidences) / len(confidences) if confidences else 0.0
        visual_score = min(1.0, 0.65 * crowd_component + 0.35 * confidence_component)
        return {"people_count": people_count, "visual_score": visual_score, "detections": detections}
