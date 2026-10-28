class DiscrepancyDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def detect(self, visual_score, audio_score):
        if visual_score is None or audio_score is None:
            return {"difference": None, "has_discrepancy": False, "comparable": False}
        difference = abs(float(visual_score) - float(audio_score))
        return {"difference": round(difference, 3), "has_discrepancy": difference >= self.threshold, "comparable": True}

