class EventClassifier:
    def classify(self, people_count, audio_event, discrepancy, fused_score=0.0):
        if discrepancy:
            return "Sensor Discrepancy"
        if audio_event == "Scream / Alarm":
            return "Emergency"
        if people_count >= 4 and audio_event == "Shouting":
            return "Crowd Disturbance"
        if people_count >= 5:
            return "Crowd Gathering"
        if fused_score >= 0.75:
            return "Elevated Activity"
        return "Normal Activity"

