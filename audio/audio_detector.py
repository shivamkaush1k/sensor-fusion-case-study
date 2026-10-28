import numpy as np


class AudioDetector:
    """Energy-based prototype audio-event detector.

    Replace this with a trained audio classifier for production use.
    """

    def __init__(self, shout_rms=0.12, alarm_rms=0.28):
        if not 0 < shout_rms < alarm_rms:
            raise ValueError("Use thresholds where 0 < shout_rms < alarm_rms")
        self.shout_rms = float(shout_rms)
        self.alarm_rms = float(alarm_rms)

    def detect(self, signal):
        if signal is None or len(signal) == 0:
            return {"audio_score": None, "audio_event": "No Audio", "rms": 0.0}
        samples = np.asarray(signal, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        score = min(1.0, rms / self.alarm_rms)
        if rms >= self.alarm_rms:
            event = "Scream / Alarm"
        elif rms >= self.shout_rms:
            event = "Shouting"
        else:
            event = "Normal Audio"
        return {"audio_score": score, "audio_event": event, "rms": rms}

