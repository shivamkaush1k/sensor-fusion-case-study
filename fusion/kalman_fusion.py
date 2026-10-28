from math import isfinite


class SensorFusionKalman:
    """Fuse normalized visual and audio risk scores with a bounded 1D filter."""

    def __init__(self, process_noise=0.02, measurement_noise=0.08):
        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError("Filter noise values must be positive")
        self.estimate = 0.0
        self.error = 1.0
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

    @staticmethod
    def _score(value, name):
        if value is None:
            return None
        score = float(value)
        if not isfinite(score):
            raise ValueError(f"{name} must be finite")
        return max(0.0, min(1.0, score))

    def fuse(self, visual_score=None, audio_score=None, visual_weight=1.0, audio_weight=1.0):
        """Return the filtered score and diagnostics for the current frame.

        Missing sensors are ignored. Weights are useful when a detector reports
        a low-confidence observation without changing the public score range.
        """
        visual = self._score(visual_score, "visual_score")
        audio = self._score(audio_score, "audio_score")
        measurements = [(visual, visual_weight), (audio, audio_weight)]
        measurements = [(value, float(weight)) for value, weight in measurements if value is not None and weight > 0]
        if not measurements:
            self.error += self.process_noise
            return {"fused_score": self.estimate, "kalman_gain": 0.0, "measurement": None, "sensor_count": 0}

        total_weight = sum(weight for _, weight in measurements)
        measurement = sum(value * weight for value, weight in measurements) / total_weight
        predicted_error = self.error + self.process_noise
        effective_noise = self.measurement_noise / total_weight
        gain = predicted_error / (predicted_error + effective_noise)
        self.estimate += gain * (measurement - self.estimate)
        self.error = (1.0 - gain) * predicted_error
        return {
            "fused_score": max(0.0, min(1.0, self.estimate)),
            "kalman_gain": gain,
            "measurement": measurement,
            "sensor_count": len(measurements),
        }

