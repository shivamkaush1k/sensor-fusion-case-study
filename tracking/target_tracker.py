"""Lightweight multi-person tracking with occlusion-aware position prediction.

This module is intended for visual analytics, such as sports footage or
facility occupancy monitoring.  It does not control or integrate with any
weapon or physical system.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    state: np.ndarray  # centre-x, centre-y, velocity-x, velocity-y
    covariance: np.ndarray
    width: float
    height: float
    confidence: float
    age: int = 1
    misses: int = 0
    detected_this_frame: bool = True

    @property
    def center(self):
        return self.state[:2]

    def bbox(self):
        cx, cy = self.center
        return [int(cx - self.width / 2), int(cy - self.height / 2), int(cx + self.width / 2), int(cy + self.height / 2)]


class TargetTracker:
    """Associate person detections and predict their image positions.

    The constant-velocity Kalman model gives a deliberately short-horizon
    estimate during missing detections; it should not be treated as identity
    verification or a long-term trajectory forecast.
    """

    def __init__(self, max_misses=20, match_distance=110.0):
        self.max_misses = max_misses
        self.match_distance = match_distance
        self.tracks = []
        self._next_id = 1

    @staticmethod
    def _measurement(box):
        x1, y1, x2, y2 = box[:4]
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]), float(x2 - x1), float(y2 - y1)

    def _predict(self, track, dt):
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        process_noise = np.diag([8.0, 8.0, 3.0, 3.0])
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise
        track.detected_this_frame = False

    def _update(self, track, box, confidence):
        measurement, width, height = self._measurement(box)
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        measurement_noise = np.eye(2) * 18.0
        innovation = measurement - observation @ track.state
        gain = track.covariance @ observation.T @ np.linalg.inv(observation @ track.covariance @ observation.T + measurement_noise)
        track.state += gain @ innovation
        track.covariance = (np.eye(4) - gain @ observation) @ track.covariance
        track.width, track.height, track.confidence = width, height, confidence
        track.misses, track.detected_this_frame = 0, True

    def update(self, detections, dt=1 / 30):
        """Update tracks from ``[{box: [x1,y1,x2,y2], confidence: float}]``."""
        for track in self.tracks:
            self._predict(track, dt)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        pairs = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                center, _, _ = self._measurement(detection["box"])
                pairs.append((float(np.linalg.norm(track.center - center)), track_index, detection_index))
        for distance, track_index, detection_index in sorted(pairs):
            if distance > self.match_distance or track_index not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self._update(self.tracks[track_index], detections[detection_index]["box"], detections[detection_index]["confidence"])
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index in unmatched_tracks:
            self.tracks[track_index].misses += 1
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            center, width, height = self._measurement(detection["box"])
            self.tracks.append(Track(self._next_id, np.array([center[0], center[1], 0.0, 0.0]), np.eye(4) * 100.0, width, height, detection["confidence"]))
            self._next_id += 1

        self.tracks = [track for track in self.tracks if track.misses <= self.max_misses]
        for track in self.tracks:
            track.age += 1
        return list(self.tracks)

    def get(self, track_id):
        return next((track for track in self.tracks if track.track_id == track_id), None)
