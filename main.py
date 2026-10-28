"""Run the multimodal sensor-fusion case study."""

import argparse
from pathlib import Path

import numpy as np

from alerts.alert_manager import AlertManager
from audio.audio_detector import AudioDetector
from audio.audio_processor import AudioProcessor
from detection.discrepancy_detector import DiscrepancyDetector
from detection.event_classifier import EventClassifier
from detection.risk_assessor import RiskAssessor
from fusion.kalman_fusion import SensorFusionKalman


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse visual occupancy and audio intensity into explainable events.")
    parser.add_argument("--source", default="0", help="Webcam index or video path for live mode.")
    parser.add_argument("--audio", default=None, help="Optional PCM WAV aligned with the video source.")
    parser.add_argument("--log-file", default="logs/events.csv", help="CSV destination for telemetry.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames in live mode; zero runs to completion.")
    parser.add_argument("--headless", action="store_true", help="Do not open a live annotated video window.")
    parser.add_argument("--demo", action="store_true", help="Run a deterministic multimodal scenario without a camera or model.")
    return parser.parse_args()


class FusionPipeline:
    def __init__(self, log_file):
        self.audio_detector = AudioDetector()
        self.fusion = SensorFusionKalman()
        self.discrepancy = DiscrepancyDetector(threshold=0.45)
        self.classifier = EventClassifier()
        self.risk = RiskAssessor()
        self.alerts = AlertManager(log_file)

    def process(self, frame, people_count, visual_score, signal=None):
        audio = self.audio_detector.detect(signal)
        fusion = self.fusion.fuse(visual_score, audio["audio_score"])
        discrepancy = self.discrepancy.detect(visual_score, audio["audio_score"])
        event = self.classifier.classify(
            people_count, audio["audio_event"], discrepancy["has_discrepancy"], fusion["fused_score"]
        )
        risk = self.risk.assess(fusion["fused_score"])
        self.alerts.save_event(
            frame, event, risk, people_count, visual_score, audio["audio_score"],
            fusion["fused_score"], fusion["kalman_gain"], discrepancy["difference"], audio["audio_event"],
        )
        return {"frame": frame, "event": event, "risk": risk, "audio": audio, "fusion": fusion, "discrepancy": discrepancy}


def demo_samples():
    """Generate quiet, crowded, audio-only, and sensor-disagreement phases."""
    for frame in range(1, 61):
        if frame < 16:
            people, visual, amplitude = 1, 0.15, 0.04
        elif frame < 31:
            people, visual, amplitude = 5, 0.95, 0.24
        elif frame < 46:
            people, visual, amplitude = 1, 0.18, 0.34
        else:
            people, visual, amplitude = 5, 0.92, 0.04
        signal = amplitude * np.sin(np.linspace(0, 2 * np.pi * 12, 1600, dtype=np.float32))
        yield frame, people, visual, signal


def run_demo(args):
    pipeline = FusionPipeline(args.log_file)
    counts = {}
    for frame, people, visual, signal in demo_samples():
        result = pipeline.process(frame, people, visual, signal)
        counts[result["event"]] = counts.get(result["event"], 0) + 1
        if frame == 1 or frame % 15 == 0:
            print(f"frame={frame:02d} event={result['event']:<20} risk={result['risk']:<8} "
                  f"visual={visual:.2f} audio={result['audio']['audio_score']:.2f} "
                  f"fused={result['fusion']['fused_score']:.2f}")
    print(f"Wrote {sum(counts.values())} telemetry rows to {Path(args.log_file)}")
    print("Events:", ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))


def run_live(args):
    import cv2
    from video.object_detector import ObjectDetector
    from video.video_processor import VideoProcessor

    video = VideoProcessor(args.source)
    audio_processor = AudioProcessor(args.audio) if args.audio else None
    if audio_processor:
        audio_processor.load_audio()
    detector = ObjectDetector()
    pipeline = FusionPipeline(args.log_file)
    fps = video.fps if video.fps > 1 else 30.0
    frame_number = 0
    try:
        while (frame := video.get_frame()) is not None:
            frame_number += 1
            detection = detector.detect(frame)
            signal = audio_processor.chunk_at((frame_number - 1) / fps) if audio_processor else None
            result = pipeline.process(frame_number, detection["people_count"], detection["visual_score"], signal)
            cv2.putText(frame, f"{result['event']} | {result['risk']} | fused {result['fusion']['fused_score']:.2f}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 120), 2)
            if not args.headless:
                cv2.imshow("Sensor Fusion", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frame_number >= args.max_frames:
                break
    finally:
        video.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {frame_number} frames; telemetry written to {Path(args.log_file)}")


def main():
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("max-frames must be non-negative")
    run_demo(args) if args.demo else run_live(args)


if __name__ == "__main__":
    main()
