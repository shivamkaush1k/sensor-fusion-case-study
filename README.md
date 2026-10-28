# Sensor Fusion Case Study

An explainable, end-to-end analytics pipeline that combines visual occupancy and audio intensity into stable event and risk decisions. It is designed for authorized footage and produces inspectable CSV telemetry rather than controlling any physical system.

## What it demonstrates

- **Visual sensing:** YOLO detects people; occupancy and mean detector confidence become a bounded visual score.
- **Audio sensing:** PCM WAV windows are normalized and classified into normal audio, shouting, or scream/alarm.
- **Fusion:** a one-dimensional Kalman filter smooths the two sensor scores and tolerates a missing sensor.
- **Reasoning:** discrepancy detection, event classification, and risk assessment remain separate testable stages.
- **Evidence:** every processed frame is written to `logs/events.csv` for the Streamlit dashboard.

## Quick start

Use Python 3.10-3.13.

```powershell
python -m pip install -r requirements.txt
python main.py --demo
streamlit run dashboard/streamlit_app.py
```

The demo runs 60 deterministic frames in four phases: quiet activity, crowd disturbance, audio-only activity, and visual/audio disagreement. It does not require a camera, model download, or WAV file.

## Live video

```powershell
# Webcam, video-only fusion
python main.py --source 0 --headless --max-frames 300

# Video plus a time-aligned PCM WAV file
python main.py --source data/sample.mp4 --audio data/sample.wav --headless
```

The first YOLO run downloads `yolov8n.pt`. For an interactive display, omit `--headless` and press `q` or `Esc` to stop.

## Architecture

```text
video/webcam -> YOLO people detector -- visual score --\
                                                   -> Kalman fusion -> event -> risk -> CSV
WAV windows  -> RMS audio detector ---- audio score -/          \\
                                                        discrepancy -> dashboard
```

The filter state is a single normalized risk estimate. Each update records the raw sensor values, fused value, Kalman gain, event, risk, and whether the sensors disagreed. If audio is absent, the visual score still flows through the filter and discrepancy is marked as not comparable.

## Project layout

```text
main.py                         CLI and end-to-end orchestration
video/object_detector.py        YOLO adapter and visual score
video/video_processor.py        Webcam/video reader
audio/audio_processor.py        Time-aligned WAV windows
audio/audio_detector.py         RMS-based audio event detector
fusion/kalman_fusion.py         Missing-sensor-aware scalar Kalman filter
detection/                      Discrepancy, event, and risk decisions
alerts/alert_manager.py         Versioned CSV telemetry writer
dashboard/streamlit_app.py      Telemetry dashboard
logs/events.csv                 Generated output
```

## Interpretation and limitations

The demo is a reproducible engineering example, not a trained threat-detection system. RMS audio is a loudness heuristic, YOLO confidence is not a semantic risk probability, and the Kalman filter introduces temporal lag by design. Camera motion, occlusion, background noise, microphone gain, and unaligned timestamps can all change the result. Calibrate thresholds against representative authorized data before using the pipeline for any operational decision.
