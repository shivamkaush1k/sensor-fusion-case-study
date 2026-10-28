"""Streamlit dashboard for the sensor-fusion telemetry log."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "events.csv"
REQUIRED_COLUMNS = {"frame", "event", "risk", "visual_score", "audio_score", "fusion_score"}

st.set_page_config(page_title="Sensor Fusion Monitor", page_icon="+", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { background: #0b1117; color: #edf4f1; }
    .block-container { max-width: 1300px; padding: 2.5rem 3rem; }
    h1, h2, h3, p, div { font-family: 'Space Grotesk', sans-serif; }
    .eyebrow { color: #7ce2b2; font: 500 .72rem 'DM Mono', monospace; letter-spacing: .16em; }
    .hero { border-bottom: 1px solid #26352f; padding-bottom: 1.4rem; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.5rem; margin: .35rem 0 .3rem; letter-spacing: -.04em; }
    .hero p { color: #9eada7; margin: 0; }
    div[data-testid='stMetric'] { background: #121c1b; border: 1px solid #26352f; padding: 1rem; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_log():
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        return None, "No telemetry yet. Run `python main.py --demo` first."
    data = pd.read_csv(LOG_FILE)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        return None, "This log uses the old schema. Run the demo again to regenerate it."
    return data, ""


data, message = load_log()
st.markdown('<div class="hero"><div class="eyebrow">MULTIMODAL OBSERVABILITY / CASE STUDY</div><h1>Sensor Fusion Monitor</h1><p>Visual occupancy and audio intensity, combined with a transparent temporal filter.</p></div>', unsafe_allow_html=True)

if data is None:
    st.info(message)
    st.stop()

latest = data.iloc[-1]
observed = int((data["event"] == "Normal Activity").sum())
discrepancies = int((data["event"] == "Sensor Discrepancy").sum())
critical = int((data["risk"] == "CRITICAL").sum())
metric_columns = st.columns(4)
metric_columns[0].metric("Frames", len(data))
metric_columns[1].metric("Latest fused score", f"{latest['fusion_score']:.2f}")
metric_columns[2].metric("Discrepancies", discrepancies)
metric_columns[3].metric("Critical frames", critical)

st.subheader(f"Latest state: {latest['event']}")
st.caption(f"Risk {latest['risk']} | frame {int(latest['frame'])} | audio event {latest.get('audio_event', 'unknown')}")
score_data = data.set_index("frame")[["visual_score", "audio_score", "fusion_score"]]
st.line_chart(score_data, height=300, color=["#67b7dc", "#f2a65a", "#7ce2b2"])

left, right = st.columns(2)
with left:
    st.subheader("Event distribution")
    st.bar_chart(data["event"].value_counts())
with right:
    st.subheader("Risk distribution")
    st.bar_chart(data["risk"].value_counts())

st.subheader("Telemetry")
st.dataframe(data.tail(25), use_container_width=True, hide_index=True)
