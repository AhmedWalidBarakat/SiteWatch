"""
SiteWatch - computer vision safety compliance monitor.
Detects people in an image and flags whether each is wearing a hard hat
and high-visibility vest, using a pretrained YOLOv8 person detector plus
HSV color heuristics (see ppe_monitor.py for the approach and tradeoffs).
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ppe_monitor import run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --sw-accent: #ef4444;
    --sw-accent-soft: rgba(239, 68, 68, 0.14);
}

h1, h2, h3, [data-testid="stSidebar"] h2 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.sw-hero {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.1rem;
}
.sw-hero .sw-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--sw-accent), #f59e0b);
    font-size: 1.3rem;
}
.sw-hero h1 {
    margin: 0 !important;
    font-size: 1.9rem !important;
    letter-spacing: -0.02em;
}
.sw-tagline {
    color: var(--sw-accent);
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0.2rem 0 1.1rem 0;
}

.sw-stat {
    background: var(--sw-accent-soft);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.55rem;
}
.sw-stat .sw-stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
}
.sw-stat .sw-stat-value {
    font-size: 0.95rem;
    font-weight: 600;
}
</style>
"""


def list_samples():
    return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))


st.set_page_config(page_title="SiteWatch", page_icon="🦺", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🦺 SiteWatch")
    st.caption("Computer vision safety compliance monitor")
    st.markdown(
        """<div class="sw-stat">
        <div class="sw-stat-label">Detector</div>
        <div class="sw-stat-value">YOLOv8n (person)</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="sw-stat">
        <div class="sw-stat-label">PPE check</div>
        <div class="sw-stat-value">HSV color heuristics</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Sample photos are public-domain U.S. government works (OSHA / NIOSH / EPA Documerica via Wikimedia Commons).")

st.markdown(
    """<div class="sw-hero"><div class="sw-badge">🦺</div><h1>SiteWatch</h1></div>
    <div class="sw-tagline">Flagging missing hard hats and hi-vis vests, automatically.</div>""",
    unsafe_allow_html=True,
)

samples = list_samples()
sample_names = [os.path.basename(p) for p in samples]
choice = st.selectbox("Choose a sample photo, or upload your own below", ["(upload my own)"] + sample_names)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
elif choice != "(upload my own)":
    path = os.path.join(SAMPLE_DIR, choice)
    image_bgr = cv2.imread(path)
else:
    image_bgr = None

if image_bgr is not None:
    with st.spinner("Detecting people and checking PPE..."):
        annotated_bgr, people = run(image_bgr)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True)

    total = len(people)
    compliant = sum(1 for p in people if p["status"] == "Compliant")

    col1, col2, col3 = st.columns(3)
    col1.metric("People detected", total)
    col2.metric("Compliant", compliant)
    col3.metric("Flagged", total - compliant)

    if people:
        df = pd.DataFrame(
            [
                {
                    "Person": i + 1,
                    "Status": p["status"],
                    "Hard hat score": round(p["hardhat_score"], 2),
                    "Vest score": round(p["vest_score"], 2),
                }
                for i, p in enumerate(people)
            ]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("Pick a sample photo or upload your own to run the compliance check.")
