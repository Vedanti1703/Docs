"""
ui_components.py
────────────────
Reusable Streamlit UI helpers and all custom CSS for the
Handwritten Notes Scanner redesign.
"""

from __future__ import annotations

import io
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

from region_detector import RegionInfo


# ────────────────────────────────────────────────
# CSS injection
# ────────────────────────────────────────────────

_CSS = """
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Global background ── */
.stApp {
    background: #F7F9FC;
}

/* ── Hide default Streamlit header ── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Custom header banner ── */
.ns-header {
    background: linear-gradient(135deg, #1F497D 0%, #2E74B5 60%, #3A8FD6 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(31,73,125,0.22);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.ns-header-text h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.ns-header-text p {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 0;
}
.ns-api-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 999px;
    padding: 6px 16px;
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 500;
    white-space: nowrap;
}
.ns-api-badge.connected { background: rgba(59,109,17,0.45); border-color: rgba(100,200,80,0.5); }
.ns-api-badge.offline   { background: rgba(180,30,30,0.40); border-color: rgba(220,80,80,0.5); }

/* ── Metric cards ── */
.ns-cards-row {
    display: flex;
    gap: 16px;
    margin: 20px 0 24px 0;
    flex-wrap: wrap;
}
.ns-card {
    flex: 1 1 120px;
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(31,73,125,0.09);
    border-top: 4px solid #2E74B5;
    text-align: center;
}
.ns-card .ns-card-val {
    font-size: 2rem;
    font-weight: 700;
    color: #1F497D;
    line-height: 1.1;
}
.ns-card .ns-card-label {
    font-size: 0.78rem;
    color: #6B7280;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Quality badge ── */
.ns-quality {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}
.ns-quality.good { background: #DCF5D0; color: #2D6A0F; }
.ns-quality.fair { background: #FFF3CC; color: #7A5700; }
.ns-quality.poor { background: #FFE0DC; color: #9B2020; }

/* ── Upload drop zone ── */
section[data-testid="stFileUploader"] > div:first-child {
    border: 2px dashed #2E74B5 !important;
    border-radius: 14px !important;
    background: #EEF5FF !important;
    padding: 36px !important;
    transition: background 0.2s ease;
}
section[data-testid="stFileUploader"] > div:first-child:hover {
    background: #DDE9FB !important;
}

/* ── Progress steps ── */
.ns-steps { display: flex; gap: 0; margin: 16px 0; }
.ns-step {
    flex: 1;
    text-align: center;
    position: relative;
    font-size: 0.78rem;
    font-weight: 500;
    color: #9CA3AF;
    padding-bottom: 28px;
}
.ns-step::before {
    content: '';
    display: block;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #E5E7EB;
    border: 2px solid #D1D5DB;
    margin: 0 auto 8px auto;
    position: relative;
    z-index: 1;
}
.ns-step::after {
    content: '';
    position: absolute;
    top: 14px; left: 50%; right: -50%;
    height: 2px;
    background: #E5E7EB;
    z-index: 0;
}
.ns-step:last-child::after { display: none; }
.ns-step.done::before   { background: #2E74B5; border-color: #1F497D; content: '✓'; color: #fff; line-height: 24px; font-size: 0.72rem; font-weight: 700; }
.ns-step.done::after    { background: #2E74B5; }
.ns-step.active::before { background: #E6F1FB; border-color: #2E74B5; animation: pulse-ring 1.2s infinite; }
.ns-step.active         { color: #1F497D; font-weight: 600; }
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(46,116,181,0.45); }
    70%  { box-shadow: 0 0 0 8px rgba(46,116,181,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,116,181,0); }
}

/* ── Side-by-side content panels ── */
.ns-panel {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(31,73,125,0.07);
    height: 100%;
}
.ns-panel h4 {
    font-size: 0.82rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 12px 0;
    border-bottom: 1px solid #F3F4F6;
    padding-bottom: 8px;
}

/* ── Region type pills ── */
.ns-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px 2px 2px 0;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.ns-pill.text     { background: #E6F1FB; color: #1F497D; }
.ns-pill.equation { background: #FFF0D6; color: #7A4500; }
.ns-pill.diagram  { background: #EDF7EE; color: #1D5C1F; }

/* ── Download button style ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1F497D, #2E74B5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s ease !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 22px;
}
.stTabs [aria-selected="true"] {
    color: #1F497D !important;
    border-bottom-color: #2E74B5 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1F497D !important;
}
section[data-testid="stSidebar"] * {
    color: #E6F1FB !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #2E74B5 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Info / success / warning overrides ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollable text preview ── */
.ns-text-preview {
    background: #FAFBFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 16px;
    max-height: 320px;
    overflow-y: auto;
    font-size: 0.88rem;
    line-height: 1.65;
    white-space: pre-wrap;
    color: #2C2C2A;
}

/* ── Inline caption below figures ── */
.ns-caption {
    text-align: center;
    font-size: 0.78rem;
    color: #9CA3AF;
    font-style: italic;
    margin-top: 4px;
    margin-bottom: 8px;
}

/* ── Export outline card ── */
.ns-outline {
    background: #FAFBFF;
    border: 1px solid #D1D9E6;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
.ns-outline li { margin: 4px 0; font-size: 0.88rem; color: #374151; }
</style>
"""


def inject_css() -> None:
    """Inject all custom CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────

def render_header(api_connected: bool) -> None:
    badge_class = "connected" if api_connected else "offline"
    badge_icon  = "🟢" if api_connected else "🔴"
    badge_label = "Groq · Llama 4 Scout connected" if api_connected else "API offline"
    st.markdown(
        f"""
        <div class="ns-header">
            <div class="ns-header-text">
                <h1>📝 Handwritten Notes Scanner</h1>
                <p>Convert handwritten notes to digital documents instantly</p>
            </div>
            <div class="ns-api-badge {badge_class}">
                {badge_icon}&nbsp;{badge_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# Image quality badge
# ────────────────────────────────────────────────

def _image_quality_score(image_bgr: np.ndarray) -> tuple[str, str, str]:
    """
    Returns (label, css_class, detail_string).
    Uses mean brightness and Laplacian variance (blur metric).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 60 or brightness > 230:
        return "Poor", "poor", f"Brightness {brightness:.0f}/255 — try better lighting"
    if blur_score < 50:
        return "Poor", "poor", f"Blur score {blur_score:.0f} — image too blurry"
    if blur_score < 200 or brightness < 100:
        return "Fair", "fair", f"Brightness {brightness:.0f}, sharpness {blur_score:.0f}"
    return "Good", "good", f"Brightness {brightness:.0f}, sharpness {blur_score:.0f}"


def render_quality_badge(image_bgr: np.ndarray) -> None:
    label, css_class, detail = _image_quality_score(image_bgr)
    icons = {"good": "✅", "fair": "⚠️", "poor": "❌"}
    icon = icons.get(css_class, "")
    st.markdown(
        f'<div class="ns-quality {css_class}">{icon} Image Quality: <b>{label}</b>'
        f'&nbsp;·&nbsp;<span style="font-weight:400">{detail}</span></div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# Count cards
# ────────────────────────────────────────────────

def render_count_cards(
    pages: int = 0,
    equations: int = 0,
    diagrams: int = 0,
    words: int = 0,
) -> None:
    st.markdown(
        f"""
        <div class="ns-cards-row">
            <div class="ns-card">
                <div class="ns-card-val">{pages}</div>
                <div class="ns-card-label">Pages Processed</div>
            </div>
            <div class="ns-card">
                <div class="ns-card-val">{equations}</div>
                <div class="ns-card-label">Equations Detected</div>
            </div>
            <div class="ns-card">
                <div class="ns-card-val">{diagrams}</div>
                <div class="ns-card-label">Diagrams Detected</div>
            </div>
            <div class="ns-card">
                <div class="ns-card-val">{words}</div>
                <div class="ns-card-label">Words Extracted</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# Step progress tracker
# ────────────────────────────────────────────────

STEPS = [
    "Detecting document",
    "Reading handwriting",
    "Detecting equations",
    "Building preview",
]


def render_steps(current: int) -> None:
    """
    Render a horizontal progress step indicator.

    Parameters
    ----------
    current : 0-based index of the **active** step.
              Steps before it are marked done, steps after pending.
    """
    items = ""
    for i, label in enumerate(STEPS):
        if i < current:
            cls = "ns-step done"
        elif i == current:
            cls = "ns-step active"
        else:
            cls = "ns-step"
        items += f'<div class="{cls}">{label}</div>'
    st.markdown(f'<div class="ns-steps">{items}</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────
# Region-aware page preview
# ────────────────────────────────────────────────

def render_region_preview(
    regions: List[RegionInfo],
    original_bgr: np.ndarray,
    max_thumb_px: int = 280,
) -> None:
    """
    Render an inline preview of all regions in reading order.
    - text regions  → styled text block
    - equation/diagram regions → thumbnail image with caption
    """
    from image_cropper import crop_region, thumbnail, region_to_jpeg_bytes

    for region in regions:
        pill_class = region.type
        pill_label = {"text": "Text", "equation": "Equation", "diagram": "Diagram"}.get(
            region.type, region.type.capitalize()
        )
        st.markdown(
            f'<span class="ns-pill {pill_class}">{pill_label}</span>',
            unsafe_allow_html=True,
        )

        if region.type == "text":
            content = (region.content or "").strip()
            if content:
                st.markdown(
                    f'<div class="ns-text-preview">{content}</div>',
                    unsafe_allow_html=True,
                )
        else:
            try:
                crop = crop_region(original_bgr, region)
                thumb = thumbnail(crop, max_dim=max_thumb_px)
                jpeg_bytes = region_to_jpeg_bytes(thumb, quality=80)
                caption = "Equation" if region.type == "equation" else "Figure / Diagram"
                st.image(jpeg_bytes, caption=caption, width="content")
            except Exception as e:
                st.caption(f"[Could not render preview: {e}]")

        st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────

def render_sidebar(api_connected: bool, stats: dict) -> dict:
    """
    Render the sidebar and return the current settings dict.

    Returns
    -------
    dict with keys:
        spell_check        : bool
        equation_detection : bool
        diagram_detection  : bool
        quality_threshold  : int  (blur score minimum)
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # API status
        if api_connected:
            st.markdown("🟢 **Groq API** — Connected")
        else:
            st.markdown("🔴 **Groq API** — Offline")

        st.markdown("---")

        # Usage stats
        st.markdown("### 📊 Session Stats")
        st.markdown(f"- Pages scanned: **{stats.get('pages', 0)}**")
        st.markdown(f"- Words extracted: **{stats.get('words', 0)}**")
        st.markdown(f"- Equations detected: **{stats.get('equations', 0)}**")
        st.markdown(f"- Diagrams detected: **{stats.get('diagrams', 0)}**")

        st.markdown("---")

        # Feature toggles
        st.markdown("### 🛠️ Features")
        spell_check = st.toggle("Spell Check", value=True, key="sb_spell")
        equation_detection = st.toggle("Equation Detection", value=True, key="sb_eq")
        diagram_detection = st.toggle("Diagram Detection", value=True, key="sb_diag")

        st.markdown("---")
        quality_threshold = st.slider(
            "Min. Image Quality (blur score)",
            min_value=0,
            max_value=300,
            value=50,
            step=10,
            key="sb_quality",
            help="Images below this blur score trigger a quality warning.",
        )

    return {
        "spell_check": spell_check,
        "equation_detection": equation_detection,
        "diagram_detection": diagram_detection,
        "quality_threshold": quality_threshold,
    }
