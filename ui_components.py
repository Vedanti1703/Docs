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
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B0F19 !important;
    color: #E2E8F0 !important;
}

/* ── Global background ── */
.stApp {
    background: #0B0F19;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(45, 116, 255, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(255, 64, 129, 0.05), transparent 25%);
}

header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Custom header banner ── */
.ns-header {
    background: rgba(18, 25, 43, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 32px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    transition: transform 0.3s ease;
}
.ns-header:hover {
    transform: translateY(-2px);
}
.ns-header-text h1 {
    font-family: 'Outfit', sans-serif;
    color: #FFFFFF;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ns-header-text p {
    color: #94A3B8;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}
.ns-api-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 8px 20px;
    color: #F8FAFC;
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.ns-api-badge.connected { background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.3); color: #34D399; }
.ns-api-badge.offline   { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); color: #F87171; }

/* ── Metric cards ── */
.ns-cards-row {
    display: flex;
    gap: 20px;
    margin: 24px 0;
    flex-wrap: wrap;
}
.ns-card {
    flex: 1 1 120px;
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 24px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.ns-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    opacity: 0.8;
}
.ns-card:hover {
    transform: translateY(-4px);
    background: rgba(30, 41, 59, 0.6);
    box-shadow: 0 12px 28px rgba(0,0,0,0.3);
    border-color: rgba(255,255,255,0.1);
}
.ns-card .ns-card-val {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.1;
}
.ns-card .ns-card-label {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Quality badge ── */
.ns-quality {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 10px;
    border: 1px solid transparent;
}
.ns-quality.good { background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.2); color: #34D399; }
.ns-quality.fair { background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.2); color: #FBBF24; }
.ns-quality.poor { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.2); color: #F87171; }

/* ── Upload drop zone ── */
section[data-testid="stFileUploader"] > div:first-child {
    border: 2px dashed rgba(148, 163, 184, 0.3) !important;
    border-radius: 20px !important;
    background: rgba(30, 41, 59, 0.3) !important;
    padding: 40px !important;
    transition: all 0.3s ease;
}
section[data-testid="stFileUploader"] > div:first-child:hover {
    background: rgba(30, 41, 59, 0.5) !important;
    border-color: #3B82F6 !important;
}
/* Ensure the text inside uploader is bright */
section[data-testid="stFileUploader"] small { color: #94A3B8 !important; }

/* ── Progress steps ── */
.ns-steps { display: flex; gap: 0; margin: 24px 0; }
.ns-step {
    flex: 1;
    text-align: center;
    position: relative;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    padding-bottom: 28px;
    transition: color 0.3s ease;
}
.ns-step::before {
    content: '';
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #1E293B;
    border: 2px solid #334155;
    margin: 0 auto 12px auto;
    position: relative;
    z-index: 1;
    transition: all 0.3s ease;
}
.ns-step::after {
    content: '';
    position: absolute;
    top: 16px; left: 50%; right: -50%;
    height: 2px;
    background: #334155;
    z-index: 0;
    transition: background 0.3s ease;
}
.ns-step:last-child::after { display: none; }
.ns-step.done::before   { background: #10B981; border-color: #059669; content: '✓'; color: #fff; line-height: 28px; font-size: 0.8rem; font-weight: 700; box-shadow: 0 0 12px rgba(16,185,129,0.4); }
.ns-step.done::after    { background: #10B981; }
.ns-step.done           { color: #10B981; }
.ns-step.active::before { background: #3B82F6; border-color: #60A5FA; animation: pulse-ring-dark 2s infinite; box-shadow: 0 0 16px rgba(59,130,246,0.5); }
.ns-step.active         { color: #F8FAFC; }
@keyframes pulse-ring-dark {
    0%   { box-shadow: 0 0 0 0 rgba(59,130,246,0.6); }
    70%  { box-shadow: 0 0 0 10px rgba(59,130,246,0); }
    100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
}

/* ── Side-by-side content panels ── */
.ns-panel {
    background: rgba(30, 41, 59, 0.3);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    height: 100%;
}
.ns-panel h4 {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #CBD5E1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 16px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 12px;
}

/* ── Region type pills ── */
.ns-pill {
    display: inline-block;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 4px 4px 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.ns-pill.text     { background: rgba(59, 130, 246, 0.15); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.3); }
.ns-pill.equation { background: rgba(168, 85, 247, 0.15); color: #D8B4FE; border: 1px solid rgba(168, 85, 247, 0.3); }
.ns-pill.diagram  { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.3); }

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    filter: brightness(1.1) !important;
}

/* Secondary Button Customization */
button[data-testid="baseButton-secondary"] {
    background: rgba(51, 65, 85, 0.8) !important;
    box-shadow: none !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
button[data-testid="baseButton-secondary"]:hover {
    background: rgba(71, 85, 105, 0.9) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px !important;
    color: #94A3B8 !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 12px 24px;
    margin-right: 8px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(51, 65, 85, 0.6);
    color: #CBD5E1 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    color: #60A5FA !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* ── Info / success / warning overrides ── */
.stAlert { 
    border-radius: 14px !important; 
    border: 1px solid rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(8px) !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: #CBD5E1;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    color: #F8FAFC !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.3) !important;
}

/* ── Scrollable text preview ── */
.ns-text-preview {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    max-height: 320px;
    overflow-y: auto;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
    color: #E2E8F0;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
}
.ns-text-preview::-webkit-scrollbar { width: 8px; }
.ns-text-preview::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
.ns-text-preview::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
.ns-text-preview::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Inline caption below figures ── */
.ns-caption {
    text-align: center;
    font-size: 0.8rem;
    color: #64748B;
    font-style: italic;
    margin-top: 8px;
    margin-bottom: 12px;
}

/* ── Export outline card ── */
.ns-outline {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
}
.ns-outline li { 
    margin: 8px 0; 
    font-size: 0.95rem; 
    color: #CBD5E1; 
}

/* General images (e.g. webcam or upload preview) */
img {
    border-radius: 12px;
}
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
