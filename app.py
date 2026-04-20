"""
app.py
──────
Handwritten Notes Scanner — complete UI redesign.

Tabs:
  1. 📁 Upload      — drag-drop zone, quality badge, 4-step progress,
                      side-by-side original ↔ region preview
  2. 📷 Webcam      — camera capture with instant preview
  3. 🔍 Review      — per-page expanders, editable text, inline thumbnails
  4. 📤 Export      — document outline, word/eq/diag counts, download buttons

Sidebar: API status, session stats, feature toggles, quality slider.
"""

from __future__ import annotations

import io
from typing import Any, List

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── local modules ────────────────────────────────────────────
from document_export import create_docx_document
from image_cropper import region_to_jpeg_bytes, thumbnail
from ocr_engine import build_ocr_engine
from preprocessing import detect_and_warp_document
from region_detector import RegionInfo, RegionList, detect_regions
from scanner import camera_input_to_cv2_image, uploaded_file_to_cv2_image
from text_postprocessor import postprocess_text
from ui_components import (
    inject_css,
    render_count_cards,
    render_header,
    render_quality_badge,
    render_region_preview,
    render_sidebar,
    render_steps,
)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        # list of dicts: {regions, original_bgr, warped_bgr, filename}
        "page_results": [],
        # aggregate stats for sidebar
        "stats": {"pages": 0, "words": 0, "equations": 0, "diagrams": 0},
        # cached export bytes — persisted so download button survives reruns
        "docx_bytes": None,
        "docx_size_kb": 0,
        "txt_content": None,
        "txt_size_kb": 0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _update_stats(regions: List[RegionInfo]) -> None:
    s = st.session_state["stats"]
    s["pages"] += 1
    for r in regions:
        if r.type == "text":
            s["words"] += len((r.content or "").split())
        elif r.type == "equation":
            s["equations"] += 1
        elif r.type == "diagram":
            s["diagrams"] += 1


# ─────────────────────────────────────────────────────────────
# Core processing pipeline
# ─────────────────────────────────────────────────────────────

def _process_image(
    image_bgr: np.ndarray,
    groq_client: Any,
    settings: dict,
    step_placeholder,
    filename: str = "page",
) -> dict | None:
    """
    Full pipeline for one image:
      warp → detect_regions → (optional: disable eq/diag) → update stats

    Returns a page_result dict or None on error.
    """
    try:
        # Step 0 — detect & straighten document
        step_placeholder.empty()
        with step_placeholder.container():
            render_steps(0)
        warped = detect_and_warp_document(image_bgr)

        # Step 1 — read handwriting / detect regions via Groq
        step_placeholder.empty()
        with step_placeholder.container():
            render_steps(1)
        regions: RegionList = detect_regions(warped, groq_client)

        # Step 2 — filter regions per user settings
        step_placeholder.empty()
        with step_placeholder.container():
            render_steps(2)

        filtered: List[RegionInfo] = []
        for r in regions:
            if r.type == "equation" and not settings.get("equation_detection", True):
                # Convert to text placeholder if eq detection disabled
                filtered.append(RegionInfo(
                    type="text", content="[ Equation — detection disabled ]",
                    top=r.top, left=r.left, width=r.width, height=r.height,
                    index=r.index,
                ))
            elif r.type == "diagram" and not settings.get("diagram_detection", True):
                filtered.append(RegionInfo(
                    type="text", content="[ Diagram — detection disabled ]",
                    top=r.top, left=r.left, width=r.width, height=r.height,
                    index=r.index,
                ))
            else:
                filtered.append(r)

        # Step 3 — build preview
        step_placeholder.empty()
        with step_placeholder.container():
            render_steps(3)

        _update_stats(filtered)

        result = {
            "regions": filtered,
            "original_bgr": image_bgr,
            "warped_bgr": warped,
            "filename": filename,
            "fallback": getattr(regions, "fallback", False),
        }
        step_placeholder.empty()
        return result

    except Exception as exc:
        step_placeholder.empty()
        st.error(f"❌ Failed to process **{filename}**: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# Plain-text helper (for webcam / legacy paths)
# ─────────────────────────────────────────────────────────────

def _page_result_to_plain_text(result: dict) -> str:
    """Concatenate all text region content in a page result."""
    parts = []
    for r in result.get("regions", []):
        if r.type == "text" and r.content:
            parts.append(r.content.strip())
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Aggregate stats computed from session state
# ─────────────────────────────────────────────────────────────

def _compute_display_stats() -> tuple[int, int, int, int]:
    pages = len(st.session_state["page_results"])
    equations = sum(
        sum(1 for r in pr["regions"] if r.type == "equation")
        for pr in st.session_state["page_results"]
    )
    diagrams = sum(
        sum(1 for r in pr["regions"] if r.type == "diagram")
        for pr in st.session_state["page_results"]
    )
    words = sum(
        sum(len((r.content or "").split()) for r in pr["regions"] if r.type == "text")
        for pr in st.session_state["page_results"]
    )
    return pages, equations, diagrams, words


# ─────────────────────────────────────────────────────────────
# Tab: Upload
# ─────────────────────────────────────────────────────────────

def _tab_upload(ocr_engine, settings: dict) -> None:
    st.markdown("### 📁 Upload Handwritten Note Images")
    st.markdown(
        "Drag and drop one or more images below. "
        "Supported: **PNG, JPG, JPEG, BMP, TIFF**"
    )

    uploaded_files = st.file_uploader(
        "Drop images here or click to browse",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center;padding:40px 0;color:#9CA3AF;font-size:0.9rem;">
                ☁️ Drag &amp; drop your handwritten note images here<br>
                <span style="font-size:0.78rem">Supports multiple pages at once</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Thumbnail + quality preview grid ─────────────────────
    st.markdown(f"**{len(uploaded_files)} file(s) selected — preview:**")
    thumb_cols = st.columns(min(len(uploaded_files), 4))
    images_bgr: List[np.ndarray] = []

    for i, f in enumerate(uploaded_files):
        try:
            img = uploaded_file_to_cv2_image(f)
            images_bgr.append(img)
            col = thumb_cols[i % 4]
            with col:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f.name, width="stretch")
                render_quality_badge(img)
        except Exception as e:
            images_bgr.append(None)
            st.warning(f"Could not load {f.name}: {e}")

    st.markdown("---")

    if st.button("🚀 Process All Images", type="primary", use_container_width=True):
        new_results = []
        step_ph = st.empty()
        prog = st.progress(0.0, text="Starting…")
        total = len(uploaded_files)

        for idx, (f, img) in enumerate(zip(uploaded_files, images_bgr)):
            if img is None:
                prog.progress((idx + 1) / total, text=f"Skipping {f.name} (load error)…")
                continue

            prog.progress(idx / total, text=f"Processing {f.name} ({idx+1}/{total})…")

            result = _process_image(
                img,
                ocr_engine.client,
                settings,
                step_ph,
                filename=f.name,
            )
            if result:
                new_results.append(result)

        prog.progress(1.0, text="Done!")
        step_ph.empty()

        st.session_state["page_results"].extend(new_results)

        if new_results:
            st.success(f"✅ Processed {len(new_results)} page(s) successfully.")
            if any(r.get("fallback") for r in new_results):
                st.info(
                    "ℹ️ One or more pages could not return structured JSON — "
                    "they were imported as plain text. Results are still usable."
                )

            # ── Side-by-side preview ──────────────────────────
            st.markdown("---")
            st.markdown("### 🔎 Results Preview")
            pages, equations, diagrams, words = _compute_display_stats()
            render_count_cards(pages, equations, diagrams, words)

            for result in new_results:
                with st.expander(f"📄 {result['filename']}", expanded=True):
                    left_col, right_col = st.columns([1, 1], gap="large")

                    with left_col:
                        st.markdown(
                            '<div class="ns-panel"><h4>Original Image</h4>',
                            unsafe_allow_html=True,
                        )
                        rgb = cv2.cvtColor(result["original_bgr"], cv2.COLOR_BGR2RGB)
                        st.image(rgb, width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with right_col:
                        st.markdown(
                            '<div class="ns-panel"><h4>Extracted Content</h4>',
                            unsafe_allow_html=True,
                        )
                        render_region_preview(
                            result["regions"], result["original_bgr"]
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No content was extracted. Check image quality and try again.")


# ─────────────────────────────────────────────────────────────
# Tab: Webcam
# ─────────────────────────────────────────────────────────────

def _tab_webcam(ocr_engine, settings: dict) -> None:
    st.markdown("### 📷 Scan Page Using Webcam")
    st.markdown(
        "Position your handwritten page in front of the camera. "
        "Ensure good lighting and the page fills most of the frame."
    )

    camera_image = st.camera_input("Capture handwritten page")

    if camera_image is None:
        return

    img_bgr = camera_input_to_cv2_image(camera_image)

    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.markdown("**Captured image**")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, width="stretch")
        render_quality_badge(img_bgr)

    with right_col:
        if st.button("🔍 Extract from Webcam", type="primary", use_container_width=True):
            step_ph = st.empty()
            result = _process_image(
                img_bgr,
                ocr_engine.client,
                settings,
                step_ph,
                filename="webcam_capture",
            )
            step_ph.empty()

            if result:
                st.session_state["page_results"].append(result)
                st.success("✅ Webcam page added to your document.")
                render_region_preview(result["regions"], result["original_bgr"])
            else:
                st.warning("Could not extract content. Try again with better lighting.")


# ─────────────────────────────────────────────────────────────
# Tab: Review
# ─────────────────────────────────────────────────────────────

def _tab_review() -> None:
    st.markdown("### 🔍 Review & Edit Extracted Content")

    page_results: List[dict] = st.session_state.get("page_results", [])

    if not page_results:
        st.info(
            "No pages processed yet. Use the **Upload** or **Webcam** tab to scan pages."
        )
        return

    pages, equations, diagrams, words = _compute_display_stats()
    render_count_cards(pages, equations, diagrams, words)
    st.markdown("---")

    st.markdown(
        "Review each page below. Edit the text in any region. "
        "Equations and diagrams are shown as thumbnails at their original position."
    )

    updated_results = []
    for page_idx, result in enumerate(page_results):
        label = f"📄 Page {page_idx + 1} — {result.get('filename', 'page')}"
        with st.expander(label, expanded=(page_idx == 0)):
            left_col, right_col = st.columns([1, 1], gap="large")

            # Left: original image
            with left_col:
                st.markdown("**Original scan**")
                rgb = cv2.cvtColor(result["original_bgr"], cv2.COLOR_BGR2RGB)
                st.image(rgb, width="stretch")

            # Right: editable regions
            with right_col:
                st.markdown("**Extracted content** *(edit text below)*")
                updated_regions: List[RegionInfo] = []

                for reg_idx, region in enumerate(result["regions"]):
                    if region.type == "text":
                        pill = "🔤 Text"
                        new_content = st.text_area(
                            pill,
                            value=(region.content or "").strip(),
                            height=100,
                            key=f"p{page_idx}_r{reg_idx}_text",
                            label_visibility="visible",
                        )
                        updated_regions.append(RegionInfo(
                            type=region.type,
                            content=new_content,
                            top=region.top, left=region.left,
                            width=region.width, height=region.height,
                            index=region.index,
                        ))

                    elif region.type == "equation":
                        st.markdown("**📐 Equation**")
                        try:
                            from image_cropper import crop_region, thumbnail, region_to_jpeg_bytes
                            crop = crop_region(result["original_bgr"], region)
                            thumb = thumbnail(crop, max_dim=300)
                            jpeg = region_to_jpeg_bytes(thumb, quality=85)
                            st.image(jpeg, caption="Equation region", width="content")
                        except Exception:
                            st.caption("[ Could not render equation thumbnail ]")
                        updated_regions.append(region)

                    elif region.type == "diagram":
                        st.markdown("**🖼️ Diagram / Figure**")
                        try:
                            from image_cropper import crop_region, thumbnail, region_to_jpeg_bytes
                            crop = crop_region(result["original_bgr"], region)
                            thumb = thumbnail(crop, max_dim=300)
                            jpeg = region_to_jpeg_bytes(thumb, quality=85)
                            st.image(jpeg, caption="Diagram region", width="content")
                        except Exception:
                            st.caption("[ Could not render diagram thumbnail ]")
                        updated_regions.append(region)

                    st.markdown(
                        "<hr style='border:none;border-top:1px solid #F3F4F6;margin:6px 0'>",
                        unsafe_allow_html=True,
                    )

                result = dict(result)          # shallow copy
                result["regions"] = updated_regions
            updated_results.append(result)

    st.session_state["page_results"] = updated_results

    st.markdown("---")
    if st.button("🗑️ Clear All Pages", type="secondary"):
        st.session_state["page_results"] = []
        st.session_state["stats"] = {"pages": 0, "words": 0, "equations": 0, "diagrams": 0}
        st.success("All pages cleared.")
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Tab: Export
# ─────────────────────────────────────────────────────────────

def _tab_export(settings: dict) -> None:
    st.markdown("### 📤 Export Document")

    page_results: List[dict] = st.session_state.get("page_results", [])

    if not page_results:
        st.info("No pages to export yet. Process images first.")
        return

    pages, equations, diagrams, words = _compute_display_stats()
    render_count_cards(pages, equations, diagrams, words)

    st.markdown("---")
    st.markdown("#### 📋 Document Outline")

    # Build outline preview
    outline_html = "<div class='ns-outline'><ul style='margin:0;padding-left:20px'>"
    for i, result in enumerate(page_results):
        fname = result.get("filename", f"page_{i+1}")
        eq_count   = sum(1 for r in result["regions"] if r.type == "equation")
        diag_count = sum(1 for r in result["regions"] if r.type == "diagram")
        txt_words  = sum(len((r.content or "").split()) for r in result["regions"] if r.type == "text")

        badges = ""
        if eq_count:
            badges += f' <span class="ns-pill equation">⚛ {eq_count} eq</span>'
        if diag_count:
            badges += f' <span class="ns-pill diagram">🖼 {diag_count} fig</span>'

        outline_html += (
            f"<li><b>Page {i+1}</b> · {fname} "
            f"· {txt_words} words{badges}</li>"
        )

    outline_html += "</ul></div>"
    st.markdown(outline_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Export format options ──────────────────────────────────
    st.markdown("#### 📥 Download")
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("**📄 Word Document (.docx)**")
        st.markdown(
            "Includes cropped equation and diagram images "
            "at their exact positions, styled headings, and lists."
        )
        if st.button("⚙️ Generate Word Document", type="primary", use_container_width=True):
            with st.spinner("Building Word document…"):
                try:
                    buf: io.BytesIO = create_docx_document(
                        page_results,
                        spell_check=settings.get("spell_check", True),
                    )
                    raw = buf.getvalue()
                    st.session_state["docx_bytes"]   = raw
                    st.session_state["docx_size_kb"] = max(1, len(raw) // 1024)
                    # invalidate old txt so stats stay fresh
                    st.session_state["txt_content"] = None
                except Exception as exc:
                    st.error(f"Failed to generate Word document: {exc}")

        # Download button lives OUTSIDE the generate-button block
        # so it survives the rerun triggered by clicking it
        if st.session_state.get("docx_bytes"):
            st.success(f"✅ Document ready — {st.session_state['docx_size_kb']} KB")
            st.download_button(
                label="⬇️ Download handwritten_notes.docx",
                data=st.session_state["docx_bytes"],
                file_name="handwritten_notes.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
                use_container_width=True,
            )

    with right_col:
        st.markdown("**📝 Plain Text (.txt)**")
        st.markdown(
            "All transcribed text only — no images. "
            "Useful for further editing or pasting into other apps."
        )
        if st.button("⚙️ Generate Plain Text", type="secondary", use_container_width=True):
            lines = []
            for i, result in enumerate(page_results):
                lines.append(f"=== Page {i+1}: {result.get('filename','page')} ===\n")
                for region in result["regions"]:
                    if region.type == "text" and region.content:
                        lines.append(region.content.strip())
                    elif region.type == "equation":
                        lines.append("[ Equation ]")
                    elif region.type == "diagram":
                        lines.append("[ Figure / Diagram ]")
                lines.append("")
            full_text = "\n".join(lines)
            st.session_state["txt_content"]  = full_text.encode("utf-8")
            st.session_state["txt_size_kb"]  = max(1, len(st.session_state["txt_content"]) // 1024)

        # Same pattern: download button outside the generate block
        if st.session_state.get("txt_content"):
            st.success(f"✅ Text ready — {st.session_state['txt_size_kb']} KB")
            st.download_button(
                label="⬇️ Download handwritten_notes.txt",
                data=st.session_state["txt_content"],
                file_name="handwritten_notes.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown(
        """
        <div style="background:#FFF8E6;border:1px solid #FFD966;border-radius:10px;
                    padding:12px 16px;font-size:0.85rem;color:#7A5700">
            <b>💡 PDF export</b> — PDF generation requires a headless browser or a paid
            conversion service and is not included to keep this app free-tier compatible.
            You can print the downloaded .docx file to PDF using Microsoft Word or LibreOffice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Handwritten Notes Scanner",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    _init_session()

    # ── Build OCR engine ─────────────────────────────────────
    api_connected = False
    ocr_engine = None
    try:
        ocr_engine = build_ocr_engine()
        api_connected = ocr_engine.backend == "groq"
    except Exception as exc:
        st.error(f"⚠️ Could not initialise Groq API: {exc}")

    # ── Header ───────────────────────────────────────────────
    render_header(api_connected)

    if not api_connected or ocr_engine is None:
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────
    settings = render_sidebar(api_connected, st.session_state["stats"])

    # ── Tabs ─────────────────────────────────────────────────
    tab_upload, tab_webcam, tab_review, tab_export = st.tabs([
        "📁 Upload Images",
        "📷 Webcam Scan",
        "🔍 Review & Edit",
        "📤 Export",
    ])

    with tab_upload:
        _tab_upload(ocr_engine, settings)

    with tab_webcam:
        _tab_webcam(ocr_engine, settings)

    with tab_review:
        _tab_review()

    with tab_export:
        _tab_export(settings)


if __name__ == "__main__":
    main()
