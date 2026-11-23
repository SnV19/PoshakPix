import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from modules.preprocessing import (
    load_image, preprocess_image, get_image_information
)
from modules.segmentation import (
    grabcut_segmentation, rembg_segmentation,
    apply_mask, get_segmentation_statistics
)
from modules.morphology import (
    refine_mask, erosion, dilation, opening, closing,
    remove_small_components, draw_contours
)
from modules.background_generation import (
    create_background, composite_garment, inpaint_image
)
from modules.consistency import evaluate_garment
from modules.evaluation import (
    compare_segmentation_methods, measure_execution_time
)
from utils.image_utils import (
    rgb_to_pil, rgba_to_pil, create_download_buffer
)
from utils.visualization import (
    plot_image_comparison, plot_metric_comparison
)


# ============================================================
#
# Keeps the existing processing modules and workflow intact.
# ============================================================

st.set_page_config(
    page_title="PoshakPix",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --ink: #20231f;
    --muted: #777a70;
    --paper: #f5f3ed;
    --panel: #fffefa;
    --line: #dedbd1;
    --olive: #59634a;
    --lime: #d8e2b8;
    --rust: #b96e4e;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1450px;
    padding: 2.2rem 3.2rem 4rem;
}
[data-testid="stSidebar"] {
    background: #20251f !important;
    border-right: 1px solid #363d32;
}
[data-testid="stSidebar"] * { color: #f5f3ed !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.55rem 0.75rem;
    border-radius: 8px;
    font-size: 0.91rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #343c30;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.22rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {
    color: #dce7bb !important;
}
.brand-lockup {
    padding: 1.2rem 0.6rem 1.5rem;
    border-bottom: 1px solid #41483c;
    margin-bottom: 1.4rem;
}
.brand-mark {
    width: 42px; height: 42px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    background: #dce7bb; color: #20251f !important;
    font: 700 22px 'Playfair Display', serif;
    margin-bottom: 0.8rem;
}
.brand-name {
    font: 600 1.35rem 'Playfair Display', serif;
    letter-spacing: -0.02em;
}
.brand-sub {
    color: #b6bcae !important; font: 10px 'DM Mono', monospace;
    letter-spacing: .16em; text-transform: uppercase; margin-top: .3rem;
}
.eyebrow {
    font: 11px 'DM Mono', monospace; letter-spacing: .16em;
    text-transform: uppercase; color: var(--olive);
    margin-bottom: .7rem;
}
.hero-panel {
    background: #e8e8dc;
    border: 1px solid #d8d9ca;
    border-radius: 5px;
    padding: 2.4rem 2.6rem;
    position: relative;
    overflow: hidden;
}
.hero-panel:after {
    content: "A"; position: absolute; right: 3rem; top: -4rem;
    font: 22rem 'Playfair Display', serif;
    color: rgba(89,99,74,.07); line-height: 1;
}
.hero-title {
    font: 600 clamp(2.6rem, 5vw, 4.7rem)/1.02 'Playfair Display', serif;
    letter-spacing: -.045em; max-width: 760px; color: #22271f;
}
.hero-copy {
    color: #656b5d; font-size: 1.02rem; max-width: 610px;
    line-height: 1.7; margin-top: 1rem;
}
.section-title {
    font: 600 2.35rem/1.12 'Playfair Display', serif;
    letter-spacing: -.035em; color: #242820; margin: .15rem 0 .35rem;
}
.section-copy { color: #74776d; margin-bottom: 1.4rem; }
.rule { border-top: 1px solid var(--line); margin: 1.5rem 0; }
.info-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 5px; padding: 1.2rem 1.25rem; min-height: 145px;
}
.info-kicker {
    color: var(--olive); font: 10px 'DM Mono', monospace;
    letter-spacing: .13em; text-transform: uppercase;
}
.info-title {
    font: 600 1.3rem 'Playfair Display', serif;
    margin: .55rem 0 .35rem; color: var(--ink);
}
.info-copy { color: var(--muted); font-size: .9rem; line-height: 1.55; }
.stat-card {
    background: #fffefa; border: 1px solid var(--line);
    padding: 1rem 1.15rem; border-radius: 5px;
}
.stat-label {
    color: #777a70; font: 10px 'DM Mono', monospace;
    text-transform: uppercase; letter-spacing: .12em;
}
.stat-value {
    font: 600 2rem 'Playfair Display', serif; color: #343d2c;
    margin-top: .25rem;
}
.step-tag {
    color: #a6ad91; font: 11px 'DM Mono', monospace;
    letter-spacing: .12em;
}
div.stButton > button, div.stDownloadButton > button {
    background: #59634a !important; color: #fffefa !important;
    border: 1px solid #59634a !important; border-radius: 3px !important;
    padding: .58rem 1.05rem !important; font-weight: 600 !important;
    min-height: 2.65rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background: #3f4935 !important; border-color: #3f4935 !important;
}
div[data-testid="stFileUploader"] {
    background: #fffefa; border: 1px dashed #b9bcae;
    border-radius: 5px; padding: .8rem;
}
div[data-testid="stMetric"] {
    background: #fffefa; border: 1px solid var(--line);
    border-radius: 5px; padding: .8rem 1rem;
}
div[data-testid="stMetric"] label {
    color: #777a70 !important; font: 10px 'DM Mono', monospace !important;
    text-transform: uppercase; letter-spacing: .1em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #343d2c !important; font: 600 2rem 'Playfair Display', serif !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line); border-radius: 5px; overflow: hidden;
}
h1, h2, h3 { color: #252920 !important; }
p, li { color: #53584e; }
[data-testid="stCaptionContainer"] { color: #85877d !important; }
.stAlert { border-radius: 4px; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------- Persistent workflow state ----------
DEFAULTS = {
    "image": None,
    "grabcut_mask": None,
    "rembg_mask": None,
    "active_mask": None,
    "processed_image": None,
    "refined_mask": None,
    "original_rgba": None,
    "preprocessing": None,
    "experiment_table": None,
    "consistency_metrics": None,
    "refinement_stages": None,
    "selected_background": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div class="brand-lockup">
      <div class="brand-mark">FS</div>
      <div class="brand-name">FashionStudio AI</div>
      <div class="brand-sub">Fashion meets pixels</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font:10px DM Mono;letter-spacing:.16em;color:#aab29e;"
        "margin:0 0 .6rem .7rem'>WORKSPACE</div>",
        unsafe_allow_html=True
    )
    page = st.radio(
        "Workspace navigation",
        [
            "Studio Overview",
            "01  Image Library",
            "02  Image Preparation",
            "03  Garment Extraction",
            "04  Mask Workshop",
            "05  Backdrop Composer",
            "06  Quality Review",
            "07  Experiment Lab",
            "Project Notes",
        ],
        label_visibility="collapsed"
    )
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top:1px solid #41483c;padding:1rem .6rem 0;"
        "color:#aeb5a4;font:10px DM Mono;letter-spacing:.1em'>"
        "HSR-AIML-055<br>GENERATIVE FASHION IMAGE STUDIO</div>",
        unsafe_allow_html=True
    )


# ---------- Helpers ----------
def section_header(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{description}</div>', unsafe_allow_html=True)


def show_rgb(image, caption=None):
    st.image(image, caption=caption, use_container_width=True)


def require_image():
    if st.session_state.image is None:
        st.warning("Add a fashion photograph in Image Library to begin.")
        return False
    return True


def require_mask():
    if st.session_state.active_mask is None:
        st.warning("Run garment extraction and select a mask first.")
        return False
    return True


def save_output(image, filename):
    os.makedirs("outputs", exist_ok=True)
    buffer = create_download_buffer(image)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="image/png"
    )


def feature_card(number, title, description):
    st.markdown(f"""
    <div class="info-card">
      <div class="info-kicker">{number} / WORKFLOW</div>
      <div class="info-title">{title}</div>
      <div class="info-copy">{description}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 00 — STUDIO OVERVIEW
# ============================================================
if page == "Studio Overview":
    st.markdown("""
    <div class="eyebrow">DIGITAL IMAGE PROCESSING / PROJECT 055</div>
    <div class="hero-panel">
      <div class="hero-title">A cleaner canvas<br>for every garment.</div>
      <div class="hero-copy">
        An image-processing workspace for isolating fashion products,
        refining garment boundaries, composing studio backdrops and
        reviewing visual consistency.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("01", "Prepare", "Resize, denoise and normalize product photographs.")
    with c2:
        feature_card("02", "Extract", "Compare classical GrabCut with pretrained U2Net segmentation.")
    with c3:
        feature_card("03", "Compose", "Refine masks and place garments on studio backgrounds.")

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>YOUR CURRENT SESSION</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>Source image</div><div class='stat-value'>{'Loaded' if st.session_state.image is not None else 'Empty'}</div></div>", unsafe_allow_html=True)
    with s2:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>Active mask</div><div class='stat-value'>{'Ready' if st.session_state.active_mask is not None else 'Pending'}</div></div>", unsafe_allow_html=True)
    with s3:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>Final output</div><div class='stat-value'>{'Created' if st.session_state.processed_image is not None else 'Pending'}</div></div>", unsafe_allow_html=True)
    with s4:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>Evaluation</div><div class='stat-value'>{'Done' if st.session_state.consistency_metrics is not None else 'Pending'}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("#### The workflow")
    st.write("Image Library  →  Preparation  →  Extraction  →  Mask Workshop  →  Backdrop Composer  →  Quality Review")
    if st.session_state.image is not None:
        show_rgb(st.session_state.image, "Current source image")


# ============================================================
# 01 — IMAGE LIBRARY
# ============================================================
elif page == "01  Image Library":
    section_header("01 / SOURCE", "Image Library", "Bring a product photograph into the studio. JPG, JPEG and PNG are supported.")
    uploaded_file = st.file_uploader(
        "Drop a fashion product image here",
        type=["jpg", "jpeg", "png"],
        help="For the first test, choose a single garment with a reasonably clear outline."
    )
    if uploaded_file is not None:
        try:
            rgb, _ = load_image(uploaded_file)
            st.session_state.image = rgb
            for key in [
                "grabcut_mask", "rembg_mask", "active_mask",
                "processed_image", "refined_mask", "original_rgba",
                "preprocessing", "experiment_table",
                "consistency_metrics", "refinement_stages"
            ]:
                st.session_state[key] = None
            st.success("Source image added. Previous processing results were reset.")
        except Exception as error:
            st.error(f"Image loading failed: {error}")

    if st.session_state.image is not None:
        left, right = st.columns([1.5, 1])
        with left:
            show_rgb(st.session_state.image, "SOURCE / ORIGINAL")
        with right:
            info = get_image_information(st.session_state.image)
            st.markdown("#### Image properties")
            for key, value in info.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            st.caption("Tip: evenly lit product images with visible garment edges generally produce cleaner masks.")


# ============================================================
# 02 — IMAGE PREPARATION
# ============================================================
elif page == "02  Image Preparation":
    section_header("02 / DIP PIPELINE", "Image Preparation", "Standardize image dimensions and reduce noise before foreground extraction.")
    if require_image():
        image = st.session_state.image
        a, b = st.columns([1, 2])
        with a:
            max_size = st.slider("Maximum image dimension", 256, 1024, 768, step=64)
            run_prep = st.button("Prepare image", type="primary", use_container_width=True)
        with b:
            st.markdown("**Operations applied**")
            st.write("Aspect-ratio-preserving resize · Gaussian denoising · Intensity normalization")
        if run_prep:
            try:
                st.session_state.preprocessing = preprocess_image(image, max_size=max_size)
                st.success("Image preparation completed.")
            except Exception as error:
                st.error(f"Preprocessing failed: {error}")
        results = st.session_state.preprocessing
        if results is not None:
            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            cols = st.columns(4)
            for col, (name, img) in zip(cols, [
                ("Original", image),
                ("Resized", results["resized"]),
                ("Gaussian denoising", results["denoised"]),
                ("Normalized", results["normalized"])
            ]):
                with col:
                    show_rgb(img, name)


# ============================================================
# 03 — GARMENT EXTRACTION
# ============================================================
elif page == "03  Garment Extraction":
    section_header("03 / SEGMENTATION", "Garment Extraction", "Compare a classical graph-cut method with pretrained salient-object segmentation.")
    if require_image():
        image = st.session_state.image
        method = st.radio(
            "Extraction mode",
            ["GrabCut", "rembg U2Net", "Compare Both"],
            horizontal=True
        )
        if st.button("Run extraction", type="primary"):
            try:
                if method in ["GrabCut", "Compare Both"]:
                    result, elapsed = measure_execution_time(grabcut_segmentation, image)
                    st.session_state.grabcut_mask = result
                    st.session_state.grabcut_elapsed = elapsed
                if method in ["rembg U2Net", "Compare Both"]:
                    result, rgba = rembg_segmentation(image)
                    st.session_state.rembg_mask = result
                    st.session_state.original_rgba = rgba
                st.success("Extraction completed.")
            except Exception as error:
                st.error(str(error))

        if st.session_state.grabcut_mask is not None or st.session_state.rembg_mask is not None:
            cols = st.columns(2)
            with cols[0]:
                st.markdown("#### A / GrabCut")
                if st.session_state.grabcut_mask is not None:
                    st.image(st.session_state.grabcut_mask, caption="Binary foreground mask", clamp=True, use_container_width=True)
                    if "grabcut_elapsed" in st.session_state:
                        st.caption(f"Runtime: {st.session_state.grabcut_elapsed:.4f} seconds")
                else:
                    st.info("Not run in this session.")
            with cols[1]:
                st.markdown("#### B / U2Net via rembg")
                if st.session_state.rembg_mask is not None:
                    st.image(st.session_state.rembg_mask, caption="Binary foreground mask", clamp=True, use_container_width=True)
                else:
                    st.info("Not run in this session.")
            if st.session_state.original_rgba is not None:
                st.markdown("#### Transparent extraction preview")
                st.image(st.session_state.original_rgba, use_container_width=True)

        st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
        st.markdown("#### Choose the mask for the next stage")
        selected = st.selectbox("Segmentation result", ["GrabCut", "rembg U2Net"])
        if st.button("Set as active mask"):
            mask = st.session_state.grabcut_mask if selected == "GrabCut" else st.session_state.rembg_mask
            if mask is not None:
                st.session_state.active_mask = mask
                st.session_state.refined_mask = None
                st.session_state.processed_image = None
                st.success(f"{selected} is now the active mask.")
            else:
                st.warning("Run the selected method first.")


# ============================================================
# 04 — MASK WORKSHOP
# ============================================================
elif page == "04  Mask Workshop":
    section_header("04 / MORPHOLOGY", "Mask Workshop", "Clean small artifacts and close minor gaps using classical morphological operations.")
    if require_mask():
        mask = st.session_state.active_mask
        left, right = st.columns([1, 1])
        with left:
            kernel_size = st.slider("Structuring element size", 1, 15, 3, step=2)
            operation = st.selectbox(
                "Operation",
                ["Complete Refinement", "Erosion", "Dilation", "Opening", "Closing", "Remove Small Components"]
            )
            apply_operation = st.button("Apply morphology", type="primary", use_container_width=True)
        with right:
            st.markdown("**Operation guide**")
            st.write("- Erosion: shrinks foreground.")
            st.write("- Dilation: expands foreground.")
            st.write("- Opening: removes small foreground noise.")
            st.write("- Closing: fills small gaps.")
        if apply_operation:
            try:
                if operation == "Complete Refinement":
                    results = refine_mask(mask, kernel_size)
                    result = remove_small_components(results["refined_mask"], min_area=100)
                    st.session_state.refinement_stages = results
                elif operation == "Erosion":
                    result = erosion(mask, kernel_size)
                    st.session_state.refinement_stages = None
                elif operation == "Dilation":
                    result = dilation(mask, kernel_size)
                    st.session_state.refinement_stages = None
                elif operation == "Opening":
                    result = opening(mask, kernel_size)
                    st.session_state.refinement_stages = None
                elif operation == "Closing":
                    result = closing(mask, kernel_size)
                    st.session_state.refinement_stages = None
                else:
                    result = remove_small_components(mask)
                    st.session_state.refinement_stages = None
                st.session_state.refined_mask = result
                st.success("Mask operation completed.")
            except Exception as error:
                st.error(f"Morphology failed: {error}")

        if st.session_state.refined_mask is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.image(mask, caption="Active input mask", clamp=True, use_container_width=True)
            with c2:
                st.image(st.session_state.refined_mask, caption="Refined mask", clamp=True, use_container_width=True)
            stages = st.session_state.refinement_stages
            if stages is not None:
                st.markdown("#### Processing stages")
                cols = st.columns(3)
                for col, key, title in zip(
                    cols,
                    ["original_mask", "opened_mask", "refined_mask"],
                    ["Original", "After opening", "After closing"]
                ):
                    with col:
                        st.image(stages[key], caption=title, clamp=True, use_container_width=True)
            if st.button("Use refined mask for composition"):
                st.session_state.active_mask = st.session_state.refined_mask
                st.session_state.processed_image = None
                st.success("Refined mask selected as the active mask.")


# ============================================================
# 05 — BACKDROP COMPOSER
# ============================================================
elif page == "05  Backdrop Composer":
    section_header("05 / COMPOSITION", "Backdrop Composer", "Place the extracted garment on a clean studio background using alpha compositing.")
    if require_image() and require_mask():
        image = st.session_state.image
        mask = st.session_state.active_mask
        left, right = st.columns([1, 1])
        with left:
            background_type = st.selectbox(
                "Studio backdrop",
                ["White", "Light Grey", "Pastel Pink", "Pastel Blue", "Pastel Green", "Cream", "Black", "Gradient", "Custom"]
            )
            custom_color = (255, 255, 255)
            if background_type == "Custom":
                selected_color = st.color_picker("Choose backdrop colour", "#E8DDF5")
                custom_color = tuple(int(selected_color[i:i + 2], 16) for i in (1, 3, 5))
            add_shadow = st.checkbox("Add soft shadow", value=True)
            generate = st.button("Compose final image", type="primary", use_container_width=True)
        with right:
            st.markdown("**Composition method**")
            st.write("The selected binary mask controls the garment transparency. The foreground is composited over the chosen background.")
            st.caption("The current implementation changes the backdrop; it does not synthesize new garment designs.")
        if generate:
            try:
                height, width = image.shape[:2]
                background = create_background(width, height, background_type, custom_color)
                output = composite_garment(image, mask, background, add_shadow)
                st.session_state.processed_image = output
                st.session_state.selected_background = background_type
                st.success("Studio image created.")
            except Exception as error:
                st.error(f"Composition failed: {error}")

        if st.session_state.processed_image is not None:
            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                show_rgb(image, "SOURCE IMAGE")
            with c2:
                show_rgb(st.session_state.processed_image, "COMPOSED RESULT")
            st.caption(f"Selected backdrop: {st.session_state.selected_background or 'Not specified'}")
            save_output(st.session_state.processed_image, "fashionstudio_output.png")

        with st.expander("Optional inpainting demonstration"):
            st.warning("This demonstration uses the garment mask as the inpainting region. It removes the garment area; it is not a garment defect-repair feature.")
            if st.button("Run inpainting demonstration"):
                try:
                    inpainted = inpaint_image(image, mask)
                    show_rgb(inpainted, "Inpainting result")
                except Exception as error:
                    st.error(f"Inpainting failed: {error}")


# ============================================================
# 06 — QUALITY REVIEW
# ============================================================
elif page == "06  Quality Review":
    section_header("06 / EVALUATION", "Quality Review", "Measure structural, pixel-level and colour-distribution similarity over the garment region.")
    if (
        st.session_state.image is not None
        and st.session_state.processed_image is not None
        and st.session_state.active_mask is not None
    ):
        original = st.session_state.image
        processed = st.session_state.processed_image
        mask = st.session_state.active_mask

        c1, c2 = st.columns(2)
        with c1:
            show_rgb(original, "ORIGINAL")
        with c2:
            show_rgb(processed, "PROCESSED")

        if st.button("Calculate quality metrics", type="primary"):
            try:
                with st.spinner("Evaluating garment region..."):
                    metrics = evaluate_garment(original, processed, mask)
                st.session_state.consistency_metrics = metrics
                st.success("Quality review completed.")
            except Exception as error:
                st.error(f"Evaluation failed: {error}")

        metrics = st.session_state.consistency_metrics
        if metrics is not None:
            a, b, c, d = st.columns(4)
            with a:
                st.metric("SSIM", metrics["SSIM"])
            with b:
                st.metric("MAE", metrics["MAE"])
            with c:
                st.metric("Histogram similarity", metrics["Histogram Similarity"])
            with d:
                st.metric("Mean Delta E 76", metrics["Mean Delta E 76"])

            st.markdown("#### How to read these scores")
            st.write("SSIM and histogram similarity: higher is generally better. MAE and Delta E 76: lower is generally better.")
            st.caption("These scores compare aligned image regions. High similarity is expected when the original garment pixels are retained during background replacement; it does not independently prove garment identity.")
            df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download quality report",
                data=df.to_csv(index=False),
                file_name="consistency_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Complete these steps first: upload an image, run segmentation, select a mask and create a composed result.")


# ============================================================
# 07 — EXPERIMENT LAB
# ============================================================
elif page == "07  Experiment Lab":
    section_header("07 / COMPARATIVE STUDY", "Experiment Lab", "Compare foreground coverage and execution time for the two segmentation approaches.")
    if require_image():
        image = st.session_state.image
        st.markdown("**Methods under comparison**")
        st.write("A — GrabCut (classical graph-cut segmentation)     B — rembg U2Net (pretrained neural segmentation)")
        if st.button("Run comparative experiment", type="primary"):
            try:
                with st.spinner("Running both segmentation methods..."):
                    grabcut_mask, grabcut_time = measure_execution_time(grabcut_segmentation, image)
                    rembg_result, rembg_time = measure_execution_time(rembg_segmentation, image)
                    rembg_mask, _ = rembg_result
                    comparison = compare_segmentation_methods(grabcut_mask, rembg_mask)
                    comparison["Execution Time (seconds)"] = [grabcut_time, rembg_time]
                    st.session_state.experiment_table = comparison
                st.success("Experiment completed.")
            except Exception as error:
                st.error(f"Experiment failed: {error}")

        df = st.session_state.experiment_table
        if df is not None:
            st.markdown("#### Results table")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Export experiment data (CSV)",
                data=df.to_csv(index=False),
                file_name="segmentation_comparison.csv",
                mime="text/csv"
            )
            st.markdown("#### Visual comparison")
            left, right = st.columns(2)
            with left:
                fig = plot_metric_comparison(
                    df["Method"].tolist(),
                    df["Foreground Percentage"].tolist(),
                    "Foreground coverage by method",
                    "Foreground (%)"
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with right:
                fig2 = plot_metric_comparison(
                    df["Method"].tolist(),
                    df["Execution Time (seconds)"].tolist(),
                    "Processing time by method",
                    "Seconds"
                )
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            st.info("Foreground coverage and runtime are descriptive measurements, not segmentation accuracy. IoU and Dice require matching ground-truth masks.")
    else:
        st.info("Upload an image in Image Library to run the experiment.")


# ============================================================
# PROJECT NOTES
# ============================================================
elif page == "Project Notes":
    section_header("PROJECT / HSR-AIML-055", "Project Notes", "Technical summary, methods and limitations for the academic demonstration.")
    st.markdown("#### Project identity")
    x, y = st.columns(2)
    with x:
        st.markdown("**Project:** Generative Fashion Product Image Studio with Identity and Garment Consistency Checks")
        st.markdown("**Course contribution:** Digital Image Processing and Computer Vision")
    with y:
        st.markdown("**Project ID:** HSR-AIML-055")
        st.markdown("**Industry reference:** Neuropixel.ai (domain reference only)")
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("#### Processing components")
    st.write("Image resizing · Gaussian filtering · GrabCut · U2Net via rembg · Binary masks · Erosion · Dilation · Opening · Closing · Connected components · Alpha compositing · Inpainting")
    st.markdown("#### Evaluation")
    st.write("SSIM · Mean Absolute Error · Histogram similarity · CIE Delta E 76 · Foreground coverage · Execution time · IoU and Dice when ground-truth masks are available")
    st.markdown("#### Known limitations")
    st.write("- GrabCut uses a rectangular initialization and may struggle with complex scenes.")
    st.write("- Pretrained segmentation may miss thin straps, lace, transparent fabric or similar-coloured regions.")
    st.write("- Morphological operations can remove fine details when the structuring element is too large.")
    st.write("- Current backdrop composition preserves the input garment; it does not generate a new garment or perform virtual try-on.")
    st.write("- The optional inpainting demonstration removes the selected mask region rather than repairing garment defects.")
    st.write("- CLIP-based identity evaluation is not part of the current active workflow.")
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.caption("Use actual experimental outputs in the report. Do not report IoU or Dice unless ground-truth masks were evaluated.")

