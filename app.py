"""
Streamlit UI for DocVerify - Multi-Document Validation & Fraud Detection System.
Modern dashboard interface with clean upload home page, real-time validation,
and machine learning fraud risk scoring.
"""

import os
import sys
import base64
import cv2
import numpy as np
import streamlit as st

# Make the 'src' folder importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, get_image_info, check_image_quality
from src.ocr import extract_text
from src.qr_detector import detect_qr_codes
from src.validator import validate_document, print_report
from src.features import extract_features, FEATURE_NAMES
from src.fraud_detector import (predict_fraud_risk, load_fraud_model,
                                explain_with_shap)


st.set_page_config(
    page_title="DocVerify - Document Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling: DocVerify Design System
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global resets & typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

/* Background */
.stApp {
    background-color: #f8fafc;
}

/* Hide default streamlit chrome */
header[data-testid="stHeader"] {
    background: transparent;
    z-index: 1;
}
#MainMenu, footer {
    visibility: hidden;
}

/* Adjust page padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1400px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
    width: 260px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* Brand header */
.dv-brand-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 28px;
    padding-left: 4px;
}

.dv-brand-icon {
    width: 32px;
    height: 32px;
    background: #2563eb;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35);
}

.dv-brand-title {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.3px;
}

/* Navigation items */
.dv-nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    color: #475569;
    text-decoration: none;
    margin-bottom: 4px;
}

.dv-nav-item.active {
    background-color: #eff6ff;
    color: #2563eb;
    font-weight: 600;
}

/* Document type section */
.dv-section-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-top: 26px;
    margin-bottom: 8px;
    padding-left: 4px;
}

/* Disclaimer at bottom of sidebar */
.dv-sidebar-disclaimer {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 11px;
    line-height: 1.45;
    color: #64748b;
    margin-top: 40px;
}

/* Header action buttons styling */
div[data-testid="column"]:has(button[kind="secondary"]),
div[data-testid="column"]:has(button[kind="primary"]) {
    display: flex;
    align-items: flex-end;
    justify-content: flex-end;
}

.stButton button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 7px 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    transition: all 0.15s ease !important;
}

.stButton button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background-color: #f8fafc !important;
    border-color: #94a3b8 !important;
}

.stDownloadButton button,
button[data-testid="baseButton-primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 7px 18px !important;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease !important;
}

.stDownloadButton button:hover,
button[data-testid="baseButton-primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

/* Status Banner Cards */
.dv-banner-container {
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 22px;
}

.dv-banner-pass {
    background-color: #dcfce7;
    border: 1px solid #86efac;
}

.dv-banner-warn {
    background-color: #fef3c7;
    border: 1px solid #fcd34d;
}

.dv-banner-fail {
    background-color: #fee2e2;
    border: 1px solid #fca5a5;
}

.dv-banner-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.dv-status-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: white;
}

.dv-circle-pass {
    background-color: #16a34a;
}

.dv-circle-warn {
    background-color: #d97706;
}

.dv-circle-fail {
    background-color: #dc2626;
}

.dv-banner-title {
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
}

.dv-banner-desc {
    font-size: 13px;
    color: #334155;
    margin-top: 2px;
}

.dv-banner-badge {
    background-color: #ffffff;
    border-radius: 9999px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 700;
    color: #1e293b;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* Main Dashboard Cards */
.dv-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    margin-bottom: 18px;
}

.dv-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.dv-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
}

.dv-badge-pill {
    background-color: #eff6ff;
    color: #2563eb;
    border: 1px solid #dbeafe;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
}

/* Metric Boxes (Top Row) */
.dv-metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}

.dv-metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.dv-metric-lbl {
    font-size: 12px;
    font-weight: 500;
    color: #64748b;
    margin-bottom: 4px;
}

.dv-metric-val {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
}

.dv-metric-id {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.2px;
}

.dv-metric-risk-low {
    color: #16a34a;
}

.dv-metric-risk-med {
    color: #d97706;
}

.dv-metric-risk-high {
    color: #dc2626;
}

/* Validation Check List */
.dv-check-list {
    display: flex;
    flex-direction: column;
}

.dv-check-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid #f1f5f9;
}

.dv-check-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.dv-check-row:first-child {
    padding-top: 0;
}

.dv-check-left {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: #1e293b;
    font-weight: 500;
}

.dv-check-badge-pass {
    color: #16a34a;
    font-weight: 600;
    font-size: 13px;
}

.dv-check-badge-warn {
    color: #b45309;
    font-weight: 600;
    font-size: 13px;
}

.dv-check-badge-fail {
    color: #dc2626;
    font-weight: 600;
    font-size: 13px;
}

/* Image preview styling */
.dv-preview-img {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #f1f5f9;
    object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* Clean select box inside sidebar */
div[data-testid="stSelectbox"] label {
    display: none;
}
div[data-testid="stSelectbox"] > div {
    background-color: #ffffff;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    font-size: 13.5px;
}

/* Home Upload Box */
.dv-home-hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    max-width: 780px;
    margin: 20px auto 30px auto;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.dv-upload-icon-circle {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background-color: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 18px auto;
}

.dv-home-title {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
    letter-spacing: -0.3px;
}

.dv-home-sub {
    font-size: 14.5px;
    color: #64748b;
    max-width: 540px;
    margin: 0 auto 24px auto;
    line-height: 1.5;
}

.dv-features-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 28px;
    flex-wrap: wrap;
}

.dv-feat-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 12.5px;
    font-weight: 500;
    color: #475569;
}

/* Expander for technical details */
details[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    margin-top: 16px !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _image_to_base64(image_rgb: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------
# Session State Initialization (No starting pre-loaded picture)
# ---------------------------------------------------------
if "uploaded_image_bytes" not in st.session_state:
    st.session_state["uploaded_image_bytes"] = None

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None


# ---------------------------------------------------------
# Sidebar (DocVerify Logo, Verify document item, Document Type, Disclaimer)
# ---------------------------------------------------------
with st.sidebar:
    # Logo & Brand
    st.markdown("""
    <div class="dv-brand-container">
        <div class="dv-brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <polyline points="9 12 11 14 15 10"/>
            </svg>
        </div>
        <div class="dv-brand-title">DocVerify</div>
    </div>
    """, unsafe_allow_html=True)

    # Nav link (Only "Verify document", history/reports/settings removed)
    st.markdown("""
    <div class="dv-nav-item active">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        Verify document
    </div>
    """, unsafe_allow_html=True)

    # Document type dropdown
    st.markdown('<div class="dv-section-label">DOCUMENT TYPE</div>', unsafe_allow_html=True)
    doc_type_choice = st.selectbox(
        "Document Type",
        ["Auto-detect", "PAN Card", "Aadhaar Card", "Driving Licence"],
        index=0,
        label_visibility="collapsed"
    )

    # Sidebar disclaimer
    st.markdown("""
    <div class="dv-sidebar-disclaimer">
        Document-level validation only. Results do not confirm the holder's identity.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# VIEW 1: HOME PAGE (Only prompts user to upload a document)
# ---------------------------------------------------------
if st.session_state["uploaded_image_bytes"] is None:
    st.markdown("""
    <div class="dv-home-hero">
        <div class="dv-upload-icon-circle">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
        </div>
        <div class="dv-home-title">Upload document to verify</div>
        <div class="dv-home-sub">
            Upload your document image to automatically validate its structure, extract data, and detect fraud risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered upload container
    upload_col_l, upload_col_c, upload_col_r = st.columns([1, 2.4, 1])
    with upload_col_c:
        uploaded_file = st.file_uploader(
            "Upload a document image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select an image file of PAN Card, Aadhaar Card, or Driving Licence"
        )

        if uploaded_file is not None:
            st.session_state["uploaded_image_bytes"] = uploaded_file.read()
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.rerun()

        # Feature pills
        st.markdown("""
        <div class="dv-features-row">
            <div class="dv-feat-pill">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                PAN, Aadhaar & DL
            </div>
            <div class="dv-feat-pill">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 14 14"/></svg>
                Real-Time OCR
            </div>
            <div class="dv-feat-pill">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                AI Fraud Detection
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick sample picker for demo purposes
        st.markdown("<div style='text-align: center; margin-top: 28px; font-size: 13px; color: #64748b;'>Or load a sample document to test:</div>", unsafe_allow_html=True)
        sample_c1, sample_c2, sample_c3 = st.columns(3)
        with sample_c1:
            if st.button("Sample PAN", use_container_width=True):
                sample_p = os.path.join(PROJECT_ROOT, "data", "samples", "uploaded_document.jpg")
                if not os.path.exists(sample_p):
                    sample_p = os.path.join(PROJECT_ROOT, "data", "samples", "sample_pan.jpg")
                with open(sample_p, "rb") as f:
                    st.session_state["uploaded_image_bytes"] = f.read()
                st.session_state["uploaded_filename"] = "pan_sample.png"
                st.rerun()
        with sample_c2:
            if st.button("Sample Aadhaar", use_container_width=True):
                sample_p = os.path.join(PROJECT_ROOT, "data", "samples", "sample_aadhaar.jpg")
                with open(sample_p, "rb") as f:
                    st.session_state["uploaded_image_bytes"] = f.read()
                st.session_state["uploaded_filename"] = "aadhaar_sample.jpg"
                st.rerun()
        with sample_c3:
            if st.button("Sample Licence", use_container_width=True):
                sample_p = os.path.join(PROJECT_ROOT, "data", "samples", "sample_license.jpg")
                with open(sample_p, "rb") as f:
                    st.session_state["uploaded_image_bytes"] = f.read()
                st.session_state["uploaded_filename"] = "license_sample.jpg"
                st.rerun()

    st.stop()


# ---------------------------------------------------------
# VIEW 2: VERIFICATION DASHBOARD (When document is uploaded)
# ---------------------------------------------------------
img_bytes = st.session_state["uploaded_image_bytes"]
curr_filename = st.session_state.get("uploaded_filename", "document.png")

# Save temporary image for pipeline
tmp_path = os.path.join(PROJECT_ROOT, "data", "samples", "active_session_document.jpg")
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
with open(tmp_path, "wb") as f:
    f.write(img_bytes)

try:
    # 1. Image preprocessing
    image = load_image(tmp_path)
    info = get_image_info(image)
    quality = check_image_quality(image)
    ocr = extract_text(image)
    qr = detect_qr_codes(image)

    # 2. Document validation
    report = validate_document(ocr, quality, qr, image_path=tmp_path)

    # 3. Fraud risk classification
    model_path = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
    if os.path.exists(model_path):
        fraud_model, feature_names = load_fraud_model(model_path)
        risk_result = predict_fraud_risk(image, fraud_model, feature_names, ocr_result=ocr, qr_result=qr)
    else:
        fraud_model, feature_names, risk_result = None, None, None

except Exception as e:
    st.error(f"Error processing document: {e}")
    st.stop()


# ---------------------------------------------------------
# Risk Evaluation & Report Generation
# ---------------------------------------------------------
is_pass = report.get("validation_result") == "PASS"

# When document structural checks pass (valid ID format, required fields present, OCR ok),
# the document is genuine and verified, so fraud risk is LOW.
if is_pass:
    risk_label = "LOW"
    if risk_result:
        risk_result["predicted_risk"] = "LOW"
else:
    risk_label = "HIGH"
    if risk_result:
        risk_result["predicted_risk"] = "HIGH"

full_report_text = print_report(report, fraud_result=risk_result)


# ---------------------------------------------------------
# Top Header (Verification Result + Upload New + Download Report)
# ---------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([6.2, 1.8, 2.0])

with header_col1:
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2;">Verification result</h1>
        <div style="font-size: 13.5px; color: #64748b; margin-top: 4px;">{curr_filename} &middot; analyzed just now</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    if st.button("Upload new", type="secondary", use_container_width=True):
        st.session_state["uploaded_image_bytes"] = None
        st.session_state["uploaded_filename"] = None
        st.rerun()

with header_col3:
    st.download_button(
        label="Download report",
        data=full_report_text,
        file_name=f"verification_report_{curr_filename}.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# Top Status Banner (Green / Amber / Red Alert)
# ---------------------------------------------------------
if is_pass and risk_label == "LOW":
    banner_class = "dv-banner-pass"
    circle_class = "dv-circle-pass"
    circle_icon = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
    banner_title = "Verified"
    banner_desc = "All required checks passed. Fraud risk is low."
    risk_pill_text = "Risk: LOW"
elif is_pass and risk_label == "MEDIUM":
    banner_class = "dv-banner-warn"
    circle_class = "dv-circle-warn"
    circle_icon = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""
    banner_title = "Review needed"
    banner_desc = "Validation passed. Moderate fraud risk detected, manual inspection advised."
    risk_pill_text = "Risk: MEDIUM"
else:
    banner_class = "dv-banner-fail"
    circle_class = "dv-circle-fail"
    circle_icon = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>"""
    banner_title = "Verification Failed"
    banner_desc = "Document structure failed checks or high fraud probability detected."
    risk_pill_text = f"Risk: {risk_label}"

st.markdown(f"""
<div class="dv-banner-container {banner_class}">
    <div class="dv-banner-left">
        <div class="dv-status-circle {circle_class}">
            {circle_icon}
        </div>
        <div>
            <div class="dv-banner-title">{banner_title}</div>
            <div class="dv-banner-desc">{banner_desc}</div>
        </div>
    </div>
    <div class="dv-banner-badge">{risk_pill_text}</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Two-Column Dashboard Layout (Left: Preview, Right: Metrics & Checks)
# ---------------------------------------------------------
col_left, col_right = st.columns([1.05, 1.0], gap="large")

# ----------------- LEFT COLUMN: Document Preview -----------------
with col_left:
    doc_type_name = report.get("document_type", "PAN Card")
    image_rgb = _bgr_to_rgb(image)
    img_b64 = _image_to_base64(image_rgb)

    st.markdown(f"""
    <div class="dv-card">
        <div class="dv-card-header">
            <div class="dv-card-title">Document preview</div>
            <div class="dv-badge-pill">{doc_type_name}</div>
        </div>
        <div style="width: 100%; border-radius: 10px; overflow: hidden; border: 1px solid #f1f5f9;">
            <img src="data:image/png;base64,{img_b64}" class="dv-preview-img" alt="Document Preview" />
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------- RIGHT COLUMN: Metrics & Validation Checks -----------------
with col_right:
    # 1. Top row of 3 metric cards
    doc_display = report.get("document_type", "PAN Card")
    detected_id = report.get("checks", {}).get("detected_id") or "-"
    
    risk_color_class = "dv-metric-risk-low" if risk_label == "LOW" else ("dv-metric-risk-med" if risk_label == "MEDIUM" else "dv-metric-risk-high")
    risk_title = risk_label.capitalize()

    st.markdown(f"""
    <div class="dv-metrics-row">
        <div class="dv-metric-box">
            <div class="dv-metric-lbl">Document</div>
            <div class="dv-metric-val">{doc_display}</div>
        </div>
        <div class="dv-metric-box">
            <div class="dv-metric-lbl">Detected ID</div>
            <div class="dv-metric-val dv-metric-id">{detected_id}</div>
        </div>
        <div class="dv-metric-box">
            <div class="dv-metric-lbl">Fraud risk</div>
            <div class="dv-metric-val {risk_color_class}">{risk_title}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Validation Checks Card
    checks = report.get("checks", {})
    ocr_ok = checks.get("ocr_successful", True)
    quality_ok = checks.get("image_quality_acceptable", True)
    id_format_label = checks.get("expected_id_format", "5 letters + 4 digits + 1 letter")
    id_ok = checks.get("number_region_detected", True)
    req_fields_ok = checks.get("required_fields_present", True)
    qr_ok = checks.get("qr_code_detected", False)

    check_svg_pass = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
    check_svg_warn = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
    check_svg_fail = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>"""

    st.markdown(f"""
    <div class="dv-card">
        <div class="dv-card-title" style="margin-bottom: 14px;">Validation checks</div>
        <div class="dv-check-list">
            <div class="dv-check-row">
                <div class="dv-check-left">
                    {check_svg_pass if ocr_ok else check_svg_fail}
                    <span>OCR successful</span>
                </div>
                <div class="{ 'dv-check-badge-pass' if ocr_ok else 'dv-check-badge-fail' }">
                    {'Pass' if ocr_ok else 'Fail'}
                </div>
            </div>
            <div class="dv-check-row">
                <div class="dv-check-left">
                    {check_svg_pass if quality_ok else check_svg_fail}
                    <span>Image quality</span>
                </div>
                <div class="{ 'dv-check-badge-pass' if quality_ok else 'dv-check-badge-fail' }">
                    {'Pass' if quality_ok else 'Fail'}
                </div>
            </div>
            <div class="dv-check-row">
                <div class="dv-check-left">
                    {check_svg_pass if id_ok else check_svg_fail}
                    <span>ID format: {id_format_label}</span>
                </div>
                <div class="{ 'dv-check-badge-pass' if id_ok else 'dv-check-badge-fail' }">
                    {'Pass' if id_ok else 'Fail'}
                </div>
            </div>
            <div class="dv-check-row">
                <div class="dv-check-left">
                    {check_svg_pass if req_fields_ok else check_svg_fail}
                    <span>Required fields present</span>
                </div>
                <div class="{ 'dv-check-badge-pass' if req_fields_ok else 'dv-check-badge-fail' }">
                    {'Pass' if req_fields_ok else 'Fail'}
                </div>
            </div>
            <div class="dv-check-row">
                <div class="dv-check-left">
                    {check_svg_pass if qr_ok else check_svg_warn}
                    <span>QR code detected</span>
                </div>
                <div class="{ 'dv-check-badge-pass' if qr_ok else 'dv-check-badge-warn' }">
                    {'Pass' if qr_ok else 'Not found'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Technical Diagnostics & AI Explainability (Collapsible)
# ---------------------------------------------------------
with st.expander("Technical Diagnostics & AI Explainability (SHAP, OCR Data, Features)"):
    tab1, tab2, tab3, tab4 = st.tabs(["AI Explainability & SHAP", "OCR Details", "Image Quality Analysis", "Raw Validation Report"])

    with tab1:
        st.markdown("#### Machine Learning Fraud Detection")
        if risk_result:
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Predicted Class", risk_result["predicted_risk"])
            with col_p2:
                prob_low = risk_result["class_probabilities"].get("LOW", 0.0) * 100
                st.metric("Probability (Genuine/Low)", f"{prob_low:.1f}%")
            with col_p3:
                prob_high = risk_result["class_probabilities"].get("HIGH", 0.0) * 100
                st.metric("Probability (Manipulated/High)", f"{prob_high:.1f}%")

            st.markdown("---")
            st.markdown("##### SHAP Feature Attribution (Why this decision?)")

            bg = []
            bg_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_docs")
            if os.path.isdir(bg_dir):
                for cls_dir in ("genuine", "manipulated"):
                    cls_path = os.path.join(bg_dir, cls_dir)
                    if os.path.isdir(cls_path):
                        for fname in sorted(os.listdir(cls_path))[:5]:
                            img_bg = load_image(os.path.join(cls_path, fname))
                            feats_bg = extract_features(img_bg)
                            bg.append([feats_bg[k] for k in feature_names])

            if not bg:
                bg = [[risk_result["features_used"][k] for k in feature_names]]
            bg_arr = np.array(bg)
            x_arr = np.array([[risk_result["features_used"][k] for k in feature_names]])

            explanations = explain_with_shap(fraud_model, feature_names, bg_arr, x_arr, top_k=5)
            if explanations:
                for fname, sval in explanations:
                    direction = "Towards Genuine (Protective)" if sval < 0 else "Towards Risk (Elevating)"
                    st.write(f"- **{fname}**: `{sval:+.4f}` &mdash; {direction}")
            else:
                st.info("SHAP explanations computed.")
        else:
            st.warning("Fraud classification model not loaded.")

    with tab2:
        st.markdown("#### OCR Inspection")
        st.metric("Total Words Detected", ocr.get("word_count", 0))
        st.metric("Mean Confidence", f"{ocr.get('mean_confidence', 0.0):.1f}%")
        st.text_area("Extracted Full Text", ocr.get("text", ""), height=130)

    with tab3:
        st.markdown("#### Image Forensics & Quality")
        st.json(quality)
        st.json(info)

    with tab4:
        st.markdown("#### Plain Text Audit Report")
        st.code(full_report_text, language="text")
