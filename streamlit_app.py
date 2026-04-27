import streamlit as st
from datetime import datetime, timezone, timedelta
from PIL import Image
import numpy as np
import cv2
import os
from roboflow import Roboflow
import pandas as pd
import io
import requests
from collections import defaultdict
import base64
from ultralytics import YOLO

# ─────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────
LOCATION_DATA = {
    "SB": {"บรรจุ": ["ห้องแต่งตัว", "ห้องบรรจุ 1", "ห้องบรรจุ 2", "ห้องบรรจุ 3", "ห้องบรรจุ Auto", "ห้อง Mix SP", "ห้องบรรจุ SP", "ห้องเก็บภาชนะ ชั้น 1", "ห้องเก็บภาชนะ ชั้น 2", "ห้อง Pack SP"]},
    "MDC": {
        "LS 1": [f"เครื่องที่ {i}" for i in range(1, 4)], "LS 2": [f"เครื่องที่ {i}" for i in range(1, 3)],
        "บริการลูกค้า": [f"เครื่องที่ {i}" for i in range(1, 12)], "พัสดุ": ["เครื่องที่ 1"], "คลังสินค้า": ["เครื่องที่ 7"],
        "Conditioning Silo": [f"เครื่องที่ {i}" for i in range(1, 4)], "SPP": [f"เครื่องที่ {i}" for i in range(1, 13)],
        "HIT": [f"เครื่องที่ {i}" for i in range(1, 6)], "Rock Sugar": [f"เครื่องที่ {i}" for i in range(1, 5)],
        "บรรจุ": [f"เครื่องที่ {i}" for i in range(1, 9)], "หม้อปั่นรีไฟน์": ["เครื่องที่ 1"],
    },
    "MPK": {
        "บรรจุ": ["1.ห้องเก็บกระสอบ 50 กก.", "2.ห้องเตรียมกระสอบ 50 กก.", "3.ห้องบรรจุ 50 กก.", "4.ห้องกระสอบ silo", "5.ห้องบรรจุ 1000 กก. Silo", "6.ห้องบรรจุ 1 กก. Silo", "7.ห้องบรรจุแพ็คโถ 5", "8.ห้องกระสอบ White 1000 กก.ใหม่", "9.ห้องบรรจุWhite 1000kg.ใหม่", "18. ห้องบรรจุ Demerara"],
        "เคี่ยวปั่นรีไฟน์": ["10.ห้องใต้หม้อปั่น", "11.ห้องสายพานห้องล่าง", "12.ห้องสายพานล่าง No.2", "13.ห้องสายพานห้องกลาง", "14.ห้องสายพานห้องบน", "15.ห้องสายพานบน No.2"],
        "Conditioning Silo": ["16.ห้องตะแกรงคัดเม็ด Nestle", "17.ห้องตะแกรงคัดเม็ด Rotex"]
    },
    "MPV": {
        "บรรจุ": [f"เครื่องที่ {i} บรรจุ" for i in range(1, 13)], "เคี่ยวปั่นรีไฟน์": ["เครื่องที่ 1 เคี่ยวปั่นรีไฟน์"],
        "Conditioning Silo": ["เครื่องที่ 1 Conditioning Silo"], "ผลิตภัณฑ์พิเศษ": [f"เครื่องที่ {i} ผลิตภัณฑ์พิเศษ" for i in range(1, 5)]
    },
    "MKS": {
        "บรรจุ": ["ILP01 ห้องเปลี่ยนเสื้อผ้าเข้าห้องบรรจุ", "ILP02 ห้อง NCS Auto", "ILP03 ห้องเก็บภาชนะ, ห้อง Auto", "ILP04 ห้องบรรจุน้ำตาล 25/50กก. ยุ้ง 2", "ILP05 ห้องบรรจุน้ำตาล 25/50กก. ยุ้ง 1", "ILP06 ห้องบรรจุน้ำตาล MG 1 กก.", "ILP07 ห้องแพ็คกล่อง MG 1 กก.", "ILP08 ห้องเก็บภาชนะ 25/50 กก.", "ILP09 ห้องเก็บภาชนะหลังห้องบรรจุ", "ILP10 ห้องเปลี่ยนเสื้อผ้าห้องผลิต MG", "ILP11 ห้องเทผลิต MG", "ILP12 ห้องเปลี่ยนเสื้อผ้าห้องเท Auto", "ILP13 ห้องเท Auto"],
        "หม้อปั่น": ["NO1 ห้องตะแกรงโยก"]
    }
}
EXCEL_FILENAME = "insect_analysis_history.xlsx"

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="AI Insect Count", layout="wide", page_icon="🦟")

# ─────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Root variables ── */
:root {
    --primary:        #2ecc71;
    --primary-dark:   #27ae60;
    --primary-glow:   rgba(46,204,113,0.18);
    --accent:         #f39c12;
    --danger:         #e74c3c;
    --bg-main:        #f4f7f4;
    --bg-card:        #ffffff;
    --bg-input:       #f8fbf8;
    --border:         #d5e8d4;
    --text-main:      #1a2e1a;
    --text-sub:       #5a7a5a;
    --text-hint:      #8aaa8a;
    --radius:         12px;
    --radius-sm:      8px;
    --shadow:         0 2px 12px rgba(0,60,0,0.08);
    --shadow-hover:   0 6px 24px rgba(0,60,0,0.14);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif !important;
    color: var(--text-main);
}
.main { background: var(--bg-main) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f0f0; }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }

/* ── Inputs ── */
div[data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stTextArea > div > textarea,
.stDateInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-main) !important;
    font-family: 'Sarabun', sans-serif !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover,
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}
div[data-baseweb="select"] span { color: var(--text-main) !important; font-weight: 500; }

/* ── Slider ── */
.stSlider [data-testid="stTickBar"] { display: none; }
.stSlider [role="slider"] {
    background: var(--primary) !important;
    border: 2px solid white !important;
    box-shadow: 0 2px 8px rgba(46,204,113,0.4) !important;
}
.stSlider [data-baseweb="slider"] > div > div > div:first-child {
    background: linear-gradient(to right, var(--primary), var(--primary-dark)) !important;
}

/* ── Primary button ── */
div.stButton > button,
.stForm [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(39,174,96,0.35) !important;
    letter-spacing: 0.3px;
}
div.stButton > button:hover,
.stForm [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(39,174,96,0.45) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: white !important;
    font-family: 'Sarabun', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(52,152,219,0.35) !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(52,152,219,0.45) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 0.5rem; }
.stRadio label {
    background: var(--bg-input);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.4rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
}
.stRadio label:has(input:checked) {
    background: var(--primary-glow);
    border-color: var(--primary);
    color: var(--primary-dark);
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Alert / success ── */
.stSuccess { border-left: 4px solid var(--primary) !important; border-radius: var(--radius-sm) !important; }
.stWarning { border-left: 4px solid var(--accent) !important; border-radius: var(--radius-sm) !important; }
.stError   { border-left: 4px solid var(--danger) !important;  border-radius: var(--radius-sm) !important; }

/* ── Section card ── */
.section-card {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.section-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Step indicator ── */
.step-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 1.5rem;
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    box-shadow: var(--shadow);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex: 1;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-hint);
    padding: 0.3rem 0;
    border-radius: 50px;
    transition: all 0.3s;
}
.step-item.active {
    color: var(--primary-dark);
    background: var(--primary-glow);
    padding: 0.3rem 1rem;
}
.step-item.done { color: var(--primary-dark); }
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: var(--border);
    color: var(--text-hint);
    font-size: 11px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-item.active .step-num { background: var(--primary); color: white; }
.step-item.done .step-num { background: var(--primary-dark); color: white; }
.step-divider { width: 20px; height: 1px; background: var(--border); flex-shrink: 0; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
.metric-card.total  { border-top: 3px solid #2ecc71; }
.metric-card.fly    { border-top: 3px solid #3498db; }
.metric-card.other  { border-top: 3px solid #e67e22; }
.metric-icon { font-size: 26px; margin-bottom: 4px; }
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card.total .metric-value { color: #27ae60; }
.metric-card.fly   .metric-value { color: #2980b9; }
.metric-card.other .metric-value { color: #d35400; }
.metric-label { font-size: 12px; color: var(--text-sub); font-weight: 600; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Progress bar ── */
.ratio-bar-wrap { margin: 0.8rem 0 1.2rem; }
.ratio-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-sub); margin-bottom: 4px; font-weight: 600; }
.ratio-bar { height: 10px; border-radius: 10px; background: #e0e0e0; overflow: hidden; }
.ratio-fill-fly   { height: 100%; background: linear-gradient(to right,#3498db,#5dade2); border-radius: 10px; transition: width 0.8s ease; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 420px; gap: 1rem; color: var(--text-hint);
}
.empty-icon { font-size: 64px; opacity: 0.5; }
.empty-title { font-size: 18px; font-weight: 700; color: var(--text-sub); }
.empty-desc { font-size: 14px; text-align: center; max-width: 260px; line-height: 1.6; }

/* ── Image caption ── */
.img-caption {
    font-size: 12px; color: var(--text-hint);
    text-align: center; margin-top: 6px; font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #27ae60 0%, #2ecc71 50%, #58d68d 100%);
    border-radius: 16px; padding: 18px 28px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 6px 24px rgba(39,174,96,0.3); margin-bottom: 24px;
">
    <div>
        <div style="font-size:26px; font-weight:800; color:white; letter-spacing:-0.5px;">
            🦟 AI Insect Count
        </div>
        <div style="font-size:13px; color:rgba(255,255,255,0.8); margin-top:2px; font-weight:500;">
            ระบบนับแมลงอัตโนมัติด้วย AI
        </div>
    </div>
    <div style="text-align:right; color:rgba(255,255,255,0.75); font-size:12px; font-family:'IBM Plex Mono',monospace;">
        v2.0
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MODEL LOAD
# ─────────────────────────────────────────
@st.cache_resource
def load_insect_model():
    try:
        model = YOLO("runs_detect_train-4_weights_best.pt")
        return model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None

model = load_insect_model()

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
keys_to_init = {
    'analysis_results': None, 'factory': "", 'department': "", 'location': "",
    'inspection_date': datetime.now(timezone(timedelta(hours=7))).date(),
    'excel_data_to_download': None, 'excel_filename': "",
    'confidence_threshold': 0.4,
}
for key, value in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─────────────────────────────────────────
#  STEP INDICATOR HELPER
# ─────────────────────────────────────────
def render_steps(active: int):
    """active = 1, 2, or 3"""
    def cls(n): return "active" if n == active else ("done" if n < active else "")
    icons = ["📋", "📷", "📊"]
    labels = ["กรอกข้อมูล", "อัปโหลดรูป", "ดูผลลัพธ์"]
    items = ""
    for i in range(1, 4):
        items += f"""
        <div class="step-item {cls(i)}">
            <span class="step-num">{i}</span>
            {icons[i-1]} {labels[i-1]}
        </div>"""
        if i < 3:
            items += '<div class="step-divider"></div>'
    st.markdown(f'<div class="step-bar">{items}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  DETERMINE ACTIVE STEP
# ─────────────────────────────────────────
has_location = all([st.session_state.factory, st.session_state.department, st.session_state.location])
has_result   = "raw_predictions" in st.session_state

if has_result:       active_step = 3
elif has_location:   active_step = 2
else:                active_step = 1

render_steps(active_step)

# ─────────────────────────────────────────
#  MAIN COLUMNS
# ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════
#  LEFT — INPUT
# ══════════════════════════════════════════
with col_left:

    # ── Section 1: Location info ──
    st.markdown('<div class="section-title">📋 ข้อมูลการตรวจ</div>', unsafe_allow_html=True)

    def on_factory_change():
        st.session_state.department = ""
        st.session_state.location = ""

    st.selectbox(
        "🏭 โรงงาน",
        ["— เลือกโรงงาน —"] + sorted(LOCATION_DATA.keys()),
        key='factory',
        on_change=on_factory_change,
        format_func=lambda x: x if x != "— เลือกโรงงาน —" else "— เลือกโรงงาน —"
    )

    def on_department_change():
        st.session_state.location = ""

    dept_disabled = not st.session_state.factory or st.session_state.factory == "— เลือกโรงงาน —"
    factory_key = st.session_state.factory if not dept_disabled else ""
    department_list = ["— เลือกหน่วยงาน —"] + (sorted(LOCATION_DATA[factory_key].keys()) if factory_key else [])
    st.selectbox("🏢 หน่วยงาน / แผนก", department_list, key='department', on_change=on_department_change, disabled=dept_disabled)

    loc_disabled = not st.session_state.department or st.session_state.department == "— เลือกหน่วยงาน —"
    dept_key = st.session_state.department if not loc_disabled else ""
    location_list = ["— เลือกพื้นที่ —"] + (sorted(LOCATION_DATA[factory_key][dept_key]) if (factory_key and dept_key) else [])
    st.selectbox("📍 พื้นที่ติดตั้ง", location_list, key='location', disabled=loc_disabled)

    st.date_input("📅 วันที่ตรวจ", key='inspection_date')

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Section 2: Upload ──
    st.markdown('<div class="section-title">📷 อัปโหลดรูปภาพ</div>', unsafe_allow_html=True)

    with st.form("analysis_form", clear_on_submit=True):
        source_option = st.radio(
            "แหล่งที่มาของรูป",
            ["📁  อัปโหลดไฟล์", "📸  ถ่ายภาพจากกล้อง"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_image = None
        if "อัปโหลดไฟล์" in source_option:
            uploaded_image = st.file_uploader(
                "ลากไฟล์มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์",
                type=["jpg", "jpeg", "png"]
            )
        else:
            uploaded_image = st.camera_input("ถ่ายภาพ", label_visibility="collapsed")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        analysis_button = st.form_submit_button("🔍  วิเคราะห์ภาพด้วย AI", use_container_width=True)

    # ── PROCESS ──
    if analysis_button:
        st.session_state.analysis_results = None
        st.session_state.excel_data_to_download = None

        f = st.session_state.factory
        d = st.session_state.department
        l = st.session_state.location

        if not all([f, d, l]) or "—" in f or "—" in d or "—" in l:
            st.warning("⚠️ กรุณากรอกข้อมูล โรงงาน, หน่วยงาน, และพื้นที่ติดตั้งให้ครบถ้วน")
        elif uploaded_image is None:
            st.warning("⚠️ กรุณาอัปโหลดหรือถ่ายภาพก่อน")
        elif not model:
            st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน")
        else:
            try:
                with st.spinner("🧠 AI กำลังวิเคราะห์ภาพ... กรุณารอสักครู่"):
                    image_pil = Image.open(uploaded_image).convert("RGB")
                    temp_path = "temp_insect_image.jpg"
                    image_pil.save(temp_path)

                    results = model.predict(source=temp_path, conf=0.10, iou=0.40, imgsz=1920, max_det=5000)
                    os.remove(temp_path)

                    predictions = []
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            predictions.append({
                                "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                                "width": x2 - x1,    "height": y2 - y1,
                                "confidence": conf,  "class": model.names[cls]
                            })

                    st.session_state.raw_predictions = predictions
                    st.session_state.original_image  = image_pil

                st.success("✅ วิเคราะห์สำเร็จ! ดูผลลัพธ์ได้ทางขวา")
            except Exception as e:
                st.error("😭 เกิดข้อผิดพลาด กรุณาลองใหม่")
                st.exception(e)

# ══════════════════════════════════════════
#  RIGHT — RESULTS
# ══════════════════════════════════════════
with col_right:

    if "raw_predictions" not in st.session_state:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">ยังไม่มีผลการวิเคราะห์</div>
            <div class="empty-desc">กรอกข้อมูลและอัปโหลดรูปภาพ<br>จากนั้นกดปุ่ม "วิเคราะห์ภาพด้วย AI"</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        predictions  = st.session_state.raw_predictions
        image_pil    = st.session_state.original_image

        # ── Confidence slider ──
        st.markdown('<div class="section-title">🎯 ปรับค่าความเชื่อมั่น (Confidence)</div>', unsafe_allow_html=True)

        col_sl, col_val = st.columns([5, 1])
        with col_sl:
            confidence_threshold = st.slider(
                "confidence", 0.0, 1.0,
                value=st.session_state.confidence_threshold,
                step=0.05, label_visibility="collapsed"
            ) 
        with col_val:
            st.markdown(
                f"<div style='text-align:center;padding-top:8px;"
                f"font-family:IBM Plex Mono,monospace;font-weight:700;"
                f"font-size:20px;color:var(--primary-dark);'>"
                f"{confidence_threshold:.2f}</div>",
                unsafe_allow_html=True
            )
        st.session_state.confidence_threshold = confidence_threshold

        # ── Filter & draw ──
        filtered = [p for p in predictions if p["confidence"] >= confidence_threshold]

        insect_count = defaultdict(int)
        image_cv2    = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        class_colors = {"fly": (46, 204, 113), "test": (231, 76, 60)}

        for pred in filtered:
            cls = pred["class"]
            insect_count[cls] += 1
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            color = class_colors.get(cls, (46, 204, 113))
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_cv2, f"{cls} {pred['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        annotated_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        fly_count   = insect_count.get("fly", 0)
        other_count = insect_count.get("test", 0)
        total       = fly_count + other_count

        # ── Metric cards ──
        fly_pct = int(fly_count / total * 100) if total > 0 else 0

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card total">
                <div class="metric-icon">🦟</div>
                <div class="metric-value">{total}</div>
                <div class="metric-label">แมลงทั้งหมด</div>
            </div>
            <div class="metric-card fly">
                <div class="metric-icon">🪰</div>
                <div class="metric-value">{fly_count}</div>
                <div class="metric-label">แมลงวัน (Fly)</div>
            </div>
            <div class="metric-card other">
                <div class="metric-icon">🐛</div>
                <div class="metric-value">{other_count}</div>
                <div class="metric-label">แมลงอื่นๆ</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Ratio bar ──
        if total > 0:
            st.markdown(f"""
            <div class="ratio-bar-wrap">
                <div class="ratio-label">
                    <span>🪰 แมลงวัน {fly_pct}%</span>
                    <span>แมลงอื่น {100 - fly_pct}%</span>
                </div>
                <div class="ratio-bar">
                    <div class="ratio-fill-fly" style="width:{fly_pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Annotated image ──
        st.markdown('<div class="section-title">🖼️ ภาพที่ผ่านการวิเคราะห์</div>', unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)
        st.markdown(
            f'<div class="img-caption">พบแมลงทั้งหมด {total} ตัว · confidence ≥ {confidence_threshold:.2f}</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Save form ──
        st.markdown('<div class="section-title">💾 บันทึกและดาวน์โหลดผล</div>', unsafe_allow_html=True)

        with st.form("save_form"):
            c1, c2 = st.columns(2)
            with c1:
                recorder_name = st.text_input("👤 ชื่อผู้บันทึก", placeholder="กรอกชื่อ-สกุล")
            with c2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                save_btn = st.form_submit_button("💾  บันทึกข้อมูล", use_container_width=True)
            notes = st.text_area("📝 หมายเหตุ (ถ้ามี)", placeholder="เพิ่มเติมหมายเหตุที่นี่...", height=80)

            if save_btn:
                if not recorder_name.strip():
                    st.warning("⚠️ กรุณากรอกชื่อผู้บันทึก")
                else:
                    try:
                        bkk_tz      = timezone(timedelta(hours=7))
                        time_in_bkk = datetime.now(bkk_tz)

                        img_buf    = io.BytesIO()
                        Image.fromarray(annotated_rgb).save(img_buf, format="JPEG")
                        img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")

                        new_record = {
                            "วันที่ตรวจ":          st.session_state.inspection_date.strftime("%Y-%m-%d"),
                            "เวลาที่บันทึก":        time_in_bkk.strftime("%H:%M:%S"),
                            "โรงงาน":              st.session_state.factory,
                            "หน่วยงาน/แผนก":       st.session_state.department,
                            "พื้นที่ติดตั้ง":       st.session_state.location,
                            "จำนวนแมลงทั้งหมด":    total,
                            "จำนวนแมลงวัน":        fly_count,
                            "จำนวนแมลงอื่นๆ":      other_count,
                            "ผู้บันทึก":            recorder_name,
                            "หมายเหตุ":             notes,
                            "รูปภาพ":              img_base64,
                        }

                        # Power Automate
                        urlPost  = "https://default097b580bb474487c888346e0bb1b5c.11.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/49eae18339ec46cd97ca8069832d0f34/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=Ei6KSW6Dt9UVgb5ZYNxh6PLtkX7dxQuOF5LAsdgVFnw"
                        requests.post(urlPost, json=new_record)

                        # Excel
                        df_new = pd.DataFrame([new_record])
                        if os.path.exists(EXCEL_FILENAME):
                            df_existing = pd.read_excel(EXCEL_FILENAME, engine='openpyxl')
                            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        else:
                            df_combined = df_new
                        df_combined.to_excel(EXCEL_FILENAME, index=False)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_combined.to_excel(writer, index=False, sheet_name='AnalysisHistory')

                        st.session_state.excel_data_to_download = output.getvalue()
                        st.session_state.excel_filename         = EXCEL_FILENAME
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาดในการบันทึก: {e}")

        # ── Download button ──
        if st.session_state.get("excel_data_to_download"):
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background: linear-gradient(135deg,#eaf6ff,#d6ecfa);
                border: 1.5px solid #aed6f1; border-radius: 12px;
                padding: 14px 18px; display:flex; align-items:center; gap:12px;
                margin-bottom:8px;
            ">
                <span style="font-size:28px">✅</span>
                <div>
                    <div style="font-weight:700;color:#1a5276;font-size:15px;">บันทึกข้อมูลสำเร็จ!</div>
                    <div style="color:#2980b9;font-size:13px;">กดปุ่มด้านล่างเพื่อดาวน์โหลดไฟล์ Excel</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="📥  ดาวน์โหลดไฟล์ Excel",
                data=st.session_state.excel_data_to_download,
                file_name=st.session_state.excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
