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

@st.cache_resource
def load_insect_model():
    """โหลดโมเดลตรวจจับแมลงจาก Roboflow"""
    try:
        API_KEY = "3ZQFofNJkviVJdyAb4mG"
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace("aiinsect").project("ai-insect")
        model = project.version(4).model
        return model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None

st.set_page_config(page_title="AI Insect Count", layout="wide", page_icon="🦟")
st.markdown(
    """
    <div style="
        background: linear-gradient(to right, #6ab04c, #badc58);
        border-radius: 12px; padding: 10px 0; text-align: center;
        font-size: 28px; font-weight: bold; color: white; margin-bottom: 20px;
    ">
        AI Insect Count
    </div>
    """, unsafe_allow_html=True
)
st.markdown("""
<style>
.stSelectbox > div > div, .stDateInput > div > div, .stTextInput > div > div > input, .stTextArea > div > textarea {
    background-color: #f0f8ff !important; border-radius: 8px !important;
}
div.stButton > button, .stForm [data-testid=stFormSubmitButton] button {
    background-color: #27ae60; color: white; font-weight: bold; border-radius: 25px;
    padding: 10px 40px; font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

model = load_insect_model()
keys_to_init = {
    'analysis_results': None, 'factory': "", 'department': "", 'location': "", 
    'inspection_date': datetime.now(timezone(timedelta(hours=7))).date(),
    'excel_data_to_download': None, 'excel_filename': ""
}
for key, value in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = value

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("กรอกข้อมูลการตรวจ")
    
    def on_factory_change(): st.session_state.department = ""; st.session_state.location = ""
    st.selectbox("โรงงาน", [""] + sorted(LOCATION_DATA.keys()), key='factory', on_change=on_factory_change)

    def on_department_change(): st.session_state.location = ""
    department_list = [""] + sorted(LOCATION_DATA[st.session_state.factory].keys()) if st.session_state.factory else [""]
    st.selectbox("หน่วยงาน/แผนก", department_list, key='department', on_change=on_department_change, disabled=not st.session_state.factory)

    location_list = [""] + sorted(LOCATION_DATA[st.session_state.factory][st.session_state.department]) if st.session_state.factory and st.session_state.department else [""]
    st.selectbox("พื้นที่ติดตั้ง", location_list, key='location', disabled=not st.session_state.department)
    
    st.date_input("วันที่ตรวจ", key='inspection_date')
    
    with st.form("analysis_form", clear_on_submit=True):
        st.subheader("อัปโหลดรูปภาพ")
        source_option = st.radio("เลือกแหล่งที่มาของรูป:", ["อัปโหลดไฟล์", "ถ่ายภาพจากกล้อง"], horizontal=True)
        uploaded_image = None
        if source_option == "อัปโหลดไฟล์":
            uploaded_image = st.file_uploader("เลือกไฟล์...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            uploaded_image = st.camera_input("ถ่ายภาพ", label_visibility="collapsed")
        
        if "confidence_threshold" not in st.session_state:
            st.session_state.confidence_threshold = 0.4

        analysis_button = st.form_submit_button("Analysis", use_container_width=True)

    if analysis_button:
        st.session_state.analysis_results = None
        st.session_state.excel_data_to_download = None
        
        if not all([st.session_state.factory, st.session_state.department, st.session_state.location]):
            st.warning("⚠️ กรุณากรอกข้อมูล โรงงาน, หน่วยงาน, และพื้นที่ติดตั้งให้ครบถ้วน")
        elif uploaded_image is None:
            st.warning("⚠️ กรุณาอัปโหลดหรือถ่ายภาพก่อน")
        elif not model:
            st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน")
        else:
            try:
                with st.spinner("🧠 กำลังวิเคราะห์ภาพ..."):
                    image_pil = Image.open(uploaded_image).convert("RGB")
                    temp_path = "temp_insect_image.jpg"
                    image_pil.save(temp_path)
                    
                    # RUN MODEL (ครั้งเดียว)
                    results_json = model.predict(temp_path, confidence=40, overlap=30).json()
                    os.remove(temp_path) 
                    predictions = results_json.get('predictions', [])

                    # เก็บข้อมูลดิบ
                    st.session_state.raw_predictions = predictions
                    st.session_state.original_image = image_pil
            
                st.success("✅ วิเคราะห์สำเร็จ!")
            except Exception as e:
                st.error("😭 เกิดข้อผิดพลาดร้ายแรง!")
                st.exception(e)



with col_right:
    st.subheader("ผลการวิเคราะห์")
    results = st.session_state.get('analysis_results')

    if "raw_predictions" in st.session_state:

        predictions = st.session_state.raw_predictions
        image_pil = st.session_state.original_image

        confidence_threshold = st.slider(
            "🎯 Confidence (ค่าความเชื่อมั่น)",
            0.0,
            1.0,
            value=st.session_state.confidence_threshold,
            step=0.05
        )

        st.session_state.confidence_threshold = confidence_threshold

        # FILTER
        filtered_predictions = [
            p for p in predictions
            if p["confidence"] >= confidence_threshold
        ]

        insect_count = defaultdict(int)

        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        class_colors = {
            "fly": (0,255,0),
            "test": (255,0,0)
        }

        for pred in filtered_predictions:

            insect_class = pred["class"]
            insect_count[insect_class] += 1

            x = int(pred["x"])
            y = int(pred["y"])
            w = int(pred["width"])
            h = int(pred["height"])

            x1 = x - w//2
            y1 = y - h//2
            x2 = x + w//2
            y2 = y + h//2

            color = class_colors.get(insect_class,(0,255,0))

            cv2.rectangle(image_cv2,(x1,y1),(x2,y2),color,2)

            label = f"{insect_class} ({pred['confidence']:.2f})"

            cv2.putText(
                image_cv2,
                label,
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        annotated_image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # COUNT
        fly_count = insect_count.get("fly",0)
        other_count = insect_count.get("test",0)
        total = fly_count + other_count

        m1,m2,m3 = st.columns(3)

        with m1:
            st.metric("จำนวนแมลงทั้งหมด", f"{total} ตัว")

        with m2:
            st.metric("แมลงวัน (Fly)", f"{fly_count} ตัว")

        with m3:
            st.metric("แมลงอื่น", f"{other_count} ตัว")
        
        st.markdown("---")
        st.markdown("#### รูปภาพที่ Label แล้ว (Picture Label)")
        st.image(annotated_image_rgb, use_container_width=True)

        st.markdown("---")
        st.subheader("📝 บันทึกและดาวน์โหลดผล")
        
        with st.form("save_form"):
            recorder_name = st.text_input("ชื่อผู้บันทึก")
            notes = st.text_area("หมายเหตุ (ถ้ามี)")
            save_button = st.form_submit_button("บันทึกและสร้างไฟล์ดาวน์โหลด")

            if save_button:
                if not recorder_name.strip():
                    st.warning("กรุณากรอกชื่อผู้บันทึก")
                else:
                    try:
                        bkk_timezone = timezone(timedelta(hours=7))
                        time_in_bkk = datetime.now(bkk_timezone)
                        
                        new_record_data = {
                            "วันที่ตรวจ": st.session_state.inspection_date.strftime("%Y-%m-%d"),
                            "เวลาที่บันทึก": time_in_bkk.strftime("%H:%M:%S"),
                            "โรงงาน": st.session_state.factory,
                            "หน่วยงาน/แผนก": st.session_state.department,
                            "พื้นที่ติดตั้ง": st.session_state.location,
                            "จำนวนแมลงทั้งหมด": total,
                            "จำนวนแมลงวัน": fly_count,
                            "จำนวนแมลงอื่นๆ": other_count,
                            "ผู้บันทึก": recorder_name,
                            "หมายเหตุ": notes
                        }

                        urlPost = "https://default097b580bb474487c888346e0bb1b5c.11.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/49eae18339ec46cd97ca8069832d0f34/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=Ei6KSW6Dt9UVgb5ZYNxh6PLtkX7dxQuOF5LAsdgVFnw"
                        response = requests.post(urlPost,json=new_record_data)
                        print(response.status_code)
                       
                        df_new = pd.DataFrame([new_record_data])
                        ##df_new.to_excel(EXCEL_FILENAME,index=False)

                        if os.path.exists(EXCEL_FILENAME):
                            df_existing = pd.read_excel(EXCEL_FILENAME, engine='openpyxl')
                            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        else:
                            df_combined = df_new
                        

                        df_combined.to_excel(EXCEL_FILENAME, index=False)
                        st.success("✅ บันทึกข้อมูลลงไฟล์หลักเรียบร้อยแล้ว!")

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_combined.to_excel(writer, index=False, sheet_name='AnalysisHistory')
                        
                        st.session_state.excel_data_to_download = output.getvalue()
                        st.session_state.excel_filename = EXCEL_FILENAME
                        st.rerun()

                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการบันทึก/สร้างไฟล์: {e}")

        if st.session_state.get("excel_data_to_download"):
            st.download_button(
                label="📥 คลิกที่นี่เพื่อดาวน์โหลดไฟล์ Excel ล่าสุด",
                data=st.session_state.excel_data_to_download,
                file_name=st.session_state.excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        with st.container(border=True, height=500):
             st.info("กรอกข้อมูลและอัปโหลดรูปภาพ จากนั้นกดปุ่ม 'Analysis' เพื่อดูผลลัพธ์")
