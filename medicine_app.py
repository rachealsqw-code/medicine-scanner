import streamlit as st
import cv2
import pytesseract
import numpy as np
import re

# Update 'USER' to your actual Windows username
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Medicine OCR Scanner", page_icon="💊")
st.title("💊 Medicine Guided Capture")

# Tab selection for better UI
tab1, tab2 = st.tabs(["📸 Live Camera", "📁 Upload Training Photo"])

def process_image(image_bytes):
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Calculate Sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Run OCR Logic
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    valid_words = [word.upper() for i, word in enumerate(data['text']) if len(word.strip()) >= 4 and int(data['conf'][i]) > 60]

    # 3. UI Feedback
    if score < 100:
        st.error(f"🔴 FRAME: RED (Blurry: {int(score)})")
        st.warning("Hold still and try again.")
    elif not valid_words:
        st.warning(f"🟡 FRAME: YELLOW (Sharp: {int(score)})")
        st.info("No clear text found. Move closer to the label.")
    else:
        st.success(f"🟢 FRAME: GREEN (Capture Success)")
        st.write("**Detected Medicine:**", ", ".join(valid_words))
    
    st.image(img, caption="Processed Image", use_container_width=True)

with tab1:
    cam_image = st.camera_input("Take a picture of the medicine label")
    if cam_image:
        process_image(cam_image.read())

with tab2:
    uploaded_file = st.file_uploader("Upload from API training photos", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        process_image(uploaded_file.read())