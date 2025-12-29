import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import os
import shutil

# --- TESSERACT PATH CONFIGURATION ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
else:
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

st.set_page_config(page_title="Medicine OCR Scanner", page_icon="💊")
st.title("💊 Guided Medicine Scanner")
st.write("Fit the medicine label inside the **Target Box** for detection.")

tab1, tab2 = st.tabs(["📸 Live Camera", "📁 Upload Training Photo"])

def process_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Could not decode image.")
        return

    # 1. CREATE TARGET BOX (Signaling Logic)
    h, w, _ = img.shape
    # Define box coordinates (Center 60% of the image)
    t, b = int(h * 0.2), int(h * 0.8)
    l, r = int(w * 0.2), int(w * 0.8)
    
    # Crop to the ROI (Region of Interest) for analysis
    roi = img[t:b, l:r]
    
    # 2. CALCULATE SHARPNESS (Gatekeeper)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

    # 3. RUN OCR (Confidence Filtering)
    data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
    valid_words = []
    for i, word in enumerate(data['text']):
        if len(word.strip()) >= 4 and int(data['conf'][i]) > 60:
            clean_word = re.sub(r'[^A-Z]', '', word.upper())
            if len(clean_word) >= 4:
                valid_words.append(clean_word)

    # 4. SIGNALING FEEDBACK
    # Draw the Target Box on the display image
    if score < 100:
        # RED SIGNAL: Blurry
        color = (0, 0, 255) # Red in BGR
        st.error(f"🔴 FRAME: RED - TOO BLURRY ({int(score)})")
        st.info("Hold steady and wait for focus.")
    elif not valid_words:
        # YELLOW SIGNAL: Sharp but no text
        color = (0, 255, 255) # Yellow in BGR
        st.warning(f"🟡 FRAME: YELLOW - READY ({int(score)})")
        st.info("Move closer or adjust angle until text is detected.")
    else:
        # GREEN SIGNAL: Success
        color = (0, 255, 0) # Green in BGR
        st.success(f"🟢 FRAME: GREEN - WORDS DETECTED!")
        st.write("### Detected Medicine:", ", ".join(set(valid_words)))

    # Draw the visual box for the user
    cv2.rectangle(img, (l, t), (r, b), color, 10)
    st.image(img, caption="Guided Viewfinder", use_container_width=True)

with tab1:
    cam_image = st.camera_input("Scanner View")
    if cam_image:
        process_image(cam_image.read())

with tab2:
    uploaded_file = st.file_uploader("Test a photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        process_image(uploaded_file.read())
