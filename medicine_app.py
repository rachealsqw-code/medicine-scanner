import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
import os
import shutil

# --- TESSERACT PATH CONFIGURATION ---
# This logic detects if you are on Windows or Linux (Cloud)
if os.name == 'nt':
    # Windows: Update 'USER' to your actual Windows username if running locally
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
else:
    # Linux (Streamlit Cloud): Automatically finds tesseract installed via packages.txt
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

st.set_page_config(page_title="Medicine OCR Scanner", page_icon="💊")
st.title("💊 Medicine Guided Capture")
st.write("Align the medicine label in the center for a GREEN frame.")

# Tab selection for Live Camera or File Upload
tab1, tab2 = st.tabs(["📸 Live Camera", "📁 Upload Training Photo"])

def process_image(image_bytes):
    # Convert image bytes to OpenCV format
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Could not decode image.")
        return

    # 1. Calculate Sharpness (Gatekeeper Logic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Run OCR Logic (Confidence-based filtering)
    # This provides a dictionary of text and confidence levels
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    valid_words = []
    for i, word in enumerate(data['text']):
        # Filter: Word length >= 4 and Confidence > 60%
        if len(word.strip()) >= 4 and int(data['conf'][i]) > 60:
            clean_word = re.sub(r'[^A-Z]', '', word.upper())
            if len(clean_word) >= 4:
                valid_words.append(clean_word)

    # 3. UI Feedback (Red/Yellow/Green Logic)
    if score < 100:
        st.error(f"🔴 FRAME: RED (Blurry: {int(score)})")
        st.warning("Hold still! The image is too blurry for the API.")
    elif not valid_words:
        st.warning(f"🟡 FRAME: YELLOW (Sharp: {int(score)})")
        st.info("No clear medicine name detected. Move closer to the label.")
    else:
        st.success(f"🟢 FRAME: GREEN (Capture Success)")
        st.write("### Detected Medicine Names:")
        for w in set(valid_words): # set() removes duplicates
            st.button(f"✅ {w}")
    
    # Show the image for user reference
    st.image(img, caption="Scan Preview", use_container_width=True)

# TAB 1: Live Camera (Works on HTTPS/Cloud)
with tab1:
    cam_image = st.camera_input("Point camera at medicine label")
    if cam_image:
        process_image(cam_image.read())

# TAB 2: File Upload (Great for testing your API training photos)
with tab2:
    uploaded_file = st.file_uploader("Choose a photo from your training set", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        process_image(uploaded_file.read())
