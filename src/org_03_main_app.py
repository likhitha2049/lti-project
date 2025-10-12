import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# --- Dynamic & Robust Path Configuration ---
# This code finds the project root folder automatically.
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    # Fallback for some environments
    PROJECT_ROOT = os.path.abspath('.')

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- Page Configuration & Model Loading ---
st.set_page_config(page_title="Multi-Model Multi-Disease Classification", page_icon="🩺", layout="centered")
st.title("🩺 Multi-Model Multi-Disease Classification System")

@st.cache_resource
def load_all_models():
    # This function now loads from the new .keras format.
    models = {
        'router': load_model(os.path.join(MODELS_DIR, 'router.keras')),
        'brain': load_model(os.path.join(MODELS_DIR, 'brain_specialist.keras')),
        'chest': load_model(os.path.join(MODELS_DIR, 'chest_specialist.keras')),
        'skin': load_model(os.path.join(MODELS_DIR, 'skin_specialist.keras'))
    }
    return models

# --- Class Labels (sorted to match generator output) ---
ROUTER_CLASSES = sorted(['brain', 'chest', 'skin'])
BRAIN_CLASSES = sorted(['brain_ct_healthy', 'brain_ct_tumor', 'brain_mri_healthy', 'brain_mri_tumor'])
CHEST_CLASSES = sorted(['chest_normal', 'chest_pneumonia'])
SKIN_CLASSES = sorted(['skin_benign', 'skin_malignant'])

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB": image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

try:
    models = load_all_models()
    st.write("Upload a medical image (Brain CT/MRI, Chest X-Ray, or Skin Lesion) for analysis.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        with st.spinner('Analyzing the image...'):
            processed_image = preprocess_image(image)
            router_prediction = models['router'].predict(processed_image)
            image_type = ROUTER_CLASSES[np.argmax(router_prediction)]
            
            st.info(f"**Step 1: Image Routing** → Identified as a **{image_type.upper()}** scan.")
            
            if image_type == 'brain':
                prediction = models['brain'].predict(processed_image)
                final_diagnosis = BRAIN_CLASSES[np.argmax(prediction)]
            elif image_type == 'chest':
                prediction = models['chest'].predict(processed_image)[0][0]
                final_diagnosis = CHEST_CLASSES[1] if prediction > 0.5 else CHEST_CLASSES[0]
            elif image_type == 'skin':
                prediction = models['skin'].predict(processed_image)[0][0]
                final_diagnosis = SKIN_CLASSES[1] if prediction > 0.5 else SKIN_CLASSES[0]

        st.success("**Step 2: Final Diagnosis Complete!**")
        
        parts = final_diagnosis.split('_')
        organ, modality, status = parts[0].capitalize(), "", " ".join(p.capitalize() for p in parts[1:])
        if organ == "Brain":
            modality = parts[1].upper()
            status = " ".join(p.capitalize() for p in parts[2:])

        col1, col2 = st.columns(2)
        col1.metric("Organ / Body Part", organ)
        if modality: col2.metric("Imaging Modality", modality)
        
        st.markdown("---")
        color = "red" if any(x in status.lower() for x in ["tumor", "pneumonia", "malignant"]) else "green"
        st.markdown(f"### <span style='color:{color};'>Diagnosis: {status}</span>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error(f"FATAL ERROR: Model files not found in the expected directory '{MODELS_DIR}'. Please ensure the 'models' folder exists in your project root (D:\\pro1) and contains all four .h5 files.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")