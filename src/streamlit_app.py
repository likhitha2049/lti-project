import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys
import os
import urllib.request

# --- Robust import for the xai helper module ---
try:
    from src import xai
except Exception:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        from src import xai
    except Exception:
        import importlib
        xai = importlib.import_module('xai')

# --- Dynamic & Robust Path Configuration ---
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Hugging Face Model URLs ---
MODEL_URLS = {
    "router": "https://huggingface.co/MunnangiLikhi/Medical-Specialist-Models/resolve/main/models/router.keras",
    "brain": "https://huggingface.co/MunnangiLikhi/Medical-Specialist-Models/resolve/main/models/brain_specialist.keras",
    "chest": "https://huggingface.co/MunnangiLikhi/Medical-Specialist-Models/resolve/main/models/chest_specialist.keras",
    "skin": "https://huggingface.co/MunnangiLikhi/Medical-Specialist-Models/resolve/main/models/skin_specialist.keras"
}

# --- Download missing models automatically ---
def ensure_models_present():
    for name, url in MODEL_URLS.items():
        local_path = os.path.join(MODELS_DIR, f"{name}_specialist.keras") if name != "router" else os.path.join(MODELS_DIR, "router.keras")
        if not os.path.exists(local_path):
            st.warning(f"Downloading {name} model from Hugging Face… (only once)")
            try:
                urllib.request.urlretrieve(url, local_path)
                st.success(f"{name.capitalize()} model downloaded successfully.")
            except Exception as e:
                st.error(f"Failed to download {name} model: {e}")
                raise e

ensure_models_present()

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Disease Classification", page_icon="🩺", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
.stImage img {border-radius: 12px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
.img-caption {text-align: center; color: #6b7280; margin-top: 8px; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("Multi-Disease Classification System")
st.write("Upload a medical image (Brain CT/MRI, Chest X-Ray, or Skin Lesion) for analysis.")


@st.cache_resource
def load_all_models():
    models = {
        'router': load_model(os.path.join(MODELS_DIR, 'router.keras')),
        'brain': load_model(os.path.join(MODELS_DIR, 'brain_specialist.keras')),
        'chest': load_model(os.path.join(MODELS_DIR, 'chest_specialist.keras')),
        'skin': load_model(os.path.join(MODELS_DIR, 'skin_specialist.keras'))
    }
    return models

# --- Class Labels ---
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
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner('Analyzing the image...'):
            processed_image, display_image = xai.prepare_image_for_model(image)
            router_prediction = models['router'].predict(processed_image)
            image_type = ROUTER_CLASSES[np.argmax(router_prediction)]
            
            st.info(f"**Step 1: Image Routing** → Identified as a **{image_type.upper()}** scan.")
            
            final_diagnosis = "" 
            
            if image_type == 'brain':
                prediction = models['brain'].predict(processed_image)
                final_diagnosis = BRAIN_CLASSES[np.argmax(prediction)]
                specialist_model = models['brain']
            elif image_type == 'chest':
                prediction = models['chest'].predict(processed_image)[0][0]
                final_diagnosis = CHEST_CLASSES[1] if prediction > 0.5 else CHEST_CLASSES[0]
                specialist_model = models['chest']
            elif image_type == 'skin':
                prediction = models['skin'].predict(processed_image)[0][0]
                final_diagnosis = SKIN_CLASSES[1] if prediction > 0.5 else SKIN_CLASSES[0]
                specialist_model = models['skin']

        st.success("**Step 2: Final Diagnosis Complete!**")
        
        st.metric("Detailed Model Output", final_diagnosis.replace("_", " ").title())
        
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

        try:
            last_conv = xai.find_last_conv_layer(specialist_model)
            if last_conv is None:
                st.warning("Could not find a suitable conv layer for Grad-CAM in the chosen model.")
            else:
                heatmap = xai.generate_grad_cam(specialist_model, processed_image, last_conv)
                if heatmap is None:
                    st.warning("Grad-CAM generation failed. Skipping explanation.")
                else:
                    try:
                        DISPLAY_SIZE = (512, 512)
                        display_image_large = display_image.resize(DISPLAY_SIZE)
                    except Exception:
                        display_image_large = display_image

                    cam_img = xai.superimpose_heatmap(display_image_large, heatmap, alpha=0.5)
                    c1, c2 = st.columns(2)
                    c1.image(display_image_large, caption='Original Uploaded Image', use_column_width=True)
                    c2.image(cam_img, caption='Explainable AI (XAI) Heatmap', use_column_width=True)
                    try:
                        report_dir = os.path.join(PROJECT_ROOT, 'reports')
                        os.makedirs(report_dir, exist_ok=True)

                        def _make_pdf():
                            try:
                                from reportlab.lib.pagesizes import letter
                                from reportlab.pdfgen import canvas
                                from reportlab.lib.utils import ImageReader
                            except Exception as e:
                                st.error("To generate PDF reports please install reportlab: pip install reportlab")
                                return None

                            import uuid, datetime, sys

                            report_id = uuid.uuid4().hex[:8]
                            ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            pdf_name = f"report_{report_id}_{ts}.pdf"
                            pdf_path = os.path.join(report_dir, pdf_name)

                            c = canvas.Canvas(pdf_path, pagesize=letter)
                            width, height = letter

                            c.setFont('Helvetica-Bold', 18)
                            c.drawString(40, height - 50, 'Medical AI Analysis Report')
                            c.setFont('Helvetica', 10)
                            c.drawString(40, height - 68, f'Report ID: {report_id}   Generated: {ts}')

                            y = height - 100
                            c.setFont('Helvetica-Bold', 12)
                            c.drawString(40, y, 'Summary')
                            y -= 18
                            c.setFont('Helvetica', 10)
                            uploaded_name = getattr(uploaded_file, 'name', 'uploaded_image')
                            c.drawString(40, y, f'Uploaded file: {uploaded_name}'); y -= 14
                            c.drawString(40, y, f'Routed image type: {image_type}'); y -= 14
                            c.drawString(40, y, f'Final diagnosis: {final_diagnosis.replace("_"," ").title()}'); y -= 14

                            try:
                                router_scores = [float(x) for x in router_prediction[0]]
                                c.drawString(40, y, f'Router scores: {router_scores}'); y -= 14
                            except Exception:
                                pass
                            try:
                                if image_type == 'brain':
                                    spec_scores = [float(x) for x in prediction[0]]
                                    c.drawString(40, y, f'Specialist scores: {spec_scores}'); y -= 14
                                else:
                                    c.drawString(40, y, f'Specialist confidence: {float(prediction):.4f}'); y -= 14
                            except Exception:
                                pass

                            c.drawString(40, y, f'Model files: {os.path.basename(os.path.join(MODELS_DIR, image_type+'_specialist.keras')) if image_type in ['brain','chest','skin'] else 'router.keras'}'); y -= 20

                            try:
                                orig_ir = ImageReader(display_image_large)
                                heat_ir = ImageReader(cam_img)
                                img_w = 220
                                img_h = 220
                                c.drawImage(orig_ir, 40, y - img_h, width=img_w, height=img_h)
                                c.drawImage(heat_ir, 60 + img_w, y - img_h, width=img_w, height=img_h)
                                y = y - img_h - 10
                            except Exception:
                                y -= 10

                            c.setFont('Helvetica-Bold', 12)
                            c.drawString(40, y, 'Heatmap Explanation')
                            y -= 16
                            c.setFont('Helvetica', 10)
                            explanation = (
                                "The Grad-CAM heatmap highlights regions the model found most important for its prediction. "
                                "Warmer colors (red/yellow) indicate higher influence on the predicted class score. "
                                "This report includes the model's routing scores and specialist confidences so clinicians can combine AI output with clinical judgement."
                            )
                            from reportlab.lib.utils import simpleSplit
                            wrapped = simpleSplit(explanation, 'Helvetica', 10, width - 80)
                            for line in wrapped:
                                c.drawString(40, y, line); y -= 12
                            y -= 6
                            c.setFont('Helvetica-Bold', 12)
                            c.drawString(40, y, 'Notes for product / ops')
                            y -= 14
                            c.setFont('Helvetica', 10)
                            env_info = f"TensorFlow {tf.__version__} | Python {sys.version.split()[0]}"
                            c.drawString(40, y, f'Environment: {env_info}')
                            c.showPage()
                            c.save()
                            return pdf_path

                        pdf_btn = st.button('Generate PDF Report')
                        if pdf_btn:
                            pdf_path = _make_pdf()
                            if pdf_path:
                                with open(pdf_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                st.success(f'Report saved to: {pdf_path}')
                                st.download_button('Download Report (PDF)', data=pdf_bytes, file_name=os.path.basename(pdf_path), mime='application/pdf')
                    except Exception as e:
                        st.warning(f"Could not prepare report UI: {e}")
        except Exception as e:
            st.warning(f"Grad-CAM step encountered an error: {e}")

except FileNotFoundError:
    st.error(f"FATAL ERROR: Model files not found in '{MODELS_DIR}'.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
