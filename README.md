# Multi-Disease Classification System 🩺

A professional, AI-powered medical image analysis application built with **Streamlit** and **TensorFlow**. This system automatically routes uploaded medical images to specialized models for detailed diagnosis and provides Explainable AI (XAI) visualisations to support clinical interpretation.

## 🌟 Features

- **Automated Routing**: Intelligently identifies the type of scan (Brain CT/MRI, Chest X-Ray, or Skin Lesion) and directs it to the appropriate specialist model.
- **Specialized Diagnosis**:
  - **Brain**: Detects tumors in both CT and MRI modalities.
  - **Chest**: Identifies Pneumonia vs. Normal scans.
  - **Skin**: Classifies lesions as Benign or Malignant.
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to highlight the specific regions of interest that influenced the model's decision.
- **Automated Model Management**: Automatically downloads the latest trained models from Hugging Face on the first run.
- **Digital Reports**: Generates comprehensive PDF analysis reports including summary data, confidence scores, and XAI heatmaps.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Image Processing**: PIL, OpenCV, NumPy
- **Explainability**: Grad-CAM (Custom Implementation)
- **Reporting**: ReportLab

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/likhitha2049/lti-project.git
   cd lti-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you want to generate PDF reports, ensure `reportlab` is installed.*

3. **Run the application**:
   ```bash
   streamlit run src/streamlit_app.py
   ```

## 📂 Project Structure

- `src/`: Core source code including the Streamlit app and XAI utilities.
- `models/`: Directory where trained `.keras` models are stored (managed automatically).
- `data/`: (Optional) Directory for local datasets.
- `reports/`: Generated PDF reports are saved here.

## 🧠 Models

The system uses four specialized models hosted on Hugging Face:
- `router`: Classifies the image source.
- `brain_specialist`: Detailed brain analysis.
- `chest_specialist`: Pneumonia detection.
- `skin_specialist`: Skin lesion classification.

---
**Disclaimer**: *This application is for demonstration and research purposes only. It should not be used for actual medical diagnosis without the supervision of a qualified healthcare professional.*
