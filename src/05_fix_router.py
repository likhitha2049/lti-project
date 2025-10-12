import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import os

# --- Dynamic & Robust Path Configuration ---
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
IMG_SIZE = (224, 224)

def rebuild_and_fix_router():
    """
    Rebuilds the router model architecture, loads the weights from the
    problematic .h5 file, and saves it in the correct .keras format.
    """
    print("--- Starting Router Model Rescue Operation ---")
    
    # --- Step 1: Rebuild the EXACT SAME architecture as during training ---
    input_tensor = tf.keras.layers.Input(shape=IMG_SIZE + (3,))
    
    base_model = MobileNetV2(
        input_tensor=input_tensor,
        include_top=False,
        weights=None  # Start with an empty structure
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    # The router had 3 output classes (brain, chest, skin)
    outputs = Dense(3, activation='softmax')(x)
    
    # Create the final model structure
    model = Model(inputs=input_tensor, outputs=outputs)
    print("Model architecture rebuilt successfully.")

    # --- Step 2: Load ONLY the weights from the old .h5 file ---
    h5_path = os.path.join(MODELS_DIR, "router.h5")
    if not os.path.exists(h5_path):
        print(f"FATAL: Cannot find source file 'router.h5' at {h5_path}")
        return
        
    try:
        model.load_weights(h5_path)
        print("Successfully loaded weights from 'router.h5'.")
    except Exception as e:
        print(f"FATAL: Failed to load weights. Error: {e}")
        return

    # --- Step 3: Save the complete, fixed model in the new .keras format ---
    keras_path = os.path.join(MODELS_DIR, "router.keras")
    model.save(keras_path)
    print(f"--- Rescue Complete! ---")
    print(f"Fixed model saved to: {keras_path}")

if __name__ == "__main__":
    rebuild_and_fix_router()