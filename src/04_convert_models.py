import os
import tensorflow as tf

# --- Dynamic & Robust Path Configuration ---
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- List of models to convert ---
MODEL_NAMES = [
    "router",
    "brain_specialist",
    "chest_specialist",
    "skin_specialist"
]

def convert_models():
    """
    Loads each model from the legacy .h5 format and re-saves it
    in the modern, more robust .keras format.
    """
    print("--- Starting Model Conversion ---")
    if not os.path.exists(MODELS_DIR):
        print(f"FATAL: Models directory not found at {MODELS_DIR}")
        return

    for name in MODEL_NAMES:
        h5_path = os.path.join(MODELS_DIR, f"{name}.h5")
        keras_path = os.path.join(MODELS_DIR, f"{name}.keras")

        if os.path.exists(h5_path):
            print(f"Converting '{name}.h5'...")
            try:
                # Load the model from the old format
                model = tf.keras.models.load_model(h5_path)
                
                # Save the model in the new format
                model.save(keras_path)
                
                print(f"  -> Successfully converted to '{name}.keras'")
            except Exception as e:
                print(f"  -> ERROR converting {name}: {e}")
        else:
            print(f"Warning: '{name}.h5' not found. Skipping.")
    
    print("\n--- Model Conversion Complete! ---")

if __name__ == "__main__":
    convert_models()