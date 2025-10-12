import os
import shutil
import glob

# --- Dynamic & Robust Path Configuration ---
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

# Corrected paths to match your D:\pro1\data\ structure
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SOURCE_DIR = os.path.join(DATA_ROOT, "processed")
DEST_DIR = os.path.join(DATA_ROOT, "prepared_data")
SETS = ["train", "val", "test"]

SPECIALIST_DATA = {
    "brain": ["brain_ct_healthy", "brain_ct_tumor", "brain_mri_healthy", "brain_mri_tumor"],
    "chest": ["chest_normal", "chest_pneumonia"],
    "skin": ["skin_benign", "skin_malignant"],
}

def copy_directory(src_path, dst_path):
    if not os.path.exists(dst_path):
        shutil.copytree(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")
    else:
        print(f"Exists: {dst_path}")

def main():
    print(f"Project Root Detected: {PROJECT_ROOT}")
    print("--- Starting Data Preparation (Using File Copy Method) ---")

    if not os.path.exists(SOURCE_DIR):
        print(f"FATAL ERROR: Source directory not found. Expected at: {SOURCE_DIR}")
        return
        
    os.makedirs(DEST_DIR, exist_ok=True)

    print("\n--- Preparing Specialist Datasets ---")
    for specialist_name, classes in SPECIALIST_DATA.items():
        for set_name in SETS:
            for class_name in classes:
                src_class_path = os.path.join(SOURCE_DIR, set_name, class_name)
                dest_class_path = os.path.join(DEST_DIR, f"{specialist_name}_specialist", set_name, class_name)
                if os.path.exists(src_class_path):
                    copy_directory(src_class_path, dest_class_path)
                else:
                    print(f"Warning: Source folder not found and skipped: {src_class_path}")

    print("\n--- Preparing Router Dataset ---")
    for set_name in SETS:
        for router_class in SPECIALIST_DATA.keys():
            dest_router_class_path = os.path.join(DEST_DIR, "router", set_name, router_class)
            os.makedirs(dest_router_class_path, exist_ok=True)
            for original_class_name in SPECIALIST_DATA[router_class]:
                src_path = os.path.join(SOURCE_DIR, set_name, original_class_name)
                if os.path.exists(src_path):
                    images = glob.glob(os.path.join(src_path, '*.*'))
                    for img_path in images:
                        img_name = os.path.basename(img_path)
                        dest_file_path = os.path.join(dest_router_class_path, f"{original_class_name}_{img_name}")
                        if not os.path.exists(dest_file_path):
                             shutil.copy2(img_path, dest_file_path)
    
    print("\n--- Data Preparation Complete! ---")
    print(f"New datasets are ready in: {DEST_DIR}")

if __name__ == "__main__":
    main()