import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB3, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os

# --- Dynamic & Robust Path Configuration ---
try:
    SRC_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_SCRIPT_DIR, os.pardir))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

# Corrected paths to match your D:\pro1\data\ structure
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
PREPARED_DATA_DIR = os.path.join(DATA_ROOT, "prepared_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- Model & Training Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30 
MODEL_MAP = {"ResNet50": ResNet50, "DenseNet121": DenseNet121, "EfficientNetB3": EfficientNetB3, "MobileNetV2": MobileNetV2}

def build_model(base_model_name, num_classes):
    """
    Builds and compiles a transfer learning model.
    This version includes a more robust fix for the EfficientNet input shape bug
    by manually loading weights.
    """
    BaseModelClass = MODEL_MAP.get(base_model_name)
    if not BaseModelClass:
        raise ValueError(f"Model name '{base_model_name}' not recognized.")
    
    # --- ROBUST FIX APPLIED HERE ---
    
    # 1. Create the model structure with the correct 3-channel input shape but NO weights.
    # This guarantees the architecture is correct from the start.
    input_tensor = tf.keras.layers.Input(shape=IMG_SIZE + (3,))
    base_model = BaseModelClass(
        input_tensor=input_tensor,
        include_top=False,
        weights=None  # <-- CRITICAL: Do not load weights here
    )

    # 2. Create a temporary model instance WITH the pre-trained weights.
    # We will use this model as a source to copy weights from.
    temp_model = BaseModelClass(
        input_shape=IMG_SIZE + (3,), # Use standard input_shape for this instance
        include_top=False,
        weights='imagenet'
    )

    # 3. Manually copy the weights from the temporary model to our main model.
    # This bypasses the faulty weight-loading mechanism of the constructor.
    for i in range(len(base_model.layers)):
        base_model.layers[i].set_weights(temp_model.layers[i].get_weights())
        
    # --- END OF FIX ---
    
    base_model.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    loss, outputs = ('binary_crossentropy', Dense(1, activation='sigmoid')(x)) if num_classes == 2 else ('categorical_crossentropy', Dense(num_classes, activation='softmax')(x))
    
    model = Model(inputs=input_tensor, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=['accuracy'])
    print(f"Model '{base_model_name}' built successfully with manual weight loading.")
    return model

def train(model_name, data_folder_name, base_model_arch):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print(f"✅ CUDA is enabled. Using {len(gpus)} GPU(s).")
    else: print("⚠️ WARNING: No GPU detected. Training on CPU.")
    
    data_dir = os.path.join(PREPARED_DATA_DIR, data_folder_name)
    save_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n--- Training Model: {model_name} ---")
    print(f"Reading data from: {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    num_classes = len(os.listdir(train_dir))

    train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255.)
    
    class_mode = 'binary' if num_classes == 2 else 'categorical'
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode=class_mode, shuffle=True)
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode=class_mode, shuffle=False)
    
    model = build_model(base_model_arch, num_classes)
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]
    model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=callbacks)
    print(f"--- Finished training {model_name}. Best model saved to {save_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--name", type=str, required=True, help="A name for the model (e.g., router, brain_specialist).")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder name inside 'prepared_data' (e.g., router, brain_specialist).")
    parser.add_argument("--arch", type=str, required=True, help=f"Model architecture. Choices: {list(MODEL_MAP.keys())}")
    args = parser.parse_args()
    train(model_name=args.name, data_folder_name=args.data_folder, base_model_arch=args.arch)