import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


def generate_grad_cam(model, img_array, layer_name):
    """
    Generates a Grad-CAM heatmap.
    - model: a tf.keras Model
    - img_array: a preprocessed image tensor shaped (1, H, W, C)
    - layer_name: name of the convolutional layer to use for Grad-CAM
    Returns: heatmap as a 2D numpy array (values in [0,1]) or None on error
    """
    if not layer_name:
        print("Error: No layer name provided for Grad-CAM.")
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except Exception as e:
        print(f"Error creating Grad-CAM model with layer '{layer_name}': {e}")
        return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        print("Error: Gradient is None. Check model structure and layer name.")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy()


def superimpose_heatmap(original_img, heatmap, alpha=0.5):
    """
    Applies the heatmap as an overlay on the original image.
    - original_img: PIL.Image or numpy array (H,W,3) in RGB
    - heatmap: 2D numpy array in [0,1]
    - alpha: blend factor for the heatmap
    Returns: a PIL.Image (RGB) with the heatmap superimposed.
    """
    # Ensure we have a numpy RGB image
    if isinstance(original_img, Image.Image):
        img = np.array(original_img)
    else:
        img = np.array(original_img)

    # Handle grayscale images
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Drop alpha channel if present
    if img.shape[2] == 4:
        img = img[..., :3]

    # Convert from RGB (PIL) to BGR for OpenCV color maps
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend heatmap with the image
    superimposed = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    # Convert back to RGB for PIL/Streamlit display
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_rgb)


def find_last_conv_layer(model):
    """
    Heuristic to locate a sensible last convolutional layer for Grad-CAM.
    Returns the layer name or None.
    """
    # Iterate layers in reverse and pick the first 2D conv-like layer.
    # Be defensive: some layer types (Concatenate, custom wrappers) may not
    # expose output_shape attribute; avoid accessing it directly.
    for layer in reversed(model.layers):
        # Direct Conv2D instance (most reliable)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        # Heuristic by class name (covers many wrappers/custom convs)
        cls_name = layer.__class__.__name__.lower()
        if 'conv' in cls_name:
            return layer.name

        # If the layer is a nested Model or has sublayers, search inside it
        if hasattr(layer, 'layers') and layer.layers:
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
                if 'conv' in sub.__class__.__name__.lower():
                    return sub.name

    return None


def prepare_image_for_model(pil_image, target_size=(224, 224)):
    """
    Resize and normalize a PIL image for the project's models.
    Returns (model_input, display_image)
    - model_input: numpy array shaped (1, H, W, C) with values in [0,1]
    - display_image: PIL.Image resized to target_size (RGB)
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    display_image = pil_image.resize(target_size)
    arr = np.array(display_image).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, display_image
