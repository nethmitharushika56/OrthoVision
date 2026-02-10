import os
from functools import lru_cache
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


BACKEND_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(BACKEND_DIR, "models", "orthovision_model.keras")
DEFAULT_TYPE_MODEL_PATH = os.path.join(BACKEND_DIR, "models", "orthovision_type_model.keras")
DEFAULT_CLASS_INDICES_PATH = os.path.join(
    BACKEND_DIR, "models", "orthovision_model.class_indices.json"
)
DEFAULT_TYPE_CLASS_INDICES_PATH = os.path.join(
    BACKEND_DIR, "models", "orthovision_type_model.class_indices.json"
)
DEFAULT_HEATMAP_PATH = os.path.join(BACKEND_DIR, "uploads", "latest_heatmap.jpg")


def _fracture_threshold() -> float:
    # Default 0.50 matches standard sigmoid binary classification.
    # Can be overridden via environment variable.
    try:
        v = float(os.environ.get("FRACTURE_THRESHOLD", "0.50"))
    except Exception:
        v = 0.50
    return max(0.01, min(0.99, v))


def ensure_dirs():
    os.makedirs(os.path.join(BACKEND_DIR, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(BACKEND_DIR, "models"), exist_ok=True)


def _load_class_indices(path: str = DEFAULT_CLASS_INDICES_PATH) -> dict:
    # Load from file if it exists, otherwise use default
    # Training should use: normal=0, fractured=1
    default = {"normal": 0, "fractured": 1}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, int) for v in data.values()):
            return data
    except Exception:
        pass
    return default


def _load_type_class_indices(path: str = DEFAULT_TYPE_CLASS_INDICES_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, int) for v in data.values()):
            return data
    except Exception:
        pass
    return {}


def _get_prediction_from_softmax(softmax_output: np.ndarray, class_indices: dict) -> tuple:
    """Get prediction from softmax output.
    Returns: (predicted_class, confidence, is_fractured, fracture_probability)
    
    For binary classification (normal vs fractured):
    - class_indices['normal'] = 0
    - class_indices['fractured'] = 1
    """
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_idx = int(np.argmax(softmax_output))
    confidence = float(softmax_output[predicted_idx])
    predicted_class = idx_to_class.get(predicted_idx, "Unknown")
    
    # For binary classification
    # fractured_idx should be 1, normal_idx should be 0
    fractured_idx = class_indices.get("fractured", 1)
    normal_idx = class_indices.get("normal", 0)
    
    # is_fractured is True if predicted class is "fractured"
    is_fractured = (predicted_idx == fractured_idx)
    
    # Fracture probability is the softmax probability for the fractured class
    if fractured_idx >= 0 and fractured_idx < len(softmax_output):
        fracture_prob = float(softmax_output[fractured_idx])
    else:
        fracture_prob = 0.0
    
    return predicted_class, confidence, is_fractured, fracture_prob


@lru_cache(maxsize=1)
def get_model(model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run training_model.py first."
        )
    model = tf.keras.models.load_model(model_path)
    # Some Keras/TF versions load Sequential models in an unbuilt state;
    # call once to ensure `inputs`/`output` are defined.
    # Using 384x384 for EfficientNetB3 model
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)
    return model


@lru_cache(maxsize=1)
def get_type_model(model_path: str = DEFAULT_TYPE_MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)
    return model


def _predict_top1(model: tf.keras.Model, img_array: np.ndarray, class_indices: dict) -> tuple[str, float, dict]:
    probs = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    idx_to_class = {v: k for k, v in class_indices.items()}
    name = idx_to_class.get(idx, "Unknown")
    all_probs = {cls: float(probs[i]) for cls, i in class_indices.items() if 0 <= int(i) < len(probs)}
    return name, conf, all_probs


def _preprocess_image(img_path: str) -> np.ndarray:
    # Using 384x384 for EfficientNetB3 model
    img = image.load_img(img_path, target_size=(384, 384))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # Use EfficientNet preprocessing instead of MobileNetV2
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    # Our model is Sequential([base_model, GAP, Dense]).
    base = model.layers[0]
    if not hasattr(base, "layers"):
        # Fallback: find a layer with 4D output.
        for layer in reversed(model.layers):
            shape = getattr(layer, "output_shape", None)
            if isinstance(shape, tuple) and len(shape) == 4:
                return layer.name
        raise ValueError("Could not determine last conv layer")

    for layer in reversed(base.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    for layer in reversed(base.layers):
        try:
            shape = getattr(layer, "output_shape", None)
            if isinstance(shape, tuple) and len(shape) == 4:
                return layer.name
        except Exception:
            pass

        try:
            out_shape = getattr(layer, "output", None)
            if out_shape is not None:
                rank = getattr(out_shape.shape, "rank", None)
                if rank == 4:
                    return layer.name
        except Exception:
            pass
    raise ValueError("Could not determine last conv layer in base model")


def _make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, target_class_idx: int = None) -> np.ndarray:
    # Simplified Grad-CAM that works with Sequential models
    # Build intermediate model that outputs conv activations
    base = model.layers[0]  # EfficientNetB3

    # In Keras 3, a loaded Sequential may not reliably expose `model.input`
    # even after predict(); however the EfficientNet base is a Functional model
    # and always has `base.input`.
    intermediate_model = tf.keras.Model(inputs=base.input, outputs=base.output)

    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Get predictions to determine target class if not provided
    preds = model(x, training=False).numpy()[0]
    if target_class_idx is None:
        target_class_idx = int(np.argmax(preds))
    
    with tf.GradientTape() as tape:
        conv_output = intermediate_model(x, training=False)
        tape.watch(conv_output)
        
        # Process through remaining layers
        gap = model.layers[1]
        head_layers = model.layers[2:]  # Dense layers
        
        x_gap = gap(conv_output)
        x_dense = x_gap
        for layer in head_layers[:-1]:  # All except last
            x_dense = layer(x_dense)
        
        logits = head_layers[-1](x_dense)  # Final output
        class_channel = logits[:, target_class_idx]
    
    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        h, w = int(conv_output.shape[1]), int(conv_output.shape[2])
        return np.zeros((h, w), dtype=np.float32)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    denom = tf.math.reduce_max(tf.maximum(heatmap, 0))
    if denom == 0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)
    heatmap = tf.maximum(heatmap, 0) / denom
    return heatmap.numpy()


def _bbox_from_heatmap(heatmap_01: np.ndarray, *, threshold: float) -> tuple[int, int, int, int] | None:
    if heatmap_01 is None or heatmap_01.size == 0:
        return None
    mask = heatmap_01 >= float(threshold)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, x1, y1

def _bbox_from_peak_component(heatmap_01: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return bbox of the connected component that contains the hottest pixel."""
    if heatmap_01 is None or heatmap_01.size == 0:
        return None

    hm = np.asarray(heatmap_01, dtype=np.float32)
    hm_max = float(np.max(hm))
    if hm_max <= 0.0:
        return None

    # Dynamic threshold based on distribution.
    # Use a high percentile so the region tracks the hottest area.
    try:
        p90 = float(np.percentile(hm, 90))
    except Exception:
        p90 = hm_max * 0.70
    thresh = max(0.15, min(hm_max * 0.70, p90))

    mask = (hm >= thresh).astype(np.uint8)
    if int(mask.max()) == 0:
        return None

    # Connected components on the thresholded mask.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None

    y_peak, x_peak = np.unravel_index(int(np.argmax(hm)), hm.shape)
    peak_label = int(labels[y_peak, x_peak])
    if peak_label == 0:
        # Peak might fall just outside threshold; fall back to a small peak box.
        h, w = hm.shape
        half = int(min(w, h) * 0.12)
        x0 = max(0, int(x_peak) - half)
        x1 = min(w - 1, int(x_peak) + half)
        y0 = max(0, int(y_peak) - half)
        y1 = min(h - 1, int(y_peak) + half)
        return x0, y0, x1, y1

    x, y, w, h, _area = stats[peak_label]
    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w - 1), int(y + h - 1)
    return x0, y0, x1, y1


def _normalize_bbox(bbox_xyxy: tuple[int, int, int, int], *, width: int, height: int) -> dict:
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))

    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    return {
        "x": float(x0) / float(width),
        "y": float(y0) / float(height),
        "w": float(w) / float(width),
        "h": float(h) / float(height),
    }


def _save_gradcam_overlay(
    img_path: str,
    heatmap: np.ndarray,
    out_path: str,
    alpha: float = 0.4,
):
    img = cv2.imread(img_path)
    if img is None:
        return out_path, None

    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    heatmap_resized = cv2.resize(heatmap, (w_img, h_img))
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Bounding box from the peak connected component (best-effort)
    hm_max = float(np.max(heatmap_resized)) if heatmap_resized.size else 0.0
    bbox = _bbox_from_peak_component(heatmap_resized)

    # If thresholding finds nothing but there is some signal, fall back to a box around the max.
    if bbox is None and hm_max > 0.0:
        max_idx = int(np.argmax(heatmap_resized))
        y_peak, x_peak = np.unravel_index(max_idx, heatmap_resized.shape)
        half = int(min(w_img, h_img) * 0.12)  # ~24% box size
        x0 = max(0, int(x_peak) - half)
        x1 = min(w_img - 1, int(x_peak) + half)
        y0 = max(0, int(y_peak) - half)
        y1 = min(h_img - 1, int(y_peak) + half)
        bbox = (x0, y0, x1, y1)

    # Save overlay image
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img
    cv2.imwrite(out_path, superimposed)

    bbox_norm = _normalize_bbox(bbox, width=w_img, height=h_img) if bbox else None
    return out_path, bbox_norm


def predict_fracture(model: tf.keras.Model, img_path: str):
    img_array = _preprocess_image(img_path)
    type_model = get_type_model()
    type_indices = _load_type_class_indices() if type_model is not None else {}
    
    print(f"\n[PREDICT] File: {img_path}")
    
    # Since we only have fracture types in the dataset, use the type model directly
    if type_model is not None and type_indices:
        # Use type model for classification
        fracture_type, fracture_type_confidence, type_all_probabilities = _predict_top1(
            type_model, img_array, type_indices
        )
        
        is_fractured = True  # Always fractured since we only classify fracture types
        confidence = fracture_type_confidence
        predicted_class = fracture_type
        
        print(f"[PREDICT] Type={fracture_type} conf={fracture_type_confidence:.4f}")
        print(f"[PREDICT] All probabilities: {type_all_probabilities}")
        
        bbox_norm = None
        heatmap_generated = False
        
        try:
            # Grad-CAM from the type model
            target_idx = type_indices.get(fracture_type)
            if target_idx is None:
                target_idx = int(np.argmax(type_model.predict(img_array, verbose=0)[0]))
            print(f"[PREDICT] Generating heatmap for {fracture_type} (idx={target_idx})")
            heatmap = _make_gradcam_heatmap(img_array, type_model, target_class_idx=int(target_idx))
            _, bbox_norm = _save_gradcam_overlay(img_path, heatmap, DEFAULT_HEATMAP_PATH)
            heatmap_generated = os.path.exists(DEFAULT_HEATMAP_PATH)
            print(f"[PREDICT] Heatmap saved successfully at {DEFAULT_HEATMAP_PATH}")
        except Exception as e:
            print(f"[PREDICT] Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
            bbox_norm = None
            heatmap_generated = False
        
        return {
            "is_fractured": True,
            "confidence": float(confidence),
            "fracture_probability": float(confidence),
            "predicted_class": fracture_type,
            "bone_part": fracture_type,
            "fracture_type": fracture_type,
            "fracture_type_confidence": float(fracture_type_confidence),
            "bbox": bbox_norm,
            "heatmap_generated": bool(heatmap_generated),
            "heatmap_url": "/get_heatmap" if heatmap_generated else None,
            "all_probabilities": {},
            "type_probabilities": type_all_probabilities,
        }
    else:
        # Fallback to binary model if type model not available
        softmax_output = model.predict(img_array, verbose=0)[0]
        class_indices = _load_class_indices()
        threshold = _fracture_threshold()

        def _p(name: str) -> float:
            idx = class_indices.get(name)
            if idx is None:
                return 0.0
            if idx < 0 or idx >= len(softmax_output):
                return 0.0
            return float(softmax_output[idx])
        
        print(f"[PREDICT] Softmax: normal={_p('normal'):.4f}, fractured={_p('fractured'):.4f}")
        print(f"[PREDICT] Class indices: {class_indices}")
        
        fractured_idx = int(class_indices.get("fractured", 1))
        normal_idx = int(class_indices.get("normal", 0))

        fracture_prob = float(softmax_output[fractured_idx]) if 0 <= fractured_idx < len(softmax_output) else 0.0
        normal_prob = float(softmax_output[normal_idx]) if 0 <= normal_idx < len(softmax_output) else 0.0

        is_fractured = bool(fracture_prob >= threshold)
        predicted_class = "fractured" if is_fractured else "normal"
        confidence = float(fracture_prob if is_fractured else normal_prob)
        
        print(
            f"[PREDICT] Result: is_fractured={is_fractured}, class={predicted_class}, "
            f"conf={confidence:.4f} (threshold={threshold:.2f})"
        )

        return {
            "is_fractured": bool(is_fractured),
            "confidence": float(confidence),
            "fracture_probability": float(fracture_prob),
            "predicted_class": predicted_class,
            "bone_part": "fractured" if is_fractured else "normal",
            "fracture_type": None,
            "fracture_type_confidence": None,
            "bbox": None,
            "heatmap_generated": False,
            "heatmap_url": None,
            "all_probabilities": {cls: float(softmax_output[idx]) for cls, idx in class_indices.items()},
            "type_probabilities": {},
        }