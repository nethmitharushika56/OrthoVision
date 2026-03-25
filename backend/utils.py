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
MODEL_INPUT_SIZE = 384
MIN_LOCALIZATION_CONF = float(os.environ.get("MIN_LOCALIZATION_CONF", "0.25"))
# Type-only model (no explicit normal class) needs strict gating to avoid false positives.
TYPE_MIN_CONF = float(os.environ.get("TYPE_MIN_CONF", "0.30"))
TYPE_MIN_MARGIN = float(os.environ.get("TYPE_MIN_MARGIN", "0.01"))
MIN_LOCALIZATION_QUALITY = float(os.environ.get("MIN_LOCALIZATION_QUALITY", "0.30"))
MIN_HEATMAP_CONF = float(os.environ.get("MIN_HEATMAP_CONF", "0.30"))
MIN_HEATMAP_MARGIN = float(os.environ.get("MIN_HEATMAP_MARGIN", "0.05"))


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
    # Try primary model path first, then fall back to type model
    if not os.path.exists(model_path):
        type_model_path = DEFAULT_TYPE_MODEL_PATH
        if os.path.exists(type_model_path):
            model_path = type_model_path
        else:
            raise FileNotFoundError(
                f"Model file not found at {model_path} or {type_model_path}. Run training first."
            )
    # Inference service does not need training compile state/loss objects.
    model = tf.keras.models.load_model(model_path, compile=False)
    # Some Keras/TF versions load Sequential models in an unbuilt state;
    # call once to ensure `inputs`/`output` are defined.
    # Using 384x384 for EfficientNetB3 model
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)
    return model


@lru_cache(maxsize=1)
def get_type_model(model_path: str = DEFAULT_TYPE_MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    # Avoid deserialization failures from training-only custom losses/metrics.
    model = tf.keras.models.load_model(model_path, compile=False)
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)
    return model


def _predict_top1(model: tf.keras.Model, img_array: np.ndarray, class_indices: dict) -> tuple[str, float, dict]:
    arr = np.asarray(img_array, dtype=np.float32)
    pred_list = [model.predict(arr, verbose=0)[0]]

    # Lightweight test-time augmentation: horizontal flip average.
    try:
        flipped = arr[:, :, ::-1, :]
        pred_list.append(model.predict(flipped, verbose=0)[0])
    except Exception:
        pass

    probs = np.mean(np.stack(pred_list, axis=0), axis=0)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    idx_to_class = {v: k for k, v in class_indices.items()}
    name = idx_to_class.get(idx, "Unknown")
    all_probs = {cls: float(probs[i]) for cls, i in class_indices.items() if 0 <= int(i) < len(probs)}
    return name, conf, all_probs


def _top2_margin(prob_map: dict) -> float:
    vals = sorted([float(v) for v in (prob_map or {}).values()], reverse=True)
    if len(vals) < 2:
        return vals[0] if vals else 0.0
    return max(0.0, vals[0] - vals[1])


def _preprocess_image(img_path: str) -> tuple[np.ndarray, dict]:
    """Preprocess while preserving aspect ratio via letterboxing.

    Returns (model_input_batch, meta) where meta is used to project Grad-CAM
    back to original image coordinates accurately.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])

    scale = min(float(MODEL_INPUT_SIZE) / float(orig_w), float(MODEL_INPUT_SIZE) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), dtype=np.uint8)

    pad_x = (MODEL_INPUT_SIZE - new_w) // 2
    pad_y = (MODEL_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    arr = canvas.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return arr, meta


def _project_heatmap_to_original(heatmap_01: np.ndarray, meta: dict) -> np.ndarray:
    """Map a model-space heatmap to original image coordinates."""
    hm = np.asarray(heatmap_01, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)

    # Bring low-res CAM map to input canvas size.
    hm_input = cv2.resize(hm, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    pad_x = int(meta.get("pad_x", 0))
    pad_y = int(meta.get("pad_y", 0))
    new_w = int(meta.get("new_w", MODEL_INPUT_SIZE))
    new_h = int(meta.get("new_h", MODEL_INPUT_SIZE))
    orig_w = int(meta.get("orig_w", MODEL_INPUT_SIZE))
    orig_h = int(meta.get("orig_h", MODEL_INPUT_SIZE))

    y0 = max(0, pad_y)
    y1 = min(MODEL_INPUT_SIZE, pad_y + new_h)
    x0 = max(0, pad_x)
    x1 = min(MODEL_INPUT_SIZE, pad_x + new_w)

    crop = hm_input[y0:y1, x0:x1]
    if crop.size == 0:
        crop = hm_input

    hm_orig = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    hm_orig = np.clip(hm_orig, 0.0, 1.0)
    return hm_orig


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
    hm = heatmap.numpy().astype(np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    return hm


def _enhance_heatmap(heatmap_01: np.ndarray) -> np.ndarray:
    """Denoise and sharpen activation map before bbox/overlay generation."""
    hm = np.asarray(heatmap_01, dtype=np.float32)
    if hm.size == 0:
        return hm

    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.clip(hm, 0.0, 1.0)

    # Smooth isolated noisy activations.
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=2.0, sigmaY=2.0)

    max_v = float(np.max(hm))
    if max_v > 0.0:
        hm = hm / max_v

    # Sharpen high-confidence regions while suppressing weak background.
    hm = np.power(hm, 1.45)
    hm = np.clip(hm, 0.0, 1.0)
    return hm


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
    """Return bbox of the most reliable connected component in activation map."""
    if heatmap_01 is None or heatmap_01.size == 0:
        return None

    hm = _enhance_heatmap(heatmap_01)
    hm_max = float(np.max(hm))
    if hm_max <= 0.0:
        return None

    h, w = hm.shape
    area = float(h * w)
    best_score = -1.0
    best_bbox = None

    # Sweep strict-to-loose thresholds and select the strongest stable component.
    for pct in (99.2, 98.5, 97.5, 96.0, 94.0, 92.0):
        thr = float(np.percentile(hm, pct))
        if thr <= 0.0:
            continue

        mask = (hm >= thr).astype(np.uint8)
        if int(mask.max()) == 0:
            continue

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            continue

        for comp_idx in range(1, num):
            x, y, bw, bh, comp_area = stats[comp_idx]
            area_ratio = float(comp_area) / area

            # Reject tiny noise and huge full-image regions.
            if area_ratio < 0.0015 or area_ratio > 0.55:
                continue

            comp_mask = labels == comp_idx
            if not np.any(comp_mask):
                continue

            comp_vals = hm[comp_mask]
            act_sum = float(np.sum(comp_vals))
            act_mean = float(np.mean(comp_vals))
            act_max = float(np.max(comp_vals))

            # Prefer regions with high integrated activation and good compactness.
            compactness = float(min(bw, bh)) / float(max(bw, bh) + 1e-6)
            score = act_sum * (0.60 + 0.40 * act_mean) * (0.70 + 0.30 * compactness) * (0.70 + 0.30 * act_max)

            if score > best_score:
                best_score = score
                best_bbox = (int(x), int(y), int(x + bw - 1), int(y + bh - 1))

    if best_bbox is not None:
        return best_bbox

    # Fallback: local box around weighted centroid of top activations.
    y_peak, x_peak = np.unravel_index(int(np.argmax(hm)), hm.shape)
    q = float(np.percentile(hm, 99.0))
    strong = hm >= q
    ys, xs = np.where(strong)
    if len(xs) == 0:
        half = int(min(w, h) * 0.10)
        x0 = max(0, int(x_peak) - half)
        x1 = min(w - 1, int(x_peak) + half)
        y0 = max(0, int(y_peak) - half)
        y1 = min(h - 1, int(y_peak) + half)
        return x0, y0, x1, y1

    wx = np.average(xs, weights=hm[ys, xs])
    wy = np.average(ys, weights=hm[ys, xs])
    half = int(min(w, h) * 0.12)
    x0 = max(0, int(wx) - half)
    x1 = min(w - 1, int(wx) + half)
    y0 = max(0, int(wy) - half)
    y1 = min(h - 1, int(wy) + half)
    return x0, y0, x1, y1


def _localization_quality(heatmap_01: np.ndarray, bbox_xyxy: tuple[int, int, int, int] | None) -> float:
    if bbox_xyxy is None:
        return 0.0

    hm = np.asarray(heatmap_01, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = np.clip(hm, 0.0, 1.0)
    total = float(np.sum(hm))
    if total <= 1e-8:
        return 0.0

    h, w = hm.shape
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, min(int(x0), w - 1))
    x1 = max(0, min(int(x1), w - 1))
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(0, min(int(y1), h - 1))

    region = hm[y0:y1 + 1, x0:x1 + 1]
    in_mass = float(np.sum(region))
    mass_ratio = in_mass / total

    area_ratio = float((x1 - x0 + 1) * (y1 - y0 + 1)) / float(h * w)
    # Prefer moderate compact regions, penalize very large boxes.
    size_score = float(np.clip(1.0 - (area_ratio / 0.35), 0.0, 1.0))

    return float(0.75 * mass_ratio + 0.25 * size_score)


def _normalize_bbox(bbox_xyxy: tuple[int, int, int, int], *, width: int, height: int) -> dict:
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))

    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)

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
    meta: dict | None = None,
    alpha: float = 0.4,
):
    img = cv2.imread(img_path)
    if img is None:
        return out_path, None

    h_img, w_img = int(img.shape[0]), int(img.shape[1])
    if meta is not None:
        heatmap_resized = _project_heatmap_to_original(heatmap, meta)
    else:
        heatmap_resized = cv2.resize(heatmap, (w_img, h_img))
    heatmap_resized = _enhance_heatmap(heatmap_resized)

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

    quality = _localization_quality(heatmap_resized, bbox)

    # Save overlay image with confidence-weighted blending to suppress low-importance noise.
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    alpha_map = np.clip(np.power(heatmap_resized, 1.6), 0.0, 1.0)
    alpha_map = np.where(alpha_map >= 0.12, alpha_map, 0.0)
    alpha_map = (alpha_map * float(alpha))[..., None]
    superimposed = (img.astype(np.float32) * (1.0 - alpha_map)) + (heatmap_color.astype(np.float32) * alpha_map)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, superimposed)

    bbox_norm = _normalize_bbox(bbox, width=w_img, height=h_img) if bbox else None
    return out_path, bbox_norm, float(quality)


def predict_fracture(model: tf.keras.Model, img_path: str):
    img_array, preprocess_meta = _preprocess_image(img_path)
    type_model = get_type_model()
    type_indices = _load_type_class_indices() if type_model is not None else {}
    
    print(f"\n[PREDICT] File: {img_path}")
    
    # If only a type model is available (no explicit "normal" class),
    # apply conservative gating to reduce false positives on non-fracture images.
    if type_model is not None and type_indices:
        # Use type model for classification
        fracture_type, fracture_type_confidence, type_all_probabilities = _predict_top1(
            type_model, img_array, type_indices
        )

        top2_margin = _top2_margin(type_all_probabilities)
        is_fractured = bool(
            float(fracture_type_confidence) >= TYPE_MIN_CONF
            and float(top2_margin) >= TYPE_MIN_MARGIN
        )
        confidence = fracture_type_confidence
        predicted_class = fracture_type if is_fractured else "normal"
        
        print(
            f"[PREDICT] Type={fracture_type} conf={fracture_type_confidence:.4f} "
            f"margin={top2_margin:.4f} gate_conf={TYPE_MIN_CONF:.2f} gate_margin={TYPE_MIN_MARGIN:.2f} "
            f"is_fractured={is_fractured}"
        )
        print(f"[PREDICT] All probabilities: {type_all_probabilities}")
        
        bbox_norm = None
        heatmap_generated = False
        localization_hidden_reason = None
        
        if is_fractured:
            try:
                # Grad-CAM from the type model
                target_idx = type_indices.get(fracture_type)
                if target_idx is None:
                    target_idx = int(np.argmax(type_model.predict(img_array, verbose=0)[0]))
                print(f"[PREDICT] Generating heatmap for {fracture_type} (idx={target_idx})")
                heatmap = _make_gradcam_heatmap(img_array, type_model, target_class_idx=int(target_idx))
                _, bbox_norm, loc_quality = _save_gradcam_overlay(img_path, heatmap, DEFAULT_HEATMAP_PATH, meta=preprocess_meta)
                heatmap_generated = os.path.exists(DEFAULT_HEATMAP_PATH)
                # Suppress localization aggressively unless confidence, margin,
                # and map quality are all strong.
                gate_conf_threshold = max(MIN_LOCALIZATION_CONF, MIN_HEATMAP_CONF)
                if (
                    float(fracture_type_confidence) < gate_conf_threshold
                    or float(top2_margin) < MIN_HEATMAP_MARGIN
                    or float(loc_quality) < MIN_LOCALIZATION_QUALITY
                ):
                    bbox_norm = None
                    heatmap_generated = False
                    localization_hidden_reason = "Localization confidence is low for this image"
                print(f"[PREDICT] Heatmap saved successfully at {DEFAULT_HEATMAP_PATH}")
            except Exception as e:
                print(f"[PREDICT] Grad-CAM failed: {e}")
                import traceback
                traceback.print_exc()
                bbox_norm = None
                heatmap_generated = False
                localization_hidden_reason = "Heatmap generation failed"
        else:
            bbox_norm = None
            heatmap_generated = False
            localization_hidden_reason = "Not classified as fractured"
        
        return {
            "is_fractured": bool(is_fractured),
            "confidence": float(confidence if is_fractured else (1.0 - confidence)),
            "fracture_probability": float(confidence if is_fractured else 0.0),
            "predicted_class": predicted_class,
            "bone_part": fracture_type if is_fractured else "normal",
            "fracture_type": fracture_type if is_fractured else None,
            "fracture_type_confidence": float(fracture_type_confidence) if is_fractured else None,
            "bbox": bbox_norm,
            "heatmap_generated": bool(heatmap_generated),
            "heatmap_url": "/get_heatmap" if heatmap_generated else None,
            "localization_hidden_reason": localization_hidden_reason,
            "warning": "Type-only model mode: conservative gating enabled for non-fracture safety",
            "all_probabilities": {},
            "type_probabilities": type_all_probabilities if is_fractured else {},
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
            "localization_hidden_reason": "Binary model path does not generate localization",
            "all_probabilities": {cls: float(softmax_output[idx]) for cls, idx in class_indices.items()},
            "type_probabilities": {},
        }