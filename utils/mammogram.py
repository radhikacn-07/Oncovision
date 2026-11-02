"""Local mammogram inference and Grad-CAM for the trained research model."""

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from utils.image_preprocessing import prepare_mammogram


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mammogram_efficientnet.keras"
METRICS_PATH = ROOT / "models" / "mammogram_metrics.json"
IMAGE_SIZE = 224
_model = None


def _validate_single_view(gray):
    """Reject obvious multi-panel screenshots outside the training domain."""
    dark = (gray <= 20).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    large_dark_panels = sum(
        int(stats[index, cv2.CC_STAT_AREA]) >= 0.08 * gray.size
        for index in range(1, count)
    )
    if large_dark_panels > 1:
        raise ValueError(
            "This appears to be a multi-panel, annotated, or report screenshot. "
            "Please upload one original single-view mammogram (CC or MLO) without text, circles, or side-by-side images."
        )


def extract_mammogram_views(raw, filename="mammogram.jpg"):
    """Split common 2-up/2x2 exported contact sheets into individual views.

    Single-view mammograms are returned unchanged. The split is intentionally
    limited to strong, regular layouts so arbitrary screenshots are not treated
    as valid clinical views.
    """
    encoded = np.frombuffer(raw, dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("The uploaded file is not a readable image.")
    height, width = gray.shape
    panels = None
    if width / height >= 1.35:
        midpoint = width // 2
        panels = [gray[:, :midpoint], gray[:, midpoint:]]
    elif height / width >= 1.35:
        x_band = gray[:, int(width * .47):int(width * .53)].mean(axis=0)
        y_band = gray[int(height * .47):int(height * .53), :].mean(axis=1)
        if np.ptp(x_band) > 70 and np.ptp(y_band) > 120:
            mid_x, mid_y = width // 2, height // 2
            panels = [gray[:mid_y, :mid_x], gray[:mid_y, mid_x:], gray[mid_y:, :mid_x], gray[mid_y:, mid_x:]]
    if not panels:
        return [(filename, raw)]
    stem = Path(filename).stem
    views = []
    for index, panel in enumerate(panels, 1):
        # Remove a narrow divider without cropping meaningful breast tissue.
        trim_y, trim_x = max(1, panel.shape[0] // 200), max(1, panel.shape[1] // 200)
        panel = panel[trim_y:-trim_y or None, trim_x:-trim_x or None]
        ok, output = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if ok:
            views.append((f"{stem}-view-{index}.jpg", output.tobytes()))
    return views or [(filename, raw)]


def model_available():
    return MODEL_PATH.is_file() and METRICS_PATH.is_file()


def model_metadata():
    if not METRICS_PATH.is_file():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _load_model():
    global _model
    if _model is None:
        if not model_available():
            raise RuntimeError("The trained mammogram model is not installed yet.")
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _prepare(raw):
    encoded = np.frombuffer(raw, dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("The uploaded file is not a readable image.")
    _validate_single_view(gray)
    # EfficientNet always receives a normalized 224x224 tensor. Pixel
    # dimensions alone must not reject an otherwise readable image: the shared
    # preprocessor fits small, portrait, landscape and large images safely.
    return prepare_mammogram(gray, IMAGE_SIZE)


def _gradcam(model, image, malignant_probability, threshold=0.5):
    import tensorflow as tf

    base = model.get_layer("efficientnetb0")
    tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        features = base(model.get_layer("augmentation")(tensor, training=False), training=False)
        tape.watch(features)
        pooled = model.get_layer("feature_pool")(features)
        pooled = model.get_layer("dropout")(pooled, training=False)
        output = model.get_layer("malignancy_probability")(pooled)
        target = output[:, 0] if malignant_probability >= threshold else 1.0 - output[:, 0]
    gradients = tape.gradient(target, features)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * features, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + tf.keras.backend.epsilon())
    heatmap = cv2.resize(heatmap.numpy(), (IMAGE_SIZE, IMAGE_SIZE))
    return heatmap


def _overlay_data_url(image, heatmap):
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.55, colored, 0.45, 0)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Unable to encode the Grad-CAM overlay.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _image_data_url(image):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Unable to encode the processed mammogram.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _attention_peak(heatmap):
    """Return a stable hotspot centroid, ignoring common border artifacts."""
    smoothed = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), 3)
    height, width = smoothed.shape
    margin_y, margin_x = max(2, int(height * 0.06)), max(2, int(width * 0.06))
    interior = smoothed.copy()
    interior[:margin_y, :] = 0
    interior[-margin_y:, :] = 0
    interior[:, :margin_x] = 0
    interior[:, -margin_x:] = 0
    positive = interior[interior > 0]
    if positive.size:
        cutoff = np.percentile(positive, 97)
        weights = np.where(interior >= cutoff, interior, 0)
        total = float(weights.sum())
        if total > 0:
            yy, xx = np.indices(weights.shape)
            return int(round(float((yy * weights).sum()) / total)), int(round(float((xx * weights).sum()) / total))
    return tuple(int(value) for value in np.unravel_index(np.argmax(interior), interior.shape))


def _attention_data_url(image, peak_y, peak_x):
    """Bake the calculated Grad-CAM peak into the image for exact alignment."""
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    color = (125, 47, 255)
    radius = 15
    cv2.circle(output, (peak_x, peak_y), radius, color, 3, cv2.LINE_AA)
    cv2.line(output, (peak_x - radius - 7, peak_y), (peak_x + radius + 7, peak_y), color, 2, cv2.LINE_AA)
    cv2.line(output, (peak_x, peak_y - radius - 7), (peak_x, peak_y + radius + 7), color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise RuntimeError("Unable to encode the attention image.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def analyze_mammogram(file_storage):
    raw = file_storage.read()
    file_storage.seek(0)
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Image is larger than the 20 MB upload limit.")
    image = _prepare(raw)
    model = _load_model()
    probability = float(model.predict(image[None, ...].astype(np.float32), verbose=0)[0, 0])
    metadata = model_metadata() or {}
    threshold = float(metadata.get("threshold", 0.5))
    label = "Malignant" if probability >= threshold else "Benign"
    class_probability = probability if label == "Malignant" else 1.0 - probability
    heatmap = _gradcam(model, image, probability, threshold)
    peak_y, peak_x = _attention_peak(heatmap)
    return {
        "class_name": label,
        "confidence": round(class_probability * 100, 1),
        "malignancy_probability": round(probability * 100, 1),
        "heatmap_data_url": _overlay_data_url(image, heatmap),
        "processed_image_data_url": _image_data_url(image),
        "attention_image_data_url": _attention_data_url(image, peak_y, peak_x),
        "attention_peak": {
            "x_percent": round(float(peak_x) / IMAGE_SIZE * 100, 1),
            "y_percent": round(float(peak_y) / IMAGE_SIZE * 100, 1),
        },
        "model_name": metadata.get("model", "EfficientNetB0 transfer-learning model"),
        "test_auc": metadata.get("auc"),
        "decision_threshold": threshold,
    }
