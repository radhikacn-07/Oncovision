"""Shared mammogram preprocessing used by training and live inference."""

import cv2
import numpy as np


IMAGE_SIZE = 224


def isolate_breast(gray):
    """Keep the dominant breast component and remove labels/scanner canvas."""
    if gray is None or gray.ndim != 2:
        raise ValueError("A grayscale mammogram is required.")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    positive = blurred[blurred > 0]
    if not positive.size:
        raise ValueError("The image contains no visible mammogram pixels.")
    # A conservative intensity threshold plus closing joins breast tissue while
    # leaving small corner labels as separate components.
    threshold = max(4, int(np.percentile(positive, 18)))
    mask = (blurred >= threshold).astype(np.uint8)
    shortest_side = min(gray.shape)
    kernel_size = max(3, min(11, shortest_side // 8))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("A breast region could not be isolated from this image.")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    tissue_mask = (labels == component).astype(np.uint8)
    x, y, width, height, _ = stats[component]
    margin_x = max(4, int(width * 0.025))
    margin_y = max(4, int(height * 0.025))
    x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
    x1 = min(gray.shape[1], x + width + margin_x)
    y1 = min(gray.shape[0], y + height + margin_y)
    cleaned = gray * tissue_mask
    return cleaned[y0:y1, x0:x1]


def prepare_mammogram(gray, image_size=IMAGE_SIZE):
    tissue = isolate_breast(gray)
    background = tissue == 0
    positive = tissue[tissue > 0]
    if positive.size:
        low, high = np.percentile(positive, (1, 99))
        if high > low:
            tissue = np.clip((tissue.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        tissue = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(tissue)
        tissue[background] = 0
    height, width = tissue.shape
    scale = min(image_size / width, image_size / height)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        tissue,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=interpolation,
    )
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    y0 = (image_size - resized.shape[0]) // 2
    x0 = (image_size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return np.repeat(canvas[..., None], 3, axis=-1)
