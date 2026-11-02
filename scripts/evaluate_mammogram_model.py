"""Evaluate the saved mammogram checkpoint on the untouched patient test set."""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from train_mammogram_model import MANIFEST, METRICS_PATH, MODEL_PATH, make_dataset


def main():
    frame = pd.read_csv(MANIFEST)
    train = frame[frame["split"].eq("train")]
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(make_dataset(test), verbose=1).reshape(-1)
    existing = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    threshold = float(existing.get("threshold", 0.5))
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["label"], predicted, labels=[0, 1]).ravel()
    metrics = {
        "dataset": "CBIS-DDSM",
        "model": "EfficientNetB0 ImageNet transfer learning",
        "image_size": 224,
        "epochs_completed": 3,
        "train_pool_images": len(train),
        "test_images": len(test),
        "test_patients": int(test["patient_id"].nunique()),
        "auc": round(float(roc_auc_score(test["label"], probabilities)), 4),
        "accuracy": round(float(accuracy_score(test["label"], predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test["label"], predicted)), 4),
        "f1": round(float(f1_score(test["label"], predicted)), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": threshold,
        "threshold_source": existing.get("threshold_source", "saved validation-derived threshold"),
        "clinical_status": "research prototype; not clinically validated",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
