"""Train an educational CBIS-DDSM mammogram classifier with patient splits."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from utils.image_preprocessing import prepare_mammogram


MANIFEST = ROOT / "models" / "mammogram_manifest.csv"
MODEL_PATH = ROOT / "models" / "mammogram_efficientnet.keras"
METRICS_PATH = ROOT / "models" / "mammogram_metrics.json"
IMAGE_SIZE = 224
BATCH_SIZE = int(os.getenv("ONCOVISION_BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("ONCOVISION_EPOCHS", "8"))
WARMUP_EPOCHS = min(int(os.getenv("ONCOVISION_WARMUP_EPOCHS", "2")), EPOCHS)
SEED = 2026


def load_image(path):
    path = path.numpy().decode("utf-8") if hasattr(path, "numpy") else str(path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return prepare_mammogram(image, IMAGE_SIZE).astype(np.float32)


def make_dataset(frame, training=False, cache_name=None):
    paths = [str(ROOT / value) for value in frame["image_path"]]
    labels = frame["label"].astype("float32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    def decode(path, label):
        image = tf.py_function(load_image, [path], Tout=tf.float32)
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_name:
        cache_path = ROOT / "models" / "cache" / f"preprocessed-v2-{cache_name}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(cache_path))
    if training:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    augmentation = tf.keras.Sequential(
        [tf.keras.layers.RandomFlip("horizontal", seed=SEED), tf.keras.layers.RandomRotation(0.02, seed=SEED)],
        name="augmentation",
    )
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="mammogram")
    x = augmentation(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="feature_pool")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="oncovision_mammogram_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )


def calibrated_threshold(labels, probabilities):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    index = int(np.argmax((true_positive_rate - false_positive_rate)[finite]))
    return float(thresholds[finite][index])


def main():
    tf.keras.utils.set_random_seed(SEED)
    frame = pd.read_csv(MANIFEST)
    official_train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(splitter.split(official_train, groups=official_train["patient_id"]))
    train, val = official_train.iloc[train_idx], official_train.iloc[val_idx]

    negatives, positives = np.bincount(train["label"].astype(int), minlength=2)
    class_weight = {0: len(train) / (2 * negatives), 1: len(train) / (2 * positives)}
    print(f"Patient-separated images: train={len(train)}, validation={len(val)}, test={len(test)}", flush=True)
    print(f"Training for {EPOCHS} epochs on {tf.config.list_physical_devices()}", flush=True)

    model = build_model()
    train_dataset = make_dataset(train, training=True, cache_name="train")
    validation_dataset = make_dataset(val, cache_name="validation")
    test_dataset = make_dataset(test, cache_name="test")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=1, factor=0.3, min_lr=1e-7),
    ]
    warmup = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=WARMUP_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Fine-tune only the final EfficientNet blocks; keep BatchNorm frozen for
    # stability with the small mammography batches.
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers[-40:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    fine_tune = model.fit(
        train_dataset, validation_data=validation_dataset,
        initial_epoch=WARMUP_EPOCHS, epochs=EPOCHS, class_weight=class_weight,
        callbacks=callbacks,
    ) if EPOCHS > WARMUP_EPOCHS else None

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    validation_probabilities = model.predict(validation_dataset, verbose=1).reshape(-1)
    threshold = calibrated_threshold(val["label"], validation_probabilities)

    probabilities = model.predict(test_dataset, verbose=1).reshape(-1)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["label"], predicted, labels=[0, 1]).ravel()
    metrics = {
        "dataset": "CBIS-DDSM",
        "model": "EfficientNetB0 ImageNet transfer learning",
        "image_size": IMAGE_SIZE,
        "epochs_completed": len(warmup.history["loss"]) + (len(fine_tune.history["loss"]) if fine_tune else 0),
        "train_images": len(train),
        "validation_images": len(val),
        "test_images": len(test),
        "test_patients": int(test["patient_id"].nunique()),
        "auc": round(float(roc_auc_score(test["label"], probabilities)), 4),
        "accuracy": round(float(accuracy_score(test["label"], predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test["label"], predicted)), 4),
        "f1": round(float(f1_score(test["label"], predicted)), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": round(threshold, 6),
        "threshold_source": "validation-set Youden J; test set untouched",
        "training_strategy": "ImageNet warmup then final-40-layer fine-tuning",
        "clinical_status": "research prototype; not clinically validated",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
