"""Train and evaluate a reproducible WDBC feature classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main():
    frame = pd.read_csv(ROOT / "data.csv")
    frame = frame.drop(columns=[column for column in frame.columns if column.startswith("Unnamed")], errors="ignore")
    target = frame.pop("diagnosis").map({"B": 0, "M": 1})
    frame = frame.drop(columns=["id"], errors="ignore")
    x_train, x_test, y_train, y_test = train_test_split(
        frame, target, test_size=0.2, random_state=2026, stratify=target
    )
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))]
    )
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (local data.csv)",
        "features": list(frame.columns),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "auc": round(float(roc_auc_score(y_test, probability)), 4),
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "clinical_status": "research prototype; not a cancer-stage model",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": list(frame.columns)}, MODEL_DIR / "wdbc_logreg.joblib")
    (MODEL_DIR / "wdbc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
