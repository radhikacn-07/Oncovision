# OncoVision

OncoVision is a Flask research prototype with separate patient and doctor experiences, local mammogram classification, Grad-CAM attention overlays, a WDBC tabular classifier, and a TNM decision-support rule engine.

## Run locally

```bash
python -m pip install -r requirements.txt
python scripts/build_mammogram_manifest.py
python scripts/train_mammogram_model.py
python scripts/train_tabular_model.py
python app.py
```

Open http://localhost:5000/. The health endpoint at `/api/health` reports whether the mammogram model and its held-out metrics are installed.

## Models and data

- Mammogram classifier: EfficientNetB0 transfer learning on local CBIS-DDSM full mammograms with patient-separated train, validation, and test partitions.
- Explainability: Grad-CAM attention overlay from the trained image model. It is not a thermal image or confirmed lesion segmentation.
- Tabular classifier: logistic regression trained on the local Wisconsin Diagnostic Breast Cancer feature CSV.
- Clinical stage: a separate TNM rule engine. WDBC has no cancer-stage labels, so its classifier is not used to invent a stage.

The upload API returns an error when the local model is unavailable. It never falls back to filename-based confidence or an unvalidated third-party vision response.

## Important note

This software and its generated reports are research/educational prototypes, not clinically validated medical devices. They do not diagnose or exclude cancer and must not replace radiologist interpretation, diagnostic imaging, pathology, or professional treatment decisions.
