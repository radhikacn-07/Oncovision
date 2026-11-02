"""Build a validated CBIS-DDSM full-mammogram classification manifest."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "3"
CSV_DIR = DATASET / "csv"
OUTPUT = ROOT / "models" / "mammogram_manifest.csv"


CASE_FILES = (
    ("mass_case_description_train_set.csv", "Mass", "Training", "train"),
    ("mass_case_description_test_set.csv", "Mass", "Test", "test"),
    ("calc_case_description_train_set.csv", "Calc", "Training", "train"),
    ("calc_case_description_test_set.csv", "Calc", "Test", "test"),
)


def build_manifest() -> pd.DataFrame:
    labels = []
    for filename, finding, source_split, split in CASE_FILES:
        frame = pd.read_csv(CSV_DIR / filename)
        frame["image_id"] = (
            finding
            + "-"
            + source_split
            + "_"
            + frame["patient_id"].astype(str)
            + "_"
            + frame["left or right breast"].astype(str)
            + "_"
            + frame["image view"].astype(str)
        )
        frame["label"] = frame["pathology"].eq("MALIGNANT").astype(int)
        frame["split"] = split
        frame["finding"] = finding.lower()
        frame["source_series_uid"] = frame["image file path"].str.split("/").str[2]
        labels.append(frame[["image_id", "patient_id", "label", "split", "finding", "source_series_uid"]])

    cases = pd.concat(labels, ignore_index=True)
    # Multiple findings can share one full image. Any malignant pathology makes
    # the image-level target malignant; finding types are retained for auditing.
    cases = (
        cases.groupby(["image_id", "patient_id", "split", "source_series_uid"], as_index=False)
        .agg(label=("label", "max"), finding=("finding", lambda x: "+".join(sorted(set(x)))))
    )

    dicom = pd.read_csv(CSV_DIR / "dicom_info.csv", low_memory=False)
    full = dicom[
        ["SeriesInstanceUID", "PatientID", "image_path", "Rows", "Columns", "Laterality", "PatientOrientation"]
    ].rename(columns={"SeriesInstanceUID": "source_series_uid", "PatientID": "metadata_image_id"})
    # A small number of series contain more than one converted JPEG. The full
    # mammogram is the largest frame in that series; masks/crops are smaller.
    full["pixel_area"] = pd.to_numeric(full["Rows"], errors="coerce") * pd.to_numeric(
        full["Columns"], errors="coerce"
    )
    full = full.sort_values("pixel_area", ascending=False).drop_duplicates("source_series_uid")
    manifest = cases.merge(full, on="source_series_uid", how="left", validate="one_to_one")
    manifest["image_path"] = manifest["image_path"].str.replace(
        "CBIS-DDSM/jpeg/", "datasets/3/jpeg/", regex=False
    )
    manifest["exists"] = manifest["image_path"].map(lambda p: (ROOT / str(p)).is_file())

    if manifest["image_path"].isna().any():
        missing = manifest.loc[manifest["image_path"].isna(), "image_id"].tolist()[:10]
        raise RuntimeError(f"No full-mammogram mapping for: {missing}")
    if not manifest["exists"].all():
        missing = manifest.loc[~manifest["exists"], "image_path"].tolist()[:10]
        raise RuntimeError(f"Missing image files: {missing}")

    patient_splits = manifest.groupby("patient_id")["split"].nunique()
    overlapping_patients = patient_splits[patient_splits > 1].index
    # Mass and calcification source CSVs reuse a few patient IDs across their
    # published train/test partitions. Move every image for those IDs to test.
    manifest.loc[manifest["patient_id"].isin(overlapping_patients), "split"] = "test"
    patient_splits = manifest.groupby("patient_id")["split"].nunique()
    leakage = patient_splits[patient_splits > 1]
    if not leakage.empty:
        raise RuntimeError(f"Train/test patient leakage detected: {leakage.index.tolist()[:10]}")

    return manifest.sort_values(["split", "patient_id", "image_id"]).reset_index(drop=True)


if __name__ == "__main__":
    result = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns="exists").to_csv(OUTPUT, index=False)
    print(f"Saved {len(result):,} images to {OUTPUT.relative_to(ROOT)}")
    print(pd.crosstab(result["split"], result["label"], margins=True).rename(columns={0: "benign", 1: "malignant"}))
    print("Patients by split:", result.groupby("split")["patient_id"].nunique().to_dict())
