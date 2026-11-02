"""Explain breast-imaging report text without inventing a diagnosis."""
from io import BytesIO
import re
import shutil
from pathlib import Path

BIRADS = {
    "0": ("Incomplete", None, "More imaging or comparison with earlier images is needed before a final category can be given.", "follow-up"),
    "1": ("Negative", "Essentially 0%", "No suspicious abnormality was reported. Continue routine screening as advised.", "reassuring"),
    "2": ("Benign", "Essentially 0%", "A non-cancerous finding was reported. Continue routine screening as advised.", "reassuring"),
    "3": ("Probably benign", ">0% to ≤2%", "This is very likely benign, but short-interval follow-up (often about 6 months) is usually advised.", "follow-up"),
    "4": ("Suspicious", ">2% to <95%", "The finding needs tissue diagnosis; a biopsy may be recommended.", "attention"),
    "4A": ("Low suspicion", ">2% to ≤10%", "The finding has low suspicion, but tissue diagnosis is generally recommended.", "attention"),
    "4B": ("Moderate suspicion", ">10% to ≤50%", "The finding has moderate suspicion and tissue diagnosis is generally recommended.", "attention"),
    "4C": ("High suspicion", ">50% to <95%", "The finding has high suspicion and tissue diagnosis is generally recommended.", "attention"),
    "5": ("Highly suggestive of malignancy", "≥95%", "Prompt tissue diagnosis and specialist follow-up are required. Imaging alone is not a biopsy diagnosis.", "urgent"),
    "6": ("Known biopsy-proven malignancy", None, "The report refers to cancer already confirmed by biopsy. Follow the specialist's plan.", "confirmed"),
}

FINDINGS = (
    (r"\bmass(?:es)?\b|\blump\b", "Mass or lump", "Its shape, margin and other features—not the word alone—determine concern."),
    (r"calcification", "Calcifications", "Small calcium deposits are mentioned. Many are benign; their pattern determines follow-up."),
    (r"architectural distortion", "Architectural distortion", "The normal arrangement of tissue appears altered and needs the stated follow-up."),
    (r"\basymmetr", "Asymmetry", "An area looks different from corresponding tissue; additional views may be requested."),
    (r"\bcyst", "Cyst", "A fluid-filled area is mentioned; simple cysts are commonly benign."),
    (r"lymph node|adenopathy", "Lymph node finding", "A lymph node is mentioned; its description determines significance."),
    (r"skin thickening|nipple retraction|nipple inversion", "Skin or nipple change", "A skin or nipple change is described and should be discussed with the clinician."),
)

def extract_report_text(upload):
    filename = (upload.filename or "report").lower()
    raw = upload.read()
    if not raw:
        raise ValueError("The uploaded report is empty.")
    if filename.endswith(".txt") or upload.mimetype == "text/plain":
        return raw.decode("utf-8", errors="replace")
    if filename.endswith(".pdf") or upload.mimetype == "application/pdf":
        try:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(raw)).pages)
        except Exception as exc:
            print("ACTUAL PDF ERROR:", repr(exc))
            raise ValueError("This PDF could not be read. Try another PDF or paste its text below.") from exc
        if not text.strip():
            raise ValueError("This scanned PDF has no selectable text. Upload a clear image or paste the report text below.")
        return text
    if filename.endswith((".jpg", ".jpeg", ".png", ".webp")) or upload.mimetype.startswith("image/"):
        tesseract = shutil.which("tesseract")
        windows_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if not tesseract and windows_tesseract.exists():
            tesseract = str(windows_tesseract)
        if not tesseract:
            raise ValueError("Image OCR is not installed on this computer yet. Paste the report text, or upload a text-based PDF/TXT file.")
        try:
            from PIL import Image
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract
            return pytesseract.image_to_string(Image.open(BytesIO(raw)))
        except Exception as exc:
            raise ValueError("Text could not be read from this image. Use a clearer image or paste its text.") from exc
    raise ValueError("Upload a PDF, TXT, JPG, PNG, or WebP report.")

def explain_report(text):
    cleaned = re.sub(r"[ \t]+", " ", text or "").strip()
    if len(cleaned) < 25:
        raise ValueError("Not enough readable text was found. Paste the Findings, Impression and Recommendation sections.")
    lower = cleaned.lower()
    common = {
        "birads": None, "risk_range": None, "risk_source": None,
        "extracted_text": cleaned[:12000],
        "disclaimer": "Educational explanation only. It does not diagnose or rule out cancer and does not replace the radiologist or treating clinician.",
    }
    # These two documents are frequently distributed as examples. Their option
    # lists must never be mistaken for a patient's selected findings.
    if "sample lay letter for abnormal finding" in lower and ("name of patient/id" in lower or "[mammogra" in lower):
        return {**common, "document_type": "sample-template", "category_label": "Abnormal finding letter template",
                "level": "attention",
                "overview": "This document is a sample letter intended for BI-RADS 4–5 abnormal findings. It says an abnormality would require biopsy, but the patient name, examination and actual finding are placeholders, so it is not a completed patient result.",
                "findings": [{"title": "Abnormal/biopsy wording detected", "explanation": "The template contains wording for an abnormality requiring biopsy. It does not show which finding or BI-RADS category applies to this patient.", "negated": False}],
                "recommendation": "If a clinic issued this as your completed letter, contact the facility promptly and ask for the signed final imaging report, exact BI-RADS category, and biopsy instructions.",
                "questions": ["Is this my completed result or an unfilled sample letter?", "What is my exact BI-RADS category?", "Which area needs biopsy and how will it be scheduled?"]}
    if ("bi-rads: 0/1/2/3/4a/4b/4c/5/6" in lower or
            ("shape: none/oval/round/irregular" in lower and "impression:" in lower)):
        return {**common, "document_type": "blank-proforma", "category_label": "Blank breast-imaging form",
                "level": "unknown",
                "overview": "This is an unfilled report form. It lists possible choices, but no patient-specific BI-RADS category, measurement, impression or finding has been selected, so it cannot be classified as normal or abnormal.",
                "findings": [{"title": "No patient finding entered", "explanation": "Words such as mass, irregular, calcifications and biopsy appear only in option lists; they are not selected results.", "negated": False}],
                "recommendation": "Ask the imaging facility for the completed and signed report containing Findings, Impression, final BI-RADS category and Recommendation.",
                "questions": ["Can I receive the completed signed report?", "What is the final BI-RADS category?", "What follow-up is actually recommended for me?"]}
    # Accept BI-RADS®, OCR replacement characters and normal punctuation, while
    # rejecting option lists (0/1/2...) and ranges (4-5) as selected categories.
    matches = re.findall(r"\bBI[\s-]*RADS[^0-9A-Z\n]{0,12}(?:category|assessment)?[^0-9\n]{0,6}([0-6](?:\s*[ABC])?)\b(?!\s*[/\-]\s*[0-6])", cleaned, re.I)
    category = re.sub(r"\s+", "", matches[-1]).upper() if matches else None
    if category in BIRADS:
        label, risk, overview, level = BIRADS[category]
    else:
        label, risk, level = "Not found in report", None, "unknown"
        if re.search(r"no (?:mammographic )?evidence of malignancy|no suspicious (?:mass|finding|abnormality)", cleaned, re.I):
            overview, level = "The wording is reassuring, but no BI-RADS category was detected. Confirm the final impression with the reporting clinic.", "reassuring"
        elif re.search(r"abnormality\s+that\s+requires\s+(?:a\s+)?biopsy|suspicious\s+(?:abnormality|finding)", cleaned, re.I):
            label, level = "Abnormal wording detected", "attention"
            overview = "The report uses abnormal or suspicious wording and recommends biopsy, but no single BI-RADS category was detected. A percentage cannot be calculated safely from this wording alone."
        else:
            overview = "No BI-RADS category was detected, so this tool cannot label the report normal or abnormal or calculate a risk range."
    findings = []
    for pattern, title, explanation in FINDINGS:
        if re.search(pattern, cleaned, re.I):
            negated = bool(re.search(rf"\b(?:no|without)\s+(?:suspicious\s+)?(?:[\w,-]+\s+){{0,3}}(?:{pattern})", cleaned, re.I))
            findings.append({"title": title, "explanation": ("The report says this was not seen. " if negated else "") + explanation, "negated": negated})
    if not findings:
        findings.append({"title": "No common finding term detected", "explanation": "Read the radiologist's Impression and Recommendation with your clinician.", "negated": False})
    recommendation = "Follow the radiologist's recommendation and discuss it with your doctor."
    for pattern in (r"(?:recommendation|recommend|follow[- ]?up)\s*[:\-]?\s*([^\n]{8,240})", r"(biopsy[^.\n]{0,180})", r"(additional imaging[^.\n]{0,180})"):
        match = re.search(pattern, cleaned, re.I)
        if match:
            recommendation = match.group(1).strip().rstrip(".") + "."
            break
    return {"birads": category, "document_type": "completed-report", "category_label": label, "risk_range": risk,
            "risk_source": "ACR BI-RADS category stated in this report" if risk else None,
            "level": level, "overview": overview, "findings": findings,
            "recommendation": recommendation,
            "questions": ["What does the final BI-RADS category mean for me?", "When and what follow-up do I need?", "Do I need comparison images, ultrasound, or biopsy?"],
            "extracted_text": cleaned[:12000],
            "disclaimer": "Educational explanation only. It does not diagnose or rule out cancer and does not replace the radiologist or treating clinician."}
