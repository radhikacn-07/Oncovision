import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_DIR = Path(__file__).resolve().parent.parent / "datasets" / "2" / "Dataset_BUSI_with_GT"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
_trained_model = None
_trained_model_error = None


def _load_trained_model():
    global _trained_model, _trained_model_error
    if _trained_model is not None:
        return _trained_model
    if _trained_model_error is not None:
        raise RuntimeError(_trained_model_error)

    try:
        import tensorflow as tf

        class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        _trained_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
        )
        return _trained_model
    except Exception as error:
        _trained_model_error = f"Unable to load the trained model: {error}"
        raise RuntimeError(_trained_model_error) from error


def _load_labels():
    labels = {}
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        index, label = line.strip().split(maxsplit=1)
        labels[int(index)] = label.title()
    return labels


def classify_uploaded_image(image):
    """Classify an uploaded image with the trained BUSI model."""
    try:
        import numpy as np
        from PIL import Image

        model = _load_trained_model()
        resized = Image.open(image).convert("RGB").resize((224, 224))
        batch = np.asarray(resized, dtype="float32") / 255.0
        scores = np.asarray(model.predict(np.expand_dims(batch, axis=0), verbose=0)).reshape(-1)
        if not scores.size:
            raise RuntimeError("The trained model returned no predictions.")
        index = int(np.argmax(scores))
        label = _load_labels().get(index, f"Class {index}")
        confidence = round(float(scores[index]) * 100, 1)
        return label, confidence
    except Exception as error:
        raise RuntimeError(str(error)) from error


def classify_with_openrouter(image):
    """Use an OpenRouter vision model when the local H5 model is unavailable."""
    keys = [
        os.getenv("OPENROUTER_API_KEY"),
        os.getenv("OPENROUTER_API_KEY_2"),
    ]
    keys = [key.strip() for key in keys if key and key.strip()]
    if not keys:
        raise RuntimeError("OpenRouter is not configured.")

    image.seek(0)
    encoded_image = base64.b64encode(image.read()).decode("ascii")
    mime_type = image.mimetype or "image/jpeg"
    model = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-001")
    prompt = (
        "Review this breast ultrasound or mammogram image for an educational prototype. "
        "Return only valid JSON with these keys: class_name (Benign, Malignant, or Normal), "
        "confidence (number from 0 to 100), explanation (one concise plain-language sentence), "
        "and next_step (one concise clinical safety recommendation). "
        "Do not claim a diagnosis; emphasize specialist review."
    )
    payload = json.dumps({
        "model": model,
        "temperature": 0.1,
        "max_tokens": 250,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}},
            ],
        }],
    }).encode("utf-8")

    last_error = None
    for key in keys:
        try:
            request = Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://127.0.0.1:5000",
                    "X-Title": "OncoVision",
                },
                method="POST",
            )
            with urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removesuffix("```").strip()
            result = json.loads(content)
            class_name = str(result.get("class_name", "")).title()
            if class_name not in {"Benign", "Malignant", "Normal"}:
                raise ValueError("The vision model returned an unsupported class.")
            confidence = round(max(0, min(100, float(result.get("confidence", 0)))), 1)
            return {
                "class_name": class_name,
                "confidence": confidence,
                "explanation": str(result.get("explanation", " specialist review is recommended.")),
                "next_step": str(result.get("next_step", "Arrange specialist review before making care decisions.")),
            }
        except (HTTPError, URLError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error

    raise RuntimeError(f"OpenRouter prediction failed: {last_error}")


def predict_stage(data):
    """Return the AJCC breast anatomic stage group from T, N and M."""
    t = (data.get("tumor") or "T2").upper()
    n = (data.get("node") or "N1").upper()
    m = (data.get("metastasis") or "M0").upper()
    grade = data.get("grade") or "Grade 2"
    try:
        age = int(data.get("age") or 58)
    except (TypeError, ValueError):
        age = 58

    valid_t, valid_n, valid_m = {"T1", "T2", "T3", "T4"}, {"N0", "N1", "N2", "N3"}, {"M0", "M1"}
    if t not in valid_t or n not in valid_n or m not in valid_m:
        raise ValueError("Unsupported TNM combination. Use T1-T4, N0-N3 and M0-M1.")

    if m == "M1":
        stage = "Stage IV"
    elif n == "N3":
        stage = "Stage IIIC"
    elif t == "T4":
        stage = "Stage IIIB" if n in {"N0", "N1", "N2"} else "Stage IIIC"
    elif (t, n) in {("T1", "N2"), ("T2", "N2"), ("T3", "N1"), ("T3", "N2")}:
        stage = "Stage IIIA"
    elif (t, n) in {("T2", "N1"), ("T3", "N0")}:
        stage = "Stage IIB"
    elif (t, n) in {("T1", "N1"), ("T2", "N0")}:
        stage = "Stage IIA"
    elif (t, n) == ("T1", "N0"):
        stage = "Stage IA"
    else:
        # Remaining supported M0 combinations are uncommon but map by
        # regional extent consistently with the AJCC anatomic grouping.
        stage = "Stage IIIA" if n == "N2" else "Stage IIB"

    summary = (
        f"AJCC anatomic TNM rules group {t} {n} {m} as {stage}. "
        "Grade and ER/PR/HER2 biomarkers provide prognostic and treatment context but do not change this anatomic stage group."
    )
    confidence = 100.0

    return {
        "stage": stage,
        "tumor": t,
        "node": n,
        "metastasis": m,
        "grade": grade,
        "confidence": confidence,
        "summary": summary,
        "display": f"{stage} ({t} {n} {m})",
    }


def explain_stage(data):
    result = predict_stage(data)
    biomarkers = [
        f"ER {data.get('er', 'Positive')}",
        f"PR {data.get('pr', 'Positive')}",
        f"HER2 {data.get('her2', 'Negative')}",
        f"Ki-67 {data.get('ki67', '20')}%",
    ]
    return (
        f"This assessment integrates TNM features ({result['tumor']} {result['node']} {result['metastasis']}) with "
        f"biomarker data ({', '.join(biomarkers)}) to support a high-level clinical summary. "
        f"The stage is {result['stage']}. {result['summary']} "
        "This prototype is intended for educational decision support and must be reviewed by the treating physician."
    )


def simulate_image_classification(filename):
    """Return demo classes for a mammogram classification workflow."""
    lower = (filename or "sample.png").lower()
    if "malig" in lower or "cancer" in lower:
        return "Malignant", 94.8
    if "benign" in lower:
        return "Benign", 91.6
    return "Benign", 88.7


def build_doctor_chatbot_response(message, context=None):
    """Local, clinician-register fallback used only in Doctor Mode."""
    context = context or {}
    normalized = (message or "").strip()
    lowered = normalized.lower()
    if not normalized:
        return "Provide the breast finding, imaging impression, TNM inputs, biomarkers, or review question to summarize."

    greetings = {"hi", "hello", "hey", "hii", "hiii", "good morning", "good evening"}
    if lowered in greetings:
        return ("Clinical Review Assistant ready. I can support breast-imaging interpretation context, differentials, "
                "TNM staging, biomarkers, model-output review, or clinical-note structure.")

    breast_terms = (
        "breast", "mammogram", "mammography", "nipple", "axilla", "axillary", "birads", "bi-rads",
        "mass", "lump", "calcification", "asymmetry", "distortion", "ductal", "lobular", "mastitis",
        "er", "pr", "her2", "ki-67", "ki67", "tnm", "stage", "tumor", "tumour", "oncology",
        "malignant", "benign", "biopsy", "grad-cam", "heatmap", "classifier", "probability",
        "mastectomy", "lumpectomy", "radiation", "systemic therapy", "pathology", "case", "clinical note",
    )
    off_topic_terms = (
        "kidney", "renal", "liver", "cardiac", "heart", "lung", "brain", "diabetes", "thyroid",
        "billing", "invoice", "cricket", "football", "weather",
    )
    if any(term in lowered for term in off_topic_terms) and not any(term in lowered for term in breast_terms):
        return ("This assistant is scoped to breast-cancer clinical support within OncoVision. "
                "Provide a related breast imaging, staging, biomarker, pathology, or management-context question.")
    if not any(term in lowered for term in breast_terms):
        return ("This assistant is scoped to breast-cancer clinical support. Please provide the relevant finding, "
                "imaging terminology, staging inputs, biomarkers, or model-review question.")

    if any(term in lowered for term in ("palpable mass", "breast mass", "new mass", "new lump")):
        return ("Differential considerations for a new palpable breast mass include:\n"
                "- Benign: simple/complicated cyst, fibroadenoma, fibrocystic change, fat necrosis, galactocele, or abscess depending on context.\n"
                "- Malignant: invasive carcinoma or other neoplasm, particularly with fixation, irregularity, skin/nipple change, or pathologic nodes.\n"
                "- Correlate age, duration, cycle/lactation/trauma history, examination, and targeted imaging. Imaging-pathology discordance requires resolution.")

    if "bi-rads" in lowered or "birads" in lowered:
        if "4" in lowered:
            return ("BI-RADS 4 indicates a suspicious abnormality for which tissue diagnosis is generally recommended.\n"
                    "- Confirm whether the report assigns 4A, 4B, or 4C and review the descriptor driving suspicion.\n"
                    "- Correlate mammographic, ultrasound, and prior-study findings.\n"
                    "- Plan image-guided sampling appropriate to the target and document radiologic-pathologic concordance.\n"
                    "Use the reporting radiologist's complete impression and current ACR guidance for case-specific management.")
        return ("BI-RADS standardizes breast-imaging assessment and management. Interpret the final category together "
                "with the descriptors, comparison studies, modality, and recommendation; an isolated category without the impression is incomplete context.")

    if any(term in lowered for term in ("er/pr", "er ", "pr ", "her2", "ki-67", "ki67", "biomarker")):
        return ("Biomarker pathway context:\n"
                "- ER/PR: hormone-receptor expression informs endocrine-sensitivity context; interpret with assay quality and complete pathology.\n"
                "- HER2: overexpression/amplification identifies a distinct biologic pathway and requires validated IHC/ISH interpretation.\n"
                "- Ki-67: proliferation marker with laboratory and interpretive variability; avoid using it as an isolated decision variable.\n"
                "Integrate biomarkers with histology, grade, anatomic/prognostic stage, comorbidity, and current multidisciplinary guidance.")

    if any(term in lowered for term in ("tnm", "ajcc", "stage", "t1", "t2", "t3", "t4", "n0", "n1", "n2", "n3", "m0", "m1")):
        return ("AJCC breast staging requires explicit T, N, and M definitions and distinction between clinical and pathologic staging. "
                "Prognostic stage also incorporates grade, ER, PR, and HER2. Provide the exact c/prefixes and inputs for a rule-based grouping; "
                "verify the result against the current AJCC staging manual because edition and special-case rules matter.")

    if any(term in lowered for term in ("grad-cam", "heatmap", "attention", "probability", "classifier", "model output")):
        return ("Model-output interpretation:\n"
                "- Probability/class confidence reflects the classifier output for its trained classes; it is not disease prevalence or a calibrated diagnostic probability unless validated as such.\n"
                "- Grad-CAM localizes features influencing the output, not a lesion boundary, measurement, or pathology confirmation.\n"
                "- Review acquisition quality, breast-region preprocessing, original views, prior imaging, and clinical/pathology correlation before using it for prioritization.")

    if any(term in lowered for term in ("treatment", "surgery", "radiation", "systemic", "therapy pathway")):
        return ("Treatment-pathway review should integrate resectability, stage, histology, biomarkers, nodal status, "
                "patient fitness/preferences, and multidisciplinary assessment. Modalities may include local surgery, radiation, "
                "and biomarker/stage-directed systemic categories; use current NCCN/local protocols for patient-specific sequencing and selection.")

    if any(term in lowered for term in ("clinical note", "structure note", "write note", "summary")):
        return ("Suggested concise structure:\n"
                "- Clinical indication and relevant history\n"
                "- Imaging/views and key finding descriptors\n"
                "- AI output and Grad-CAM context, explicitly labeled research support\n"
                "- Radiologic assessment/differential and concordance considerations\n"
                "- Recommended workup or multidisciplinary next step\n"
                "- Limitations and responsible clinician interpretation")

    if "pain" in lowered or "mastalgia" in lowered:
        return ("Breast-pain differential depends on focality, cyclicity, laterality, examination, age, and associated findings. "
                "Consider cyclical mastalgia, chest-wall/musculoskeletal pain, cystic change, trauma/fat necrosis, medication-related causes, "
                "and inflammatory/infective pathology; malignancy is less commonly represented by pain alone. Focal persistent non-cyclical pain "
                "or pain with a mass, skin/nipple change, discharge, or nodes warrants targeted clinical-imaging correlation.")

    report_id = context.get("report_id")
    if report_id:
        return (f"Latest queued case {report_id}: classifier result {context.get('class_name')}, malignancy probability "
                f"{context.get('malignancy_probability')}%, review status {context.get('review_status')}. "
                "Specify whether you need image-feature context, Grad-CAM limitations, staging correlation, or a structured review note.")
    return ("Provide the specific breast finding, BI-RADS impression, TNM/biomarker inputs, or model output. "
            "I can return a focused differential, staging context, interpretation framework, or note structure without assigning an unverified diagnosis.")


def build_chatbot_response(message, mode, context=None):
    context = context or {}
    if mode == "doctor":
        return build_doctor_chatbot_response(message, context)
    normalized = (message or "").strip()
    if not normalized:
        return "Please share a question about your imaging, symptoms, or treatment plan so I can help with a safer summary."

    lowered = normalized.lower()
    report_id = context.get("report_id")
    classification = context.get("class_name")
    probability = context.get("malignancy_probability")
    status = context.get("review_status")
    note = context.get("doctor_notes")
    greetings = {"hi", "hello", "hey", "hii", "hiii", "namaste", "salam", "good morning", "good evening"}
    if lowered in greetings:
        return ("Hello! I can help with breast health, breast symptoms, mammograms, screening, breast cancer, "
                "reports and safer next steps. What would you like to understand?")

    breast_terms = (
        "breast", "mammogram", "mammography", "nipple", "underarm", "axilla", "birads", "bi-rads",
        "mastectomy", "lumpectomy", "ductal", "lobular", "her2", "estrogen receptor", "progesterone receptor",
        "breastfeeding", "stan cancer", "ganth", "lump", "chest lump", "self exam", "self-exam",
    )
    related_terms = (
        "cancer", "tumor", "tumour", "malignant", "benign", "biopsy", "screening", "oncologist",
        "chemotherapy", "radiation", "hormone therapy", "stage", "grad-cam", "heatmap", "report", "result",
        "case", "review",
    )
    off_topic_terms = (
        "kidney", "renal", "liver", "hepatitis", "heart", "cardiac", "lung", "asthma", "brain", "migraine",
        "stomach", "gastric", "diabetes", "blood sugar", "thyroid", "skin rash", "knee", "dental", "tooth",
    )
    is_breast_topic = any(term in lowered for term in breast_terms) or any(term in lowered for term in related_terms)
    if any(term in lowered for term in off_topic_terms) and not any(term in lowered for term in breast_terms):
        return ("I’m the OncoVision Breast Health Assistant, so I can only answer questions about breast health, "
                "mammograms and breast cancer. Please ask a qualified clinician about that other health topic.")
    if not is_breast_topic:
        return ("I can help only with breast-related topics—for example breast changes, lumps, pain, nipple symptoms, "
                "mammograms, screening, breast cancer, reports or treatment questions. Please rephrase your question within that scope.")

    side = "left" if "left" in lowered else "right" if "right" in lowered else "affected"
    if ("nipple" in lowered or "nippl" in lowered) and any(term in lowered for term in ("invert", "retract", "turn in", "pulled in", "dislocat", "position", "direction", "flat", "sunken")):
        return (f"A nipple that has always pointed inward can be a normal variation. A new change in the {side} nipple—such as "
                "turning inward, flattening, changing direction or being pulled to one side—should be examined by a clinician, "
                "especially with a lump, discharge, crusting or skin dimpling. Do not try to force it back into position. "
                "Was this present since puberty, or is it a recent change?")
    if any(term in lowered for term in ("allerg", "itch", "itchy", "eczema", "rash", "irritat", "burning skin", "dry skin")):
        return (f"Itching or a rash on the {side} breast can come from soap, detergent, fabric, sweat, friction, eczema or infection. "
                "Avoid new fragranced products, keep the area dry, and do not apply steroid or antibiotic creams unless advised. "
                "Arrange assessment if it persists, spreads, blisters, involves only one nipple, causes crusting or discharge, or occurs with a lump. "
                "Seek prompt care for rapidly spreading redness, fever or severe pain.")
    if any(term in lowered for term in ("mastitis", "abscess", "pus", "breast infection")):
        return (f"A breast infection or abscess may cause a painful warm red area, swelling, fever or pus. It needs prompt clinical "
                "assessment because antibiotics or drainage may be required. Seek same-day care if you have fever, feel unwell, "
                "are breastfeeding, or the swelling is worsening. Do not puncture or squeeze the area yourself.")
    if any(term in lowered for term in ("injury", "injured", "hit my breast", "trauma", "bruise", "bruising")):
        return (f"After an injury, the {side} breast may develop bruising, tenderness, swelling or a firm area from healing fat tissue. "
                "Use a supportive bra and a wrapped cold pack for short periods if comfortable. Get urgent care for severe swelling, "
                "uncontrolled bleeding or breathing difficulty; arrange examination if a lump, pain or discoloration persists.")
    if any(term in lowered for term in ("pregnan", "breastfeed", "breast feed", "lactat", "milk supply", "blocked duct", "engorg")):
        return ("Pregnancy and breastfeeding commonly change breast size, tenderness, lumpiness and nipple appearance. A blocked duct "
                "or mastitis can cause a painful localized area; fever, spreading redness or feeling unwell needs prompt assessment. "
                "Continue feeding or expressing if your maternity clinician says it is safe, and contact an obstetric, lactation or breast-care professional for individualized advice.")
    if any(term in lowered for term in ("bigger breast", "breast is bigger", "smaller breast", "breast is smaller", "one breast", "uneven", "asymmetr", "different size", "breast size")):
        return ("Some difference in breast size is common and can change with puberty, menstrual cycles, pregnancy or weight. A new or "
                "rapid one-sided size or shape change—particularly with swelling, skin dimpling, redness, a lump or nipple change—should "
                "be examined. When did you first notice the asymmetry, and is it stable or increasing?")
    if any(term in lowered for term in ("male breast", "man breast", "boy breast", "gynecomastia", "gynaecomastia")):
        return ("Men can develop breast enlargement or tenderness from hormonal changes, medicines or other conditions. A firm one-sided "
                "lump, nipple discharge, inversion or skin change needs medical assessment because men can also develop breast cancer, "
                "although it is uncommon. A clinician can review medicines and decide whether imaging is needed.")
    if any(term in lowered for term in ("implant", "silicone", "augmentation", "implant rupture")):
        return ("Breast implants can be associated with firmness, pain, shape change, fluid or possible rupture. A new change should be "
                "reviewed by a breast or plastic-surgery clinician; the appropriate imaging depends on implant type and symptoms. "
                "Seek urgent care for marked redness, fever, rapidly increasing swelling or severe pain.")
    if any(term in lowered for term in ("cyst", "fibroadenoma", "fibrocystic")):
        return ("Breast cysts and fibroadenomas are common benign conditions, but a new lump cannot be reliably identified by touch alone. "
                "A clinician may use examination and ultrasound or mammography to characterize it. Return for review if a known lump grows, "
                "becomes persistently painful or develops associated skin or nipple changes.")
    if any(term in lowered for term in ("dense breast", "breast density", "dense tissue")):
        return ("Dense breasts contain more fibroglandular tissue, which can make mammograms harder to interpret and is also associated "
                "with some increase in breast-cancer risk. Density is determined from mammography, not by touch. Ask your clinician whether "
                "your density and personal risk justify any additional imaging under local guidance.")
    if "bi-rads" in lowered or "birads" in lowered:
        return ("BI-RADS is a standardized radiology assessment system. Categories range from incomplete assessment through benign findings "
                "to increasing levels of suspicion; category 6 refers to known biopsy-proven cancer. The exact next step depends on the category "
                "and report wording, so share the category or discuss it with the reporting radiologist.")
    if any(term in lowered for term in ("ultrasound", "sonography", "breast mri", "mri breast")):
        return ("Breast ultrasound is often used to evaluate a symptom or distinguish fluid-filled from solid findings. Breast MRI is used "
                "for selected high-risk screening or diagnostic situations and does not replace every mammogram. The appropriate test depends "
                "on age, symptoms, risk and prior imaging, so it should be selected by a clinician.")
    if any(term in lowered for term in ("period", "menstrual", "before periods", "cycle", "pms")) and any(term in lowered for term in ("pain", "tender", "lump", "swollen")):
        return ("Breast tenderness or lumpiness in both breasts can fluctuate with the menstrual cycle and often improves after a period. "
                "Track the timing for one or two cycles and use a supportive bra. A focal, one-sided or persistent change, or any lump, discharge, "
                "skin change or redness, should still be examined by a clinician.")
    if "nipple" in lowered and any(term in lowered for term in ("pain", "sore", "hurt", "tender")):
        return (f"Pain or tenderness around the {side} nipple can occur from friction, hormonal changes, infection, "
                "eczema or other non-cancerous causes. Check whether there is redness, cracking, a rash, discharge, "
                "a new lump or nipple inversion. Arrange a clinician visit if it is new, one-sided, persistent, worsening, "
                "or accompanied by any of those changes. How long has the pain been present, and is there discharge or a visible change?")
    if "discharge" in lowered or "fluid from nipple" in lowered or "nipple fluid" in lowered:
        return (f"Nipple discharge can have several causes. Discharge from the {side} nipple should be assessed sooner if "
                "it occurs without squeezing, is bloody or clear, comes from one opening, or is accompanied by a lump or skin change. "
                "Do not repeatedly squeeze the nipple. Please tell me its color, whether it happens by itself, and whether it is one-sided.")
    if any(term in lowered for term in ("lump", "ganth", "hard mass", "thickening")):
        return (f"A new {side}-breast or underarm lump needs a clinical examination, although many lumps are benign cysts or "
                "fibroadenomas. Note its location, how long it has been present, whether it changes with your menstrual cycle, "
                "and whether there are skin or nipple changes. Please arrange a breast-clinic or doctor appointment rather than trying to diagnose it by touch.")
    if any(term in lowered for term in ("redness", "red", "warm", "fever", "infection")):
        return (f"Redness or warmth of the {side} breast can be caused by infection or inflammation and needs prompt assessment, "
                "especially with fever, rapidly increasing swelling, severe pain or feeling unwell. Seek same-day medical care for "
                "those warning signs. If breastfeeding, mention that to the clinician because mastitis is one possible cause.")
    if any(term in lowered for term in ("dimpl", "orange peel", "peau d", "skin pulling", "skin change")):
        return (f"New dimpling, puckering, thickening or orange-peel-like skin on the {side} breast should be examined by a "
                "clinician. It can have different causes, but a persistent new skin change should not be watched indefinitely or assessed from a photo alone. "
                "Please arrange an appointment and mention when you first noticed it.")
    if "swelling" in lowered or "swollen" in lowered:
        return (f"New swelling of the {side} breast or underarm may come from infection, hormonal change, lymph-node enlargement "
                "or other causes. A clinician should assess persistent or one-sided swelling. Seek urgent care if it is rapidly worsening, "
                "very painful, red and hot, or associated with fever or breathing difficulty.")
    if ("breast" in lowered and any(term in lowered for term in ("pain", "painful", "hurt", "hurts", "tender", "tenderness", "sore"))):
        return (f"{side.title()}-breast pain is commonly related to hormonal changes, muscle or chest-wall strain, medication, "
                "an ill-fitting bra or benign breast conditions; pain alone is less commonly a cancer sign. Please seek assessment "
                "if it is focal, one-sided, persistent, worsening, or occurs with a lump, skin change, discharge, redness or fever. "
                "Is the pain in one exact spot, and how long has it lasted?")
    if any(term in lowered for term in ("unusual change", "symptom", "sign", "change in my breast")):
        return ("A new breast or underarm lump, skin dimpling, shape change, nipple inversion or unusual discharge, "
                "persistent one-sided pain, redness or swelling should be assessed by a clinician. Many breast changes "
                "are not cancer, but a new or persistent change should not be self-diagnosed. Arrange a breast-clinic or "
                "doctor appointment. Tell me the exact change—lump, pain, discharge, skin change, redness or swelling—so I can give more specific guidance.")
    if any(term in lowered for term in ("what is breast cancer", "breast cancer kya", "define breast cancer")):
        return ("Breast cancer develops when abnormal cells in breast tissue grow uncontrollably. Different types behave "
                "differently, so diagnosis uses clinical examination, imaging and usually tissue testing. Early evaluation "
                "can widen treatment options, but only a qualified clinician can diagnose it.")
    if any(term in lowered for term in ("screen", "mammogram age", "how often", "frequency")):
        return ("Mammography can find breast changes before symptoms appear. The appropriate starting age and frequency "
                "depend on your country’s guidance and personal risk, including age, family history and genetic factors. "
                "Ask your clinician for an individual screening schedule; a new symptom needs assessment rather than waiting for routine screening.")
    if any(term in lowered for term in ("self exam", "self-exam", "check breast", "examine")):
        return ("Breast self-awareness means knowing your usual appearance and feel and reporting a new persistent change. "
                "A self-check cannot diagnose or rule out cancer and does not replace recommended screening. You can open "
                "the Self-Examination Guide in the menu for a careful step-by-step overview.")
    if any(term in lowered for term in ("risk", "family history", "genetic", "brca", "prevent")):
        return ("Breast-cancer risk can be influenced by age, close family history, inherited variants such as BRCA1/2, "
                "prior chest radiation, reproductive or hormonal factors, alcohol and weight after menopause. Having a risk "
                "factor does not mean cancer is certain. Discuss strong family history with a clinician or genetic counselor.")
    if any(term in lowered for term in ("biopsy", "tissue test")):
        return ("A breast biopsy removes a small tissue sample so a pathologist can examine the cells. It may be recommended "
                "after an examination or imaging finding, but needing a biopsy does not automatically mean cancer. Your care "
                "team should explain the biopsy type, preparation, possible discomfort and when results are expected.")
    if any(term in lowered for term in ("treatment", "chemo", "radiation", "surgery", "mastectomy", "lumpectomy", "hormone therapy")):
        return ("Breast-cancer treatment may include breast-conserving surgery or mastectomy, radiation, chemotherapy, "
                "hormone therapy, HER2-targeted treatment or other medicines. The plan depends on pathology, stage, biomarkers, "
                "overall health and patient preferences, so it must be decided with the oncology team.")
    if any(term in lowered for term in ("stage", "tnm")):
        return ("Breast-cancer stage describes extent of disease using tumor size, lymph nodes and distant spread, together "
                "with pathology and biomarkers. Stage cannot be safely determined from a photo or symptom alone; it requires "
                "the treating team’s examination, imaging and pathology information.")
    if any(term in lowered for term in ("benign", "malignant")) and "report" not in lowered:
        return ("Benign means non-cancerous; malignant means cancerous. Imaging can estimate suspicion, but those words should "
                "not be assigned from an AI screenshot alone. A radiologist interprets the original study and pathology may be needed for confirmation.")

    if not report_id and any(term in lowered for term in ("result", "report", "heat", "grad", "spot")):
        return ("No patient mammogram is available in the review queue yet." if mode == "doctor" else
                "You do not have a saved mammogram result yet. Upload a single-view mammogram first, then I can explain its report.")
    if mode == "doctor":
        if any(word in lowered for word in ("latest", "summary", "result", "case", "review")):
            return (f"Latest case {report_id}: model classification {classification}, malignancy probability "
                    f"{probability}%, review status {status}. Grad-CAM is model attention only and should be "
                    "compared with the original mammogram before recording your interpretation.")
        if "heat" in lowered or "grad" in lowered or "spot" in lowered:
            return "The Grad-CAM overlay indicates pixels that most influenced this EfficientNet output; it is not a lesion contour, measurement, or pathology confirmation."
        return f"Case {report_id} is {status}. I can summarize its model result, explain Grad-CAM limitations, or help structure a concise review note."
    if any(word in lowered for word in ("result", "report", "mean", "explain", "latest")):
        answer = (f"Your latest report {report_id} was classified {classification} by the research model, with a "
                  f"{probability}% estimated malignancy probability. Its clinician review status is {status}.")
        if note:
            answer += f" The clinician note says: {note}"
        return answer + " This does not confirm or exclude cancer."
    if "heat" in lowered or "grad" in lowered or "spot" in lowered:
        return "The colored Grad-CAM image shows where the model focused. It is not a thermal scan and does not prove that the highlighted area is cancer."
    if "next" in lowered or "doctor" in lowered:
        return f"Your report is currently {status}. Please arrange review by a radiologist or breast specialist and follow their advice about additional imaging or pathology."
    return ("I can provide general breast-health education, but I may need a more specific question. You can ask about breast "
            "symptoms, mammogram screening, risk factors, biopsy, cancer stages, treatment, or your saved report. I cannot diagnose disease.")
