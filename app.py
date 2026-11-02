from functools import wraps
import base64
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from io import BytesIO
import os
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, abort
from utils.clinical import predict_stage, explain_stage, simulate_image_classification
from utils.mammogram import analyze_mammogram, extract_mammogram_views, model_available, model_metadata
from werkzeug.datastructures import FileStorage
from utils.storage import (UPLOAD_DIR, create_analysis, create_clinical_prediction, dashboard_stats, get_analysis,
                           get_clinical_prediction,
                           latest_patient_analysis, list_all_analyses,
                           list_clinical_predictions, list_latest_patient_analyses,
                           list_patient_analyses, review_analysis)
from utils.report_explainer import extract_report_text, explain_report
from oncovision_ai_assistant_route import ai_assistant
from utils.oncologist_finder import oncologist_finder
from utils.access_control import (
    access_control, active_patient_ids, init_access_control_db,
    requires_patient_access,
)
from utils.auth_storage import authenticate_user, register_user

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("ONCOVISION_SECRET_KEY", "change-this-key-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
app.register_blueprint(ai_assistant)
app.register_blueprint(oncologist_finder)
app.register_blueprint(access_control)
init_access_control_db()


@app.after_request
def prevent_stale_html(response):
    """Keep navigation and authenticated pages from remaining stale in browsers."""
    if response.content_type and response.content_type.startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def require_mode(mode):
    """Restrict a route to the authenticated patient or doctor experience."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = session.get("user")
            if not user:
                return redirect(url_for("login", mode=mode, next=request.path))
            if user.get("mode") != mode:
                destination = "doctor_dashboard" if user.get("mode") == "doctor" else "patient_dashboard"
                return redirect(url_for(destination))
            return view(*args, **kwargs)
        return wrapped
    return decorator

PORTAL_PAGES = {
    "analysis-history": ("Analysis History", "Review previous mammogram analyses and their available results."),
    "ai-explanations": ("AI Explanations", "Review plain-language explanations linked to available analysis context."),
    "patient-records": ("Patient Records", "Browse patient records available in this prototype session."),
    "prediction-history": ("Prediction History", "Review clinical stage predictions saved during this session."),
    "reports": ("Analytics & Reports", "View a consolidated summary of image analyses and clinical predictions."),
    "profile": ("Profile", "Update the profile information shown throughout OncoVision."),
    "help": ("Help & Support", "Find guidance for screening, clinical assessment, and account access."),
    "resources": ("Clinical Resources", "Reference educational resources and clinical workflow guidance."),
    "health-knowledge": ("Health Knowledge", "Read educational breast-health information and screening guidance."),
    "forgot-password": ("Reset Password", "Enter your account email to request a password-reset link."),
    "create-account": ("Create Account", "Create a prototype patient or doctor account."),
    "about": ("About OncoVision", "OncoVision is an educational AI-assisted decision-support prototype."),
    "privacy": ("Privacy Policy", "Review how this prototype handles submitted information."),
    "terms": ("Terms & Conditions", "Review the terms governing use of this educational prototype."),
    "contact": ("Contact Us", "Send a support request to the OncoVision team."),
}


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/api/health")
def health_api():
    return jsonify({
        "status": "ok",
        "mammogram_model": "ready" if model_available() else "training-or-missing",
        "mammogram_metrics": model_metadata(),
    })


@app.get("/login")
def login():
    user = session.get("user")
    if user:
        destination = "doctor_dashboard" if user.get("mode") == "doctor" else "patient_dashboard"
        return redirect(url_for(destination))
    return render_template("login.html", account_created=request.args.get("created") == "true")


@app.post("/login")
def login_post():
    email = request.form.get("email") or request.form.get("username")
    mode = request.form.get("mode") or "patient"
    identity = (email or "Demo User").strip()
    registered = authenticate_user(identity, request.form.get("password", ""), mode)
    if not registered:
        session.pop("user", None)
        return render_template("login.html", login_error="Invalid email or password"), 401
    session["user"] = registered
    if mode == "doctor":
        return redirect(url_for("doctor_dashboard"))
    return redirect(url_for("patient_dashboard"))


@app.post("/api/auth/register")
def register_api():
    payload = request.get_json(silent=True) or {}
    account, errors = register_user(
        payload.get("name"), payload.get("email"), payload.get("password"), payload.get("role")
    )
    if errors:
        return jsonify({"error": next(iter(errors.values())), "errors": errors}), 409 if "email" in errors and "exists" in errors["email"] else 400
    return jsonify({"message": "Account created — please log in", "account": account}), 201


@app.get("/patient/dashboard")
@require_mode("patient")
def patient_dashboard():
    identity = session["user"]["identity"]
    records = list_patient_analyses(identity)
    benign = sum(record["class_name"] == "Benign" for record in records)
    malignant = sum(record["class_name"] == "Malignant" for record in records)
    total = len(records)
    dashboard_records = []
    for record in records:
        item = dict(record)
        try:
            saved_at = datetime.fromisoformat(item["created_at"])
            if saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
            saved_at = saved_at.astimezone(ZoneInfo("Asia/Kolkata"))
            item["date_display"] = saved_at.strftime("%d %b %Y")
            item["time_display"] = saved_at.strftime("%I:%M %p")
        except (TypeError, ValueError):
            item["date_display"] = str(item.get("created_at") or "")[:10]
            item["time_display"] = ""
        dashboard_records.append(item)
    return render_template(
        "patient_dashboard.html",
        recent_analyses=dashboard_records,
        patient_stats={"total": total, "benign": benign, "malignant": malignant},
        overview={
            "benign_percent": round(benign * 100 / total) if total else 0,
            "malignant_percent": round(malignant * 100 / total) if total else 0,
        },
    )


@app.get("/doctor/dashboard")
@require_mode("doctor")
def doctor_dashboard():
    allowed_patients = active_patient_ids(session["user"]["identity"])
    image_reviews = list_all_analyses(allowed_patients)
    return render_template(
        "doctor_dashboard.html",
        image_reviews=image_reviews,
        pending_reviews=sum(record["review_status"] == "Pending review" for record in image_reviews),
    )


@app.get("/api/dashboard/stats")
@require_mode("doctor")
def doctor_dashboard_stats_api():
    allowed_patients = active_patient_ids(session["user"]["identity"])
    return jsonify({
        "status": "ok",
        **dashboard_stats(len(allowed_patients), allowed_patients),
    })


def _doctor_report_data(doctor_identity):
    """Build one newest-first report feed from persisted image and TNM records."""
    analyses = list_all_analyses(active_patient_ids(doctor_identity))
    predictions = list_clinical_predictions(doctor_identity)
    reports = [
        {
            "id": row["id"], "patient": row["patient_name"],
            "report_type": "Mammogram Analysis", "kind": "mammogram",
            "date_generated": row["created_at"], "reference": row["report_id"],
            "result": row["class_name"],
            "view_url": url_for("doctor_report_view", report_kind="mammogram", report_id=row["id"]),
        }
        for row in analyses
    ]
    reports.extend(
        {
            "id": row["id"], "patient": row["patient_name"],
            "report_type": "Clinical Stage Prediction", "kind": "prediction",
            "date_generated": row["created_at"], "reference": row["patient_id"],
            "result": row["predicted_stage"],
            "view_url": url_for("doctor_report_view", report_kind="prediction", report_id=row["id"]),
        }
        for row in predictions
    )
    reports.sort(key=lambda item: item["date_generated"], reverse=True)
    benign = sum(row["class_name"] == "Benign" for row in analyses)
    malignant = sum(row["class_name"] == "Malignant" for row in analyses)
    return {
        "image_analyses": len(analyses),
        "saved_predictions": len(predictions),
        "reports_ready": len(reports),
        "breakdown": {"benign": benign, "malignant": malignant},
        "reports": reports,
    }


@app.get("/api/reports")
@require_mode("doctor")
def reports_api():
    return jsonify({"status": "ok", **_doctor_report_data(session["user"]["identity"])})


@app.get("/api/patients/<path:patient_identity>/ai-summary")
@require_mode("doctor")
@requires_patient_access(lambda _request, patient_identity: patient_identity)
def patient_ai_summary_api(patient_identity):
    record = latest_patient_analysis(patient_identity)
    if not record:
        return jsonify({"status": "not_found", "error": "No mammogram analysis was found for this patient."}), 404
    return jsonify({
        "status": "ok", "patient": record["patient_name"], "result": record["class_name"],
        "probability": record["malignancy_probability"], "confidence": record["confidence"],
        "model_notes": record["explanation"], "review_status": record["review_status"],
        "report_id": record["report_id"],
    })


@app.get("/doctor/clinical-assessment")
@require_mode("doctor")
def clinical_assessment():
    return render_template("doctor_clinical_assessment.html")


@app.get("/patient/find-oncologist")
@require_mode("patient")
def find_oncologist():
    return render_template("find_oncologist.html")


@app.get("/patient/upload-predict")
@require_mode("patient")
def upload_predict():
    return render_template(
        "upload_predict.html",
        mammogram_model_available=model_available(),
        mammogram_model_metadata=model_metadata(),
    )


@app.get("/patient/self-examination")
@require_mode("patient")
def self_examination():
    return render_template("self_examination.html")


@app.get("/patient/report-explainer")
@require_mode("patient")
def report_explainer():
    return render_template("report_explainer.html")


@app.post("/api/report-explain")
@require_mode("patient")
def report_explain_api():
    pasted_text = request.form.get("report_text", "").strip()
    upload = request.files.get("report")
    try:
        text = pasted_text or (extract_report_text(upload) if upload and upload.filename else "")
        if not text:
            raise ValueError("Upload a report or paste the report text first.")
        result = explain_report(text)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    result["filename"] = upload.filename if upload and upload.filename else "Pasted report text"
    return jsonify(result)


@app.post("/api/stage-predict")
@require_mode("doctor")
def stage_predict_api():
    payload = request.get_json(silent=True) or {}
    try:
        prediction = predict_stage(payload)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(prediction)


@app.post("/api/analysis")
@require_mode("patient")
def image_analysis_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files["image"]
    if not image.filename:
        return jsonify({"error": "Please choose an image file"}), 400
    if image.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        return jsonify({"error": "Please upload a JPG, PNG, or WebP image."}), 400
    try:
        result = analyze_mammogram(image)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    result["model_source"] = "local-cbis-ddsm"
    result["explanation"] = (
        f"The local research model estimated a {result['malignancy_probability']}% malignancy probability "
        f"and classified this image as {result['class_name']}. The color overlay is Grad-CAM model attention, "
        "not a confirmed lesion boundary or diagnosis."
    )
    result["next_step"] = (
        "A radiologist or breast specialist must interpret the original mammogram with clinical history and, "
        "when appropriate, additional imaging or pathology."
    )
    user = session["user"]
    token = uuid4().hex
    extension = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[image.mimetype]
    original_name = f"{token}-original{extension}"
    processed_name = f"{token}-processed.png"
    heatmap_name = f"{token}-gradcam.png"
    image.seek(0)
    (UPLOAD_DIR / original_name).write_bytes(image.read())
    processed_encoded = result["processed_image_data_url"].split(",", 1)[1]
    (UPLOAD_DIR / processed_name).write_bytes(base64.b64decode(processed_encoded))
    heatmap_encoded = result["heatmap_data_url"].split(",", 1)[1]
    (UPLOAD_DIR / heatmap_name).write_bytes(base64.b64decode(heatmap_encoded))
    report_id = f"OV-{token[:10].upper()}"
    analysis_id = create_analysis({
        "report_id": report_id,
        "patient_identity": user["identity"],
        "patient_name": user["name"],
        "original_filename": image.filename,
        "original_file": original_name,
        "processed_file": processed_name,
        "heatmap_file": heatmap_name,
        **result,
    })
    result["analysis_id"] = analysis_id
    result["report_id"] = report_id
    result["review_status"] = "Pending review"
    history = session.setdefault("analysis_history", [])
    history.append({key: result[key] for key in ("class_name", "confidence", "malignancy_probability", "model_name")})
    session["analysis_history"] = history[-20:]
    return jsonify(result)


@app.post("/api/analyses")
@require_mode("patient")
def multi_image_analysis_api():
    uploads = request.files.getlist("images")
    if not uploads:
        return jsonify({"error": "No mammogram images uploaded"}), 400
    if len(uploads) > 4:
        return jsonify({"error": "Upload a maximum of 4 image files at once."}), 400
    expanded = []
    for upload in uploads:
        if upload.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return jsonify({"error": f"{upload.filename}: upload a JPG, PNG, or WebP image."}), 400
        raw = upload.read()
        if len(raw) > 20 * 1024 * 1024:
            return jsonify({"error": f"{upload.filename}: image is larger than 20 MB."}), 400
        try:
            expanded.extend(extract_mammogram_views(raw, upload.filename))
        except ValueError as error:
            return jsonify({"error": f"{upload.filename}: {error}"}), 400
    if len(expanded) > 8:
        return jsonify({"error": "The selected files contain more than 8 mammogram views. Upload a smaller study."}), 400

    user = session["user"]
    results, failures = [], []
    for filename, raw in expanded:
        image = FileStorage(stream=BytesIO(raw), filename=filename, content_type="image/jpeg")
        try:
            result = analyze_mammogram(image)
        except (ValueError, RuntimeError) as error:
            failures.append({"filename": filename, "error": str(error)})
            continue
        result["model_source"] = "local-cbis-ddsm"
        result["explanation"] = (
            f"The local research model estimated a {result['malignancy_probability']}% malignancy probability "
            f"and classified this view as {result['class_name']}. The color overlay is Grad-CAM model attention, "
            "not a confirmed lesion boundary or diagnosis."
        )
        result["next_step"] = (
            "A radiologist or breast specialist must interpret every original view together with clinical history "
            "and, when appropriate, additional imaging or pathology."
        )
        token = uuid4().hex
        original_name, processed_name, heatmap_name = f"{token}-original.jpg", f"{token}-processed.png", f"{token}-gradcam.png"
        (UPLOAD_DIR / original_name).write_bytes(raw)
        (UPLOAD_DIR / processed_name).write_bytes(base64.b64decode(result["processed_image_data_url"].split(",", 1)[1]))
        (UPLOAD_DIR / heatmap_name).write_bytes(base64.b64decode(result["heatmap_data_url"].split(",", 1)[1]))
        report_id = f"OV-{token[:10].upper()}"
        analysis_id = create_analysis({
            "report_id": report_id, "patient_identity": user["identity"], "patient_name": user["name"],
            "original_filename": filename, "original_file": original_name, "processed_file": processed_name,
            "heatmap_file": heatmap_name, **result,
        })
        result.update({"analysis_id": analysis_id, "report_id": report_id, "review_status": "Pending review", "filename": filename})
        results.append(result)
    if not results:
        return jsonify({"error": "None of the extracted mammogram views could be analysed.", "failures": failures}), 400
    return jsonify({"results": results, "failures": failures, "source_files": len(uploads), "extracted_views": len(expanded)})


@app.get("/chat")
def chat():
    if not session.get("user"):
        return redirect(url_for("login", next=request.path))
    return render_template("chatbot.html", mode=session["user"]["mode"])


@app.get("/patient/results")
@require_mode("patient")
def patient_results():
    records = list_patient_analyses(session["user"]["identity"])
    return render_template("patient_results.html", records=records)


@app.route("/patient/profile", methods=["GET", "POST"])
@require_mode("patient")
def patient_profile():
    user = session["user"]
    saved = False
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if not name or not email:
            return render_template("patient_profile.html", profile=user, records=list_patient_analyses(user["identity"]), error="Name and email are required."), 400
        user.update({
            "name": name,
            "email": email,
            "phone": request.form.get("phone", "").strip(),
            "date_of_birth": request.form.get("date_of_birth", "").strip(),
            "blood_group": request.form.get("blood_group", "").strip(),
            "emergency_contact": request.form.get("emergency_contact", "").strip(),
            "address": request.form.get("address", "").strip(),
        })
        session["user"] = user
        session.modified = True
        saved = True
        # Keep the login identity stable so an email edit does not disconnect
        # the patient from reports saved under the original login.
        records = list_patient_analyses(user["identity"])
    else:
        records = list_patient_analyses(user["identity"])
    return render_template("patient_profile.html", profile=user, records=records, saved=saved)


@app.route("/doctor/profile", methods=["GET", "POST"])
@require_mode("doctor")
def doctor_profile():
    """Show and update the currently authenticated clinician's profile."""
    user = session["user"]
    saved = False
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if not name or not email:
            error = "Name and email are required."
        else:
            user.update({
                "name": name,
                "email": email,
                "phone": request.form.get("phone", "").strip(),
                "specialization": request.form.get("specialization", "").strip(),
                "registration_number": request.form.get("registration_number", "").strip(),
                "hospital": request.form.get("hospital", "").strip(),
                "experience": request.form.get("experience", "").strip(),
                "address": request.form.get("address", "").strip(),
            })
            session["user"] = user
            session.modified = True
            saved = True
    allowed_patients = active_patient_ids(user["identity"])
    stats = dashboard_stats(len(allowed_patients), allowed_patients)
    status = 400 if error else 200
    return render_template(
        "doctor_profile.html", profile=user, stats=stats, saved=saved, error=error,
        active_page="profile", page_title="Doctor Profile",
    ), status


@app.get("/doctor/image-reviews")
@require_mode("doctor")
def doctor_image_reviews():
    allowed_patients = active_patient_ids(session["user"]["identity"])
    return render_template(
        "doctor_image_reviews.html",
        records=list_latest_patient_analyses(allowed_patients),
    )


def _patient_for_analysis(_request, analysis_id, *_args, **_kwargs):
    record = get_analysis(analysis_id)
    return record["patient_identity"] if record else None


def _patient_for_report(_request, report_kind, report_id, **_kwargs):
    if report_kind != "mammogram":
        return None
    record = get_analysis(report_id)
    return record["patient_identity"] if record else None


@app.get("/api/doctor/mammogram-reviews/<int:analysis_id>")
@require_mode("doctor")
@requires_patient_access(_patient_for_analysis)
def doctor_mammogram_review_detail_api(analysis_id):
    record = get_analysis(analysis_id)
    if not record:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify({
        "id": record["id"], "report_id": record["report_id"],
        "patient_identity": record["patient_identity"],
        "patient_name": record["patient_name"], "original_filename": record["original_filename"],
        "class_name": record["class_name"], "confidence": record["confidence"],
        "malignancy_probability": record["malignancy_probability"],
        "explanation": record["explanation"], "review_status": record["review_status"],
        "doctor_notes": record["doctor_notes"] or "", "created_at": record["created_at"],
        "processed_url": url_for("analysis_file", analysis_id=record["id"], kind="processed"),
        "heatmap_url": url_for("analysis_file", analysis_id=record["id"], kind="heatmap"),
    })


@app.post("/api/doctor/image-reviews/<int:analysis_id>")
@require_mode("doctor")
@requires_patient_access(_patient_for_analysis)
def doctor_review_api(analysis_id):
    if not get_analysis(analysis_id):
        return jsonify({"error": "Analysis not found"}), 404
    payload = request.get_json(silent=True) or {}
    status = payload.get("status", "Reviewed")
    if status not in {"Reviewed", "Follow-up required", "Pending review"}:
        return jsonify({"error": "Unsupported review status"}), 400
    record = review_analysis(analysis_id, session["user"]["name"], status, str(payload.get("notes", "")).strip())
    return jsonify({"message": "Doctor review saved", "record": dict(record)})


@app.get("/analysis-file/<int:analysis_id>/<kind>")
@requires_patient_access(_patient_for_analysis)
def analysis_file(analysis_id, kind):
    user = session.get("user")
    record = get_analysis(analysis_id)
    if not user or not record or kind not in {"original", "processed", "heatmap"}:
        abort(404)
    if user["mode"] == "patient" and record["patient_identity"] != user["identity"]:
        abort(403)
    if kind == "heatmap":
        filename = record["heatmap_file"]
    elif kind == "processed" and record["processed_file"]:
        filename = record["processed_file"]
    else:
        filename = record["original_file"]
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/doctor/reports/<report_kind>/<int:report_id>")
@require_mode("doctor")
@requires_patient_access(_patient_for_report)
def doctor_report_view(report_kind, report_id):
    if report_kind == "mammogram":
        record = get_analysis(report_id)
    elif report_kind == "prediction":
        record = get_clinical_prediction(report_id, session["user"]["identity"])
    else:
        abort(404)
    if not record:
        abort(404)
    return render_template(
        "doctor_report_view.html", report_kind=report_kind, record=record,
        active_page="reports", page_title="Report Details",
    )


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/portal/<page>")
def portal(page):
    if page not in PORTAL_PAGES:
        return render_template("portal_page.html", title="Page Not Found", description="The requested OncoVision page is unavailable.", page=page), 404
    if page == "create-account":
        return render_template("create_account.html")
    user = session.get("user")
    public_pages = {"forgot-password", "create-account", "about", "privacy", "terms", "contact"}
    doctor_pages = {"patient-records", "prediction-history", "reports", "resources"}
    patient_pages = {"analysis-history", "ai-explanations", "health-knowledge"}
    if page not in public_pages and not user:
        return redirect(url_for("login", next=request.path))
    if user and page in doctor_pages and user.get("mode") != "doctor":
        return redirect(url_for("patient_dashboard"))
    if user and page in patient_pages and user.get("mode") != "patient":
        return redirect(url_for("doctor_dashboard"))
    title, description = PORTAL_PAGES[page]
    records = list_clinical_predictions(user["identity"]) if user and user.get("mode") == "doctor" and page in {"prediction-history", "reports"} else []
    return render_template(
        "portal_page.html", title=title, description=description, page=page,
        records=records, mode=(user or {}).get("mode"), active_page=page,
        page_title=title,
    )


@app.post("/api/clinical-records")
@require_mode("doctor")
def save_clinical_record():
    payload = request.get_json(silent=True) or {}
    required = ("tumor", "node", "metastasis", "grade", "er", "pr", "her2")
    if any(not payload.get(field) for field in required):
        return jsonify({"error": "Missing required clinical fields"}), 400
    patient_name = str(payload.get("patient_name") or "").strip()
    patient_id = str(payload.get("patient_id") or "").strip()
    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400
    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400
    try:
        age = int(payload.get("age"))
        ki67 = float(payload.get("ki67"))
    except (TypeError, ValueError):
        return jsonify({"error": "Patient age and Ki-67 must be valid numbers"}), 400
    if not 1 <= age <= 130:
        return jsonify({"error": "Patient age must be between 1 and 130"}), 400
    if not 0 <= ki67 <= 100:
        return jsonify({"error": "Ki-67 must be between 0 and 100"}), 400
    prediction = predict_stage(payload)
    values = {
        **payload, **prediction,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "age": age,
        "ki67": ki67,
    }
    record = create_clinical_prediction(session["user"]["identity"], values)
    saved = dict(record)
    saved["stage"] = saved["predicted_stage"]
    saved["display"] = f'{saved["predicted_stage"]} ({saved["tumor"]} {saved["node"]} {saved["metastasis"]})'
    return jsonify({"message": "Record saved to patient history", "record": saved}), 201


@app.route("/api/predictions", methods=["GET", "POST"])
@require_mode("doctor")
def predictions_api():
    if request.method == "GET":
        return jsonify({"status": "ok", "predictions": [dict(row) for row in list_clinical_predictions(session["user"]["identity"])]})
    return save_clinical_record()


@app.get("/api/predictions/<int:prediction_id>")
@require_mode("doctor")
def prediction_detail_api(prediction_id):
    record = get_clinical_prediction(prediction_id, session["user"]["identity"])
    if not record:
        return jsonify({"error": "Prediction not found"}), 404
    return jsonify({"status": "ok", "prediction": dict(record)})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("ONCOVISION_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.getenv("ONCOVISION_PORT", "5000")),
    )
