import pytest
import numpy as np
import cv2
import json
import sqlite3
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from app import app
from werkzeug.security import check_password_hash
from utils.clinical import predict_stage, explain_stage, simulate_image_classification
from utils.report_explainer import explain_report
from utils.mammogram import _attention_peak, extract_mammogram_views
from utils.oncologist_finder import haversine_km
from utils.storage import list_all_analyses
import utils.oncologist_finder as finder_module
import oncovision_ai_assistant_route as assistant_module
import utils.storage as storage_module


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ONCOVISION_USERS_FILE", str(tmp_path / "test-users.json"))
    monkeypatch.setenv("ONCOVISION_ACCESS_DB", str(tmp_path / "test-access.sqlite3"))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def login_as(client, mode):
    credentials = {"name": f"Test {mode.title()}", "email": f"{mode}@example.com", "password": "testpassword8", "role": mode}
    client.post("/api/auth/register", json=credentials)
    return client.post("/login", data={"username": credentials["email"], "password": credentials["password"], "mode": mode})


def test_temporary_patient_access_code_scopes_and_expires(client, monkeypatch, tmp_path):
    analyses = list_all_analyses()
    if not analyses:
        pytest.skip("No persisted mammogram fixture is available")
    record = analyses[0]
    access_db = tmp_path / "test-access.sqlite3"

    with client.session_transaction() as current_session:
        current_session["user"] = {
            "name": record["patient_name"], "identity": record["patient_identity"], "mode": "patient"
        }
    profile = client.get("/patient/profile")
    assert b"Share My Data With a Doctor" in profile.data
    generated = client.post("/api/access/generate-code")
    assert generated.status_code == 200
    code = generated.get_json()["code"]
    assert len(code) == 6 and code.isdigit()

    with client.session_transaction() as current_session:
        current_session["user"] = {"name": "Access Doctor", "identity": "access-doctor@example.com", "mode": "doctor"}
    denied = client.get(f'/api/doctor/mammogram-reviews/{record["id"]}')
    assert denied.status_code == 403
    assert denied.get_json()["status"] == "access_denied"
    assert record["patient_name"].encode() not in client.get("/doctor/image-reviews").data

    claimed = client.post("/api/access/claim-code", json={"code": code})
    assert claimed.status_code == 200
    assert claimed.get_json()["patient_id"] == record["patient_identity"]
    active = client.get("/api/access/my-active-patients").get_json()["active_patients"]
    assert record["patient_identity"] in {item["patient_id"] for item in active}
    assert client.get(f'/api/doctor/mammogram-reviews/{record["id"]}').status_code == 200
    assert record["patient_name"].encode() in client.get("/doctor/image-reviews").data

    with sqlite3.connect(access_db) as db:
        db.execute(
            "UPDATE access_grants SET expires_at = ? WHERE code = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), code),
        )
    expired = client.get(f'/api/doctor/mammogram-reviews/{record["id"]}')
    assert expired.status_code == 403
    assert "expired" in expired.get_json()["error"].lower()


def test_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"OncoVision" in response.data
    hero = client.get("/static/images/oncovision-hero-v2.png")
    assert hero.status_code == 200
    assert len(hero.data) > 100_000


def test_patient_dashboard_hero_has_single_wide_patient_card(client):
    login_as(client, "patient")
    page = client.get("/patient/dashboard")
    assert b'class="mode-cards patient-only-card"' in page.data
    stylesheet = client.get("/static/css/styles.css")
    assert b".welcome-ref .patient-only-card" in stylesheet.data
    assert b"grid-template-columns:minmax(320px,400px)" in stylesheet.data
    assert b"white-space:nowrap" in stylesheet.data


def test_create_account_hashes_password_and_redirects_to_login(client, monkeypatch, tmp_path):
    users_file = tmp_path / "users.json"
    monkeypatch.setenv("ONCOVISION_USERS_FILE", str(users_file))
    page = client.get("/portal/create-account")
    assert page.status_code == 200
    assert b'name="name" type="text"' in page.data
    assert b'name="email" type="email"' in page.data
    assert b'name="password" type="password"' in page.data
    assert b'minlength="8"' in page.data
    assert b'name="role" value="patient"' in page.data
    assert b'class="create-account-layout"' in page.data
    assert b"Your breast-health journey, securely connected." in page.data
    assert b"window.location.assign('/login?created=true')" in page.data

    payload = {"name": "Radhika Naik", "email": "radhika@example.com", "password": "securepass8", "role": "patient"}
    created = client.post("/api/auth/register", json=payload)
    assert created.status_code == 201
    saved = json.loads(users_file.read_text(encoding="utf-8"))["radhika@example.com"]
    assert "password" not in saved
    assert saved["password_hash"] != payload["password"]
    assert check_password_hash(saved["password_hash"], payload["password"])
    assert client.post("/api/auth/register", json=payload).status_code == 409

    login = client.post("/login", data={"username": payload["email"], "password": payload["password"], "mode": "patient"})
    assert login.status_code == 302
    assert login.headers["Location"].endswith("/patient/dashboard")
    client.get("/logout")
    bad_login = client.post("/login", data={"username": payload["email"], "password": "wrong-password", "mode": "patient"})
    assert bad_login.status_code == 401


def test_create_account_rejects_invalid_email_and_short_password(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ONCOVISION_USERS_FILE", str(tmp_path / "users.json"))
    response = client.post("/api/auth/register", json={"name": "A", "email": "invalid", "password": "short", "role": "patient"})
    assert response.status_code == 400
    errors = response.get_json()["errors"]
    assert set(errors) == {"name", "email", "password"}


@pytest.mark.parametrize("role,dashboard", [("patient", "/patient/dashboard"), ("doctor", "/doctor/dashboard")])
def test_login_strictly_verifies_password_for_both_roles(client, role, dashboard):
    account = {"name": f"Secure {role.title()}", "email": f"secure-{role}@example.com", "password": "correct-password-8", "role": role}
    assert client.post("/api/auth/register", json=account).status_code == 201

    correct = client.post("/login", data={"username": account["email"], "password": account["password"], "mode": role})
    assert correct.status_code == 302
    assert correct.headers["Location"].endswith(dashboard)
    client.get("/logout")

    wrong = client.post("/login", data={"username": account["email"], "password": "definitely-wrong", "mode": role})
    assert wrong.status_code == 401
    assert b"Invalid email or password" in wrong.data
    with client.session_transaction() as current_session:
        assert "user" not in current_session

    missing = client.post("/login", data={"username": f"missing-{role}@example.com", "password": "anything-here", "mode": role})
    assert missing.status_code == 401
    assert b"Invalid email or password" in missing.data
    with client.session_transaction() as current_session:
        assert "user" not in current_session


def test_real_oncologist_location_apis_and_map_ui(client, monkeypatch):
    assert 0 < haversine_km(13.3409, 77.1010, 13.3379, 77.1173) < 5
    live_hospitals = [
        {"name": f"Live Hospital {index}", "doctor": "Hospital Care Team", "specialization": "Hospital", "city": "Tumakuru", "district": "Tumakuru", "lat": 13.34 + index / 1000, "lng": 77.10, "rating": 0, "phone": "", "distance_km": float(index), "source": "OpenStreetMap"}
        for index in range(1, 4)
    ]
    monkeypatch.setattr("utils.oncologist_finder.fetch_real_hospitals", lambda lat, lng, radius: live_hospitals)
    monkeypatch.setattr("utils.oncologist_finder.geocode_city", lambda city: (13.3409, 77.1010))
    nearby = client.get("/api/nearby-hospitals?lat=13.3409&lng=77.1010&radius_km=25")
    assert nearby.status_code == 200
    nearby_data = nearby.get_json()
    assert nearby_data["status"] == "ok"
    assert nearby_data["count"] >= 3
    distances = [hospital["distance_km"] for hospital in nearby_data["hospitals"]]
    assert distances == sorted(distances)
    assert all(hospital["district"] == "Tumakuru" for hospital in nearby_data["hospitals"])

    city = client.get("/api/hospitals-by-city?city=Tumkuru&radius_km=60").get_json()
    assert city["count"] >= 3
    assert all(hospital["district"] == "Tumakuru" for hospital in city["hospitals"])

    login_as(client, "patient")
    page = client.get("/patient/find-oncologist")
    assert page.status_code == 200
    assert b'id="useMyLocation"' in page.data
    assert b"Use Current Location" in page.data
    assert b'class="use-current-location"' in page.data
    assert b"navigator.geolocation.getCurrentPosition" in page.data
    assert b"/api/hospitals-by-city" in page.data
    assert b"leaflet@1.9.4" in page.data
    assert b".leaflet-container img" in page.data
    assert b"max-width: none !important" in page.data
    assert b"#leafletMap { height: 585px" in page.data
    assert b"SearchAreaControl" in page.data
    assert page.data.count(b"L.tileLayer(") == 1
    assert b"ResizeObserver(refreshMapSize)" in page.data
    assert b"googleMapsUrl" in page.data
    assert b"https://www.google.com/maps/search/?api=1&query=" in page.data
    assert b"window.open(googleMapsUrl(hospital),'_blank','noopener,noreferrer')" in page.data
    assert b"radius_km=25" in page.data
    assert b'value="Mangalore, Karnataka"' not in page.data
    assert b"initMap();searchCity()" not in page.data
    assert b"window.isSecureContext" in page.data
    assert b"navigator.permissions" in page.data
    assert b"LOCATION_CACHE_KEY" in page.data
    assert b"OpenStreetMap hospital" in page.data


def test_city_geocoder_does_not_force_karnataka(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"lat": "19.0760", "lon": "72.8777"}]

    def fake_get(url, params, headers, timeout):
        captured.update(params)
        return Response()

    finder_module.geocode_city.cache_clear()
    monkeypatch.setattr(finder_module.requests, "get", fake_get)
    assert finder_module.geocode_city("Mumbai") == (19.076, 72.8777)
    assert captured["q"] == "Mumbai"
    assert "Karnataka" not in captured["q"]


def test_shared_hamburger_menu_has_responsive_toggle_behavior(client):
    login_as(client, "patient")
    page = client.get("/patient/find-oncologist")
    assert b'class="icon-button patient-menu-button"' in page.data
    assert b'aria-controls="patientSidebar"' in page.data
    script = client.get("/static/js/oncovision.js")
    assert script.status_code == 200
    assert b"sidebar-collapsed" in script.data
    assert b"sidebar-mobile-open" in script.data
    assert b"sidebar-menu-overlay" in script.data
    assert b"event.key === 'Escape'" in script.data
    assert b"data-sidebar-close" in page.data
    assert b"dashboard-menu-open" in script.data
    stylesheet = client.get("/static/css/styles.css")
    assert b".dashboard-shell.sidebar-collapsed" in stylesheet.data
    assert b".sidebar.mobile-open" in stylesheet.data
    assert b"viewport-safe dashboard navigation for phones" in stylesheet.data
    assert b"max-height:100dvh" in stylesheet.data
    dashboard = client.get("/patient/dashboard")
    assert b'aria-controls="patientSidebar"' in dashboard.data
    assert b'aria-label="Close navigation"' in dashboard.data
    assert b'bi bi-x-lg' in dashboard.data
    assert b"icon.className = 'bi bi-list'" in script.data
    assert b"The close action belongs inside the visible sidebar" in stylesheet.data


def test_dashboard_sidebar_and_header_remain_stable_while_scrolling(client):
    login_as(client, "patient")
    stylesheet = client.get("/static/css/styles.css")
    assert stylesheet.status_code == 200
    assert b"Stable dashboard navigation" in stylesheet.data
    assert b".dashboard-shell>.sidebar{" in stylesheet.data
    assert b"position:sticky" in stylesheet.data
    assert b".dashboard-main>.topbar{" in stylesheet.data


def test_sidebar_close_button_is_inside_menu_and_header_button_opens_it(client):
    login_as(client, "patient")
    page = client.get("/patient/dashboard")
    assert b'id="patientSidebar"' in page.data
    assert b'data-sidebar-close' in page.data
    assert b'aria-controls="patientSidebar"' in page.data
    assert b'class="sidebar-brand-head"' in page.data
    assert page.data.count(b'class="ov-brand"') == 1
    stylesheet = client.get("/static/css/styles.css")
    assert b"The close action belongs inside the visible sidebar" in stylesheet.data
    assert b".dashboard-shell>.sidebar .sidebar-close-button{" in stylesheet.data


def test_collapsed_sidebar_keeps_compact_logo_in_header(client):
    login_as(client, "patient")
    dashboard = client.get("/patient/dashboard")
    assert b'class="collapsed-topbar-brand"' in dashboard.data
    assert b"Onco<span>Vision</span>" in dashboard.data
    stylesheet = client.get("/static/css/styles.css")
    assert b".dashboard-shell.sidebar-collapsed .collapsed-topbar-brand" in stylesheet.data


def test_landing_has_mobile_navigation_and_viewport(client):
    response = client.get("/")
    assert b'name="viewport"' in response.data
    assert b'class="mobile-menu"' in response.data
    css = client.get("/static/css/styles.css")
    assert css.status_code == 200
    assert b"Complete phone layout" in css.data
    assert b"overflow-x:hidden" in css.data
    assert b"oncovision-hero-v2.png" in css.data


def test_every_landing_navigation_target_exists(client):
    response = client.get("/")
    for section_id in ("home", "about", "features", "how-it-works", "patients", "doctors", "contact"):
        assert f'id="{section_id}"'.encode() in response.data
        assert f'href="#{section_id}"'.encode() in response.data
    assert b'aria-controls="landingMenu"' in response.data
    assert b'aria-expanded="false"' in response.data


def test_landing_audience_images_are_available(client):
    response = client.get("/")
    for image_name in ("patient-screening-consultation.png", "doctor-clinical-review.png"):
        assert image_name.encode() in response.data
        image = client.get(f"/static/images/{image_name}")
        assert image.status_code == 200
        assert len(image.data) > 100_000


def test_patient_dashboard(client):
    login_as(client, "patient")
    response = client.get("/patient/dashboard")
    assert response.status_code == 200
    assert b"Patient" in response.data or b"Dashboard" in response.data
    html = response.data.decode()
    assert html.index("Patient Mode") < html.index("Explain Doctor Report")
    assert html.index("Mammogram Analysis") < html.index("Explain Doctor Report") < html.index("My Results & Reviews")


def test_patient_dashboard_history_is_scoped_and_links_to_full_history(client, monkeypatch):
    requested_identities = []

    def patient_records(identity):
        requested_identities.append(identity)
        return [{
            "id": record_id,
            "report_id": f"OV-PRIVATE{record_id}",
            "class_name": "Benign",
            "confidence": 88.4,
            "created_at": "2026-09-10T08:30:00+00:00",
        } for record_id in range(91, 96)]

    monkeypatch.setattr("app.list_patient_analyses", patient_records)
    login_as(client, "patient")
    response = client.get("/patient/dashboard")
    assert response.status_code == 200
    assert requested_identities == ["patient@example.com"]
    assert b"OV-PRIVATE91" in response.data
    assert b"MAM-2026-0012" not in response.data
    assert b'href="/patient/results"' in response.data
    assert b'href="/patient/results#analysis-91"' in response.data
    assert b"My records only" in response.data
    assert b'id="historyToggle"' in response.data
    assert b'class="history-extra-row"' in response.data
    assert b"history-expanded" in response.data
    assert b"10 Sep 2026" in response.data
    assert b"02:00 PM IST" in response.data
    assert b'data-label="Report ID"' in response.data
    stylesheet = client.get("/static/css/styles.css")
    assert b".patient-recent-panel .data-table thead{display:none}" in stylesheet.data
    assert b".patient-dashboard-shell{width:100%;max-width:100%;overflow-x:clip}" in stylesheet.data
    assert b".patient-recent-panel.history-expanded .table-scroll{max-height:420px" in stylesheet.data


def test_login_name_is_shown_on_patient_pages(client):
    client.post("/api/auth/register", json={"name": "ABC", "email": "abc@example.com", "password": "secure-test-8", "role": "patient"})
    client.post("/login", data={"username": "abc@example.com", "password": "secure-test-8", "mode": "patient"})
    dashboard = client.get("/patient/dashboard")
    oncologists = client.get("/patient/find-oncologist")
    assert b"Hello, ABC" in dashboard.data
    assert b'<div class="avatar">A</div>' in dashboard.data
    assert b"<strong>ABC</strong>" in oncologists.data
    assert b"Aisha" not in dashboard.data


def test_email_login_uses_readable_name(client):
    client.post("/api/auth/register", json={"name": "rida", "email": "rida@example.com", "password": "secure-test-8", "role": "patient"})
    client.post("/login", data={"username": "rida@example.com", "password": "secure-test-8", "mode": "patient"})
    response = client.get("/patient/dashboard")
    assert b"Hello, rida" in response.data
    assert b"Hello, rida@example.com" not in response.data


def test_stage_prediction_rules():
    payload = {"tumor": "T2", "node": "N1", "metastasis": "M0", "grade": "Grade 2"}
    result = predict_stage(payload)
    assert result["stage"] == "Stage IIB"
    assert "T2" in result["display"]


def test_explain_stage_has_clinical_summary():
    summary = explain_stage({"tumor": "T2", "node": "N1", "metastasis": "M0", "er": "Positive", "pr": "Positive", "her2": "Negative", "ki67": 20})
    assert "Stage IIB" in summary
    assert "ER" in summary


@pytest.mark.parametrize("tumor,node,metastasis,expected", [
    ("T1", "N0", "M0", "Stage IA"),
    ("T2", "N0", "M0", "Stage IIA"),
    ("T2", "N1", "M0", "Stage IIB"),
    ("T3", "N2", "M0", "Stage IIIA"),
    ("T4", "N1", "M0", "Stage IIIB"),
    ("T1", "N3", "M0", "Stage IIIC"),
    ("T1", "N0", "M1", "Stage IV"),
])
def test_ajcc_anatomic_stage_groups(tumor, node, metastasis, expected):
    assert predict_stage({"tumor": tumor, "node": node, "metastasis": metastasis})["stage"] == expected


def test_image_simulation_classes():
    cls, confidence = simulate_image_classification("malignant_scan.png")
    assert cls == "Malignant"
    assert confidence > 90


def test_mammogram_report_is_hidden_before_upload(client):
    login_as(client, "patient")
    response = client.get("/patient/upload-predict")
    assert response.status_code == 200
    assert b'id="medicalReport" hidden' in response.data
    assert b"88.7%" not in response.data
    assert b'id="printReport"' in response.data
    assert b'id="mammogramPreview"' in response.data
    assert b'id="gradcamPreview"' in response.data
    assert b'attention_image_data_url' in response.data
    assert b'id="mammogramInput"' in response.data
    assert b'multiple hidden' in response.data
    assert b'id="studySelector"' in response.data
    assert b"Highest-Attention Region" in response.data
    assert b"Prototype / Non-Diagnostic Report" in response.data


def test_all_interactive_pages_render(client):
    login_as(client, "patient")
    for route in ["/patient/dashboard", "/patient/find-oncologist", "/patient/upload-predict", "/patient/report-explainer", "/patient/self-examination", "/patient/profile", "/chat", "/portal/ai-explanations", "/portal/health-knowledge", "/portal/help"]:
        assert client.get(route).status_code == 200
    assert client.get("/portal/settings").status_code == 404
    login_as(client, "doctor")
    for route in ["/doctor/dashboard", "/doctor/clinical-assessment", "/doctor/image-reviews", "/doctor/profile", "/chat", "/portal/reports", "/portal/prediction-history", "/portal/resources", "/portal/help"]:
        assert client.get(route).status_code == 200


def test_every_doctor_page_uses_one_consistent_sidebar(client):
    login_as(client, "doctor")
    routes = [
        "/doctor/dashboard",
        "/doctor/image-reviews",
        "/doctor/clinical-assessment",
        "/portal/prediction-history",
        "/portal/reports",
        "/chat",
        "/portal/resources",
        "/doctor/profile",
        "/portal/help",
    ]
    expected_labels = [
        b"Dashboard",
        b"Clinical Workflow",
        b"Mammogram Review Queue",
        b"Clinical Assessment",
        b"Prediction History",
        b"Reports",
        b"Clinical Support",
        b"Doctor AI Assistant",
        b"Clinical Resources",
        b"Doctor Profile",
        b"Help &amp; Support",
        b"Logout",
    ]
    for route in routes:
        page = client.get(route)
        assert page.status_code == 200
        assert page.data.count(b'id="doctorSidebar"') == 1
        assert page.data.count(b'class="ov-brand"') == 1
        assert all(label in page.data for label in expected_labels)
        assert b"Chat with AI Assistant" not in page.data
        assert b"Other Modules" not in page.data
        assert b">Account<" not in page.data


def test_logged_in_doctor_has_professional_profile(client):
    login_as(client, "doctor")
    dashboard = client.get("/doctor/dashboard")
    assert b'href="/doctor/profile"' in dashboard.data
    assert b"Doctor Profile" in dashboard.data

    page = client.get("/doctor/profile")
    assert page.status_code == 200
    assert b"Professional Information" in page.data
    assert b"Specialization" in page.data
    assert b"Medical registration number" in page.data
    assert b"Hospital / Clinic" in page.data
    assert b"Clinical Workspace" in page.data

    updated = client.post("/doctor/profile", data={
        "name": "Dr Profile Test", "email": "doctor@example.com",
        "phone": "9876543210", "specialization": "Radiology",
        "registration_number": "KMC-123", "hospital": "OncoVision Clinic",
        "experience": "8", "address": "Tumakuru",
    })
    assert updated.status_code == 200
    assert b"Doctor profile updated successfully" in updated.data
    assert b"Radiology" in updated.data


def test_doctor_reports_page_uses_live_database_records(client):
    login_as(client, "doctor")
    page = client.get("/portal/reports")
    assert page.status_code == 200
    assert b'data-report-stat="image_analyses"' in page.data
    assert b'id="reportsTable"' in page.data
    assert b"No reports generated yet" in page.data
    assert b"<b>12</b><span>Image analyses</span>" not in page.data
    assert b'class="report-result-pill ${resultClass}"' in page.data
    assert b'class="report-result">${escapeHtml(report.result)}' not in page.data

    response = client.get("/api/reports")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["reports_ready"] == payload["image_analyses"] + payload["saved_predictions"]
    assert payload["breakdown"]["benign"] + payload["breakdown"]["malignant"] == payload["image_analyses"]
    if payload["reports"]:
        report = payload["reports"][0]
        assert report["report_type"] in {"Mammogram Analysis", "Clinical Stage Prediction"}
        detail = client.get(report["view_url"])
        assert detail.status_code == 200
        assert b"Print / Save PDF" in detail.data
        assert b"Research Prototype / Non-Diagnostic Report" in detail.data


def test_active_session_cannot_return_to_role_login_until_logout(client):
    login_as(client, "doctor")
    response = client.get("/login")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/doctor/dashboard")
    client.get("/logout")
    assert client.get("/login").status_code == 200


def test_patient_chat_answers_breast_topics_and_refuses_other_organs(client):
    login_as(client, "patient")
    greeting = client.post("/api/chatbot", json={"message": "hi"}).get_json()["response"]
    symptom = client.post("/api/chatbot", json={"message": "I noticed an unusual change in my left breast"}).get_json()["response"]
    education = client.post("/api/chatbot", json={"message": "What is breast cancer?"}).get_json()["response"]
    off_topic = client.post("/api/chatbot", json={"message": "Tell me about kidney disease"}).get_json()["response"]
    unrelated = client.post("/api/chatbot", json={"message": "Who won the cricket match?"}).get_json()["response"]
    nipple_pain = client.post("/api/chatbot", json={"message": "pain on my nipples"}).get_json()["response"]
    discharge = client.post("/api/chatbot", json={"message": "clear discharge from my right nipple"}).get_json()["response"]
    lump = client.post("/api/chatbot", json={"message": "I found a lump in my left breast"}).get_json()["response"]
    assert "breast health" in greeting.lower()
    assert "assessed by a clinician" in symptom.lower()
    assert "abnormal cells" in education.lower()
    assert "only answer questions about breast health" in off_topic.lower()
    assert "only with breast-related topics" in unrelated.lower()
    assert "friction" in nipple_pain.lower()
    assert "without squeezing" in discharge.lower()
    assert "left-breast" in lump.lower()
    assert len({symptom, nipple_pain, discharge, lump}) == 4


def test_openrouter_compatible_ai_assistant_endpoint_has_safe_fallback(client):
    response = client.post("/api/ai-assistant", json={"message": "I have pain in my right breast"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["provider"] == "local-fallback"
    assert payload["reply"] == payload["response"]
    assert "right-breast pain" in payload["reply"].lower()
    assert "not a medical diagnosis" in payload["reply"].lower()
    assert client.post("/api/ai-assistant", json={"message": ""}).status_code == 400


def test_patient_chat_understands_short_follow_up_from_same_conversation(client):
    login_as(client, "patient")
    client.delete("/api/chat-history")
    first = client.post("/api/chatbot", json={"message": "I have pain in my left breast"}).get_json()
    follow_up = client.post(
        "/api/chatbot",
        json={"message": "what should I do bro", "conversation_id": first["conversation_id"]},
    ).get_json()["response"]
    assert "only with breast-related topics" not in follow_up.lower()
    assert "breast pain" in follow_up.lower()
    assert "assessment" in follow_up.lower()

    for question in ("is it serious?", "which test is needed?", "ab kya karu?"):
        response = client.post(
            "/api/chatbot",
            json={"message": question, "conversation_id": first["conversation_id"]},
        ).get_json()["response"]
        assert "only with breast-related topics" not in response.lower()
        assert "breast pain" in response.lower()

    new_topic = client.post(
        "/api/chatbot",
        json={"message": "What does a breast biopsy mean?", "conversation_id": first["conversation_id"]},
    ).get_json()["response"]
    assert "tissue sample" in new_topic.lower()

    unrelated = client.post(
        "/api/chatbot",
        json={"message": "Who won the cricket match?", "conversation_id": first["conversation_id"]},
    ).get_json()["response"]
    assert "only with breast-related topics" in unrelated.lower()


def test_chat_history_is_persistent_user_scoped_and_clearable(client):
    assert client.get("/api/chat-history").status_code == 401
    login_as(client, "patient")
    client.delete("/api/chat-history")
    response = client.post("/api/chatbot", json={"message": "What is a mammogram?"})
    assert response.status_code == 200
    history = client.get("/api/chat-history").get_json()
    assert history["count"] == 2
    assert [item["role"] for item in history["messages"]] == ["user", "assistant"]
    assert history["messages"][0]["content"] == "What is a mammogram?"

    client.get("/logout")
    login_as(client, "doctor")
    client.delete("/api/chat-history")
    assert client.get("/api/chat-history").get_json()["count"] == 0

    client.get("/logout")
    login_as(client, "patient")
    assert client.get("/api/chat-history").get_json()["count"] == 2
    cleared = client.delete("/api/chat-history").get_json()
    assert cleared["deleted"] == 2
    assert client.get("/api/chat-history").get_json()["count"] == 0
    page = client.get("/chat")
    assert b'id="clearChat"' in page.data
    assert b"loadHistory" in page.data


def test_chat_supports_saved_threads_new_chat_and_daily_rollover(client, monkeypatch):
    credentials = {
        "name": "Thread Tester", "email": f"thread-test-{uuid4().hex}@example.com",
        "password": "testpassword8", "role": "patient",
    }
    client.post("/api/auth/register", json=credentials)
    client.post("/login", data={"username": credentials["email"], "password": credentials["password"], "mode": "patient"})

    monkeypatch.setattr(storage_module, "_local_date", lambda: "2026-09-10")
    first = client.get("/api/chat-conversations").get_json()
    first_id = first["active_conversation_id"]
    sent = client.post("/api/chatbot", json={"message": "Explain screening", "conversation_id": first_id})
    assert sent.status_code == 200

    second = client.post("/api/chat-conversations").get_json()["conversation"]
    assert second["id"] != first_id
    client.post("/api/chatbot", json={"message": "What is a biopsy?", "conversation_id": second["id"]})
    conversations = client.get("/api/chat-conversations").get_json()["conversations"]
    assert {first_id, second["id"]}.issubset({item["id"] for item in conversations})
    assert {"Explain screening", "What is a biopsy?"}.issubset({item["title"] for item in conversations})
    first_messages = client.get(f"/api/chat-conversations/{first_id}/messages").get_json()["messages"]
    second_messages = client.get(f"/api/chat-conversations/{second['id']}/messages").get_json()["messages"]
    assert "Explain screening" in {item["content"] for item in first_messages}
    assert "What is a biopsy?" in {item["content"] for item in second_messages}

    monkeypatch.setattr(storage_module, "_local_date", lambda: "2026-09-11")
    next_day = client.get("/api/chat-conversations").get_json()
    assert next_day["active_conversation_id"] not in {first_id, second["id"]}

    page = client.get("/chat")
    assert b'id="chatHistoryDrawer"' in page.data
    assert b'id="newChat"' in page.data
    assert b"Saved privately to your account" in page.data
    assert b"deleteConversation" in page.data


def test_chat_conversation_can_be_deleted_without_affecting_other_threads(client):
    login_as(client, "patient")
    first = client.post("/api/chat-conversations").get_json()["conversation"]
    second = client.post("/api/chat-conversations").get_json()["conversation"]
    client.post("/api/chatbot", json={"message": "Explain screening", "conversation_id": first["id"]})
    client.post("/api/chatbot", json={"message": "Explain biopsy", "conversation_id": second["id"]})

    deleted = client.delete(f"/api/chat-conversations/{first['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/chat-conversations/{first['id']}/messages").status_code == 404
    remaining = client.get(f"/api/chat-conversations/{second['id']}/messages")
    assert remaining.status_code == 200
    assert remaining.get_json()["count"] == 2


def test_chat_conversation_delete_is_account_scoped(client):
    login_as(client, "patient")
    conversation = client.post("/api/chat-conversations").get_json()["conversation"]
    client.get("/logout")
    login_as(client, "doctor")
    assert client.delete(f"/api/chat-conversations/{conversation['id']}").status_code == 404


def test_doctor_assistant_uses_clinical_prompt_register_and_chips(client):
    login_as(client, "doctor")
    page = client.get("/chat")
    assert page.status_code == 200
    assert b"Differential for a new palpable mass" in page.data
    assert b"Interpreting BI-RADS 4 findings" in page.data
    assert b"ER/PR/HER2 implications for treatment pathway" in page.data
    assert b"What breast changes need a doctor?" not in page.data
    assert b'placeholder="Ask a clinical question..."' in page.data
    assert b"Clinical support only" in page.data
    assert b"Verify against current guidelines and clinical judgment" in page.data
    assert b'placeholder="Ask a breast-health question..."' not in page.data
    assert b"Not emergency or diagnostic advice" not in page.data

    pain = client.post("/api/chatbot", json={"message": "Clinical differential for focal breast pain"}).get_json()["response"]
    assert "differential" in pain.lower()
    assert "clinical-imaging correlation" in pain.lower()
    assert "i'm sorry" not in pain.lower()
    assert "encourage your patient" not in pain.lower()
    assert "see a doctor" not in pain.lower()
    assert pain.endswith("AI-generated support — verify against current guidelines and clinical judgment for patient-specific decisions.")

    birads = client.post("/api/chatbot", json={"message": "Interpreting BI-RADS 4 findings"}).get_json()["response"]
    assert "tissue diagnosis" in birads.lower()
    assert "radiologic-pathologic concordance" in birads.lower()
    off_topic = client.post("/api/chatbot", json={"message": "Explain a kidney billing question"}).get_json()["response"]
    assert "scoped to breast-cancer clinical support" in off_topic.lower()


def test_patient_chat_keeps_patient_labels_and_ai_replies_are_justified(client):
    login_as(client, "patient")
    page = client.get("/chat")
    assert page.status_code == 200
    assert b'placeholder="Ask a breast-health question..."' in page.data
    assert b"Breast-health questions only" in page.data
    assert b"Not emergency or diagnostic advice" in page.data
    assert b'placeholder="Ask a clinical question..."' not in page.data
    assert b"Clinical support only" not in page.data

    stylesheet = client.get("/static/css/styles.css")
    css = b"".join(stylesheet.data.split())
    assert b".chat-bubble.bot{" in css
    assert b"text-align:justify" in css
    assert b"text-justify:inter-word" in css
    assert b"hyphens:auto" in css


def test_chat_backend_routes_provider_to_role_specific_prompt(client, monkeypatch):
    observed_modes = []
    def fake_provider(message, mode="patient", context=None, history=None):
        observed_modes.append(mode)
        return "Role-specific provider response."
    monkeypatch.setattr(assistant_module, "_openrouter_reply", fake_provider)

    login_as(client, "doctor")
    doctor_reply = client.post("/api/chatbot", json={"message": "Review this breast case"}).get_json()["response"]
    assert observed_modes[-1] == "doctor"
    assert "AI-generated support" in doctor_reply
    assert "not a medical diagnosis" not in doctor_reply.lower()

    client.get("/logout")
    login_as(client, "patient")
    patient_reply = client.post("/api/chatbot", json={"message": "I have breast pain"}).get_json()["response"]
    assert observed_modes[-1] == "patient"
    assert "not a medical diagnosis" in patient_reply.lower()
    assert "AI-generated support" not in patient_reply


def test_self_examination_guide_has_working_interactive_controls(client):
    login_as(client, "patient")
    response = client.get("/patient/self-examination")
    assert response.status_code == 200
    assert response.data.count(b'class="self-step-card"') == 6
    assert b'id="guideDialog"' in response.data
    assert b'id="symptomTracker"' in response.data
    assert b"localStorage.setItem" in response.data
    assert b"Find a Specialist Near You" in response.data
    assert response.data.count(b"reference-step-art") >= 6
    assert b"guideDetails" in response.data
    assert b"How to do it" in response.data
    assert b"What to notice" in response.data
    assert b"Know what is normal for you" in response.data
    assert b"Awareness-not diagnosis" in response.data.replace(b"\xe2\x80\x94", b"-")
    assert b"NHS breast/chest checking guidance" in response.data
    assert b"American Cancer Society" not in response.data
    assert b"youtube-nocookie.com/embed/-ygucOBbKJA" in response.data
    assert b"controls=1" in response.data
    assert b"allowfullscreen" in response.data
    stylesheet = client.get("/static/css/styles.css")
    assert b"Self-examination: strict phone-width layout" in stylesheet.data
    assert b".self-video-frame{position:relative!important;display:block!important;width:100%!important" in stylesheet.data
    assert b".self-step-grid{display:grid;grid-template-columns:minmax(0,1fr)" in stylesheet.data
    image = client.get("/static/images/self-examination-reference.jpg")
    assert image.status_code == 200
    assert len(image.data) > 50_000


def test_report_explainer_uses_report_birads_without_fake_percentages(client):
    login_as(client, "patient")
    page = client.get("/patient/report-explainer")
    assert page.status_code == 200
    assert b'id="reportExplanation" class="report-result" hidden' in page.data
    assert b"Explain Doctor Report" in client.get("/patient/dashboard").data

    response = client.post("/api/report-explain", data={"report_text": "Findings: no suspicious mass or calcification. Assessment: BI-RADS 1 Negative. Recommendation: routine annual screening."})
    data = response.get_json()
    assert response.status_code == 200
    assert data["birads"] == "1"
    assert data["risk_range"] == "Essentially 0%"
    assert data["level"] == "reassuring"

    no_category = client.post("/api/report-explain", data={"report_text": "Breast imaging report. There is an asymmetry. Please compare with prior examination."}).get_json()
    assert no_category["risk_range"] is None
    assert no_category["category_label"] == "Not found in report"


def test_report_explainer_birads_subcategories_and_role_protection(client):
    result = explain_report("Impression: BI-RADS 4A low-suspicion mass. Recommendation: ultrasound-guided biopsy.")
    assert result["risk_range"] == ">2% to ≤10%"
    assert "biopsy" in result["recommendation"].lower()
    assert client.get("/patient/report-explainer").status_code == 302
    login_as(client, "doctor")
    assert client.get("/patient/report-explainer").headers["Location"].endswith("/doctor/dashboard")


def test_report_explainer_does_not_treat_template_options_as_patient_results():
    abnormal_template = explain_report("Sample Lay Letter for Abnormal Finding (to be used with BI-RADS 4-5) Name of Patient/ID Dear Patient: your [mammogram] showed an abnormality that requires a biopsy.")
    assert abnormal_template["document_type"] == "sample-template"
    assert abnormal_template["category_label"] == "Abnormal finding letter template"
    assert abnormal_template["risk_range"] is None
    assert "biopsy" in abnormal_template["overview"].lower()

    blank_form = explain_report("Breast imaging report Shape: None/Oval/Round/Irregular Impression: BI-RADS: 0/1/2/3/4A/4B/4C/5/6 Further assessment advised: None/Referral for core biopsy")
    assert blank_form["document_type"] == "blank-proforma"
    assert blank_form["birads"] is None
    assert blank_form["findings"][0]["title"] == "No patient finding entered"


def test_attention_peak_is_calculated_from_each_heatmap_and_ignores_border():
    first = np.zeros((224, 224), dtype=np.float32)
    second = np.zeros((224, 224), dtype=np.float32)
    first[45:55, 55:65] = 1
    second[145:155, 155:165] = 1
    first[0, 0] = 2  # Strong border artifact should not win.
    peak_one = _attention_peak(first)
    peak_two = _attention_peak(second)
    assert 40 <= peak_one[0] <= 60 and 50 <= peak_one[1] <= 70
    assert 140 <= peak_two[0] <= 160 and 150 <= peak_two[1] <= 170
    assert peak_one != peak_two


def test_common_composite_mammogram_layouts_are_split_into_views(client):
    two_panel = np.zeros((220, 500), dtype=np.uint8)
    two_panel[20:200, 20:230] = 90
    two_panel[20:200, 270:480] = 150
    ok, encoded = cv2.imencode(".jpg", two_panel)
    assert ok
    views = extract_mammogram_views(encoded.tobytes(), "study.jpg")
    assert len(views) == 2
    assert views[0][0] == "study-view-1.jpg"
    login_as(client, "patient")
    page = client.get("/patient/upload-predict")
    assert b"common 2-panel/4-panel exports" in page.data
    assert b"/api/analyses" in page.data


def test_chat_covers_broader_breast_health_topics(client):
    login_as(client, "patient")
    questions = {
        "dislocate of nipple": "force it back",
        "allergic itchy rash on breast": "fragranced products",
        "breast infection with pus": "antibiotics or drainage",
        "breast injury and bruise": "cold pack",
        "pain while breastfeeding": "pregnancy and breastfeeding",
        "one breast is bigger": "difference in breast size",
        "male breast lump": "men can also develop",
        "breast implant pain": "implant type",
        "what is a breast cyst": "common benign conditions",
        "what are dense breasts": "fibroglandular tissue",
        "what does BI-RADS mean": "standardized radiology",
        "breast ultrasound or MRI": "appropriate test",
    }
    answers = []
    for question, expected in questions.items():
        response = client.post("/api/chatbot", json={"message": question}).get_json()["response"]
        assert expected.lower() in response.lower()
        answers.append(response)
    assert len(set(answers)) == len(questions)


def test_save_clinical_record_and_history(client):
    login_as(client, "doctor")
    payload = {
        "patient_name": "Manual Patient", "patient_id": "PID-1001",
        "tumor": "T2", "node": "N1", "metastasis": "M0",
        "grade": "Grade 2", "er": "Positive", "pr": "Positive",
        "her2": "Negative", "ki67": "25", "age": "48",
    }
    response = client.post("/api/clinical-records", json=payload)
    assert response.status_code == 201
    assert response.get_json()["record"]["stage"] == "Stage IIB"
    history = client.get("/portal/prediction-history")
    assert history.status_code == 200
    predictions = client.get("/api/predictions").get_json()["predictions"]
    assert any(record["predicted_stage"] == "Stage IIB" for record in predictions)


def test_clinical_resources_are_authoritative_and_useful(client):
    login_as(client, "doctor")
    page = client.get("/portal/resources")
    assert page.status_code == 200
    assert b"AJCC Cancer Staging Systems" in page.data
    assert b"NCCN Breast Cancer Guidelines" in page.data
    assert b"WHO Global Breast Cancer Framework" in page.data
    assert b"Biomarker &amp; Pathology Reference" in page.data
    assert b"Grad-CAM" in page.data


def test_save_clinical_record_validates_required_fields(client):
    login_as(client, "doctor")
    response = client.post("/api/clinical-records", json={"tumor": "T2"})
    assert response.status_code == 400


def test_clinical_assessment_uses_manual_patient_and_neutral_defaults(client):
    login_as(client, "doctor")
    page = client.get("/doctor/clinical-assessment")
    assert page.status_code == 200
    assert b'name="patient_name"' in page.data
    assert b'name="patient_id"' in page.data
    assert b'placeholder="Enter patient name"' in page.data
    assert b'Ananya Sharma' not in page.data
    assert b'name="ki67" type="number" min="0" max="100" step="0.1" value="0"' in page.data
    assert b'name="age" type="number" min="1" max="130" step="1" value="0"' in page.data
    assert b'<option value="" selected disabled>Select T stage</option>' in page.data


def test_save_clinical_record_requires_manual_patient_identity(client):
    login_as(client, "doctor")
    payload = {
        "tumor": "T2", "node": "N1", "metastasis": "M0",
        "grade": "Grade 2", "er": "Positive", "pr": "Positive",
        "her2": "Negative", "ki67": 25, "age": 48,
    }
    response = client.post("/api/clinical-records", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"] == "Patient name is required"


def test_patient_cannot_access_doctor_routes(client):
    login_as(client, "patient")
    response = client.get("/doctor/clinical-assessment")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/patient/dashboard")
    assert client.post("/api/stage-predict", json={}).status_code == 302


def test_doctor_cannot_access_patient_routes(client):
    login_as(client, "doctor")
    response = client.get("/patient/upload-predict")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/doctor/dashboard")
    assert client.post("/api/analysis").status_code == 302


def test_logout_clears_role_session(client):
    login_as(client, "patient")
    assert client.get("/logout").status_code == 302
    assert "/login" in client.get("/patient/dashboard").headers["Location"]
