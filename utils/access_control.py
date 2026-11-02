"""Time-limited patient-to-doctor access grants for OncoVision."""

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, session


access_control = Blueprint("access_control", __name__)
CODE_VALIDITY_MINUTES = 30
ROOT = Path(__file__).resolve().parents[1]


def _db_path():
    configured = os.getenv("ONCOVISION_ACCESS_DB")
    return Path(configured) if configured else ROOT / "instance" / "access_control.sqlite3"


def _utcnow():
    return datetime.now(timezone.utc)


def _ensure_schema(conn):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS access_grants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            code TEXT NOT NULL,
            doctor_id TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            claimed_at TEXT,
            revoked INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_access_code ON access_grants(code)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_access_doctor_patient ON access_grants(doctor_id, patient_id, expires_at)"
    )


def _get_db():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def init_access_control_db():
    with _get_db() as conn:
        conn.commit()


def _generate_unique_code(conn):
    now = _utcnow().isoformat()
    for _ in range(50):
        code = f"{secrets.randbelow(1_000_000):06d}"
        exists = conn.execute(
            "SELECT 1 FROM access_grants WHERE code = ? AND revoked = 0 AND expires_at > ?",
            (code, now),
        ).fetchone()
        if not exists:
            return code
    raise RuntimeError("Could not allocate a unique access code")


def has_active_access(doctor_id, patient_id):
    if not doctor_id or not patient_id:
        return False
    with _get_db() as conn:
        row = conn.execute(
            """SELECT 1 FROM access_grants
               WHERE doctor_id = ? AND patient_id = ? AND revoked = 0 AND expires_at > ?
               ORDER BY expires_at DESC LIMIT 1""",
            (str(doctor_id), str(patient_id), _utcnow().isoformat()),
        ).fetchone()
    return row is not None


def active_patient_ids(doctor_id):
    """Return the distinct patient identities this doctor may access now."""
    if not doctor_id:
        return []
    with _get_db() as conn:
        rows = conn.execute(
            """SELECT patient_id, MAX(expires_at) AS expires_at
               FROM access_grants
               WHERE doctor_id = ? AND revoked = 0 AND expires_at > ?
               GROUP BY patient_id ORDER BY expires_at DESC""",
            (str(doctor_id), _utcnow().isoformat()),
        ).fetchall()
    return [row["patient_id"] for row in rows]


def requires_patient_access(get_patient_id_from_request):
    """Gate a patient-specific view for the owning patient or an authorized doctor."""
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            user = session.get("user")
            if not user:
                return jsonify({"status": "access_denied", "error": "Login required"}), 401
            patient_id = get_patient_id_from_request(request, *args, **kwargs)
            # A None identifier is used for doctor-owned saved records, which remain available.
            if patient_id is None:
                return view_func(*args, **kwargs)
            allowed = (
                user.get("mode") == "patient" and user.get("identity") == patient_id
            ) or (
                user.get("mode") == "doctor" and has_active_access(user.get("identity"), patient_id)
            )
            if allowed:
                return view_func(*args, **kwargs)
            message = "Your access to this patient's data has expired — ask them for a new code."
            if request.path.startswith("/api/") or request.path.startswith("/analysis-file/"):
                return jsonify({"status": "access_denied", "error": message}), 403
            return render_template(
                "access_expired.html", message=message, active_page="reports", page_title="Access expired"
            ), 403
        return wrapped
    return decorator


@access_control.post("/api/access/generate-code")
def generate_code():
    user = session.get("user")
    if not user or user.get("mode") != "patient":
        return jsonify({"error": "Only patients can generate access codes"}), 403
    now = _utcnow()
    expires = now + timedelta(minutes=CODE_VALIDITY_MINUTES)
    with _get_db() as conn:
        # Only one unclaimed code should be shareable at a time.
        conn.execute(
            "UPDATE access_grants SET revoked = 1 WHERE patient_id = ? AND doctor_id IS NULL AND revoked = 0",
            (user["identity"],),
        )
        code = _generate_unique_code(conn)
        conn.execute(
            """INSERT INTO access_grants (patient_id, code, created_at, expires_at)
               VALUES (?, ?, ?, ?)""",
            (user["identity"], code, now.isoformat(), expires.isoformat()),
        )
    return jsonify({
        "status": "ok", "code": code, "expires_in_minutes": CODE_VALIDITY_MINUTES,
        "expires_at": expires.isoformat(),
    })


@access_control.post("/api/access/claim-code")
def claim_code():
    user = session.get("user")
    if not user or user.get("mode") != "doctor":
        return jsonify({"error": "Only doctors can claim access codes"}), 403
    code = str((request.get_json(silent=True) or {}).get("code", "")).strip()
    if len(code) != 6 or not code.isdigit():
        return jsonify({"status": "invalid", "message": "Enter a valid 6-digit access code."}), 400
    now = _utcnow()
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM access_grants WHERE code = ? AND revoked = 0 ORDER BY id DESC LIMIT 1",
            (code,),
        ).fetchone()
        if row is None:
            return jsonify({"status": "invalid", "message": "Invalid access code."}), 404
        if datetime.fromisoformat(row["expires_at"]) <= now:
            return jsonify({"status": "expired", "message": "This access code has expired."}), 410
        if row["doctor_id"] and row["doctor_id"] != user["identity"]:
            return jsonify({"status": "invalid", "message": "This access code has already been used."}), 409
        conn.execute(
            "UPDATE access_grants SET doctor_id = ?, claimed_at = ? WHERE id = ?",
            (user["identity"], now.isoformat(), row["id"]),
        )
    return jsonify({
        "status": "ok", "patient_id": row["patient_id"],
        "access_valid_until": row["expires_at"],
    })


@access_control.get("/api/access/my-active-patients")
def my_active_patients():
    user = session.get("user")
    if not user or user.get("mode") != "doctor":
        return jsonify({"error": "Doctor login required"}), 401
    now = _utcnow().isoformat()
    with _get_db() as conn:
        rows = conn.execute(
            """SELECT patient_id, MAX(expires_at) AS expires_at
               FROM access_grants
               WHERE doctor_id = ? AND revoked = 0 AND expires_at > ?
               GROUP BY patient_id ORDER BY expires_at DESC""",
            (user["identity"], now),
        ).fetchall()
    return jsonify({
        "status": "ok",
        "active_patients": [dict(row) for row in rows],
    })

