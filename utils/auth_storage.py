"""Small local account store for the OncoVision prototype."""

import json
import os
import re
from pathlib import Path
from threading import RLock

from werkzeug.security import check_password_hash, generate_password_hash


_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_LOCK = RLock()


def _users_file():
    configured = os.getenv("ONCOVISION_USERS_FILE")
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / "instance" / "users.json"


def _read_users():
    path = _users_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def register_user(name, email, password, role):
    name = str(name or "").strip()
    email = str(email or "").strip().lower()
    password = str(password or "")
    role = str(role or "").strip().lower()
    errors = {}
    if len(name) < 2:
        errors["name"] = "Please enter your full name."
    if not _EMAIL_PATTERN.fullmatch(email):
        errors["email"] = "Please enter a valid email address."
    if len(password) < 8:
        errors["password"] = "Password must be at least 8 characters."
    if role not in {"patient", "doctor"}:
        errors["role"] = "Please choose Patient or Doctor."
    if errors:
        return None, errors

    with _LOCK:
        users = _read_users()
        if email in users:
            return None, {"email": "An account with this email already exists."}
        users[email] = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": role,
        }
        path = _users_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(users, indent=2), encoding="utf-8")
        temporary.replace(path)
        return {key: users[email][key] for key in ("name", "email", "role")}, None


def authenticate_user(email, password, role):
    email = str(email or "").strip().lower()
    with _LOCK:
        account = _read_users().get(email)
    if not account:
        return None
    if account.get("role") != role or not check_password_hash(account.get("password_hash", ""), password or ""):
        return False
    return {"name": account["name"], "identity": email, "mode": account["role"]}


def count_users_by_role(role):
    """Return the number of registered accounts for a role."""
    with _LOCK:
        users = _read_users()
    return sum(account.get("role") == role for account in users.values())
