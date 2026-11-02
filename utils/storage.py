"""SQLite persistence for the connected patient-to-doctor review workflow."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "instance"
UPLOAD_DIR = INSTANCE_DIR / "uploads"
DB_PATH = INSTANCE_DIR / "oncovision.sqlite3"
APP_TIMEZONE = ZoneInfo("Asia/Kolkata")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _local_date():
    return datetime.now(APP_TIMEZONE).date().isoformat()


def connect():
    INSTANCE_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS image_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                patient_identity TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                original_file TEXT NOT NULL,
                processed_file TEXT,
                heatmap_file TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                malignancy_probability REAL NOT NULL,
                explanation TEXT NOT NULL,
                next_step TEXT NOT NULL,
                model_name TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending review',
                doctor_name TEXT,
                doctor_notes TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            )
            """
        )
        columns = {row[1] for row in db.execute("PRAGMA table_info(image_analyses)")}
        if "processed_file" not in columns:
            db.execute("ALTER TABLE image_analyses ADD COLUMN processed_file TEXT")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identity TEXT NOT NULL,
                user_mode TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                provider TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_identity ON chat_messages(user_identity, user_mode, id)"
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS chat_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identity TEXT NOT NULL,
                user_mode TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New chat',
                conversation_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_conversations_identity "
            "ON chat_conversations(user_identity, user_mode, updated_at DESC)"
        )
        chat_columns = {row[1] for row in db.execute("PRAGMA table_info(chat_messages)")}
        if "conversation_id" not in chat_columns:
            db.execute("ALTER TABLE chat_messages ADD COLUMN conversation_id INTEGER")

        # Preserve messages created before multi-conversation chat was added.
        legacy_groups = db.execute(
            """SELECT user_identity, user_mode, MIN(created_at), MAX(created_at)
               FROM chat_messages WHERE conversation_id IS NULL
               GROUP BY user_identity, user_mode"""
        ).fetchall()
        for identity, mode, created_at, updated_at in legacy_groups:
            date_value = (created_at or _utc_now())[:10]
            cursor = db.execute(
                """INSERT INTO chat_conversations
                   (user_identity, user_mode, title, conversation_date, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (identity, mode, "Previous conversation", date_value,
                 created_at or _utc_now(), updated_at or created_at or _utc_now()),
            )
            db.execute(
                """UPDATE chat_messages SET conversation_id = ?
                   WHERE user_identity = ? AND user_mode = ? AND conversation_id IS NULL""",
                (cursor.lastrowid, identity, mode),
            )
        db.execute(
            """CREATE TABLE IF NOT EXISTS clinical_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_identity TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                tumor TEXT NOT NULL, node TEXT NOT NULL, metastasis TEXT NOT NULL,
                grade TEXT NOT NULL, er TEXT, pr TEXT, her2 TEXT, ki67 REAL,
                age INTEGER, predicted_stage TEXT NOT NULL, confidence REAL NOT NULL,
                summary TEXT NOT NULL, notes TEXT, created_at TEXT NOT NULL
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_prediction_doctor ON clinical_predictions(doctor_identity, id)")


def create_analysis(values):
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO image_analyses (
                report_id, patient_identity, patient_name, original_filename,
                original_file, processed_file, heatmap_file, class_name, confidence,
                malignancy_probability, explanation, next_step, model_name,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["report_id"], values["patient_identity"], values["patient_name"],
                values["original_filename"], values["original_file"], values["processed_file"], values["heatmap_file"],
                values["class_name"], values["confidence"], values["malignancy_probability"],
                values["explanation"], values["next_step"], values["model_name"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def list_patient_analyses(identity):
    with connect() as db:
        return db.execute(
            "SELECT * FROM image_analyses WHERE patient_identity = ? ORDER BY id DESC", (identity,)
        ).fetchall()


def list_all_analyses(patient_identities=None):
    with connect() as db:
        if patient_identities is None:
            return db.execute("SELECT * FROM image_analyses ORDER BY id DESC").fetchall()
        identities = list(dict.fromkeys(patient_identities))
        if not identities:
            return []
        placeholders = ",".join("?" for _ in identities)
        return db.execute(
            f"SELECT * FROM image_analyses WHERE patient_identity IN ({placeholders}) ORDER BY id DESC",
            identities,
        ).fetchall()


def list_latest_patient_analyses(patient_identities=None):
    """Return only each patient's newest case for the queue overview."""
    with connect() as db:
        if patient_identities is not None:
            identities = list(dict.fromkeys(patient_identities))
            if not identities:
                return []
            placeholders = ",".join("?" for _ in identities)
            return db.execute(
                f"""SELECT a.* FROM image_analyses a
                   JOIN (SELECT patient_identity, MAX(id) AS latest_id
                         FROM image_analyses WHERE patient_identity IN ({placeholders})
                         GROUP BY patient_identity) p ON p.latest_id = a.id
                   ORDER BY a.id DESC""",
                identities,
            ).fetchall()
        return db.execute(
            """SELECT a.* FROM image_analyses a
               JOIN (SELECT patient_identity, MAX(id) AS latest_id
                     FROM image_analyses GROUP BY patient_identity) p
                 ON p.latest_id = a.id
               ORDER BY a.id DESC"""
        ).fetchall()


def get_analysis(analysis_id):
    with connect() as db:
        return db.execute("SELECT * FROM image_analyses WHERE id = ?", (analysis_id,)).fetchone()


def dashboard_stats(registered_patients=0, patient_identities=None):
    """Compute doctor dashboard totals from persisted analyses."""
    analyses = list_all_analyses(patient_identities)
    latest = list_latest_patient_analyses(patient_identities)
    return {
        "patients": int(registered_patients),
        "stable": sum(row["class_name"] == "Benign" or row["malignancy_probability"] < 50 for row in latest),
        "high_risk": sum(row["class_name"] == "Malignant" or row["malignancy_probability"] >= 50 for row in latest),
        "ai_cases": len(analyses),
    }


def latest_patient_analysis(identity):
    with connect() as db:
        return db.execute(
            "SELECT * FROM image_analyses WHERE patient_identity = ? ORDER BY id DESC LIMIT 1",
            (identity,),
        ).fetchone()


def review_analysis(analysis_id, doctor_name, status, notes):
    with connect() as db:
        db.execute(
            """UPDATE image_analyses
               SET review_status = ?, doctor_name = ?, doctor_notes = ?, reviewed_at = ?
               WHERE id = ?""",
            (status, doctor_name, notes, datetime.now(timezone.utc).isoformat(), analysis_id),
        )
    return get_analysis(analysis_id)


def create_chat_conversation(identity, mode, title="New chat", conversation_date=None):
    """Create an account- and mode-scoped chat thread."""
    now = _utc_now()
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO chat_conversations
               (user_identity, user_mode, title, conversation_date, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (identity, mode, title, conversation_date or _local_date(), now, now),
        )
        return cursor.lastrowid


def get_chat_conversation(conversation_id, identity, mode):
    with connect() as db:
        return db.execute(
            """SELECT id, title, conversation_date, created_at, updated_at
               FROM chat_conversations
               WHERE id = ? AND user_identity = ? AND user_mode = ?""",
            (conversation_id, identity, mode),
        ).fetchone()


def get_or_create_daily_chat(identity, mode):
    """Return today's newest thread, creating the day's first thread if needed."""
    today = _local_date()
    with connect() as db:
        row = db.execute(
            """SELECT id, title, conversation_date, created_at, updated_at
               FROM chat_conversations
               WHERE user_identity = ? AND user_mode = ? AND conversation_date = ?
               ORDER BY updated_at DESC, id DESC LIMIT 1""",
            (identity, mode, today),
        ).fetchone()
    if row:
        return row
    conversation_id = create_chat_conversation(identity, mode, conversation_date=today)
    return get_chat_conversation(conversation_id, identity, mode)


def list_chat_conversations(identity, mode, limit=50):
    safe_limit = max(1, min(int(limit), 100))
    with connect() as db:
        return db.execute(
            """SELECT c.id, c.title, c.conversation_date, c.created_at, c.updated_at,
                      COUNT(m.id) AS message_count,
                      COALESCE((SELECT content FROM chat_messages last_message
                                WHERE last_message.conversation_id = c.id
                                ORDER BY last_message.id DESC LIMIT 1), '') AS preview
               FROM chat_conversations c
               LEFT JOIN chat_messages m ON m.conversation_id = c.id
               WHERE c.user_identity = ? AND c.user_mode = ?
               GROUP BY c.id
               ORDER BY c.updated_at DESC, c.id DESC LIMIT ?""",
            (identity, mode, safe_limit),
        ).fetchall()


def add_chat_message(identity, mode, role, content, provider=None, conversation_id=None):
    """Persist one user-specific chat turn and return its database id."""
    if role not in {"user", "assistant"}:
        raise ValueError("Chat role must be user or assistant.")
    if conversation_id is None:
        conversation_id = get_or_create_daily_chat(identity, mode)["id"]
    if not get_chat_conversation(conversation_id, identity, mode):
        raise ValueError("Chat conversation was not found for this user.")
    now = _utc_now()
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO chat_messages
               (user_identity, user_mode, role, content, provider, created_at, conversation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (identity, mode, role, content, provider, now, conversation_id),
        )
        if role == "user":
            current = db.execute(
                "SELECT title FROM chat_conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if current and current["title"] == "New chat":
                compact_title = " ".join(content.split())[:54].strip() or "New chat"
                db.execute(
                    "UPDATE chat_conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (compact_title, now, conversation_id),
                )
            else:
                db.execute(
                    "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
        else:
            db.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return cursor.lastrowid


def list_chat_messages(identity, mode, limit=100, conversation_id=None):
    """Return a user's newest messages in chronological display order."""
    safe_limit = max(1, min(int(limit), 200))
    with connect() as db:
        if conversation_id is None:
            rows = db.execute(
                """SELECT id, role, content, provider, created_at, conversation_id
                   FROM chat_messages
                   WHERE user_identity = ? AND user_mode = ?
                   ORDER BY id DESC LIMIT ?""",
                (identity, mode, safe_limit),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT id, role, content, provider, created_at, conversation_id
                   FROM chat_messages
                   WHERE user_identity = ? AND user_mode = ? AND conversation_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (identity, mode, conversation_id, safe_limit),
            ).fetchall()
    return list(reversed(rows))


def clear_chat_messages(identity, mode, conversation_id=None):
    """Delete one thread, or all history for backward-compatible callers."""
    with connect() as db:
        if conversation_id is None:
            cursor = db.execute(
                "DELETE FROM chat_messages WHERE user_identity = ? AND user_mode = ?",
                (identity, mode),
            )
        else:
            cursor = db.execute(
                """DELETE FROM chat_messages
                   WHERE user_identity = ? AND user_mode = ? AND conversation_id = ?""",
                (identity, mode, conversation_id),
            )
            db.execute(
                """UPDATE chat_conversations SET title = 'New chat', updated_at = ?
                   WHERE id = ? AND user_identity = ? AND user_mode = ?""",
                (_utc_now(), conversation_id, identity, mode),
            )
        return cursor.rowcount


def delete_chat_conversation(conversation_id, identity, mode):
    """Permanently delete one account- and mode-scoped conversation."""
    with connect() as db:
        conversation = db.execute(
            """SELECT id FROM chat_conversations
               WHERE id = ? AND user_identity = ? AND user_mode = ?""",
            (conversation_id, identity, mode),
        ).fetchone()
        if not conversation:
            return False
        db.execute(
            """DELETE FROM chat_messages
               WHERE conversation_id = ? AND user_identity = ? AND user_mode = ?""",
            (conversation_id, identity, mode),
        )
        db.execute(
            """DELETE FROM chat_conversations
               WHERE id = ? AND user_identity = ? AND user_mode = ?""",
            (conversation_id, identity, mode),
        )
    return True


def create_clinical_prediction(doctor_identity, values):
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO clinical_predictions (
                doctor_identity, patient_name, patient_id, tumor, node, metastasis,
                grade, er, pr, her2, ki67, age, predicted_stage, confidence,
                summary, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doctor_identity, values["patient_name"], values["patient_id"], values["tumor"],
             values["node"], values["metastasis"], values["grade"], values.get("er"),
             values.get("pr"), values.get("her2"), values.get("ki67"), values.get("age"),
             values["stage"], values["confidence"], values["summary"], values.get("notes", ""),
             datetime.now(timezone.utc).isoformat()),
        )
        prediction_id = cursor.lastrowid
    return get_clinical_prediction(prediction_id, doctor_identity)


def list_clinical_predictions(doctor_identity):
    with connect() as db:
        return db.execute(
            "SELECT * FROM clinical_predictions WHERE doctor_identity = ? ORDER BY id DESC",
            (doctor_identity,),
        ).fetchall()


def get_clinical_prediction(prediction_id, doctor_identity):
    with connect() as db:
        return db.execute(
            "SELECT * FROM clinical_predictions WHERE id = ? AND doctor_identity = ?",
            (prediction_id, doctor_identity),
        ).fetchone()


init_db()
