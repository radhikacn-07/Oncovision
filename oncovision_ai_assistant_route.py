"""OncoVision breast-health AI assistant routes.

Run this module directly to start the complete OncoVision Flask application::

    python oncovision_ai_assistant_route.py

The assistant is intentionally limited to breast-health education. It does not
provide a diagnosis or replace a clinician.
"""

import os

from flask import Blueprint, current_app, jsonify, request, session
from dotenv import load_dotenv

from utils.clinical import build_chatbot_response
from utils.storage import (
    add_chat_message,
    clear_chat_messages,
    create_chat_conversation,
    delete_chat_conversation,
    get_chat_conversation,
    get_or_create_daily_chat,
    list_all_analyses,
    list_chat_conversations,
    list_chat_messages,
    list_patient_analyses,
)

load_dotenv()


ai_assistant = Blueprint("ai_assistant", __name__)


def _requested_conversation(user, payload=None):
    """Resolve a thread without ever allowing cross-account access."""
    payload = payload or {}
    raw_id = payload.get("conversation_id") or request.args.get("conversation_id")
    if raw_id not in (None, ""):
        try:
            conversation_id = int(raw_id)
        except (TypeError, ValueError):
            return None
        return get_chat_conversation(conversation_id, user["identity"], user["mode"])
    return get_or_create_daily_chat(user["identity"], user["mode"])

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
MANDATORY_NOTE = (
    "⚠️ This is general information, not a medical diagnosis. If your symptoms "
    "are severe, persistent, or worsening, please consult a doctor or oncologist "
    "as soon as possible."
)
SYSTEM_PROMPT = """
You are the Patient Breast Health Assistant inside OncoVision. Speak directly
to a patient using simple, warm, calm and non-alarming language.

SCOPE: Answer only breast-health questions, including breast pain, lumps,
swelling, nipple discharge or inversion, skin changes, redness, itching,
dimpling, tenderness, mammography, biopsy, risk factors, breast cancer stages,
general treatment education, self-examination, cysts, fibroadenomas, mastitis,
hormonal pain, infections, breast-health anxiety, and a saved breast report
when context is supplied.

For in-scope questions: acknowledge the concern; explain possible common causes
without diagnosing; offer only low-risk comfort steps when relevant; clearly
identify red flags; and recommend an appropriate clinical review. Red flags
include a new hard lump, blood-stained discharge, orange-peel dimpling, sudden
nipple inversion, rapid swelling, a non-healing sore, or a lump persisting
beyond one menstrual cycle.

For anything outside breast health, do not answer the question. Politely state
that you specialize only in breast health, suggest a general physician or the
relevant specialist, and invite a breast-health question.

Never diagnose, prescribe medicines or dosages, promise that a symptom is
benign, or give false reassurance. Keep responses concise and patient-friendly.
End every response with this exact note on a new line:
"⚠️ This is general information, not a medical diagnosis. If your symptoms are
severe, persistent, or worsening, please consult a doctor or oncologist as soon
as possible."
""".strip()

# Patient and clinician assistants deliberately use separate instructions.
SYSTEM_PROMPT_PATIENT = SYSTEM_PROMPT
SYSTEM_PROMPT_DOCTOR = """
You are the Clinical Review Assistant inside OncoVision, a breast-cancer
decision-support platform. You are speaking to a licensed clinician
(oncologist, radiologist, surgeon, or general physician), not a patient.
Use clinical, professional register — the way one clinician would brief
another. Do not use patient-comfort language, do not say "your patient
should see a doctor" (the user IS the doctor), and do not soften technical
content.

SCOPE: Support the clinician on breast-cancer-related clinical questions,
including:
- Differential diagnosis considerations for breast symptoms/findings
- Interpretation context for BI-RADS categories, mammographic findings,
  and imaging terminology
- TNM staging, AJCC staging logic, and how T/N/M/Grade/ER/PR/HER2/Ki-67
  map to clinical stage and prognosis categories
- Biomarker significance (ER/PR/HER2/Ki-67) for treatment-pathway context
- General treatment modality overviews (surgical, radiation, systemic
  therapy categories) as reference, not prescriptive dosing
- How to interpret the AI model's output (probability score, Grad-CAM
  attention regions), including its limitations as decision support
- Structuring clinical notes/summaries for a case, if asked
- Pointing clinicians to current NCCN, AJCC, and WHO sources when citable
  clinical guidance is required

STYLE:
- Be direct and technically precise; assume clinical literacy.
- Discuss differentials, staging logic, and biomarker implications in
  appropriate clinical depth.
- Do not make a definitive diagnosis for an unverified case, prescribe exact
  drug dosages, or claim certainty unsupported by the available data. Flag
  where clinical correlation, pathology, or further workup is required.
- The mammogram classifier, Grad-CAM, and stage prediction are decision-support
  aids. Mention this briefly where relevant, not in every paragraph.
- Keep responses focused and efficient. Prefer bullets for considerations.

OUT OF SCOPE: For requests unrelated to breast-cancer clinical practice,
state briefly that you are scoped to breast-cancer clinical support in
OncoVision and invite a related clinical question.

End each response with this exact short line:
"AI-generated support — verify against current guidelines and clinical judgment for patient-specific decisions."
""".strip()

DOCTOR_MANDATORY_NOTE = (
    "AI-generated support — verify against current guidelines and clinical "
    "judgment for patient-specific decisions."
)


def _with_mandatory_note(reply, mode="patient"):
    reply = (reply or "").strip()
    note = DOCTOR_MANDATORY_NOTE if mode == "doctor" else MANDATORY_NOTE
    if note not in reply:
        reply = f"{reply}\n\n{note}"
    return reply


def _contextual_fallback_message(message, history):
    """Attach the prior topic when an offline message is a genuine follow-up."""
    lowered = (message or "").strip().lower()
    if not lowered or len(lowered) > 220:
        return message
    if lowered in {"hi", "hello", "hey", "hii", "hiii", "namaste", "salam", "good morning", "good evening"}:
        return message
    unrelated_terms = (
        "cricket", "football", "match score", "weather", "movie", "song",
        "recipe", "coding", "python code", "stock market", "bitcoin",
        "politics", "election", "travel", "hotel", "game", "kidney",
        "renal", "liver", "heart", "cardiac", "lung", "brain", "diabetes",
        "thyroid", "dental", "tooth", "billing", "invoice",
    )
    if any(term in lowered for term in unrelated_terms):
        return message
    topic_terms = (
        "breast", "mammogram", "mammography", "nipple", "underarm", "axilla",
        "lump", "mass", "pain", "tender", "rash", "itch", "discharge",
        "swelling", "redness", "cancer", "tumor", "tumour", "biopsy",
        "screening", "oncologist", "benign", "malignant", "report", "result",
        "treatment", "chemotherapy", "radiation", "stage", "tnm", "bi-rads",
        "birads", "her2", "ki-67", "grad-cam", "heatmap", "pathology",
    )
    references_prior_topic = any(
        marker in f" {lowered} " for marker in (
            " it ", " this ", " that ", " these ", " those ", " them ",
            " its ", " about it ", " for this ", " that mean ", " uske ",
            " iske ", " ye ", " woh ", " ab kya ",
        )
    )
    # A message naming a complete new breast topic should stand on its own.
    if any(term in lowered for term in topic_terms) and not references_prior_topic:
        return message
    previous_user_message = next(
        (
            item["content"] for item in reversed(history or [])
            if item["role"] == "user"
            and any(term in item["content"].lower() for term in topic_terms)
        ),
        "",
    )
    if not previous_user_message:
        return message
    return f"Previous question: {previous_user_message}\nFollow-up question: {message}"


def _openrouter_reply(message, mode="patient", context=None, history=None):
    """Return an OpenRouter response, or None when it is not configured."""
    if current_app.config.get("TESTING"):
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    system_prompt = SYSTEM_PROMPT_DOCTOR if mode == "doctor" else SYSTEM_PROMPT_PATIENT
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append(
            {
                "role": "system",
                "content": "Saved OncoVision context (do not treat as a diagnosis): " + str(context),
            }
        )
    for item in (history or [])[-20:]:
        role = item["role"] if item["role"] in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": item["content"]})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.4,
        messages=messages,
    )
    return response.choices[0].message.content


@ai_assistant.post("/api/ai-assistant")
def ai_assistant_api():
    """OpenRouter-backed public API described in the integration example."""
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required", "status": "error"}), 400

    user = session.get("user") or {}
    mode = user.get("mode", "patient")
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    conversation = _requested_conversation(user, payload) if user.get("identity") else None
    if user.get("identity") and not conversation:
        return jsonify({"error": "Conversation not found.", "status": "error"}), 404
    history = (
        list_chat_messages(user["identity"], mode, 20, conversation["id"])
        if conversation else []
    )
    try:
        reply = _openrouter_reply(message, mode, context, history)
        provider = "openrouter"
        if reply is None:
            reply = build_chatbot_response(_contextual_fallback_message(message, history), mode, context)
            provider = "local-fallback"
        reply = _with_mandatory_note(reply, mode)
        if user.get("identity"):
            add_chat_message(user["identity"], mode, "user", message, conversation_id=conversation["id"])
            add_chat_message(user["identity"], mode, "assistant", reply, provider, conversation["id"])
        return jsonify({"reply": reply, "response": reply, "status": "ok", "provider": provider,
                        "conversation_id": conversation["id"] if conversation else None})
    except Exception as exc:
        # Keep the patient-facing tool useful during temporary provider errors.
        reply = _with_mandatory_note(
            build_chatbot_response(_contextual_fallback_message(message, history), mode, context), mode
        )
        if user.get("identity"):
            add_chat_message(user["identity"], mode, "user", message, conversation_id=conversation["id"])
            add_chat_message(user["identity"], mode, "assistant", reply, "local-fallback", conversation["id"])
        return jsonify(
            {
                "reply": reply,
                "response": reply,
                "status": "fallback",
                "provider": "local-fallback",
                "detail": str(exc),
                "conversation_id": conversation["id"] if conversation else None,
            }
        )


@ai_assistant.post("/api/chatbot")
def chatbot_api():
    """Answer a breast-health question using the signed-in user's context."""
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to use the assistant."}), 401

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a breast-health question."}), 400

    mode = user.get("mode", "patient")
    conversation = _requested_conversation(user, payload)
    if not conversation:
        return jsonify({"error": "This conversation was not found."}), 404
    if mode == "patient":
        analyses = list_patient_analyses(user["identity"])
    else:
        analyses = list_all_analyses()

    latest = dict(analyses[0]) if analyses else {}
    supplied_context = payload.get("context")
    if not isinstance(supplied_context, dict):
        supplied_context = {}
    context = {**supplied_context, **latest}
    history = list_chat_messages(user["identity"], mode, 20, conversation["id"])

    try:
        response = _openrouter_reply(message, mode, context, history)
    except Exception:
        response = None
    provider = "openrouter" if response else "local-fallback"
    if not response:
        response = build_chatbot_response(_contextual_fallback_message(message, history), mode, context)
    response = _with_mandatory_note(response, mode)
    add_chat_message(user["identity"], mode, "user", message, conversation_id=conversation["id"])
    add_chat_message(user["identity"], mode, "assistant", response, provider, conversation["id"])
    return jsonify(
        {
            "response": response,
            "reply": response,
            "status": "ok",
            "provider": provider,
            "mode": mode,
            "context_report_id": latest.get("report_id"),
            "conversation_id": conversation["id"],
        }
    )


@ai_assistant.get("/api/chat-conversations")
def chat_conversations_api():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to view conversations."}), 401
    active = get_or_create_daily_chat(user["identity"], user["mode"])
    conversations = [dict(row) for row in list_chat_conversations(user["identity"], user["mode"])]
    return jsonify({
        "status": "ok",
        "active_conversation_id": active["id"],
        "conversations": conversations,
    })


@ai_assistant.post("/api/chat-conversations")
def create_chat_conversation_api():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to start a conversation."}), 401
    conversation_id = create_chat_conversation(user["identity"], user["mode"])
    conversation = dict(get_chat_conversation(conversation_id, user["identity"], user["mode"]))
    return jsonify({"status": "ok", "conversation": conversation}), 201


@ai_assistant.get("/api/chat-conversations/<int:conversation_id>/messages")
def chat_conversation_messages_api(conversation_id):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to view conversations."}), 401
    conversation = get_chat_conversation(conversation_id, user["identity"], user["mode"])
    if not conversation:
        return jsonify({"error": "Conversation not found."}), 404
    messages = [
        dict(row) for row in list_chat_messages(
            user["identity"], user["mode"], 200, conversation_id
        )
    ]
    return jsonify({"conversation": dict(conversation), "messages": messages, "count": len(messages)})


@ai_assistant.delete("/api/chat-conversations/<int:conversation_id>")
def delete_chat_conversation_api(conversation_id):
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to delete a conversation."}), 401
    deleted = delete_chat_conversation(
        conversation_id, user["identity"], user["mode"]
    )
    if not deleted:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"status": "ok", "deleted_conversation_id": conversation_id})


@ai_assistant.get("/api/chat-history")
def chat_history_api():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to view chat history."}), 401
    conversation = _requested_conversation(user)
    if not conversation:
        return jsonify({"error": "Conversation not found."}), 404
    messages = [dict(row) for row in list_chat_messages(
        user["identity"], user["mode"], 100, conversation["id"]
    )]
    return jsonify({"messages": messages, "count": len(messages),
                    "conversation_id": conversation["id"]})


@ai_assistant.delete("/api/chat-history")
def clear_chat_history_api():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Please log in to clear chat history."}), 401
    raw_id = request.args.get("conversation_id")
    if raw_id:
        try:
            conversation_id = int(raw_id)
        except ValueError:
            return jsonify({"error": "Invalid conversation id."}), 400
        if not get_chat_conversation(conversation_id, user["identity"], user["mode"]):
            return jsonify({"error": "Conversation not found."}), 404
        deleted = clear_chat_messages(user["identity"], user["mode"], conversation_id)
    else:
        # Preserve the original API contract for existing clients.
        deleted = clear_chat_messages(user["identity"], user["mode"])
    return jsonify({"status": "ok", "deleted": deleted})


if __name__ == "__main__":
    # Import after defining the blueprint to avoid a circular import while the
    # main application registers this module.
    from app import app

    app.run(host="0.0.0.0", port=5000, debug=False)
