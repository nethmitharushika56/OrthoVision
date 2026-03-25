import os
from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient


DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
MONGODB_URI = os.environ.get("MONGODB_URI", DEFAULT_MONGODB_URI)
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "orthovision")

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
_db = _client[MONGODB_DB_NAME]
_users = _db["users"]
_analyses = _db["analyses"]


def _normalize_user_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {
        "id": str(doc.get("_id")),
        "email": doc.get("email"),
        "name": doc.get("name"),
        "phone": doc.get("phone"),
        "institution": doc.get("institution"),
        "profileImage": doc.get("profileImage"),
    }


def _normalize_analysis_doc(doc: dict[str, Any]) -> dict[str, Any]:
    created_at = doc.get("createdAt")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    return {
        "id": str(doc.get("_id")),
        "userEmail": doc.get("userEmail"),
        "prediction": doc.get("prediction"),
        "isFractured": doc.get("isFractured"),
        "fractureType": doc.get("fractureType"),
        "confidence": doc.get("confidence"),
        "bonePart": doc.get("bonePart"),
        "createdAt": created_at,
    }


def init_db() -> None:
    # Verify connectivity and ensure indexes for common lookups.
    _client.admin.command("ping")
    _users.create_index([("email", ASCENDING)], unique=True)
    _analyses.create_index([("userEmail", ASCENDING), ("createdAt", DESCENDING)])


def upsert_user(payload: dict) -> dict:
    email = str(payload.get("email", "")).strip().lower()
    name = str(payload.get("name", "")).strip()
    if not email or not name:
        raise ValueError("Both 'email' and 'name' are required")

    now = datetime.utcnow()
    update = {
        "$set": {
            "email": email,
            "name": name,
            "phone": payload.get("phone"),
            "institution": payload.get("institution"),
            "profileImage": payload.get("profileImage"),
            "updatedAt": now,
        },
        "$setOnInsert": {"createdAt": now},
    }
    _users.update_one({"email": email}, update, upsert=True)
    doc = _users.find_one({"email": email})
    user = _normalize_user_doc(doc)
    return user or {}


def get_user_by_email(email: str) -> dict | None:
    value = str(email or "").strip().lower()
    if not value:
        return None
    return _normalize_user_doc(_users.find_one({"email": value}))


def create_analysis(payload: dict) -> dict:
    doc = {
        "userEmail": (payload.get("userEmail") or "").strip().lower() or None,
        "prediction": payload.get("prediction"),
        "isFractured": payload.get("isFractured"),
        "fractureType": payload.get("fractureType"),
        "confidence": payload.get("confidence"),
        "bonePart": payload.get("bonePart"),
        "resultJson": payload.get("resultJson"),
        "createdAt": datetime.utcnow(),
    }
    result = _analyses.insert_one(doc)
    saved = _analyses.find_one({"_id": ObjectId(result.inserted_id)})
    return _normalize_analysis_doc(saved or doc)


def list_analyses(user_email: str | None = None, limit: int = 100) -> list[dict]:
    query: dict[str, Any] = {}
    if user_email:
        query["userEmail"] = user_email.strip().lower()

    cursor = _analyses.find(query).sort("createdAt", DESCENDING).limit(max(1, min(limit, 500)))
    return [_normalize_analysis_doc(item) for item in cursor]


def check_db_health() -> bool:
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False
