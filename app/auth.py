"""
API Key authentication for MnemonicAi.
Handles key generation, validation, storage, and usage tracking.
"""

import secrets
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


# ─── Database Setup ───────────────────────────────────────────────

DB_PATH = "mnemonicai_data/auth.db"

def init_auth_db():
    """Call this once on startup to create the auth tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            stripe_customer_id TEXT,
            created_at TEXT NOT NULL,
            tier TEXT DEFAULT 'free',
            status TEXT DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT DEFAULT 'default',
            created_at TEXT NOT NULL,
            last_used TEXT,
            status TEXT DEFAULT 'active',
            monthly_limit INTEGER DEFAULT 100,
            requests_this_month INTEGER DEFAULT 0,
            reset_date TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_log_key ON usage_log(api_key_id);
    """)
    conn.commit()
    conn.close()


# ─── Key Generation ───────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.
    Returns: (full_key, key_hash, key_prefix)
    
    The full key is shown ONCE to the user.
    We store only the hash (like a password).
    The prefix is stored for display purposes (e.g., "mn_live_a3f2...").
    """
    raw_token = secrets.token_urlsafe(32)
    full_key = f"mn_live_{raw_token}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:12] + "..."
    return full_key, key_hash, key_prefix


def hash_api_key(key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


# ─── Key Validation ───────────────────────────────────────────────

def validate_api_key(raw_key: str) -> Optional[dict]:
    """
    Validate an API key and return the associated user info.
    Returns None if invalid/inactive.
    """
    if not raw_key:
        return None

    key_hash = hash_api_key(raw_key)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT ak.id, ak.user_id, ak.status, ak.monthly_limit,
               ak.requests_this_month, ak.reset_date, 
               u.tier, u.status as user_status, u.email
        FROM api_keys ak
        JOIN users u ON ak.user_id = u.id
        WHERE ak.key_hash = ? AND ak.status = 'active' AND u.status = 'active'
        """,
        (key_hash,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    # Check monthly limit
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if row["reset_date"] < today:
        # Reset counter for new month
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE api_keys SET requests_this_month = 0, reset_date = ? WHERE id = ?",
            (today, row["id"])
        )
        conn.commit()
        conn.close()
        current_count = 0
    else:
        current_count = row["requests_this_month"]

    if current_count >= row["monthly_limit"]:
        return None  # Rate limited

    return {
        "key_id": row["id"],
        "user_id": row["user_id"],
        "email": row["email"],
        "tier": row["tier"],
        "requests_this_month": current_count,
        "monthly_limit": row["monthly_limit"],
    }


def log_usage(key_id: str, endpoint: str, tokens: int = 0):
    """Log an API request for usage tracking."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO usage_log (api_key_id, endpoint, timestamp, tokens_used) VALUES (?, ?, ?, ?)",
        (key_id, endpoint, now, tokens)
    )
    conn.execute(
        "UPDATE api_keys SET requests_this_month = requests_this_month + 1, last_used = ? WHERE id = ?",
        (now, key_id)
    )
    conn.commit()
    conn.close()


# ─── FastAPI Middleware ───────────────────────────────────────────

async def auth_middleware(request: Request, call_next):
    """
    FastAPI middleware that validates API keys on every request.
    Skips auth for health checks, docs, and the checkout/signup endpoints.
    """
    # Paths that don't need auth
    PUBLIC_PATHS = {
        "/", "/health", "/docs", "/openapi.json", "/redoc",
        "/signup", "/create-checkout-session", "/session-status",
        "/stripe-webhook", "/complete.html", "/checkout.html"
    }

    # Allow public paths
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # Allow OPTIONS (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)

    # Extract API key from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    elif request.headers.get("X-API-Key"):
        api_key = request.headers.get("X-API-Key")
    else:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Missing API key. Pass it as 'Authorization: Bearer mn_live_...' or 'X-API-Key: mn_live_...'", "type": "auth_error"}}
        )

    # Validate
    user_info = validate_api_key(api_key)
    if not user_info:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid or inactive API key. Check your key at https://mnemonicai.org/dashboard", "type": "auth_error"}}
        )

    # Attach user info to request state so endpoints can use it
    request.state.user_id = user_info["user_id"]
    request.state.user_email = user_info["email"]
    request.state.user_tier = user_info["tier"]
    request.state.key_id = user_info["key_id"]

    # Process the request
    response = await call_next(request)

    # Log usage after the request completes
    log_usage(user_info["key_id"], request.url.path)

    return response
