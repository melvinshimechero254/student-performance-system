"""SQLite-backed user accounts (register / verify / password reset)."""
from __future__ import annotations

import secrets
import re
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone

from werkzeug.security import check_password_hash, generate_password_hash

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

UTC = timezone.utc


def ensure_schema(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                api_key TEXT,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(conn, "users", "is_admin", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "users", "api_key", "TEXT")
        _ensure_column(conn, "users", "failed_attempts", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "users", "locked_until", "TEXT")
        _ensure_column(conn, "users", "email", "TEXT")
        _ensure_column(conn, "users", "is_approved", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(conn, "users", "approved_by", "INTEGER")
        _ensure_column(conn, "users", "approved_at", "TEXT")
        _ensure_column(conn, "users", "approval_note", "TEXT")
        _ensure_column(conn, "users", "reset_token_hash", "TEXT")
        _ensure_column(conn, "users", "reset_token_expires_at", "TEXT")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _validate_username(username: str) -> tuple[bool, str]:
    u = (username or "").strip()
    if not u:
        return False, "Username is required."
    if not _USERNAME_RE.match(u):
        return False, "Username must be 3–32 characters (letters, numbers, underscore)."
    return True, u


def _validate_password(password: str) -> tuple[bool, str]:
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    # Basic strength signal (kept lightweight, no extra deps).
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(not c.isalnum() for c in password)
    if sum([has_lower, has_upper, has_digit, has_symbol]) < 3:
        return False, "Password must include at least 3 of: lowercase, uppercase, number, symbol."
    return True, password  # echo normalized input (no trimming for passwords)


def _normalize_email(email: str) -> tuple[bool, str]:
    normalized = (email or "").strip().lower()
    if not normalized:
        return False, "Email is required."
    if not _EMAIL_RE.match(normalized):
        return False, "Email format is invalid."
    return True, normalized


def register_user(
    db_path: str, username: str, password: str, email: str
) -> tuple[bool, str, int | None, str | None]:
    ok, msg = _validate_username(username)
    if not ok:
        return False, msg, None, None
    username = msg
    ok, pwd_ok = _validate_password(password)
    if not ok:
        return False, pwd_ok, None, None
    password = pwd_ok
    ok, email_ok = _normalize_email(email)
    if not ok:
        return False, email_ok, None, None
    email = email_ok
    pwd_hash = generate_password_hash(password)
    api_key = secrets.token_urlsafe(32)
    try:
        with sqlite3.connect(db_path) as conn:
            email_taken = conn.execute(
                "SELECT 1 FROM users WHERE lower(email) = lower(?) LIMIT 1",
                (email,),
            ).fetchone()
            if email_taken:
                return False, "That email is already registered.", None, None
            existing_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            is_admin = 1 if int(existing_users or 0) == 0 else 0
            is_approved = 1 if is_admin else 0
            cur = conn.execute(
                """
                INSERT INTO users (username, email, password_hash, api_key, is_admin, is_approved, approved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    pwd_hash,
                    api_key,
                    is_admin,
                    is_approved,
                    datetime.now(UTC).isoformat() if is_approved else None,
                ),
            )
            conn.commit()
            return True, "Registered.", cur.lastrowid, username
    except sqlite3.IntegrityError:
        return False, "That username is already taken.", None, None


def verify_user(db_path: str, identifier: str, password: str) -> tuple[int, str, bool] | None:
    ident = (identifier or "").strip()
    if not ident:
        return None
    if not password:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, password_hash, is_admin, failed_attempts, locked_until
            FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
            """,
            (ident, ident),
        ).fetchone()
        if not row:
            return None
        user_id, canonical_name, pwd_hash, is_admin, failed_attempts, locked_until = row

        now = datetime.now(UTC)
        if locked_until:
            try:
                until = datetime.fromisoformat(locked_until)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=UTC)
            except ValueError:
                until = None
            if until and until > now:
                return None

        if check_password_hash(pwd_hash, password):
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?",
                (user_id,),
            )
            conn.commit()
            return user_id, canonical_name, bool(is_admin)

        # Failed attempt: increment & maybe lock.
        failed_attempts = int(failed_attempts or 0) + 1
        lock_until = None
        if failed_attempts >= 5:
            lock_until = (now + timedelta(minutes=10)).isoformat()
        conn.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            (failed_attempts, lock_until, user_id),
        )
        conn.commit()
        return None


def get_user_by_id(db_path: str, user_id: int) -> tuple[int, str, bool] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, is_admin FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    if not row:
        return None
    uid, uname, is_admin = row
    return int(uid), str(uname), bool(is_admin)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_user_approved(db_path: str, user_id: int) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT is_approved FROM users WHERE id = ?", (int(user_id),)).fetchone()
    if not row:
        return False
    return bool(int(row[0]))


def change_password(db_path: str, user_id: int, current_password: str, new_password: str) -> tuple[bool, str]:
    ok, pwd_ok = _validate_password(new_password)
    if not ok:
        return False, pwd_ok
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if not row:
            return False, "User not found."
        if not check_password_hash(row[0], current_password or ""):
            return False, "Current password is incorrect."
        if check_password_hash(row[0], pwd_ok):
            return False, "New password must be different from current password."
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?, failed_attempts = 0, locked_until = NULL
            WHERE id = ?
            """,
            (generate_password_hash(pwd_ok), int(user_id)),
        )
        conn.commit()
    return True, "Password updated."


def request_password_reset(db_path: str, username_or_email: str) -> tuple[str | None, str | None]:
    candidate = (username_or_email or "").strip()
    if not candidate:
        return None, None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, email FROM users
            WHERE lower(username) = lower(?) OR (email IS NOT NULL AND lower(email) = lower(?))
            """,
            (candidate, candidate),
        ).fetchone()
        if not row:
            return None, None
        user_id, email = row
        if not email:
            return None, None
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
        conn.execute(
            """
            UPDATE users
            SET reset_token_hash = ?, reset_token_expires_at = ?
            WHERE id = ?
            """,
            (_hash_token(token), expires_at, int(user_id)),
        )
        conn.commit()
        return token, str(email)


def reset_password_with_token(db_path: str, token: str, new_password: str) -> tuple[bool, str]:
    if not token:
        return False, "Invalid reset token."
    ok, pwd_ok = _validate_password(new_password)
    if not ok:
        return False, pwd_ok

    token_hash = _hash_token(token)
    now = datetime.now(UTC)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, reset_token_expires_at
            FROM users
            WHERE reset_token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if not row:
            return False, "Invalid or expired reset token."
        user_id, expires_at = row
        if not expires_at:
            return False, "Invalid or expired reset token."
        try:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
        except ValueError:
            return False, "Invalid or expired reset token."
        if expiry < now:
            return False, "Invalid or expired reset token."

        conn.execute(
            """
            UPDATE users
            SET password_hash = ?,
                failed_attempts = 0,
                locked_until = NULL,
                reset_token_hash = NULL,
                reset_token_expires_at = NULL
            WHERE id = ?
            """,
            (generate_password_hash(pwd_ok), int(user_id)),
        )
        conn.commit()
        return True, "Password reset successful."
