"""SQLite-backed user accounts with roles, courses, students, and interventions."""
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

VALID_ROLES = ("admin", "lecturer", "student")


def ensure_schema(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")

        # ── users ──────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_approved INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                api_key TEXT,
                reg_number TEXT,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                approved_by INTEGER,
                approved_at TEXT,
                approval_note TEXT,
                reset_token_hash TEXT,
                reset_token_expires_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # migrate existing DBs
        for col, ddl in [
            ("role", "TEXT NOT NULL DEFAULT 'student'"),
            ("is_active", "INTEGER NOT NULL DEFAULT 1"),
            ("is_approved", "INTEGER NOT NULL DEFAULT 0"),
            ("approved_by", "INTEGER"),
            ("approved_at", "TEXT"),
            ("approval_note", "TEXT"),
            ("reset_token_hash", "TEXT"),
            ("reset_token_expires_at", "TEXT"),
            ("api_key", "TEXT"),
            ("reg_number", "TEXT"),
            ("email", "TEXT"),
        ]:
            _ensure_column(conn, "users", col, ddl)

        # ── courses ────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT UNIQUE NOT NULL,
                lecturer_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lecturer_id) REFERENCES users(id)
            )
        """)

        # ── student_courses ────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS student_courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                risk_level TEXT NOT NULL DEFAULT 'none',
                risk_updated_at TEXT,
                acknowledged_at TEXT,
                enrolled_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, course_id),
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (course_id) REFERENCES courses(id)
            )
        """)

        # ── cat_scores ─────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                cat_number INTEGER NOT NULL,
                score REAL NOT NULL,
                entered_by INTEGER,
                entered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, course_id, cat_number),
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (course_id) REFERENCES courses(id)
            )
        """)

        # ── attendance_records ─────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                attendance_pct REAL NOT NULL,
                recorded_by INTEGER,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (course_id) REFERENCES courses(id)
            )
        """)

        # ── assignment_scores ──────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assignment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                score_pct REAL NOT NULL,
                submitted_pct REAL NOT NULL DEFAULT 100,
                entered_by INTEGER,
                entered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (course_id) REFERENCES courses(id)
            )
        """)

        # ── interventions ──────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                lecturer_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT,
                intervened_at TEXT DEFAULT CURRENT_TIMESTAMP,
                follow_up_needed INTEGER DEFAULT 0,
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (course_id) REFERENCES courses(id),
                FOREIGN KEY (lecturer_id) REFERENCES users(id)
            )
        """)

        # ── predictions ────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                student_id INTEGER,
                course_id INTEGER,
                mode TEXT NOT NULL,
                attendance REAL,
                cat1_score REAL,
                cat2_score REAL,
                cat_average REAL,
                assignment_score REAL,
                assignments_submitted REAL,
                predicted_final_score REAL,
                prediction TEXT NOT NULL,
                confidence REAL,
                payload_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


def _ensure_column(conn, table, column, ddl):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# ── Validation helpers ──────────────────────────────────────────────────────

def _validate_username(username):
    u = (username or "").strip()
    if not u:
        return False, "Username is required."
    if not _USERNAME_RE.match(u):
        return False, "Username must be 3–32 characters (letters, numbers, underscore)."
    return True, u


def _validate_password(password):
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(not c.isalnum() for c in password)
    if sum([has_lower, has_upper, has_digit, has_symbol]) < 3:
        return False, "Password must include at least 3 of: lowercase, uppercase, number, symbol."
    return True, password


def _normalize_email(email):
    normalized = (email or "").strip().lower()
    if not normalized:
        return False, "Email is required."
    if not _EMAIL_RE.match(normalized):
        return False, "Email format is invalid."
    return True, normalized


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── User registration ───────────────────────────────────────────────────────

def register_user(db_path, username, password, email, role="student", reg_number=None):
    ok, msg = _validate_username(username)
    if not ok:
        return False, msg, None, None
    username = msg
    ok, pwd_ok = _validate_password(password)
    if not ok:
        return False, pwd_ok, None, None
    ok, email_ok = _normalize_email(email)
    if not ok:
        return False, email_ok, None, None
    email = email_ok
    if role not in VALID_ROLES:
        role = "student"
    pwd_hash = generate_password_hash(password)
    api_key = secrets.token_urlsafe(32)
    try:
        with sqlite3.connect(db_path) as conn:
            if conn.execute("SELECT 1 FROM users WHERE lower(email)=lower(?) LIMIT 1", (email,)).fetchone():
                return False, "That email is already registered.", None, None
            existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            # First user ever → admin, auto-approved
            if int(existing or 0) == 0:
                is_admin = 1
                is_approved = 1
                role = "admin"
            else:
                is_admin = 1 if role == "admin" else 0
                is_approved = 0  # all non-first users wait for approval
            cur = conn.execute("""
                INSERT INTO users
                  (username, email, password_hash, api_key, role, is_admin, is_approved,
                   approved_at, reg_number)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                username, email, pwd_hash, api_key, role, is_admin, is_approved,
                datetime.now(UTC).isoformat() if is_approved else None,
                reg_number,
            ))
            conn.commit()
            return True, "Registered.", cur.lastrowid, username
    except sqlite3.IntegrityError:
        return False, "That username is already taken.", None, None


# ── Authentication ──────────────────────────────────────────────────────────

def verify_user(db_path, identifier, password):
    ident = (identifier or "").strip()
    if not ident or not password:
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT id, username, password_hash, is_admin, role,
                   failed_attempts, locked_until
            FROM users
            WHERE lower(username)=lower(?) OR lower(email)=lower(?)
            LIMIT 1
        """, (ident, ident)).fetchone()
        if not row:
            return None
        now = datetime.now(UTC)
        if row["locked_until"]:
            try:
                until = datetime.fromisoformat(row["locked_until"])
                if until.tzinfo is None:
                    until = until.replace(tzinfo=UTC)
                if until > now:
                    return None
            except ValueError:
                pass
        if check_password_hash(row["password_hash"], password):
            conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?", (row["id"],))
            conn.commit()
            return row["id"], row["username"], bool(row["is_admin"]), row["role"]
        # bad password
        attempts = int(row["failed_attempts"] or 0) + 1
        lock = (now + timedelta(minutes=10)).isoformat() if attempts >= 5 else None
        conn.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                     (attempts, lock, row["id"]))
        conn.commit()
        return None


def get_user_by_id(db_path, user_id):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, is_admin, role FROM users WHERE id=?", (int(user_id),)
        ).fetchone()
    if not row:
        return None
    return int(row["id"]), str(row["username"]), bool(row["is_admin"]), str(row["role"])


def is_user_approved(db_path, user_id):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT is_approved FROM users WHERE id=?", (int(user_id),)).fetchone()
    return bool(int(row[0])) if row else False


def change_password(db_path, user_id, current_password, new_password):
    ok, pwd_ok = _validate_password(new_password)
    if not ok:
        return False, pwd_ok
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id=?", (int(user_id),)).fetchone()
        if not row:
            return False, "User not found."
        if not check_password_hash(row[0], current_password or ""):
            return False, "Current password is incorrect."
        if check_password_hash(row[0], pwd_ok):
            return False, "New password must be different from current password."
        conn.execute("""
            UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE id=?
        """, (generate_password_hash(pwd_ok), int(user_id)))
        conn.commit()
    return True, "Password updated."


def request_password_reset(db_path, username_or_email):
    candidate = (username_or_email or "").strip()
    if not candidate:
        return None, None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT id, email FROM users
            WHERE lower(username)=lower(?) OR (email IS NOT NULL AND lower(email)=lower(?))
        """, (candidate, candidate)).fetchone()
        if not row or not row[1]:
            return None, None
        user_id, email = row
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
        conn.execute("""
            UPDATE users SET reset_token_hash=?, reset_token_expires_at=? WHERE id=?
        """, (_hash_token(token), expires_at, int(user_id)))
        conn.commit()
        return token, str(email)


def reset_password_with_token(db_path, token, new_password):
    if not token:
        return False, "Invalid reset token."
    ok, pwd_ok = _validate_password(new_password)
    if not ok:
        return False, pwd_ok
    token_hash = _hash_token(token)
    now = datetime.now(UTC)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("""
            SELECT id, reset_token_expires_at FROM users WHERE reset_token_hash=?
        """, (token_hash,)).fetchone()
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
        conn.execute("""
            UPDATE users
            SET password_hash=?, failed_attempts=0, locked_until=NULL,
                reset_token_hash=NULL, reset_token_expires_at=NULL
            WHERE id=?
        """, (generate_password_hash(pwd_ok), int(user_id)))
        conn.commit()
    return True, "Password reset successful."


# ── Course helpers ──────────────────────────────────────────────────────────

def get_courses_for_lecturer(db_path, lecturer_id):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("""
            SELECT id, name, code FROM courses WHERE lecturer_id=? ORDER BY name
        """, (int(lecturer_id),)).fetchall()


def get_at_risk_students_for_lecturer(db_path, lecturer_id):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("""
            SELECT u.id, u.username, u.email, u.reg_number,
                   c.name AS course_name, c.code AS course_code, c.id AS course_id,
                   sc.risk_level, sc.risk_updated_at, sc.acknowledged_at
            FROM student_courses sc
            JOIN users u ON u.id = sc.student_id
            JOIN courses c ON c.id = sc.course_id
            WHERE c.lecturer_id = ?
              AND sc.risk_level IN ('yellow','orange','red')
            ORDER BY
              CASE sc.risk_level WHEN 'red' THEN 1 WHEN 'orange' THEN 2 ELSE 3 END,
              sc.risk_updated_at DESC
        """, (int(lecturer_id),)).fetchall()


def get_all_at_risk_students(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("""
            SELECT u.id AS student_id, u.username, u.reg_number,
                   c.name AS course_name, c.code AS course_code,
                   lec.username AS lecturer_name,
                   sc.risk_level, sc.risk_updated_at,
                   (SELECT COUNT(*) FROM interventions i
                    WHERE i.student_id=u.id AND i.course_id=c.id) AS intervention_count
            FROM student_courses sc
            JOIN users u ON u.id = sc.student_id
            JOIN courses c ON c.id = sc.course_id
            LEFT JOIN users lec ON lec.id = c.lecturer_id
            WHERE sc.risk_level IN ('yellow','orange','red')
            ORDER BY
              CASE sc.risk_level WHEN 'red' THEN 1 WHEN 'orange' THEN 2 ELSE 3 END
        """).fetchall()
