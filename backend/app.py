from __future__ import annotations

import os
import secrets
import sqlite3
import time
import io
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
import logging

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, jsonify, redirect, render_template,
    request, send_file, session, url_for,
)

from auth_service import (
    change_password, ensure_schema, get_user_by_id,
    is_user_approved, register_user, request_password_reset,
    reset_password_with_token, verify_user,
    get_courses_for_lecturer, get_at_risk_students_for_lecturer,
    get_all_at_risk_students,
)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "student_performance_model.pkl")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "dev-secret-change-me"
app.config.setdefault("AUTH_DATABASE", os.path.join(BASE_DIR, "users.db"))
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
app.config.setdefault("SESSION_COOKIE_SECURE", False)
app.config.setdefault("PERMANENT_SESSION_LIFETIME", 60 * 60 * 8)

logger = logging.getLogger("student_dashboard")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(os.path.join(LOGS_DIR, "app.log"), maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# ── Load model bundle ───────────────────────────────────────────────────────
_bundle = joblib.load(MODEL_PATH)
if isinstance(_bundle, dict):
    _clf = _bundle["classifier"]
    _reg = _bundle.get("regressor")
    MODEL_FEATURES = _bundle.get("features", ["Attendance", "CAT1_Score", "CAT2_Score",
                                               "CAT_Average", "Assignment_Score", "Assignments_Submitted"])
else:
    # legacy single model
    _clf = _bundle
    _reg = None
    MODEL_FEATURES = ["Attendance", "CAT_Score", "Assignment_Score", "Final_Exam"]

UTC = timezone.utc
ATTENDANCE_MIN = 70.0  # minimum for exam eligibility


# ── DB helpers ──────────────────────────────────────────────────────────────

def _db_path():
    return app.config["AUTH_DATABASE"]


def _connect_db():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


@app.before_request
def _boot_db():
    ensure_schema(_db_path())


# ── Current user ────────────────────────────────────────────────────────────

def _current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    result = get_user_by_id(_db_path(), int(user_id))
    if not result:
        return None
    uid, uname, is_admin, role = result
    with _connect_db() as conn:
        row = conn.execute("SELECT is_active FROM users WHERE id=?", (uid,)).fetchone()
    if row and int(row["is_active"]) == 0:
        session.clear()
        return None
    return {"id": uid, "username": uname, "is_admin": bool(is_admin), "role": role}


def _log_event(event, **fields):
    logger.info(str({"event": event, **fields}))


# ── CSRF ────────────────────────────────────────────────────────────────────

def _csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def _require_csrf():
    if app.config.get("TESTING"):
        return True
    sent = request.form.get("csrf_token", "")
    return bool(sent) and secrets.compare_digest(sent, session.get("csrf_token", ""))


@app.context_processor
def _inject_globals():
    u = _current_user() or {}
    return {
        "csrf_token": _csrf_token,
        "current_username": u.get("username"),
        "current_is_admin": bool(u.get("is_admin", False)),
        "current_role": u.get("role", "student"),
    }


# ── Rate limiting ───────────────────────────────────────────────────────────

_RATE: dict = {}


def _rate_limit(key, limit, window_s):
    if app.config.get("TESTING"):
        return True
    now = time.time()
    bucket = [t for t in _RATE.get(key, []) if now - t < window_s]
    if len(bucket) >= limit:
        _RATE[key] = bucket
        return False
    bucket.append(now)
    _RATE[key] = bucket
    return True


# ── Decorators ──────────────────────────────────────────────────────────────

def login_required_page(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _current_user():
            return view(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return wrapped


def login_required_api(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _current_user():
            return view(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        u = _current_user()
        if u and u["is_admin"]:
            return view(*args, **kwargs)
        return render_template("forbidden.html"), 403
    return wrapped


def lecturer_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        u = _current_user()
        if u and u["role"] in ("lecturer", "admin"):
            return view(*args, **kwargs)
        return render_template("forbidden.html"), 403
    return wrapped


# ── Email (Resend) ──────────────────────────────────────────────────────────

def _send_email(to: str, subject: str, body: str) -> bool:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        logger.error("RESEND_API_KEY not set")
        return False
    try:
        import resend
        resend.api_key = api_key
        resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": "melvinshimechero@gmail.com",  # demo: route all to admin
            "subject": subject,
            "text": body,
        })
        return True
    except Exception as exc:
        logger.error("Resend email failed: %s", exc)
        return False


def _send_password_reset_email(recipient_email, reset_url):
    return _send_email(
        recipient_email,
        "Password reset instructions",
        f"A password reset was requested.\n\nReset link:\n{reset_url}\n\n"
        "Ignore this if you did not request it.",
    )


def _notify_lecturer_at_risk(lecturer_email, lecturer_name, student_name, course_name, risk_level):
    emoji = {"yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(risk_level, "⚠️")
    return _send_email(
        lecturer_email,
        f"{emoji} At-Risk Alert: {student_name} in {course_name}",
        f"Dear {lecturer_name},\n\n"
        f"Student {student_name} has been flagged as {risk_level.upper()} risk "
        f"in your course '{course_name}'.\n\n"
        f"Please log in to review their performance and record an intervention:\n"
        f"{url_for('lecturer_dashboard', _external=True)}\n\n"
        f"Early intervention significantly improves student outcomes.",
    )


# ── ML helpers ──────────────────────────────────────────────────────────────

def _build_features(attendance, cat1, cat2, assignment_score, assignments_submitted):
    cat_avg = round((cat1 + cat2) / 2, 2)
    return {
        "Attendance": attendance,
        "CAT1_Score": cat1,
        "CAT2_Score": cat2,
        "CAT_Average": cat_avg,
        "Assignment_Score": assignment_score,
        "Assignments_Submitted": assignments_submitted,
    }


def _predict(features: dict):
    arr = np.array([[features[f] for f in MODEL_FEATURES]])
    label = str(_clf.predict(arr)[0])
    confidence = round(float(max(_clf.predict_proba(arr)[0])) * 100, 2)
    predicted_final = None
    if _reg is not None:
        predicted_final = round(float(_reg.predict(arr)[0]), 1)
        predicted_final = max(0.0, min(70.0, predicted_final))
    return label, confidence, predicted_final


def _feature_importance():
    vals = getattr(_clf, "feature_importances_", None)
    if vals is None:
        return {}
    return {k: round(float(v), 4) for k, v in zip(MODEL_FEATURES, vals)}


# ── At-risk engine ──────────────────────────────────────────────────────────

def _compute_risk_level(attendance, cat1, cat2, assignment_score, label):
    """
    Three-tier progressive risk system:
      yellow  — early warning (attendance dip, week 1-4)
      orange  — mid-semester concern (CAT1 weak)
      red     — urgent (ML predicts Fail/At Risk + grade evidence)
      none    — on track
    """
    cat_avg = (cat1 + cat2) / 2

    if label in ("Fail", "At Risk") and (attendance < 70 or cat_avg < 9):
        return "red"
    if label in ("Fail", "At Risk"):
        return "orange"
    if attendance < 75 or cat_avg < 12:
        return "yellow"
    return "none"


def _update_student_risk(db_path, student_id, course_id, risk_level):
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            UPDATE student_courses
            SET risk_level=?, risk_updated_at=?
            WHERE student_id=? AND course_id=?
        """, (risk_level, now, int(student_id), int(course_id)))
        conn.commit()


# ── Auth routes ─────────────────────────────────────────────────────────────

def _safe_next(url):
    if not url or not isinstance(url, str):
        return None
    if not url.startswith("/") or url.startswith("//"):
        return None
    return url


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(_safe_next(request.args.get("next")) or url_for("home"))
    if request.method == "POST":
        if not _require_csrf():
            return render_template("login.html", error="Security check failed."), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "login"), 10, 60):
            return render_template("login.html", error="Too many attempts. Wait a minute."), 429
        identifier = request.form.get("identifier", "") or request.form.get("username", "")
        password = request.form.get("password", "")
        selected_role = request.form.get("role", "")
        next_url = request.form.get("next", "")
        result = verify_user(_db_path(), identifier, password)
        if result is None:
            return render_template("login.html", error="Invalid username or password.", next_url=next_url)
        user_id, canonical_name, is_admin, role = result
        if not is_user_approved(_db_path(), int(user_id)):
            return render_template("login.html",
                                   error="Your account is pending admin approval.", next_url=next_url), 403
        # Role mismatch check — UI role selector vs actual DB role
        if selected_role and selected_role != role:
            return render_template("login.html",
                                   error=f"This account is not registered as '{selected_role}'. "
                                         f"Please select '{role}' from the role dropdown.",
                                   next_url=next_url)
        session["user_id"] = user_id
        session["username"] = canonical_name
        session["is_admin"] = bool(is_admin)
        session["role"] = role
        _log_event("login_success", user_id=user_id, role=role)
        # Role-based redirect
        if role == "admin" or is_admin:
            return redirect(url_for("admin_page"))
        if role == "lecturer":
            return redirect(url_for("lecturer_dashboard"))
        return redirect(url_for("student_dashboard"))
    return render_template("login.html", next_url=request.args.get("next", ""))


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        if not _require_csrf():
            return render_template("register.html", error="Security check failed."), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "register"), 6, 60):
            return render_template("register.html", error="Too many attempts. Wait a minute."), 429
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        role = request.form.get("role", "student")
        reg_number = request.form.get("reg_number", "").strip() or None
        # Students must provide reg number
        if role == "student" and not reg_number:
            return render_template("register.html", error="Registration number is required for students.")
        ok, msg, user_id, canonical = register_user(
            _db_path(), username, password, email, role=role, reg_number=reg_number
        )
        if not ok:
            return render_template("register.html", error=msg)
        session.clear()
        return redirect(url_for("login", message="Registration complete. Account pending admin approval."))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required_page
def change_password_page():
    if request.method == "POST":
        if not _require_csrf():
            return render_template("change_password.html", error="Security check failed."), 400
        ok, msg = change_password(
            _db_path(), int(session["user_id"]),
            request.form.get("current_password", ""),
            request.form.get("new_password", ""),
        )
        if not ok:
            return render_template("change_password.html", error=msg), 400
        return render_template("change_password.html", success="Password changed successfully.")
    return render_template("change_password.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password_page():
    if request.method == "POST":
        if not _require_csrf():
            return render_template("forgot_password.html", error="Security check failed."), 400
        identifier = request.form.get("identifier", "")
        token, recipient = request_password_reset(_db_path(), identifier)
        if token and recipient:
            reset_url = url_for("reset_password_page", token=token, _external=True)
            _send_password_reset_email(recipient, reset_url)
        return render_template("forgot_password.html",
                               success="If an account exists, reset instructions have been sent.")
    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password_page(token):
    if request.method == "POST":
        if not _require_csrf():
            return render_template("reset_password.html", token=token, error="Security check failed."), 400
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        if new_pw != confirm:
            return render_template("reset_password.html", token=token, error="Passwords do not match."), 400
        ok, msg = reset_password_with_token(_db_path(), token, new_pw)
        if not ok:
            return render_template("reset_password.html", token=token, error=msg), 400
        return render_template("reset_password.html", token=token, success=msg)
    return render_template("reset_password.html", token=token)


# ── Home (role-aware redirect) ──────────────────────────────────────────────

@app.route("/")
@login_required_page
def home():
    u = _current_user()
    if u["is_admin"] or u["role"] == "admin":
        return redirect(url_for("admin_page"))
    if u["role"] == "lecturer":
        return redirect(url_for("lecturer_dashboard"))
    return redirect(url_for("student_dashboard"))


# ── Student dashboard ───────────────────────────────────────────────────────

@app.route("/student")
@login_required_page
def student_dashboard():
    u = _current_user()
    if u["role"] not in ("student", "admin"):
        return redirect(url_for("home"))
    with _connect_db() as conn:
        # get courses this student is enrolled in
        enrollments = conn.execute("""
            SELECT sc.course_id, sc.risk_level, sc.acknowledged_at,
                   c.name AS course_name, c.code,
                   lec.username AS lecturer_name
            FROM student_courses sc
            JOIN courses c ON c.id = sc.course_id
            LEFT JOIN users lec ON lec.id = c.lecturer_id
            WHERE sc.student_id = ?
            ORDER BY sc.risk_level DESC
        """, (u["id"],)).fetchall()

        # get latest scores per course
        scores = {}
        for e in enrollments:
            cid = e["course_id"]
            cats = conn.execute("""
                SELECT cat_number, score FROM cat_scores
                WHERE student_id=? AND course_id=?
                ORDER BY cat_number
            """, (u["id"], cid)).fetchall()
            att = conn.execute("""
                SELECT attendance_pct FROM attendance_records
                WHERE student_id=? AND course_id=?
                ORDER BY id DESC LIMIT 1
            """, (u["id"], cid)).fetchone()
            ass = conn.execute("""
                SELECT score_pct, submitted_pct FROM assignment_scores
                WHERE student_id=? AND course_id=?
                ORDER BY id DESC LIMIT 1
            """, (u["id"], cid)).fetchone()

            cat_list = [dict(c) for c in cats]
            cat_avg = round(sum(c["score"] for c in cat_list) / len(cat_list), 1) if cat_list else None
            scores[cid] = {
                "cats": cat_list,
                "cat_avg": cat_avg,
                "attendance": att["attendance_pct"] if att else None,
                "assignment_score": ass["score_pct"] if ass else None,
                "assignments_submitted": ass["submitted_pct"] if ass else None,
            }

        # get latest prediction for student
        predictions = conn.execute("""
            SELECT prediction, confidence, predicted_final_score, created_at, course_id
            FROM predictions
            WHERE user_id=? AND student_id=?
            ORDER BY id DESC LIMIT 10
        """, (u["id"], u["id"])).fetchall()

        # get interventions received
        interventions = conn.execute("""
            SELECT i.action, i.outcome, i.intervened_at,
                   lec.username AS lecturer_name, c.name AS course_name
            FROM interventions i
            JOIN users lec ON lec.id = i.lecturer_id
            JOIN courses c ON c.id = i.course_id
            WHERE i.student_id=?
            ORDER BY i.intervened_at DESC LIMIT 10
        """, (u["id"],)).fetchall()

    return render_template("student_dashboard.html",
                           u=u, enrollments=enrollments, scores=scores,
                           predictions=predictions, interventions=interventions)


@app.route("/student/acknowledge/<int:course_id>", methods=["POST"])
@login_required_page
def student_acknowledge(course_id):
    u = _current_user()
    if not _require_csrf():
        return redirect(url_for("student_dashboard"))
    with _connect_db() as conn:
        conn.execute("""
            UPDATE student_courses SET acknowledged_at=?
            WHERE student_id=? AND course_id=?
        """, (datetime.now(UTC).isoformat(), u["id"], int(course_id)))
        conn.commit()
    return redirect(url_for("student_dashboard"))


# ── Lecturer dashboard ──────────────────────────────────────────────────────

@app.route("/lecturer")
@lecturer_required
def lecturer_dashboard():
    u = _current_user()
    at_risk = get_at_risk_students_for_lecturer(_db_path(), u["id"])
    courses = get_courses_for_lecturer(_db_path(), u["id"])
    with _connect_db() as conn:
        # count interventions done this month
        interventions_this_month = conn.execute("""
            SELECT COUNT(*) FROM interventions
            WHERE lecturer_id=?
              AND strftime('%Y-%m', intervened_at)=strftime('%Y-%m','now')
        """, (u["id"],)).fetchone()[0]
        # total students in lecturer's courses
        total_students = conn.execute("""
            SELECT COUNT(DISTINCT sc.student_id)
            FROM student_courses sc
            JOIN courses c ON c.id=sc.course_id
            WHERE c.lecturer_id=?
        """, (u["id"],)).fetchone()[0]
    return render_template("lecturer_dashboard.html",
                           u=u, at_risk=at_risk, courses=courses,
                           interventions_this_month=interventions_this_month,
                           total_students=total_students)


@app.route("/lecturer/student/<int:student_id>/course/<int:course_id>")
@lecturer_required
def lecturer_student_detail(student_id, course_id):
    u = _current_user()
    with _connect_db() as conn:
        # verify this student is in this lecturer's course
        course = conn.execute("""
            SELECT * FROM courses WHERE id=? AND lecturer_id=?
        """, (int(course_id), u["id"])).fetchone()
        if not course and not u["is_admin"]:
            return render_template("forbidden.html"), 403

        student = conn.execute(
            "SELECT id, username, email, reg_number FROM users WHERE id=?",
            (int(student_id),)
        ).fetchone()
        cats = conn.execute("""
            SELECT cat_number, score FROM cat_scores
            WHERE student_id=? AND course_id=? ORDER BY cat_number
        """, (int(student_id), int(course_id))).fetchall()
        att = conn.execute("""
            SELECT attendance_pct, recorded_at FROM attendance_records
            WHERE student_id=? AND course_id=? ORDER BY id DESC LIMIT 1
        """, (int(student_id), int(course_id))).fetchone()
        ass = conn.execute("""
            SELECT score_pct, submitted_pct FROM assignment_scores
            WHERE student_id=? AND course_id=? ORDER BY id DESC LIMIT 1
        """, (int(student_id), int(course_id))).fetchone()
        sc = conn.execute("""
            SELECT risk_level FROM student_courses
            WHERE student_id=? AND course_id=?
        """, (int(student_id), int(course_id))).fetchone()
        prev_interventions = conn.execute("""
            SELECT action, outcome, intervened_at, follow_up_needed
            FROM interventions
            WHERE student_id=? AND course_id=?
            ORDER BY intervened_at DESC
        """, (int(student_id), int(course_id))).fetchall()
        latest_pred = conn.execute("""
            SELECT prediction, confidence, predicted_final_score, created_at
            FROM predictions
            WHERE student_id=? AND course_id=?
            ORDER BY id DESC LIMIT 1
        """, (int(student_id), int(course_id))).fetchone()

    # compute prediction on the fly if we have enough data
    cat_list = [dict(c) for c in cats]
    cat1 = cat_list[0]["score"] if len(cat_list) > 0 else 0
    cat2 = cat_list[1]["score"] if len(cat_list) > 1 else cat1
    attendance = att["attendance_pct"] if att else 0
    assignment_score = ass["score_pct"] if ass else 0
    assignments_submitted = ass["submitted_pct"] if ass else 0
    feats = _build_features(attendance, cat1, cat2, assignment_score, assignments_submitted)
    label, confidence, predicted_final = _predict(feats)

    return render_template("lecturer_student_detail.html",
                           u=u, student=student, course=course,
                           cats=cat_list, attendance=att, assignment=ass,
                           risk_level=sc["risk_level"] if sc else "none",
                           prev_interventions=prev_interventions,
                           latest_pred=latest_pred,
                           live_label=label, live_confidence=confidence,
                           predicted_final=predicted_final, feats=feats)


@app.route("/lecturer/intervene", methods=["POST"])
@lecturer_required
def lecturer_intervene():
    u = _current_user()
    if not _require_csrf():
        return redirect(url_for("lecturer_dashboard"))
    student_id = int(request.form.get("student_id", 0))
    course_id = int(request.form.get("course_id", 0))
    action = request.form.get("action", "").strip()
    outcome = request.form.get("outcome", "").strip()
    follow_up = 1 if request.form.get("follow_up_needed") else 0
    if not action:
        return redirect(url_for("lecturer_dashboard"))
    with _connect_db() as conn:
        conn.execute("""
            INSERT INTO interventions
              (student_id, course_id, lecturer_id, action, outcome, follow_up_needed)
            VALUES (?,?,?,?,?,?)
        """, (student_id, course_id, u["id"], action, outcome, follow_up))
        conn.commit()
    _log_event("intervention_logged", lecturer=u["id"], student=student_id, course=course_id)
    return redirect(url_for("lecturer_student_detail", student_id=student_id, course_id=course_id))


@app.route("/lecturer/enter-scores", methods=["GET", "POST"])
@lecturer_required
def lecturer_enter_scores():
    u = _current_user()
    courses = get_courses_for_lecturer(_db_path(), u["id"])
    message = None
    error = None

    if request.method == "POST":
        if not _require_csrf():
            return render_template("lecturer_enter_scores.html", courses=courses, error="Security check failed.")
        student_id_raw = request.form.get("student_id", "")
        course_id = int(request.form.get("course_id", 0))
        cat_number = int(request.form.get("cat_number", 1))
        cat_score = float(request.form.get("cat_score", 0))
        attendance = float(request.form.get("attendance", 0))
        assignment_score = float(request.form.get("assignment_score", 0))
        assignments_submitted = float(request.form.get("assignments_submitted", 100))

        # resolve student by reg number or username
        with _connect_db() as conn:
            student = conn.execute("""
                SELECT u.id FROM users u
                JOIN student_courses sc ON sc.student_id=u.id
                WHERE sc.course_id=?
                  AND (lower(u.reg_number)=lower(?) OR lower(u.username)=lower(?))
            """, (course_id, student_id_raw, student_id_raw)).fetchone()
            if not student:
                return render_template("lecturer_enter_scores.html", courses=courses,
                                       error=f"Student '{student_id_raw}' not found in this course.")
            sid = student["id"]

            # upsert CAT score
            conn.execute("""
                INSERT INTO cat_scores (student_id, course_id, cat_number, score, entered_by)
                VALUES (?,?,?,?,?)
                ON CONFLICT(student_id, course_id, cat_number)
                DO UPDATE SET score=excluded.score, entered_by=excluded.entered_by,
                              entered_at=CURRENT_TIMESTAMP
            """, (sid, course_id, cat_number, cat_score, u["id"]))

            # upsert attendance
            conn.execute("""
                INSERT INTO attendance_records (student_id, course_id, attendance_pct, recorded_by)
                VALUES (?,?,?,?)
            """, (sid, course_id, attendance, u["id"]))

            # upsert assignment
            conn.execute("""
                INSERT INTO assignment_scores
                  (student_id, course_id, score_pct, submitted_pct, entered_by)
                VALUES (?,?,?,?,?)
            """, (sid, course_id, assignment_score, assignments_submitted, u["id"]))

            conn.commit()

            # re-read all CAT scores for this student
            all_cats = conn.execute("""
                SELECT score FROM cat_scores WHERE student_id=? AND course_id=?
            """, (sid, course_id)).fetchall()
            cat1 = all_cats[0]["score"] if len(all_cats) > 0 else cat_score
            cat2 = all_cats[1]["score"] if len(all_cats) > 1 else cat1

        # run prediction
        feats = _build_features(attendance, cat1, cat2, assignment_score, assignments_submitted)
        label, confidence, predicted_final = _predict(feats)
        risk = _compute_risk_level(attendance, cat1, cat2, assignment_score, label)

        # store prediction
        with _connect_db() as conn:
            conn.execute("""
                INSERT INTO predictions
                  (user_id, student_id, course_id, mode, attendance,
                   cat1_score, cat2_score, cat_average, assignment_score,
                   assignments_submitted, predicted_final_score, prediction, confidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (u["id"], sid, course_id, "lecturer_entry",
                  attendance, cat1, cat2, feats["CAT_Average"],
                  assignment_score, assignments_submitted,
                  predicted_final, label, confidence))
            conn.commit()

        # update risk level
        _update_student_risk(_db_path(), sid, course_id, risk)

        # notify lecturer by email if newly at-risk (orange or red)
        if risk in ("orange", "red"):
            with _connect_db() as conn:
                lec = conn.execute(
                    "SELECT email, username FROM users WHERE id=?", (u["id"],)
                ).fetchone()
                stu = conn.execute(
                    "SELECT username FROM users WHERE id=?", (sid,)
                ).fetchone()
                crs = conn.execute(
                    "SELECT name FROM courses WHERE id=?", (course_id,)
                ).fetchone()
            if lec and stu and crs:
                _notify_lecturer_at_risk(
                    lec["email"], lec["username"],
                    stu["username"], crs["name"], risk
                )

        message = (f"Scores saved. Prediction: {label} ({confidence}%). "
                   f"Predicted final exam score: {predicted_final}/70. "
                   f"Risk level: {risk.upper()}.")

    return render_template("lecturer_enter_scores.html",
                           courses=courses, message=message, error=error)


# ── Admin ───────────────────────────────────────────────────────────────────

@app.route("/admin")
@admin_required
def admin_page():
    with _connect_db() as conn:
        totals = conn.execute("""
            SELECT
              (SELECT COUNT(*) FROM users) AS users_count,
              (SELECT COUNT(*) FROM users WHERE role='student') AS students_count,
              (SELECT COUNT(*) FROM users WHERE role='lecturer') AS lecturers_count,
              (SELECT COUNT(*) FROM predictions) AS predictions_count,
              (SELECT COUNT(*) FROM predictions WHERE prediction='Pass') AS pass_count,
              (SELECT COUNT(*) FROM predictions WHERE prediction='Fail') AS fail_count,
              (SELECT COUNT(*) FROM predictions WHERE prediction='At Risk') AS at_risk_count,
              (SELECT COUNT(*) FROM interventions) AS interventions_count,
              (SELECT COUNT(*) FROM student_courses WHERE risk_level='red') AS red_count,
              (SELECT COUNT(*) FROM student_courses WHERE risk_level='orange') AS orange_count,
              (SELECT COUNT(*) FROM student_courses WHERE risk_level='yellow') AS yellow_count
        """).fetchone()
        recent_users = conn.execute("""
            SELECT id, username, email, role, is_admin, is_active, is_approved, created_at
            FROM users ORDER BY id DESC LIMIT 25
        """).fetchall()
        pending_users = conn.execute("""
            SELECT id, username, email, role, created_at
            FROM users WHERE is_approved=0 ORDER BY id DESC
        """).fetchall()
        at_risk = get_all_at_risk_students(_db_path())
        courses = conn.execute("""
            SELECT c.id, c.name, c.code, u.username AS lecturer_name,
                   COUNT(sc.student_id) AS student_count
            FROM courses c
            LEFT JOIN users u ON u.id=c.lecturer_id
            LEFT JOIN student_courses sc ON sc.course_id=c.id
            GROUP BY c.id
            ORDER BY c.name
        """).fetchall()
    return render_template("admin.html", totals=totals, recent_users=recent_users,
                           pending_users=pending_users, at_risk=at_risk, courses=courses)


@app.route("/admin/user/<int:user_id>/approve", methods=["POST"])
@admin_required
def admin_approve_user(user_id):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        conn.execute("""
            UPDATE users SET is_approved=1, approved_by=?, approved_at=?, is_active=1 WHERE id=?
        """, (int(session["user_id"]), datetime.now(UTC).isoformat(), int(user_id)))
        conn.commit()
    return redirect(url_for("admin_page"))


@app.route("/admin/user/<int:user_id>/reject", methods=["POST"])
@admin_required
def admin_reject_user(user_id):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    if int(user_id) == int(session.get("user_id", -1)):
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        conn.execute("""
            UPDATE users SET is_approved=0, is_active=0, approval_note='Rejected by admin' WHERE id=?
        """, (int(user_id),))
        conn.commit()
    return redirect(url_for("admin_page"))


@app.route("/admin/user/<int:user_id>/toggle-active", methods=["POST"])
@admin_required
def admin_toggle_active(user_id):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    if int(user_id) == int(session.get("user_id", -1)):
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        row = conn.execute("SELECT is_active FROM users WHERE id=?", (user_id,)).fetchone()
        if row:
            conn.execute("UPDATE users SET is_active=? WHERE id=?",
                         (0 if int(row["is_active"]) else 1, user_id))
            conn.commit()
    return redirect(url_for("admin_page"))


@app.route("/admin/courses/add", methods=["POST"])
@admin_required
def admin_add_course():
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper()
    lecturer_id = request.form.get("lecturer_id", "")
    if not name or not code:
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        try:
            conn.execute("""
                INSERT INTO courses (name, code, lecturer_id) VALUES (?,?,?)
            """, (name, code, int(lecturer_id) if lecturer_id else None))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    return redirect(url_for("admin_page"))


@app.route("/admin/enroll", methods=["POST"])
@admin_required
def admin_enroll_student():
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    student_id = int(request.form.get("student_id", 0))
    course_id = int(request.form.get("course_id", 0))
    with _connect_db() as conn:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO student_courses (student_id, course_id) VALUES (?,?)
            """, (student_id, course_id))
            conn.commit()
        except sqlite3.IntegrityError:
            pass
    return redirect(url_for("admin_page"))


# ── Predict API (updated for new features) ──────────────────────────────────

@app.route("/predict", methods=["POST"])
@login_required_api
def predict():
    u = _current_user()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    try:
        cat1 = float(data.get("CAT1_Score", data.get("CAT_Score", 0)))
        cat2 = float(data.get("CAT2_Score", cat1))
        attendance = float(data["Attendance"])
        assignment_score = float(data.get("Assignment_Score", 0))
        assignments_submitted = float(data.get("Assignments_Submitted", 100))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    feats = _build_features(attendance, cat1, cat2, assignment_score, assignments_submitted)
    label, confidence, predicted_final = _predict(feats)
    risk = _compute_risk_level(attendance, cat1, cat2, assignment_score, label)

    with _connect_db() as conn:
        conn.execute("""
            INSERT INTO predictions
              (user_id, mode, attendance, cat1_score, cat2_score, cat_average,
               assignment_score, assignments_submitted, predicted_final_score, prediction, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (u["id"], "single", attendance, cat1, cat2, feats["CAT_Average"],
              assignment_score, assignments_submitted, predicted_final, label, confidence))
        conn.commit()

    fi = _feature_importance()
    explanation = sorted(
        [{"feature": k, "importance": v} for k, v in fi.items()],
        key=lambda x: -x["importance"]
    )

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "predicted_final_exam_score": predicted_final,
        "predicted_final_out_of": 70,
        "risk_level": risk,
        "explainability": explanation,
        "note": (
            "Predicted final exam score is estimated from pre-exam data only. "
            "No final exam score was used as input."
        ),
    })


@app.route("/explainability")
@login_required_api
def explainability():
    return jsonify({
        "model_features": MODEL_FEATURES,
        "feature_importance": _feature_importance(),
        "note": "Final exam score is NOT used as a feature — enabling early warning before end of semester.",
        "classes": list(_clf.classes_),
    })


@app.route("/history")
@login_required_page
def history_page():
    u = _current_user()
    with _connect_db() as conn:
        rows = conn.execute("""
            SELECT mode, attendance, cat1_score, cat2_score, cat_average,
                   assignment_score, predicted_final_score,
                   prediction, confidence, created_at
            FROM predictions WHERE user_id=?
            ORDER BY id DESC LIMIT 200
        """, (u["id"],)).fetchall()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(debug=True, port=port)
