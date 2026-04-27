from __future__ import annotations

import os
import secrets
import sqlite3
import time
import io
import smtplib
import ssl
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
import logging
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from auth_service import (
    change_password,
    ensure_schema,
    get_user_by_id,
    is_user_approved,
    register_user,
    request_password_reset,
    reset_password_with_token,
    verify_user,
)

# Load environment variables from project-root .env for local/dev runs.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "student_performance_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "predicted_results.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_secret = os.environ.get("FLASK_SECRET_KEY") or "dev-secret-change-me"
# Flask sets SECRET_KEY=None by default, so setdefault() won't help.
if not app.secret_key:
    app.secret_key = _secret
app.config.setdefault("AUTH_DATABASE", os.path.join(BASE_DIR, "users.db"))
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
# If you terminate TLS at a proxy, set this True in deployment.
app.config.setdefault("SESSION_COOKIE_SECURE", False)
app.config.setdefault("PERMANENT_SESSION_LIFETIME", 60 * 60 * 8)  # 8 hours

logger = logging.getLogger("student_dashboard")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOGS_DIR, "app.log")
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

model = joblib.load(MODEL_PATH)

REQUIRED_COLUMNS = ["Attendance", "CAT_Score", "Assignment_Score", "Final_Exam"]
UTC = timezone.utc
RUBRIC_WEIGHTS = {"CAT_Score": 0.15, "Assignment_Score": 0.15, "Final_Exam": 0.70}
ATTENDANCE_EXAM_MIN = 70.0


def _db_path() -> str:
    return app.config["AUTH_DATABASE"]


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_user_flags() -> None:
    with _connect_db() as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "is_active" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            conn.commit()
        if "is_approved" in cols:
            conn.execute("UPDATE users SET is_approved = 1 WHERE is_admin = 1 AND is_approved = 0")
            conn.commit()


@app.before_request
def _ensure_auth_db() -> None:
    ensure_schema(_db_path())
    _ensure_user_flags()
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                attendance REAL,
                cat_score REAL,
                assignment_score REAL,
                final_exam REAL,
                prediction TEXT NOT NULL,
                confidence REAL,
                payload_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _safe_next_url(next_url: str | None) -> str | None:
    if not next_url or not isinstance(next_url, str):
        return None
    if not next_url.startswith("/") or next_url.startswith("//"):
        return None
    return next_url


def _current_user() -> dict | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    user = get_user_by_id(_db_path(), int(user_id))
    if not user:
        return None
    uid, uname, is_admin = user
    with _connect_db() as conn:
        row = conn.execute("SELECT is_active FROM users WHERE id = ?", (int(uid),)).fetchone()
    if row and int(row["is_active"]) == 0:
        session.clear()
        return None
    return {"id": uid, "username": uname, "is_admin": bool(is_admin)}


def _log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    logger.info(str(payload))


def _should_expose_reset_link() -> bool:
    return bool(app.config.get("TESTING")) or os.environ.get("SHOW_RESET_LINKS", "0") == "1"


def _send_password_reset_email(recipient_email: str, reset_url: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_sender = os.environ.get("SMTP_SENDER", smtp_user).strip()
    smtp_use_ssl = os.environ.get("SMTP_USE_SSL", "0") == "1"
    smtp_use_starttls = os.environ.get("SMTP_USE_STARTTLS", "1") == "1"
    if not smtp_host or not smtp_sender:
        return False

    msg = EmailMessage()
    msg["Subject"] = "Password reset instructions"
    msg["From"] = smtp_sender
    msg["To"] = recipient_email
    msg.set_content(
        "A password reset was requested for your account.\n\n"
        f"Use this link to reset your password:\n{reset_url}\n\n"
        "If you did not request this, you can ignore this email."
    )

    context = ssl.create_default_context()
    try:                                          # ← ADD THIS
        if smtp_use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10, context=context) as smtp:
                if smtp_user:
                    smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
                if smtp_use_starttls:
                    smtp.starttls(context=context)
                if smtp_user:
                    smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
        return True
    except Exception as exc:                      # ← ADD THIS
        app.logger.error("SMTP send failed: %s", exc)
        return False


def _write_reset_link_log(recipient_email: str, reset_url: str) -> None:
    log_path = os.path.join(LOGS_DIR, "password_reset_links.log")
    stamp = datetime.now(UTC).isoformat()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} recipient={recipient_email} reset_url={reset_url}\n")


def _store_prediction(
    user_id: int,
    mode: str,
    row: dict,
    prediction: str,
    confidence: float | None,
    payload_json: str | None = None,
) -> None:
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                user_id, mode, attendance, cat_score, assignment_score, final_exam,
                prediction, confidence, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                mode,
                row.get("Attendance"),
                row.get("CAT_Score"),
                row.get("Assignment_Score"),
                row.get("Final_Exam"),
                prediction,
                confidence,
                payload_json,
            ),
        )
        conn.commit()


def _output_file(prefix: str, user_id: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUTS_DIR, f"{prefix}_u{user_id}_{stamp}.csv")


def _feature_importance() -> dict[str, float]:
    vals = getattr(model, "feature_importances_", None)
    if vals is None:
        return {k: 0.0 for k in REQUIRED_COLUMNS}
    return {k: round(float(v), 4) for k, v in zip(REQUIRED_COLUMNS, vals)}


def _local_explanation(row: dict[str, float]) -> list[dict]:
    imp = _feature_importance()
    out = []
    for key in REQUIRED_COLUMNS:
        value = float(row[key])
        out.append(
            {
                "feature": key,
                "value": value,
                "contribution": round(imp[key] * (value / 100.0), 4),
            }
        )
    out.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return out


def _rubric_explanation(row: dict[str, float]) -> dict:
    attendance = float(row["Attendance"])
    eligible_for_exam = attendance >= ATTENDANCE_EXAM_MIN
    breakdown = []
    total = 0.0
    for metric, weight in RUBRIC_WEIGHTS.items():
        raw = float(row[metric])
        contribution = round(raw * weight, 2)
        breakdown.append(
            {
                "metric": metric,
                "weight_percent": int(weight * 100),
                "raw_score": raw,
                "weighted_contribution": contribution,
            }
        )
        total += contribution
    breakdown.sort(key=lambda x: x["weighted_contribution"], reverse=True)
    return {
        "attendance_threshold": ATTENDANCE_EXAM_MIN,
        "attendance": attendance,
        "eligible_for_exam": eligible_for_exam,
        "weighted_total": round(total, 2),
        "breakdown": breakdown,
        "note": (
            "Attendance below 70% means learner is not eligible to sit exam."
            if not eligible_for_exam
            else "Attendance requirement met."
        ),
    }


def _api_key_for_user(user_id: int) -> str | None:
    with _connect_db() as conn:
        row = conn.execute("SELECT api_key FROM users WHERE id = ?", (int(user_id),)).fetchone()
    if not row:
        return None
    return row["api_key"]


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


_RATE: dict[tuple[str, str], list[float]] = {}


def _rate_limit(key: tuple[str, str], limit: int, window_s: int) -> bool:
    if app.config.get("TESTING"):
        return True
    now = time.time()
    bucket = _RATE.get(key, [])
    bucket = [t for t in bucket if now - t < window_s]
    if len(bucket) >= limit:
        _RATE[key] = bucket
        return False
    bucket.append(now)
    _RATE[key] = bucket
    return True


def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _require_csrf() -> bool:
    if app.config.get("TESTING"):
        return True
    sent = request.form.get("csrf_token", "")
    return bool(sent) and secrets.compare_digest(sent, session.get("csrf_token", ""))


@app.context_processor
def _inject_csrf():
    u = _current_user() or {}
    return {
        "csrf_token": _csrf_token,
        "current_username": u.get("username"),
        "current_is_admin": bool(u.get("is_admin", False)),
    }


def admin_required_page(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        u = _current_user()
        if u and u["is_admin"]:
            return view(*args, **kwargs)
        return render_template("forbidden.html"), 403

    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        dest = _safe_next_url(request.args.get("next")) or url_for("home")
        return redirect(dest)
    next_url = request.args.get("next", "")
    ui_message = request.args.get("message", "")
    if request.method == "POST":
        if not _require_csrf():
            return render_template("login.html", error="Security check failed. Refresh and try again.", next_url=next_url), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "login"), limit=10, window_s=60):
            return render_template("login.html", error="Too many attempts. Please wait a minute.", next_url=next_url), 429
        identifier = request.form.get("identifier", "") or request.form.get("username", "")
        password = request.form.get("password", "")
        next_url = request.form.get("next", "") or ""
        admin_login = request.form.get("login_mode") == "admin"
        verified = verify_user(_db_path(), identifier, password)
        if verified is None:
            return render_template(
                "login.html",
                error="Invalid username or password.",
                next_url=next_url,
            )
        user_id, canonical_name, is_admin = verified
        if not is_user_approved(_db_path(), int(user_id)):
            return render_template(
                "login.html",
                error="Your account is pending admin approval.",
                next_url=next_url,
                message="Ask an administrator to approve your account.",
            ), 403
        if admin_login and not bool(is_admin):
            return render_template(
                "login.html",
                error="Admin login requires an admin account.",
                next_url=next_url,
            ), 403
        with _connect_db() as conn:
            active_row = conn.execute("SELECT is_active FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if active_row and int(active_row["is_active"]) == 0:
            return render_template("login.html", error="Account is disabled. Contact admin.", next_url=next_url), 403
        session["user_id"] = user_id
        session["username"] = canonical_name
        session["is_admin"] = bool(is_admin)
        session["admin_authenticated"] = bool(admin_login and is_admin)
        _log_event("login_success", user_id=user_id, username=canonical_name)
        dest = _safe_next_url(next_url) or url_for("home")
        return redirect(dest)
    return render_template("login.html", next_url=next_url, message=ui_message)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        if not _require_csrf():
            return render_template("register.html", error="Security check failed. Refresh and try again."), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "register"), limit=6, window_s=60):
            return render_template("register.html", error="Too many attempts. Please wait a minute."), 429
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        ok, msg, user_id, canonical = register_user(_db_path(), username, password, email)
        if not ok:
            return render_template("register.html", error=msg)
        session["user_id"] = user_id
        session["username"] = canonical or username.strip()
        approved = is_user_approved(_db_path(), int(user_id))
        session["is_admin"] = False
        session["admin_authenticated"] = False
        _log_event("register_success", user_id=user_id, username=session["username"])
        if not approved:
            session.clear()
            return redirect(url_for("login", message="Registration complete. Account pending admin approval."))
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    _log_event("logout", user_id=session.get("user_id"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required_page
def change_password_page():
    if request.method == "POST":
        if not _require_csrf():
            return render_template("change_password.html", error="Security check failed. Refresh and try again."), 400
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if new_password != confirm_password:
            return render_template("change_password.html", error="New password and confirmation do not match."), 400
        ok, msg = change_password(_db_path(), int(session.get("user_id")), current_password, new_password)
        if not ok:
            return render_template("change_password.html", error=msg), 400
        _log_event("password_changed", user_id=session.get("user_id"))
        return render_template("change_password.html", success="Password changed successfully.")
    return render_template("change_password.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password_page():
    if request.method == "POST":
        if not _require_csrf():
            return render_template("forgot_password.html", error="Security check failed. Refresh and try again."), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "forgot-password"), limit=6, window_s=60):
            return render_template("forgot_password.html", error="Too many attempts. Please wait a minute."), 429
        identifier = request.form.get("identifier", "")
        token, recipient_email = request_password_reset(_db_path(), identifier)
        reset_url = None
        email_sent = False
        if token:
            reset_url = url_for("reset_password_page", token=token, _external=True)
            try:
                email_sent = _send_password_reset_email(recipient_email, reset_url)
            except Exception as exc:
                _log_event(
                    "password_reset_email_failed",
                    identifier=identifier,
                    recipient_email=recipient_email,
                    error=str(exc),
                )
            if not email_sent and reset_url:
                # Dev fallback: keep reset links discoverable when SMTP is not configured.
                _write_reset_link_log(recipient_email, reset_url)
            _log_event("password_reset_requested", identifier=identifier, recipient_email=recipient_email)
        public_link = reset_url if (_should_expose_reset_link() and reset_url) else None
        return render_template(
            "forgot_password.html",
            success=(
                "If an account exists, reset instructions have been sent."
                if email_sent
                else "If an account exists, reset instructions will be available shortly."
            ),
            reset_url=public_link,
        )
    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password_page(token: str):
    if request.method == "POST":
        if not _require_csrf():
            return render_template("reset_password.html", token=token, error="Security check failed. Refresh and try again."), 400
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        if new_password != confirm_password:
            return render_template("reset_password.html", token=token, error="Password confirmation does not match."), 400
        ok, msg = reset_password_with_token(_db_path(), token, new_password)
        if not ok:
            return render_template("reset_password.html", token=token, error=msg), 400
        _log_event("password_reset_success")
        return render_template("reset_password.html", token=token, success=msg)
    return render_template("reset_password.html", token=token)


@app.route("/")
@login_required_page
def home():
    return render_template("index.html")


@app.route("/history")
@login_required_page
def history_page():
    u = _current_user()
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT mode, attendance, cat_score, assignment_score, final_exam,
                   prediction, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 200
            """,
            (u["id"],),
        ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/history/export")
@login_required_page
def history_export():
    u = _current_user()
    out_path = _output_file("history", u["id"])
    with _connect_db() as conn:
        data = pd.read_sql_query(
            """
            SELECT mode AS Mode,
                   attendance AS Attendance,
                   cat_score AS CAT_Score,
                   assignment_score AS Assignment_Score,
                   final_exam AS Final_Exam,
                   prediction AS Prediction,
                   confidence AS Confidence,
                   created_at AS Created_At
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            conn,
            params=(u["id"],),
        )
    data.to_csv(out_path, index=False)
    return send_file(out_path, as_attachment=True)


@app.route("/history/export.xlsx")
@login_required_page
def history_export_excel():
    u = _current_user()
    with _connect_db() as conn:
        data = pd.read_sql_query(
            """
            SELECT mode AS Mode,
                   attendance AS Attendance,
                   cat_score AS CAT_Score,
                   assignment_score AS Assignment_Score,
                   final_exam AS Final_Exam,
                   prediction AS Prediction,
                   confidence AS Confidence,
                   created_at AS Created_At
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            conn,
            params=(u["id"],),
        )
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="History")
    out.seek(0)
    filename = f"history_u{u['id']}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        out,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/history/export.pdf")
@login_required_page
def history_export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return jsonify({"error": "PDF export dependency missing. Install reportlab."}), 500

    u = _current_user()
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT created_at, mode, attendance, cat_score, assignment_score, final_exam, prediction, confidence
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 300
            """,
            (u["id"],),
        ).fetchall()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, f"Prediction History Report - {u['username']}")
    y -= 18
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, y, f"Generated: {datetime.now(UTC).isoformat()}")
    y -= 24
    pdf.drawString(40, y, "Time")
    pdf.drawString(170, y, "Mode")
    pdf.drawString(220, y, "Scores (A/CAT/ASS/FIN)")
    pdf.drawString(410, y, "Pred")
    pdf.drawString(460, y, "Conf")
    y -= 12

    for row in rows:
        if y < 40:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 9)
        pdf.drawString(40, y, str(row["created_at"])[:19])
        pdf.drawString(170, y, str(row["mode"]))
        scores = f"{row['attendance']}/{row['cat_score']}/{row['assignment_score']}/{row['final_exam']}"
        pdf.drawString(220, y, scores[:33])
        pdf.drawString(410, y, str(row["prediction"]))
        conf = "-" if row["confidence"] is None else str(row["confidence"])
        pdf.drawString(460, y, conf[:8])
        y -= 12

    pdf.save()
    buffer.seek(0)
    filename = f"history_u{u['id']}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")


@app.route("/my-api-key")
@login_required_api
def my_api_key():
    u = _current_user()
    return jsonify({"api_key": _api_key_for_user(u["id"])})


@app.route("/admin")
@admin_required_page
def admin_page():
    with _connect_db() as conn:
        totals = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM users) AS users_count,
                (SELECT COUNT(*) FROM predictions) AS predictions_count,
                (SELECT COUNT(*) FROM predictions WHERE prediction = 'Pass') AS pass_count,
                (SELECT COUNT(*) FROM predictions WHERE prediction = 'Fail') AS fail_count
            """
        ).fetchone()
        recent_users = conn.execute(
            """
            SELECT id, username, is_admin, is_active, is_approved, created_at
            FROM users ORDER BY id DESC LIMIT 25
            """
        ).fetchall()
        pending_users = conn.execute(
            """
            SELECT id, username, created_at
            FROM users
            WHERE is_approved = 0
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()
        daily = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
            FROM predictions
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day DESC
            LIMIT 14
            """
        ).fetchall()
    return render_template("admin.html", totals=totals, recent_users=recent_users, pending_users=pending_users, daily=daily)


@app.route("/admin/user/<int:user_id>/approve", methods=["POST"])
@admin_required_page
def admin_approve_user(user_id: int):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET is_approved = 1, approved_by = ?, approved_at = ?, approval_note = NULL, is_active = 1
            WHERE id = ?
            """,
            (
                int(session.get("user_id")),
                datetime.now(UTC).isoformat(),
                int(user_id),
            ),
        )
        conn.commit()
    _log_event("admin_approve_user", actor=session.get("user_id"), target=user_id)
    return redirect(url_for("admin_page"))


@app.route("/admin/user/<int:user_id>/reject", methods=["POST"])
@admin_required_page
def admin_reject_user(user_id: int):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    if int(user_id) == int(session.get("user_id", -1)):
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        conn.execute(
            """
            UPDATE users
            SET is_approved = 0, is_active = 0, approved_by = ?, approved_at = ?, approval_note = ?
            WHERE id = ?
            """,
            (
                int(session.get("user_id")),
                datetime.now(UTC).isoformat(),
                "Rejected by admin",
                int(user_id),
            ),
        )
        conn.commit()
    _log_event("admin_reject_user", actor=session.get("user_id"), target=user_id)
    return redirect(url_for("admin_page"))


@app.route("/admin/user/<int:user_id>/role", methods=["POST"])
@admin_required_page
def admin_toggle_role(user_id: int):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        row = conn.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            next_role = 0 if int(row["is_admin"]) else 1
            conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (next_role, user_id))
            conn.commit()
            _log_event("admin_toggle_role", actor=session.get("user_id"), target=user_id, is_admin=next_role)
    return redirect(url_for("admin_page"))


@app.route("/admin/user/<int:user_id>/active", methods=["POST"])
@admin_required_page
def admin_toggle_active(user_id: int):
    if not _require_csrf():
        return redirect(url_for("admin_page"))
    if int(user_id) == int(session.get("user_id", -1)):
        return redirect(url_for("admin_page"))
    with _connect_db() as conn:
        row = conn.execute("SELECT is_active FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            next_active = 0 if int(row["is_active"]) else 1
            conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (next_active, user_id))
            conn.commit()
            _log_event("admin_toggle_active", actor=session.get("user_id"), target=user_id, is_active=next_active)
    return redirect(url_for("admin_page"))


@app.route("/predict", methods=["POST"])
@login_required_api
def predict():
    u = _current_user()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        row = {
            "Attendance": float(data["Attendance"]),
            "CAT_Score": float(data["CAT_Score"]),
            "Assignment_Score": float(data["Assignment_Score"]),
            "Final_Exam": float(data["Final_Exam"]),
        }
        features = np.array([[row[c] for c in REQUIRED_COLUMNS]])
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    prediction = model.predict(features)[0]
    confidence = round(float(max(model.predict_proba(features)[0])) * 100, 2)
    explanation = _local_explanation(row)
    rubric = _rubric_explanation(row)
    _store_prediction(
        u["id"],
        "single",
        row,
        str(prediction),
        confidence,
    )
    _log_event("single_prediction", user_id=u["id"], prediction=str(prediction), confidence=confidence)
    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
            "explainability": explanation[:3],
            "rubric_explainability": rubric,
        }
    )


@app.route("/predict/multi-subject", methods=["POST"])
@login_required_api
def predict_multi_subject():
    """Accept subject-wise scores, aggregate, then run overall prediction."""
    u = _current_user()
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    subjects = data.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        return jsonify({"error": "Provide a non-empty 'subjects' array."}), 400
    try:
        attendance = float(data["Attendance"])
        if attendance < 0 or attendance > 100:
            raise ValueError("Attendance must be between 0 and 100.")
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid attendance: {exc}"}), 400

    subject_rows = []
    cat_values = []
    ass_values = []
    fin_values = []
    for idx, subj in enumerate(subjects, start=1):
        if not isinstance(subj, dict):
            return jsonify({"error": f"Subject item #{idx} must be an object."}), 400
        name = str(subj.get("subject") or f"Subject {idx}")
        try:
            cat = float(subj["CAT_Score"])
            ass = float(subj["Assignment_Score"])
            fin = float(subj["Final_Exam"])
        except (KeyError, ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid subject input at #{idx}: {exc}"}), 400
        if any(v < 0 or v > 100 for v in (cat, ass, fin)):
            return jsonify({"error": f"Scores for {name} must be between 0 and 100."}), 400
        weighted_total = round(cat * 0.15 + ass * 0.15 + fin * 0.70, 2)
        subject_rows.append(
            {
                "subject": name,
                "CAT_Score": cat,
                "Assignment_Score": ass,
                "Final_Exam": fin,
                "weighted_total": weighted_total,
            }
        )
        cat_values.append(cat)
        ass_values.append(ass)
        fin_values.append(fin)

    row = {
        "Attendance": round(attendance, 2),
        "CAT_Score": round(float(np.mean(cat_values)), 2),
        "Assignment_Score": round(float(np.mean(ass_values)), 2),
        "Final_Exam": round(float(np.mean(fin_values)), 2),
    }
    features = np.array([[row[c] for c in REQUIRED_COLUMNS]])
    prediction = model.predict(features)[0]
    confidence = round(float(max(model.predict_proba(features)[0])) * 100, 2)
    rubric = _rubric_explanation(row)
    _store_prediction(
        u["id"],
        "multi_subject",
        row,
        str(prediction),
        confidence,
        payload_json=str({"subjects_count": len(subject_rows), "subjects": subject_rows}),
    )
    _log_event(
        "multi_subject_prediction",
        user_id=u["id"],
        subject_count=len(subject_rows),
        prediction=str(prediction),
        confidence=confidence,
    )
    return jsonify(
        {
            "subjects": subject_rows,
            "aggregate_features": row,
            "prediction": prediction,
            "confidence": confidence,
            "rubric_explainability": rubric,
            "note": "Overall prediction uses average CAT/Assignment/Final scores across subjects plus attendance.",
        }
    )


@app.route("/explainability")
@login_required_api
def explainability():
    return jsonify(
        {
            "feature_importance": _feature_importance(),
            "rubric": {
                "weights_percent": {
                    "CAT_Score": int(RUBRIC_WEIGHTS["CAT_Score"] * 100),
                    "Assignment_Score": int(RUBRIC_WEIGHTS["Assignment_Score"] * 100),
                    "Final_Exam": int(RUBRIC_WEIGHTS["Final_Exam"] * 100),
                },
                "attendance_min_for_exam": ATTENDANCE_EXAM_MIN,
                "summary": "CAT and Assignment are complementary at 15% each; Final exam is 70%; attendance must be at least 70%.",
            },
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    with _connect_db() as conn:
        owner = conn.execute("SELECT id FROM users WHERE api_key = ?", (api_key,)).fetchone()
    if not owner:
        return jsonify({"error": "Invalid API key"}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    try:
        row = {
            "Attendance": float(data["Attendance"]),
            "CAT_Score": float(data["CAT_Score"]),
            "Assignment_Score": float(data["Assignment_Score"]),
            "Final_Exam": float(data["Final_Exam"]),
        }
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    features = np.array([[row[c] for c in REQUIRED_COLUMNS]])
    prediction = model.predict(features)[0]
    confidence = round(float(max(model.predict_proba(features)[0])) * 100, 2)
    _store_prediction(int(owner["id"]), "api", row, str(prediction), confidence)
    _log_event("api_prediction", user_id=int(owner["id"]), prediction=str(prediction), confidence=confidence)
    return jsonify({"prediction": prediction, "confidence": confidence})


@app.route("/upload", methods=["POST"])
@login_required_api
def upload():
    u = _current_user()
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return jsonify({"error": f"Missing column(s): {', '.join(missing)}"}), 400

    work = df[REQUIRED_COLUMNS].copy()
    invalid_mask = pd.Series(False, index=work.index)
    reasons = pd.Series("", index=work.index, dtype="object")

    for col in REQUIRED_COLUMNS:
        numeric = pd.to_numeric(work[col], errors="coerce")
        bad = numeric.isna()
        invalid_mask |= bad
        reasons[bad] = reasons[bad].where(reasons[bad] != "", f"Invalid {col}")
        work[col] = numeric

    out_of_range = (work[REQUIRED_COLUMNS] < 0).any(axis=1) | (work[REQUIRED_COLUMNS] > 100).any(axis=1)
    invalid_mask |= out_of_range
    reasons[out_of_range] = reasons[out_of_range].where(
        reasons[out_of_range] != "",
        "Scores must be between 0 and 100",
    )

    valid = work[~invalid_mask].copy()
    invalid = df[invalid_mask].copy()
    if not invalid.empty:
        invalid["Error"] = reasons[invalid_mask].values

    if valid.empty:
        return jsonify({"error": "No valid rows found in uploaded CSV.", "invalid_rows": int(invalid.shape[0])}), 400

    valid["Prediction"] = model.predict(valid[REQUIRED_COLUMNS])

    pass_rate = round((valid["Prediction"] == "Pass").mean() * 100, 2)

    output_csv = _output_file("predictions", u["id"])
    valid.to_csv(output_csv, index=False)
    session["last_output_file"] = output_csv

    error_csv = None
    if not invalid.empty:
        error_csv = _output_file("invalid_rows", u["id"])
        invalid.to_csv(error_csv, index=False)
    session["last_error_file"] = error_csv

    for _, row in valid.iterrows():
        _store_prediction(
            u["id"],
            "bulk",
            {
                "Attendance": float(row["Attendance"]),
                "CAT_Score": float(row["CAT_Score"]),
                "Assignment_Score": float(row["Assignment_Score"]),
                "Final_Exam": float(row["Final_Exam"]),
            },
            str(row["Prediction"]),
            None,
        )
    _log_event("bulk_prediction", user_id=u["id"], valid_rows=int(valid.shape[0]), invalid_rows=int(invalid.shape[0]))

    return jsonify({
        "preview": valid.head(10).to_dict(orient="records"),
        "download": "/download",
        "pass_rate": pass_rate,
        "valid_rows": int(valid.shape[0]),
        "invalid_rows": int(invalid.shape[0]),
        "download_errors": "/download-errors" if error_csv else None,
    })


@app.route("/download")
@login_required_page
def download():
    output_csv = session.get("last_output_file")
    if not output_csv or not os.path.exists(output_csv):
        return jsonify({"error": "No results file available. Run a bulk upload first."}), 404
    return send_file(output_csv, as_attachment=True)


@app.route("/download-errors")
@login_required_page
def download_errors():
    path = session.get("last_error_file")
    if not path or not os.path.exists(path):
        return jsonify({"error": "No invalid rows file available."}), 404
    return send_file(path, as_attachment=True)


@app.route("/openapi.json")
def openapi_spec():
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Student Performance API", "version": "1.0.0"},
        "paths": {
            "/api/predict": {
                "post": {
                    "summary": "Predict with API key",
                    "parameters": [
                        {
                            "name": "X-API-Key",
                            "in": "header",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": REQUIRED_COLUMNS,
                                    "properties": {
                                        "Attendance": {"type": "number"},
                                        "CAT_Score": {"type": "number"},
                                        "Assignment_Score": {"type": "number"},
                                        "Final_Exam": {"type": "number"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Prediction result"},
                        "401": {"description": "Invalid API key"},
                        "400": {"description": "Invalid payload"},
                    },
                }
            }
        },
    }
    return jsonify(spec)


@app.route("/api/docs")
def api_docs():
    return render_template("api_docs.html")


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(debug=True, port=port)
