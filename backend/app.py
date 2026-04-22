from __future__ import annotations

import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
import logging

import joblib
import numpy as np
import pandas as pd
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

from auth_service import ensure_schema, get_user_by_id, register_user, verify_user

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
        if u and u["is_admin"] and session.get("admin_authenticated"):
            return view(*args, **kwargs)
        return render_template("forbidden.html"), 403

    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        dest = _safe_next_url(request.args.get("next")) or url_for("home")
        return redirect(dest)
    next_url = request.args.get("next", "")
    if request.method == "POST":
        if not _require_csrf():
            return render_template("login.html", error="Security check failed. Refresh and try again.", next_url=next_url), 400
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
        if not _rate_limit((ip, "login"), limit=10, window_s=60):
            return render_template("login.html", error="Too many attempts. Please wait a minute.", next_url=next_url), 429
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        next_url = request.form.get("next", "") or ""
        admin_login = request.form.get("login_mode") == "admin"
        verified = verify_user(_db_path(), username, password)
        if verified is None:
            return render_template(
                "login.html",
                error="Invalid username or password.",
                next_url=next_url,
            )
        user_id, canonical_name, is_admin = verified
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
    return render_template("login.html", next_url=next_url)


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
        password = request.form.get("password", "")
        ok, msg, user_id, canonical = register_user(_db_path(), username, password)
        if not ok:
            return render_template("register.html", error=msg)
        session["user_id"] = user_id
        session["username"] = canonical or username.strip()
        session["is_admin"] = False
        session["admin_authenticated"] = False
        _log_event("register_success", user_id=user_id, username=session["username"])
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    _log_event("logout", user_id=session.get("user_id"))
    session.clear()
    return redirect(url_for("login"))


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
            "SELECT id, username, is_admin, is_active, created_at FROM users ORDER BY id DESC LIMIT 25"
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
    return render_template("admin.html", totals=totals, recent_users=recent_users, daily=daily)


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
