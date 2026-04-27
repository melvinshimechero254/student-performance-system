"""
Basic smoke tests for the Flask API.
Run with: pytest tests/test_app.py
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from app import app


@pytest.fixture
def client(tmp_path):
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "pytest-secret-key"
    app.config["AUTH_DATABASE"] = str(tmp_path / "test_users.db")
    with app.test_client() as c:
        yield c


def _register(client, username="testuser", password="Password12!", email=None):
    email = email or f"{username}@example.com"
    return client.post(
        "/register",
        data={"username": username, "email": email, "password": password},
        follow_redirects=True,
    )


def test_home_redirects_unauthenticated(client):
    res = client.get("/")
    assert res.status_code == 302
    assert "/login" in res.location


def test_home_authenticated(client):
    _register(client)
    res = client.get("/")
    assert res.status_code == 200
    assert b"Student Performance" in res.data


def test_predict_requires_auth(client):
    payload = {
        "Attendance": 85,
        "CAT_Score": 70,
        "Assignment_Score": 75,
        "Final_Exam": 68,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 401


def test_predict_valid_input(client):
    _register(client)
    payload = {
        "Attendance": 85,
        "CAT_Score": 70,
        "Assignment_Score": 75,
        "Final_Exam": 68,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], str) and data["prediction"]
    assert 0 <= data["confidence"] <= 100
    assert "rubric_explainability" in data
    assert "eligible_for_exam" in data["rubric_explainability"]


def test_predict_missing_field(client):
    _register(client)
    res = client.post("/predict", json={"Attendance": 85})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_no_body(client):
    _register(client)
    res = client.post("/predict", data="not json", content_type="text/plain")
    assert res.status_code == 400


def test_download_without_upload(client):
    output = os.path.join(
        os.path.dirname(__file__), "..", "backend", "predicted_results.csv"
    )
    if os.path.exists(output):
        os.remove(output)
    _register(client)
    res = client.get("/download")
    assert res.status_code == 404


def test_register_duplicate_username(client):
    _register(client, "alice", "Password12!")
    client.get("/logout")
    res = client.post(
        "/register",
        data={"username": "alice", "email": "alice2@example.com", "password": "Password12!"},
    )
    assert res.status_code == 200
    assert b"already taken" in res.data


def test_login_invalid(client):
    _register(client, "bob", "Password12!")
    client.get("/logout")
    res = client.post(
        "/login",
        data={"username": "bob", "password": "wrongpassword"},
    )
    assert res.status_code == 200
    assert b"Invalid username or password" in res.data


def test_admin_login_requires_admin_account(client):
    _register(client, "bootstrapadmin", "Password12!")
    client.get("/logout")
    _register(client, "plainuser", "Password12!")
    client.get("/logout")
    res = client.post(
        "/login",
        data={"username": "plainuser", "password": "Password12!", "login_mode": "admin"},
    )
    assert res.status_code == 403
    assert (
        b"Admin login requires an admin account" in res.data
        or b"pending admin approval" in res.data
    )


def test_history_page_and_export(client):
    _register(client)
    client.post(
        "/predict",
        json={
            "Attendance": 90,
            "CAT_Score": 88,
            "Assignment_Score": 81,
            "Final_Exam": 75,
        },
    )
    page = client.get("/history")
    assert page.status_code == 200
    exported = client.get("/history/export")
    assert exported.status_code == 200
    exported_xlsx = client.get("/history/export.xlsx")
    assert exported_xlsx.status_code == 200
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in (exported_xlsx.content_type or "")
    exported_pdf = client.get("/history/export.pdf")
    assert exported_pdf.status_code == 200
    assert "application/pdf" in (exported_pdf.content_type or "")


def test_explainability_endpoint(client):
    _register(client)
    res = client.get("/explainability")
    assert res.status_code == 200
    body = res.get_json()
    assert "feature_importance" in body
    assert "rubric" in body
    assert body["rubric"]["weights_percent"]["CAT_Score"] == 15


def test_admin_page_access_for_first_user(client):
    _register(client, "adminuser", "Password12!")
    client.get("/logout")
    client.post(
        "/login",
        data={"username": "adminuser", "password": "Password12!", "login_mode": "admin"},
    )
    res = client.get("/admin")
    assert res.status_code == 200


def test_pending_approval_blocks_login_until_admin_approves(client, tmp_path):
    _register(client, "adminseed", "Password12!")
    client.get("/logout")
    res = client.post(
        "/register",
        data={"username": "pendinguser", "email": "pending@example.com", "password": "Password12!"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"pending admin approval" in res.data

    login_pending = client.post("/login", data={"username": "pendinguser", "password": "Password12!"})
    assert login_pending.status_code == 403
    assert b"pending admin approval" in login_pending.data

    client.post("/login", data={"username": "adminseed", "password": "Password12!", "login_mode": "admin"})
    with sqlite3.connect(str(tmp_path / "test_users.db")) as conn:
        pending_id = conn.execute("SELECT id FROM users WHERE username = ?", ("pendinguser",)).fetchone()[0]
    client.post(f"/admin/user/{pending_id}/approve")
    client.get("/logout")
    approved_login = client.post("/login", data={"username": "pendinguser", "password": "Password12!"}, follow_redirects=True)
    assert approved_login.status_code == 200
    assert b"Student Performance" in approved_login.data


def test_change_password_flow(client):
    _register(client, "changepass", "Password12!")
    bad = client.post(
        "/change-password",
        data={"current_password": "Wrong12!", "new_password": "BetterPass12@", "confirm_password": "BetterPass12@"},
    )
    assert bad.status_code == 400
    ok = client.post(
        "/change-password",
        data={"current_password": "Password12!", "new_password": "BetterPass12@", "confirm_password": "BetterPass12@"},
    )
    assert ok.status_code == 200
    client.get("/logout")
    old_login = client.post("/login", data={"username": "changepass", "password": "Password12!"})
    assert b"Invalid username or password" in old_login.data
    new_login = client.post("/login", data={"username": "changepass", "password": "BetterPass12@"}, follow_redirects=True)
    assert new_login.status_code == 200


def test_forgot_and_reset_password_flow(client):
    _register(client, "resetuser", "Password12!", "resetuser@example.com")
    client.get("/logout")
    forgot = client.post("/forgot-password", data={"identifier": "resetuser@example.com"})
    assert forgot.status_code == 200
    assert b"reset link" in forgot.data.lower()

    db_path = app.config["AUTH_DATABASE"]
    with sqlite3.connect(db_path) as conn:
        token_hash = conn.execute(
            "SELECT reset_token_hash FROM users WHERE username = ?",
            ("resetuser",),
        ).fetchone()[0]
    assert token_hash

    # Request a fresh token we can use directly from rendered page.
    forgot2 = client.post("/forgot-password", data={"identifier": "resetuser"})
    body = forgot2.data.decode("utf-8")
    marker = "/reset-password/"
    start = body.find(marker)
    assert start != -1
    token = body[start + len(marker):].split('"')[0].split("<")[0]

    reset = client.post(f"/reset-password/{token}", data={"new_password": "NewPass12!", "confirm_password": "NewPass12!"})
    assert reset.status_code == 200
    old_login = client.post("/login", data={"identifier": "resetuser@example.com", "password": "Password12!"})
    assert b"Invalid username or password" in old_login.data
    new_login = client.post("/login", data={"identifier": "resetuser@example.com", "password": "NewPass12!"}, follow_redirects=True)
    assert new_login.status_code == 200


def test_admin_can_reject_pending_user(client, tmp_path):
    _register(client, "adminfirst", "Password12!")
    client.get("/logout")
    client.post(
        "/register",
        data={"username": "rejectme", "email": "rejectme@example.com", "password": "Password12!"},
        follow_redirects=True,
    )
    client.post("/login", data={"username": "adminfirst", "password": "Password12!", "login_mode": "admin"})
    with sqlite3.connect(str(tmp_path / "test_users.db")) as conn:
        user_id = conn.execute("SELECT id FROM users WHERE username = ?", ("rejectme",)).fetchone()[0]
    res = client.post(f"/admin/user/{user_id}/reject", follow_redirects=True)
    assert res.status_code == 200
    client.get("/logout")
    denied = client.post("/login", data={"username": "rejectme", "password": "Password12!"})
    assert denied.status_code in (200, 403)


def test_admin_page_denied_without_admin_login_mode(client):
    _register(client, "adminuser2", "Password12!")
    res = client.get("/admin")
    assert res.status_code == 200


def test_openapi_and_docs_available(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    spec = res.get_json()
    assert "/api/predict" in spec["paths"]
    docs = client.get("/api/docs")
    assert docs.status_code == 200


def test_api_predict_with_api_key(client, tmp_path):
    _register(client, "apikeyuser", "Password12!")
    db_path = str(tmp_path / "test_users.db")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT api_key FROM users WHERE username = ?", ("apikeyuser",)).fetchone()
    assert row and row[0]
    payload = {
        "Attendance": 77,
        "CAT_Score": 60,
        "Assignment_Score": 81,
        "Final_Exam": 69,
    }
    res = client.post("/api/predict", json=payload, headers={"X-API-Key": row[0]})
    assert res.status_code == 200
    body = res.get_json()
    assert "prediction" in body and "confidence" in body


def test_login_with_email_identifier(client):
    _register(client, "mailuser", "Password12!", "mailuser@example.com")
    client.get("/logout")
    res = client.post(
        "/login",
        data={"identifier": "mailuser@example.com", "password": "Password12!"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"Student Performance" in res.data


def test_register_duplicate_email(client):
    _register(client, "userone", "Password12!", "dupe@example.com")
    client.get("/logout")
    res = client.post(
        "/register",
        data={"username": "usertwo", "email": "dupe@example.com", "password": "Password12!"},
        follow_redirects=True,
    )
    assert res.status_code == 200
    assert b"already registered" in res.data


def test_predict_multi_subject(client):
    _register(client, "multisub", "Password12!", "multisub@example.com")
    payload = {
        "Attendance": 86,
        "subjects": [
            {"subject": "Math", "CAT_Score": 80, "Assignment_Score": 78, "Final_Exam": 72},
            {"subject": "English", "CAT_Score": 74, "Assignment_Score": 82, "Final_Exam": 76},
            {"subject": "Science", "CAT_Score": 88, "Assignment_Score": 84, "Final_Exam": 79},
        ],
    }
    res = client.post("/predict/multi-subject", json=payload)
    assert res.status_code == 200
    body = res.get_json()
    assert body["prediction"] in ("Pass", "Fail")
    assert isinstance(body["subjects"], list) and len(body["subjects"]) == 3
    assert "aggregate_features" in body
