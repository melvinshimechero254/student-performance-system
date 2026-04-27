# Student Performance Prediction System

A Flask web application that predicts student pass/fail outcomes using a trained machine learning model.

## Project Structure

```
student_performance_system/
├── backend/
│   ├── app.py                  # Flask application
│   └── templates/
│       └── index.html          # Dashboard UI
├── data/
│   └── student_performance.csv # Training dataset
├── models/
│   └── student_performance_model.pkl
├── notebooks/
│   ├── data_simulation.ipynb
│   └── model_training.ipynb
├── tests/
│   └── test_app.py
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev + notebook dependencies
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
python app.py
# Open http://127.0.0.1:5000
```

## Test

```bash
pip install pytest
pytest tests/
```

## API Endpoints

| Method | Endpoint    | Description              |
|--------|-------------|--------------------------|
| GET    | `/`         | Dashboard UI             |
| POST   | `/predict`  | Single student prediction|
| POST   | `/predict/multi-subject`  | Multi-subject prediction (aggregated) |
| POST   | `/upload`   | Bulk CSV prediction      |
| GET    | `/download` | Download results CSV     |
| GET    | `/history`  | User prediction history  |
| GET    | `/admin`    | Admin dashboard (admin only) |
| GET    | `/explainability` | Feature importance JSON |
| POST   | `/api/predict` | API-key protected prediction |
| GET    | `/openapi.json` | OpenAPI schema |
| GET    | `/api/docs` | Swagger UI documentation |

### `/predict` payload

```json
{
  "Attendance": 85,
  "CAT_Score": 72,
  "Assignment_Score": 78,
  "Final_Exam": 65
}
```

### `/predict/multi-subject` payload

Use this when a learner has multiple subjects and you want one overall outcome.
The API averages CAT, Assignment, and Final Exam across provided subjects, combines
with Attendance, and returns one overall pass/fail prediction + confidence.

```json
{
  "Attendance": 86,
  "subjects": [
    { "subject": "Math", "CAT_Score": 80, "Assignment_Score": 78, "Final_Exam": 72 },
    { "subject": "English", "CAT_Score": 74, "Assignment_Score": 82, "Final_Exam": 76 }
  ]
}
```

## Deployment Notes

- Set a strong secret key in production:
  - `FLASK_SECRET_KEY` should be a long random value and never committed to git.
- Enable secure session cookies behind HTTPS:
  - Set `SESSION_COOKIE_SECURE=True` in your deployment environment.
- Recommended workflow:
  - Make changes locally/staging first, run tests, then deploy.
  - Avoid editing code directly on production servers.
- Logs:
  - Structured app events are written to `backend/logs/app.log` with rotation.

## Environment Configuration

Copy `.env.example` to `.env` and set real values before deployment.

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

### Required for production security

- `FLASK_SECRET_KEY`
  - Required. Use a strong random secret (at least 32 bytes).
- `SHOW_RESET_LINKS`
  - Keep `0` in production. Only set `1` for local demos/testing.

### SMTP (forgot password email)

Set these values to enable password reset emails:

- `SMTP_HOST` (e.g. `smtp.gmail.com`, `smtp.office365.com`)
- `SMTP_PORT` (default `587`)
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SMTP_SENDER` (from address; defaults to `SMTP_USER` if omitted)

If SMTP is not configured, forgot-password requests will still be accepted,
but reset links will not be emailed in production.
