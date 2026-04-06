# Fixora

Fixora is an AI-enhanced developer learning platform that converts programming errors into targeted learning experiences. The project includes a Django backend API, user authentication, bug analysis workflows, and an intelligence layer that classifies errors, identifies missing concepts, and generates practice challenges.

## Project Structure

- `backend/` - Django backend application
  - `accounts/` - user registration, authentication, onboarding, and profile handling
  - `bugs/` - bug submission and analysis endpoints
  - `backend/` - Django project configuration
- `bug_intelligence/` - AI pipeline for error classification and challenge generation
  - `analyze.py` - primary analysis pipeline
  - `main.py` - command-line test harness for interactive analysis
  - `core/` - classifier, concept detection, evaluator, and challenge generator
  - `data/` - reference concepts and error mapping resources

## Key Features

- Error classification and categorization
- Root concept identification for programming errors
- Personalized challenge generation based on user input
- REST API endpoints for analysis and account management
- Basic user onboarding and streak tracking via profile updates

## Backend API

The Django backend exposes these main endpoints:

- `POST /api/accounts/register/` - register a new user
- `POST /api/accounts/login/` - obtain JWT access and refresh tokens
- `POST /api/accounts/refresh/` - refresh JWT token
- `POST /api/accounts/onboarding/` - complete user onboarding with experience and language preferences
- `POST /api/analyze/` - submit a bug or error message for analysis

## How It Works

1. A user submits a bug or error text.
2. The system classifies the error type and confidence.
3. It determines the likely missing programming concept.
4. It generates a practice challenge with a code snippet and hint.
5. User progress is tracked via profile updates and streaks.

## Running the Project

1. Create and activate a Python virtual environment.
2. Install required dependencies:

```bash
pip install django djangorestframework djangorestframework-simplejwt
```

3. Run Django migrations:

```bash
cd backend
python manage.py migrate
```

4. Start the development server:

```bash
python manage.py runserver
```

## Testing the AI Layer

To run the standalone AI pipeline from `bug_intelligence`:

```bash
cd bug_intelligence
python main.py
```

This will prompt for an error, language, and user level, then show classification, root concept, and a generated challenge.

## Notes

- The backend uses SQLite by default via `backend/db.sqlite3`.
- Django settings indicate the project was generated with Django 6.0.1.
- The AI pipeline currently supports Python, C, and C++ for error analysis.

## Contribution

- Extend the AI pipeline with additional bug categories and concepts.
- Add frontend integration for submitting errors and reviewing challenges.
- Add tests for API endpoints and the analysis pipeline.

