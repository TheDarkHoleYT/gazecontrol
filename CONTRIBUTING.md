# Contributing to GazeControl

## Setup

```bash
git clone <repo-url> gazecontrol && cd gazecontrol
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Branch off `main`: `git checkout -b feat/my-feature`
2. Write code + tests
3. `pre-commit run --all-files` — lint, format, type-check must pass
4. `pytest --cov=gazecontrol --cov-fail-under=70` — coverage must stay ≥ 70 %
5. Open PR against `main`

## Code standards

- Python 3.11+, type hints required for all public APIs
- Docstrings in Google style
- Log in English; user-facing messages may be in Italian
- No `print()` in library code — use `logging.getLogger(__name__)`
- No magic numbers — promote to `settings.py`

## Running tests

```bash
# All tests (Win32 tests skipped on Linux)
pytest

# With coverage report
pytest --cov=gazecontrol --cov-report=html

# Only fast unit tests
pytest -m "not slow and not integration"
```
