"""ASGI entry point.

Run with: ``uvicorn services.api.app:app --reload``.
Importing ``services.api.main`` has no side effects, which keeps tests isolated.
"""

from services.api.main import create_app

app = create_app()
