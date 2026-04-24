#!/usr/bin/env python3
"""
FILE(2) — OpenEnv Backend Entry Point
Run with: python run.py
Or: uvicorn run:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from backend.api.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV", "production") == "development",
        workers=1,
        log_level="info",
    )
