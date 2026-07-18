"""
api/index.py
------------
Entry point Vercel uses to run this Flask app as a serverless function.

Vercel's Python runtime looks inside the /api folder for files that expose
a WSGI-compatible variable called `app`. Here we just import the real
Flask app object that's defined in the project's app.py (one level up)
and re-expose it so Vercel can serve it.
"""

import os
import sys

# Make sure the project root (one level up from /api) is on the import path,
# so we can import app.py, recommender.py, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402  (the actual Flask application)

# Vercel's Python runtime expects a WSGI app object named `app` in this file.
