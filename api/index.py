from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.web_app import AppHandler  # noqa: E402


class handler(AppHandler, BaseHTTPRequestHandler):
    pass
