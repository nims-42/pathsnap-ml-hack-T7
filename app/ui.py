"""Evaluator UI (runs on Laptop 2). Plain HTML + Leaflet, served by the standard library.

    python app/ui.py                 then open http://127.0.0.1:8080

The API address comes from app/config.json ("api_url") or the PATHSNAP_API env var,
e.g. http://192.168.43.10:8000. It can also be edited in the page itself.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PORT = int(os.environ.get("PATHSNAP_UI_PORT", 8080))
API = os.environ.get("PATHSNAP_API") or json.load(open(HERE / "config.json"))["api_url"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/config":
            self._send(json.dumps({"api_url": API}).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"PathSnap UI on http://127.0.0.1:{PORT}   (API: {API})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
