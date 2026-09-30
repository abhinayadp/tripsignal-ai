import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chatbot import TravelDisruptionAssistant

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
assistant = TravelDisruptionAssistant()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if self.path == "/health":
            self._send(200, json.dumps({"status": "ok", "documents": len(assistant.documents)}).encode(), "application/json")
            return
        if self.path in routes:
            filename, content_type = routes[self.path]
            self._send(200, (WEB_DIR / filename).read_bytes(), content_type)
            return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send(404, b"Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        result = assistant.answer_with_trace(payload.get("query", ""))
        self._send(200, json.dumps(result).encode(), "application/json")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"TripSignal AI running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
