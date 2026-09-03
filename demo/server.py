from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from advoice.demo import analyze_base64_wav, write_demo_result


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
SAMPLE_AUDIO = ROOT / "assets" / "synthetic_picture_description.wav"
SAMPLE_TRANSCRIPT = ROOT / "assets" / "synthetic_picture_description.txt"
SAMPLE_RESULT = ROOT / "output" / "synthetic_case_result.json"


def ensure_sample() -> None:
    if not SAMPLE_AUDIO.exists():
        from generate_sample import main as generate_sample

        generate_sample()
    write_demo_result(SAMPLE_AUDIO, SAMPLE_TRANSCRIPT, SAMPLE_RESULT)


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(WEB), **kwargs)

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/sample":
            self._json(json.loads(SAMPLE_RESULT.read_text(encoding="utf-8")))
            return
        if self.path == "/api/sample-audio":
            raw = SAMPLE_AUDIO.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/assets/synthetic_picture_description.wav":
            raw = SAMPLE_AUDIO.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/output/synthetic_case_result.json":
            self._json(json.loads(SAMPLE_RESULT.read_text(encoding="utf-8")))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 28 * 1024 * 1024:
                raise ValueError("Invalid or oversized request.")
            payload = json.loads(self.rfile.read(length))
            self._json(analyze_base64_wav(payload))
        except Exception as exc:
            self._json({"error": type(exc).__name__, "message": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local ADvoice reproducibility demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ensure_sample()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"ADvoice demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
