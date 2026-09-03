from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from advoice.demo import (
    analyze_base64_wav,
    analyze_local_manifest_case,
    parse_byte_range,
    write_demo_result,
)


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
SAMPLE_AUDIO = ROOT / "assets" / "synthetic_picture_description.wav"
SAMPLE_TRANSCRIPT = ROOT / "assets" / "synthetic_picture_description.txt"
SAMPLE_RESULT = ROOT / "output" / "synthetic_case_result.json"
LOCAL_MANIFEST = ROOT / "local_cases.json"
LOCAL_OUTPUT = ROOT / "local_output"
DEMO_ANALYZER_CODE = Path(analyze_local_manifest_case.__code__.co_filename)


def ensure_sample() -> None:
    if not SAMPLE_AUDIO.exists():
        from generate_sample import main as generate_sample

        generate_sample()
    write_demo_result(SAMPLE_AUDIO, SAMPLE_TRANSCRIPT, SAMPLE_RESULT)


def local_cases() -> dict[str, dict]:
    if not LOCAL_MANIFEST.exists():
        return {}
    payload = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    return {str(case["demo_case_id"]): case for case in payload.get("cases", [])}


def public_case_summary(case: dict) -> dict:
    return {
        key: case[key]
        for key in (
            "demo_case_id",
            "dataset_id",
            "channel_id",
            "channel_name_zh",
            "task_name_zh",
            "description_zh",
            "evidence_focus_zh",
            "research_label",
            "language",
        )
    }


def analyze_local(case_id: str) -> dict:
    case = local_cases().get(case_id)
    if case is None:
        raise KeyError(case_id)
    output = LOCAL_OUTPUT / f"{case_id}.json"
    dependencies = [Path(str(case["audio_path"])), LOCAL_MANIFEST, DEMO_ANALYZER_CODE]
    transcript_value = str(case.get("transcript_path", "")).strip()
    if transcript_value:
        dependencies.append(Path(transcript_value))
    newest_dependency = max(path.stat().st_mtime for path in dependencies if path.exists())
    if not output.exists() or output.stat().st_mtime < newest_dependency:
        result = analyze_local_manifest_case(case)
        LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(output.read_text(encoding="utf-8"))


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

    def _file(self, file_path: Path) -> None:
        size = file_path.stat().st_size
        try:
            selected = parse_byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE.value)
            return
        start, end = selected or (0, size - 1)
        self.send_response(
            HTTPStatus.PARTIAL_CONTENT.value if selected else HTTPStatus.OK.value
        )
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
        )
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if selected:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with file_path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers commonly cancel the initial full media request before
            # reopening it as a byte-range request for seeking.
            return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/sample":
            self._json(json.loads(SAMPLE_RESULT.read_text(encoding="utf-8")))
            return
        if path == "/api/cases":
            cases = [
                {
                    "demo_case_id": "synthetic_case_001",
                    "dataset_id": "PUBLIC_SYNTHETIC_DEMO",
                    "channel_id": "public_demo",
                    "channel_name_zh": "合成公开案例",
                    "task_name_zh": "公开流程验证",
                    "description_zh": "不含患者数据，用于检查公开代码和回溯界面。",
                    "evidence_focus_zh": ["指标证据", "状态卡", "片段回溯"],
                    "research_label": "UNLABELED",
                    "language": "en",
                }
            ]
            cases.extend(public_case_summary(case) for case in local_cases().values())
            self._json({"cases": cases, "local_restricted_cases_available": bool(local_cases())})
            return
        if path.startswith("/api/case/"):
            try:
                self._json(analyze_local(path.rsplit("/", 1)[-1]))
            except KeyError:
                self._json({"error": "case_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/api/case-audio/"):
            case_id = path.rsplit("/", 1)[-1]
            case = local_cases().get(case_id)
            if case is None:
                self._json({"error": "case_not_found"}, HTTPStatus.NOT_FOUND)
                return
            audio_path = Path(str(case["audio_path"]))
            self._file(audio_path)
            return
        if path == "/api/sample-audio":
            raw = SAMPLE_AUDIO.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/assets/synthetic_picture_description.wav":
            raw = SAMPLE_AUDIO.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/output/synthetic_case_result.json":
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
