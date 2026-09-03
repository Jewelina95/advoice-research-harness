from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from advoice.demo import (
    PUBLIC_DEMO_CASES,
    analyze_base64_wav,
    analyze_local_manifest_case,
    analyze_public_case,
    parse_byte_range,
    public_case_summaries,
    write_public_demo_bundle,
)


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"
COHORT_RESULT = OUTPUT / "adress_2020_cohort_summary.json"
LOCAL_MANIFEST = ROOT / "local_cases.json"
LOCAL_OUTPUT = ROOT / "local_output"
DEMO_ANALYZER_CODE = Path(analyze_local_manifest_case.__code__.co_filename)


def ensure_public_bundle() -> None:
    required_audio = [ASSETS / str(case["audio_file"]) for case in PUBLIC_DEMO_CASES.values()]
    if not all(path.exists() for path in required_audio):
        from generate_sample import main as generate_samples

        generate_samples()
    write_public_demo_bundle(ASSETS, OUTPUT)


def local_cases() -> dict[str, dict]:
    if not LOCAL_MANIFEST.exists():
        return {}
    payload = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    return {str(case["demo_case_id"]): case for case in payload.get("cases", [])}


def _local_case_summary(case: dict) -> dict:
    def value(key: str, legacy: str, default: object) -> object:
        return case[key] if key in case else case.get(legacy, default)

    return {
        "case_id": str(case["demo_case_id"]),
        "dataset_id": str(case["dataset_id"]),
        "channel_id": str(case["channel_id"]),
        "channel_name": str(value("channel_name", "channel_name_zh", case["channel_id"])),
        "task_name": str(value("task_name", "task_name_zh", case.get("task_type", "Task"))),
        "description": str(value("description", "description_zh", "Restricted local case")),
        "evidence_focus": list(value("evidence_focus", "evidence_focus_zh", [])),
        "research_label": str(case.get("research_label", "UNAVAILABLE")),
        "language": str(case.get("language", "")),
        "data_scope": "local_restricted_not_for_redistribution",
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


def case_result(case_id: str) -> dict:
    if case_id in PUBLIC_DEMO_CASES:
        path = OUTPUT / f"{case_id}.json"
        if not path.exists():
            return analyze_public_case(case_id, ASSETS)
        return json.loads(path.read_text(encoding="utf-8"))
    return analyze_local(case_id)


def case_audio(case_id: str) -> Path:
    if case_id in PUBLIC_DEMO_CASES:
        return ASSETS / str(PUBLIC_DEMO_CASES[case_id]["audio_file"])
    case = local_cases().get(case_id)
    if case is None:
        raise KeyError(case_id)
    return Path(str(case["audio_path"]))


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
        self.send_response(HTTPStatus.PARTIAL_CONTENT.value if selected else HTTPStatus.OK.value)
        self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
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
            return

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/cases":
            cases = public_case_summaries()
            cases.extend(_local_case_summary(case) for case in local_cases().values())
            self._json({"cases": cases, "local_restricted_cases_available": bool(local_cases())})
            return
        if path.startswith("/api/case/"):
            try:
                self._json(case_result(path.rsplit("/", 1)[-1]))
            except KeyError:
                self._json({"error": "case_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/api/case-audio/"):
            try:
                self._file(case_audio(path.rsplit("/", 1)[-1]))
            except KeyError:
                self._json({"error": "case_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if path == "/api/cohort":
            self._json(json.loads(COHORT_RESULT.read_text(encoding="utf-8")))
            return
        if path.startswith("/assets/"):
            selected = ASSETS / path.rsplit("/", 1)[-1]
            if selected.is_file():
                self._file(selected)
                return
        if path.startswith("/output/"):
            selected = OUTPUT / path.rsplit("/", 1)[-1]
            if selected.is_file():
                self._json(json.loads(selected.read_text(encoding="utf-8")))
                return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/run-case/"):
            case_id = path.rsplit("/", 1)[-1]
            try:
                if case_id in PUBLIC_DEMO_CASES:
                    self._json(analyze_public_case(case_id, ASSETS))
                else:
                    self._json(analyze_local(case_id))
            except KeyError:
                self._json({"error": "case_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if path != "/api/analyze":
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
    ensure_public_bundle()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"ADvoice demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
