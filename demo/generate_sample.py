from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

SAMPLE_PATTERNS: dict[str, list[tuple[float, float, float]]] = {
    "synthetic_clinical_interview.wav": [
        (1.0, 150.0, 0.17), (0.4, 0.0, 0.0), (2.0, 172.0, 0.20),
        (0.8, 0.0, 0.0), (1.3, 165.0, 0.18), (0.5, 0.0, 0.0),
    ] * 4,
    "synthetic_picture_description.wav": [
        (1.8, 170.0, 0.22), (0.7, 0.0, 0.0), (2.4, 185.0, 0.20),
        (1.1, 0.0, 0.0), (1.5, 160.0, 0.18), (0.5, 0.0, 0.0),
        (2.1, 195.0, 0.21),
    ] * 3,
    "synthetic_structured_task.wav": [
        (0.8, 190.0, 0.20), (0.25, 0.0, 0.0), (0.7, 205.0, 0.18),
        (0.35, 0.0, 0.0), (0.9, 180.0, 0.19), (0.55, 0.0, 0.0),
    ] * 6,
    "synthetic_public_speech.wav": [
        (2.8, 135.0, 0.15), (0.3, 0.0, 0.0), (3.2, 155.0, 0.23),
        (0.45, 0.0, 0.0), (2.5, 145.0, 0.12), (0.25, 0.0, 0.0),
    ] * 3,
}

TRANSCRIPTS = {
    "synthetic_clinical_interview.txt": (
        "Interviewer: Tell me about your morning.\n"
        "Participant: I made tea, then I looked for my keys. I paused because I could not remember where I had put them.\n"
        "Interviewer: What happened next?\n"
        "Participant: I found them near the door and walked to the shop."
    ),
    "synthetic_picture_description.txt": (
        "The boy is reaching for a cookie and the stool is tipping. The mother is washing dishes while the water runs over the sink. "
        "The girl is standing beside him and the curtains are moving near the window."
    ),
    "synthetic_structured_task.txt": (
        "Dog, cat, horse, sheep, cow, goat, rabbit, tiger, lion, elephant, giraffe, monkey, bear, fox, deer, zebra."
    ),
    "synthetic_public_speech.txt": (
        "Today I want to describe the work we completed over the past year. We met with several groups, reviewed the evidence, "
        "and changed the plan when the results did not support the first approach."
    ),
}


def _tone(sample_rate: int, seconds: float, frequency: float, amplitude: float) -> list[float]:
    count = int(sample_rate * seconds)
    return [
        amplitude
        * (0.72 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / max(count - 1, 1)))
        for index in range(count)
    ]


def _silence(sample_rate: int, seconds: float) -> list[float]:
    return [0.0] * int(sample_rate * seconds)


def _write_wav(path: Path, pattern: list[tuple[float, float, float]]) -> None:
    sample_rate = 16_000
    signal: list[float] = []
    for seconds, frequency, amplitude in pattern:
        signal.extend(
            _silence(sample_rate, seconds)
            if frequency == 0.0
            else _tone(sample_rate, seconds, frequency, amplitude)
        )
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(
            b"".join(
                struct.pack("<h", max(-32768, min(32767, int(value * 32767))))
                for value in signal
            )
        )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for filename, pattern in SAMPLE_PATTERNS.items():
        _write_wav(ASSETS / filename, pattern)
    for filename, transcript in TRANSCRIPTS.items():
        (ASSETS / filename).write_text(transcript + "\n", encoding="utf-8")
    print(f"Generated {len(SAMPLE_PATTERNS)} synthetic cases in {ASSETS}")


if __name__ == "__main__":
    main()
