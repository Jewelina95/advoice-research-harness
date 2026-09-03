from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "synthetic_picture_description.wav"


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


def main() -> None:
    sample_rate = 16_000
    signal: list[float] = []
    pattern = [
        (1.8, 170.0, 0.22),
        (0.7, 0.0, 0.0),
        (2.4, 185.0, 0.20),
        (1.1, 0.0, 0.0),
        (1.5, 160.0, 0.18),
        (0.5, 0.0, 0.0),
        (2.1, 195.0, 0.21),
    ] * 3
    for seconds, frequency, amplitude in pattern:
        signal.extend(
            _silence(sample_rate, seconds)
            if frequency == 0.0
            else _tone(sample_rate, seconds, frequency, amplitude)
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(
            b"".join(
                struct.pack("<h", max(-32768, min(32767, int(value * 32767))))
                for value in signal
            )
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
