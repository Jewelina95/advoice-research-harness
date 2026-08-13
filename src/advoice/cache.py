from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .utils import hash_values, json_dump, json_load, now_utc


@dataclass
class StageResult:
    name: str
    status: str
    fingerprint: str
    outputs: list[str]


class StageCache:
    def __init__(self, cache_dir: Path, force: bool = False) -> None:
        self.cache_dir = cache_dir
        self.force = force
        cache_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        name: str,
        inputs: Iterable[object],
        outputs: Iterable[Path],
        function: Callable[[], None],
    ) -> StageResult:
        output_paths = list(outputs)
        fingerprint = hash_values(inputs)
        marker = self.cache_dir / f"{name}.json"
        previous = json_load(marker, {})
        valid = (
            not self.force
            and previous.get("fingerprint") == fingerprint
            and all(path.exists() for path in output_paths)
        )
        if valid:
            return StageResult(name, "cached", fingerprint, [str(p) for p in output_paths])
        function()
        missing = [str(path) for path in output_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Stage {name} did not create required outputs: {missing}")
        json_dump(
            {
                "stage": name,
                "fingerprint": fingerprint,
                "completed_at_utc": now_utc(),
                "outputs": [str(path) for path in output_paths],
            },
            marker,
        )
        return StageResult(name, "executed", fingerprint, [str(p) for p in output_paths])

