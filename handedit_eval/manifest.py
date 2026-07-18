from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .bank import enrich_record


@dataclass
class ManifestRecord:
    data: Dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data.get("id", ""))


def read_manifest_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            records.append(enrich_record(row))
    return records


def write_manifest_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_record(record: Dict[str, Any]) -> None:
    required = ["id", "src_path", "pred_path"]
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"Manifest record is missing required fields: {missing}")


def iter_validated_records(path: str) -> Iterator[Dict[str, Any]]:
    for row in read_manifest_jsonl(path):
        validate_record(row)
        yield row
