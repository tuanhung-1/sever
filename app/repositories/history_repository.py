from __future__ import annotations

import json
import os
import time
from typing import Any


def append_jsonl_record(file_path: str, packet: dict[str, Any], lock: Any = None) -> None:
    record = {"saved_at": int(time.time()), **packet}

    def write_record() -> None:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if lock is None:
        write_record()
        return

    with lock:
        write_record()


def read_jsonl_records(file_path: str, limit: int = 50, lock: Any = None) -> list[dict[str, Any]]:
    if limit <= 0 or not os.path.exists(file_path):
        return []

    def read_lines() -> list[str]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.readlines()

    lines = read_lines() if lock is None else _read_lines_locked(lock, read_lines)
    records = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records[-limit:][::-1]


def _read_lines_locked(lock: Any, reader) -> list[str]:
    with lock:
        return reader()
