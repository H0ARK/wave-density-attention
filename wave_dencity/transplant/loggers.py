from __future__ import annotations

import json
import sqlite3
import time
from typing import Iterable


class StepLogger:
    def __init__(
        self,
        log_path: str,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        sqlite_path: str | None = None,
        sqlite_table: str = "training_log",
        sqlite_every: int = 1,
    ):
        self.log_path = log_path
        self.f = open(log_path, "a", encoding="utf-8")
        self.include = set(include) if include else None
        self.exclude = set(exclude) if exclude else set()
        self.sqlite_path = sqlite_path
        self.sqlite_table = sqlite_table
        self.sqlite_every = max(1, int(sqlite_every))
        self._sqlite_conn = None
        if sqlite_path:
            self._sqlite_conn = sqlite3.connect(sqlite_path)
            self._ensure_sqlite_table()

    def _filter_data(self, data: dict) -> dict:
        if self.include is None:
            filtered = dict(data)
        else:
            filtered = {k: data[k] for k in self.include if k in data}
        if self.exclude:
            for key in list(filtered.keys()):
                if key in self.exclude:
                    filtered.pop(key, None)
        return filtered

    def _ensure_sqlite_table(self) -> None:
        if self._sqlite_conn is None:
            return
        self._sqlite_conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.sqlite_table} ("
            "step INTEGER PRIMARY KEY, time REAL, data TEXT)"
        )
        self._sqlite_conn.commit()

    def _write_sqlite(self, step: int, entry: dict) -> None:
        if self._sqlite_conn is None:
            return
        payload = json.dumps(entry)
        self._sqlite_conn.execute(
            f"INSERT OR REPLACE INTO {self.sqlite_table} (step, time, data) VALUES (?, ?, ?)",
            (step, entry["time"], payload),
        )
        self._sqlite_conn.commit()

    def log(self, step: int, data: dict):
        filtered = self._filter_data(data)
        entry = {"step": step, "time": time.time(), **filtered}
        self.f.write(json.dumps(entry) + "\n")
        self.f.flush()
        if self._sqlite_conn is not None and (step % self.sqlite_every == 0):
            self._write_sqlite(step, entry)

    def close(self):
        self.f.close()
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
