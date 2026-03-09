from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2

from detector import DetectionResult


@dataclass
class DatabaseConfig:
    backend: str = "sqlite"  # "sqlite", "postgres", "influx"
    sqlite_path: str = "potato_qc.db"

    # Placeholders for future extension
    postgres_dsn: str = ""
    influx_url: str = ""
    influx_token: str = ""
    influx_org: str = ""
    influx_bucket: str = ""

    snapshot_dir: str = "snapshots"


class EventLogger:
    """
    Database logging for detection events and snapshots.

    Fully implemented for SQLite. PostgreSQL and InfluxDB can be added
    via the same interface if required.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()

        if self.config.backend != "sqlite":
            # For this implementation we fully support SQLite and expose a
            # clear error for other backends to keep dependencies minimal.
            raise NotImplementedError(
                f"Backend '{self.config.backend}' is not implemented in this sample. "
                "Use 'sqlite' or extend EventLogger with your desired backend."
            )

        os.makedirs(self.config.snapshot_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                object_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                track_id INTEGER,
                snapshot_path TEXT
            )
            """
        )
        self.conn.commit()

    def log_event(
        self,
        det: DetectionResult,
        frame,
        save_snapshot: bool = True,
    ) -> None:
        """
        Log an event to the database and optionally store an image snapshot.
        """
        ts = datetime.utcnow().isoformat()
        snapshot_path: Optional[str] = None

        if save_snapshot:
            filename = f"{ts.replace(':', '_').replace('.', '_')}_id{det.track_id or 0}.jpg"
            snapshot_path = os.path.join(self.config.snapshot_dir, filename)
            cv2.imwrite(snapshot_path, frame)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO detection_events(timestamp, object_class, confidence, track_id, snapshot_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, det.cls_name, float(det.confidence), det.track_id, snapshot_path),
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()

