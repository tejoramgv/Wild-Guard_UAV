"""Persistent detection storage (SQLite) with CSV export and deduplication."""

import csv
import sqlite3
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    stamp REAL NOT NULL,
    drone_x REAL NOT NULL,
    drone_y REAL NOT NULL,
    drone_z REAL NOT NULL,
    source TEXT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    image_path TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_class_stamp
    ON detections (class_name, stamp);
"""

COLUMNS = ["class_name", "confidence", "x", "y", "z", "stamp",
           "drone_x", "drone_y", "drone_z", "source",
           "bbox_x", "bbox_y", "bbox_w", "bbox_h", "image_path"]


@dataclass
class DetectionRecord:
    class_name: str
    confidence: float
    x: float
    y: float
    z: float
    stamp: float
    drone_x: float
    drone_y: float
    drone_z: float
    source: str
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 0
    bbox_h: int = 0
    image_path: str = ""

    def row(self):
        data = asdict(self)
        return tuple(data[c] for c in COLUMNS)


class DetectionStore:
    """SQLite-backed detection history.

    ``insert_unique`` suppresses repeat frames of the same target: a detection
    is a duplicate when the class matches, the estimate is within
    ``dedup_radius`` metres, and the previous accepted stamp is within
    ``dedup_seconds``. This keeps a stationary animal from creating one row per
    camera frame while still recording it again after a real gap.
    """

    def __init__(self, path="detections.sqlite3", dedup_radius=1.5,
                 dedup_seconds=5.0):
        self.path = str(path)
        self.dedup_radius = dedup_radius
        self.dedup_seconds = dedup_seconds
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _is_duplicate(self, rec):
        lo = rec.stamp - self.dedup_seconds
        rows = self.conn.execute(
            "SELECT x, y, z FROM detections WHERE class_name = ? AND stamp >= ? AND stamp <= ?",
            (rec.class_name, lo, rec.stamp)).fetchall()
        for x, y, z in rows:
            if math.dist((rec.x, rec.y, rec.z), (x, y, z)) <= self.dedup_radius:
                return True
        return False

    def insert(self, rec):
        self.conn.execute(
            "INSERT INTO detections (" + ",".join(COLUMNS) + ",created_at) VALUES ("
            + ",".join(["?"] * len(COLUMNS)) + ",?)",
            rec.row() + (time.time(),))
        self.conn.commit()
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def insert_unique(self, rec):
        """Insert unless a recent nearby detection of the same class exists.

        Returns the new row id, or ``None`` when suppressed as a duplicate.
        """
        if self._is_duplicate(rec):
            return None
        return self.insert(rec)

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

    def counts_by_class(self):
        rows = self.conn.execute(
            "SELECT class_name, COUNT(*) FROM detections GROUP BY class_name "
            "ORDER BY COUNT(*) DESC").fetchall()
        return {name: count for name, count in rows}

    def fetch_all(self, class_name=None, since=None, until=None):
        query = "SELECT " + ",".join(COLUMNS) + " FROM detections"
        clauses, params = [], []
        if class_name is not None:
            clauses.append("class_name = ?")
            params.append(class_name)
        if since is not None:
            clauses.append("stamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("stamp <= ?")
            params.append(until)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY stamp, id"
        return self.conn.execute(query, params).fetchall()

    def fetch_records(self, **kwargs):
        return [DetectionRecord(*row) for row in self.fetch_all(**kwargs)]

    def export_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(self.fetch_all())
        return str(path)
