"""Detection association, deduplication and SQLite storage tests."""

import csv

from forest_bringup.detection_logic import (
    Detection2D, DetectionHistory, associate, iou)
from forest_bringup.storage import DetectionRecord, DetectionStore


def _det(cx, cy, stamp=1.0, source="rgb", cls="person", conf=0.9, w=40, h=80):
    return Detection2D(cls, conf, cx, cy, w, h, stamp, source)


def test_iou():
    a, b = _det(100, 100), _det(100, 100)
    assert iou(a, b) == 1.0
    c = _det(500, 500)
    assert iou(a, c) == 0.0


def test_associate_rgb_thermal_match():
    rgb = [_det(100, 100, source="rgb")]
    thm = [_det(104, 102, source="thermal", conf=0.8)]
    fused = associate(rgb, thm, max_time_delta=0.2, min_iou=0.2)
    assert len(fused) == 1 and fused[0].sources == ("rgb", "thermal")


def test_associate_keeps_unmatched_and_respects_time():
    rgb = [_det(100, 100, stamp=1.0), _det(300, 300, stamp=1.0)]
    thm = [_det(100, 100, stamp=3.0, source="thermal")]
    fused = associate(rgb, thm, max_time_delta=0.2)
    assert len(fused) == 3
    assert {f.sources for f in fused} == {("rgb",), ("thermal",)}


def test_associate_class_mismatch_with_aliases():
    rgb = [_det(100, 100, cls="deer")]
    thm = [_det(100, 100, cls="elk", source="thermal")]
    assert len(associate(rgb, thm)) == 2
    merged = associate(rgb, thm, aliases={"elk": "deer"})
    assert len(merged) == 1


def test_detection_history_dedup():
    hist = DetectionHistory(radius=2.0, seconds=5.0)
    assert hist.check_and_add("person", (0, 0, 0), 10.0) is True
    assert hist.check_and_add("person", (0.5, 0, 0), 11.0) is False
    assert hist.check_and_add("person", (10, 0, 0), 11.0) is True
    assert hist.check_and_add("person", (0.5, 0, 0), 20.0) is True


def test_sqlite_store_and_csv(tmp_path):
    db = tmp_path / "detections.sqlite3"
    csv_path = tmp_path / "out" / "detections.csv"
    with DetectionStore(db, dedup_radius=1.0, dedup_seconds=5.0) as store:
        rec = DetectionRecord("person", 0.9, 1, 2, 3, 100.0,
                              0, 0, 5, "rgb", 10, 20, 30, 40, "frame.jpg")
        assert store.insert_unique(rec) is not None
        assert store.count() == 1
        # Same target next frame -> suppressed.
        rec2 = DetectionRecord("person", 0.8, 1.2, 2.0, 3.0, 100.1,
                               0, 0, 5, "thermal")
        assert store.insert_unique(rec2) is None
        assert store.count() == 1
        # Far away -> stored.
        rec3 = DetectionRecord("deer", 0.7, 20, 20, 0, 101.0, 0, 0, 5, "rgb")
        assert store.insert_unique(rec3) is not None
        assert store.count() == 2
        assert len(store.fetch_all(class_name="deer")) == 1
        assert len(store.fetch_records()) == 2
        store.export_csv(csv_path)
    with open(csv_path) as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "class_name"
    assert len(rows) == 3
