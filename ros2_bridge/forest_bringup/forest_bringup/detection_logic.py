"""2D detection association (RGB + thermal) and duplicate suppression."""

from dataclasses import dataclass
import math


@dataclass
class Detection2D:
    class_name: str
    confidence: float
    cx: float
    cy: float
    w: float
    h: float
    stamp: float
    source: str
    image_path: str = ""

    def bbox(self):
        """(x1, y1, x2, y2) in pixels."""
        return (self.cx - self.w / 2.0, self.cy - self.h / 2.0,
                self.cx + self.w / 2.0, self.cy + self.h / 2.0)


@dataclass
class FusedDetection:
    class_name: str
    confidence: float
    detections: tuple = ()
    sources: tuple = ()
    stamp: float = 0.0

    @property
    def primary(self):
        return max(self.detections, key=lambda d: d.confidence)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a.bbox()
    bx1, by1, bx2, by2 = b.bbox()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def class_compatible(a, b, aliases=None):
    if a == b:
        return True
    aliases = aliases or {}
    return aliases.get(a, a) == aliases.get(b, b)


def associate(rgb, thermal, max_time_delta=0.15, min_iou=0.2, aliases=None):
    """Greedily match RGB and thermal detections of the same class.

    Matching is timestamp-aware (|dt| <= max_time_delta) and spatially aware
    (IoU >= min_iou). Unmatched detections are still returned as
    single-source records, so nothing is silently dropped.
    """
    remaining = list(thermal)
    fused = []
    for det in sorted(rgb, key=lambda d: -d.confidence):
        best, best_iou = None, 0.0
        for cand in remaining:
            if abs(cand.stamp - det.stamp) > max_time_delta:
                continue
            if not class_compatible(det.class_name, cand.class_name, aliases):
                continue
            score = iou(det, cand)
            if score >= min_iou and score > best_iou:
                best, best_iou = cand, score
        if best is not None:
            remaining.remove(best)
            fused.append(FusedDetection(
                class_name=det.class_name,
                confidence=max(det.confidence, best.confidence),
                detections=(det, best),
                sources=("rgb", "thermal"),
                stamp=min(det.stamp, best.stamp)))
        else:
            fused.append(FusedDetection(det.class_name, det.confidence,
                                        (det,), ("rgb",), det.stamp))
    for det in remaining:
        fused.append(FusedDetection(det.class_name, det.confidence,
                                    (det,), ("thermal",), det.stamp))
    return fused


class DetectionHistory:
    """Suppress repeat world-localised detections of the same target."""

    def __init__(self, radius=1.5, seconds=5.0):
        self.radius = radius
        self.seconds = seconds
        self._tracks = []  # list[(class_name, x, y, z, stamp)]

    def is_duplicate(self, class_name, position, stamp):
        for cls, x, y, z, t in self._tracks:
            if cls != class_name:
                continue
            if stamp - t > self.seconds or t > stamp:
                continue
            if math.dist(position, (x, y, z)) <= self.radius:
                return True
        return False

    def add(self, class_name, position, stamp):
        self._tracks.append((class_name, position[0], position[1], position[2], stamp))
        # Bound memory growth; targets older than the dedup window are useless.
        cutoff = stamp - self.seconds * 4.0
        self._tracks = [t for t in self._tracks if t[4] >= cutoff]

    def check_and_add(self, class_name, position, stamp):
        if self.is_duplicate(class_name, position, stamp):
            return False
        self.add(class_name, position, stamp)
        return True
