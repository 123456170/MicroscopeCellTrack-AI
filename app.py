import os
import io
import json
import math
import time
import sqlite3
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =============================================================================
# App configuration
# =============================================================================

st.set_page_config(
    page_title="MicroscopeCellTrack AI",
    page_icon="🦠",
    layout="wide",
)

APP_TITLE = "MicroscopeCellTrack AI"
SIM_BADGE = "SYNTHETIC SIMULATION — NOT REAL SCIENTIFIC EVIDENCE"
DEFAULT_UM_PER_PIXEL = 0.20
FRAGMENT_INTERVAL = 0.18
DB_FILENAME = "microscope_celltrack_history.db"

PAGES = [
    "Live Experiment",
    "Tracking",
    "Measurements",
    "Timeline",
    "Calibration",
    "History",
    "Analytics",
]

DETECTOR_DEFAULTS = {
    "invert": False,
    "min_area": 35,
    "max_area_fraction": 0.25,
    "bg_kernel": 51,
    "clahe_clip": 2.5,
    "open_kernel": 3,
    "close_kernel": 7,
}

TRACK_DEFAULTS = {
    "max_distance": 45.0,
    "max_miss": 12,
    "min_hits": 2,
    "velocity_alpha": 0.65,
    "area_weight": 12.0,
}


# =============================================================================
# Session state helpers
# =============================================================================

def _set_default(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def initialize_defaults() -> None:
    defaults = {
        "page": PAGES[0],
        "playing": True,
        "frame_idx": 0,
        "play_fps": 10,
        "show_trails": True,
        "show_labels": True,
        "show_uncertainty": True,
        "um_per_pixel": DEFAULT_UM_PER_PIXEL,
        "calibration_source": "Deterministic synthetic stage micrometer",
        "video_bytes": None,
        "video_name": "MicroscopeCellTrack_demo.mp4",
        "highlight_anomaly": False,
    }
    for k, v in defaults.items():
        _set_default(k, v)


def get_calibration() -> Tuple[float, bool]:
    try:
        um = float(st.session_state.get("um_per_pixel", DEFAULT_UM_PER_PIXEL))
    except Exception:
        um = float("nan")

    if not np.isfinite(um) or um <= 0 or um > 100:
        return DEFAULT_UM_PER_PIXEL, False
    return um, True


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        x = float(x)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


# =============================================================================
# SQLite experiment history
# =============================================================================

def get_db_path() -> str:
    candidates = [
        os.path.abspath(DB_FILENAME),
        os.path.join(tempfile.gettempdir(), DB_FILENAME),
    ]
    for path in candidates:
        try:
            conn = sqlite3.connect(path)
            conn.execute("SELECT 1")
            conn.close()
            return path
        except Exception:
            pass
    return ":memory:"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            experiment_name TEXT,
            source TEXT,
            n_frames INTEGER,
            fps REAL,
            um_per_pixel REAL,
            calibration_source TEXT,
            n_tracks INTEGER,
            n_events INTEGER,
            mota REAL,
            id_switches INTEGER,
            rmse_px REAL,
            tracking_precision REAL,
            tracking_recall REAL,
            notes TEXT
        )
        """
    )
    return conn


def _sql_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    return v


def save_experiment(record: Dict[str, Any]) -> Tuple[bool, Any]:
    try:
        fields = [
            "created_at",
            "experiment_name",
            "source",
            "n_frames",
            "fps",
            "um_per_pixel",
            "calibration_source",
            "n_tracks",
            "n_events",
            "mota",
            "id_switches",
            "rmse_px",
            "tracking_precision",
            "tracking_recall",
            "notes",
        ]
        values = [_sql_value(record.get(f)) for f in fields]
        sql = f"INSERT INTO experiments ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})"
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        last_id = cur.lastrowid
        conn.close()
        return True, last_id
    except Exception as e:
        return False, str(e)


def load_history() -> pd.DataFrame:
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM experiments ORDER BY id DESC",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# Geometry / visualization helpers
# =============================================================================

def color_for_track(track_id: int) -> Tuple[int, int, int]:
    hue = int((int(track_id) * 47) % 180)
    hsv = np.uint8([[[hue, 220, 235]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def ellipse_perimeter(major: float, minor: float) -> float:
    a = max(float(major) / 2.0, 1e-3)
    b = max(float(minor) / 2.0, 1e-3)
    return float(math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def eccentricity(major: float, minor: float) -> float:
    major = float(major)
    minor = float(minor)
    a = max(major, minor) / 2.0
    b = min(major, minor) / 2.0
    if a <= 1e-6:
        return 0.0
    b = max(b, 1e-6)
    val = 1.0 - (b * b) / (a * a)
    return float(math.sqrt(max(0.0, val)))


# =============================================================================
# Detector / segmentation
# =============================================================================

def detect_frame(gray: np.ndarray, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    stats = {
        "mean_intensity": 0.0,
        "blur_score": 0.0,
        "n_detections": 0,
    }

    if gray is None or gray.size == 0:
        return [], stats

    if cfg.get("invert", False):
        gray = cv2.bitwise_not(gray)

    h, w = gray.shape[:2]
    stats["mean_intensity"] = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    stats["blur_score"] = blur_score

    med = cv2.medianBlur(gray, 3)

    bg_k = max(15, int(cfg.get("bg_kernel", 51)) | 1)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_k, bg_k))
    background = cv2.morphologyEx(med, cv2.MORPH_OPEN, bg_kernel)

    fg = cv2.subtract(med, background)
    fg = cv2.normalize(fg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(cfg.get("clahe_clip", 2.5)), tileGridSize=(8, 8))
    fg = clahe.apply(fg)

    _, thresh = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(thresh > 0)) > 0.60:
        thresh = cv2.bitwise_not(thresh)

    open_k = max(1, int(cfg.get("open_kernel", 3)) | 1)
    close_k = max(1, int(cfg.get("close_kernel", 7)) | 1)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = float(cfg.get("min_area", 35))
    max_area = float(cfg.get("max_area_fraction", 0.25)) * h * w

    detections = []

    blur_norm = float(np.clip(blur_score / 150.0, 0.0, 1.0))

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        if len(cnt) >= 5:
            (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
        else:
            (x_circle, y_circle), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = float(x_circle), float(y_circle)
            ma = MA = float(2.0 * radius)
            angle = 0.0

        if MA <= 2.0 or ma <= 0.0:
            continue

        if MA < ma:
            MA, ma = ma, MA

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

        roi_raw = gray[y1:y2, x1:x2]
        roi_fg = fg[y1:y2, x1:x2]

        mean_intensity = float(roi_raw.mean()) if roi_raw.size else 0.0
        mean_response = float(roi_fg.mean()) if roi_fg.size else 0.0
        response_score = float(np.clip(mean_response / 255.0, 0.0, 1.0))
        area_score = float(np.clip(area / 500.0, 0.0, 1.0))

        confidence = float(
            0.35 * solidity
            + 0.25 * area_score
            + 0.25 * blur_norm
            + 0.15 * response_score
        )
        confidence = float(np.clip(confidence, 0.05, 0.99))

        pos_uncertainty_px = float(0.6 + 2.5 * (1.0 - confidence) + 35.0 / (blur_score + 15.0))
        area_uncertainty_px = float(area * (0.07 + 0.30 * (1.0 - confidence)))

        detections.append(
            {
                "centroid": (float(cx), float(cy)),
                "area": area,
                "major_axis": float(MA),
                "minor_axis": float(ma),
                "angle": float(angle),
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "confidence": confidence,
                "pos_uncertainty_px": pos_uncertainty_px,
                "area_uncertainty_px": area_uncertainty_px,
                "solidity": float(solidity),
                "mean_intensity": mean_intensity,
            }
        )

    stats["n_detections"] = len(detections)
    return detections, stats


# =============================================================================
# Tracker
# =============================================================================

class Track:
    def __init__(self, track_id: int, detection: Dict[str, Any], frame_idx: int, min_hits: int):
        self.id = int(track_id)
        self.centroid = np.array(detection["centroid"], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.last_frame = int(frame_idx)
        self.hits = 1
        self.miss = 0
        self.min_hits = int(min_hits)
        self.state = "tentative" if min_hits > 1 else "active"

        self.area = float(detection.get("area", 0.0))
        self.major = float(detection.get("major_axis", 0.0))
        self.minor = float(detection.get("minor_axis", 0.0))
        self.angle = float(detection.get("angle", 0.0))
        self.conf = float(detection.get("confidence", 0.5))
        self.pos_unc = float(detection.get("pos_uncertainty_px", 1.0))
        self.area_unc = float(detection.get("area_uncertainty_px", 0.0))
        self.mean_intensity = float(detection.get("mean_intensity", 0.0))
        self.speed = 0.0
        self.trail = [tuple(self.centroid.astype(int))]

    def predict(self) -> np.ndarray:
        return self.centroid + self.velocity

    def update(self, detection: Dict[str, Any], frame_idx: int, velocity_alpha: float) -> None:
        new_centroid = np.array(detection["centroid"], dtype=float)
        dt = max(1, int(frame_idx) - int(self.last_frame))

        inst_velocity = (new_centroid - self.centroid) / float(dt)
        self.velocity = velocity_alpha * inst_velocity + (1.0 - velocity_alpha) * self.velocity
        self.speed = float(np.linalg.norm(self.velocity))

        self.centroid = new_centroid
        self.last_frame = int(frame_idx)
        self.hits += 1
        self.miss = 0

        self.area = float(detection.get("area", self.area))
        self.major = float(detection.get("major_axis", self.major))
        self.minor = float(detection.get("minor_axis", self.minor))
        self.angle = float(detection.get("angle", self.angle))
        self.conf = float(detection.get("confidence", self.conf))
        self.pos_unc = float(detection.get("pos_uncertainty_px", self.pos_unc))
        self.area_unc = float(detection.get("area_uncertainty_px", self.area_unc))
        self.mean_intensity = float(detection.get("mean_intensity", self.mean_intensity))

        if self.hits >= self.min_hits:
            self.state = "active"

        self.trail.append(tuple(self.centroid.astype(int)))
        if len(self.trail) > 80:
            self.trail.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "centroid": tuple(float(x) for x in self.centroid),
            "x": float(self.centroid[0]),
            "y": float(self.centroid[1]),
            "area": self.area,
            "major_axis": self.major,
            "minor_axis": self.minor,
            "angle": self.angle,
            "confidence": self.conf,
            "pos_uncertainty_px": self.pos_unc,
            "area_uncertainty_px": self.area_unc,
            "mean_intensity": self.mean_intensity,
            "speed_px_per_frame": self.speed,
            "state": self.state,
            "hits": self.hits,
            "miss": self.miss,
            "last_frame": self.last_frame,
            "trail": list(self.trail),
        }


class CellTracker:
    def __init__(
        self,
        max_distance: float = 45.0,
        max_miss: int = 12,
        min_hits: int = 2,
        velocity_alpha: float = 0.65,
        area_weight: float = 12.0,
    ):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_distance = float(max_distance)
        self.max_miss = int(max_miss)
        self.min_hits = int(min_hits)
        self.velocity_alpha = float(velocity_alpha)
        self.area_weight = float(area_weight)

    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        frame_idx = int(frame_idx)

        active_tracks = [tr for tr in self.tracks if tr.state != "deleted"]

        pairs = []
        for ti, track in enumerate(active_tracks):
            predicted = track.predict()
            for di, det in enumerate(detections):
                d = float(np.linalg.norm(predicted - np.array(det["centroid"], dtype=float)))
                if d <= self.max_distance:
                    area_diff = abs(
                        math.log1p(float(det.get("area", 1.0))) - math.log1p(float(track.area))
                    )
                    cost = d + self.area_weight * area_diff
                    pairs.append((cost, d, ti, di))

        pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_detections = set()

        for cost, d, ti, di in pairs:
            if ti in matched_tracks or di in matched_detections:
                continue
            active_tracks[ti].update(detections[di], frame_idx, self.velocity_alpha)
            matched_tracks.add(ti)
            matched_detections.add(di)

        for ti, track in enumerate(active_tracks):
            if ti not in matched_tracks:
                track.miss += 1
                if track.miss > self.max_miss:
                    track.state = "deleted"

        for di, det in enumerate(detections):
            if di not in matched_detections:
                new_track = Track(self.next_id, det, frame_idx, self.min_hits)
                self.next_id += 1
                self.tracks.append(new_track)

        output = []
        for track in self.tracks:
            if track.state != "deleted" and track.last_frame == frame_idx:
                output.append(track.to_dict())

        self.tracks = [
            tr
            for tr in self.tracks
            if tr.state != "deleted" or (frame_idx - tr.last_frame) < 30
        ]

        return output


# =============================================================================
# Synthetic deterministic demo generator
# =============================================================================

def generate_demo(
    n_frames: int = 180,
    h: int = 320,
    w: int = 480,
    seed: int = 2026,
) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    xg = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yg = np.linspace(0.0, 1.0, h, dtype=np.float32)
    base_bg = 18.0 + 22.0 * xg[None, :] + 14.0 * yg[:, None]

    cells: List[Dict[str, Any]] = []
    for i in range(1, 8):
        vx = float(rng.uniform(-1.8, 1.8))
        vy = float(rng.uniform(-1.8, 1.8))
        if abs(vx) < 0.5:
            vx = 0.9 if vx >= 0 else -0.9
        if abs(vy) < 0.5:
            vy = 0.7 if vy >= 0 else -0.7

        cells.append(
            {
                "id": i,
                "x": float(rng.uniform(70, w - 70)),
                "y": float(rng.uniform(70, h - 70)),
                "vx": vx,
                "vy": vy,
                "rx": float(rng.uniform(9.0, 16.0)),
                "ry": float(rng.uniform(7.0, 13.0)),
                "angle": float(rng.uniform(0.0, 180.0)),
                "spin": float(rng.uniform(-2.0, 2.0)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
                "alive": True,
                "intensity": float(rng.uniform(150.0, 220.0)),
            }
        )

    frames: List[np.ndarray] = []
    gt_frames: List[List[Dict[str, Any]]] = []
    division_done = False

    events = [
        {
            "frame": 57,
            "track_id": None,
            "type": "missing_frame",
            "description": "Simulated corrupted / missing frame interval",
            "confidence": 0.99,
            "source": "simulation",
        },
        {
            "frame": 70,
            "track_id": None,
            "type": "occlusion",
            "description": "Simulated occluding debris crosses field of view",
            "confidence": 0.98,
            "source": "simulation",
        },
        {
            "frame": 80,
            "track_id": 3,
            "type": "division",
            "description": "Simulated division of track 3 into daughter tracks 101 and 102",
            "confidence": 0.99,
            "source": "simulation",
        },
        {
            "frame": 120,
            "track_id": None,
            "type": "experimental_change",
            "description": "Simulated perturbation: illumination reduction and motility increase",
            "confidence": 0.99,
            "source": "simulation",
        },
        {
            "frame": 150,
            "track_id": 5,
            "type": "track_lost",
            "description": "Simulated cell death / loss of track 5",
            "confidence": 0.97,
            "source": "simulation",
        },
    ]

    for t in range(n_frames):
        speed_factor = 1.0 if t < 120 else 1.9
        illum = 1.0 + 0.08 * math.sin(2.0 * math.pi * t / 70.0)
        if t >= 120:
            illum *= 0.68

        img = base_bg.copy() * illum
        img += 3.0 * np.sin(2.0 * math.pi * (yg[:, None] * 3.0 + t * 0.010))
        img += 2.0 * np.cos(2.0 * math.pi * (xg[None, :] * 2.0 - t * 0.008))

        if t == 80 and not division_done:
            parent = None
            for c in cells:
                if c["id"] == 3 and c["alive"]:
                    parent = c
                    break
            if parent is not None:
                parent["alive"] = False
                cells.append(
                    {
                        "id": 101,
                        "x": parent["x"] - 8.0,
                        "y": parent["y"],
                        "vx": -1.5,
                        "vy": -0.8,
                        "rx": parent["rx"] * 0.72,
                        "ry": parent["ry"] * 0.72,
                        "angle": parent["angle"],
                        "spin": 2.0,
                        "phase": 0.0,
                        "alive": True,
                        "intensity": parent["intensity"],
                    }
                )
                cells.append(
                    {
                        "id": 102,
                        "x": parent["x"] + 8.0,
                        "y": parent["y"],
                        "vx": 1.6,
                        "vy": 0.9,
                        "rx": parent["rx"] * 0.72,
                        "ry": parent["ry"] * 0.72,
                        "angle": parent["angle"] + 45.0,
                        "spin": -2.0,
                        "phase": 1.0,
                        "alive": True,
                        "intensity": parent["intensity"],
                    }
                )
                division_done = True

        gt: List[Dict[str, Any]] = []

        for c in cells:
            if not c["alive"]:
                continue

            if c["id"] == 5 and t > 135:
                c["rx"] = max(2.0, c["rx"] * 0.985)
                c["ry"] = max(2.0, c["ry"] * 0.985)
                c["intensity"] = max(60.0, c["intensity"] * 0.99)
                if t >= 165:
                    c["alive"] = False
                    continue

            c["x"] += c["vx"] * speed_factor + 0.25 * math.sin(c["phase"] + t * 0.12)
            c["y"] += c["vy"] * speed_factor + 0.25 * math.cos(c["phase"] + t * 0.10)
            c["angle"] = (c["angle"] + c["spin"]) % 180.0

            margin = float(max(c["rx"], c["ry"]) + 6.0)
            if c["x"] < margin or c["x"] > w - margin:
                c["vx"] *= -1.0
                c["x"] = float(np.clip(c["x"], margin, w - margin))
            if c["y"] < margin or c["y"] > h - margin:
                c["vy"] *= -1.0
                c["y"] = float(np.clip(c["y"], margin, h - margin))

            axes = (int(max(2.0, c["rx"])), int(max(2.0, c["ry"])))
            color = float(np.clip(c["intensity"] * illum, 35.0, 245.0))

            cv2.ellipse(
                img,
                (int(c["x"]), int(c["y"])),
                axes,
                float(c["angle"]),
                0,
                360,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.ellipse(
                img,
                (int(c["x"]), int(c["y"])),
                (max(1, int(axes[0] * 0.65)), max(1, int(axes[1] * 0.65))),
                float(c["angle"]),
                0,
                360,
                color * 1.12,
                -1,
                lineType=cv2.LINE_AA,
            )

            gt.append(
                {
                    "id": int(c["id"]),
                    "centroid": (float(c["x"]), float(c["y"])),
                    "major_axis": float(2.0 * c["rx"]),
                    "minor_axis": float(2.0 * c["ry"]),
                    "angle": float(c["angle"]),
                    "area": float(math.pi * c["rx"] * c["ry"]),
                }
            )

        if 70 <= t < 85:
            x_bar = int((t - 70) / 14.0 * w)
            cv2.rectangle(img, (x_bar - 18, 0), (x_bar + 18, h), 12, -1)

        if t in (57, 58):
            img = rng.normal(28.0, 22.0, (h, w)).astype(np.float32)

        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        img = np.clip(img + rng.normal(0.0, 3.2, (h, w)), 0, 255).astype(np.uint8)

        frames.append(img)
        gt_frames.append(gt)

    meta = {
        "fps": 10.0,
        "width": w,
        "height": h,
        "true_um_per_pixel": DEFAULT_UM_PER_PIXEL,
        "seed": seed,
        "source": "Deterministic synthetic microscopy simulation",
        "anomaly_frame": 120,
        "division_frame": 80,
        "events": events,
    }

    return frames, gt_frames, meta


# =============================================================================
# Measurements, events, evaluation
# =============================================================================

def build_measurements(tracks_by_frame: List[List[Dict[str, Any]]], meta: Dict[str, Any]) -> pd.DataFrame:
    fps = float(meta.get("fps", 10.0))
    rows = []

    for frame_idx, track_list in enumerate(tracks_by_frame):
        for tr in track_list:
            if tr.get("state", "active") == "deleted":
                continue

            major = float(tr.get("major_axis", 0.0))
            minor = float(tr.get("minor_axis", 0.0))
            centroid = tr.get("centroid", (0.0, 0.0))

            rows.append(
                {
                    "frame": int(frame_idx),
                    "time_s": float(frame_idx) / fps,
                    "track_id": int(tr.get("id", -1)),
                    "x_px": float(centroid[0]),
                    "y_px": float(centroid[1]),
                    "area_px": float(tr.get("area", 0.0)),
                    "major_px": major,
                    "minor_px": minor,
                    "angle_deg": float(tr.get("angle", 0.0)) % 180.0,
                    "eccentricity": eccentricity(major, minor),
                    "perimeter_px": ellipse_perimeter(major, minor),
                    "mean_intensity": float(tr.get("mean_intensity", 0.0)),
                    "speed_px_per_frame": float(tr.get("speed_px_per_frame", 0.0)),
                    "confidence": float(tr.get("confidence", 0.0)),
                    "pos_uncertainty_px": float(tr.get("pos_uncertainty_px", 1.0)),
                    "area_uncertainty_px": float(tr.get("area_uncertainty_px", 0.0)),
                    "track_state": tr.get("state", "active"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "frame", "time_s", "track_id", "x_px", "y_px", "area_px",
                "major_px", "minor_px", "angle_deg", "eccentricity",
                "perimeter_px", "mean_intensity", "speed_px_per_frame",
                "confidence", "pos_uncertainty_px", "area_uncertainty_px",
                "track_state", "step_px", "cumulative_distance_px",
            ]
        )

    df = pd.DataFrame(rows)
    df.sort_values(["track_id", "frame"], inplace=True)

    dx = df.groupby("track_id")["x_px"].diff().fillna(0.0)
    dy = df.groupby("track_id")["y_px"].diff().fillna(0.0)
    df["step_px"] = np.sqrt(dx * dx + dy * dy)
    df["cumulative_distance_px"] = df.groupby("track_id")["step_px"].cumsum()

    df.sort_values(["frame", "track_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calibrate_measurements(df: pd.DataFrame, um_per_pixel: float, fps: float) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["track_label"] = []
        return out

    out = df.copy()
    um = float(um_per_pixel)
    fps = float(fps)

    out["um_per_pixel"] = um
    out["track_label"] = "ID " + out["track_id"].astype(str)

    out["x_um"] = out["x_px"] * um
    out["y_um"] = out["y_px"] * um
    out["area_um2"] = out["area_px"] * um * um
    out["area_uncertainty_um2"] = out["area_uncertainty_px"] * um * um
    out["major_um"] = out["major_px"] * um
    out["minor_um"] = out["minor_px"] * um
    out["perimeter_um"] = out["perimeter_px"] * um

    out["speed_um_per_s"] = out["speed_px_per_frame"] * fps * um
    out["speed_um_per_min"] = out["speed_um_per_s"] * 60.0
    out["speed_uncertainty_um_per_s"] = (
        out["speed_um_per_s"] * (0.10 + 0.50 * (1.0 - out["confidence"]))
        + out["pos_uncertainty_px"] * fps * um
    )

    out["step_um"] = out["step_px"] * um
    out["cumulative_distance_um"] = out["cumulative_distance_px"] * um
    out["pos_uncertainty_um"] = out["pos_uncertainty_px"] * um

    return out


def detect_events(
    measurements_df: pd.DataFrame,
    frame_stats_df: pd.DataFrame,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    fps = float(meta.get("fps", 10.0))
    events: List[Dict[str, Any]] = []

    for ev in meta.get("events", []):
        e = ev.copy()
        e.setdefault("source", "simulation")
        e.setdefault("confidence", 0.95)
        events.append(e)

    stats_df = frame_stats_df.copy() if frame_stats_df is not None else pd.DataFrame()

    if not stats_df.empty and "mean_intensity" in stats_df:
        roll_mean = stats_df["mean_intensity"].rolling(15, min_periods=5).mean()
        roll_std = stats_df["mean_intensity"].rolling(15, min_periods=5).std().fillna(1.0)
        z = (stats_df["mean_intensity"] - roll_mean) / (roll_std + 1e-6)

        anomaly_frames = stats_df.loc[z.abs() > 3.5, "frame"].tolist()
        for fr in anomaly_frames:
            fr = int(fr)
            already = any(
                abs(fr - int(e.get("frame", -999) or -999)) < 5
                and e.get("type") in ("illumination_change", "experimental_change")
                for e in events
            )
            if not already:
                events.append(
                    {
                        "frame": fr,
                        "track_id": None,
                        "type": "illumination_change",
                        "description": "Detected abrupt global illumination / intensity change",
                        "confidence": 0.70,
                        "source": "algorithm",
                    }
                )

    if not measurements_df.empty and "speed_px_per_frame" in measurements_df:
        for tid, grp in measurements_df.groupby("track_id"):
            grp = grp.sort_values("frame")
            speed = grp["speed_px_per_frame"]
            roll = speed.rolling(12, min_periods=5).mean()
            std = speed.rolling(12, min_periods=5).std().fillna(0.5)
            z = (speed - roll) / (std + 1e-6)

            candidates = grp[(z > 2.8) & (speed > 1.2)]
            if len(candidates) >= 3:
                first = candidates.iloc[0]
                events.append(
                    {
                        "frame": int(first["frame"]),
                        "track_id": int(tid),
                        "type": "motion_anomaly",
                        "description": f"Sustained speed increase for track {int(tid)}",
                        "confidence": 0.65,
                        "source": "algorithm",
                    }
                )

    if not measurements_df.empty:
        if not stats_df.empty and "frame" in stats_df:
            n_frames = int(stats_df["frame"].max()) + 1
        else:
            n_frames = int(measurements_df["frame"].max()) + 1

        last_frames = measurements_df.groupby("track_id")["frame"].max()
        for tid, lf in last_frames.items():
            if int(lf) < n_frames - 3:
                events.append(
                    {
                        "frame": int(lf),
                        "track_id": int(tid),
                        "type": "track_lost",
                        "description": f"Track {int(tid)} lost before end of experiment",
                        "confidence": 0.60,
                        "source": "algorithm",
                    }
                )

    df = pd.DataFrame(events)

    if df.empty:
        df = pd.DataFrame(
            columns=["frame", "time_s", "track_id", "type", "description", "confidence", "source"]
        )
        return df

    if "frame" not in df:
        df["frame"] = 0

    df["time_s"] = df["frame"].astype(float) / fps

    for col in ["track_id", "type", "description", "confidence", "source"]:
        if col not in df:
            df[col] = None

    df = df.drop_duplicates(subset=["frame", "type", "track_id"], keep="first")
    df = df.sort_values(["frame", "type"]).reset_index(drop=True)
    return df


def evaluate_tracking(
    gt_frames: List[List[Dict[str, Any]]],
    tracks_by_frame: List[List[Dict[str, Any]]],
    max_dist: float = 22.0,
) -> Dict[str, float]:
    total_gt = 0
    tp = 0
    fn = 0
    fp = 0
    id_switches = 0
    errors: List[float] = []
    gt_to_track: Dict[Any, int] = {}

    n = min(len(gt_frames), len(tracks_by_frame))

    for t in range(n):
        gt_list = gt_frames[t] or []
        track_list = tracks_by_frame[t] or []

        total_gt += len(gt_list)

        pairs = []
        for gi, gt in enumerate(gt_list):
            gt_c = np.array(gt["centroid"], dtype=float)
            for ti, tr in enumerate(track_list):
                tr_c = np.array(tr["centroid"], dtype=float)
                d = float(np.linalg.norm(gt_c - tr_c))
                if d <= max_dist:
                    pairs.append((d, gi, ti))

        pairs.sort(key=lambda x: x[0])

        matched_gt = set()
        matched_tr = set()

        for d, gi, ti in pairs:
            if gi in matched_gt or ti in matched_tr:
                continue

            matched_gt.add(gi)
            matched_tr.add(ti)
            tp += 1
            errors.append(d)

            gt_id = gt_list[gi]["id"]
            tr_id = track_list[ti]["id"]

            if gt_id in gt_to_track and gt_to_track[gt_id] != tr_id:
                id_switches += 1
            gt_to_track[gt_id] = tr_id

        fn += len(gt_list) - len(matched_gt)
        fp += len(track_list) - len(matched_tr)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    mota = 1.0 - (fn + fp + id_switches) / max(1, total_gt)
    rmse_px = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")

    return {
        "total_gt": int(total_gt),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "id_switches": int(id_switches),
        "precision": float(precision),
        "recall": float(recall),
        "MOTA": float(mota),
        "RMSE_px": float(rmse_px),
    }


def run_pipeline(
    frames: List[np.ndarray],
    gt_frames: List[List[Dict[str, Any]]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    tracker = CellTracker(**TRACK_DEFAULTS)

    tracks_by_frame: List[List[Dict[str, Any]]] = []
    frame_stats: List[Dict[str, float]] = []

    for idx, frame in enumerate(frames):
        detections, stats = detect_frame(frame, DETECTOR_DEFAULTS)
        active_tracks = tracker.update(idx, detections)

        tracks_by_frame.append(active_tracks)

        stats = dict(stats)
        stats["frame"] = idx
        stats["n_detections"] = len(detections)
        frame_stats.append(stats)

    frame_stats_df = pd.DataFrame(frame_stats)
    measurements_df = build_measurements(tracks_by_frame, meta)
    events_df = detect_events(measurements_df, frame_stats_df, meta)
    evaluation = evaluate_tracking(gt_frames, tracks_by_frame, max_dist=22.0)

    evaluation["n_tracks"] = int(measurements_df["track_id"].nunique()) if not measurements_df.empty else 0

    return {
        "tracks_by_frame": tracks_by_frame,
        "frame_stats": frame_stats_df,
        "measurements": measurements_df,
        "events": events_df,
        "evaluation": evaluation,
    }


# =============================================================================
# Overlay rendering
# =============================================================================

def render_overlay(frame_idx: int) -> np.ndarray:
    demo = st.session_state.get("demo")
    result = st.session_state.get("result")

    if demo is None or result is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    frames = demo["frames"]
    frame_idx = int(frame_idx) % len(frames)

    gray = frames[frame_idx]
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    tracks = result["tracks_by_frame"][frame_idx]
    events_df = result.get("events")

    show_trails = bool(st.session_state.get("show_trails", True))
    show_labels = bool(st.session_state.get("show_labels", True))
    show_uncertainty = bool(st.session_state.get("show_uncertainty", True))

    um_per_pixel, calibration_valid = get_calibration()
    fps = float(demo["meta"].get("fps", 10.0))

    for tr in tracks:
        track_id = int(tr.get("id", 0))
        color = color_for_track(track_id)

        centroid = tr.get("centroid", (0.0, 0.0))
        center = (int(float(centroid[0])), int(float(centroid[1])))

        major = max(1, int(float(tr.get("major_axis", 6.0)) / 2.0))
        minor = max(1, int(float(tr.get("minor_axis", 6.0)) / 2.0))
        angle = float(tr.get("angle", 0.0))

        if show_trails:
            trail = tr.get("trail", [])
            for i in range(1, len(trail)):
                cv2.line(rgb, trail[i - 1], trail[i], color, 1, cv2.LINE_AA)

        if show_uncertainty:
            radius = int(5 + 3.0 * float(tr.get("pos_uncertainty_px", 1.0)))
            cv2.circle(rgb, center, radius, color, 1, cv2.LINE_AA)

        cv2.ellipse(rgb, center, (major, minor), angle, 0, 360, color, 2, cv2.LINE_AA)
        cv2.circle(rgb, center, 2, color, -1, cv2.LINE_AA)

        if show_labels:
            speed_um = float(tr.get("speed_px_per_frame", 0.0)) * fps * um_per_pixel
            label = f"ID {track_id}"
            sublabel = f"{speed_um:.2f} um/s"

            cv2.putText(
                rgb,
                label,
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                label,
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                sublabel,
                (center[0] + 6, center[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                sublabel,
                (center[0] + 6, center[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    if events_df is not None and not events_df.empty:
        event_rows = events_df[events_df["frame"].astype(float) == float(frame_idx)]
        if not event_rows.empty:
            txt = "; ".join([str(x) for x in event_rows["type"].unique()])
            cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 24), (70, 0, 0), -1)
            cv2.putText(
                rgb,
                f"EVENT: {txt}",
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Scale bar
    if calibration_valid and um_per_pixel > 0:
        candidates = [1, 2, 5, 10, 20, 50, 100, 200]
        bar_um = candidates[0]
        for candidate in candidates:
            px = candidate / um_per_pixel
            if 40 <= px <= 180:
                bar_um = candidate
                break

        px_len = int(bar_um / um_per_pixel)
        px_len = min(px_len, rgb.shape[1] - 80)

        x0 = 10
        y0 = rgb.shape[0] - 14
        cv2.line(rgb, (x0, y0), (x0 + px_len, y0), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(
            rgb,
            f"{bar_um} um",
            (x0 + px_len + 6, y0 + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        rgb,
        "SYNTHETIC DEMO - NOT EVIDENCE",
        (6, rgb.shape[0] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )

    return rgb


# =============================================================================
# Video export
# =============================================================================

def generate_video(duration_s: float, output_fps: int, progress_bar=None) -> Optional[bytes]:
    demo = st.session_state.get("demo")
    if demo is None:
        return None

    n_frames = len(demo["frames"])
    n_out = max(8, int(duration_s * output_fps))

    try:
        import imageio.v2 as imageio
    except Exception:
        st.error("imageio is required for video export. Install imageio and imageio-ffmpeg.")
        return None

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"microscope_celltrack_{int(time.time())}.mp4",
    )

    try:
        writer = imageio.get_writer(
            tmp_path,
            fps=int(output_fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )
    except Exception as e:
        st.error(f"Could not initialize MP4 writer: {e}")
        return None

    try:
        for i in range(n_out):
            src = int(i * n_frames / max(1, n_out)) % n_frames
            frame = render_overlay(src)
            writer.append_data(frame)

            if progress_bar is not None and (i % 4 == 0 or i == n_out - 1):
                progress_bar.progress(float((i + 1) / n_out))
    finally:
        writer.close()

    with open(tmp_path, "rb") as f:
        data = f.read()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return data


# =============================================================================
# State initialization
# =============================================================================

def init_state() -> None:
    if st.session_state.get("demo_ready", False):
        return

    with st.spinner("Loading deterministic synthetic microscope demo..."):
        try:
            frames, gt_frames, meta = generate_demo()
            result = run_pipeline(frames, gt_frames, meta)
        except Exception as e:
            st.error(f"Failed to initialize demo: {e}")
            st.stop()

        st.session_state.demo = {
            "frames": frames,
            "gt_frames": gt_frames,
            "meta": meta,
        }
        st.session_state.result = result
        st.session_state.measurements_df = result["measurements"]
        st.session_state.events_df = result["events"]
        st.session_state.frame_stats = result["frame_stats"]
        st.session_state.eval_metrics = result["evaluation"]

        st.session_state.demo_ready = True
        st.session_state.playing = True
        st.session_state.frame_idx = 0
        st.session_state.demo_start = time.time()


# =============================================================================
# Summary / export helpers
# =============================================================================

def build_summary() -> Dict[str, Any]:
    demo = st.session_state.get("demo", {})
    meta = demo.get("meta", {})
    eval_metrics = st.session_state.get("eval_metrics", {})
    events_df = st.session_state.get("events_df", pd.DataFrame())

    um, _ = get_calibration()

    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "experiment_name": "MicroscopeCellTrack synthetic demo",
        "source": meta.get("source", "simulation"),
        "n_frames": len(demo.get("frames", [])),
        "fps": float(meta.get("fps", 10.0)),
        "um_per_pixel": float(um),
        "calibration_source": st.session_state.get("calibration_source", ""),
        "n_tracks": int(eval_metrics.get("n_tracks", 0)),
        "n_events": int(len(events_df)),
        "mota": safe_float(eval_metrics.get("MOTA"), float("nan")),
        "id_switches": int(eval_metrics.get("id_switches", 0)),
        "rmse_px": safe_float(eval_metrics.get("RMSE_px"), float("nan")),
        "tracking_precision": safe_float(eval_metrics.get("precision"), float("nan")),
        "tracking_recall": safe_float(eval_metrics.get("recall"), float("nan")),
        "notes": "Synthetic deterministic demo; not validated scientific evidence.",
    }


# =============================================================================
# Live panel fragment
# =============================================================================

def live_panel_inner() -> None:
    demo = st.session_state.get("demo")
    result = st.session_state.get("result")

    if demo is None or result is None:
        st.warning("Demo not initialized.")
        return

    n_frames = len(demo["frames"])
    meta = demo["meta"]

    if st.session_state.get("playing", False):
        step = max(1, int(round(float(st.session_state.get("play_fps", 10)) * FRAGMENT_INTERVAL)))
        st.session_state.frame_idx = (int(st.session_state.get("frame_idx", 0)) + step) % n_frames

        demo_start = st.session_state.get("demo_start", time.time())
        if time.time() - demo_start > 60:
            st.session_state.playing = False

    idx = int(st.session_state.get("frame_idx", 0)) % n_frames

    c1, c2, c3, c4 = st.columns(4)

    if c1.button("Pause" if st.session_state.playing else "Play", key="play_toggle"):
        st.session_state.playing = not st.session_state.playing
        if st.session_state.playing:
            st.session_state.demo_start = time.time()

    if c2.button("⟨ Step", key="step_back"):
        st.session_state.playing = False
        st.session_state.frame_idx = (idx - 1) % n_frames

    if c3.button("Step ⟩", key="step_forward"):
        st.session_state.playing = False
        st.session_state.frame_idx = (idx + 1) % n_frames

    if c4.button("Reset", key="reset_demo"):
        st.session_state.playing = True
        st.session_state.frame_idx = 0
        st.session_state.demo_start = time.time()

    selected = st.slider(
        "Frame",
        min_value=0,
        max_value=n_frames - 1,
        value=idx,
        disabled=bool(st.session_state.playing),
    )

    if selected != idx:
        st.session_state.playing = False
        st.session_state.frame_idx = int(selected)
        idx = int(selected)

    img = render_overlay(idx)
    st.image(img, caption=f"Frame {idx + 1}/{n_frames} | Synthetic live demo", use_container_width=True)

    tracks = result["tracks_by_frame"][idx]
    active_tracks = [tr for tr in tracks if tr.get("state", "active") == "active"]
    mean_conf = float(np.mean([tr.get("confidence", 0.0) for tr in tracks])) if tracks else 0.0

    events_df = result.get("events", pd.DataFrame())
    events_here = 0
    if events_df is not None and not events_df.empty:
        events_here = int((events_df["frame"].astype(float) == float(idx)).sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Frame", f"{idx + 1}/{n_frames}")
    m2.metric("Active tracks", len(active_tracks))
    m3.metric("Detections", len(tracks))
    m4.metric("Mean confidence", f"{mean_conf:.2f}")
    m5.metric("Events here", events_here)

    st.progress(float((idx + 1) / n_frames))

    if st.session_state.get("highlight_anomaly", False):
        st.warning(
            "Controlled experimental change is highlighted. This is a simulated perturbation "
            "for demonstration and does not represent a real biological response."
        )


if hasattr(st, "fragment"):
    live_panel = st.fragment(run_every=timedelta(seconds=FRAGMENT_INTERVAL))(live_panel_inner)
else:
    live_panel = live_panel_inner


# =============================================================================
# Pages
# =============================================================================

def page_live() -> None:
    st.subheader("Live Experiment — Deterministic Synthetic Demo")
    st.caption(SIM_BADGE)
    st.info(
        "This dashboard automatically loads a deterministic synthetic microscopy experiment, "
        "starts live tracking, and allows interaction. It is a software demonstration, not "
        "validated scientific evidence."
    )

    live_panel()

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Controlled experimental change / anomaly")
        meta = st.session_state.demo.get("meta", {})

        if st.button("Jump to controlled anomaly frame", key="jump_anomaly"):
            st.session_state.playing = False
            st.session_state.frame_idx = int(meta.get("anomaly_frame", 120))
            st.session_state.highlight_anomaly = True

        if st.button("Jump to division event", key="jump_division"):
            st.session_state.playing = False
            st.session_state.frame_idx = int(meta.get("division_frame", 80))
            st.session_state.highlight_anomaly = False

        if st.button("Jump to track-loss event", key="jump_track_loss"):
            st.session_state.playing = False
            st.session_state.frame_idx = 150
            st.session_state.highlight_anomaly = False

        st.caption(
            "These events are simulated for demonstration. They help validate tracking, "
            "uncertainty handling, and event detection logic."
        )

    with col_right:
        st.markdown("#### Live demo video export")
        st.caption("Generates an MP4 from the live synthetic tracking overlay. Maximum 60 seconds.")

        duration = st.slider("Video duration in seconds (max 60)", 5, 60, 10, step=1)
        output_fps = st.select_slider("Output FPS", options=[4, 6, 8, 10, 12], value=8)

        if st.button("Generate live demo video", type="primary", key="generate_video_btn"):
            progress = st.progress(0.0)
            status = st.empty()
            status.info("Rendering synthetic demo video...")

            video_bytes = generate_video(duration, output_fps, progress)

            if video_bytes:
                st.session_state.video_bytes = video_bytes
                st.session_state.video_name = f"MicroscopeCellTrack_demo_{duration}s.mp4"
                status.success("Video ready.")
            else:
                status.error("Video generation failed. Try a shorter duration or install imageio-ffmpeg.")

        if st.session_state.get("video_bytes"):
            try:
                st.video(
                    st.session_state.video_bytes,
                    format="video/mp4",
                    autoplay=True,
                    muted=True,
                )
            except Exception:
                st.video(st.session_state.video_bytes)

            st.download_button(
                "Download live demo MP4",
                data=st.session_state.video_bytes,
                file_name=st.session_state.get("video_name", "MicroscopeCellTrack_demo.mp4"),
                mime="video/mp4",
            )


def page_tracking() -> None:
    st.subheader("Tracking Review")
    st.caption("Persistent IDs, detection overlays, and per-frame track states.")

    demo = st.session_state.demo
    result = st.session_state.result
    n_frames = len(demo["frames"])

    idx = st.slider("Frame", 0, n_frames - 1, 0, key="tracking_frame_slider")
    img = render_overlay(idx)
    st.image(img, caption=f"Frame {idx + 1}/{n_frames}", use_container_width=True)

    tracks = result["tracks_by_frame"][idx]
    table = []
    for tr in tracks:
        row = {k: v for k, v in tr.items() if k != "trail"}
        row["centroid"] = f"({tr['centroid'][0]:.1f}, {tr['centroid'][1]:.1f})"
        table.append(row)

    df = pd.DataFrame(table)

    if df.empty:
        st.info("No tracks on this frame.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download frame track table CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"frame_{idx + 1}_tracks.csv",
            mime="text/csv",
        )


def page_measurements() -> None:
    st.subheader("Measurements")
    st.caption("Calibrated morphology and motion measurements with uncertainty estimates.")

    df = st.session_state.measurements_df

    if df.empty:
        st.warning("No measurements available.")
        return

    um, valid = get_calibration()
    fps = float(st.session_state.demo["meta"].get("fps", 10.0))
    calibrated = calibrate_measurements(df, um, fps)

    if not valid:
        st.warning("Calibration invalid. Using fallback demo calibration.")

    track_ids = sorted(calibrated["track_id"].unique().tolist())
    selected = st.multiselect("Track IDs", track_ids, default=track_ids)

    filtered = calibrated[calibrated["track_id"].isin(selected)]

    st.metric("Measurement rows", len(filtered))

    if PLOTLY_OK and not filtered.empty:
        fig_speed = px.line(
            filtered,
            x="time_s",
            y="speed_um_per_s",
            color="track_label",
            hover_data=["confidence", "pos_uncertainty_um", "speed_uncertainty_um_per_s"],
            title="Speed over time",
        )
        st.plotly_chart(fig_speed, use_container_width=True)

        fig_area = px.line(
            filtered,
            x="time_s",
            y="area_um2",
            color="track_label",
            hover_data=["area_uncertainty_um2", "confidence"],
            title="Area over time",
        )
        st.plotly_chart(fig_area, use_container_width=True)

    st.dataframe(filtered.head(1000), use_container_width=True)

    st.download_button(
        "Download measurements CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="measurements.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download measurements JSON",
        data=filtered.to_json(orient="records", date_format="iso").encode("utf-8"),
        file_name="measurements.json",
        mime="application/json",
    )


def page_timeline() -> None:
    st.subheader("Timeline & Trajectories")
    st.caption("Calibrated trajectories, temporal measurements, and detected events.")

    df = st.session_state.measurements_df
    events_df = st.session_state.events_df

    if df.empty:
        st.warning("No measurements available.")
        return

    um, _ = get_calibration()
    fps = float(st.session_state.demo["meta"].get("fps", 10.0))
    calibrated = calibrate_measurements(df, um, fps)

    if PLOTLY_OK and not calibrated.empty:
        fig_traj = px.line(
            calibrated,
            x="x_um",
            y="y_um",
            color="track_label",
            line_group="track_label",
            hover_data=["frame", "speed_um_per_s"],
            title="Cell trajectories",
        )
        fig_traj.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_traj, use_container_width=True)

        fig_speed = px.line(
            calibrated,
            x="time_s",
            y="speed_um_per_s",
            color="track_label",
            title="Speed timeline with events",
        )

        if events_df is not None and not events_df.empty:
            unique_events = events_df.drop_duplicates(subset=["frame", "type"])
            for _, ev in unique_events.iterrows():
                fig_speed.add_vline(
                    x=float(ev["time_s"]),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=str(ev["type"]),
                    annotation_position="top right",
                )

        st.plotly_chart(fig_speed, use_container_width=True)

    st.markdown("#### Event table")
    if events_df is None or events_df.empty:
        st.info("No events detected.")
    else:
        st.dataframe(events_df, use_container_width=True)
        st.download_button(
            "Download events CSV",
            data=events_df.to_csv(index=False).encode("utf-8"),
            file_name="events.csv",
            mime="text/csv",
        )


def page_calibration() -> None:
    st.subheader("Calibration")
    st.caption("Convert pixel measurements into physical units when a scale is available.")

    um, valid = get_calibration()

    st.write(f"Current calibration: **{um:.4f} µm/pixel**")
    if not valid:
        st.warning("Current calibration is invalid. Fallback demo calibration is being used.")

    val = st.number_input(
        "Micrometres per pixel",
        min_value=0.0001,
        max_value=100.0,
        value=float(um),
        step=0.001,
        format="%.4f",
        key="calibration_number_input",
    )

    source = st.text_input(
        "Calibration source",
        value=str(st.session_state.get("calibration_source", "User input")),
        key="calibration_source_input",
    )

    c1, c2 = st.columns(2)

    if c1.button("Apply calibration"):
        if val > 0:
            st.session_state.um_per_pixel = float(val)
            st.session_state.calibration_source = source
            st.success("Calibration applied.")
        else:
            st.error("Calibration must be positive.")

    if c2.button("Use demo true calibration (0.2 µm/pixel)"):
        st.session_state.um_per_pixel = DEFAULT_UM_PER_PIXEL
        st.session_state.calibration_source = "Deterministic synthetic stage micrometer"
        st.success("Demo calibration restored.")

    st.info(
        f"Example: 100 px = {100 * um:.2f} µm. "
        f"1000 px² = {1000 * um * um:.2f} µm²."
    )

    st.warning(
        "If calibration is missing or invalid, measurements should be treated as pixel-based "
        "only. Calibration errors are surfaced and fallback values are clearly labeled."
    )


def page_history() -> None:
    st.subheader("Experiment History")
    st.caption("SQLite experiment history plus CSV/JSON export.")

    summary = build_summary()

    st.markdown("#### Current experiment summary")
    st.json(json_safe(summary))

    if st.button("Save current experiment to SQLite"):
        ok, info = save_experiment(summary)
        if ok:
            st.success(f"Saved experiment row ID: {info}")
        else:
            st.error(f"Could not save experiment: {info}")

    history = load_history()

    st.markdown("#### Stored experiments")
    if history.empty:
        st.info("No experiments saved yet.")
    else:
        st.dataframe(history, use_container_width=True)
        st.download_button(
            "Download history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="experiment_history.csv",
            mime="text/csv",
        )

    st.download_button(
        "Download current summary JSON",
        data=json.dumps(json_safe(summary), indent=2).encode("utf-8"),
        file_name="experiment_summary.json",
        mime="application/json",
    )


def page_analytics() -> None:
    st.subheader("Analytics & Ground-Truth Evaluation")
    st.caption(
        "Ground truth is synthetic and deterministic. These metrics validate software behavior, "
        "not biological truth."
    )

    eval_metrics = st.session_state.get("eval_metrics", {})
    frame_stats = st.session_state.get("frame_stats", pd.DataFrame())
    measurements = st.session_state.get("measurements_df", pd.DataFrame())

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("MOTA", f"{safe_float(eval_metrics.get('MOTA')):.3f}")
    c2.metric("Precision", f"{safe_float(eval_metrics.get('precision')):.3f}")
    c3.metric("Recall", f"{safe_float(eval_metrics.get('recall')):.3f}")
    c4.metric("ID switches", int(eval_metrics.get("id_switches", 0)))
    c5.metric("RMSE px", f"{safe_float(eval_metrics.get('RMSE_px')):.2f}")

    if PLOTLY_OK and frame_stats is not None and not frame_stats.empty:
        fig_intensity = px.line(
            frame_stats,
            x="frame",
            y="mean_intensity",
            title="Global intensity QC",
        )
        st.plotly_chart(fig_intensity, use_container_width=True)

        fig_detections = px.line(
            frame_stats,
            x="frame",
            y="n_detections",
            title="Detections per frame",
        )
        st.plotly_chart(fig_detections, use_container_width=True)

    if measurements is not None and not measurements.empty:
        um, _ = get_calibration()
        fps = float(st.session_state.demo["meta"].get("fps", 10.0))
        calibrated = calibrate_measurements(measurements, um, fps)

        summary = (
            calibrated.groupby("track_label")
            .agg(
                frames=("frame", "count"),
                mean_speed_um_per_s=("speed_um_per_s", "mean"),
                max_speed_um_per_s=("speed_um_per_s", "max"),
                mean_area_um2=("area_um2", "mean"),
                total_distance_um=("cumulative_distance_um", "max"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

        st.markdown("#### Per-track summary")
        st.dataframe(summary, use_container_width=True)

        if PLOTLY_OK:
            fig_box = px.box(
                calibrated,
                x="track_label",
                y="area_um2",
                title="Area distribution by track",
            )
            st.plotly_chart(fig_box, use_container_width=True)


# =============================================================================
# Main app
# =============================================================================

initialize_defaults()
init_state()

with st.sidebar:
    st.title(APP_TITLE)
    st.caption(SIM_BADGE)

    st.radio("Dashboard", PAGES, key="page")

    st.divider()

    st.checkbox("Trails", key="show_trails")
    st.checkbox("Labels", key="show_labels")
    st.checkbox("Uncertainty", key="show_uncertainty")

    st.slider("Playback FPS", 2, 30, key="play_fps")

    st.caption(
        "Live autoplay stops after 60 seconds. Press Play to continue. "
        "Researcher review is required before any scientific interpretation."
    )

    st.caption(
        "This is a deterministic synthetic demonstration. It does not claim scientific "
        "validation or experimental causality."
    )


page = st.session_state.get("page", PAGES[0])

if page == "Live Experiment":
    page_live()
elif page == "Tracking":
    page_tracking()
elif page == "Measurements":
    page_measurements()
elif page == "Timeline":
    page_timeline()
elif page == "Calibration":
    page_calibration()
elif page == "History":
    page_history()
elif page == "Analytics":
    page_analytics()
else:
    page_live()