"""Pinned workbook acquisition and strict, lossless research dataset adapters.

Convention: x.shape == (T,), Y.shape == (J, T). No smoothing, imputation,
standardization, removal of constant channels, or temporal subsampling.
"""
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import urllib.request
import truststore
truststore.inject_into_ssl()

import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "data/sources.json").read_text())
METADATA_COLUMNS = ["source_sync_time_s", "sensor_timestamp_ticks_text", "video_time_s",
                    "clip_time_s", "nearest_video_frame", "nearest_video_time_s",
                    "sensor_minus_video_ms", "hand_active"]


@dataclass
class Dataset:
    name: str
    x: np.ndarray
    Y: np.ndarray
    names: list
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.Y = np.ascontiguousarray(self.Y, dtype=np.float64)
        if self.x.ndim != 1 or self.Y.ndim != 2:
            raise ValueError("Expected x=(T,), Y=(J,T)")
        if self.Y.shape != (len(self.names), len(self.x)) or len(self.x) < 6:
            raise ValueError("Inconsistent dimensions or fewer than six observations")
        if not np.isfinite(self.x).all() or not np.isfinite(self.Y).all():
            raise ValueError(f"{self.name}: missing/nonfinite data; no implicit imputation")
        if not np.all(np.diff(self.x) >= 0) or self.x[1] >= self.x[-2]:
            raise ValueError("Time coordinates must be nondecreasing with a nonempty interior domain")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Dimension names must be unique")
        return self


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(name, data_dir=ROOT / "data/raw", offline=False):
    """Download only when absent; verify pinned SHA-256 even for local inputs."""
    spec = SOURCES[name]
    path = Path(data_dir) / spec["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"Missing {path}. Download {spec['url']}")
        print(f"Downloading {name}: {spec['url']}", flush=True)
        temp = path.with_suffix(".download")
        try:
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "slope-change-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=120) as src, temp.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
            if sha256(temp) != spec["sha256"]:
                raise ValueError(f"{name}: downloaded checksum mismatch")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"{path}: checksum mismatch; expected pinned workbook")
    return path


def load_era5(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb["Annual Temperature"].iter_rows(values_only=True)
        header = next(rows)
        indices = [header.index(str(y)) for y in range(2000, 2026)]
        names, values, coordinates = [], [], []
        for row in rows:
            if row[0] is None:
                continue
            coordinates.append(list(ast.literal_eval(row[0])))
            names.append(str(row[0]))
            values.append([row[i] for i in indices])
        if len(values) != 40401:
            raise ValueError("Expected all 40,401 grid cells")
        return Dataset("era5", np.arange(2000, 2026), values, names,
                       {"units": "deg C", "x_label": "Year", "coordinates": coordinates,
                        "description": "Annual equal-month means of daily 17:00 UTC t2m samples; land and ocean"}).validate()
    finally:
        wb.close()


def load_holoassist(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        matrices, names, times = [], [], []
        for hand in ("Left_hand", "Right_hand"):
            rows = wb[hand].iter_rows(values_only=True)
            header = list(next(rows))
            if header[:8] != METADATA_COLUMNS or len(header[8:]) != 546:
                raise ValueError(f"Unexpected {hand} schema")
            body = list(rows)
            times.append(np.asarray([r[3] for r in body], dtype=float))
            matrices.append(np.asarray([r[8:] for r in body], dtype=float).T)
            names.extend(f"{hand}:{h}" for h in header[8:])
        if not np.array_equal(times[0], times[1]):
            raise ValueError("Hand timestamps differ; refusing an implicit join/interpolation")
        return Dataset("holoassist", times[0], np.vstack(matrices), names,
                       {"units": "mixed (matrix entries, m, flags)", "x_label": "Clip time (s)",
                        "repeated_timestamp_count": int(len(times[0])-len(np.unique(times[0]))),
                        "description": "Both hands, all 546 exported measurement columns per hand; all 240 readings and repeated times retained"}).validate()
    finally:
        wb.close()


def load_cmapss(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = wb["Train sensors"].iter_rows(values_only=True)
        header = next(rows)
        body = []
        for row in rows:
            if row[0] == 1:
                body.append(row)
            elif body:
                break
        sensors = [r for r in body if str(r[1]).startswith("Sensor ")]
        rul_rows = [r for r in body if str(r[2]).upper() == "RUL"]
        if len(sensors) != 21 or len(rul_rows) != 1:
            raise ValueError("Expected 21 sensors and one RUL row for training engine 1")
        rul = np.asarray([v for v in rul_rows[0][5:] if v is not None], dtype=float)
        # Total RUL is the initial RUL, 191; x_i=191+1-RUL_i = 1,...,192.
        x = float(rul[0]) + 1.0 - rul
        if rul[-1] != 0 or not np.array_equal(x, np.asarray(header[5:5+len(x)], dtype=float)):
            raise ValueError("RUL-derived observation index differs from workbook cycles")
        Y = np.asarray([r[5:5+len(x)] for r in sensors], dtype=float)
        names = [f"{r[1]} ({r[2]})" for r in sensors]
        return Dataset("cmapss", x, Y, names,
                       {"units": [r[4] for r in sensors], "symbols": [r[2] for r in sensors],
                        "sensor_descriptions": [r[3] for r in sensors], "initial_RUL": float(rul[0]),
                        "x_label": "Operating cycle", "description": "FD001 training engine 1; 21 sensors, settings and RUL excluded"}).validate()
    finally:
        wb.close()


def load_dataset(name, data_dir=ROOT / "data/raw", offline=False):
    path = acquire(name, data_dir, offline)
    data = {"era5": load_era5, "holoassist": load_holoassist, "cmapss": load_cmapss}[name](path)
    data.metadata.update(source=SOURCES[name], J=data.Y.shape[0], T=len(data.x),
                         constant_dimensions=int(np.count_nonzero(np.ptp(data.Y, axis=1) == 0)))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    for name in SOURCES:
        data = load_dataset(name, offline=args.offline)
        print(name, data.Y.shape, (data.x[0], data.x[-1]), data.metadata["constant_dimensions"])
