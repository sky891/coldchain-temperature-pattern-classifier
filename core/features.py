# core/features.py
from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

DTFMT = "%Y-%m-%d %H:%M:%S"

# ----- 시간축 유틸 -----
def parse_points(items: Iterable) -> Tuple[List[datetime], List[float]]:
    """
    items: Pydantic 객체(DataPoint) 또는 dict 유사: .DATETIME / .TEMP or ["DATETIME"]/["TEMP"]
    반환: 정렬된 (timestamps, temps)
    """
    xs, ys = [], []
    for it in items:
        t = getattr(it, "DATETIME", None) or it["DATETIME"]
        v = getattr(it, "TEMP", None) or it["TEMP"]
        xs.append(datetime.strptime(str(t), DTFMT))
        ys.append(float(v))
    # 정렬
    pair = sorted(zip(xs, ys), key=lambda x: x[0])
    xs, ys = [p[0] for p in pair], [p[1] for p in pair]
    return xs, ys

def needs_resample(xs: List[datetime], target_min: float = 10.0, tol_min: float = 1.0) -> bool:
    """평균 간격이 10±tol 분인지 점검. 포인트가 너무 적으면 True(리샘플 필요)."""
    if len(xs) < 3:
        return True
    diffs = [(xs[i]-xs[i-1]).total_seconds()/60 for i in range(1, len(xs))]
    avg = sum(diffs)/len(diffs)
    return not all(abs(d - target_min) <= tol_min for d in diffs)

def resample_to_10min(xs: List[datetime], ys: List[float]) -> List[float]:
    """선형보간 기반 10분 격자 리샘플."""
    if not xs:
        return []
    cur, end = xs[0], xs[-1]
    grid = []
    while cur <= end:
        grid.append(cur)
        cur += timedelta(minutes=10)

    def interp(ts):
        if ts <= xs[0]:
            return ys[0]
        if ts >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if xs[i-1] <= ts <= xs[i]:
                t0, t1 = xs[i-1], xs[i]
                y0, y1 = ys[i-1], ys[i]
                r = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
                return y0 + r * (y1 - y0)
        return ys[-1]

    return [interp(t) for t in grid]

# ----- 길이 맞춤 & 특징 생성 -----
def _pad_or_trim_1d(arr: np.ndarray, seq_len: int, pad_val: float) -> np.ndarray:
    """오른쪽 패딩, 왼쪽 잘라서 tail 유지."""
    x = np.asarray(arr, dtype="float32")
    if len(x) >= seq_len:
        return x[-seq_len:]
    out = np.full(seq_len, pad_val, dtype="float32")
    out[: len(x)] = x
    return out

def make_features_5ch(arr_1d: np.ndarray, lo: float, hi: float,
                      seq_len: int, pad_val: float) -> np.ndarray:
    """
    (seq_len, 5): [temp, rel_pos, upper_exc, lower_exc, grad]
    - rel_pos = (x - mid) / ((hi - lo)/2)
    - upper_exc = max(0, x - hi)
    - lower_exc = max(0, lo - x)
    - grad = np.gradient(x) (패딩 구간은 pad_val)
    """
    x = _pad_or_trim_1d(np.asarray(arr_1d, dtype="float32"), seq_len, pad_val)

    m   = (x != pad_val)
    v   = x[m]
    span = max(1e-6, float(hi - lo))
    mid  = (lo + hi) * 0.5

    rel_pos   = np.full_like(x, pad_val)
    upper_exc = np.full_like(x, pad_val)
    lower_exc = np.full_like(x, pad_val)
    grad      = np.full_like(x, pad_val)

    if m.any():
        rel_pos[m]   = (v - mid) / (span * 0.5)
        upper_exc[m] = np.maximum(0.0, v - hi)
        lower_exc[m] = np.maximum(0.0, lo - v)
        if v.size >= 2:
            grad[m] = np.gradient(v).astype("float32")

    temp = x.astype("float32")
    return np.stack([temp, rel_pos, upper_exc, lower_exc, grad], axis=-1)  # (L,5)