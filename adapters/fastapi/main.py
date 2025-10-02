# adapters/fastapi/main.py

import os
import json
import math
import numpy as np
from functools import lru_cache
from typing import List, Union, Tuple, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.predictor import predict_pattern, calc_risk
from core.features import parse_points, needs_resample, resample_to_10min

# =========================
# 운영 임계값 (필요 시 조정)
# =========================
# 공통
MIN_POINTS_10MIN = 6  # 10분 간격 기준 최소 1시간 (=6포인트)

# 시작(Start) 판정(연속 점수 계산에 사용)
START_OUT_HEAD_RATIO = 0.6   # 초반 OOR 비율 ≥ 60%
START_IN_TAIL_RATIO  = 0.8   # 후반 InRange 비율 ≥ 80%
START_CENTER_GAIN    = 0.5   # 후반이 중앙(mid)으로 0.5℃ 이상 근접

# 오픈(Open) 판정(연속 점수 계산에 사용)
OPEN_SPIKE_JUMP      = 3.0   # 한 스텝 급격 점프(℃)
OPEN_SPIKE_OUTSIDE   = 0.5   # 점프 지점이 hi+0.5 또는 lo-0.5 이상/이하
OPEN_RECOVER_STEPS   = 3     # 이후 3스텝(=30분) 내 범위 복귀 시 오픈으로 간주

# 하향/상향(Trend) 판정(연속 점수 계산에 사용)
SLOPE_DOWN_TH        = -0.02 # 스텝당 ℃ (10분 기준)
SLOPE_UP_TH          =  0.02
RUN_BELOW_LO_STEPS   = 2     # 하한 미만 연속 스텝(20분)
RUN_ABOVE_HI_STEPS   = 2     # 상한 초과 연속 스텝(20분)
HEAD_TAIL_DROP_TH    = 0.8   # head_mean - tail_mean ≥ 0.8℃ → 하향
HEAD_TAIL_RISE_TH    = 0.8   # tail_mean - head_mean ≥ 0.8℃ → 상향

# 정상(Normal) 판정 보강 용 임계(현재는 참고용; 연속 점수에서 활용)
NORMAL_MAX_STEP      = 0.5   # 인접 스텝 간 최대 변화
NORMAL_ABS_SLOPE     = 0.01  # |slope|
NORMAL_STD           = 0.4   # 표준편차

PATTERNS = ["정상", "시작", "오픈", "상향", "하향"]


app = FastAPI(title="Temperature Test API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def load_temp_limits() -> Dict[str, Tuple[float, float]]:
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "temp_limits.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Tuple[float, float]] = {}
    for dt, val in data.items():
        # 키 정규화(대소문자/공백 차이 방지)
        key = str(dt).strip().upper()
        if isinstance(val, dict):
            out[key] = (float(val["lo"]), float(val["hi"]))
        else:
            lo, hi = val
            out[key] = (float(lo), float(hi))
    return out

def get_temp_limits(device_type: str) -> Tuple[float, float]:
    # 입력도 정규화
    key = (device_type or "").strip().upper()
    limits = load_temp_limits()
    return limits.get(key, (-100.0, 100.0))

# ====== 요청 스키마 ======
class SensorPayload(BaseModel):
    IMEI: str
    Temperature: str
    deviceType: str

class DataPoint(BaseModel):
    DATETIME: str
    TEMP: Union[str, float]

class PatternPayload(BaseModel):
    IMEI: str
    deviceType: str
    deviceDataList: List[DataPoint]

# ====== 도우미 ======
def compute_basic_stats(arr: np.ndarray, lo: float, hi: float) -> dict:
    """패턴 공통으로 쓰는 통계 계산"""
    t = np.arange(len(arr))
    slope = np.polyfit(t, arr, 1)[0] if len(arr) >= 2 else 0.0
    std = float(np.std(arr))

    # 앞/뒤 윈도우
    k_head = max(2, len(arr) // 3)
    k_tail = max(3, len(arr) // 2)
    head = arr[:k_head]
    tail = arr[-k_tail:]
    head_mean = float(np.mean(head)) if len(head) else float(arr[0])
    tail_mean = float(np.mean(tail)) if len(tail) else float(arr[-1])
    mid = 0.5 * (lo + hi)

    # in/out 비율
    in_head = float(((head >= lo) & (head <= hi)).mean()) if len(head) else 0.0
    in_tail = float(((tail >= lo) & (tail <= hi)).mean()) if len(tail) else 0.0

    # 하한/상한 연속 run-length
    below = arr < lo
    above = arr > hi

    def max_run(mask: np.ndarray) -> int:
        if not mask.any():
            return 0
        edges = np.where(np.concatenate(([True], mask[1:] != mask[:-1], [True])))[0]
        runs = edges[1::2] - edges[:-1:2]
        return int(runs.max())

    max_run_below_lo = max_run(below)
    max_run_above_hi = max_run(above)

    # 최대 스텝 변화
    max_step = float(np.max(np.abs(np.diff(arr)))) if len(arr) >= 2 else 0.0

    return dict(
        slope=float(slope),
        std=std,
        head_mean=head_mean,
        tail_mean=tail_mean,
        mid=mid,
        in_head=in_head,
        in_tail=in_tail,
        max_run_below_lo=max_run_below_lo,
        max_run_above_hi=max_run_above_hi,
        max_step=max_step,
    )

def detect_open(arr: np.ndarray, lo: float, hi: float) -> bool:
    """오픈: 특정 시점 급격한 범위 밖 이탈 + 단시간 복귀 (참고용: 연속 점수에서는 크기/복귀 정도만 반영)"""
    if len(arr) < 3:
        return False
    diffs = np.diff(arr)
    # 급격 점프 발생 인덱스 찾기 (i→i+1)
    cand_idx = np.where(np.abs(diffs) >= OPEN_SPIKE_JUMP)[0]
    for i in cand_idx:
        v1, v2 = arr[i], arr[i + 1]
        # 점프 후 위치가 충분히 범위 밖인지
        if (v2 > hi + OPEN_SPIKE_OUTSIDE) or (v2 < lo - OPEN_SPIKE_OUTSIDE):
            # 이후 OPEN_RECOVER_STEPS 내 복귀 확인
            end = min(len(arr), i + 1 + OPEN_RECOVER_STEPS + 1)
            if np.any((arr[i + 1 : end] >= lo) & (arr[i + 1 : end] <= hi)):
                return True
    return False

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# ----- (신규) 모델 확률 표준화: 어떤 형태로 와도 dict로 변환 -----
LABEL_MAP_EN2KO = {
    "normal": "정상",
    "start":  "시작",
    "open":   "오픈",
    "up":     "상향", "upward": "상향", "rise": "상향", "increasing": "상향",
    "down":   "하향", "downward": "하향", "fall": "하향", "decreasing": "하향",
}

def coerce_probs_to_dict(probs) -> Dict[str, float]:
    """
    다양한 형태(list/tuple/dict/리스트-페어/리스트-딕트)를
    {라벨:확률} dict로 표준화.
    - 영문 라벨은 한글 라벨로 매핑.
    - 없는 키는 0.0으로 채움.
    """
    out = {k: 0.0 for k in PATTERNS}

    # 이미 dict이면: 키 정규화 + 매핑
    if isinstance(probs, dict):
        for k, v in probs.items():
            if k is None:
                continue
            key = str(k).strip()
            ko = LABEL_MAP_EN2KO.get(key.lower(), key)
            if ko in out:
                try:
                    out[ko] = float(v)
                except Exception:
                    pass
        return out

    # list/tuple인 경우
    if isinstance(probs, (list, tuple)):
        # 1) [p1, p2, p3, p4, p5] 형태 (PATTERNS 순서 가정)
        if len(probs) == len(PATTERNS) and all(isinstance(x, (int, float)) for x in probs):
            for i, k in enumerate(PATTERNS):
                out[k] = float(probs[i])
            return out

        # 2) [("정상", 0.7), ("하향", 0.2), ...] 또는 [{"label":"정상","prob":0.7}, ...]
        for item in probs:
            lab = None; p = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lab, p = item[0], item[1]
            elif isinstance(item, dict):
                lab = item.get("label") or item.get("name") or item.get("pattern")
                p   = item.get("prob")  or item.get("p")    or item.get("score") or item.get("value")
            if lab is None or p is None:
                continue
            key = str(lab).strip()
            ko = LABEL_MAP_EN2KO.get(key.lower(), key)
            if ko in out:
                try:
                    out[ko] = float(p)
                except Exception:
                    pass
        return out

    # 기타 타입은 전부 0.0
    return out

def compute_rule_scores(arr: np.ndarray, lo: float, hi: float, stats: dict) -> dict:
    """
    하드 '게이트 통과/실패' 대신 각 패턴의 '연속 점수(0~1)'를 산출.
    - 정상: in-range 비율, |slope| 작음, std 작음
    - 시작: 초반 OOR↑, 후반 in↑, 중앙근접 이득
    - 오픈: 스파이크 크기 + 복귀 정도
    - 상향/하향: 기울기, run-length, head/tail 차이
    """
    scores = {k: 0.0 for k in PATTERNS}

    in_head = stats["in_head"]
    in_tail = stats["in_tail"]
    slope = stats["slope"]
    std = stats["std"]
    head = stats["head_mean"]
    tail = stats["tail_mean"]
    mid = stats["mid"]
    run_dn = stats["max_run_below_lo"]
    run_up = stats["max_run_above_hi"]

    in_all = 0.5 * in_head + 0.5 * in_tail
    drop = head - tail
    rise = -drop

    # --- 정상 ---
    normal_in = in_all
    normal_slope = 1.0 - clamp01(abs(slope) / 0.05)  # 0.05/step 이상이면 감점
    normal_std = 1.0 - clamp01(std / 0.8)  # std 0.8 이상이면 감점
    scores["정상"] = clamp01(0.6 * normal_in + 0.2 * normal_slope + 0.2 * normal_std)

    # --- 시작 ---
    center_gain = abs(head - mid) - abs(tail - mid)
    start_head_oor = clamp01((1.0 - in_head - 0.6) / 0.4)  # 0.6↑부터 점수
    start_tail_in = clamp01((in_tail - 0.6) / 0.4)  # 0.6↑부터 점수
    start_center = clamp01((center_gain - 0.3) / 0.7)  # 0.3~1.0 스케일
    scores["시작"] = clamp01(0.45 * start_head_oor + 0.35 * start_tail_in + 0.20 * start_center)

    # --- 오픈 ---
    diffs = np.abs(np.diff(arr)) if len(arr) >= 2 else np.array([0.0])
    big = float(np.max(diffs)) if diffs.size else 0.0
    open_scale = clamp01((big - 3.0) / 4.0)  # 3~7℃ 점프를 0~1로
    open_recover = clamp01(in_tail)  # tail 구간 복귀 정도
    scores["오픈"] = clamp01(0.75 * open_scale + 0.25 * open_recover)

    # --- 상향/하향 ---
    up_slope = clamp01((slope - 0.02) / 0.08)
    up_run = clamp01((run_up - 1) / 4)
    up_rise = clamp01((rise - 0.6) / 1.4)
    scores["상향"] = clamp01(0.5 * up_slope + 0.25 * up_run + 0.25 * up_rise)

    dn_slope = clamp01((-slope - 0.02) / 0.08)
    dn_run = clamp01((run_dn - 1) / 4)
    dn_drop = clamp01((drop - 0.6) / 1.4)
    scores["하향"] = clamp01(0.5 * dn_slope + 0.25 * dn_run + 0.25 * dn_drop)

    # 큰 스파이크/추세가 있으면 '정상' 점수 감쇠
    antagonist = max(scores["상향"], scores["하향"], scores["오픈"])
    scores["정상"] *= (1.0 - 0.6 * antagonist)

    # ε 추가: 완전 0/1 방지
    eps = 1e-6
    for k in scores:
        scores[k] = clamp01(scores[k] + eps)
    return scores

# ====== 엔드포인트 ======
@app.post("/predict_stream")
def predict_stream(req: SensorPayload):
    try:
        temp = float(req.Temperature)
    except ValueError:
        raise HTTPException(400, "Invalid Temperature")

    lo, hi = get_temp_limits(req.deviceType)
    risk = calc_risk(temp, lo, hi)
    status = "out_of_range" if risk >= 1.0 else "in_range"
    return {"status": status, "risk": round(risk, 4), "lo": lo, "hi": hi}

@app.post("/predict_pattern")
def predict_pattern_api(req: PatternPayload):
    # 1) 시간/온도 파싱
    xs, ys = parse_points(req.deviceDataList)

    # 2) 10±1분 간격 확인/리샘플
    series = resample_to_10min(xs, ys) if needs_resample(xs) else ys

    # 2.5) 최소 길이 가드 — 10분 간격 기준 1시간 미만 거절
    if len(series) < MIN_POINTS_10MIN:
        raise HTTPException(422, "데이터가 너무 짧습니다(최소 1시간 이상 권장).")

    # 3) 보관온도 한계(포함) 로드
    lo, hi = get_temp_limits(req.deviceType)

    # ========= GATE 0: '정상' 최우선 (inclusive) =========
    # 전 구간이 lo~hi(포함) 안이면 바로 정상 확정
    in_range_all = (min(series) >= lo) and (max(series) <= hi)
    if in_range_all:
        return {
            "pattern": "정상",
            "probs": {p: (1.0 if p == "정상" else 0.0) for p in PATTERNS},
            "policy_overridden": True,
            "diagnostics": {
                "gate": "storage_range_all",
                "lo": lo, "hi": hi,
                "len_series": len(series),
            },
        }

    # 4) 공통 통계 계산 (Gate 0 통과 못한 케이스만)
    arr = np.asarray(series, dtype=float)
    stats = compute_basic_stats(arr, lo, hi)

    # ====== 블렌딩 방식: 규칙 점수(연속값) × 모델 확률 ======
    # 규칙 점수 → 확률화
    rule_scores = compute_rule_scores(arr, lo, hi, stats)

    # 모델 확률 (어떤 포맷이 와도 dict로 강제 변환)
    model_label, model_probs_raw = predict_pattern(series, device_type=req.deviceType)
    model_probs = coerce_probs_to_dict(model_probs_raw)

    # 누락 키 보정 및 정규화 유틸
    def _normalize(d: dict) -> dict:
        s = sum(max(0.0, float(v)) for v in d.values()) or 1.0
        return {k: max(0.0, float(v)) / s for k, v in d.items()}

    rule_probs = _normalize(rule_scores)
    model_probs = _normalize(model_probs)

    # 가중 합성 (운영에서 W_RULE/W_MODEL 튜닝)
    W_RULE, W_MODEL = 0.6, 0.4
    mixed = {k: W_RULE * rule_probs.get(k, 0.0) + W_MODEL * model_probs.get(k, 0.0) for k in PATTERNS}

    # 신뢰도 과대 방지: 클리핑 후 재정규화
    floor, ceil = 0.02, 0.95
    for k in mixed:
        mixed[k] = max(floor, min(ceil, float(mixed[k])))
    mixed = _normalize(mixed)

    # 탑2 마진이 작으면 살짝 평탄화 (100% 과신 방지)
    top2 = sorted(mixed.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < 0.15:
        for k in mixed:
            mixed[k] = 0.5 * mixed[k] + 0.5 * (1.0 / len(PATTERNS))
        mixed = _normalize(mixed)

    final_label = max(mixed.items(), key=lambda kv: kv[1])[0]

    return {
        "pattern": final_label,
        "probs": mixed,
        "policy_overridden": False,
        "diagnostics": {
            "gate": "rule+model_blend",
            "rule_scores": rule_scores,
            "model_probs": model_probs,
            "lo": lo, "hi": hi,
            "len_series": len(series),
            "slope": round(stats["slope"], 4),
            "std": round(stats["std"], 3),
            "in_head_ratio": round(stats["in_head"], 2),
            "in_tail_ratio": round(stats["in_tail"], 2),
            "max_run_below_lo": stats["max_run_below_lo"],
            "max_run_above_hi": stats["max_run_above_hi"],
        },
    }

# ====== 한눈에 보기 좋은 확률 UI: /ui ======
@app.get("/ui", response_class=HTMLResponse)
def simple_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Temperature Pattern Viewer</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" crossorigin="anonymous"></script>
  <style>
    :root{
      --bg:#0b1020; --fg:#e8ecf3; --muted:#9aa5b1; --card:#121933;
      --accent:#5b8cff; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
      --ring: 0 10px 30px rgba(91,140,255,.18);
    }
    *{box-sizing:border-box}
    body{font-family: ui-sans-serif,system-ui,Segoe UI,Roboto,Apple Color Emoji,Noto Color Emoji;
         margin:24px;background:linear-gradient(180deg,#0b1020,#0e1430);color:var(--fg)}
    h1{font-size:20px;margin:0 0 12px}
    .grid{display:grid;gap:18px;grid-template-columns:1.1fr .9fr}
    @media (max-width: 980px){ .grid{grid-template-columns:1fr} }
    .card{background:var(--card);border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:16px;box-shadow:var(--ring)}
    textarea{width:100%;height:180px;border-radius:12px;border:1px solid rgba(255,255,255,.08);
             background:#0c132b;color:var(--fg);padding:12px;font-family:ui-monospace,Consolas,Menlo,monospace}
    button{padding:10px 16px;border-radius:10px;border:1px solid rgba(255,255,255,.12);cursor:pointer;background:#10183a;color:var(--fg)}
    .small{color:var(--muted);font-size:12px}
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .badge{display:inline-flex;align-items:center;gap:6px;border:1px solid rgba(255,255,255,.1);
           padding:4px 10px;border-radius:999px;background:rgba(255,255,255,.06);font-weight:600}
    .big{font-size:22px;font-weight:800}
    .muted{color:var(--muted)}
    .kvs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .kvs div{background:#0e1533;border:1px solid rgba(255,255,255,.06);padding:10px;border-radius:12px}
    canvas{max-height:280px}
    .bars{display:grid;gap:10px}
    .bar{background:#0d1430;border-radius:999px;border:1px solid rgba(255,255,255,.08);height:14px;overflow:hidden}
    .bar > i{display:block;height:100%;background:linear-gradient(90deg,#5b8cff,#6ca8ff)}
    .pill{display:inline-block;min-width:52px;text-align:right;font-variant-numeric:tabular-nums}
    .list{display:grid;gap:8px}
    .line{display:flex;align-items:center;gap:10px}
    pre{background:#0c132b;color:#dde6f8;border-radius:12px;padding:12px;overflow:auto}
  </style>
</head>
<body>
  <h1>Temperature Pattern Viewer</h1>

  <div class="grid">
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div>
          <b>입력 페이로드 (PatternPayload)</b>
          <div class="small">테스트 JSON 붙여넣고 “분석” 클릭</div>
        </div>
        <div class="row">
          <button id="btnNormal">정상 샘플</button>
          <button id="btnStart">시작 샘플</button>
        </div>
      </div>
      <textarea id="payload">{
  "IMEI": "demo",
  "deviceType": "L100",
  "deviceDataList": [
    {"DATETIME":"2025-01-01 01:00:00","TEMP":"4.9"},
    {"DATETIME":"2025-01-01 01:10:00","TEMP":"5.0"},
    {"DATETIME":"2025-01-01 01:20:00","TEMP":"5.1"},
    {"DATETIME":"2025-01-01 01:30:00","TEMP":"5.2"},
    {"DATETIME":"2025-01-01 01:40:00","TEMP":"5.2"},
    {"DATETIME":"2025-01-01 01:50:00","TEMP":"5.3"}
  ]
}</textarea>
      <div class="row" style="margin-top:12px">
        <button id="analyzeBtn">분석</button>
        <span id="status" class="small"></span>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div><b>예측 결과</b></div>
        <div id="topBadge" class="badge">-</div>
      </div>

      <div class="row" style="gap:18px;margin-top:12px;align-items:flex-start">
        <div style="flex:1">
          <canvas id="probDonut"></canvas>
        </div>
        <div style="flex:1">
          <canvas id="probBars"></canvas>
        </div>
      </div>

      <div class="list" style="margin-top:14px" id="probList"></div>

      <div class="kvs" style="margin-top:14px">
        <div><div class="small">게이트</div><div id="kvGate">-</div></div>
        <div><div class="small">길이(len)</div><div id="kvLen">-</div></div>
        <div><div class="small">보관온도</div><div id="kvRange">-</div></div>
        <div><div class="small">slope / std</div><div id="kvStats">-</div></div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:18px">
    <b>온도 시계열</b>
    <canvas id="tempChart" style="margin-top:8px"></canvas>
  </div>

  <div class="card" style="margin-top:18px">
    <b>로그</b>
    <pre id="log"></pre>
  </div>

  <script>
    const ORDER = ['정상','시작','오픈','상향','하향'];

    function $(id){ return document.getElementById(id); }
    function log(s){ const el=$('log'); el.textContent += s+'\\n'; el.scrollTop=el.scrollHeight; }

    let donutChart, barsChart, tempChart;

    function toPercentArray(probs){
      // probs 객체를 ORDER 순서의 % 배열로 변환 (없으면 0)
      const vals = ORDER.map(k => (probs?.[k] ?? 0));
      const sum = vals.reduce((a,b)=>a+b,0) || 1;
      return vals.map(v => Math.round(v*1000/sum)/10); // 소수1자리
    }

    function renderDonut(probs){
      const data = toPercentArray(probs);
      if(donutChart) donutChart.destroy();
      donutChart = new Chart($('probDonut'), {
        type: 'doughnut',
        data: { labels: ORDER, datasets: [{ data }] },
        options: { plugins:{ legend:{ position:'bottom' } }, cutout:'60%' }
      });
    }

    function renderBars(probs){
      const data = toPercentArray(probs);
      if(barsChart) barsChart.destroy();
      barsChart = new Chart($('probBars'), {
        type: 'bar',
        data: { labels: ORDER, datasets: [{ data, borderWidth:0 }] },
        options: {
          indexAxis:'y',
          plugins:{ legend:{display:false}, tooltip:{callbacks:{label:(ctx)=>ctx.raw+'%'}} },
          scales:{ x:{ suggestedMin:0, suggestedMax:100, ticks:{ callback:(v)=>v+'%' } } }
        }
      });
    }

    function renderList(probs){
      const data = ORDER.map(k => ({ k, v: (probs?.[k] ?? 0) }));
      const sum = data.reduce((a,b)=>a+b.v,0) || 1;
      $('probList').innerHTML = data.map(({k,v})=>{
        const p = Math.round(v*1000/sum)/10;
        return `
          <div class="line">
            <span class="badge" style="min-width:64px">${k}</span>
            <span class="pill">${p.toFixed(1)}%</span>
            <div class="bar" style="flex:1"><i style="width:${p}%"></i></div>
          </div>`;
      }).join('');
    }

    function renderTopBadge(probs){
      const entries = ORDER.map(k => [k, probs?.[k] ?? 0]);
      entries.sort((a,b)=>b[1]-a[1]);
      const [k,v] = entries[0] || ['-',0];
      const p = Math.round((v/(entries.reduce((s,[,x])=>s+x,0)||1))*1000)/10;
      $('topBadge').innerHTML = `<span>${k}</span> <span class="muted">${p.toFixed(1)}%</span>`;
    }

    function renderTemps(xs, ys, lo, hi){
      if(tempChart) tempChart.destroy();
      tempChart = new Chart($('tempChart'), {
        type:'line',
        data:{ labels: xs, datasets:[
          { label:'TEMP', data: ys, yAxisID:'y' },
          { label:'LO', data: ys.map(_=>lo), pointRadius:0, borderDash:[6,6], yAxisID:'y' },
          { label:'HI', data: ys.map(_=>hi), pointRadius:0, borderDash:[6,6], yAxisID:'y' },
        ]},
        options:{ plugins:{ legend:{ position:'bottom' } }, elements:{ point:{ radius:2 } } }
      });
    }

    function fillSample(which){
      const normal = {
        "IMEI":"demo","deviceType":"L100",
        "deviceDataList":[
          {"DATETIME":"2025-01-01 01:00:00","TEMP":"4.9"},
          {"DATETIME":"2025-01-01 01:10:00","TEMP":"5.0"},
          {"DATETIME":"2025-01-01 01:20:00","TEMP":"5.1"},
          {"DATETIME":"2025-01-01 01:30:00","TEMP":"5.2"},
          {"DATETIME":"2025-01-01 01:40:00","TEMP":"5.2"},
          {"DATETIME":"2025-01-01 01:50:00","TEMP":"5.3"}
        ]};
      const start = {
        "IMEI":"demo","deviceType":"L100",
        "deviceDataList":[
          {"DATETIME":"2025-01-01 01:00:00","TEMP":"10.5"},
          {"DATETIME":"2025-01-01 01:10:00","TEMP":"8.5"},
          {"DATETIME":"2025-01-01 01:20:00","TEMP":"7.6"},
          {"DATETIME":"2025-01-01 01:30:00","TEMP":"7.2"},
          {"DATETIME":"2025-01-01 01:40:00","TEMP":"7.0"},
          {"DATETIME":"2025-01-01 01:50:00","TEMP":"6.8"}
        ]};
      $('payload').value = JSON.stringify(which==='start'?start:normal, null, 2);
    }

    async function analyze(){
      $('status').textContent = '요청 중...';
      try{
        const payload = JSON.parse($('payload').value);
        const resp = await fetch('/predict_pattern', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });
        const text = await resp.text();
        log('[CLIENT] HTTP ' + resp.status);
        if(!resp.ok){ $('status').textContent = '오류 ' + resp.status + ': ' + text; return; }
        const data = JSON.parse(text);

        // --- 확률 렌더 ---
        renderDonut(data.probs || {});
        renderBars(data.probs || {});
        renderList(data.probs || {});
        renderTopBadge(data.probs || {});

        // --- 메타/시계열 ---
        const xs = payload.deviceDataList.map(d=>d.DATETIME);
        const ys = payload.deviceDataList.map(d=>parseFloat(d.TEMP));
        const di = data.diagnostics || {};
        $('kvGate').textContent = di.gate ?? '-';
        $('kvLen').textContent  = di.len_series ?? '-';
        $('kvRange').textContent= (di.lo ?? '-') + ' ~ ' + (di.hi ?? '-');
        $('kvStats').textContent= (di.slope ?? di.slope_per_step ?? '-') + ' / ' + (di.std ?? '-');
        renderTemps(xs, ys, di.lo ?? null, di.hi ?? null);

        $('status').innerHTML = '<span class="small">완료</span>';
      }catch(e){
        $('status').textContent = '클라이언트 오류: ' + e.message;
        log('[CLIENT] ' + e.message);
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      $('analyzeBtn').addEventListener('click', analyze);
      $('btnNormal').addEventListener('click', () => fillSample('normal'));
      $('btnStart').addEventListener('click', () => fillSample('start'));
      log('[CLIENT] UI ready');
    });
  </script>
</body>
</html>
    """