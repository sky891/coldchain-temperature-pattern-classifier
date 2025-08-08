# adapters/fastapi/main.py

import os
import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing    import List, Dict
from pydantic import BaseModel
from core.preprocess import excel_bytes_to_series, json_to_series
from core.predictor import predict_pattern, calc_risk

app = FastAPI(title="Temperature Drift API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def load_temp_limits() -> dict[str, tuple[float, float]]:
    """
    adapters/fastapi/config/temp_limits.json 을 읽어서
    {'TT19EX': (lo,hi), ...} 구조로 반환
    """
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        "config", "temp_limits.json"
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 리스트를 튜플로, 숫자형으로 바꿔 줍니다
    return { dt: (float(lo), float(hi)) for dt, (lo, hi) in data.items() }

def get_temp_limits(device_type: str) -> tuple[float, float]:
    limits = load_temp_limits()
    # 설정에 없으면 기본값 (예: -100~100)
    return limits.get(device_type, (-100.0, 100.0))

# 입력 페이로드 정의
class SensorPayload(BaseModel):
    IMEI:        str
    Temperature: str
    deviceType:  str

# 실시간 위험도 API
@app.post("/predict_stream")
def predict_stream(req: SensorPayload):
    # 온도 파싱
    try:
        temp = float(req.Temperature)
    except ValueError:
        raise HTTPException(400, "Invalid Temperature")

    # 허용 범위 꺼내기
    lo, hi = get_temp_limits(req.deviceType)

    # 위험도 계산
    risk   = calc_risk(temp, lo, hi)
    status = "out_of_range" if risk >= 1.0 else "in_range"

    return {
        "status": status,
        "risk":   round(risk, 4),
        "lo":     lo,
        "hi":     hi
    }

# # 2) 파일 전체 패턴 예측
# @app.post("/predict_file")
# async def predict_file(file: UploadFile = File(...)):
#     data = await file.read()
#     try:
#         series = excel_bytes_to_series(data)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     label, probs = predict_pattern(series)
#     return {"pattern": label, "probs": probs}

class PatternPayload(BaseModel):
    IMEI:           str
    deviceType:     str
    deviceDataList: List[Dict]   # device-data API 리스폰스 배열

@app.post("/predict_pattern")
def predict_pattern_api(req: PatternPayload):
    # 1) points 배열로 변환
    points = [
      {"time": item["DATETIME"], "temp": float(item["TEMP"])}
      for item in req.deviceDataList
    ]
    # 2) 시계열 NumPy 배열로 변환
    series = json_to_series({"points": points})
    # 3) 패턴 예측
    label, probs = predict_pattern(series)
    return {"pattern": label, "probs": probs}
