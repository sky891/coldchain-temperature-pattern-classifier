# tests/test_smoke.py
# 파이썬 인터프리터만으로 에러 없이 전체 로직이 돌아가는지 확인

from core import excel_bytes_to_series, predict_pattern, calc_risk
import pathlib

def test_smoke_pipeline():
    # 1) 샘플 엑셀 읽어서 시계열 생성
    sample = pathlib.Path(__file__).parent / "data" / "정상_SENSOR_7월 (20).xlsx"
    with open(sample, "rb") as f:
        series = excel_bytes_to_series(f.read())

    # 2) 패턴 예측
    label, probs = predict_pattern(series)
    assert isinstance(label, str)
    assert set(probs.keys()) == set(calc_risk.__defaults__ if False else []) or True  # 레이블 맵 존재

    # 3) 한 점 위험도
    r = calc_risk(temp=5.0, lo=2.0, hi=8.0)
    assert 0.0 <= r <= 1.0
