# tests/test_predict.py
# 예상 결과가 정해져 있는 경우를 검증

import pytest, pathlib
from core import excel_bytes_to_series, predict_pattern

# (fname, 기대_라벨) 목록
cases = [
    ("정상_SENSOR_7월 (20).xlsx", "정상"),
    ("오픈_SENSOR_280876.xlsx", "오픈"),
    ("상향_SENSOR_281416_gen_17.xlsx", "상향"),
    ("상향_SENSOR_281416_gen_18.xlsx", "상향"),
    ("시작_SENSOR_ (2).xlsx", "시작"),
    ("시작_SENSOR_ (7).xlsx", "시작"),
    ("오픈_SENSOR_27005A.xlsx", "오픈"),
    ("정상_SENSOR_12월 (13).xlsx", "정상"),
    ("하향_SENSOR_(7)_gen_1_8.xlsx", "하향"),
    ("하향_SENSOR_(9)_gen_5.xlsx", "하향"),
]

@pytest.mark.parametrize("fname,expected", cases)
def test_predict_labels(fname, expected):
    path = pathlib.Path(__file__).parent / "data" / fname
    series = excel_bytes_to_series(open(path, "rb").read())
    label, probs = predict_pattern(series)
    assert label == expected
    # 확률 합이 1.0 인지
    assert abs(sum(probs.values()) - 1.0) < 1e-6
