import numpy as np
from .model_loader import get_model, PAD_VAL
LABEL2IDX = {'정상':0,'시작':1,'오픈':2,'상향':3,'하향':4}
IDX2LBL   = {v:k for k,v in LABEL2IDX.items()}

def calc_risk(temp:float, lo:float, hi:float) -> float:
    """온도가 범위 안일 때 0~1 위험도, 범위 밖이면 1.0 바로 사용"""
    if temp < lo or temp > hi:
        return 1.0
    mid, half = (lo+hi)/2, (hi-lo)/2
    return min(1.0, (abs(temp-mid)/half)**2)   # 거리^2 파워커브

def predict_pattern(series: np.ndarray):
    model, max_len = get_model()

    # 1) 길이가 max_len보다 크면 뒤쪽(최신) max_len개만 사용
    if len(series) > max_len:
        series = series[-max_len:]

    x = np.full((1,max_len,1), PAD_VAL, 'float32')
    x[0, :len(series), 0] = series
    prob = model.predict(x, verbose=0)[0]
    top  = int(prob.argmax())
    return IDX2LBL[top], { c: float(prob[i]) for c,i in LABEL2IDX.items() }
