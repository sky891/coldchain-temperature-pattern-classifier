# core/predictor.py
import os, json, pickle
import numpy as np
import tensorflow as tf
from .model_loader import get_model, PAD_VAL  # 쓰지 않으면 지워도 됌
from core.features import make_features_5ch  # featrue 추가

LABEL2IDX = {'정상':0,'시작':1,'오픈':2,'상향':3,'하향':4}
IDX2LBL   = {v:k for k,v in LABEL2IDX.items()}

def calc_risk(temp:float, lo:float, hi:float) -> float:
    if temp < lo or temp > hi:
        return 1.0
    mid, half = (lo+hi)/2, (hi-lo)/2
    return min(1.0, (abs(temp-mid)/half)**2)

def _load_limits(device_type: str):
    cfg_path = os.path.join('adapters', 'fastapi', 'config', 'temp_limits.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    rec = cfg.get(device_type)
    if rec is None:
        raise ValueError(f"unknown device_type: {device_type}")
    if isinstance(rec, dict):
        return float(rec['lo']), float(rec['hi'])
    lo, hi = rec
    return float(lo), float(hi)

def predict_pattern(series, device_type: str):
    meta = pickle.load(open('core/assets/meta.pkl','rb'))
    max_len = int(meta.get('max_len') or meta.get('SEQ_LEN') or 512)
    pad_val = float(meta.get('PAD_VAL', -999.0))

    model = tf.keras.models.load_model('core/assets/pattern_entire.keras', compile=False)

    lo, hi = _load_limits(device_type)
    arr = np.asarray(series, dtype='float32')              # (T,)
    x5  = make_features_5ch(arr, lo, hi, max_len, pad_val) # (L,5)
    x = x5[None, ...]  # (1, L, 5)


    prob = model.predict(x, verbose=0)[0]

    idx2label = meta.get('idx2label') or meta.get('IDX2LABEL')
    if isinstance(idx2label, dict):
        idx2label = {int(k): v for k, v in idx2label.items()}
    label = idx2label[int(np.argmax(prob))]
    return label, prob.tolist()
