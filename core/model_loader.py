import os, pickle, tensorflow as tf, threading

MODEL_F = os.path.join(os.path.dirname(__file__), "assets/pattern_entire.keras")
META_F  = os.path.join(os.path.dirname(__file__), "assets/meta.pkl")
PAD_VAL = -999.0

_lock   = threading.Lock()
_model  = None
_maxlen = None

def get_model():
    global _model, _maxlen
    with _lock:
        if _model is None:
            _model  = tf.keras.models.load_model(MODEL_F, compile=False)
            _maxlen = pickle.load(open(META_F,"rb"))["max_len"]
    return _model, _maxlen
