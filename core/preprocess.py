import pandas as pd, numpy as np, re, unicodedata
from io import BytesIO

WIN_MIN = 10
PAD_VAL = -999.0

def json_to_series(payload: dict) -> np.ndarray:
    """
    payload 예시:
    {
      "points": [
         {"time":"2025-05-26T09:00:00+09:00","temp":4.5},
         {"time":"2025-05-26T09:10:00+09:00","temp":4.7}
      ]
    }
    """
    df = pd.DataFrame(payload["points"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = (df.set_index("time")
            .resample(f"{WIN_MIN}min").mean()
            .interpolate().ffill().bfill())
    return df["temp"].to_numpy("float32")

def excel_bytes_to_series(data: bytes) -> np.ndarray:
    for hdr in (9, 0):
        try:
            df = pd.read_excel(BytesIO(data), header=hdr)
            df.columns = df.columns.str.strip()
            df = (df.assign(측정시간=lambda d: pd.to_datetime(d['측정시간'], errors='coerce'),
                            temp      =lambda d: pd.to_numeric(d['온도(℃)'], errors='coerce'))
                    .dropna(subset=['측정시간'])
                    .groupby('측정시간', as_index=False).mean()
                    .set_index('측정시간')
                    .resample(f"{WIN_MIN}min").mean()
                    .interpolate('linear').ffill().bfill())
            return df['temp'].to_numpy('float32')
        except Exception:
            continue
    raise ValueError("엑셀 포맷 불일치: '측정시간','온도(℃)' 열이 없습니다.")
