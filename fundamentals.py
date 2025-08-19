# fundamentals.py
from __future__ import annotations
import pandas as pd
from datetime import datetime, timedelta
from pykrx.stock import get_market_fundamental

def yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def prev_business_day(today: datetime | None = None) -> str:
    d = today or datetime.now()
    # KRX 휴일 처리 간소화: 전날부터 거꾸로(최대 5일) 스캔
    for i in range(1, 6):
        day = d - timedelta(days=i)
        if day.weekday() < 5:  # Mon~Fri
            return yyyymmdd(day)
    return yyyymmdd(d - timedelta(days=1))

def fetch_fundamental_map(tickers: list[str]) -> dict[str, dict]:
    """
    tickers: '000660' 형식 (KRX 코드)
    return: { '000660': {'PER': 8.3, 'PBR': 1.1, 'DIV': 1.9}, ...}
    """
    ref = prev_business_day()
    try:
        df = get_market_fundamental(ref, market="ALL")  # KOSPI+KOSDAQ
        df = df.rename_axis("Ticker").reset_index()
        df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)
        df = df[["Ticker", "PER", "PBR", "DIV"]]
    except Exception:
        return {}

    need = set(tickers)
    out = {}
    for _, r in df.iterrows():
        t = r["Ticker"]
        if t in need:
            out[t] = {
                "PER": _safe_float(r.get("PER")),
                "PBR": _safe_float(r.get("PBR")),
                "DIV": _safe_float(r.get("DIV")),
            }
    return out

def _safe_float(v):
    try:
        x = float(v)
        if pd.isna(x): return None
        return x
    except Exception:
        return None
