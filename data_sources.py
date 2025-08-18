import pandas as pd
import datetime as dt
from typing import Optional, List
import yfinance as yf
from pykrx import stock

def fetch_krx(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, symbol)
    df = df.rename(columns={'시가':'Open','고가':'High','저가':'Low','종가':'Close','거래량':'Volume'})
    df.index = pd.to_datetime(df.index)
    df = df[['Open','High','Low','Close','Volume']].astype(float)
    return df

def fetch_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
    df = df[['Open','High','Low','Close','Volume']].astype(float)
    return df

def date_range_for_lookback(lookback_days: int, buffer_days: int = 30):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days + buffer_days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def get_all_krx_tickers(
    exclude_etf_etn_spac: bool = True,
    name_filters_exclude: Optional[List[str]] = None,
    limit: int = 0
) -> List[str]:
    """코스피+코스닥 전체 티커 반환 (간단한 이름 기반 필터 제공)."""
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    all_codes = list(dict.fromkeys(list(kospi) + list(kosdaq)))

    name_map = {code: stock.get_market_ticker_name(code) for code in all_codes}

    if exclude_etf_etn_spac:
        default_keys = ["ETF", "ETN", "스팩", "SPAC"]
        keys = set(default_keys + (name_filters_exclude or []))
        def _is_excluded(name: str) -> bool:
            if not isinstance(name, str):
                return False
            up = name.upper()
            return any(k.upper() in up for k in keys)
        all_codes = [c for c in all_codes if not _is_excluded(name_map.get(c, ""))]

    if limit and limit > 0:
        all_codes = all_codes[:limit]

    return all_codes

def get_krx_name(code: str) -> str:
    try:
        return stock.get_market_ticker_name(code)
    except Exception:
        return code
