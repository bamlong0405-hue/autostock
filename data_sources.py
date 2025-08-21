# data_sources.py (clean, optimized)

from __future__ import annotations

import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, List, Dict

import yfinance as yf
from pykrx import stock


# =========================
# 기본 시세 소스
# =========================
def fetch_krx(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    start/end: 'YYYYMMDD' 문자열 (pykrx 포맷)
    """
    df = stock.get_market_ohlcv_by_date(start, end, symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
    )
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def fetch_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    start/end: 'YYYY-MM-DD' (yfinance 포맷)
    """
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # 컬럼 정리
    df = df.rename(
        columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def date_range_for_lookback(lookback_days: int, buffer_days: int = 30):
    """
    분석 lookback 일수 + 버퍼일수로 pykrx/yahoo 두 포맷 날짜를 함께 반환
    """
    end = dt.datetime.now()
    start = end - dt.timedelta(days=lookback_days + buffer_days)
    # pykrx 포맷, yahoo 포맷 동시 반환
    return (
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )


# =========================
# 티커/이름 유틸
# =========================
def get_all_krx_tickers(
    exclude_etf_etn_spac: bool = True,
    name_filters_exclude: Optional[List[str]] = None,
    limit: int = 0,
) -> List[str]:
    """
    코스피+코스닥 전체 티커 반환 (간단한 이름 기반 필터 제공)
    """
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    all_codes = list(dict.fromkeys(list(kospi) + list(kosdaq)))

    if exclude_etf_etn_spac or (name_filters_exclude and len(name_filters_exclude) > 0):
        name_map = {code: stock.get_market_ticker_name(code) for code in all_codes}
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


# =========================
# 보드세트 & 회전율(시총 대비 거래대금) 스냅샷
# =========================

@lru_cache(maxsize=1)
def get_board_sets(date_str: str | None = None) -> Dict[str, set]:
    """
    KOSPI / KOSDAQ 소속 심볼 세트 리턴
    date_str: 'YYYYMMDD' (없으면 오늘 기준)
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    try:
        kospi = set(stock.get_market_ticker_list(date_str, market="KOSPI"))
    except Exception:
        kospi = set()
    try:
        kosdaq = set(stock.get_market_ticker_list(date_str, market="KOSDAQ"))
    except Exception:
        kosdaq = set()
    return {"KOSPI": kospi, "KOSDAQ": kosdaq}


def _fetch_cap_and_value(date_str: str) -> pd.DataFrame:
    """
    특정 영업일 기준 시총/거래대금 스냅샷을 한 번에 수집
    반환: index=ticker, cols=['시가총액','거래대금']
    """
    cap_kospi = stock.get_market_cap_by_ticker(date_str, market="KOSPI")[["시가총액"]]
    cap_kosdaq = stock.get_market_cap_by_ticker(date_str, market="KOSDAQ")[["시가총액"]]
    cap = pd.concat([cap_kospi, cap_kosdaq], axis=0, sort=False)

    val_kospi = stock.get_market_trading_value_by_ticker(date_str, market="KOSPI")[["거래대금"]]
    val_kosdaq = stock.get_market_trading_value_by_ticker(date_str, market="KOSDAQ")[["거래대금"]]
    val = pd.concat([val_kospi, val_kosdaq], axis=0, sort=False)

    df = cap.join(val, how="outer")
    return df


def attach_turnover_krx(symbols: List[str], date_str: str | None = None) -> Dict[str, Dict]:
    """
    회전율 스냅샷(당일 또는 가장 최근 영업일)을 빠르게 계산
    - symbols: 6자리 KRX 티커 리스트
    - return: { '000660': {'turnover_pct': float|None, 'board': 'KOSPI'|'KOSDAQ'|None}, ... }
    """
    df = pd.DataFrame()
    use_date = None

    if date_str is None:
        # 오늘부터 최대 7일 이전까지 영업일 탐색 (휴일/주말 대비)
        for i in range(0, 7):
            try_date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            try:
                tmp = _fetch_cap_and_value(try_date)
            except Exception:
                tmp = pd.DataFrame()
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                df = tmp
                use_date = try_date
                break
    else:
        try:
            df = _fetch_cap_and_value(date_str)
            use_date = date_str
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        # 실패 시 None 채워서 반환
        return {s: {"turnover_pct": None, "board": None} for s in symbols}

    brd = get_board_sets(use_date)
    out: Dict[str, Dict] = {}
    for s in symbols:
        row = df.loc[s] if s in df.index else None
        cap = float(row["시가총액"]) if row is not None and pd.notna(row.get("시가총액")) else None
        val = float(row["거래대금"]) if row is not None and pd.notna(row.get("거래대금")) else None
        pct = (val / cap * 100.0) if (cap and val and cap > 0) else None

        board = None
        if s in brd.get("KOSPI", set()):
            board = "KOSPI"
        elif s in brd.get("KOSDAQ", set()):
            board = "KOSDAQ"

        out[s] = {"turnover_pct": pct, "board": board}
    return out
