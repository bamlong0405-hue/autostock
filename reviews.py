# reviews.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import math

def _fmt(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except Exception:
        return None

def make_quick_review(row: dict, tail_df: Optional[pd.DataFrame], cfg: dict) -> List[str]:
    """
    row: results의 한 행(dict). 예: {"symbol":"000660","rsi":..., "wr":..., "per":..., "pbr":...}
    tail_df: 해당 종목의 최근 구간 피처 DF (Close, MA_M, OBV_slope, Bullish, Bull_Engulf 등 포함 가능)
    cfg: config.toml 로드된 dict
    return: 한줄 리뷰 리스트
    """
    lines: List[str] = []

    # --- 기술 요약 ---
    rsi = _fmt(row.get("rsi"))
    wr  = _fmt(row.get("wr"))
    gap = _fmt(row.get("ma20_gap_pct"))

    if rsi is not None:
        if rsi <= 30:
            lines.append(f"RSI {rsi} (과매도권)")
        elif rsi >= 70:
            lines.append(f"RSI {rsi} (과매수권)")
        else:
            lines.append(f"RSI {rsi} (중립)")

    if wr is not None:
        if wr <= -80:
            lines.append(f"Williams %R {wr} (저점권)")
        elif wr >= -20:
            lines.append(f"Williams %R {wr} (고점권)")
        else:
            lines.append(f"Williams %R {wr} (중립)")

    if gap is not None:
        if gap < -5:
            lines.append(f"MA20 이격 {gap}% (평균선 하단, 되돌림 여지)")
        elif gap > 5:
            lines.append(f"MA20 이격 {gap}% (평균선 상단, 과열 유의)")
        else:
            lines.append(f"MA20 이격 {gap}%")

    if isinstance(tail_df, pd.DataFrame) and not tail_df.empty:
        last = tail_df.iloc[-1]
        obv_slope = _fmt(last.get("OBV_slope"), 5)
        if obv_slope is not None:
            if obv_slope > 0:
                lines.append(f"OBV 기울기 {obv_slope} (수급 개선)")
            elif obv_slope < 0:
                lines.append(f"OBV 기울기 {obv_slope} (수급 약화)")
        if bool(last.get("Bull_Engulf", False)):
            lines.append("상승 장악형 캔들 포착")
        if bool(last.get("Bullish", False)):
            lines.append("양봉/강세 캔들 신호")

    # --- 펀더멘털 힌트 (있을 때만) ---
    per = _fmt(row.get("per"))
    pbr = _fmt(row.get("pbr"))
    div = _fmt(row.get("div"))

    # 임계값은 config에서 오버라이드 가능
    fv = cfg.get("fundamental_view", {})
    per_low  = float(fv.get("per_low", 8))
    per_high = float(fv.get("per_high", 25))
    pbr_low  = float(fv.get("pbr_low", 1.0))
    div_good = float(fv.get("div_good", 2.0))  # 배당수익률(%)

    if per is not None:
        if per <= per_low:
            lines.append(f"PER {per} (저평가 구간 가능)")
        elif per >= per_high:
            lines.append(f"PER {per} (성장 기대 반영)")

    if pbr is not None:
        if pbr <= pbr_low:
            lines.append(f"PBR {pbr} (자산가치 대비 저평가 가능)")

    if div is not None and div >= div_good:
        lines.append(f"배당수익률 {div}% (배당 매력)")

    # 비어있으면 최소 한 줄 보장
    if not lines:
        lines.append("핵심 지표 정상 계산 (특이사항 없음)")

    return lines
