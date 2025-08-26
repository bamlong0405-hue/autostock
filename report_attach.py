# report_attach.py
import io
import os
import math
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt

# (선택) 간단 리뷰 문구 생성 — 없으면 무시
try:
    from reviews import make_quick_review
except Exception:
    def make_quick_review(row_info, tail_df, cfg):
        return []

HTML_HEADER = """<!doctype html><html><head><meta charset="utf-8">
<title>BUY Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Pretendard, Arial, sans-serif; margin: 24px; }
h1 { font-size: 20px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 24px 0 8px; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.meta { color: #666; font-size: 12px; margin-bottom: 8px; }
.kv { font-size: 13px; }
.kv b { display: inline-block; width: 120px; }
img { display:block; max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
ul { margin: 6px 0 0 18px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; margin-right:4px; background:#eef2ff; border:1px solid #dbeafe; }
.box { margin-top:8px; padding:10px; border:1px solid #e5e7eb; border-radius:10px; background:#fafafa }
.note{ color:#555; font-size:12px; }
.table { margin-top:6px; font-size:13px; border-collapse:collapse }
.table td { padding:2px 6px; border-bottom:1px dashed #eee }
.subtle { color:#888; font-size:12px }
</style>
</head><body>
"""

HTML_FOOTER = "</body></html>"


# -----------------------------
# 작은 유틸
# -----------------------------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _small_price_chart(df: pd.DataFrame, name: str) -> str:
    """
    최근 60봉 종가/MA20/볼밴 표시
    """
    d = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.plot(d.index, d["Close"], label="Close")
    if "MA_M" in d.columns:
        ax.plot(d.index, d["MA_M"], label="MA20")
    # 볼밴 (있으면)
    if {"BB_UPPER", "BB_LOWER"}.issubset(set(d.columns)):
        ax.plot(d.index, d["BB_UPPER"], linewidth=0.8, label="BB Upper")
        ax.plot(d.index, d["BB_LOWER"], linewidth=0.8, label="BB Lower")
    ax.set_title(name)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    return _fig_to_base64(fig)


def _near(x: float, y: float, tol: float) -> bool:
    """x가 y에 tol 비율 이내로 가까운가 (예: tol=0.01 → 1%)"""
    if x is None or y is None or not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return False
    if y == 0:
        return abs(x - y) <= tol
    return abs(x - y) / abs(y) <= tol


def _fib_zone_label(close: float, low: float, high: float, tol_ratio: float = 0.01) -> Optional[str]:
    """
    최근 N봉 high/low 기준으로 38.2 / 50 / 61.8 % 되돌림 근처면 레이블 반환
    """
    if any(v is None for v in (close, low, high)) or math.isclose(high, low):
        return None
    span = high - low
    f382 = high - 0.382 * span
    f500 = high - 0.500 * span
    f618 = high - 0.618 * span
    if _near(close, f382, tol_ratio): return "피보나치 38.2%"
    if _near(close, f500, tol_ratio): return "피보나치 50%"
    if _near(close, f618, tol_ratio): return "피보나치 61.8%"
    return None


def _combo_signals(last: pd.Series, tail: pd.DataFrame) -> Dict[str, Any]:
    """
    컨플루언스(지표 조합) 검출:
      - BB_POS_PCT ≤ 20, RSI ≤ 35
      - MACD 골든/데드 크로스
      - WR ≤ -80
      - MA20 하회(과도 괴리 음수)
      - 거래량 급증 (당일 > 20일 평균 * 1.5)
      - 피보나치 되돌림 레벨 근접
    반환: {'hits':[...], 'score':float}
    """
    hits: List[str] = []
    score = 0.0

    def _get(name):
        v = last.get(name, None)
        try:
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    # 1) 볼밴 위치 & RSI
    bb_pos = _get("BB_POS_PCT")
    rsi = _get("RSI")
    if bb_pos is not None and bb_pos <= 20:
        hits.append("볼밴 하단권")
        score += 0.8
    if rsi is not None and rsi <= 35:
        hits.append("RSI 저평가 구간")
        score += 0.8

    # 2) MACD 골든/데드
    macd = _get("MACD"); macd_sig = _get("MACD_SIG")
    macd_prev = tail["MACD"].iloc[-2] if "MACD" in tail.columns and len(tail) >= 2 else None
    macd_sig_prev = tail["MACD_SIG"].iloc[-2] if "MACD_SIG" in tail.columns and len(tail) >= 2 else None
    macd_golden = (
        macd is not None and macd_sig is not None and macd_prev is not None and macd_sig_prev is not None
        and macd > macd_sig and macd_prev <= macd_sig_prev
    )
    macd_dead = (
        macd is not None and macd_sig is not None and macd_prev is not None and macd_sig_prev is not None
        and macd < macd_sig and macd_prev >= macd_sig_prev
    )
    if macd_golden:
        hits.append("MACD 골든크로스")
        score += 1.0
    if macd_dead:
        hits.append("MACD 데드크로스(주의)")
        score -= 0.8

    # 3) WR 저점권
    wr = _get("WR")
    if wr is not None and wr <= -80:
        hits.append("WR 저점권")
        score += 0.6

    # 4) MA20 괴리(음수면 하단)
    close = _get("Close"); ma20 = _get("MA_M")
    if close is not None and ma20 and ma20 > 0:
        gap_pct = (close - ma20) / ma20 * 100.0
        if gap_pct <= -3.0:
            hits.append(f"20일선 하단(-{abs(gap_pct):.1f}%)")
            score += 0.5

    # 5) 거래량 급증
    vol = _get("Volume"); vol_ma20 = _get("VOL_MA20")
    if vol is not None and vol_ma20 and vol_ma20 > 0 and vol > (vol_ma20 * 1.5):
        hits.append("거래량 급증")
        score += 0.6

    # 6) 피보나치 레벨
    if {"High", "Low", "Close"}.issubset(tail.columns) and len(tail) >= 40:
        swing_low = float(tail["Low"].tail(40).min())
        swing_high = float(tail["High"].tail(40).max())
        fib = _fib_zone_label(close, swing_low, swing_high, tol_ratio=0.01)
        if fib:
            hits.append(f"{fib} 지지·저항 부근")
            score += 0.4

    # 7) 캔들 장악형(있으면 가점)
    bull_eng = bool(last.get("Bull_Engulf", False))
    if bull_eng:
        hits.append("상승 장악형")
        score += 0.6

    return {"hits": hits, "score": round(score, 2)}


# -----------------------------
# 본문 생성
# -----------------------------
def build_buy_attachment(
    buy_rows: List[dict],
    details: Dict[str, pd.DataFrame],
    cfg: dict,
    out_path: str = "output/buy_report.html",
    max_charts: int = 10,
    aux_info: Optional[Dict[str, Any]] = None,
    embed_charts: bool = False,
    turnover_map: Optional[Dict[str, Dict[str, Any]]] = None,  # {sym:{turnover_pct, board}}
) -> str:
    """
    buy_rows : signal == BUY 인 dict 리스트
    details  : 심볼 -> 피처 DataFrame (Close/MA_M/RSI/WR/OBV_slope/VOL_MA20/BB_* 포함 권장)
    aux_info : 심볼 -> {"news":[{title,link,published}], "filings":[{rpt,link,rcpdt}]}
    turnover_map: 심볼 -> {"turnover_pct": float(일일%), "board":"KOSPI/KOSDAQ"}
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html: List[str] = [HTML_HEADER, f"<h1>BUY Report</h1><div class='meta'>Generated at {ts}</div>"]

    # 첨부 용량 보호: 상한
    take = buy_rows[:max_charts]

    for r in take:
        sym = r["symbol"]
        mkt = r.get("market", "")
        name = r.get("name", sym)
        df = details.get(sym)

        # 차트
        chart_uri = ""
        if embed_charts and isinstance(df, pd.DataFrame) and not df.empty:
            chart_uri = _small_price_chart(df, f"{name} ({sym})")

        # 기본 키값
        rsi = r.get("rsi", None)
        wr  = r.get("wr", None)
        gap = r.get("ma20_gap_pct", None)

        # 카드 시작
        html.append("<div class='card'>")
        html.append(f"<h2>{name} <span class='subtle'>({mkt}:{sym})</span></h2>")

        # 종가 / 전일 거래량 / 회전율
        last_close = None
        prev_vol = None
        tov_txt = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            last_close = float(last.get("Close")) if pd.notna(last.get("Close")) else None
            if len(df) >= 2:
                prev_vol = float(df["Volume"].iloc[-2]) if pd.notna(df["Volume"].iloc[-2]) else None

        if turnover_map and sym in turnover_map:
            tinfo = turnover_map[sym]
            tov = tinfo.get("turnover_pct")
            brd = tinfo.get("board")
            tov_txt = f"{tov:.2f}%/day ({brd})" if isinstance(tov, (int, float)) else None

        html.append("<table class='table'>")
        if last_close is not None:
            html.append(f"<tr><td><b>종가</b></td><td>{last_close:,.2f}</td></tr>")
        if prev_vol is not None:
            html.append(f"<tr><td><b>전일 거래량</b></td><td>{int(prev_vol):,}</td></tr>")
        if tov_txt:
            html.append(f"<tr><td><b>회전율</b></td><td>{tov_txt}</td></tr>")
        if rsi is not None:
            html.append(f"<tr><td><b>RSI</b></td><td>{float(rsi):.2f}</td></tr>")
        if wr is not None:
            html.append(f"<tr><td><b>Williams %R</b></td><td>{float(wr):.2f}</td></tr>")
        if gap is not None:
            html.append(f"<tr><td><b>MA20 Gap</b></td><td>{float(gap):.2f}%</td></tr>")
        html.append("</table>")

        if chart_uri:
            html.append(f"<img src='{chart_uri}' alt='chart'/>")

        # ---- 기술 근거(기존) ----
        reasons: List[str] = []
        if rsi is not None:
            reasons.append("RSI 값이 낮아 '싸졌을' 가능성을 시사")
        if wr is not None:
            reasons.append("윌리엄스 %R이 저점권이면 '바닥에서 반등' 기대")
        if gap is not None:
            reasons.append("20일선과의 괴리로 과열/침체 정도를 파악")

        # OBV/캔들/볼밴 위치 텍스트
        if isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            if "OBV_slope" in df.columns and pd.notna(last.get("OBV_slope", None)):
                reasons.append("OBV 기울기로 수급(매수/매도 에너지) 방향 참고")
            if "Bull_Engulf" in df.columns and bool(last.get("Bull_Engulf", False)):
                reasons.append("캔들 패턴: 상승 장악형 포착")
            if "BB_POS_PCT" in df.columns and pd.notna(last.get("BB_POS_PCT", None)):
                reasons.append(f"볼린저밴드 위치 {float(last['BB_POS_PCT']):.1f}% (0% 하단·100% 상단)")

        if reasons:
            html.append("<ul>")
            for t in reasons:
                html.append(f"<li>{t}</li>")
            html.append("</ul>")

        # ---- ★ 지표 조합(컨플루언스) 박스 ----
        if isinstance(df, pd.DataFrame) and len(df) >= 3:
            tail = df.tail(60).copy()
            last = tail.iloc[-1]
            combo = _combo_signals(last, tail)
            hits = combo["hits"]; score = combo["score"]

            if hits:
                html.append("<div class='box'>")
                html.append("<div style='font-weight:600;margin-bottom:6px'>지표 조합 신호</div>")
                html.append("<div class='note'>여러 지표가 동시에 같은 방향을 가리키면 신뢰도가 높아집니다.</div>")
                html.append("<div style='margin:6px 0'>")
                for h in hits:
                    html.append(f"<span class='badge'>{h}</span>")
                html.append("</div>")
                html.append(f"<div class='subtle'>Combo score: <b>{score:.2f}</b> "
                            "(MACD/RSI/볼밴/WR/MA20괴리/거래량/피보 기준)</div>")
                html.append("</div>")

        # ---- (선택) 리뷰(간단 요약) ----
        try:
            tail_df = df.tail(60).copy() if isinstance(df, pd.DataFrame) and not df.empty else None
            review_lines = make_quick_review(r, tail_df, cfg)
            if review_lines:
                html.append("<div class='box'><div style='font-weight:600;margin-bottom:6px'>리뷰(요약)</div><ul style='margin:0 0 0 18px;padding:0'>")
                for ln in review_lines:
                    html.append(f"<li style='margin:2px 0'>{ln}</li>")
                html.append("</ul></div>")
        except Exception as e:
            html.append(f"<div class='subtle'>[리뷰 생성 오류: {e}]</div>")

        # ---- 뉴스 ----
        if aux_info and sym in aux_info and aux_info[sym].get("news"):
            html.append("<div class='box'><div style='font-weight:600;margin-bottom:6px'>최근 뉴스</div><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["news"]:
                title = it.get("title", "")
                link = it.get("link", "#")
                pub  = it.get("published", "")
                html.append(f"<li style='margin:2px 0'><a href='{link}' target='_blank'>{title}</a> "
                            f"<span class='subtle'>({pub})</span></li>")
            html.append("</ul></div>")

        # ---- 공시(DART) ----
        if aux_info and sym in aux_info and aux_info[sym].get("filings"):
            html.append("<div class='box'><div style='font-weight:600;margin-bottom:6px'>최근 공시(DART)</div><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["filings"]:
                rpt   = it.get("rpt", "")
                link  = it.get("link", "#")
                rcpdt = it.get("rcpdt", "")
                html.append(f"<li style='margin:2px 0'><a href='{link}' target='_blank'>{rpt}</a> "
                            f"<span class='subtle'>({rcpdt})</span></li>")
            html.append("</ul></div>")

        html.append("</div>")  # /card

    html.append(HTML_FOOTER)
    html_str = "".join(html)

    os.makedirs(cfg["general"]["output_dir"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path
