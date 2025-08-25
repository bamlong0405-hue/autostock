# report_attach.py
import io
import os
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt

# reviews 모듈이 없어도 동작하도록 안전하게 import
try:
    from reviews import make_quick_review  # 선택적
except Exception:
    make_quick_review = None

HTML_HEADER = """<!doctype html><html><head><meta charset="utf-8">
<title>BUY Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Pretendard, Arial, sans-serif; margin: 24px; }
h1 { font-size: 20px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 24px 0 8px; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.meta { color: #666; font-size: 12px; margin-bottom: 8px; }
.kv { font-size: 13px; display: grid; grid-template-columns: repeat(2, minmax(200px, 1fr)); gap: 4px 16px; }
.kv b { display: inline-block; width: 120px; }
img { display:block; max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
ul { margin: 6px 0 0 18px; }
.small { color:#666; font-size:12px; }
</style>
</head><body>
"""

HTML_FOOTER = "</body></html>"

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _small_price_chart(df: pd.DataFrame, name: str) -> str:
    # 최근 60봉 종가/MA20 + BB 대역(있으면)
    d = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.plot(d.index, d.get("Close"), label="Close")
    if "MA_M" in d.columns:
        ax.plot(d.index, d["MA_M"], label="MA20")
    # 볼린저밴드가 있으면 얇은 선으로
    if "BB_UPPER" in d.columns and "BB_LOWER" in d.columns:
        ax.plot(d.index, d["BB_UPPER"], linewidth=0.8, label="BB Upper")
        ax.plot(d.index, d["BB_LOWER"], linewidth=0.8, label="BB Lower")
    ax.set_title(name)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    return _fig_to_base64(fig)

def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"

def _fmt_float(x, nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

def build_buy_attachment(
    buy_rows: List[dict],
    details: Dict[str, pd.DataFrame],
    cfg: dict,
    out_path: str = "output/buy_report.html",
    max_charts: int = 10,
    aux_info: Optional[Dict[str, Any]] = None,  # ← 뉴스/공시 전달: {sym: {"news":[...], "filings":[...]}}
) -> str:
    """
    buy_rows: signal == BUY 인 종목 dict 리스트
              예) {"market": "KRX", "symbol": "000000", "name": "...",
                   "rsi":..., "wr":..., "ma20_gap_pct":..., "turnover_pct": ...}
    details : 심볼 -> 피처 DataFrame (Close/Volume/MA_M/RSI/WR/OBV_slope/BB_* 등)
    aux_info: 심볼 -> {"news":[{title,link,published}], "filings":[{rpt,link,rcpdt}]}
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html: List[str] = [HTML_HEADER, f"<h1>BUY Report</h1><div class='meta'>Generated at {ts}</div>"]

    take = buy_rows[:max_charts]
    for r in take:
        sym = r["symbol"]
        name = r.get("name", sym)
        df = details.get(sym)

        # 차트
        chart_uri = ""
        if isinstance(df, pd.DataFrame) and not df.empty:
            chart_uri = _small_price_chart(df, f"{name} ({sym})")

        # 최신 row
        last = df.iloc[-1] if isinstance(df, pd.DataFrame) and not df.empty else {}
        close_val = last.get("Close", r.get("close"))
        vol_val = last.get("Volume", None)

        # 회전율(%) 우선순위: row값 -> df의 Turnover -> 없음
        tnp = r.get("turnover_pct", None)
        if tnp is None and isinstance(last, pd.Series) and "Turnover" in last.index and pd.notna(last["Turnover"]):
            try:
                tnp = float(last["Turnover"]) * 100.0
            except Exception:
                tnp = None

        # 기본 지표 (rsi/wr/gap)
        rsi = r.get("rsi", None)
        wr  = r.get("wr", None)
        gap = r.get("ma20_gap_pct", None)

        # 추천 근거(기술) 텍스트
        reasons: List[str] = []
        if rsi is not None:
            reasons.append(f"RSI {float(rsi):.2f} — 과매도/반등 가능성 체크")
        if wr is not None:
            reasons.append(f"Williams %R {float(wr):.2f} — 저점권(-80↓) 근접 시 반등 모멘텀")
        if gap is not None:
            reasons.append(f"20일선 괴리 {float(gap):.2f}% — 과열/과매도 판단 근거")

        if isinstance(df, pd.DataFrame) and not df.empty:
            if "OBV_slope" in df.columns and pd.notna(last.get("OBV_slope", None)):
                reasons.append(f"OBV 추세(slope) {float(last['OBV_slope']):.5f} — 수급 흐름 참고")
            if "BB_POS_PCT" in df.columns and pd.notna(last.get("BB_POS_PCT", None)):
                reasons.append(f"볼린저밴드 위치 {float(last['BB_POS_PCT']):.1f}% — 0% 하단, 100% 상단")
            if "Bull_Engulf" in df.columns and bool(last.get("Bull_Engulf", False)):
                reasons.append("상승 장악형 캔들 포착")
            if "Bullish" in df.columns and bool(last.get("Bullish", False)):
                reasons.append("양봉/강세 캔들 신호")

        # 카드 시작
        html.append("<div class='card'>")
        html.append(f"<h2>{name} <span class='small'>({r['market']}:{sym})</span></h2>")

        # 핵심 값: 종가 / 전일 거래량 / 회전율
        html.append("<div class='kv'>")
        html.append(f"<div><b>종가</b> { _fmt_int(round(close_val)) if close_val is not None else '-' }</div>")
        html.append(f"<div><b>전일 거래량</b> { _fmt_int(vol_val) if vol_val is not None else '-' }</div>")
        html.append(f"<div><b>회전율(%)</b> { _fmt_float(tnp, 2) if tnp is not None else '-' }</div>")
        # 보조 지표
        if rsi is not None: html.append(f"<div><b>RSI</b> {_fmt_float(rsi,2)}</div>")
        if wr  is not None: html.append(f"<div><b>Williams %R</b> {_fmt_float(wr,2)}</div>")
        if gap is not None: html.append(f"<div><b>MA20 Gap</b> {_fmt_float(gap,2)}%</div>")
        html.append("</div>")  # /kv

        if chart_uri:
            html.append(f"<img src='{chart_uri}' alt='chart'/>")

        if reasons:
            html.append("<ul>")
            for t in reasons:
                html.append(f"<li>{t}</li>")
            html.append("</ul>")

        # (선택) 리뷰 요약 박스
        if make_quick_review is not None:
            try:
                tail_df = df.tail(60).copy() if isinstance(df, pd.DataFrame) and not df.empty else None
                review_lines = make_quick_review(r, tail_df, cfg)
                if review_lines:
                    html.append("<div style='margin-top:8px;padding:10px;border:1px solid #e5e7eb;border-radius:10px;background:#fafafa'>")
                    html.append("<div style='font-weight:600;margin-bottom:6px'>리뷰(요약)</div>")
                    html.append("<ul style='margin:0 0 0 18px;padding:0'>")
                    for ln in review_lines:
                        html.append(f"<li style='margin:2px 0'>{ln}</li>")
                    html.append("</ul></div>")
            except Exception as e:
                html.append(f"<div class='small'>[리뷰 생성 오류: {e}]</div>")

        # 뉴스 섹션
        if aux_info and sym in aux_info and aux_info[sym].get("news"):
            html.append("<div style='margin-top:8px;padding:10px;border:1px solid #e5e7eb;border-radius:10px'>")
            html.append("<div style='font-weight:600;margin-bottom:6px'>최근 뉴스</div><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["news"]:
                title = it.get("title", "")
                link  = it.get("link", "#")
                pub   = it.get("published", "")
                html.append(f"<li style='margin:2px 0'><a href='{link}' target='_blank'>{title}</a> "
                            f"<span style='color:#888'>({pub})</span></li>")
            html.append("</ul></div>")

        # 공시 섹션
        if aux_info and sym in aux_info and aux_info[sym].get("filings"):
            html.append("<div style='margin-top:8px;padding:10px;border:1px solid #e5e7eb;border-radius:10px'>")
            html.append("<div style='font-weight:600;margin-bottom:6px'>최근 공시(DART)</div><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["filings"]:
                rpt   = it.get("rpt", "")
