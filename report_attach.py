# report_attach.py
import io
import os
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt

# 선택: 리뷰 함수가 없으면 무시되도록 안전 처리
try:
    from reviews import make_quick_review  # 리뷰 요약 (선택적 사용)
except Exception:
    def make_quick_review(*args, **kwargs):
        return []

HTML_HEADER = """<!doctype html><html><head><meta charset="utf-8">
<title>BUY Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Pretendard, Arial, sans-serif; margin: 24px; }
h1 { font-size: 20px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 24px 0 8px; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
.meta { color: #666; font-size: 12px; margin-bottom: 8px; }
.kv { font-size: 13px; line-height: 1.4; }
.kv b { display: inline-block; width: 130px; color: #333; }
img { display:block; max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; margin-top: 6px; }
ul { margin: 6px 0 0 18px; }
.small { color:#666; font-size:12px; }
.section { margin-top:8px; padding:10px; border:1px solid #e5e7eb; border-radius:10px; }
.section h3 { margin:0 0 6px; font-size:14px; }
</style>
</head><body>
"""

HTML_FOOTER = "</body></html>"

def _fig_to_base64(fig, dpi: int = 72) -> str:
    """작은 DPI로 저장해 용량 최소화"""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _small_price_chart(df: pd.DataFrame, name: str) -> str:
    """최근 60봉 종가/MA20(+선택 BB대역)을 매우 작은 이미지로"""
    d = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(4.5, 2.2))  # 소형
    ax.plot(d.index, d.get("Close"), label="Close", linewidth=0.9)
    if "MA_M" in d.columns:
        ax.plot(d.index, d["MA_M"], label="MA20", linewidth=0.9)

    # BB 대역을 얇은 라인으로 (있을 때만)
    has_bb = {"BB_UPPER", "BB_LOWER"}.issubset(set(d.columns))
    if has_bb:
        try:
            ax.plot(d.index, d["BB_UPPER"], linewidth=0.6, label="BB Upper")
            ax.plot(d.index, d["BB_LOWER"], linewidth=0.6, label="BB Lower")
        except Exception:
            pass

    ax.set_title(name, fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.4)
    ax.legend(fontsize=8)
    return _fig_to_base64(fig, dpi=72)

def _fmt2(x, suffix=""):
    try:
        return f"{float(x):,.2f}{suffix}"
    except Exception:
        return "-"

def _fmt0(x, suffix=""):
    try:
        return f"{int(x):,}{suffix}"
    except Exception:
        return "-"

def build_buy_attachment(
    buy_rows: List[dict],
    details: Dict[str, pd.DataFrame],
    cfg: dict,
    out_path: str = "output/buy_report.html",
    max_charts: int = 10,
    aux_info: Optional[Dict[str, Any]] = None,   # {sym: {"news":[...], "filings":[...]}}
    embed_charts: bool = False,                  # 첨부 용량 감소를 위해 기본 False
    turnover_map: Optional[Dict[str, Dict[str, Any]]] = None,  # {sym: {"turnover_pct":..., "board":...}}
) -> str:
    """
    buy_rows : signal == BUY 인 종목 dict 리스트 (results 원소)
    details  : 심볼 -> 피처 DataFrame (Close/Volume/MA_M/OBV_slope/BB_* 포함 권장)
    aux_info : 심볼 -> {"news":[{title,link,published}], "filings":[{rpt,link,rcpdt}]}
    turnover_map: 심볼 -> {"turnover_pct": float, "board": "KOSPI"/"KOSDAQ"}
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
        if embed_charts and isinstance(df, pd.DataFrame) and not df.empty:
            chart_uri = _small_price_chart(df, f"{name} ({sym})")

        # 기본 지표(요약)
        rsi = r.get("rsi", None)
        wr  = r.get("wr", None)
        gap = r.get("ma20_gap_pct", None)

        # 종가 / 전일 거래량 / 회전율
        close_val = r.get("close", None)
        prev_vol = None
        if isinstance(df, pd.DataFrame) and len(df) >= 2 and "Volume" in df.columns:
            prev_vol = df["Volume"].shift(1).iloc[-1]

        turn_pct = None
        board    = None
        if turnover_map and sym in turnover_map:
            turn_pct = turnover_map[sym].get("turnover_pct")
            board    = turnover_map[sym].get("board")

        # 추천 근거(기술)
        reasons: List[str] = []
        if rsi is not None:
            reasons.append(f"RSI {float(rsi):.2f} — 과매도/반등 가능성 체크")
        if wr is not None:
            reasons.append(f"Williams %R {float(wr):.2f} — 저점권(-80↓) 근접 시 반등 모멘텀")
        if gap is not None:
            reasons.append(f"MA20 괴리 {float(gap):.2f}% — 과열/과매도 판단 참고")

        if isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            if "OBV_slope" in df.columns and pd.notna(last.get("OBV_slope", None)):
                reasons.append(f"OBV 추세(slope) {float(last['OBV_slope']):.5f} — 수급 흐름 참고")
            if "Bull_Engulf" in df.columns and bool(last.get("Bull_Engulf", False)):
                reasons.append("상승 장악형 캔들 포착")
            if "Bullish" in df.columns and bool(last.get("Bullish", False)):
                reasons.append("양봉/강세 캔들 신호")
            if "BB_POS_PCT" in df.columns and pd.notna(last.get("BB_POS_PCT", None)):
                reasons.append(f"볼린저밴드 위치 {float(last['BB_POS_PCT']):.1f}% — 0% 하단, 100% 상단")

        # 카드
        html.append("<div class='card'>")
        html.append(f"<h2>{name} <span class='small'>({r['market']}:{sym}{' · '+board if board else ''})</span></h2>")

        # Key-Value
        html.append("<div class='kv'>")
        if close_val is not None: html.append(f"<div><b>종가</b> {_fmt2(close_val)}</div>")
        if prev_vol is not None:  html.append(f"<div><b>전일 거래량</b> {_fmt0(prev_vol)}</div>")
        if (turn_pct is not None): html.append(f"<div><b>회전율</b> {_fmt2(turn_pct, '%')}</div>")
        if rsi is not None:       html.append(f"<div><b>RSI</b> {_fmt2(rsi)}</div>")
        if wr  is not None:       html.append(f"<div><b>Williams %R</b> {_fmt2(wr)}</div>")
        if gap is not None:       html.append(f"<div><b>MA20 Gap</b> {_fmt2(gap, '%')}</div>")
        html.append("</div>")

        if chart_uri:
            html.append(f"<img src='{chart_uri}' alt='chart'/>")

        if reasons:
            html.append("<ul>")
            for t in reasons:
                html.append(f"<li>{t}</li>")
            html.append("</ul>")

        # (선택) 리뷰 요약 박스
        try:
            tail_df = df.tail(60).copy() if isinstance(df, pd.DataFrame) and not df.empty else None
            review_lines = make_quick_review(r, tail_df, cfg)
            if review_lines:
                html.append("<div class='section'><h3>리뷰(요약)</h3><ul style='margin:0 0 0 18px;padding:0'>")
                for ln in review_lines:
                    html.append(f"<li style='margin:2px 0'>{ln}</li>")
                html.append("</ul></div>")
        except Exception as e:
            html.append(f"<div class='small'>[리뷰 생성 오류: {e}]</div>")

        # 뉴스
        if aux_info and sym in aux_info and aux_info[sym].get("news"):
            html.append("<div class='section'><h3>최근 뉴스</h3><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["news"]:
                title = it.get("title", "")
                link  = it.get("link", "#")
                pub   = it.get("published", "")
                html.append(f"<li style='margin:2px 0'><a href='{link}' target='_blank'>{title}</a> "
                            f"<span class='small'>({pub})</span></li>")
            html.append("</ul></div>")

        # 공시(DART)
        if aux_info and sym in aux_info and aux_info[sym].get("filings"):
            html.append("<div class='section'><h3>최근 공시(DART)</h3><ul style='margin:0 0 0 18px;padding:0'>")
            for it in aux_info[sym]["filings"]:
                rpt   = it.get("rpt", "")
                link  = it.get("link", "#")
                rcpdt = it.get("rcpdt", "")
                html.append(f"<li style='margin:2px 0'><a href='{link}' target='_blank'>{rpt}</a> "
                            f"<span class='small'>({rcpdt})</span></li>")
            html.append("</ul></div>")

        html.append("</div>")  # /card

    html.append(HTML_FOOTER)
    html_str = "".join(html)

    os.makedirs(cfg["general"]["output_dir"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path
