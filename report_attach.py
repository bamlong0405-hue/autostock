# report_attach.py
import io
import base64
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

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

def _small_price_chart(df: pd.DataFrame, name: str):
    # 최근 60봉 종가/MA20
    d = df.tail(60)
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.plot(d.index, d["Close"], label="Close")
    if "MA_M" in d.columns:
        ax.plot(d.index, d["MA_M"], label="MA20")
    ax.set_title(name)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    return _fig_to_base64(fig)

def build_buy_attachment(
    buy_rows: List[dict],
    details: Dict[str, pd.DataFrame],
    cfg: dict,
    out_path: str = "output/buy_report.html",
    max_charts: int = 10,
) -> str:
    """
    buy_rows: main.py에서 만든 results 중 signal == BUY 만 추린 dict 리스트
              예) {"market": "KRX", "symbol": "000000", "name": "...", "rsi":..., "wr":..., "ma20_gap_pct":...}
    details : 심볼 -> 피처 DataFrame (Close/MA_M/RSI/WR/OBV_slope 포함 권장)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = [HTML_HEADER, f"<h1>BUY Report</h1><div class='meta'>Generated at {ts}</div>"]

    take = buy_rows[:max_charts]
    for r in take:
        sym = r["symbol"]
        name = r.get("name", sym)
        df = details.get(sym)

        # 차트
        chart_uri = ""
        if isinstance(df, pd.DataFrame) and not df.empty:
            chart_uri = _small_price_chart(df, f"{name} ({sym})")

        # 추천 근거 텍스트
        rsi = r.get("rsi", None)
        wr  = r.get("wr", None)
        gap = r.get("ma20_gap_pct", None)

        reasons = []
        # 블로그 모드일 수도, 기본 모드일 수도 있으므로 공통 근거 문구
        if rsi is not None:
            reasons.append(f"RSI {float(rsi):.2f} — 과매도권에서 되돌림/반등 구간 여부 체크")
        if wr is not None:
            reasons.append(f"Williams %R {float(wr):.2f} — 저점권(-80 이하시) 근접이면 반등 모멘텀")
        if gap is not None:
            reasons.append(f"20일선 괴리 {float(gap):.2f}% — 과열/과매도 판단 근거")

        # details df에서 추가 근거 추출 (있을 때만)
        if isinstance(df, pd.DataFrame) and not df.empty:
            last = df.iloc[-1]
            if "OBV_slope" in df.columns and pd.notna(last.get("OBV_slope", None)):
                reasons.append(f"OBV 추세(slope) {float(last['OBV_slope']):.5f} — 수급 흐름 참고")
            if "Bull_Engulf" in df.columns and bool(last.get("Bull_Engulf", False)):
                reasons.append("상승 장악형 캔들 포착")
            if "Bullish" in df.columns and bool(last.get("Bullish", False)):
                reasons.append("양봉/강세 캔들 신호")

        html.append("<div class='card'>")
        html.append(f"<h2>{name} <span style='color:#888;font-weight:normal;'>({r['market']}:{sym})</span></h2>")
        html.append("<div class='kv'>")
        if rsi is not None: html.append(f"<div><b>RSI</b> {float(rsi):.2f}</div>")
        if wr  is not None: html.append(f"<div><b>Williams %R</b> {float(wr):.2f}</div>")
        if gap is not None: html.append(f"<div><b>MA20 Gap</b> {float(gap):.2f}%</div>")
        html.append("</div>")

        if chart_uri:
            html.append(f"<img src='{chart_uri}' alt='chart'/>")

        if reasons:
            html.append("<ul>")
            for t in reasons:
                html.append(f"<li>{t}</li>")
            html.append("</ul>")

        html.append("</div>")  # card

    html.append(HTML_FOOTER)
    html_str = "".join(html)

    # 저장
    import os
    os.makedirs(cfg["general"]["output_dir"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path
