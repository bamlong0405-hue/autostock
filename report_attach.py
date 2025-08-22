# report_attach.py
import io
import os
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt

# ----- (선택) 리뷰 생성기: 없으면 더미로 대체 -----
try:
    from reviews import make_quick_review
except Exception:
    def make_quick_review(row: dict, tail_df: Optional[pd.DataFrame], cfg: dict) -> List[str]:
        # 최소한의 대체 요약
        out = []
        if row.get("rsi") is not None:
            out.append(f"RSI {float(row['rsi']):.1f}")
        if row.get("wr") is not None:
            out.append(f"WR {float(row['wr']):.1f}")
        if row.get("ma20_gap_pct") is not None:
            out.append(f"MA20 Gap {float(row['ma20_gap_pct']):.2f}%")
        return out[:3]

HTML_HEADER = """<!doctype html><html><head><meta charset="utf-8">
<title>BUY Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Pretendard, Arial, sans-serif; margin: 24px; color:#111;}
h1 { font-size: 20px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 6px 0 10px; }
.card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px 16px; margin: 14px 0; }
.meta { color: #6b7280; font-size: 12px; margin-bottom: 8px; }
.kv { font-size: 13px; display:flex; flex-wrap:wrap; gap:10px;}
.kv div b { display: inline-block; min-width: 110px; color:#374151;}
img { display:block; max-width: 100%; height: auto; border: 1px solid #f3f4f6; border-radius: 10px; }
ul { margin: 8px 0 0 18px; }
.badge { display:inline-block; font-size:12px; padding:2px 8px; border:1px solid #e5e7eb; border-radius:999px; color:#374151; background:#f9fafb; margin-left:6px;}
.section { margin-top:8px; padding:10px; border:1px solid #e5e7eb; border-radius:10px; background:#fafafa;}
.section .title { font-weight:600; margin-bottom:6px; }
.small { color:#6b7280; font-size:12px; }
</style>
</head><body>
"""

HTML_FOOTER = "</body></html>"

# ---------- helpers ----------
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _fmt(v, nd=2, none_text="-"):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return none_text
        f = float(v)
        if nd == 0:
            return f"{int(round(f))}"
        return f"{f:.{nd}f}"
    except Exception:
        return none_text

def _small_price_chart(df: pd.DataFrame, name: str, show_bb: bool = True) -> str:
    """
    최근 60봉 Close/MA20(+BB 선택) 간단 라인차트
    """
    d = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(d.index, d["Close"], label="Close")
    if "MA_M" in d.columns:
        ax.plot(d.index, d["MA_M"], label="MA20")
    if show_bb and ("BB_UPPER" in d.columns and "BB_LOWER" in d.columns and "BB_MID" in d.columns):
        # BB 대역을 얇은 라인
