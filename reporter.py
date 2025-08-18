import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 1000) -> str:
    if max_rows and max_rows > 0:
        df = df.head(max_rows)
    return df.to_html(classes="table table-striped", border=0, index=False)

def _charts_html(details: dict[str, pd.DataFrame], limit: int = 80) -> str:
    parts = []
    count = 0
    for sym, df in details.items():
        if limit and count >= limit:
            break
        buf = io.BytesIO()
        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, title=f"{sym} Close (with MA20)")
        if 'MA_M' in df:
            df['MA_M'].plot(ax=ax)
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        parts.append(f"<h3>{sym}</h3><img src='data:image/png;base64,{b64}' style='max-width:800px;'>")
        count += 1
    return ''.join(parts)

def build_html_report(
    summary_rows: list[dict],
    details: dict[str, pd.DataFrame],
    cfg: dict,
    *,
    show_charts: bool = True,
    max_charts: int = 80,
    max_table_rows: int = 1000
) -> str:
    cols = ['market','symbol','name','signal','close','rsi','wr','ma20_gap_pct']
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        # 안전 정렬
        for c in ['market','symbol','signal']:
            if c not in summary_df.columns: summary_df[c] = ""
        summary_df = summary_df[cols]
        summary_df = summary_df.sort_values(["signal","symbol"])
    summary_html = dataframe_to_html_table(summary_df, max_rows=max_table_rows)

    charts_html = _charts_html(details, limit=max_charts) if show_charts else ""

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          .table {{ border-collapse: collapse; width: 100%; }}
          .table td, .table th {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
          h1 {{ margin-bottom: 0; }}
          .muted {{ color: #666; }}
        </style>
      </head>
      <body>
        <h1>Daily Signals</h1>
        <div class="muted">{cfg['general'].get('timezone','')}</div>
        <h2>Summary</h2>
        {summary_html}
        {"<h2>Charts</h2>" + charts_html if show_charts else ""}
      </body>
    </html>
    """
    return html
