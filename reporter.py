import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 1000) -> str:
    return df.tail(max_rows).to_html(classes="table table-striped", border=0)

def build_html_report(summary_rows, details, cfg: dict) -> str:
    summary_df = pd.DataFrame(summary_rows).sort_values(["signal","symbol"])
    summary_html = dataframe_to_html_table(summary_df[['market','symbol','name','signal','close','rsi','wr','ma20_gap_pct']])

    chart_html_parts = []
    for sym, df in details.items():
        buf = io.BytesIO()
        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, title=f"{sym} Close (with MA20)")
        df['MA_M'].plot(ax=ax)
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart_html_parts.append(f"<h3>{sym}</h3><img src='data:image/png;base64,{b64}' style='max-width:800px;'>")

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          .table {{ border-collapse: collapse; }}
          .table td, .table th {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
          h1 {{ margin-bottom: 0; }}
          .muted {{ color: #666; }}
        </style>
      </head>
      <body>
        <h1>Daily Signals</h1>
        <div class="muted">{cfg['general']['timezone']}</div>
        <h2>Summary</h2>
        {summary_html}
        <h2>Charts</h2>
        {''.join(chart_html_parts)}
      </body>
    </html>
    """
    return html
