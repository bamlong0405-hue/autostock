import os
import pandas as pd
from reporter import build_html_report, dataframe_to_html_table

def save_backtest_outputs(results: list[dict], outdir: str, top_n: int = 50):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results).sort_values(["CAGR"], ascending=False)
    csv_path = os.path.join(outdir, "backtest_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    cols = ["market","symbol","name","CAGR","MDD","Sharpe","Trades","LastEquity"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    html_table = dataframe_to_html_table(df[cols].head(top_n), max_rows=top_n)
    html = f"""
    <html><head><meta charset="utf-8"><style>
      body {{ font-family: Arial, sans-serif; }}
      .table {{ border-collapse: collapse; width: 100%; }}
      .table td, .table th {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
    </style></head><body>
      <h1>Backtest Summary (Top {top_n})</h1>
      {html_table}
    </body></html>
    """
    html_path = os.path.join(outdir, "backtest_summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return csv_path, html_path
