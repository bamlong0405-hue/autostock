import os
import sys
import toml
import pandas as pd
from datetime import datetime
from data_sources import (
    fetch_krx, fetch_yahoo, date_range_for_lookback,
    get_all_krx_tickers, get_krx_name
)
from strategy import build_features, generate_signal
from reporter import build_html_report
from mailer import send_mail_html

def get_name(symbol: str, market: str) -> str:
    if market == "KRX":
        return get_krx_name(symbol)
    return symbol

def analyze_symbol(symbol: str, market: str, cfg: dict):
    lb = cfg['general']['lookback_days']
    krx_start, krx_end, yah_start, yah_end = date_range_for_lookback(lb)

    if market == "KRX":
        df = fetch_krx(symbol, krx_start, krx_end)
    else:
        df = fetch_yahoo(symbol, yah_start, yah_end)

    if df.empty or len(df) < max(60, lb // 2):
        return pd.DataFrame(), {}

    feat = build_features(df, cfg)
    feat['Signal'] = generate_signal(feat, cfg)
    latest = feat.dropna().iloc[-1]

    info = {
        "market": market,
        "symbol": symbol,
        "name": get_name(symbol, market),
        "signal": latest['Signal'],
        "close": round(float(latest['Close']), 3),
        "rsi": round(float(latest['RSI']), 2),
        "wr": round(float(latest['WR']), 2),
        "ma20_gap_pct": round((latest['Close'] - latest['MA_M']) / latest['MA_M'] * 100, 2) if latest['MA_M'] else None
    }
    return feat, info

def load_universe(cfg: dict):
    uv = cfg.get('universe', {})
    krx_mode = uv.get('krx_mode', 'LIST').upper()

    if krx_mode == "ALL":
        krx_codes = get_all_krx_tickers(
            exclude_etf_etn_spac=uv.get('exclude_etf_etn_spac', True),
            name_filters_exclude=uv.get('name_filters_exclude', []),
            limit=int(uv.get('max_symbols', 0) or 0),
        )
    else:
        krx_codes = uv.get('krx', [])

    yah = uv.get('yahoo', [])
    return krx_codes, yah

def main():
    cfg = toml.load("config.toml")
    os.makedirs(cfg['general']['output_dir'], exist_ok=True)

    krx_codes, yah_codes = load_universe(cfg)

    results = []
    details = {}

    for i, sym in enumerate(krx_codes, 1):
        feat, info = analyze_symbol(sym, "KRX", cfg)
        if not feat.empty:
            results.append(info)
            details[sym] = feat[['Close','MA_M']].copy()
        if i % 200 == 0:
            print(f"[KRX] processed {i}/{len(krx_codes)}")

    for sym in yah_codes:
        feat, info = analyze_symbol(sym, "YAHOO", cfg)
        if not feat.empty:
            results.append(info)
            details[sym] = feat[['Close','MA_M']].copy()

    if not results:
        html = "<html><body><h1>No data</h1></body></html>"
        all_html = html
    else:
        # ① 전체 리포트 (모든 종목, HOLD 포함)
        all_html = build_html_report(results, details, cfg)

        # ② 메일용 리포트 (BUY/SELL만)
        email_only = cfg.get("email_options", {}).get("email_only_signals", True)
        max_email_charts = int(cfg.get("email_options", {}).get("max_email_charts", 80))

        if email_only:
            filtered = [r for r in results if str(r.get("signal")) in ("BUY","SELL")]
            filtered_symbols = [r["symbol"] for r in filtered][:max_email_charts]
            filtered_details = {sym: details[sym] for sym in filtered_symbols if sym in details}

            if not filtered:
                html = "<html><body><h1>No BUY/SELL signals today</h1></body></html>"
            else:
                html = build_html_report(filtered, filtered_details, cfg)
        else:
            html = all_html

    # 파일 저장: 전체 리포트 (HOLD 포함 여부 옵션)
    save_all = cfg.get("email_options", {}).get("include_hold_in_report", True)
    report_path = os.path.join(cfg['general']['output_dir'], cfg['general']['report_filename'])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(all_html if save_all else html)

    smtp_user = cfg['email']['from_addr']
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("ERROR: GMAIL_APP_PASSWORD env not set", file=sys.stderr)
        sys.exit(1)
    send_mail_html(
        smtp_user=smtp_user,
        app_password=app_password,
        to_addrs=cfg['email']['to_addrs'],
        subject=cfg['email']['subject'],
        html=html,
        from_name=cfg['email']['from_name']
    )

    print(f"Report written: {report_path} ({datetime.now()})")

if __name__ == "__main__":
    main()
