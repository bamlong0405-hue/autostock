# main.py
import os
import sys
import time
import toml
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 프로젝트 모듈 ===
from data_sources import (
    fetch_krx, fetch_yahoo, date_range_for_lookback,
    get_all_krx_tickers, get_krx_name, get_board_sets, attach_turnover_krx
)
from strategy import build_features, generate_signal
from reporter import build_html_report
from mailer import send_mail_html
from backtest import simulate_symbol
from bt_reporter import save_backtest_outputs
from fundamentals import fetch_fundamental_map
from news import fetch_news_headlines
from dart_client import latest_filings
from report_attach import build_buy_attachment

# -----------------------
# 공용 로그(즉시 flush)
# -----------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# -----------------------
# 헬퍼
# -----------------------
def get_name(symbol: str, market: str) -> str:
    if market == "KRX":
        try:
            return get_krx_name(symbol)
        except Exception:
            return symbol
    return symbol

def analyze_symbol(symbol: str, market: str, cfg: dict):
    """개별 심볼 분석: (features, info, ohlc_df) 반환"""
    lb = int(cfg['general']['lookback_days'])
    krx_start, krx_end, yah_start, yah_end = date_range_for_lookback(lb)

    if market == "KRX":
        df = fetch_krx(symbol, krx_start, krx_end)
    else:
        df = fetch_yahoo(symbol, yah_start, yah_end)

    if df is None or df.empty or len(df) < max(60, lb // 2):
        return pd.DataFrame(), {}, df

    feat = build_features(df, cfg)
    feat['Signal'] = generate_signal(feat, cfg)
    latest = feat.dropna().iloc[-1]

    info = {
        "market": market,
        "symbol": symbol,
        "name": get_name(symbol, market),
        "signal": str(latest['Signal']),
        "close": round(float(latest['Close']), 3),
        "rsi": round(float(latest.get('RSI', float('nan'))), 2) if 'RSI' in feat.columns else None,
        "wr": round(float(latest.get('WR', float('nan'))), 2) if 'WR' in feat.columns else None,
    }
    if 'MA_M' in feat.columns and pd.notna(latest.get('MA_M')):
        try:
            info["ma20_gap_pct"] = round((latest['Close'] - latest['MA_M']) / latest['MA_M'] * 100, 2)
        except Exception:
            info["ma20_gap_pct"] = None
    else:
        info["ma20_gap_pct"] = None

    return feat, info, df

def load_universe(cfg: dict):
    uv = cfg.get('universe', {})
    krx_mode = str(uv.get('krx_mode', 'LIST')).upper()
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

def maybe_run_backtest(cfg: dict, feats_map: dict, df_map: dict):
    """선택적 백테스트 실행 (config.backtest.enabled)"""
    bt = cfg.get('backtest', {})
    if not bt.get('enabled', False):
        return None

    lb = int(bt.get('lookback_days', 900))
    m_symbol = bt.get('market_filter_symbol', '069500')
    _, _, _, _ = date_range_for_lookback(lb)

    # 시장 상태(>MA) 시리즈
    m_df = df_map.get(m_symbol)
    if m_df is None or m_df.empty:
        try:
            ks, ke, _, _ = date_range_for_lookback(lb)
            m_df = fetch_krx(m_symbol, ks, ke)
        except Exception:
            m_df = pd.DataFrame()

    market_ok = None
    if m_df is not None and not m_df.empty:
        ma_days = int(bt.get('market_ma_days', 200))
        m_ma = m_df['Close'].rolling(ma_days).mean()
        market_ok = (m_df['Close'] > m_ma)
        market_ok.name = 'market_ok'

    liq_lb = int(bt.get('liquidity_lookback', 20))
    liq_min = float(bt.get('liquidity_min_avg_value', 0.0))
    max_hold = int(bt.get('max_hold_days', 0))
    atr_period = int(bt.get('atr_period', 14))
    stop_mult = float(bt.get('stop_atr_mult', 0.0))
    trail_mult = float(bt.get('trail_atr_mult', 0.0))
    cost_bps = float(bt.get('cost_bps_per_side', 25.0))
    top_n = int(bt.get('summary_top_n', 50))

    results = []
    for sym, feat in feats_map.items():
        df = df_map.get(sym)
        if df is None or df.empty:
            continue
        avg_value = (df['Close'] * df['Volume']).rolling(liq_lb).mean()
        mk = None
        if market_ok is not None:
            mk = market_ok.reindex(df.index).fillna(method='ffill').fillna(False)

        metrics, eq, trades = simulate_symbol(
            signals=feat['Signal'],
            ohlc=df,
            market_ok=mk,
            avg_value=avg_value,
            liquidity_min_value=liq_min,
            max_hold_days=max_hold,
            atr_period=atr_period,
            stop_atr_mult=stop_mult,
            trail_atr_mult=trail_mult,
            cost_bps_per_side=cost_bps
        )
        metrics.update({"market": "KRX", "symbol": sym, "name": get_krx_name(sym)})
        results.append(metrics)

    outdir = cfg['general']['output_dir']
    csv_path, html_path = save_backtest_outputs(results, outdir, top_n=top_n)
    return {"csv": csv_path, "html": html_path, "count": len(results)}

# -----------------------
# 메인
# -----------------------
def main():
    cfg = toml.load("config.toml")
    os.makedirs(cfg['general']['output_dir'], exist_ok=True)
    workers = int(cfg.get('general', {}).get('workers', 8))

    # 1) 유니버스 로드
    krx_codes, yah_codes = load_universe(cfg)
    log(f"Universe: KRX={len(krx_codes)}, YAHOO={len(yah_codes)}")

    # 2) (옵션) 보드/유동성 선필터
    try:
        if cfg.get('filters', {}).get('use_board_filter', True):
            boards = get_board_sets()  # {"KOSPI": set([...]), "KOSDAQ": set([...]) ...}
            allow = set()
            if cfg.get('filters', {}).get('board_allow_kospi', True):
                allow |= boards.get("KOSPI", set())
            if cfg.get('filters', {}).get('board_allow_kosdaq', True):
                allow |= boards.get("KOSDAQ", set())
            before = len(krx_codes)
            krx_codes = [s for s in krx_codes if s in allow] if allow else krx_codes
            log(f"Board filter: {before} → {len(krx_codes)}")
    except Exception as e:
        log(f"[WARN] board filter skipped: {e}")

    try:
        if cfg.get('filters', {}).get('use_liquidity_filter', False):
            liq = attach_turnover_krx(krx_codes)  # 구현: turnover_20d_pct / value_20d_avg 컬럼 기대
            min_turn = float(cfg['filters'].get('min_turnover_20d_pct', 0.3))  # %
            min_value = float(cfg['filters'].get('min_value_20d_avg', 1e9))    # 10억
            keep = set(liq.loc[
                (liq['turnover_20d_pct'] >= min_turn) & (liq['value_20d_avg'] >= min_value),
                'ticker'
            ])
            before = len(krx_codes)
            krx_codes = [s for s in krx_codes if s in keep] if keep else krx_codes
            log(f"Liquidity filter: {before} → {len(krx_codes)}")
    except Exception as e:
        log(f"[WARN] liquidity filter skipped: {e}")

    # (옵션) 개발 중 상한
    max_syms = int(cfg.get('general', {}).get('max_symbols', 0) or 0)
    if max_syms > 0 and len(krx_codes) > max_syms:
        krx_codes = krx_codes[:max_syms]
        log(f"Cap universe to {max_syms} symbols (dev/test)")

    # 3) 분석 실행 (KRX 병렬)
    results = []
    details = {}
    feats_map = {}
    df_map = {}

    log(f"Start KRX scan (workers={workers}): {len(krx_codes)} symbols")
    def _proc_krx(sym):
        try:
            feat, info, df = analyze_symbol(sym, "KRX", cfg)
            return sym, feat, info, df, None
        except Exception as e:
            return sym, None, None, None, e

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_proc_krx, sym) for sym in krx_codes]
        for fut in as_completed(futs):
            sym, feat, info, df, err = fut.result()
            done += 1
            if err:
                if done % 50 == 0:
                    log(f"[KRX] error {sym}: {err} (progress {done}/{len(krx_codes)})")
                continue
            if feat is not None and not feat.empty:
                results.append(info)
                save_cols = [c for c in [
                    'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                    'Bullish','Bearish','Bull_Engulf','BB_POS_PCT'
                ] if c in feat.columns]
                details[sym] = feat[save_cols].copy()
                feats_map[sym] = feat
                df_map[sym] = df
            if done % 200 == 0:
                log(f"[KRX] processed {done}/{len(krx_codes)}")
    log("KRX scan done.")

    # 4) (선택) 야후 심볼
    if yah_codes:
        log(f"Start Yahoo scan: {len(yah_codes)} symbols")
        for sym in yah_codes:
            try:
                feat, info, df = analyze_symbol(sym, "YAHOO", cfg)
                if not feat.empty:
                    results.append(info)
                    save_cols = [c for c in [
                        'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                        'Bullish','Bearish','Bull_Engulf','BB_POS_PCT'
                    ] if c in feat.columns]
                    details[sym] = feat[save_cols].copy()
                    feats_map[sym] = feat
                    df_map[sym] = df
            except Exception as e:
                log(f"[YAHOO] error {sym}: {e}")
        log("Yahoo scan done.")

    # 5) 펀더멘털/뉴스/공시 수집 (BUY 위주)
    # 펀더멘털(PER/PBR/DIV)
    try:
        krx_tickers = [r["symbol"] for r in results if r.get("market") == "KRX"]
        funda_map = fetch_fundamental_map(krx_tickers) if krx_tickers else {}
    except Exception as e:
        log(f"[WARN] fundamentals fetch skipped: {e}")
        funda_map = {}

    for r in results:
        if r.get("market") == "KRX" and r.get("symbol") in funda_map:
            f = funda_map[r["symbol"]]
            r["per"] = f.get("PER")
            r["pbr"] = f.get("PBR")
            r["div"] = f.get("DIV")

    # 최신 뉴스/공시 (메일 용량 고려: BUY만)
    email_opts = cfg.get("email_options", {})
    max_news = int(email_opts.get("max_news_per_symbol", 3))
    max_filings = int(email_opts.get("max_filings_per_symbol", 3))
    aux_info = {}  # {sym: {"news":[...], "filings":[...]}}

    buy_rows_all = [r for r in results if str(r.get("signal")).upper() == "BUY"]
    if buy_rows_all:
        for r in buy_rows_all:
            sym = r["symbol"]
            name = r.get("name") or sym
            try:
                news_items = fetch_news_headlines(name, max_items=max_news, lang="ko")
            except Exception as e:
                news_items = []
                log(f"[WARN] news fetch fail {name}: {e}")
            try:
                filings = latest_filings(name, max_items=max_filings)
            except Exception as e:
                filings = []
                log(f"[WARN] dart fetch fail {name}: {e}")

            if news_items or filings:
                aux_info[sym] = {"news": news_items, "filings": filings}

    # 6) 리포트 생성
    if not results:
        html = "<html><body><h1>No data</h1></body></html>"
        all_html = html
    else:
        # 전체 리포트(파일/아티팩트용)
        all_html = build_html_report(
            results, details, cfg,
            show_charts=True,
            max_charts=50,
            max_table_rows=100000
        )

        # 메일용: BUY/SELL만, 차트 제외(용량↓)
        email_only = bool(email_opts.get("email_only_signals", True))
        include_charts = bool(email_opts.get("include_charts", False))
        max_email_rows = int(email_opts.get("max_email_rows", 300))
        max_email_charts = int(email_opts.get("max_email_charts", 0))

        if email_only:
            filtered = [r for r in results if str(r.get("signal")).upper() in ("BUY","SELL")]
            if not filtered:
                html = "<html><body><h1>No BUY/SELL signals today</h1></body></html>"
            else:
                symbols = [r["symbol"] for r in filtered]
                if include_charts and max_email_charts > 0:
                    symbols = symbols[:max_email_charts]
                    filtered_details = {s: details[s] for s in symbols if s in details}
                else:
                    filtered_details = {}
                html = build_html_report(
                    filtered, filtered_details, cfg,
                    show_charts=include_charts and max_email_charts > 0,
                    max_charts=max_email_charts,
                    max_table_rows=max_email_rows
                )
        else:
            html = build_html_report(
                results, {}, cfg,
                show_charts=False,
                max_table_rows=max_email_rows
            )

    # 7) 파일 저장(전체 리포트)
    save_all = bool(cfg.get("email_options", {}).get("include_hold_in_report", True))
    report_path = os.path.join(cfg['general']['output_dir'], cfg['general']['report_filename'])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(all_html if save_all else html)
    log(f"Report written: {report_path}")

    # 8) 첨부 리포트(소형 차트+근거+뉴스/공시/리뷰)
    attachment_paths = []
    if bool(email_opts.get("attach_buy_report", True)) and buy_rows_all:
        try:
            attach_path = build_buy_attachment(
                buy_rows=buy_rows_all,
                details=details,
                cfg=cfg,
                out_path=f"{cfg['general']['output_dir'].rstrip('/')}/buy_report.html",
                max_charts=int(email_opts.get("max_buy_charts", 10)),
                aux_info=aux_info
            )
            attachment_paths.append(attach_path)
            log(f"Attachment created: {attach_path}")
        except Exception as e:
            log(f"[WARN] build_buy_attachment failed: {e}")

    # 9) 메일 전송 (용량 폴백 포함)
    smtp_user = cfg['email']['from_addr']
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        log("ERROR: GMAIL_APP_PASSWORD env not set")
        sys.exit(1)

    try:
        send_mail_html(
            smtp_user=smtp_user,
            app_password=app_password,
            to_addrs=cfg['email']['to_addrs'],
            subject=cfg['email']['subject'],
            html=html,
            from_name=cfg['email']['from_name'],
            attachments=attachment_paths if attachment_paths else None
        )
        log("Email sent.")
    except Exception as e:
        log(f"[WARN] email send failed: {e}")
        # 552 등 용량 초과 폴백: 표만 150행
        fallback_html = build_html_report(
            [r for r in results if str(r.get("signal")).upper() in ("BUY","SELL")],
            {},
            cfg,
            show_charts=False,
            max_table_rows=150
        )
        try:
            send_mail_html(
                smtp_user=smtp_user,
                app_password=app_password,
                to_addrs=cfg['email']['to_addrs'],
                subject=cfg['email']['subject'] + " (fallback)",
                html=fallback_html,
                from_name=cfg['email']['from_name'],
                attachments=None
            )
            log("Sent fallback email (reduced size).")
        except Exception as e2:
            log(f"Email failed even after fallback: {e2}")
            sys.exit(1)

    # 10) 요약 & (옵션) 백테스트
    total = len(results)
    buys  = sum(1 for r in results if str(r.get("signal")).upper() == "BUY")
    sells = sum(1 for r in results if str(r.get("signal")).upper() == "SELL")
    holds = total - buys - sells
    log(f"[SUMMARY] total={total}, BUY={buys}, SELL={sells}, HOLD={holds}")

    bt_out = maybe_run_backtest(cfg, feats_map, df_map)
    if bt_out:
        log(f"Backtest: {bt_out['count']} symbols → {bt_out['csv']}, {bt_out['html']}")

    log(f"Done at {datetime.now()}")

if __name__ == "__main__":
    main()
