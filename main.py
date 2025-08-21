import os
import sys
import toml
import pandas as pd
from datetime import datetime

from data_sources import (
    fetch_krx, fetch_yahoo, date_range_for_lookback,
    get_all_krx_tickers, get_krx_name,
    get_board_sets, attach_turnover_krx,   # NEW
)
from strategy import build_features, generate_signal
from reporter import build_html_report
from mailer import send_mail_html

# (선택) 리뷰/펀더멘탈/뉴스/공시 확장 사용 중이면 임포트 유지
try:
    from fundamentals import fetch_fundamental_map
except Exception:
    fetch_fundamental_map = None
try:
    from news import fetch_news_headlines
except Exception:
    fetch_news_headlines = None
try:
    from dart_client import latest_filings
except Exception:
    latest_filings = None

# -----------------------------------------------------------------------------
def get_name(symbol: str, market: str) -> str:
    return get_krx_name(symbol) if market == "KRX" else symbol

def _build_regime_series(cfg: dict) -> pd.Series | None:
    try:
        m_sym = cfg.get("mode", {}).get("regime_symbol", "069500")
        lb = int(cfg["general"]["lookback_days"]) + int(cfg.get("mode", {}).get("regime_ma_days", 200)) + 10
        ks, ke, _, _ = date_range_for_lookback(lb)
        mdf = fetch_krx(m_sym, ks, ke)
        if mdf is None or mdf.empty:
            return None
        n = int(cfg.get("mode", {}).get("regime_ma_days", 200))
        ma = mdf["Close"].rolling(n).mean()
        regime_bull = (mdf["Close"] > ma)
        regime_bull.name = "REGIME_BULL"
        return regime_bull
    except Exception:
        return None

def analyze_symbol(symbol: str,
                   market: str,
                   cfg: dict,
                   regime_bull: pd.Series | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    lb = cfg['general']['lookback_days']
    krx_start, krx_end, yah_start, yah_end = date_range_for_lookback(lb)

    if market == "KRX":
        df = fetch_krx(symbol, krx_start, krx_end)
    else:
        df = fetch_yahoo(symbol, yah_start, yah_end)

    if df is None or df.empty or len(df) < max(60, lb // 2):
        return pd.DataFrame(), {}, (df if isinstance(df, pd.DataFrame) else pd.DataFrame())

    feat = build_features(df, cfg)

    # 레짐 오버레이: 피처에 붙여 두면 디버깅도 쉬움
    if isinstance(regime_bull, pd.Series) and not regime_bull.empty:
        feat["REGIME_BULL"] = regime_bull.reindex(feat.index).ffill().fillna(False)

    feat['Signal'] = generate_signal(feat, cfg)
    feat = feat.dropna(subset=["Close"])  # 안전

    if feat.empty:
        return pd.DataFrame(), {}, df

    latest = feat.iloc[-1]
    cur_signal = str(latest['Signal'])

    # 약세장(레짐 OFF)일 때 BUY 차단 옵션
    if cfg.get("mode", {}).get("regime_disable_buy_in_bear", True):
        is_bull = bool(latest.get("REGIME_BULL", True))
        if cur_signal == "BUY" and not is_bull:
            cur_signal = "HOLD"

    # 최근 BUY 이후 N일 쿨다운
    cooldown = int(cfg.get("signals", {}).get("cooldown_days", 0) or 0)
    if cooldown > 0 and cur_signal == "BUY":
        buy_idx = feat.index[feat['Signal'].astype(str).str.upper() == "BUY"]
        if len(buy_idx) >= 2:
            # 직전 BUY와의 일수 차이
            prev = buy_idx[-2]
            days = (feat.index[-1] - prev).days if hasattr(prev, "to_pydatetime") or hasattr(prev, "tzinfo") else cooldown
            if days < cooldown:
                cur_signal = "HOLD"

    info = {
        "market": market,
        "symbol": symbol,
        "name": get_name(symbol, market),
        "signal": cur_signal,
        "close": round(float(latest['Close']), 3),
        "rsi": round(float(latest['RSI']), 2) if pd.notna(latest.get("RSI", pd.NA)) else None,
        "wr":  round(float(latest['WR']), 2)  if pd.notna(latest.get("WR", pd.NA))  else None,
        "ma20_gap_pct": (
            round((latest['Close'] - latest['MA_M']) / latest['MA_M'] * 100, 2)
            if pd.notna(latest.get("MA_M", pd.NA)) and latest['MA_M'] else None
        ),
        # 디버깅/랭킹용
        "obv_slope": float(latest['OBV_slope']) if pd.notna(latest.get("OBV_slope", pd.NA)) else None,
        "regime_bull": bool(latest.get("REGIME_BULL", True)),
    }
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

# -----------------------------------------------------------------------------
def main():
    cfg = toml.load("config.toml")
    os.makedirs(cfg['general']['output_dir'], exist_ok=True)

    # 심볼 유니버스
    krx_codes, yah_codes = load_universe(cfg)

    # 레짐 시리즈
    regime_series = _build_regime_series(cfg)

    results: list[dict] = []
    details: dict[str, pd.DataFrame] = {}
    feats_map: dict[str, pd.DataFrame] = {}
    df_map: dict[str, pd.DataFrame] = {}

    # --- KRX ---
    for i, sym in enumerate(krx_codes, 1):
        feat, info, df = analyze_symbol(sym, "KRX", cfg, regime_bull=regime_series)
        if not feat.empty and info:
            results.append(info)
            keep_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','REGIME_BULL','BB_POS_PCT'
            ] if c in feat.columns]
            details[sym]  = feat[keep_cols].copy()
            feats_map[sym] = feat
            df_map[sym]    = df
        if i % 200 == 0:
            print(f"[KRX] processed {i}/{len(krx_codes)}")

    # --- Yahoo (옵션) ---
    for sym in yah_codes:
        feat, info, df = analyze_symbol(sym, "YAHOO", cfg, regime_bull=regime_series)
        if not feat.empty and info:
            results.append(info)
            keep_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','REGIME_BULL','BB_POS_PCT'
            ] if c in feat.columns]
            details[sym]  = feat[keep_cols].copy()
            feats_map[sym] = feat
            df_map[sym]    = df

    # --- 펀더멘털 (선택) ---
    funda_map = {}
    if fetch_fundamental_map is not None:
        try:
            krx_tickers = [r["symbol"] for r in results if r.get("market") == "KRX"]
            funda_map = fetch_fundamental_map(krx_tickers) or {}
            # info 확장
            for r in results:
                t = r.get("symbol")
                if r.get("market") == "KRX" and t in funda_map:
                    r.update({
                        "per": funda_map[t].get("PER"),
                        "pbr": funda_map[t].get("PBR"),
                        "div": funda_map[t].get("DIV"),
                    })
        except Exception as e:
            print(f"[WARN] fundamentals failed: {e}")

    # --- 회전율(시총 대비 거래대금 %) 부착 ---
    turn_info = attach_turnover_krx([r["symbol"] for r in results if r.get("market")=="KRX"])
    for r in results:
        if r.get("market") != "KRX":
            continue
        t = r["symbol"]
        ti = turn_info.get(t, {})
        r["turnover_pct"] = ti.get("turnover_pct")
        r["board"] = ti.get("board")

    # ---------- 리포트 생성 (전체) ----------
    if not results:
        html = "<html><body><h1>No data</h1></body></html>"
        all_html = html
    else:
        all_html = build_html_report(
            results, details, cfg,
            show_charts=True,
            max_charts=50,
            max_table_rows=100000
        )

        # ---------- 메일용 요약 ----------
        email_opts = cfg.get("email_options", {})
        email_only = bool(email_opts.get("email_only_signals", True))
        max_email_charts = int(email_opts.get("max_email_charts", 0))
        include_charts   = bool(email_opts.get("include_charts", False))
        max_email_rows   = int(email_opts.get("max_email_rows", 300))

        # BUY/SELL 선별
        base_list = [r for r in results if str(r.get("signal")).upper() in ("BUY","SELL")] if email_only else list(results)

        # ① 회전율 필터( BUY 에만 적용 )
        if cfg.get("filters", {}).get("use_turnover", True):
            k_min = float(cfg["filters"].get("turnover_min_pct_kospi", 0.15))
            q_min = float(cfg["filters"].get("turnover_min_pct_kosdaq", 0.25))
            def pass_turnover(row: dict) -> bool:
                if str(row.get("signal")).upper() != "BUY":
                    return True
                pct   = row.get("turnover_pct")
                board = row.get("board")
                if pct is None or board is None:
                    return False
                thr = k_min if board == "KOSPI" else q_min
                return pct >= thr
            base_list = [r for r in base_list if pass_turnover(r)]

        # ② 혼잡도 제한( BUY 너무 많을 때 상한 )
        oc = cfg.get("overcrowd", {})
        if oc.get("enabled", True):
            max_buys = int(oc.get("max_buys", 120))
            rank_key = str(oc.get("rank_metric", "turnover_pct"))
            buys  = [r for r in base_list if str(r.get("signal")).upper()=="BUY"]
            sells = [r for r in base_list if str(r.get("signal")).upper()=="SELL"]
            if len(buys) > max_buys:
                # 랭킹 기준: turnover_pct / obv_slope / ma20_gap
                def score(r: dict) -> float:
                    v = r.get({
                        "turnover_pct":"turnover_pct",
                        "obv_slope":"obv_slope",
                        "ma20_gap":"ma20_gap_pct"
                    }.get(rank_key, "turnover_pct"))
                    try:
                        return float(v) if v is not None else -1e9
                    except Exception:
                        return -1e9
                buys = sorted(buys, key=score, reverse=True)[:max_buys]
            base_list = buys + sells

        # ③ 차트 포함 여부
        if not base_list:
            html = "<html><body><h1>No BUY/SELL signals today</h1></body></html>"
            filtered_details = {}
        else:
            symbols = [r["symbol"] for r in base_list]
            if include_charts and max_email_charts > 0:
                symbols = symbols[:max_email_charts]
                filtered_details = {s: details[s] for s in symbols if s in details}
            else:
                filtered_details = {}

            html = build_html_report(
                base_list, filtered_details, cfg,
                show_charts=include_charts and max_email_charts > 0,
                max_charts=max_email_charts,
                max_table_rows=max_email_rows
            )

    # ---------- 전체 리포트 저장 ----------
    save_all = cfg.get("email_options", {}).get("include_hold_in_report", True)
    report_path = os.path.join(cfg['general']['output_dir'], cfg['general']['report_filename'])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(all_html if save_all else html)

    # ---------- (선택) BUY 첨부 리포트 ----------
    attachment_paths = []
    try:
        from report_attach import build_buy_attachment
        email_opts = cfg.get("email_options", {})
        if bool(email_opts.get("attach_buy_report", True)):
            buy_rows = [r for r in results if str(r.get("signal")).upper() == "BUY"]
            if cfg.get("filters", {}).get("use_turnover", True):
                k_min = float(cfg["filters"].get("turnover_min_pct_kospi", 0.15))
                q_min = float(cfg["filters"].get("turnover_min_pct_kosdaq", 0.25))
                buy_rows = [
                    r for r in buy_rows
                    if r.get("turnover_pct") is not None and r.get("board") is not None and
                       r["turnover_pct"] >= (k_min if r["board"]=="KOSPI" else q_min)
                ]
            if buy_rows:
                attach_path = build_buy_attachment(
                    buy_rows=buy_rows,
                    details=details,
                    cfg=cfg,
                    out_path=f"{cfg['general']['output_dir'].rstrip('/')}/buy_report.html",
                    max_charts=int(email_opts.get("max_buy_charts", 10)),
                )
                attachment_paths.append(attach_path)
    except Exception as e:
        print(f"[WARN] build_buy_attachment failed: {e}")

    # ---------- 메일 전송 ----------
    smtp_user   = cfg['email']['from_addr']
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("ERROR: GMAIL_APP_PASSWORD env not set", file=sys.stderr)
        sys.exit(1)

    # 요약 로그
    total = len(results)
    buys  = sum(1 for r in results if str(r.get("signal")).upper()=="BUY")
    sells = sum(1 for r in results if str(r.get("signal")).upper()=="SELL")
    holds = total - buys - sells
    print(f"[SUMMARY] total={total}, BUY={buys}, SELL={sells}, HOLD={holds}")

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
    except Exception as e:
        print(f"[WARN] email send failed: {e}")
        # 552 등 용량 에러 폴백
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
            print("Sent fallback email (reduced size).")
        except Exception as e2:
            print(f"Email failed even after fallback: {e2}", file=sys.stderr)
            sys.exit(1)

    print(f"Report written: {report_path} ({datetime.now()})")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
