# main.py
import os
import sys
import toml
import pandas as pd
from datetime import datetime

# --- 프로젝트 모듈 ---
from data_sources import (
    fetch_krx, fetch_yahoo, date_range_for_lookback,
    get_all_krx_tickers, get_krx_name,
    get_board_sets
)
import data_sources as _ds

from strategy import build_features, generate_signal
from reporter import build_html_report
from mailer import send_mail_html

# (옵션) 백테스트
try:
    from backtest import simulate_symbol
    from bt_reporter import save_backtest_outputs
except Exception:
    simulate_symbol = None
    save_backtest_outputs = None

# (옵션) 펀더멘털/뉴스/공시
try:
    from fundamentals import fetch_fundamental_map  # {'000660': {'PER':..,'PBR':..,'DIV':..}, ...}
except Exception:
    fetch_fundamental_map = None

try:
    from news import fetch_news_headlines  # (name, max_items, lang) -> [{'title','link','published'}]
except Exception:
    fetch_news_headlines = None

try:
    from dart_client import latest_filings  # (corp_name, max_items) -> [{'rpt','link','rcpdt'}]
except Exception:
    latest_filings = None


# -------------------------------
# 유틸
# -------------------------------
def get_name(symbol: str, market: str) -> str:
    if market == "KRX":
        return get_krx_name(symbol)
    return symbol


def analyze_symbol(symbol: str, market: str, cfg: dict):
    lb = int(cfg['general']['lookback_days'])
    krx_start, krx_end, yah_start, yah_end = date_range_for_lookback(lb)

    if market == "KRX":
        df = fetch_krx(symbol, krx_start, krx_end)
    else:
        df = fetch_yahoo(symbol, yah_start, yah_end)

    if df is None or df.empty or len(df) < max(60, lb // 2):
        # (feat빈값, info빈딕트, df 그대로)
        return pd.DataFrame(), {}, df

    feat = build_features(df, cfg)
    sig = generate_signal(feat, cfg)
    feat['Signal'] = sig

    # 마지막 유효 행
    latest_idx = sig.last_valid_index()
    if latest_idx is None:
        return pd.DataFrame(), {}, df
    latest = feat.loc[latest_idx]

    info = {
        "market": market,
        "symbol": symbol,
        "name": get_name(symbol, market),
        "signal": str(latest['Signal']).upper(),
        "close": round(float(latest['Close']), 3),
        "rsi": round(float(latest['RSI']), 2) if pd.notna(latest.get('RSI')) else None,
        "wr": round(float(latest['WR']), 2) if pd.notna(latest.get('WR')) else None,
        "ma20_gap_pct": round(((latest['Close'] - latest['MA_M']) / latest['MA_M'] * 100), 2)
                        if pd.notna(latest.get('MA_M')) and latest['MA_M'] else None
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


def maybe_run_backtest(cfg: dict, feats_map: dict, df_map: dict):
    if not cfg.get('backtest', {}).get('enabled', False):
        return None
    if simulate_symbol is None or save_backtest_outputs is None:
        return None

    bt = cfg['backtest']
    lb = int(bt.get('lookback_days', 900))
    m_symbol = bt.get('market_filter_symbol', '069500')

    # 시장 필터용 DF
    m_df = df_map.get(m_symbol)
    if m_df is None or m_df.empty:
        ks, ke, _, _ = date_range_for_lookback(lb)
        try:
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
            mk = market_ok.reindex(df.index).ffill().fillna(False)

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


# -------------------------------
# BUY 랭킹(상위 후보만 메일/첨부)
# -------------------------------
def rank_buy_candidates(buy_rows, extras, top_n=30):
    """
    buy_rows: [{'symbol','market','name','rsi','wr',...}, ...]
    extras:   {'turnover_pct': {sym: pct or None}, 'obv_slope': {sym: val or None}}
    """
    rows = []
    for r in buy_rows:
        sym = r["symbol"]
        rsi = float(r.get("rsi")) if r.get("rsi") is not None else 50.0
        wr  = float(r.get("wr"))  if r.get("wr")  is not None else -50.0
        obv = extras.get("obv_slope", {}).get(sym) or 0.0
        tov = extras.get("turnover_pct", {}).get(sym) or 0.0

        # 간단 정규화
        rsi_score = max(0.0, (50.0 - rsi) / 50.0)     # RSI 낮을수록 +
        wr_score  = max(0.0, (-20.0 - wr) / 80.0)     # WR -100~-20 구간 +
        obv_score = max(0.0, obv)                     # 양수일수록 +
        tov_score = max(0.0, min(tov, 1.0))           # 0~1%/day까지만 가점

        score = (0.35 * rsi_score) + (0.35 * wr_score) + (0.20 * obv_score) + (0.10 * tov_score)
        rows.append((score, r))

    rows.sort(key=lambda x: x[0], reverse=True)
    ranked = [r for _, r in rows[:top_n]]
    return ranked


def main():
    cfg = toml.load("config.toml")
    os.makedirs(cfg['general']['output_dir'], exist_ok=True)

    # 유니버스
    krx_codes, yah_codes = load_universe(cfg)

    results = []
    details = {}
    feats_map = {}
    df_map = {}

    # ---- KRX 스캔 ----
    for i, sym in enumerate(krx_codes, 1):
        feat, info, df = analyze_symbol(sym, "KRX", cfg)
        if not feat.empty:
            results.append(info)

            # 리포트에 필요한 최소 열만 저장(메모리 보호)
            save_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','BB_POS_PCT','BB_UPPER','BB_LOWER'
            ] if c in feat.columns]
            details[sym] = feat[save_cols].copy()
            feats_map[sym] = feat
            df_map[sym] = df

        if i % 200 == 0:
            print(f"[KRX] processed {i}/{len(krx_codes)}")

    # ---- Yahoo (선택) ----
    for sym in yah_codes:
        feat, info, df = analyze_symbol(sym, "YAHOO", cfg)
        if not feat.empty:
            results.append(info)
            save_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','BB_POS_PCT','BB_UPPER','BB_LOWER'
            ] if c in feat.columns]
            details[sym] = feat[save_cols].copy()
            feats_map[sym] = feat
            df_map[sym] = df

    if not results:
        html = "<html><body><h1>No data</h1></body></html>"
        all_html = html
    else:
        # ---------------------------
        # 회전율 필터 / 맵
        # ---------------------------
        filt = cfg.get("filters", {})
        use_turnover = bool(filt.get("use_turnover", False))
        # 오늘(또는 최근 영업일) 스냅샷 거래대금/시총으로 회전율 계산
        turnover_info = _ds.attach_turnover_krx([r["symbol"] for r in results if r["market"] == "KRX"])
        # board(KOSPI/KOSDAQ) 기준 임계치
        kospi_min = float(filt.get("turnover_min_pct_kospi", 0.15))
        kosdaq_min = float(filt.get("turnover_min_pct_kosdaq", 0.25))

        # 레짐(시장국면) BUY 무효 옵션
        reg = cfg.get("mode", {})
        regime_disable = bool(reg.get("regime_disable_buy_in_bear", False))
        regime_symbol = reg.get("regime_symbol", "069500")
        regime_days = int(reg.get("regime_ma_days", 200))

        regime_ok = True
        if regime_disable:
            try:
                ks, ke, _, _ = date_range_for_lookback(regime_days + 30)
                r_df = fetch_krx(regime_symbol, ks, ke)
                if not r_df.empty:
                    ma = r_df['Close'].rolling(regime_days).mean()
                    regime_ok = bool(r_df['Close'].iloc[-1] > ma.iloc[-1])
            except Exception:
                regime_ok = True  # 장애 시 무시

        # BUY/SELL/HOLD 수 집계
        raw_counts = pd.Series([str(r.get("signal")).upper() for r in results]).value_counts()
        print(f"[DEBUG] raw BUY/HOLD/SELL = "
              f"{raw_counts.get('BUY',0)} {raw_counts.get('HOLD',0)} {raw_counts.get('SELL',0)}")

        # 1) 회전율 필터 적용 (옵션)
        filtered_results = []
        for r in results:
            if str(r.get("signal")).upper() != "BUY" or not use_turnover:
                filtered_results.append(r)
                continue
            info = turnover_info.get(r["symbol"], {})
            tov = info.get("turnover_pct")
            brd = info.get("board")
            pass_cond = True
            if tov is None or brd is None:
                pass_cond = False
            else:
                if brd == "KOSPI":
                    pass_cond = (tov >= kospi_min)
                elif brd == "KOSDAQ":
                    pass_cond = (tov >= kosdaq_min)
                else:
                    pass_cond = False

            if pass_cond:
                filtered_results.append(r)
        results = filtered_results

        print(f"[DEBUG] after turnover => BUY = "
              f"{sum(1 for x in results if str(x.get('signal')).upper()=='BUY')}")

        # 2) 레짐: 약세장이면 BUY 무효화
        if regime_disable and not regime_ok:
            for r in results:
                if str(r.get("signal")).upper() == "BUY":
                    r["signal"] = "HOLD"

        # 3) BUY 과다 시 상한/랭킹 컷팅
        overcrowd = cfg.get("overcrowd", {})
        if bool(overcrowd.get("enabled", True)):
            max_buys = int(overcrowd.get("max_buys", 120))
            buy_rows_now = [r for r in results if str(r.get("signal")).upper() == "BUY"]

            if len(buy_rows_now) > max_buys:
                # 랭킹에 필요한 보조 맵
                obv_map = {}
                for r in buy_rows_now:
                    sym = r["symbol"]
                    if sym in details and "OBV_slope" in details[sym].columns and not details[sym].empty:
                        obv_map[sym] = float(details[sym]["OBV_slope"].iloc[-1])
                    else:
                        obv_map[sym] = 0.0

                tov_map = {s: turnover_info.get(s, {}).get("turnover_pct") or 0.0 for s in [r["symbol"] for r in buy_rows_now]}

                ranked = rank_buy_candidates(
                    buy_rows_now,
                    extras={"turnover_pct": tov_map, "obv_slope": obv_map},
                    top_n=max_buys
                )
                # ranked 외 나머지 BUY는 HOLD로 다운그레이드
                ranked_syms = {r["symbol"] for r in ranked}
                new_results = []
                for r in results:
                    if str(r.get("signal")).upper() == "BUY" and r["symbol"] not in ranked_syms:
                        r = dict(r)
                        r["signal"] = "HOLD"
                    new_results.append(r)
                results = new_results

        print(f"[DEBUG] after overcrowd => BUY = "
              f"{sum(1 for x in results if str(x.get('signal')).upper()=='BUY')}")

        # ---------------------------
        # 펀더멘털(옵션)
        # ---------------------------
        try:
            if fetch_fundamental_map:
                krx_tickers = [r["symbol"] for r in results if r.get("market") == "KRX"]
                funda_map = fetch_fundamental_map(krx_tickers)
            else:
                funda_map = {}
        except Exception:
            funda_map = {}

        for r in results:
            if r.get("market") != "KRX":
                continue
            t = r["symbol"]
            if t in funda_map:
                r.update({
                    "per": funda_map[t].get("PER"),
                    "pbr": funda_map[t].get("PBR"),
                    "div": funda_map[t].get("DIV"),
                })

        # ---------------------------
        # 뉴스/공시: BUY만 (첨부 리포트용)
        # ---------------------------
        aux_info = {}
        email_opts = cfg.get("email_options", {})
        max_news = int(email_opts.get("max_news_per_symbol", 3) or 3)
        max_fil  = int(email_opts.get("max_filings_per_symbol", 3) or 3)

        have_dart_key = bool(os.environ.get("DART_API_KEY") or os.environ.get("OPEN_DART_API_KEY"))
        if not have_dart_key and latest_filings:
            print("[WARN] DART_API_KEY/OPEN_DART_API_KEY not set. Filings will be empty.")

        for r in results:
            if str(r.get("signal")).upper() != "BUY":
                continue
            sym = r["symbol"]
            name = r.get("name") or sym

            news_items = []
            if fetch_news_headlines:
                try:
                    news_items = fetch_news_headlines(name, max_items=max_news, lang="ko")
                except Exception:
                    news_items = []

            filings = []
            if latest_filings and have_dart_key:
                try:
                    filings = latest_filings(name, max_items=max_fil)
                except Exception:
                    filings = []

            if news_items or filings:
                aux_info[sym] = {"news": news_items, "filings": filings}

        # ---------------------------
        # 전체 리포트(파일 저장용)
        # ---------------------------
        all_html = build_html_report(
            results, details, cfg,
            show_charts=True,
            max_charts=50,
            max_table_rows=100000
        )

        # ---------------------------
        # 메일 본문: BUY/SELL만, 차트 제외(기본)
        # ---------------------------
        email_opts = cfg.get("email_options", {})
        email_only = bool(email_opts.get("email_only_signals", True))
        include_charts = bool(email_opts.get("include_charts", False))
        max_email_charts = int(email_opts.get("max_email_charts", 0))
        max_email_rows = int(email_opts.get("max_email_rows", 300))

        # BUY 우선 랭킹 리스트 (첨부와 동일 기준)
        buy_rows_all = [r for r in results if str(r.get("signal")).upper() == "BUY"]
        obv_map2 = {r["symbol"]: (float(details[r["symbol"]]["OBV_slope"].iloc[-1])
                                  if r["symbol"] in details and "OBV_slope" in details[r["symbol"]].columns and not details[r["symbol"]].empty else 0.0)
                    for r in buy_rows_all}
        tov_map2 = {r["symbol"]: (_ds.attach_turnover_krx([r["symbol"]]).get(r["symbol"], {}).get("turnover_pct") or 0.0)
                    for r in buy_rows_all} if buy_rows_all else {}
        # 상위 25%(10~40) 기본
        email_buy_rows = rank_buy_candidates(
            buy_rows_all,
            extras={"turnover_pct": tov_map2, "obv_slope": obv_map2},
            top_n=min(40, max(10, int(len(buy_rows_all) * 0.25))) if buy_rows_all else 0
        )

        if email_only:
            filtered = (
                email_buy_rows +
                [r for r in results if str(r.get("signal")).upper() == "SELL"]
            )
            if not filtered:
                html = "<html><body><h1>No BUY/SELL signals today</h1></body></html>"
                filtered_details = {}
            else:
                # 차트 포함을 켜더라도 본문은 가볍게(보통 False)
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
                results, {}, cfg, show_charts=False, max_table_rows=max_email_rows
            )

    # ---------- 파일 저장 (전체 리포트) ----------
    save_all = cfg.get("email_options", {}).get("include_hold_in_report", True)
    report_path = os.path.join(cfg['general']['output_dir'], cfg['general']['report_filename'])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(all_html if save_all else html)

    # ---------- 메일 전송 준비 (첨부 리포트 생성) ----------
    smtp_user = cfg['email']['from_addr']
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("ERROR: GMAIL_APP_PASSWORD env not set", file=sys.stderr)
        sys.exit(1)

    email_opts = cfg.get("email_options", {})
    attach_buy_report = bool(email_opts.get("attach_buy_report", True))
    max_buy_charts    = int(email_opts.get("max_buy_charts", 10))   # 30으로 올려도 됨
    embed_charts      = bool(email_opts.get("attach_embed_charts", False))  # 기본 False(용량 절감)

    # (1) BUY만 추려서 보조 데이터 준비
    buy_rows = [r for r in results if str(r.get("signal")).upper() == "BUY"]

    # 뉴스/공시 수집 (이미 위에서 만든 aux_info가 있으면 재사용)
    try:
        aux_info
    except NameError:
        aux_info = {}
        try:
            email_opts = cfg.get("email_options", {})
            max_news = int(email_opts.get("max_news_per_symbol", 3))
            max_filings = int(email_opts.get("max_filings_per_symbol", 3))
        except Exception:
            max_news, max_filings = 3, 3

        if buy_rows:
            for r in buy_rows:
                sym = r["symbol"]
                name = r.get("name") or sym
                try:
                    if fetch_news_headlines:
                        news_items = fetch_news_headlines(name, max_items=max_news, lang="ko")
                    else:
                        news_items = []
                except Exception:
                    news_items = []
                try:
                    if latest_filings and (os.environ.get("DART_API_KEY") or os.environ.get("OPEN_DART_API_KEY")):
                        filings = latest_filings(name, max_items=max_filings)
                    else:
                        filings = []
                except Exception:
                    filings = []
                if news_items or filings:
                    aux_info[sym] = {"news": news_items, "filings": filings}

    # (2) 회전율(당일) 맵: {sym: {"turnover_pct":..., "board":...}}
    turnover_map = {}
    try:
        buy_syms = [r["symbol"] for r in buy_rows if r.get("market") == "KRX"]
        if buy_syms:
            turnover_map = _ds.attach_turnover_krx(buy_syms) or {}
    except Exception as e:
        print(f"[WARN] turnover_map fail: {e}")
        turnover_map = {}

    # (3) 첨부 리포트 생성
    attachment_paths  = []
    if attach_buy_report and buy_rows:
        try:
            from report_attach import build_buy_attachment
            attach_path = build_buy_attachment(
                buy_rows=buy_rows,
                details=details,                 # 차트/전일 거래량 산출용 원시 피처 DF
                cfg=cfg,
                out_path=f"{cfg['general']['output_dir'].rstrip('/')}/buy_report.html",
                max_charts=max_buy_charts,       # 30으로 늘리면 embed_charts=False 권장
                aux_info=aux_info,               # 뉴스/공시
                embed_charts=True,       # False면 용량 매우 작음(권장)
                turnover_map=turnover_map,       # 회전율 표시
            )
            attachment_paths.append(attach_path)
        except Exception as e:
            print(f"[WARN] build_buy_attachment failed: {e}")

    # 요약 로그
    total = len(results)
    buys  = sum(1 for r in results if str(r.get("signal")).upper()=="BUY")
    sells = sum(1 for r in results if str(r.get("signal")).upper()=="SELL")
    holds = total - buys - sells
    print(f"[SUMMARY] total={total}, BUY={buys}, SELL={sells}, HOLD={holds}")

    # ---------- 메일 전송 ----------
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
        # 폴백: 표만, 150행, 첨부 없음
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


if __name__ == "__main__":
    main()
