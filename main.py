# main.py
import os
import sys
import toml
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# ---- 프로젝트 내부 모듈 ----
from data_sources import (
    fetch_krx, fetch_yahoo, date_range_for_lookback,
    get_all_krx_tickers, get_krx_name,
    get_board_sets, attach_turnover_krx,   # 회전율 보조
)

from strategy import build_features, generate_signal
from reporter import build_html_report
from mailer import send_mail_html
from bt_reporter import save_backtest_outputs

# (선택) 펀더멘탈/뉴스/DART 클라이언트는 없을 수도 있으므로 안전 임포트
try:
    from fundamentals import fetch_fundamental_map
except Exception:
    def fetch_fundamental_map(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        return {}
try:
    from news import fetch_news_headlines
except Exception:
    def fetch_news_headlines(query: str, max_items: int = 3, lang: str = "ko") -> List[Dict[str, str]]:
        return []
try:
    from dart_client import latest_filings
except Exception:
    def latest_filings(company_name: str, max_items: int = 3) -> List[Dict[str, str]]:
        # dart_client 모듈이 없으면 빈 리스트 반환
        return []


# --------------- 유틸 ---------------
def get_name(symbol: str, market: str) -> str:
    if market == "KRX":
        return get_krx_name(symbol)
    return symbol


def _safe_latest(feat: pd.DataFrame) -> Optional[pd.Series]:
    """dropna 후 비어있으면 None 반환(인덱스 오류 방지)."""
    if feat is None or feat.empty:
        return None
    x = feat.dropna()
    if x.empty:
        return None
    return x.iloc[-1]


def _build_regime_series(cfg: dict) -> Optional[pd.Series]:
    """레짐(시장 국면) 시리즈: Close > MA(regime_ma_days) => True"""
    mode = cfg.get("mode", {})
    if not mode:
        return None
    symbol = mode.get("regime_symbol")
    if not symbol:
        return None

    days = int(mode.get("regime_ma_days", 200))
    lb_days = max(days * 2, 260)
    ks, ke, _, _ = date_range_for_lookback(lb_days)
    try:
        df = fetch_krx(symbol, ks, ke)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    ma = df["Close"].rolling(days, min_periods=days//2).mean()
    regime = (df["Close"] > ma).astype(bool)
    regime.name = "REGIME_BULL"
    return regime


def _apply_regime_on_info(info: dict, regime: Optional[pd.Series], cfg: dict) -> dict:
    """레짐 차단 옵션 적용: 약세면 BUY→HOLD로 덮기."""
    if regime is None:
        return info

    disable = bool(cfg.get("mode", {}).get("regime_disable_buy_in_bear", False))
    if not disable:
        return info

    # latest 날짜에 맞춘 레짐 값 reindex
    # info에는 날짜가 없으므로 단순히 regime의 마지막 값 사용(일봉 동기화를 위해)
    is_bull = bool(regime.tail(1).iloc[0]) if len(regime) else True
    if str(info.get("signal")).upper() == "BUY" and not is_bull:
        info = {**info, "signal": "HOLD", "_why": "regime_blocked"}
    return info


def _count(rows: List[dict], label: str) -> int:
    return sum(1 for r in rows if str(r.get("signal")).upper() == label.upper())


# --------------- 심볼 분석 ---------------
def analyze_symbol(symbol: str, market: str, cfg: dict) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    lb = int(cfg['general']['lookback_days'])
    krx_start, krx_end, yah_start, yah_end = date_range_for_lookback(lb)

    if market == "KRX":
        df = fetch_krx(symbol, krx_start, krx_end)
    else:
        df = fetch_yahoo(symbol, yah_start, yah_end)

    if df is None or df.empty or len(df) < max(60, lb // 2):
        return pd.DataFrame(), {}, (df if df is not None else pd.DataFrame())

    feat = build_features(df, cfg)
    feat['Signal'] = generate_signal(feat, cfg)

    latest = _safe_latest(feat)
    if latest is None:
        return pd.DataFrame(), {}, df

    # ma20 괴리
    ma_gap = None
    if pd.notna(latest.get('MA_M')):
        try:
            ma_gap = round((float(latest['Close']) - float(latest['MA_M'])) / float(latest['MA_M']) * 100, 2)
        except Exception:
            ma_gap = None

    info = {
        "market": market,
        "symbol": symbol,
        "name": get_name(symbol, market),
        "signal": str(latest['Signal']).upper(),
        "close": round(float(latest['Close']), 3),
        "rsi": round(float(latest['RSI']), 2) if pd.notna(latest.get('RSI')) else None,
        "wr": round(float(latest['WR']), 2) if pd.notna(latest.get('WR')) else None,
        "ma20_gap_pct": ma_gap
    }
    return feat, info, df


# --------------- 유니버스 로드 ---------------
def load_universe(cfg: dict) -> Tuple[List[str], List[str]]:
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


# --------------- 백테스트(옵션) ---------------
def maybe_run_backtest(cfg: dict, feats_map: dict, df_map: dict) -> Optional[Dict[str, Any]]:
    bt = cfg.get('backtest', {})
    if not bt.get('enabled', False):
        return None

    # 시장 필터용 심볼 DF 만들기
    lb = int(bt.get('lookback_days', 900))
    m_symbol = bt.get('market_filter_symbol', '069500')
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

    # 백테스트 엔진
    from backtest import simulate_symbol

    results = []
    for sym, feat in feats_map.items():
        df = df_map.get(sym)
        if df is None or df.empty or feat is None or feat.empty:
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


# --------------- 메인 ---------------
def main():
    cfg = toml.load("config.toml")
    os.makedirs(cfg['general']['output_dir'], exist_ok=True)

    # 심볼 유니버스
    krx_codes, yah_codes = load_universe(cfg)

    # 레짐 시리즈(옵션)
    regime_series = _build_regime_series(cfg)

    results: List[dict] = []
    details: Dict[str, pd.DataFrame] = {}
    feats_map: Dict[str, pd.DataFrame] = {}
    df_map: Dict[str, pd.DataFrame] = {}

    # ---------- KRX ----------
    for i, sym in enumerate(krx_codes, 1):
        feat, info, df = analyze_symbol(sym, "KRX", cfg)
        if not feat.empty:
            # 레짐 차단 반영
            info = _apply_regime_on_info(info, regime_series, cfg)
            results.append(info)

            save_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','BB_MID','BB_UPPER','BB_LOWER','BB_POS_PCT'
            ] if c in feat.columns]
            details[sym] = feat[save_cols].copy()
            feats_map[sym] = feat
            df_map[sym] = df

        if i % 200 == 0:
            print(f"[KRX] processed {i}/{len(krx_codes)}")

    # ---------- Yahoo (선택) ----------
    for sym in yah_codes:
        feat, info, df = analyze_symbol(sym, "YAHOO", cfg)
        if not feat.empty:
            info = _apply_regime_on_info(info, regime_series, cfg)
            results.append(info)

            save_cols = [c for c in [
                'Open','High','Low','Close','MA_M','RSI','WR','OBV','OBV_slope',
                'Bullish','Bearish','Bull_Engulf','BB_MID','BB_UPPER','BB_LOWER','BB_POS_PCT'
            ] if c in feat.columns]
            details[sym] = feat[save_cols].copy()
            feats_map[sym] = feat
            df_map[sym] = df

    # ---- 1차 요약 로그
    print("[DEBUG] raw BUY/HOLD/SELL =", _count(results, "BUY"), _count(results, "HOLD"), _count(results, "SELL"))

    # ---------- 펀더멘털(선택) ----------
    try:
        krx_syms = [r["symbol"] for r in results if r.get("market") == "KRX"]
        funda_map = fetch_fundamental_map(krx_syms) if krx_syms else {}
    except Exception as e:
        print(f"[WARN] fundamentals fetch failed: {e}")
        funda_map = {}

    for r in results:
        if r.get("market") == "KRX":
            t = r["symbol"]
            if t in funda_map:
                r.update({
                    "per": funda_map[t].get("PER"),
                    "pbr": funda_map[t].get("PBR"),
                    "div": funda_map[t].get("DIV"),
                })

    # ---------- 회전율 필터(옵션) ----------
    filters = cfg.get("filters", {})
    use_turnover = bool(filters.get("use_turnover", False))
    turn_map: Dict[str, Dict[str, Any]] = {}
    if use_turnover:
        try:
            symbols = [r["symbol"] for r in results if r.get("market") == "KRX"]
            # attach_turnover_krx: {'000660': {'turnover_pct': float, 'board': 'KOSPI'|'KOSDAQ'}}
            turn_map = attach_turnover_krx(symbols)
        except Exception as e:
            print(f"[WARN] turnover fetch failed: {e}")
            turn_map = {}

    def pass_turnover(sym: str, info: Optional[Dict[str, Any]], filters: dict) -> bool:
        """값 없음(None)은 '통과' 처리해서 전멸 방지."""
        if not filters.get("use_turnover", False):
            return True
        th_kospi  = float(filters.get("turnover_min_pct_kospi", 0.0))
        th_kosdaq = float(filters.get("turnover_min_pct_kosdaq", 0.0))
        if info is None:
            return True
        pct   = info.get("turnover_pct", None)
        board = info.get("board", None)
        if pct is None or board is None:
            return True
        if board == "KOSPI":
            return pct >= th_kospi
        if board == "KOSDAQ":
            return pct >= th_kosdaq
        return True

    # BUY만 회전율 필터 적용(SELL/HOLD는 그대로)
    filtered_results: List[dict] = []
    for r in results:
        if str(r.get("signal")).upper() == "BUY" and r.get("market") == "KRX":
            sym = r["symbol"]
            info = turn_map.get(sym)
            if not pass_turnover(sym, info, filters):
                # 컷된 건 디버그 표시만
                print(f"[DEBUG] BUY filtered by turnover: {sym} ({info})")
                continue
        filtered_results.append(r)

    print("[DEBUG] after turnover => BUY =", _count(filtered_results, "BUY"))

    # ---------- overcrowd 컷(옵션) ----------
    oc = cfg.get("overcrowd", {})
    if bool(oc.get("enabled", False)):
        max_buys = int(oc.get("max_buys", 120))
        metric = str(oc.get("rank_metric", "turnover_pct")).lower()

        buy_rows = [r for r in filtered_results if str(r.get("signal")).upper() == "BUY"]
        hold_sell_rows = [r for r in filtered_results if str(r.get("signal")).upper() != "BUY"]

        if len(buy_rows) > max_buys:
            # 점수 계산
            def _score(row):
                sym = row["symbol"]
                if metric == "turnover_pct":
                    x = turn_map.get(sym, {}).get("turnover_pct", None)
                    return (x if isinstance(x, (int, float)) else -1.0)
                elif metric == "obv_slope":
                    f = feats_map.get(sym)
                    return float(f["OBV_slope"].iloc[-1]) if f is not None and not f.empty else -1.0
                elif metric == "ma20_gap":
                    return abs(float(row.get("ma20_gap_pct") or 0.0)) * -1.0  # 괴리 과도한 순서로 정렬(예시)
                return 0.0

            buy_rows.sort(key=_score, reverse=True)
            buy_rows = buy_rows[:max_buys]
            filtered_results = buy_rows + hold_sell_rows

        print("[DEBUG] after overcrowd => BUY =", _count(filtered_results, "BUY"))

    # ---------- 리포트(전체/메일) ----------
    if not filtered_results:
        html = "<html><body><h1>No data</h1></body></html>"
        all_html = html
    else:
        # 전체 리포트(파일 저장용) — HOLD 포함 여부 config로 제어
        all_html = build_html_report(
            filtered_results, details, cfg,
            show_charts=True, max_charts=50, max_table_rows=100000
        )

        # 메일용: BUY/SELL만, 차트 제외(용량 보호)
        email_opts = cfg.get("email_options", {})
        email_only = email_opts.get("email_only_signals", True)
        include_charts = bool(email_opts.get("include_charts", False))
        max_email_charts = int(email_opts.get("max_email_charts", 0))
        max_email_rows = int(email_opts.get("max_email_rows", 300))

        if email_only:
            filtered = [r for r in filtered_results if str(r.get("signal")).upper() in ("BUY","SELL")]
            if not filtered:
                html = "<html><body><h1>No BUY/SELL signals today</h1></body></html>"
                filtered_details = {}
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
                filtered_results, {}, cfg,
                show_charts=False, max_table_rows=max_email_rows
            )

    # 전체 리포트 저장
    save_all = cfg.get("email_options", {}).get("include_hold_in_report", True)
    report_path = os.path.join(cfg['general']['output_dir'], cfg['general']['report_filename'])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(all_html if save_all else html)

    # ---------- BUY 첨부(차트/근거/리뷰/뉴스/DART) ----------
    # BUY만 추려서 뉴스/공시 수집 (용량 보호)
    email_opts = cfg.get("email_options", {})
    attach_buy = bool(email_opts.get("attach_buy_report", True))
    max_buy_charts = int(email_opts.get("max_buy_charts", 10))
    attachment_paths: List[str] = []

    aux_info: Dict[str, Dict[str, Any]] = {}
    if attach_buy:
        buy_rows = [r for r in filtered_results if str(r.get("signal")).upper() == "BUY"]
        if buy_rows:
            # 뉴스/공시
            try:
                max_news = int(email_opts.get("max_news_per_symbol", 3))
                max_filings = int(email_opts.get("max_filings_per_symbol", 3))
            except Exception:
                max_news, max_filings = 3, 3

            has_dart_key = bool(os.environ.get("DART_API_KEY", "")) or bool(os.environ.get("OPEN_DART_API_KEY", ""))
            if not has_dart_key:
                print("[WARN] DART_API_KEY/OPEN_DART_API_KEY not set. Filings will be empty.")

            for r in buy_rows[:max_buy_charts]:
                sym = r["symbol"]
                name = r.get("name") or sym
                news_items = []
                filings = []
                # 뉴스(회사명 기반)
                try:
                    news_items = fetch_news_headlines(name, max_items=max_news, lang="ko") or []
                except Exception as e:
                    print(f"[WARN] news fetch failed for {name}: {e}")
                # 공시(DART)
                try:
                    filings = latest_filings(name, max_items=max_filings) or []
                    if not filings and has_dart_key:
                        print(f"[INFO] No DART filings for {name} (check corp-code mapping).")
                except Exception as e:
                    print(f"[WARN] dart fetch failed for {name}: {e}")

                if news_items or filings:
                    aux_info[sym] = {"news": news_items, "filings": filings}

            # 첨부 리포트 생성
            from report_attach import build_buy_attachment
            try:
                attach_path = build_buy_attachment(
                    buy_rows=buy_rows,
                    details=details,
                    cfg=cff(cfg),
                    out_path=f"{cfg['general']['output_dir'].rstrip('/')}/buy_report.html",
                    max_charts=max_buy_charts,
                    aux_info=aux_info,  # ★ 뉴스/공시 전달
                )
                attachment_paths.append(attach_path)
            except Exception as e:
                print(f"[WARN] build_buy_attachment failed: {e}")

    # 요약 로그
    total = len(filtered_results)
    buys  = _count(filtered_results, "BUY")
    sells = _count(filtered_results, "SELL")
    holds = total - buys - sells
    print(f"[SUMMARY] total={total}, BUY={buys}, SELL={sells}, HOLD={holds}")
    print(f"Report written: {report_path} ({datetime.now()})")

    # ---------- 메일 발송 ----------
    smtp_user = cfg['email']['from_addr']
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not app_password:
        print("ERROR: GMAIL_APP_PASSWORD env not set", file=sys.stderr)
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
    except Exception as e:
        # 552 등 용량 에러 폴백
        print(f"[WARN] email send failed: {e}")
        fallback_html = build_html_report(
            [r for r in filtered_results if str(r.get("signal")).upper() in ("BUY","SELL")],
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


def cff(cfg: dict) -> dict:
    """config defensive copy (리포트에 그대로 넘겨도 안전하도록)."""
    # 필요 시 얕은 복사로 충분
    return dict(cfg)


if __name__ == "__main__":
    main()
