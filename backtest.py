import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()

def simulate_symbol(
    signals: pd.Series,
    ohlc: pd.DataFrame,
    *,
    market_ok: pd.Series = None,
    avg_value: pd.Series = None,
    liquidity_min_value: float = 0.0,
    max_hold_days: int = 0,
    atr_period: int = 14,
    stop_atr_mult: float = 0.0,
    trail_atr_mult: float = 0.0,
    cost_bps_per_side: float = 25.0
):
    """
    Long-only. BUY/SELL 신호 사용.
    체결 규칙: 신호 발생일(D) 기준, 다음 거래일 시가(D+1 Open)에서 체결.
    market_ok False면 매수 금지. avg_value < liquidity_min_value면 매수 금지.
    종료 규칙: SELL 신호, max_hold_days, ATR 손절/트레일 조건 중 먼저 히트 시 D+1 시가로 청산.
    """
    df = ohlc.sort_index().copy()
    sig = signals.reindex(df.index).fillna("HOLD")

    if market_ok is not None:
        m_ok = market_ok.reindex(df.index).fillna(False)
    else:
        m_ok = pd.Series(True, index=df.index)

    if avg_value is not None:
        av = avg_value.reindex(df.index).fillna(0.0)
    else:
        av = pd.Series(np.inf, index=df.index)

    atrv = atr(df, period=atr_period)

    cost = cost_bps_per_side/10000.0
    pos = 0
    entry_px = None
    peak_px = None
    entry_idx = None

    equity = 1.0
    equity_curve = []
    trades = []
    last_close = df['Close'].shift(1)

    for i, dt in enumerate(df.index[:-1]):  # 마지막 날은 체결 불가
        today = dt
        nxt = df.index[i+1]
        s = sig.loc[today]

        # 보유 중 일간 마크투마켓 (전일 대비 종가 수익률)
        if pos == 1 and not pd.isna(last_close.loc[today]) and last_close.loc[today] > 0:
            daily_ret = (df['Close'].loc[today] / last_close.loc[today]) - 1.0
            equity *= (1.0 + daily_ret)

        # 청산 조건 평가 (체결은 다음날 시가)
        exit_reason = None
        if pos == 1:
            if s == "SELL":
                exit_reason = "SELL"
            if exit_reason is None and max_hold_days and entry_idx is not None:
                if (pd.Timestamp(today) - pd.Timestamp(entry_idx)).days >= max_hold_days:
                    exit_reason = "MAX_HOLD"
            if exit_reason is None and stop_atr_mult and not pd.isna(atrv.loc[today]) and entry_px is not None:
                stop_line = entry_px - stop_atr_mult * atrv.loc[today]
                if df['Close'].loc[today] <= stop_line:
                    exit_reason = "STOP"
            if exit_reason is None and trail_atr_mult and not pd.isna(atrv.loc[today]):
                peak_px = max(peak_px or df['Close'].loc[today], df['Close'].loc[today])
                trail_line = peak_px - trail_atr_mult * atrv.loc[today]
                if df['Close'].loc[today] <= trail_line:
                    exit_reason = "TRAIL"

            if exit_reason is not None:
                exit_px = df['Open'].loc[nxt] * (1.0 - cost)
                if entry_px:
                    equity *= (exit_px / entry_px)
                trades.append({
                    "entry_date": entry_idx, "entry_px": entry_px,
                    "exit_date": nxt, "exit_px": exit_px,
                    "reason": exit_reason
                })
                pos = 0
                entry_px = None
                entry_idx = None
                peak_px = None

        # 진입 (체결은 다음날 시가)
        if pos == 0 and s == "BUY":
            if m_ok.loc[today] and av.loc[today] >= liquidity_min_value:
                entry_px = df['Open'].loc[nxt] * (1.0 + cost)
                entry_idx = nxt
                peak_px = entry_px
                pos = 1

        equity_curve.append((today, equity))

    # 마지막 날 기록
    last_day = df.index[-1]
    equity_curve.append((last_day, equity))
    eq = pd.Series([e for (_, e) in equity_curve], index=[d for (d, _) in equity_curve])

    metrics = performance(eq)
    metrics["Trades"] = len(trades)
    return metrics, eq, trades

def performance(eq: pd.Series):
    eq = eq.dropna()
    if eq.empty or len(eq) < 5:
        return {"CAGR": 0.0, "MDD": 0.0, "Vol": 0.0, "Sharpe": np.nan, "LastEquity": float(eq.iloc[-1]) if len(eq)>0 else 1.0}
    ret = eq.pct_change().dropna()
    cum = float(eq.iloc[-1])
    mdd = ((eq/eq.cummax())-1).min()
    ann = cum**(252/len(eq)) - 1 if len(eq)>0 else 0
    vol = float(ret.std()*np.sqrt(252)) if len(ret)>2 else 0.0
    sharpe = float(ann/vol) if vol>0 else np.nan
    return {"CAGR": float(ann), "MDD": float(mdd), "Vol": vol, "Sharpe": sharpe, "LastEquity": cum}
