import pandas as pd
from indicators import williams_r, rsi, obv, moving_average, candle_bearish, candle_bullish, slope

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out['WR'] = williams_r(out, cfg['signals']['williams_r_period'])
    out['RSI'] = rsi(out['Close'], cfg['signals']['rsi_period'])
    out['OBV'] = obv(out)
    out['OBV_slope'] = slope(out['OBV'], cfg['signals']['obv_trend_lookback'])

    out['MA_S'] = moving_average(out['Close'], cfg['signals']['ma_short'])
    out['MA_M'] = moving_average(out['Close'], cfg['signals']['ma_mid'])
    out['MA_L'] = moving_average(out['Close'], cfg['signals']['ma_long'])

    if cfg['signals'].get('use_bearish_candle', True):
        out['Bearish'] = candle_bearish(out)
    else:
        out['Bearish'] = False

    if cfg['signals'].get('use_bullish_candle', True):
        out['Bullish'] = candle_bullish(out)
    else:
        out['Bullish'] = False

    return out

def generate_signal(feat: pd.DataFrame, cfg: dict) -> pd.Series:
    s = pd.Series(index=feat.index, dtype="object")

    buy = (
        (feat['WR'] <= cfg['signals']['williams_r_buy']) &
        (feat['RSI'] <= cfg['signals']['rsi_buy']) &
        (feat['OBV_slope'] >= cfg['signals']['obv_slope_min'])
    )
    if cfg['signals'].get('require_above_ma20_for_buy', True):
        buy &= (feat['Close'] > feat['MA_M'])
    if cfg['signals'].get('use_bullish_candle', True):
        buy &= feat['Bullish']

    sell = (
        (feat['WR'] >= cfg['signals']['williams_r_sell']) &
        (feat['RSI'] >= cfg['signals']['rsi_sell']) &
        (feat['OBV_slope'] <= 0)
    )
    if cfg['signals'].get('require_below_ma20_for_sell', True):
        sell &= (feat['Close'] < feat['MA_M'])
    if cfg['signals'].get('use_bearish_candle', True):
        sell &= feat['Bearish']

    s[buy] = "BUY"
    s[sell] = "SELL"
    s[(~buy) & (~sell)] = "HOLD"

    return s
