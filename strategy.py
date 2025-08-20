import pandas as pd
import numpy as np
from indicators import (
    williams_r, rsi, obv, moving_average, candle_bearish, candle_bullish, slope
)

# -------- 공통 유틸 --------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def macd_lines(close: pd.Series, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    macd_sig = _ema(macd, signal)
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df['Open'], df['Close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    # 전일 음봉 & 당일 양봉 & 당일 몸통이 전일 몸통을 감쌈
    return (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)

def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df['Open'], df['Close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)

def bullish_divergence_proxy(df: pd.DataFrame, rsi_s: pd.Series, lookback=40, half=20) -> pd.Series:
    """
    프록시 방식의 상승 다이버전스:
    - 최근 half창 저가 < 이전 half창 저가  (가격 저점 낮아짐)
    - 최근 half창 RSI 최저 > 이전 half창 RSI 최저 (RSI 저점 높아짐)
    """
    low = df['Low']
    prev_low = low.shift(half).rolling(half, min_periods=max(2, half//2)).min()
    curr_low = low.rolling(half, min_periods=max(2, half//2)).min()

    prev_rsi_min = rsi_s.shift(half).rolling(half, min_periods=max(2, half//2)).min()
    curr_rsi_min = rsi_s.rolling(half, min_periods=max(2, half//2)).min()

    return (curr_low < prev_low) & (curr_rsi_min > prev_rsi_min)

def bearish_divergence_proxy(df: pd.DataFrame, rsi_s: pd.Series, lookback=40, half=20) -> pd.Series:
    high = df['High']
    prev_high = high.shift(half).rolling(half, min_periods=max(2, half//2)).max()
    curr_high = high.rolling(half, min_periods=max(2, half//2)).max()

    prev_rsi_max = rsi_s.shift(half).rolling(half, min_periods=max(2, half//2)).max()
    curr_rsi_max = rsi_s.rolling(half, min_periods=max(2, half//2)).max()

    return (curr_high > prev_high) & (curr_rsi_max < prev_rsi_max)

# -------- 피처 생성 --------
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    # 기본 지표
    out['WR']  = williams_r(out, cfg['signals']['williams_r_period'])
    out['RSI'] = rsi(out['Close'], cfg['signals']['rsi_period'])
    out['OBV'] = obv(out)
    out['OBV_slope'] = slope(out['OBV'], cfg['signals']['obv_trend_lookback'])

    out['MA_S'] = moving_average(out['Close'], cfg['signals']['ma_short'])
    out['MA_M'] = moving_average(out['Close'], cfg['signals']['ma_mid'])
    out['MA_L'] = moving_average(out['Close'], cfg['signals']['ma_long'])

    # 캔들(옵션)
    out['Bearish'] = candle_bearish(out) if cfg['signals'].get('use_bearish_candle', True) else False
    out['Bullish'] = candle_bullish(out) if cfg['signals'].get('use_bullish_candle', True) else False

    # 안전장치: inf/NaN 정리
    out = out.replace([float('inf'), float('-inf')], pd.NA)
    for col in ['WR','RSI','OBV','OBV_slope','MA_S','MA_M','MA_L','Open','High','Low','Close','Volume']:
        if col in out.columns:
            out[col] = out[col].ffill()

    # OHLCV 보존
    for col in ['Open','High','Low','Close','Volume']:
        if col not in out.columns and col in df.columns:
            out[col] = df[col]

    # --- 블로그 모드용 보조지표 (MACD/거래량 평균) ---
    bcfg = cfg.get('blog_signals', {})
    m_fast = int(bcfg.get('macd_fast', 12))
    m_slow = int(bcfg.get('macd_slow', 26))
    m_sig  = int(bcfg.get('macd_signal', 9))

    macd, macd_sig, macd_hist = macd_lines(out['Close'], m_fast, m_slow, m_sig)
    out['MACD'] = macd
    out['MACD_SIG'] = macd_sig
    out['MACD_HIST'] = macd_hist

    out['VOL_MA20'] = out['Volume'].rolling(20, min_periods=10).mean()

    # 블로그 모드용 캔들(장악형)
    out['Bull_Engulf'] = bullish_engulfing(out)
    out['Bear_Engulf'] = bearish_engulfing(out)

    # --- Bollinger Bands (report 참고용) ---
    bb_cfg = cfg.get('bbands', {})
    if bb_cfg.get('enabled', True):
        n = int(bb_cfg.get('period', 20))
        k = float(bb_cfg.get('stdev', 2.0))
        roll = out['Close'].rolling(window=n, min_periods=n)
        ma = roll.mean()
        std = roll.std()

        out['BB_MID']   = ma
        out['BB_UPPER'] = ma + k * std
        out['BB_LOWER'] = ma - k * std
        # 분모 0 방지
        with np.errstate(invalid='ignore', divide='ignore'):
            out['BB_WIDTH'] = (out['BB_UPPER'] - out['BB_LOWER']) / ma
            band = (out['BB_UPPER'] - out['BB_LOWER'])
            pos = (out['Close'] - out['BB_LOWER']) / band * 100.0
        out['BB_POS_PCT'] = pos.clip(lower=0, upper=100)

    return out

# -------- 기본(기존) 전략 신호: 점수식 --------
def _generate_signal_default(feat: pd.DataFrame, cfg: dict) -> pd.Series:
    sigcfg = cfg['signals']

    wr_buy   = float(sigcfg.get('williams_r_buy', -80))
    wr_sell  = float(sigcfg.get('williams_r_sell', -20))
    rsi_buy  = float(sigcfg.get('rsi_buy', 30))
    rsi_sell = float(sigcfg.get('rsi_sell', 70))
    obv_min  = float(sigcfg.get('obv_slope_min', 0.0))

    buy_th  = int(sigcfg.get('buy_score_threshold', 2))
    sell_th = int(sigcfg.get('sell_score_threshold', 2))

    buy_score = (
        (feat['WR']  <= wr_buy ).astype(int) +
        (feat['RSI'] <= rsi_buy).astype(int) +
        (feat['OBV_slope'] >= obv_min).astype(int)
    )
    sell_score = (
        (feat['WR']  >= wr_sell ).astype(int) +
        (feat['RSI'] >= rsi_sell).astype(int) +
        (feat['OBV_slope'] <= 0).astype(int)
    )

    if sigcfg.get('use_bullish_candle', False):
        buy_score += feat['Bullish'].astype(int)
    if sigcfg.get('use_bearish_candle', False):
        sell_score += feat['Bearish'].astype(int)

    if sigcfg.get('require_above_ma20_for_buy', False):
        buy_score += (feat['Close'] > feat['MA_M']).astype(int)
    if sigcfg.get('require_below_ma20_for_sell', False):
        sell_score += (feat['Close'] < feat['MA_M']).astype(int)

    s = pd.Series("HOLD", index=feat.index, dtype="object")
    s = s.mask(buy_score  >= buy_th,  "BUY")
    s = s.mask(sell_score >= sell_th, "SELL")
    return s.astype(str).str.upper()

# -------- 블로그 전략 신호: 다이버전스 + MACD + 장악형 + 거래량 --------
def _generate_signal_blog(feat: pd.DataFrame, cfg: dict) -> pd.Series:
    bcfg = cfg.get('blog_signals', {})
    rsi_period = int(bcfg.get('rsi_period', 14))
    lookback   = int(bcfg.get('divergence_lookback', 40))
    half       = int(bcfg.get('div_half_window', max(10, lookback//2)))

    need_engulf = bool(bcfg.get('require_bullish_engulfing', True))
    need_vol    = bool(bcfg.get('require_volume_surge', True))
    vol_mult    = float(bcfg.get('volume_surge_multiplier', 1.5))

    # 지표 준비
    rsi_s = rsi(feat['Close'], rsi_period)

    macd = feat['MACD']; macd_sig = feat['MACD_SIG']
    macd_golden = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
    macd_dead   = (macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))

    bull_div = bullish_divergence_proxy(feat, rsi_s, lookback, half)
    bear_div = bearish_divergence_proxy(feat, rsi_s, lookback, half)

    bull_eng = feat['Bull_Engulf']
    bear_eng = feat['Bear_Engulf']

    vol_avg = feat['VOL_MA20']
    vol_surge = feat['Volume'] > (vol_avg * vol_mult)

    # 조건 결합
    buy = bull_div & macd_golden
    if need_engulf:
        buy &= bull_eng
    if need_vol:
        buy &= vol_surge

    sell = bear_div & macd_dead
    # 필요하면 매도에 장악형/볼륨도 추가:
    # if bcfg.get('require_bearish_engulfing', False): sell &= bear_eng
    # if bcfg.get('require_volume_surge_for_sell', False): sell &= (feat['Volume'] > vol_avg * vol_mult)

    s = pd.Series("HOLD", index=feat.index, dtype="object")
    s = s.mask(buy, "BUY")
    s = s.mask(sell, "SELL")
    return s.astype(str).str.upper()

# -------- 외부 노출 함수 --------
def generate_signal(feat: pd.DataFrame, cfg: dict) -> pd.Series:
    mode = cfg.get('mode', {}).get('signal_mode', 'default').lower()
    if mode == "blog":
        return _generate_signal_blog(feat, cfg)
    return _generate_signal_default(feat, cfg)
