import pandas as pd
import numpy as np

def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High'].rolling(period).max()
    low = df['Low'].rolling(period).min()
    wr = -100 * (high - df['Close']) / (high - low)
    return wr

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff()).fillna(0.0)
    vol = df['Volume'].fillna(0.0)
    return (direction * vol).cumsum()

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def candle_bearish(df: pd.DataFrame) -> pd.Series:
    return (df['Close'] < df['Open'])

def candle_bullish(df: pd.DataFrame) -> pd.Series:
    return (df['Close'] > df['Open'])

def slope(series: pd.Series, lookback: int) -> pd.Series:
    return (series - series.shift(lookback)) / lookback
