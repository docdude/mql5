from itertools import combinations
from typing import Union

import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import talib
from feature_engine.selection import DropCorrelatedFeatures
from loguru import logger
from matplotlib import pyplot as plt
from numba import njit, prange


def get_period_returns(close: pd.Series, **time_delta_kwargs) -> pd.Series:
    """
    Compute periodic returns for a given time period, robust to non-consecutive trading days.

    This function calculates returns by finding the closing price from a specified
    time duration (days, hours, minutes) in the past. It handles cases where
    the prior period might not be a trading day by using `searchsorted` to find
    the nearest valid previous index.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) Periodic returns (percentage changes), aligned to the prior valid trading period
    """
    # Find previous valid trading day for each date
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(**time_delta_kwargs))

    # Drop indices that are before the start of the 'close' Series
    prev_idx = prev_idx[prev_idx > 0]

    # Align current and previous closes
    curr_idx = close.index[close.shape[0] - prev_idx.shape[0] :]
    prev_close = close.iloc[prev_idx - 1].values

    ret = close.loc[curr_idx] / prev_close - 1
    return ret


def get_period_vol(close: pd.Series, lookback: int = 100, **time_delta_kwargs) -> pd.Series:
    """
    Compute the exponentially weighted moving volatility of periodic returns.

    This function first calculates periodic returns using `get_period_returns`
    and then applies an Exponentially Weighted Moving (EWM) standard deviation
    to these returns to estimate volatility.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param lookback: (int) lookback window (default is 100)
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) Periodic volatility values
    """
    ret = get_period_returns(close, **time_delta_kwargs)
    vol = ret.ewm(span=lookback).std()
    return vol


@njit(parallel=True)
def rolling_autocorr_numba(data: np.ndarray, lookback: int) -> np.ndarray:
    """
    Compute rolling autocorrelation for a 1D NumPy array.

    This function calculates the autocorrelation between `data[t]` and `data[t-1]` within a
    rolling window of `lookback` size using Numba for performance.

    :param data: (np.ndarray) A 1D NumPy array of numerical data (e.g., returns).
    :param lookback: (int) The size of the rolling window for autocorrelation calculation.
    return: (np.ndarray) A NumPy array containing the rolling autocorrelation values.
    """
    result = np.full(len(data), np.nan)
    for i in prange(lookback - 1, len(data)):
        window = data[i - lookback + 1 : i + 1]
        # [0, 1] extracts the correlation between the two series (not self-correlation)
        result[i] = np.corrcoef(window[:-1], window[1:])[0, 1]
    return result


def get_period_autocorr(close: pd.Series, lookback: int = 100, **time_delta_kwargs) -> pd.Series:
    """
    Estimates rolling periodic autocorrelation of closing prices.

    This function first calculates the periodic returns using `get_period_returns`
    and then computes the rolling autocorrelation of these returns using the
    Numba-optimized `rolling_autocorr_numba` function.

    :param close: (pd.Series) closing prices, indexed by datetime
    :param lookback: (int) The window equivalent of the Simple Moving Average for the Exponentially Weighted Moving
                average calculation (default is 100)
    :param time_delta_kwargs: Time components for calculating period returns:
    - **days**: (int) Number of days
    - **hours**: (int) Number of hours
    - **minutes**: (int) Number of minutes
    - **seconds**: (int) Number of seconds
    return: (pd.Series) of rolling periodic autocorrelation values, indexed by the datetime index of the input `close` Series.
    """
    ret = get_period_returns(close, **time_delta_kwargs)
    acorr = rolling_autocorr_numba(ret.to_numpy(), lookback)
    df0 = pd.Series(acorr, index=ret.index)
    return df0


def get_lagged_returns(
    prices: Union[pd.Series, pd.DataFrame],
    lags: list,
    nperiods: int = 3,
) -> pd.DataFrame:
    """
    Compute various lagged returns for a given price series.

    This function calculates returns for specified lag periods and then creates
    additional lagged features.

    :param prices: (pd.Series or pd.DataFrame) close prices, indexed by datetime
    :param lags: (list) A list of integers, where each integer represents a
                 lag period for which returns should be calculated.
    :param nperiods: (int) The number of additional lagged versions to create for each
                     return series. For example, if `nperiods=3` and `lags=[1]`,
                     it will create `returns_1_lag_1`, `returns_1_lag_2`, `returns_1_lag_3`.
    return: (pd.DataFrame) A pandas DataFrame containing the calculated returns and
            their lagged versions.
    """
    q = 0.0001  # Quantile cut-off for winsorizing extreme prices
    df = pd.DataFrame()

    for lag in lags:
        # Calculate 1-period geometric mean return of the lag period and
        # winsorize extreme values by clipping.
        df[f"returns_{lag}"] = (
            prices.pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))  # winsorize
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )

    # Create additional lagged versions of the calculated returns
    for t in range(1, nperiods + 1):
        for lag in lags:
            df[f"returns_{lag}_lag_{t}"] = df[f"returns_{lag}"].shift(t * lag)

    df.rename(columns={"returns_1": "returns"}, inplace=True)
    return df

def get_serial_correlation_features(close: pd.Series, window=10):
    """Serial correlation features"""
    df = pd.DataFrame(index=close.index)
    ret = np.log(close).diff()
    sma_returns = ret.rolling(window, min_periods=3)
    df["returns_autocorr"] = rolling_autocorr_numba(ret.values, lookback=window)
    for lag in range(1, 6):
        df[f"returns_autocorr_lag_{lag}"] = df["returns_autocorr"].shift(lag)
    return df

def get_return_dist_features(close, window=10):
    """Distribution of return features"""
    df = pd.DataFrame(index=close.index)
    ret = np.log(close).diff()
    sma_returns = ret.rolling(window, min_periods=3)
    df["returns_normalized"] = (ret - sma_returns.mean()) / sma_returns.std()
    df[f"returns_skew"] = sma_returns.skew()
    df[f"returns_kurtosis"] = sma_returns.kurt()
    return df


def get_MA_diffs(close, windows, verbose=False):
    """
    Moving average differences.

    :param close: (pd.Series) Close prices
    :param windows: (list) list of windows to create differences for, e.g. (10, 20, 50)
    return: (pd.DataFrame) A DataFrame containing moving average differences.
    """
    df = pd.DataFrame(index=close.index)
    sma = {window: close.rolling(window, closed="left").mean() for window in windows}

    # Create differences of all unique combinations of windows
    for win in combinations(windows, 2):
        fast_window, slow_window = sorted(win)
        df[f"sma_diff_{fast_window}_{slow_window}"] = sma[fast_window] - sma[slow_window]

    dcf = DropCorrelatedFeatures(threshold=0.8)
    out = dcf.fit_transform(df)
    dropped = df.columns.difference(out.columns).to_list()
    if len(dropped) > 0:
        logger.info(
            f"\nDropped features with correlation > 0.8: \n\t{dropped}"
            f"\nKept features: \n\t{out.columns.to_list()}"
        )
        if verbose:
            corr_matrix = df.corr()
            # Set the figure size for better readability
            plt.figure(figsize=(12, 4))

            # Create the heatmap with the mask
            sns.heatmap(
                corr_matrix,
                cmap="coolwarm",  # Choose a colormap
                linewidths=0.5,  # Add lines to separate the cells
                annot=True,  # Annotate with the correlation values
                fmt=".2f",  # Format the annotations to two decimal places
                cbar_kws={"shrink": 0.8},  # Shrink the color bar
            )

            plt.title("Correlation Matrix")
            plt.show()

    return out


def get_yang_zhang_vol(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret**2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret**2).rolling(window=window).sum()
    sigma_rs_sq = (
        1
        / (window - 1)
        * (high_close_ret * high_open_ret + low_close_ret * low_open_ret)
        .rolling(window=window)
        .sum()
    )

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)


def create_meta_features(data, lookback_window=10, bb_period=20, bb_std=2):
    """
    Create features for meta-labeling model
    """
    df = data.copy()
    features = pd.DataFrame(index=df.index)

    # Calculate spread if bid/ask columns exist
    if "ask_close" in df.columns and "bid_close" in df.columns:
        # For OHLC bars created with price='bid_ask'
        features["rel_spread"] = (df["ask_close"] - df["bid_close"]) / df["close"]
    else:
        # If no bid/ask data available, skip spread feature or approximate with high-low
        logger.warning("No bid/ask data found. Skipping spread feature.")
        # Alternative: approximate with high-low range
        # features["rel_spread"] = (df["high"] - df["low"]) / df["close"]

    # Bollinger features
    bb_feat = df.ta.bbands(bb_period, bb_std)
    features["bb_bandwidth"] = bb_feat.filter(regex="BBB")
    features["bb_percentage"] = bb_feat.filter(regex="BBP")

    # Price-based features
    # NOTE: Returns are lagged so no need to apply shift
    lagged_ret = get_lagged_returns(df.close, lags=[1, 5, 10], nperiods=3)
    features = features.join(lagged_ret)

    features["vol"] = get_yang_zhang_vol(df.open, df.high, df.low, df.close, window=5)
    features[f"vol_{bb_period}"] = get_yang_zhang_vol(
        df.open, df.high, df.low, df.close, window=bb_period
    )
    for t in range(1, 6):
        features[f"vol_lag_{t}"] = features["vol"].shift(t)
    
    features["autocorr"] = rolling_autocorr_numba(
        features["returns"].values, lookback=lookback_window
    )
    for t in range(1, 6):
        features[f"autocorr_{t}"] = features["autocorr"].shift(t)

    for num_hours in (1, 4, 24):
        features[f"H{num_hours}_vol"] = get_period_vol(df.close, lookback=100, hours=num_hours)
    features.columns = features.columns.str.replace("H24", "D1")

    features["returns_skew"] = features["returns"].rolling(lookback_window).skew()
    features["returns_kurt"] = features["returns"].rolling(lookback_window).kurt()

    # Technical indicators
    # Volatility
    features["tr"] = df.ta.true_range()
    features["atr"] = df.ta.atr()

    # Moving average differences
    ma_diffs = get_MA_diffs(df.close, windows=(5, 20, 50, 100))
    ma_diffs = ma_diffs.div(features["atr"], axis=0)  # Normalize by ATR
    features = features.join(ma_diffs)

    # Momentum
    mom_feat = pd.concat((df.ta.mom(10), df.ta.mom(50), df.ta.mom(100)), axis=1)
    mom_feat.columns = mom_feat.columns.str.lower()
    features = features.join(mom_feat)  # Momentum indicators
    features["rsi"] = df.ta.rsi()
    stochrsi = df.ta.stochrsi()
    features["stoch_rsi_k"] = stochrsi.iloc[:, 0]  # Stochastic RSI %K
    features["stoch_rsi_d"] = stochrsi.iloc[:, 1]

    # Trend
    adx = df.ta.adx()  # ADX
    adx.columns = [
        x.split("_")[0].lower() for x in adx.columns
    ]  # Rename columns to match convention
    adx["dm_net"] = adx["dmp"] - adx["dmn"]
    features = features.join(adx)  # Concatenate ADX columns [['adx', 'dm_net']]
    features["macd"], _, features["macd_hist"] = talib.MACD(
        df.close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # ============================================================================
    # TIER 1 FEATURES: Volume-Based & MA Bias
    # ============================================================================
    
    # Volume features (if volume data available)
    if "volume" in df.columns:
        # On-Balance Volume: cumulative volume flow
        features["obv"] = talib.OBV(df.close, df.volume)
        # Normalize OBV by rolling mean to make it stationary
        features["obv_norm"] = features["obv"] / features["obv"].rolling(50).mean()
        
        # Accumulation/Distribution: volume-weighted price momentum
        features["ad"] = talib.AD(df.high, df.low, df.close, df.volume)
        # Normalize AD
        features["ad_norm"] = features["ad"] / features["ad"].rolling(50).mean()
        
        # Accumulation/Distribution Oscillator: fast-slow AD difference
        features["adosc"] = talib.ADOSC(df.high, df.low, df.close, df.volume, 
                                        fastperiod=3, slowperiod=10)
    else:
        logger.warning("No volume data found. Skipping OBV, AD, ADOSC features.")
    
    # MA Bias features: normalized distance from moving average
    # Ratio-based (scale-invariant) vs difference-based
   # for period in [5, 10, 20, 60]:
    #    sma = df.close.rolling(period).mean()
     #   features[f"bias_{period}"] = df.close / sma
        
    # ============================================================================
    # TIER 2 FEATURES: Stochastic & Acceleration
    # ============================================================================
    """
    # Classic Stochastic Oscillator (different from Stochastic RSI)
    # Uses high/low ranges instead of close prices
    stoch = df.ta.stoch(fastk_period=14, slowk_period=3, slowd_period=3)
    if stoch is not None and len(stoch.columns) >= 2:
        features["stoch_k"] = stoch.iloc[:, 0]  # %K fast
        features["stoch_d"] = stoch.iloc[:, 1]  # %D slow
        features["stoch_divergence"] = features["stoch_k"] - features["stoch_d"]
    
    # Acceleration features: rate of change of momentum
    # Measures if momentum is accelerating or decelerating
    for period in [5, 10, 20, 60]:
        # acc(n) = close.shift(n) / (close.shift(2*n) + close) * 2
        # This measures acceleration by comparing recent vs older momentum
        numerator = df.close.shift(period)
        denominator = df.close.shift(2 * period) + df.close.shift(1)  # Shift current close!
        features[f"acc_{period}"] = (numerator / denominator) * 2
    """
    # ============================================================================
    # LOOKAHEAD PREVENTION
    # ============================================================================
    
    # Features that are ALREADY lagged (don't shift these):
    # - All 'returns_*_lag_*' columns from get_lagged_returns()
    # - 'vol_lag_*' columns (already shifted)
    # - 'autocorr_*' columns (already shifted, except base 'autocorr')
    already_lagged = [col for col in features.columns if '_lag_' in col]
    
    # Shift all features by 1 period to prevent lookahead, EXCEPT already-lagged ones
    for col in features.columns:
        if col not in already_lagged:
            features[col] = features[col].shift(1)
    
    # Handle inf and NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features
