from abc import ABC, abstractmethod
from os import close

import pandas as pd
import talib


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1 for long, -1 for short, 0 for no position)"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        pass

    @abstractmethod
    def get_objective(self) -> str:
        """Return strategy objective"""
        pass


class BollingerStrategy(BaseStrategy):
    """
    BollingerStrategy implements a mean reversion trading strategy using Bollinger Bands.
    Attributes:
        window (int): The lookback period for calculating Bollinger Bands.
        num_std (float): The number of standard deviations for the bands.
        objective (str): The strategy objective, default is "mean_reversion".
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on Bollinger Bands. Returns a Series where
            1 indicates a buy signal (price below lower band), -1 indicates a sell signal
            (price above upper band), and 0 indicates no signal.
        get_strategy_name() -> str:
            Returns the name of the strategy including window and standard deviation parameters.
        get_objective() -> str:
            Returns the objective of the strategy.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, objective: str = "mean_reversion"):
        self.window = window
        self.num_std = num_std
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals using Bollinger Bands"""
        close = data["close"]

        # Calculate Bollinger Bands
        upper_band, _, lower_band = talib.BBANDS(
            close, timeperiod=self.window, nbdevup=self.num_std, nbdevdn=self.num_std
        )

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="side")
        signals[(close >= upper_band)] = -1  # Sell signal (mean reversion)
        signals[(close <= lower_band)] = 1  # Buy signal (mean reversion)
        return signals

    def get_strategy_name(self) -> str:
        return f"Bollinger_w{self.window}_std{self.num_std}"

    def get_objective(self) -> str:
        return self.objective


class MACrossoverStrategy(BaseStrategy):
    """
    MACrossoverStrategy implements a moving average crossover trend-following strategy.
    Attributes:
        fast_window (int): Window size for the fast moving average.
        slow_window (int): Window size for the slow moving average.
        objective (str): The objective of the strategy (default: "trend_following").
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Generates trading signals based on the crossover of fast and slow moving averages.
            Returns a Series with values: 1 for long, -1 for short, and 0 for neutral.
        get_strategy_name() -> str:
            Returns the name of the strategy, including the fast and slow window sizes.
        get_objective() -> str:
            Returns the objective of the strategy.
    """

    def __init__(
        self, fast_window: int = 10, slow_window: int = 30, objective: str = "trend_following"
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.objective = objective

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals based on MA crossover"""
        close = data["close"]

        # Calculate moving averages
        fast_ma = talib.MA(close, self.fast_window)
        slow_ma = talib.MA(close, self.slow_window)

        # Generate signals
        signals = pd.Series(0, index=data.index, dtype="int8", name="side")
        signals[(fast_ma > slow_ma)] = 1  # Long signal when fast MA crosses above slow MA
        signals[(fast_ma < slow_ma)] = -1  # Short signal when fast MA crosses below slow MA
        return signals

    def get_strategy_name(self) -> str:
        return f"MACrossover_{self.fast_window}_{self.slow_window}"

    def get_objective(self) -> str:
        return self.objective

class BollingerMACrossoverStrategy:
    """
    Combines Bollinger Bands and Moving Average signals for multi-class classification.
    
    NOTE: This strategy does NOT inherit from BaseStrategy because it returns 
    signal types 0-6 for ML classification, not standard 1/-1/0 trading signals.
    
    Attributes:
        window (int): Bollinger Bands period (default 20)
        num_std (float): Number of standard deviations for bands (default 2.0)
        ma_window (int): Moving average period (default 20)
        objective (str): Strategy objective (default "combined")
    
    Methods:
        generate_signals(data: pd.DataFrame) -> pd.Series:
            Returns signal types 0-6:
            0 = No signal
            1 = Bullish bounce from lower Bollinger Band
            2 = Bearish rejection at upper Bollinger Band
            3 = Continuation at upper band (high resistance)
            4 = Continuation at lower band (low support)
            5 = Bullish bounce from MA20
            6 = Bearish rejection at MA20
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, ma_window: int = 20, objective: str = "combined"):
        self.window = window
        self.num_std = num_std
        self.ma_window = ma_window
        self.objective = objective


    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate combined signals from both strategies - returns signal types 0-6"""
        close = data["close"]
        low = data["low"]
        high = data["high"]
        high1 = high.shift(1)
        low1 = low.shift(1)
        close1 = close.shift(1)
        
        # Calculate Bollinger Bands
        upper_band, _, lower_band = talib.BBANDS(
            close, timeperiod=self.window, nbdevup=self.num_std, nbdevdn=self.num_std
        )
        MA20 = talib.MA(close, timeperiod=self.ma_window)
        
        # Define conditions (matching training_data.py exactly)
        con1 = (close >= lower_band) & (low1 <= lower_band) & (close > close1)
        con2 = (close <= upper_band) & (high1 >= upper_band) & (close1 > close)
        con3 = (high >= upper_band) & (high1 >= upper_band) & (close > close1)
        con4 = (low <= lower_band) & (low1 <= lower_band) & (close1 > close)
        con5 = (close >= MA20) & (low1 <= MA20) & (close > close1)
        con6 = (close <= MA20) & (high1 >= MA20) & (close1 > close)

        # Generate signals (0 = no signal, 1-6 = specific signal types)
        signals = pd.Series(0, index=data.index, dtype="int8", name="side")
        signals.loc[con1] = 1
        signals.loc[con2] = 2
        signals.loc[con3] = 3
        signals.loc[con4] = 4
        signals.loc[con5] = 5
        signals.loc[con6] = 6
        
        return signals


    def get_strategy_name(self) -> str:
        return f"BollingerMA_Crossover"

    def get_objective(self) -> str:
        return self.objective
 