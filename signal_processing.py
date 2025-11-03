import pandas as pd
from typing import Tuple, Union
import logging

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import talib
from loguru import logger
from filters import cusum_filter
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

class BollingerMeanReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""

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

def get_entries(
    strategy: 'BaseStrategy',
    data: pd.DataFrame,
    filter_events: bool = False,
    filter_threshold: Union[float, pd.Series] = None,
    on_crossover: bool = True,
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """Get timestamps and position information for entry events.

    This function processes signals from a given `BaseStrategy` to identify trade
    entry points. It can apply a CUSUM filter to isolate significant events or,
    by default, detect entries at signal crossover points.

    Args:
        strategy (BaseStrategy): The trading strategy object that generates the
            primary signals.
        data (pd.DataFrame): A pandas DataFrame containing the input data, expected
            to have a 'close' column if `filter_events` is True.
        filter_events (bool, optional): If True, a CUSUM filter is applied to the
            signals to identify significant events. Defaults to False.
        filter_threshold (Union[float, pd.Series], optional): The threshold for the
            CUSUM filter. Must be a float or a pandas Series. Defaults to None.
        on_crossover (bool, optional): If True, only events where the signal changes
            from the previous period are considered entry points. Defaults to True.
            Works in combination with filter_events when both are True.

    Raises:
        ValueError: If `filter_events` is True and `filter_threshold` is not a
            `float` or `pd.Series`.
            
    Returns:
        Tuple[pd.Series, pd.DatetimeIndex]: A tuple containing:
            side (pd.Series): A Series with the same index as the input data,
                where each value represents the trading position (-1 for short,
                1 for long, 0 for no position).
            t_events (pd.DatetimeIndex): A DatetimeIndex of the timestamps for
                each detected entry event.
                
    Notes:
        When both filter_events=True and on_crossover=True:
        - For sparse signals (e.g., Bollinger Band touches): Returns CUSUM events where signals exist
        - For dense signals (e.g., MA crossover always in position): Returns crossovers near CUSUM events
        - This ensures meaningful event filtering for any strategy type
    """
    primary_signals = strategy.generate_signals(data)
    signal_mask = primary_signals != 0

    # Step 1: Apply crossover detection if requested
    if on_crossover:
        crossover_mask = primary_signals != primary_signals.shift()
        signal_mask &= crossover_mask
    
    # Step 2: Apply CUSUM filter if requested
    if filter_events:
        try:
            close = data.close
        except AttributeError as e:
            logger.error(f"Dataframe must have a 'close' column: {e}")
            raise e

        if not isinstance(filter_threshold, (pd.Series, float)):
            raise ValueError("filter_threshold must be a Series or a float")
        elif isinstance(filter_threshold, pd.Series):
            filter_threshold = filter_threshold.copy().dropna()
            close = close.reindex(filter_threshold.index)

        # Get CUSUM-filtered event timestamps
        filtered_events = cusum_filter(close, filter_threshold)
        
        # Combine with signal mask: keep events that pass both filters
        signal_mask &= primary_signals.index.isin(filtered_events)

    t_events = primary_signals.index[signal_mask]

    side = pd.Series(index=data.index, name="side")
    side.loc[t_events] = primary_signals.loc[t_events]
    side = side.ffill().fillna(0).astype("int8")

    # Generate descriptive message
    if filter_events and on_crossover:
        s = " generated by crossover + CUSUM filter"
    elif filter_events:
        s = " generated by CUSUM filter"
    elif on_crossover:
        s = " generated by crossover"
    else:
        s = " (all non-zero signals)"

    logger.info(f"Generated {len(t_events):,} trade events{s}.")

    return side, t_events