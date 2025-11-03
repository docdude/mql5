
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import logging
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Step 1: Data Extraction

def get_ticks(symbol, start_date, end_date):
    """
    Downloads tick data from the MT5 terminal.

    Args:
        symbol (str): Financial instrument (e.g., currency pair or stock).
        start_date, end_date (str or datetime): Time range for data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Tick data with a datetime index.
    """
    from tqdm import tqdm
    
    if not mt5.initialize():
        logging.error("MT5 connection not established.")
        raise RuntimeError("MT5 connection error.")

    start_date = pd.Timestamp(start_date, tz='UTC') if isinstance(start_date, str) else (
        start_date if start_date.tzinfo is not None else pd.Timestamp(start_date, tz='UTC')
    )
    end_date = pd.Timestamp(end_date, tz='UTC') if isinstance(end_date, str) else (
        end_date if end_date.tzinfo is not None else pd.Timestamp(end_date, tz='UTC')
    )

    try:
        with tqdm(total=5, desc=f"Downloading {symbol} ticks", unit="step") as pbar:
            # Step 1: Download ticks from MT5
            pbar.set_description(f"Fetching {symbol} ticks from MT5")
            ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
            pbar.update(1)
            
            if ticks is None or len(ticks) == 0:
                logging.warning(f"No ticks returned for {symbol}")
                return None
            
            # Step 2: Create DataFrame
            pbar.set_description("Creating DataFrame")
            df = pd.DataFrame(ticks)
            pbar.update(1)
            
            # Step 3: Convert timestamps
            pbar.set_description("Converting timestamps")
            df['time'] = pd.to_datetime(df['time_msc'], unit='ms')
            pbar.update(1)
            
            # Step 4: Set index and clean
            pbar.set_description("Setting index")
            df.set_index('time', inplace=True)
            df.drop('time_msc', axis=1, inplace=True)
            pbar.update(1)
            
            # Step 5: Remove empty columns
            pbar.set_description("Removing empty columns")
            df = df[df.columns[df.any()]]
            pbar.update(1)
            
            pbar.set_description(f"✓ Downloaded {len(df):,} ticks")
        
        df.info()
    except Exception as e:
        logging.error(f"Error while downloading ticks: {e}")
        return None

    return df


# Step 2: Data Cleaning

def clean_tick_data(df: pd.DataFrame,
                    n_digits: int,
                    timezone: str = 'UTC'
                    ):
    """
    Clean and validate Forex tick data with comprehensive quality checks.

    Args:
        df: DataFrame containing tick data with bid/ask prices and timestamp index
        n_digits: Number of decimal places in instrument price.
        timezone: Timezone to localize/convert timestamps to (default: UTC)

    Returns:
        Cleaned DataFrame or None if empty after cleaning
    """
    if df.empty:
        return None

    df = df.copy(deep=False)  # Work on a copy to avoid modifying the original DataFrame 
    n_initial = df.shape[0] # Store initial row count for reporting

    # 1. Ensure proper datetime index
    # Use errors='coerce' to turn unparseable dates into NaT and then drop them.
    if not isinstance(df.index, pd.DatetimeIndex):
        original_index_name = df.index.name
        df.index = pd.to_datetime(df.index, errors='coerce')
        nan_idx_count = df.index.isnull().sum()
        if nan_idx_count > 0:
            logging.info(f"Dropped {nan_idx_count:,} rows with unparseable timestamps.")
            df = df[~df.index.isnull()]
        if original_index_name:
            df.index.name = original_index_name
    
    if df.empty: # Check if empty after index cleaning
        logging.warning("Warning: DataFrame empty after initial index cleaning")
        return None

    # 2. Timezone handling
    if df.index.tz is None:
        df = df.tz_localize(timezone)
    elif str(df.index.tz) != timezone.upper():
        df = df.tz_convert(timezone)
    
    # 3. Price validity checks
    # Apply rounding and then filtering
    df['bid'] = df['bid'].round(n_digits)
    df['ask'] = df['ask'].round(n_digits)

    # Validate prices
    price_filter = (
        (df['bid'] > 0) &
        (df['ask'] > 0) &
        (df['ask'] > df['bid'])
    )
    
    n_before_price_filter = df.shape[0]
    df = df[price_filter]
    n_filtered_prices = n_before_price_filter - df.shape[0]
    if n_filtered_prices > 0:
        logging.info(f"Filtered {n_filtered_prices:,} ({n_filtered_prices / n_before_price_filter:.2%}) invalid prices.")

    if df.empty: # Check if empty after price cleaning
        logging.warning("Warning: DataFrame empty after price cleaning")
        return None
    
    # Dropping NA values
    initial_rows_before_na = df.shape[0]
    if df.isna().any().any(): # Use .any().any() to check if any NA exists in the whole DF
        na_counts = df.isna().sum()
        na_cols = na_counts[na_counts > 0]
        if not na_cols.empty:
            logging.info(f'Dropped NA values from columns: \n{na_cols}')
            df.dropna(inplace=True)

    n_dropped_na = initial_rows_before_na - df.shape[0]
    if n_dropped_na > 0:
        logging.info(f"Dropped {n_dropped_na:,} ({n_dropped_na / n_before_price_filter:.2%}) rows due to NA values.")

    if df.empty: # Check if empty after NA cleaning
        logging.warning("Warning: DataFrame empty after NA cleaning")
        return None
    
    # 4. Microsecond handling
    if not df.index.microsecond.any():
        logging.warning("Warning: No timestamps with microsecond precision found")
    
    # 5. Duplicate handling
    duplicate_mask = df.index.duplicated(keep='last')
    dup_count = duplicate_mask.sum()
    if dup_count > 0:
        logging.info(f"Removed {dup_count:,} ({dup_count / n_before_price_filter:.2%}) duplicate timestamps.")
        df = df[~duplicate_mask]

    if df.empty: # Check if empty after duplicate cleaning
        logging.warning("Warning: DataFrame empty after duplicate cleaning")
        return None

    # 6. Chronological order
    if not df.index.is_monotonic_increasing:
        logging.info("Sorting DataFrame by index to ensure chronological order.")
        df.sort_index(inplace=True)

    # 7. Final validation and reporting
    if df.empty:
        logging.warning("Warning: DataFrame empty after all cleaning steps.")
        return None
    
    n_final = df.shape[0]
    n_cleaned = n_initial - n_final
    percentage_cleaned = (n_cleaned / n_initial) if n_initial > 0 else 0
    logging.info(f"Cleaned {n_cleaned:,} of {n_initial:,} ({percentage_cleaned:.2%}) datapoints.")

    return df


# Step 3: Create Bars and Convert to End-Time

## Resampling Frequency Conversion

def set_resampling_freq(timeframe: str) -> str:
    """
    Converts an MT5 timeframe to a pandas resampling frequency.

    Args:
        timeframe (str): MT5 timeframe (e.g., 'M1', 'H1', 'D1', 'W1').

    Returns:
        str: Pandas frequency string.
    """
    timeframe = timeframe.upper()
    nums = [x for x in timeframe if x.isnumeric()]
    if not nums:
        raise ValueError("Timeframe must include numeric values (e.g., 'M1').")
    
    x = int(''.join(nums))
    if timeframe == 'W1':
        freq = 'W-FRI'
    elif timeframe == 'D1':
        freq = 'B'
    elif timeframe.startswith('H'):
        freq = f'{x}H'
    elif timeframe.startswith('M'):
        freq = f'{x}min'
    elif timeframe.startswith('S'):
        freq = f'{x}S'
    else:
        raise ValueError("Valid timeframes include W1, D1, Hx, Mx, Sx.")
    
    return freq


def calculate_ticks_per_period(df: pd.DataFrame, timeframe: str = "M1", method: str = 'median', verbose: bool = True) -> int:
    """
    Dynamically calculates the average number of ticks per given timeframe.

    Args:
        df (pd.DataFrame): Tick data.
        timeframe (str): MT5 timeframe.
        method (str): 'median' or 'mean' for the calculation.
        verbose (bool): Whether to print the result.

    Returns:
        int: Rounded average ticks per period.
    """
    freq = set_resampling_freq(timeframe)
    resampled = df.resample(freq).size()
    fn = getattr(np, method)
    num_ticks = fn(resampled.values)
    num_rounded = int(np.round(num_ticks))
    num_digits = len(str(num_rounded)) - 1
    rounded_ticks = int(round(num_rounded, -num_digits))
    rounded_ticks = max(1, rounded_ticks)
    
    if verbose:
        t0 = df.index[0].date()
        t1 = df.index[-1].date()
        logging.info(f"From {t0} to {t1}, {method} ticks per {timeframe}: {num_ticks:,} rounded to {rounded_ticks:,}")
    
    return rounded_ticks


## Grouping and Bar Creation

def flatten_column_names(df):
    '''
    Joins tuples created by dataframe aggregation 
    with a list of functions into a unified name.
    '''
    return ["_".join(map(str, col)).strip() for col in df.columns.values]


def make_bar_type_grouper(
        df: pd.DataFrame,
        bar_type: str = 'tick',
        bar_size: int = 100,
        timeframe: str = 'M1'
):
    """
    Create a grouped object for aggregating tick data into time/tick/dollar/volume bars.

    Args:
        df: DataFrame with tick data (index should be datetime for time bars).
        bar_type: Type of bar ('time', 'tick', 'dollar', 'volume').
        bar_size: Number of ticks/dollars/volume per bar (ignored for time bars).
        timeframe: Timeframe for resampling (e.g., 'H1', 'D1', 'W1').

    Returns:
        - GroupBy object for aggregation
        - Calculated bar_size (for tick/dollar/volume bars)
    """
    # Create working copy (shallow is sufficient)
    df = df.copy(deep=False)  # OPTIMIZATION: Shallow copy here only once
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index('time')
        except KeyError:
            raise TypeError("Could not set 'time' as index")

    # Sort if needed
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Time bars
    if bar_type == 'time':
        freq = set_resampling_freq(timeframe)
        bar_group = (df.resample(freq, closed='left', label='right') # includes data upto, but not including, the end of the period
                    if not freq.startswith(('B', 'W')) 
                    else df.resample(freq))
        return bar_group, 0  # bar_size not used

    # Dynamic bar sizing
    if bar_size == 0:
        if bar_type == 'tick':
            bar_size = calculate_ticks_per_period(df, timeframe)
        else:
            raise NotImplementedError(f"{bar_type} bars require non-zero bar_size")

    # Non-time bars
    df['time'] = df.index  # Add without copying
    
    if bar_type == 'tick':
        bar_id = np.arange(len(df)) // bar_size
    elif bar_type in ('volume', 'dollar'):
        if 'volume' not in df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        
        # Optimized cumulative sum
        cum_metric = (df['volume'] * df['bid'] if bar_type == 'dollar' 
                      else df['volume'])
        cumsum = cum_metric.cumsum()
        bar_id = (cumsum // bar_size).astype(int)
    else:
        raise NotImplementedError(f"{bar_type} bars not implemented")

    return df.groupby(bar_id), bar_size


def make_bars(tick_df: pd.DataFrame,
              bar_type: str = 'tick',
              bar_size: int = 0,
              timeframe: str = 'M1',
              price: str = 'midprice',
              verbose=True):
    '''
    Create OHLC data by sampling ticks using timeframe or a threshold.

    Parameters
    ----------
    tick_df: pd.DataFrame
        tick data
    bar_type: str
        type of bars to create from ['tick', 'time', 'volume', 'dollar']
    bar_size: int 
        default 0. bar_size when bar_type != 'time'
    timeframe: str
        MT5 timeframe (e.g., 'M5', 'H1', 'D1', 'W1').
        Used for time bars, or for tick bars if bar_size = 0.
    price: str
        default midprice. If 'bid_ask', columns (bid_open, ..., bid_close), 
        (ask_open, ..., ask_close) are included.
    verbose: bool
        print information about the data

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, median_price, tick_volume, volume]
    '''    
    if 'midprice' not in tick_df:
        tick_df['midprice'] = (tick_df['bid'] + tick_df['ask']) / 2

    bar_group, bar_size_ = make_bar_type_grouper(tick_df, bar_type, bar_size, timeframe)
    ohlc_df = bar_group['midprice'].ohlc().astype('float64')
    ohlc_df['tick_volume'] = bar_group['bid'].count() if bar_type != 'tick' else bar_size_
    
    if price == 'bid_ask':
        # Aggregate OHLC data for every bar_size rows
        bid_ask_df = bar_group.agg({k: 'ohlc' for k in ('bid', 'ask')})
        # Flatten MultiIndex columns
        col_names = flatten_column_names(bid_ask_df)
        bid_ask_df.columns = col_names
        ohlc_df = ohlc_df.join(bid_ask_df)

    if 'volume' in tick_df:
        ohlc_df['volume'] = bar_group['volume'].sum()

    if bar_type == 'time':
        ohlc_df.ffill(inplace=True)
    else:
        end_time =  bar_group['time'].last()
        ohlc_df.index = end_time + pd.Timedelta(microseconds=1) # ensure end time is after event
        # Note: 'time' column only exists in the grouped df, not in ohlc_df, so no need to drop it

        # drop last bar due to insufficient ticks
        if len(tick_df) % bar_size_ > 0: 
            ohlc_df = ohlc_df.iloc[:-1]

    if verbose:
        if bar_type != 'time':
            tm = f'{bar_size_:,}'
            if bar_type == 'tick' and bar_size == 0:
                tm = f'{timeframe} - {bar_size_:,} ticks'
            timeframe = tm
        print(f'\nTick data - {tick_df.shape[0]:,} rows')
        print(f'{bar_type}_bar {timeframe}')
        ohlc_df.info()
    
    # Remove timezone info from DatetimeIndex
    try:
        ohlc_df = ohlc_df.tz_convert(None)
    except:
        pass
    
    return ohlc_df


# Volatility Analysis Plotting

def plot_volatility_analysis_of_bars(df, symbol, start, end, freq, thres=.01, bins=100):
    """
    Plot the volatility analysis of bars using Plotly.
    df: DataFrame containing the data with 'open' and 'close' columns.
    symbol: Symbol of the asset.    
    start: Start date of the data.
    end: End date of the data.
    freq: Frequency of the data.
    thres: Threshold for filtering large values, e.g., 1-.01 for 99th quantile.
    bins: Number of bins for the histogram.
    """
    abs_price_changes = (df['close'] / df['open'] - 1).mul(100).abs()
    thres = abs_price_changes.quantile(1 - thres)
    abs_price_changes = abs_price_changes[abs_price_changes < thres] # filter out large values for visualization

    # Calculate Histogram
    counts, bins = np.histogram(abs_price_changes, bins=bins)
    bins = bins[:-1] # remove the last bin edge

    # Calculate Proportions
    total_counts = len(abs_price_changes)
    proportion_candles_right = []
    proportion_price_change_right = []

    for i in range(len(bins)):
        candles_right = abs_price_changes[abs_price_changes >= bins[i]]
        count_right = len(candles_right)
        proportion_candles_right.append(count_right / total_counts)
        proportion_price_change_right.append(np.sum(candles_right) / np.sum(abs_price_changes))

    fig = go.Figure()

    # Histogram with Hover Template
    fig.add_trace(
        go.Bar(x=bins, y=counts, 
               name='Histogram absolute price change (%)',
               marker=dict(color='#1f77b4'), 
               hovertemplate='<b>Bin: %{x:.2f}</b><br>Frequency: %{y}',  # Custom hover text
               yaxis='y1',
               opacity=.65))

    ms = 3 # marker size
    lw = .5 # line width
    
    # Proportion of Candles at the Right with Hover Text
    fig.add_trace(
        go.Scatter(x=bins, y=proportion_candles_right, 
                   name='Proportion of candles at the right',
                   mode='lines+markers', 
                   marker=dict(color='red', size=ms), 
                   line=dict(width=lw),
                   hovertext=[f"Bin: {x:.2f}, Proportion: {y:.4f}" 
                              for x, y in zip(bins, proportion_candles_right)],  # Hover text list
                   hoverinfo='text',  # Show only the 'text' from hovertext
                   yaxis='y2'))
    

    # Proportion Price Change Produced by Candles at the Right with Hover Text
    fig.add_trace(
        go.Scatter(x=bins, y=proportion_price_change_right, 
                   name='Proportion price change produced by candles at the right',
                   mode='lines+markers', 
                   marker=dict(color='green', size=ms), 
                   line=dict(width=lw),
                   hovertext=[f"Bin: {x:.2f}, Proportion: {y:.4f}" 
                              for x, y in zip(bins, proportion_price_change_right)], # Hover text list
                   hoverinfo='text',  # Show only the 'text' from hovertext
                   yaxis='y2'))
    
    # Indices of proportion_price_change_right at 10% intervals
    search_idx = [.01, .05] + np.linspace(.1, 1., 10).tolist()
    price_idxs = np.searchsorted(sorted(proportion_candles_right), search_idx, side='right')
    for ix in price_idxs:  # Add annotations for every step-th data point as an example
        x = bins[-ix]
        y = proportion_candles_right[-ix]
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{y:.4f}",  # Display the proportion value with 4 decimal points
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-15,  # Offset for the annotation text
            font=dict(color="salmon"),
            arrowcolor="red",
            yref='y2'
        )

        y = proportion_price_change_right[-ix]
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{y:.4f}",  # Display the proportion value with 4 decimal points
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-25,  # Offset for the annotation text
            font=dict(color="lightgreen"),
            arrowcolor="green",
            yref='y2'
        )

    # Layout Configuration with Legend Inside
    fig.update_layout(
        title=f'Volatility Analysis of {symbol} {freq} from {start} to {end}',
        xaxis_title='Absolute price change (%)',
        yaxis_title='Frequency',
        yaxis2=dict(
            title='Proportion',
            overlaying='y',
            side='right',
            gridcolor='#444'  # Set grid color for the secondary y-axis
        ),
        plot_bgcolor='#222',  # Dark gray background
        paper_bgcolor='#222',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#444'),  # Set grid color for the primary x-axis
        yaxis=dict(gridcolor='#444'),   # Set grid color for the primary y-axis
        legend=dict(
            x=0.3,  # Adjust x coordinate (0 to 1)
            y=0.95,  # Adjust y coordinate (0 to 1)
            traceorder="normal",  # Optional: maintain trace order
            font=dict(color="white")  # Optional: set legend text color
        ),
        # width=750,  # Set width of the figure
        # height=480,  # Set height of the figure
    )

    return fig

