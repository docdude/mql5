"""
AAPL Historical Tick Data Analysis and Bar Creation

This script demonstrates how to:
1. Connect to MT5 and download AAPL tick data
2. Create multiple bar types (time, tick, volume, dollar)
3. Analyze and visualize the results
4. Save processed data for further analysis

Usage:
    python aapl_bars_analysis.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Import functions from bars.py
from bars import (
    get_ticks,
    clean_tick_data,
    make_bars,
    plot_volatility_analysis_of_bars,
    set_resampling_freq
)

# Import MT5 login
from mt5_login import login_mt5, mt5_logged_in

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def check_symbol_availability(symbol: str) -> bool:
    """
    Check if a symbol is available in MT5.
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        bool: True if symbol is available, False otherwise
    """
    terminal_info = mt5.terminal_info()
    if terminal_info is None or not terminal_info.connected:
        logging.error("MT5 not logged in. Please login first.")
        return False
    
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        logging.error(f"Symbol {symbol} not found")
        return False
    
    if not symbol_info.visible:
        logging.info(f"Symbol {symbol} is not visible, attempting to enable...")
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to enable symbol {symbol}")
            return False
        logging.info(f"Symbol {symbol} enabled successfully")
    
    logging.info(f"Symbol: {symbol}")
    logging.info(f"  Description: {symbol_info.description}")
    logging.info(f"  Currency: {symbol_info.currency_base}/{symbol_info.currency_profit}")
    logging.info(f"  Digits: {symbol_info.digits}")
    logging.info(f"  Point: {symbol_info.point}")
    logging.info(f"  Trade Mode: {symbol_info.trade_mode}")
    
    return True


def download_tick_data(start_date: str, end_date: str, symbol: str = "AAPL") -> pd.DataFrame:
    """
    Download tick data from MT5.
    
    Args:
        start_date: Start date (YYYY-MM-DD format or datetime)
        end_date: End date (YYYY-MM-DD format or datetime)
        symbol: Trading symbol (default: AAPL)
        
    Returns:
        pd.DataFrame: Cleaned tick data
    """
    logging.info(f"Downloading {symbol} tick data from {start_date} to {end_date}")
    
    # Check if symbol is available
    if not check_symbol_availability(symbol):
        return None
    
    # Download ticks
    try:
        tick_df = get_ticks(symbol, start_date, end_date)
        
        if tick_df is None or tick_df.empty:
            logging.error("No tick data received")
            return None
        
        logging.info(f"Downloaded {len(tick_df):,} ticks")
        
        # Clean the data
        logging.info("Cleaning tick data...")
        tick_df = clean_tick_data(tick_df, n_digits=5,timezone='UTC')
        
        if tick_df is None or tick_df.empty:
            logging.error("No data remaining after cleaning")
            return None
        
        logging.info(f"Cleaned data: {len(tick_df):,} valid ticks")
        
        return tick_df
        
    except Exception as e:
        logging.exception(f"Error downloading tick data: {e}")
        return None


def create_all_bar_types(tick_df: pd.DataFrame, timeframe: str = 'M5', 
                         tick_bar_size: int = 0, volume_bar_size: int = 0,
                         dollar_bar_size: int = 0) -> dict:
    """
    Create all bar types from tick data.
    
    Args:
        tick_df: DataFrame with tick data
        timeframe: Timeframe for time-based bars (e.g., 'M1', 'M5', 'H1')
        tick_bar_size: Size for tick bars (0 = auto-calculate)
        volume_bar_size: Size for volume bars (0 = manual specification needed)
        dollar_bar_size: Size for dollar bars (0 = manual specification needed)
        
    Returns:
        dict: Dictionary containing all bar types
    """
    bars_dict = {}
    
    logging.info("=" * 60)
    logging.info("Creating Time Bars")
    logging.info("=" * 60)
    try:
        time_bars = make_bars(tick_df, bar_type='time', timeframe=timeframe, verbose=True)
        bars_dict['time'] = time_bars
        logging.info(f"Created {len(time_bars):,} time bars ({timeframe})")
    except Exception as e:
        logging.error(f"Failed to create time bars: {e}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Creating Tick Bars")
    logging.info("=" * 60)
    try:
        tick_bars = make_bars(tick_df, bar_type='tick', bar_size=tick_bar_size, 
                              timeframe=timeframe, verbose=True)
        bars_dict['tick'] = tick_bars
        logging.info(f"Created {len(tick_bars):,} tick bars")
    except Exception as e:
        logging.error(f"Failed to create tick bars: {e}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Creating Volume Bars")
    logging.info("=" * 60)
    try:
        # Check if volume data exists
        if 'volume' not in tick_df.columns or tick_df['volume'].sum() == 0:
            logging.warning("No volume data available in tick data.")
            logging.warning("Volume bars require actual traded volume (not available for forex).")
            logging.warning("Skipping volume bars.")
        elif volume_bar_size == 0:
            logging.warning("Volume bar size not specified. Skipping volume bars.")
            logging.warning("Set volume_bar_size to a positive value (e.g., 1000, 5000)")
        else:
            volume_bars = make_bars(tick_df, bar_type='volume', bar_size=volume_bar_size, 
                                   timeframe=timeframe, verbose=True)
            bars_dict['volume'] = volume_bars
            logging.info(f"Created {len(volume_bars):,} volume bars")
    except KeyError as e:
        logging.warning(f"Volume column not found: {e}. Forex symbols don't have real volume data.")
    except Exception as e:
        logging.error(f"Failed to create volume bars: {e}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Creating Dollar Bars")
    logging.info("=" * 60)
    try:
        # Check if volume data exists
        if 'volume' not in tick_df.columns or tick_df['volume'].sum() == 0:
            logging.warning("No volume data available in tick data.")
            logging.warning("Dollar bars require actual traded volume (not available for forex).")
            logging.warning("Skipping dollar bars.")
        elif dollar_bar_size == 0:
            logging.warning("Dollar bar size not specified. Skipping dollar bars.")
            logging.warning("Set dollar_bar_size to a positive value (e.g., 100000, 500000)")
        else:
            dollar_bars = make_bars(tick_df, bar_type='dollar', bar_size=dollar_bar_size, 
                                   timeframe=timeframe, verbose=True)
            bars_dict['dollar'] = dollar_bars
            logging.info(f"Created {len(dollar_bars):,} dollar bars")
    except KeyError as e:
        logging.warning(f"Volume column not found: {e}. Forex symbols don't have real volume data.")
    except Exception as e:
        logging.error(f"Failed to create dollar bars: {e}")
    
    return bars_dict


def save_bars_to_csv(bars_dict: dict, symbol: str, output_dir: str = "data"):
    """
    Save all bar types to CSV files.
    
    Args:
        bars_dict: Dictionary of bar DataFrames
        symbol: Trading symbol
        output_dir: Output directory for CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for bar_type, df in bars_dict.items():
        if df is not None and not df.empty:
            filename = output_path / f"{symbol}_{bar_type}_bars_{timestamp}.csv"
            df.to_csv(filename)
            logging.info(f"Saved {bar_type} bars to {filename}")


def analyze_bars(bars_dict: dict, symbol: str):
    """
    Analyze and compare different bar types.
    
    Args:
        bars_dict: Dictionary of bar DataFrames
        symbol: Trading symbol
    """
    logging.info("\n" + "=" * 60)
    logging.info("Bar Type Analysis")
    logging.info("=" * 60)
    
    results = []
    for bar_type, df in bars_dict.items():
        if df is not None and not df.empty:
            returns = (df['close'] / df['open'] - 1) * 100
            
            analysis = {
                'Bar Type': bar_type,
                'Count': len(df),
                'Mean Return (%)': returns.mean(),
                'Std Return (%)': returns.std(),
                'Min Return (%)': returns.min(),
                'Max Return (%)': returns.max(),
                'Mean Volume': df['volume'].mean() if 'volume' in df.columns else np.nan,
                'Mean Tick Volume': df['tick_volume'].mean() if 'tick_volume' in df.columns else np.nan
            }
            results.append(analysis)
    
    analysis_df = pd.DataFrame(results)
    logging.info(f"\n{analysis_df.to_string(index=False)}")
    
    return analysis_df


def main():
    """Main execution function."""
    
    # Configuration
    SYMBOL = "EURUSD"
    ACCOUNT = "demo"  # Change to your account name
    
    # Date range (adjust as needed)
    # Note: MT5 demo brokers typically only have tick data for the last 1-7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)  # Last 1 day
    
    # Bar configuration
    TIMEFRAME = "M5"  # 5-minute time bars
    TICK_BAR_SIZE = 0  # 0 = auto-calculate based on timeframe
    
    # NOTE: Volume and Dollar bars require actual traded volume data
    # Forex symbols (like EURUSD) only have "tick volume" (price changes), not real volume
    # These will be skipped automatically for forex. They work for stocks/futures with volume data.
    VOLUME_BAR_SIZE = 0  # Set to 0 to skip (forex doesn't have real volume)
    DOLLAR_BAR_SIZE = 0  # Set to 0 to skip (forex doesn't have real volume)
    
    logging.info("=" * 60)
    logging.info("AAPL Tick Data Analysis and Bar Creation")
    logging.info("=" * 60)
    
    # Step 1: Login to MT5
    logging.info("\nStep 1: Connecting to MT5...")
    try:
        login_mt5(ACCOUNT, verbose=True)
        # Check actual connection status
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            logging.error("Failed to login to MT5. Exiting.")
            return
    except Exception as e:
        logging.exception(f"MT5 login error: {e}")
        return
    
    # Step 2: Download tick data
    logging.info(f"\nStep 2: Downloading {SYMBOL} tick data...")
    tick_df = download_tick_data(start_date, end_date, SYMBOL)
    
    if tick_df is None or tick_df.empty:
        logging.error("Failed to get tick data. Exiting.")
        mt5.shutdown()
        return
    
    # Display tick data info
    logging.info(f"\nTick Data Summary:")
    logging.info(f"  Start: {tick_df.index[0]}")
    logging.info(f"  End: {tick_df.index[-1]}")
    logging.info(f"  Total Ticks: {len(tick_df):,}")
    logging.info(f"  Columns: {list(tick_df.columns)}")
    logging.info(f"\n{tick_df.head()}")
    
    # Step 3: Create all bar types
    logging.info(f"\nStep 3: Creating bars...")
    bars_dict = create_all_bar_types(
        tick_df, 
        timeframe=TIMEFRAME,
        tick_bar_size=TICK_BAR_SIZE,
        volume_bar_size=VOLUME_BAR_SIZE,
        dollar_bar_size=DOLLAR_BAR_SIZE
    )
    
    # Step 4: Analyze bars
    logging.info(f"\nStep 4: Analyzing bars...")
    analysis_df = analyze_bars(bars_dict, SYMBOL)
    
    # Step 5: Save to CSV
    logging.info(f"\nStep 5: Saving bars to CSV...")
    save_bars_to_csv(bars_dict, SYMBOL)
    
    # Step 6: Create volatility plots (optional)
    logging.info(f"\nStep 6: Creating volatility plots...")
    try:
        for bar_type, df in bars_dict.items():
            if df is not None and not df.empty and len(df) > 10:
                logging.info(f"Creating volatility plot for {bar_type} bars...")
                fig = plot_volatility_analysis_of_bars(
                    df, 
                    SYMBOL, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    f"{bar_type}_{TIMEFRAME}",
                    thres=0.01,
                    bins=50
                )
                
                # Save plot
                output_path = Path("plots")
                output_path.mkdir(exist_ok=True)
                plot_file = output_path / f"{SYMBOL}_{bar_type}_volatility.html"
                fig.write_html(str(plot_file))
                logging.info(f"Saved volatility plot to {plot_file}")
    except Exception as e:
        logging.error(f"Error creating plots: {e}")
    
    # Cleanup
    mt5.shutdown()
    logging.info("\n" + "=" * 60)
    logging.info("Analysis Complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
