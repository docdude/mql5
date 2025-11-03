"""
Utilities for loading saved bar data into pandas DataFrames.

This module provides functions to load previously processed bar data
from CSV files saved by aapl_bars_analysis.py or similar scripts.

Usage:
    from load_data import load_bars, load_all_bars
    
    # Load specific bar type
    time_bars = load_bars('EURUSD', 'time', date='20251101_120000')
    
    # Load all bar types for a symbol
    bars_dict = load_all_bars('EURUSD', date='20251101_120000')
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_bars(
    symbol: str,
    bar_type: str,
    date: Optional[str] = None,
    data_dir: str = "data"
) -> Optional[pd.DataFrame]:
    """
    Load bar data from CSV file.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'AAPL', 'SPY')
        bar_type: Type of bars ('time', 'tick', 'volume', 'dollar')
        date: Optional date string in format YYYYMMDD_HHMMSS
              If None, loads the most recent file
        data_dir: Directory where CSV files are stored
        
    Returns:
        pd.DataFrame: Bar data with datetime index, or None if not found
        
    Example:
        >>> df = load_bars('EURUSD', 'time', date='20251101_120000')
        >>> df = load_bars('EURUSD', 'tick')  # Load most recent
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"Data directory not found: {data_path}")
        return None
    
    # Build filename pattern
    if date:
        pattern = f"{symbol}_{bar_type}_bars_{date}.csv"
        files = list(data_path.glob(pattern))
    else:
        # Find most recent file
        pattern = f"{symbol}_{bar_type}_bars_*.csv"
        files = sorted(data_path.glob(pattern), reverse=True)
    
    if not files:
        logging.error(f"No files found matching pattern: {pattern}")
        return None
    
    filepath = files[0]
    logging.info(f"Loading {bar_type} bars from: {filepath.name}")
    
    try:
        # Load CSV with proper datetime parsing
        df = pd.read_csv(
            filepath,
            parse_dates=['time'],
            index_col='time'
        )
        
        logging.info(f"Loaded {len(df):,} {bar_type} bars")
        logging.info(f"  Start: {df.index[0]}")
        logging.info(f"  End: {df.index[-1]}")
        logging.info(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None


def load_all_bars(
    symbol: str,
    date: Optional[str] = None,
    data_dir: str = "data",
    bar_types: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all available bar types for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'AAPL')
        date: Optional date string in format YYYYMMDD_HHMMSS
              If None, loads the most recent files
        data_dir: Directory where CSV files are stored
        bar_types: List of bar types to load. If None, loads all available.
                   Options: ['time', 'tick', 'volume', 'dollar']
        
    Returns:
        dict: Dictionary with bar_type as key and DataFrame as value
        
    Example:
        >>> bars = load_all_bars('EURUSD')
        >>> time_bars = bars['time']
        >>> tick_bars = bars['tick']
    """
    if bar_types is None:
        bar_types = ['time', 'tick', 'volume', 'dollar']
    
    bars_dict = {}
    
    logging.info("=" * 60)
    logging.info(f"Loading {symbol} bars from {data_dir}/")
    logging.info("=" * 60)
    
    for bar_type in bar_types:
        df = load_bars(symbol, bar_type, date=date, data_dir=data_dir)
        if df is not None:
            bars_dict[bar_type] = df
    
    if not bars_dict:
        logging.warning(f"No bar data found for {symbol}")
    else:
        logging.info(f"\nSuccessfully loaded {len(bars_dict)} bar types: {list(bars_dict.keys())}")
    
    return bars_dict


def list_available_data(symbol: Optional[str] = None, data_dir: str = "data") -> pd.DataFrame:
    """
    List all available bar data files.
    
    Args:
        symbol: Optional symbol to filter by. If None, shows all symbols.
        data_dir: Directory where CSV files are stored
        
    Returns:
        pd.DataFrame: Summary of available files with columns:
                     [symbol, bar_type, date, filepath, size_mb]
                     
    Example:
        >>> available = list_available_data()
        >>> print(available)
        >>> eurusd_data = list_available_data('EURUSD')
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error(f"Data directory not found: {data_path}")
        return pd.DataFrame()
    
    # Find all bar CSV files
    pattern = f"{symbol}_*_bars_*.csv" if symbol else "*_bars_*.csv"
    files = list(data_path.glob(pattern))
    
    if not files:
        logging.info(f"No bar data files found in {data_dir}/")
        return pd.DataFrame()
    
    # Parse file information
    records = []
    for f in files:
        parts = f.stem.split('_')
        if len(parts) >= 4:
            sym = parts[0]
            bar_type = parts[1]
            date = '_'.join(parts[3:])  # Handle YYYYMMDD_HHMMSS format
            size_mb = f.stat().st_size / (1024 * 1024)
            
            records.append({
                'symbol': sym,
                'bar_type': bar_type,
                'date': date,
                'filepath': str(f),
                'size_mb': round(size_mb, 2)
            })
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        df = df.sort_values(['symbol', 'date', 'bar_type'], ascending=[True, False, True])
        logging.info(f"Found {len(df)} bar data files:")
        print(df.to_string(index=False))
    
    return df


def load_latest_bars(
    symbol: str,
    bar_type: str,
    data_dir: str = "data"
) -> Optional[pd.DataFrame]:
    """
    Convenience function to load the most recent bar data.
    
    Args:
        symbol: Trading symbol
        bar_type: Type of bars
        data_dir: Data directory
        
    Returns:
        pd.DataFrame: Most recent bar data
        
    Example:
        >>> latest_time_bars = load_latest_bars('EURUSD', 'time')
    """
    return load_bars(symbol, bar_type, date=None, data_dir=data_dir)


if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 60)
    print("Available Bar Data")
    print("=" * 60)
    list_available_data()
    
    print("\n" + "=" * 60)
    print("Loading EURUSD bars")
    print("=" * 60)
    bars = load_all_bars('EURUSD')
    
    if bars:
        print("\nLoaded bar types:")
        for bar_type, df in bars.items():
            print(f"  {bar_type}: {len(df):,} bars")
