"""
Quick test script to verify MT5 connection and data availability.

Usage:
    python test_mt5_connection.py
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

from mt5_login import login_mt5, mt5_logged_in


def test_connection():
    """Test MT5 connection."""
    print("INFO: Testing MT5 connection...")
    
    # Try to login (update account name as needed)
    account = input("Enter your account name (from credentials file): ").strip()
    
    try:
        login_mt5(account, verbose=True)
        
        # Check actual MT5 terminal info to verify connection
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            print("ERROR: Failed to connect to MT5")
            return False
        
        print("\n✓ MT5 connection successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Connection error: {e}")
        return False


def test_symbol(symbol: str = "SPY"):
    """Test if a symbol is available."""
    print(f"\nINFO: Testing symbol availability: {symbol}")
    
    # Check symbol info
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"WARNING: ✗ Symbol {symbol} not found")
        return False
    
    print(f"✓ Symbol {symbol} found!")
    print(f"  Description: {symbol_info.description}")
    print(f"  Currency: {symbol_info.currency_base}/{symbol_info.currency_profit}")
    print(f"  Visible: {symbol_info.visible}")
    
    if not symbol_info.visible:
        print(f"  Attempting to enable {symbol}...")
        if mt5.symbol_select(symbol, True):
            print(f"  ✓ {symbol} enabled")
        else:
            print(f"  ✗ Failed to enable {symbol}")
            return False
    
    return True


def test_tick_data(symbol: str = "SPY", days: int = 1):
    """Test if tick data is available."""
    print(f"\nINFO: Testing tick data availability for {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"  Requesting ticks from {start_date} to {end_date}")
    
    try:
        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            print(f"WARNING: ✗ No tick data available for {symbol}")
            print("  Try:")
            print("    - Reducing date range")
            print("    - Checking if market was open")
            print("    - Verifying broker provides tick data for this symbol")
            return False
        
        print(f"✓ Received {len(ticks):,} ticks!")
        print(f"  First tick: {datetime.fromtimestamp(ticks[0][0])}")
        print(f"  Last tick: {datetime.fromtimestamp(ticks[-1][0])}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error retrieving tick data: {e}")
        return False


def list_available_symbols(search_pattern: str = ""):
    """List available symbols."""
    print("\nINFO: Listing available symbols...")
    
    symbols = mt5.symbols_get()
    
    if symbols is None or len(symbols) == 0:
        print("WARNING: No symbols found")
        return
    
    print(f"Total symbols available: {len(symbols)}")
    
    if search_pattern:
        filtered = [s for s in symbols if search_pattern.upper() in s.name.upper()]
        print(f"Symbols matching '{search_pattern}': {len(filtered)}")
        
        for sym in filtered[:20]:  # Show first 20 matches
            print(f"  {sym.name} - {sym.description}")
        
        if len(filtered) > 20:
            print(f"  ... and {len(filtered) - 20} more")
    else:
        print("Common symbols:")
        common = ["SPY", "EURUSD", "GBPUSD", "USDJPY", "AAPL", "MSFT", "GOOGL"]
        for sym_name in common:
            sym = mt5.symbol_info(sym_name)
            if sym:
                print(f"  ✓ {sym_name} - {sym.description}")
            else:
                print(f"  ✗ {sym_name} - Not available")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MT5 Connection and Data Availability Test")
    print("=" * 60)
    
    # Test 1: Connection
    if not test_connection():
        print("\nERROR: Connection test failed. Please check your credentials.")
        return
    
    # Test 2: List symbols
    list_available_symbols()
    
    # Test 3: Check specific symbol
    print()
    symbol = input("Enter symbol to test (default: SPY): ").strip().upper() or "SPY"
    
    if not test_symbol(symbol):
        print(f"\nWARNING: {symbol} not available. Try another symbol.")
        
        # Show alternatives
        print("\nSearching for similar symbols...")
        list_available_symbols(symbol[:2])  # Search by first 2 letters
        
        mt5.shutdown()
        return
    
    # Test 4: Tick data
    if not test_tick_data(symbol, days=2):
        print(f"\nWARNING: No recent tick data for {symbol}. Try:")
        print("  1. A different date range")
        print("  2. A different symbol (e.g., EURUSD)")
        print("  3. Checking if markets are open")
    
    # Cleanup
    mt5.shutdown()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nIf all tests passed, you're ready to run: python spy_bars_analysis.py")


if __name__ == "__main__":
    main()
