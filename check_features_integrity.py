import pandas as pd
import numpy as np
from pathlib import Path

print('='*80)
print('FEATURE CSV INTEGRITY CHECK')
print('='*80)

files = {
    'Trend-Scanning': 'data/features_trend_scanning.csv',
    'Triple-Barrier': 'data/features_triple_barrier.csv'
}

for name, filepath in files.items():
    print(f'\n{name} Features')
    print('-'*80)
    
    # Load data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Basic info
    print(f'Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
    print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    
    # Data quality checks
    print(f'\nData Quality:')
    print(f'  Missing values: {df.isnull().sum().sum():,}')
    print(f'  Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum():,}')
    print(f'  Duplicate rows: {df.duplicated().sum():,}')
    print(f'  Duplicate timestamps: {df.index.duplicated().sum():,}')
    
    # Check for constant columns (all zeros or all same value)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f'  WARNING - Constant columns ({len(constant_cols)}): {constant_cols[:5]}...')
    else:
        print(f'  ✓ No constant columns')
    
    # Check for columns with too many zeros
    zero_heavy = []
    for col in df.columns:
        zero_pct = (df[col] == 0).sum() / len(df) * 100
        if zero_pct > 95:
            zero_heavy.append((col, zero_pct))
    
    if zero_heavy:
        print(f'  WARNING - Columns with >95% zeros ({len(zero_heavy)}):')
        for col, pct in zero_heavy[:5]:
            print(f'    {col}: {pct:.1f}% zeros')
    else:
        print(f'  ✓ No zero-heavy columns')
    
    # Statistical summary
    print(f'\nStatistical Summary:')
    numeric_df = df.select_dtypes(include=[np.number])
    print(f'  Mean of means: {numeric_df.mean().mean():.6f}')
    print(f'  Mean of stds: {numeric_df.std().mean():.6f}')
    print(f'  Min value: {numeric_df.min().min():.6f}')
    print(f'  Max value: {numeric_df.max().max():.6f}')
    
    # Check first and last rows for zeros (potential warm-up issues)
    first_100_zeros = (df.head(100) == 0).all(axis=1).sum()
    last_100_zeros = (df.tail(100) == 0).all(axis=1).sum()
    
    print(f'\nWarm-up/Cool-down Check:')
    print(f'  First 100 rows (all zeros): {first_100_zeros}')
    print(f'  Last 100 rows (all zeros): {last_100_zeros}')
    
    # Index continuity
    print(f'\nIndex Integrity:')
    print(f'  Monotonic increasing: {df.index.is_monotonic_increasing}')
    print(f'  Unique timestamps: {df.index.is_unique}')
    
    # Show some sample values
    print(f'\nSample Values (first non-zero row):')
    non_zero_mask = (df != 0).any(axis=1)
    if non_zero_mask.any():
        first_non_zero_idx = non_zero_mask.idxmax()
        first_non_zero = df.loc[first_non_zero_idx]
        non_zero_cols = first_non_zero[first_non_zero != 0]
        print(f'  Timestamp: {first_non_zero_idx}')
        print(f'  Non-zero columns: {len(non_zero_cols)}/{len(df.columns)}')
        if len(non_zero_cols) > 0:
            print(f'  Sample: {list(non_zero_cols.head(5).items())}')

print('\n' + '='*80)
print('INTEGRITY CHECK COMPLETE')
print('='*80)
