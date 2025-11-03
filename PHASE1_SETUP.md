# Phase 1 Setup Checklist

## Prerequisites

Before running `concurrency_weights.ipynb`, ensure you have:

### ✓ Required Files
- [ ] `optimized_concurrent.py` - Already exists in workspace
- [ ] `optimized_attribution.py` - Already exists in workspace  
- [ ] `load_data.py` - Already exists in workspace
- [ ] `sides.ipynb` - Should be completed with labeled events

### ✓ Required Data
You need to save labeled events from `sides.ipynb`:

```python
# Add this cell to sides.ipynb to save events
events.to_csv('data/EURUSD_triple_barrier_events.csv')
print(f"Saved {len(events)} events to data/EURUSD_triple_barrier_events.csv")
```

Expected columns in events DataFrame:
- `t1` - End time when barrier was hit
- `bin` - Binary label (0 or 1)
- `ret` - Return of the trade
- `side` - Position side (1=long, -1=short)

### ✓ Required Directories
- [x] `data/` - Created automatically by notebook
- [x] `results/` - Created automatically
- [ ] `models/` - Will be created in Phase 2

### ✓ Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

All should be installed in your `.venv` already.

---

## Running Phase 1

1. **Save events from sides.ipynb**
   ```python
   # In sides.ipynb, add a cell:
   events.to_csv('data/EURUSD_labeled_events.csv')
   ```

2. **Open concurrency_weights.ipynb**
   - File is ready to run
   - Uses optimized functions (5-10x faster)
   - All cells can be run sequentially

3. **Expected Runtime**
   - Concurrent events: ~5-10 seconds
   - Uniqueness weights: ~3-5 seconds  
   - Return attribution: ~5-10 seconds
   - Time decay: ~2-3 seconds
   - Total: ~15-30 seconds

4. **Expected Outputs**
   Files saved to `data/`:
   - `EURUSD_sample_weights.csv` - All weight variants
   - `EURUSD_events_with_weights.csv` - Events with weights attached
   - `EURUSD_concurrent_events.csv` - Concurrency time series

   Plots saved to `results/`:
   - `concurrent_events.png` - Concurrency over time
   - `uniqueness_weights.png` - Uniqueness analysis
   - `return_attribution_weights.png` - Return weights
   - `time_decay_weights.png` - Time decay pattern
   - `weight_correlation_heatmap.png` - Weight correlations
   - `all_weights_comparison.png` - Side-by-side comparison

---

## Troubleshooting

### Error: "Events file not found"
**Solution**: Run this in sides.ipynb:
```python
events.to_csv('data/EURUSD_labeled_events.csv')
```

### Error: "Module not found"
**Solution**: Check imports are correct:
```python
from optimized_concurrent import get_num_conc_events_optimized
from optimized_attribution import get_weights_by_return_optimized
```

### Slow performance
**Check**: Make sure using optimized versions, not standard concurrent.py/attribution.py

### Index mismatch errors
**Check**: events DataFrame and close Series should have compatible datetime indices

---

## Key Metrics to Watch

After running Phase 1, check these values:

1. **Mean Concurrency**: Should be 10-50 for forex tick data
   - Higher = More overlap = More need for weighting

2. **Mean Uniqueness**: Should be 0.2-0.5 typically
   - This becomes your `max_samples` parameter in Phase 2
   - Lower = More overlap in your labels

3. **Weight Correlations**:
   - Uniqueness vs Time Decay: Usually 0.3-0.6
   - Return Attribution: Often very different from others

---

## Next Steps

After Phase 1 completes successfully:

1. Verify all CSV files saved to `data/`
2. Check all plots saved to `results/`
3. Note the "Recommended max_samples" value from final output
4. Proceed to Phase 2: Model Training

---

## Article Reference

Based on: **MQL5 Article 19850 - Label Concurrency**
https://www.mql5.com/en/articles/19850

Key findings used in this notebook:
- Uniqueness weighting improves F1 by 6-10%
- Return attribution causes model collapse (included for comparison only)
- Average uniqueness should be used as max_samples parameter
