# MQL5 Machine Learning Trading System

Financial machine learning framework for algorithmic trading using MQL5/MetaTrader 5, implementing methods from "Advances in Financial Machine Learning" by Marcos López de Prado.

## 📊 Features

- **Labeling Methods**
  - Triple-Barrier Method with meta-labeling
  - Trend-Scanning Labels
  - Fixed-horizon labels with volatility-based sizing

- **Feature Engineering**
  - Technical indicators (90+ features)
  - Time-based features (session, time-of-day)
  - Volatility features (realized vol, vol-of-vol)
  - Microstructure features (order flow proxies)

- **Sample Weighting**
  - Label uniqueness (concurrent events)
  - Return attribution
  - Time-decay weighting

- **Models**
  - Random Forest with meta-labeling
  - Sample-weighted training
  - Feature importance analysis

## 📁 Project Structure

```
mql5/
├── bars.py                          # Bar generation (time, tick, volume, dollar)
├── filters.py                       # CUSUM filter for event detection
├── fractals.py                      # Fractal pattern detection
├── triple_barrier.py                # Triple-barrier labeling
├── trend_scanning.py                # Trend-scanning labels
├── moving_averages.py               # MA indicators
├── bollinger_features.py            # Bollinger Band features
├── volatility.py                    # Volatility calculations
├── returns.py                       # Return calculations
├── time_features.py                 # Time-based features
├── strategies.py                    # Trading strategies
├── optimized_concurrent.py          # Sample uniqueness weights
├── optimized_attribution.py         # Return attribution weights
├── mt5_login.py                     # MT5 connection
├── feature_triple_barrier.ipynb     # Feature generation (triple-barrier)
├── training_data_triple_barrier.ipynb  # Training data prep
├── training_data_trend_scanning.ipynb  # Trend-scanning data prep
├── models.ipynb                     # Model training (triple-barrier)
├── models_ts.ipynb                  # Model training (trend-scanning)
└── data/                            # Data storage (gitignored)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal
- MT5 account (demo or live)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mql5.git
cd mql5

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### MT5 Configuration

1. Update `mt5_login.py` with your credentials:
```python
account = YOUR_ACCOUNT_NUMBER
password = "YOUR_PASSWORD"
server = "YOUR_BROKER_SERVER"
```

2. Ensure MT5 terminal is running

### Workflow

1. **Generate Features**: Run `feature_triple_barrier.ipynb`
2. **Prepare Training Data**: Run `training_data_triple_barrier.ipynb` or `training_data_trend_scanning.ipynb`
3. **Train Model**: Run `models.ipynb` or `models_ts.ipynb`
4. **Evaluate**: Analyze metrics, confusion matrices, feature importance

## 📚 Methods Implemented

### Triple-Barrier Labeling
- Defines profit-taking, stop-loss, and time-based barriers
- Meta-labeling: predicts probability of profit given primary strategy signal
- Sample weighting based on label uniqueness

### Trend-Scanning
- Fits linear regressions over multiple time horizons
- Selects window with highest |t-value|
- Volatility-based masking to avoid low-vol regimes

### Meta-Labeling
- Binary classification: TAKE_TRADE (1) vs SKIP_TRADE (0)
- Filters false positives from primary strategy
- Improves precision while maintaining strategy recall

## 🎯 Current Performance

**Triple-Barrier Model (Bollinger Strategy)**
- Test Accuracy: ~55-60%
- ROC AUC: ~0.60-0.65
- Dataset: ~10k labeled signals
- Top features: Session volatility, time-of-day, trend indicators

**Trend-Scanning Model (MA Crossover)**
- Test Accuracy: ~55-60%
- ROC AUC: ~0.58-0.62
- Dataset: ~800 labeled crossovers
- Top features: Session volatility, dayofweek, realized vol

## 🔧 Technologies

- **Data**: MetaTrader 5 Python API
- **Processing**: pandas, numpy, numba
- **ML**: scikit-learn (Random Forest)
- **Visualization**: matplotlib, seaborn
- **Optimization**: multiprocess, numba JIT

## 📖 References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- [MQL5 Article #19253](https://www.mql5.com/en/articles/19253) - Trend-Scanning Implementation

## ⚠️ Disclaimer

This is educational/research code. Not financial advice. Past performance does not guarantee future results. Trading involves risk of loss.

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.
