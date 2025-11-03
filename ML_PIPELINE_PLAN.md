# ML Pipeline Implementation Plan

## Overview
This phased approach builds a complete ML trading pipeline based on MQL5 Article 19850 (Label Concurrency) using existing scripts in the workspace. The pipeline follows the methodology: Labeling → Concurrency/Weights → Model Training → Evaluation → Feature Analysis.

---

## PHASE 1: Concurrency & Sample Weights (concurrency_weights.ipynb)

**Objective**: Compute sample weights to address label concurrency using three methods

**Prerequisites**: 
- Completed `sides.ipynb` with labeled events (events DataFrame with 'bin', 't1', 'ret' columns)
- EURUSD tick bars DataFrame

**Notebook Structure**:

### Cell 1: Load Labeled Events
```python
# Load from sides.ipynb results or saved CSV
import pandas as pd
import numpy as np
from concurrent import get_num_conc_events, get_av_uniqueness_from_triple_barrier
from attribution import get_weights_by_return, get_weights_by_time_decay
from load_data import load_bars

# Load bar data
df = load_bars('EURUSD', 'tick')
close = df['close']

# Load events (from sides.ipynb or saved file)
# events should have columns: ['t1', 'bin', 'ret', 'side']
events = pd.read_csv('data/EURUSD_labeled_events.csv', index_col=0, parse_dates=True)
```

### Cell 2: Compute Concurrent Events Count
```python
# Count how many events are active at each timestamp
num_conc_events = get_num_conc_events(
    events=events, 
    close=close, 
    num_threads=4, 
    verbose=True
)

# Visualize concurrency
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
num_conc_events.plot()
plt.title('Number of Concurrent Events Over Time')
plt.ylabel('Concurrent Events')
plt.show()

print(f"Mean concurrency: {num_conc_events.mean():.2f}")
print(f"Max concurrency: {num_conc_events.max():.0f}")
```

### Cell 3: Compute Uniqueness Weights
```python
# Calculate average uniqueness (Method 1 from article)
uniqueness_weights = get_av_uniqueness_from_triple_barrier(
    triple_barrier_events=events,
    close_series=close,
    num_threads=4,
    num_conc_events=num_conc_events,
    verbose=True
)

# Display statistics
print("\nUniqueness Weights Statistics:")
print(uniqueness_weights.describe())

# Visualize distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
uniqueness_weights['tW'].hist(bins=50)
plt.xlabel('Uniqueness Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Uniqueness Weights')

plt.subplot(1, 2, 2)
uniqueness_weights['tW'].plot()
plt.ylabel('Uniqueness')
plt.title('Uniqueness Over Time')
plt.tight_layout()
plt.show()
```

### Cell 4: Compute Return Attribution Weights
```python
# Calculate return-based weights (WARNING: Use cautiously per article findings)
return_weights = get_weights_by_return(
    triple_barrier_events=events,
    close_series=close,
    num_threads=4,
    num_conc_events=num_conc_events,
    verbose=True
)

print("\nReturn Attribution Weights Statistics:")
print(return_weights.describe())

# Note: Article shows this method can cause model collapse
# Use only for comparison, not production
```

### Cell 5: Compute Time Decay Weights
```python
# Apply time decay to uniqueness weights
time_decay_weights = get_weights_by_time_decay(
    triple_barrier_events=events,
    close_series=close,
    num_threads=4,
    last_weight=0.5,  # Most recent gets weight 1.0, oldest gets 0.5
    linear=False,  # Use exponential decay
    av_uniqueness=uniqueness_weights,
    verbose=True
)

print("\nTime Decay Weights Statistics:")
print(time_decay_weights.describe())

# Visualize decay pattern
plt.figure(figsize=(14, 5))
time_decay_weights.plot()
plt.title('Time Decay Weights (Exponential)')
plt.ylabel('Weight')
plt.show()
```

### Cell 6: Compare All Weighting Methods
```python
# Combine all weights into single DataFrame
weights_df = pd.DataFrame({
    'uniqueness': uniqueness_weights['tW'],
    'return_attr': return_weights,
    'time_decay': time_decay_weights,
    'combined': uniqueness_weights['tW'] * time_decay_weights
})

# Normalize combined weights
weights_df['combined'] = weights_df['combined'] * len(weights_df) / weights_df['combined'].sum()

print("\nCorrelation between weighting methods:")
print(weights_df.corr())

# Visualize all methods
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(['uniqueness', 'return_attr', 'time_decay', 'combined']):
    ax = axes[idx // 2, idx % 2]
    weights_df[col].hist(bins=50, ax=ax)
    ax.set_title(f'{col.replace("_", " ").title()} Distribution')
    ax.set_xlabel('Weight')
plt.tight_layout()
plt.show()
```

### Cell 7: Save Weights for Model Training
```python
# Save all weight variants for use in models.ipynb
weights_df.to_csv('data/EURUSD_sample_weights.csv')
print("Saved sample weights to data/EURUSD_sample_weights.csv")

# Recommendation from article: Use 'uniqueness' or 'combined' for training
print(f"\nRecommended max_samples for BaggingClassifier: {weights_df['uniqueness'].mean():.3f}")
```

**Key Insights from Article**:
- Uniqueness weighting consistently improves F1 score (6.7% boost for Bollinger, 10.2% for MA Crossover)
- Return attribution can cause model collapse (avoid in meta-labeling)
- Time decay gives more weight to recent observations
- Set `max_samples=uniqueness.mean()` in BaggingClassifier

---

## PHASE 2: Model Training (models.ipynb)

**Objective**: Train Random Forest with sample weights and proper cross-validation

**Prerequisites**:
- Labeled events with features
- Sample weights from Phase 1

**Notebook Structure**:

### Cell 1: Load Data and Weights
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score

# Load events with features (from features.py or feature engineering)
events = pd.read_csv('data/EURUSD_labeled_events_with_features.csv', index_col=0, parse_dates=True)

# Load sample weights
weights_df = pd.read_csv('data/EURUSD_sample_weights.csv', index_col=0, parse_dates=True)

# Prepare X, y
feature_cols = [col for col in events.columns if col not in ['t1', 'bin', 'ret', 'side']]
X = events[feature_cols]
y = events['bin']

# Sample weights
sample_weights = weights_df['uniqueness'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")
print(f"Positive class ratio: {y.mean():.2%}")
```

### Cell 2: Implement PurgedKFold Cross-Validation
```python
from sklearn.model_selection import BaseCrossValidator

class PurgedKFold(BaseCrossValidator):
    """
    Cross-validation with purging and embargo to prevent data leakage
    From MQL5 Article 19850
    """
    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have same index")
        
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)]
        
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg:]))
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Initialize CV
cv = PurgedKFold(n_splits=5, t1=events['t1'], pct_embargo=0.01)
```

### Cell 3: Train Baseline Random Forest (No Weights)
```python
# Baseline model without sample weights
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Fit on full data for initial assessment
rf_baseline.fit(X, y)
print("Baseline Random Forest (no weights):")
print(f"Training accuracy: {rf_baseline.score(X, y):.3f}")
```

### Cell 4: Train Weighted Random Forest
```python
# Model with uniqueness-based sample weights
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Fit with sample weights
rf_weighted.fit(X, y, sample_weight=sample_weights)
print("\nWeighted Random Forest (uniqueness weights):")
print(f"Training accuracy: {rf_weighted.score(X, y):.3f}")
```

### Cell 5: Train Bagging Classifier with Constrained Samples
```python
# Method 1 from article: Constrain bootstrap sample size
avg_uniqueness = sample_weights.mean()

bagging_clf = BaggingClassifier(
    estimator=RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42
    ),
    n_estimators=10,
    max_samples=avg_uniqueness,  # Key parameter from article
    max_features=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

bagging_clf.fit(X, y, sample_weight=sample_weights)
print(f"\nBagging Classifier (max_samples={avg_uniqueness:.3f}):")
print(f"Training accuracy: {bagging_clf.score(X, y):.3f}")
```

### Cell 6: Save Models
```python
import joblib

# Save all trained models
joblib.dump(rf_baseline, 'models/rf_baseline.pkl')
joblib.dump(rf_weighted, 'models/rf_weighted.pkl')
joblib.dump(bagging_clf, 'models/bagging_weighted.pkl')

print("Models saved to models/ directory")
```

**Model Architecture Notes**:
- Use `class_weight='balanced'` for imbalanced classes
- Set `max_samples=uniqueness.mean()` in BaggingClassifier
- Apply sample_weight in `.fit()` method
- Use PurgedKFold for temporal data leakage prevention

---

## PHASE 3: Evaluation (evaluation.ipynb)

**Objective**: Comprehensive model performance assessment using multiple metrics

**Prerequisites**:
- Trained models from Phase 2
- Sample weights for scoring

**Notebook Structure**:

### Cell 1: Load Models and Data
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report

# Load models
rf_baseline = joblib.load('models/rf_baseline.pkl')
rf_weighted = joblib.load('models/rf_weighted.pkl')
bagging_clf = joblib.load('models/bagging_weighted.pkl')

# Load data
events = pd.read_csv('data/EURUSD_labeled_events_with_features.csv', index_col=0, parse_dates=True)
weights_df = pd.read_csv('data/EURUSD_sample_weights.csv', index_col=0, parse_dates=True)

feature_cols = [col for col in events.columns if col not in ['t1', 'bin', 'ret', 'side']]
X = events[feature_cols]
y = events['bin']
sample_weights = weights_df['uniqueness'].values
```

### Cell 2: Implement Probability Weighted Accuracy (PWA)
```python
def probability_weighted_accuracy(y_true, y_prob, sample_weight=None):
    """
    PWA metric from Article 19850
    Weights correct predictions by confidence level
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    
    # Predicted class and probability
    pred_idx = np.argmax(y_prob, axis=1)
    p_n = y_prob[np.arange(len(y_true)), pred_idx]
    
    # Correctness indicator
    y_n = (pred_idx == y_true).astype(int)
    
    # Confidence weights
    n_classes = y_prob.shape[1]
    baseline = 1.0 / n_classes
    conf_w = p_n - baseline
    
    # PWA calculation
    numerator = np.sum(sample_weight * y_n * conf_w)
    denominator = np.sum(sample_weight * conf_w)
    
    if np.isclose(denominator, 0.0):
        return 0.5
    
    return numerator / denominator

print("PWA metric implemented")
```

### Cell 3: Cross-Validation with All Metrics
```python
from models import PurgedKFold  # Import from models.ipynb

cv = PurgedKFold(n_splits=5, t1=events['t1'], pct_embargo=0.01)

def evaluate_model_cv(model, X, y, cv, sample_weights, model_name):
    """Evaluate model with cross-validation"""
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'neg_log_loss': [],
        'pwa': []
    }
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]
        
        # Train
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Score
        scores['accuracy'].append(accuracy_score(y_test, y_pred, sample_weight=w_test))
        scores['precision'].append(precision_score(y_test, y_pred, sample_weight=w_test, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, sample_weight=w_test, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, sample_weight=w_test, zero_division=0))
        scores['neg_log_loss'].append(-log_loss(y_test, y_prob, sample_weight=w_test))
        scores['pwa'].append(probability_weighted_accuracy(y_test.values, y_prob, w_test))
    
    # Convert to DataFrame
    scores_df = pd.DataFrame(scores)
    scores_df['model'] = model_name
    
    return scores_df

# Evaluate all models
results_baseline = evaluate_model_cv(rf_baseline, X, y, cv, sample_weights, 'RF Baseline')
results_weighted = evaluate_model_cv(rf_weighted, X, y, cv, sample_weights, 'RF Weighted')
results_bagging = evaluate_model_cv(bagging_clf, X, y, cv, sample_weights, 'Bagging')

# Combine results
all_results = pd.concat([results_baseline, results_weighted, results_bagging], ignore_index=True)
```

### Cell 4: Performance Comparison Table
```python
# Create summary table (similar to Article 19850)
summary = all_results.groupby('model').agg(['mean', 'std'])

print("\n=== Model Performance Comparison ===")
print(summary.round(3))

# Save results
summary.to_csv('results/model_comparison.csv')
```

### Cell 5: Visualization - Performance Metrics
```python
import matplotlib.pyplot as plt
import seaborn as sns

metrics = ['accuracy', 'f1', 'pwa', 'neg_log_loss']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = all_results.pivot(columns='model', values=metric)
    data.boxplot(ax=ax)
    ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_comparison_boxplots.png', dpi=300)
plt.show()
```

### Cell 6: Confusion Matrix Comparison
```python
# Train on full dataset and get confusion matrices
models = {
    'RF Baseline': rf_baseline,
    'RF Weighted': rf_weighted,
    'Bagging': bagging_clf
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X, y, sample_weight=sample_weights)
    y_pred = model.predict(X)
    
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
    axes[idx].set_title(f'{name}\nConfusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300)
plt.show()
```

### Cell 7: Key Findings Report
```python
# Generate comparison similar to article tables
print("\n=== KEY FINDINGS ===")
print("\n1. Uniqueness Weighting Impact:")
baseline_f1 = results_baseline['f1'].mean()
weighted_f1 = results_weighted['f1'].mean()
improvement = ((weighted_f1 - baseline_f1) / baseline_f1) * 100
print(f"   F1 Score improvement: {improvement:.1f}%")

print("\n2. Probability-Weighted Accuracy:")
baseline_pwa = results_baseline['pwa'].mean()
weighted_pwa = results_weighted['pwa'].mean()
pwa_improvement = ((weighted_pwa - baseline_pwa) / baseline_pwa) * 100
print(f"   PWA improvement: {pwa_improvement:.1f}%")

print("\n3. Recommendation:")
if weighted_f1 > baseline_f1:
    print("   ✓ Use sample weighting for production model")
else:
    print("   ✗ Sample weighting did not improve performance")
```

**Evaluation Metrics Explanation**:
- **Accuracy**: Overall correctness (misleading with imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall (critical for meta-labeling)
- **PWA**: Confidence-weighted accuracy (aligns with position sizing)
- **Neg Log-Loss**: Prediction confidence assessment (lower is better)

---

## PHASE 4: Feature Importance (extension to models.ipynb)

**Objective**: Understand which features drive model predictions

**Add to models.ipynb**:

### Cell: Feature Importance Analysis
```python
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from Random Forest
importances = rf_weighted.feature_importances_
feature_names = X.columns

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Top 20 features
top_20 = importance_df.head(20)

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/feature_importances.png', dpi=300)
plt.show()

print("\nTop 10 Features:")
print(importance_df.head(10))
```

### Cell: Permutation Importance
```python
from sklearn.inspection import permutation_importance

# Compute permutation importance
perm_importance = permutation_importance(
    rf_weighted, X, y, 
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Create DataFrame
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Visualize
plt.figure(figsize=(10, 8))
top_perm = perm_df.head(20)
plt.barh(range(len(top_perm)), top_perm['importance_mean'])
plt.yticks(range(len(top_perm)), top_perm['feature'])
plt.xlabel('Permutation Importance')
plt.title('Top 20 Permutation Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/permutation_importance.png', dpi=300)
plt.show()
```

### Cell: SHAP Values (Optional - requires shap library)
```python
try:
    import shap
    
    # Create explainer
    explainer = shap.TreeExplainer(rf_weighted)
    shap_values = explainer.shap_values(X.sample(1000))  # Sample for speed
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[1], X.sample(1000), plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('results/shap_importance.png', dpi=300)
    plt.show()
    
    print("SHAP analysis complete")
except ImportError:
    print("SHAP library not installed. Run: pip install shap")
```

---

## PHASE 5 (Optional): Deep Learning Models (deep_learning.ipynb)

**Objective**: Implement LSTM and WaveNet for temporal pattern recognition

**Note**: This is an advanced extension. Start only after Phases 1-4 are complete.

**Notebook Structure**:

### Cell 1: Prepare Sequential Data
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
events = pd.read_csv('data/EURUSD_labeled_events_with_features.csv', index_col=0, parse_dates=True)

# Create sequences (e.g., 50 timesteps)
def create_sequences(data, labels, seq_length=50):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(labels[i+seq_length])
    
    return np.array(sequences), np.array(targets)

# Prepare features
feature_cols = [col for col in events.columns if col not in ['t1', 'bin', 'ret', 'side']]
X = events[feature_cols].values
y = events['bin'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences
X_seq, y_seq = create_sequences(X_scaled, y, seq_length=50)

print(f"Sequence shape: {X_seq.shape}")
print(f"Targets shape: {y_seq.shape}")
```

### Cell 2: LSTM Architecture
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# Build LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()
```

### Cell 3: Train LSTM
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Split data
split_idx = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('models/lstm_best.h5', save_best_only=True)
]

# Train
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.show()
```

### Cell 4: Prediction Heatmaps
```python
# Generate predictions
y_pred_proba = lstm_model.predict(X_val)

# Create heatmap of predictions over time
plt.figure(figsize=(14, 6))
plt.imshow([y_pred_proba[:500].flatten()], aspect='auto', cmap='RdYlGn', interpolation='nearest')
plt.colorbar(label='Prediction Probability')
plt.title('LSTM Prediction Heatmap (First 500 samples)')
plt.xlabel('Time Step')
plt.ylabel('Probability')
plt.tight_layout()
plt.savefig('results/lstm_prediction_heatmap.png', dpi=300)
plt.show()
```

### Cell 5: WaveNet Architecture (Simplified)
```python
from tensorflow.keras.layers import Conv1D, Activation, Add

def residual_block(x, filters, kernel_size, dilation_rate):
    """WaveNet-style residual block"""
    conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    conv = Activation('relu')(conv)
    conv = Conv1D(filters, 1)(conv)
    return Add()([x, conv])

# Build WaveNet-style model
inputs = tf.keras.Input(shape=(X_seq.shape[1], X_seq.shape[2]))
x = Conv1D(32, 1)(inputs)

# Stack dilated convolutions
for i in range(8):
    x = residual_block(x, 32, kernel_size=2, dilation_rate=2**i)

x = Activation('relu')(x)
x = Conv1D(1, 1)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Activation('sigmoid')(x)

wavenet_model = tf.keras.Model(inputs=inputs, outputs=outputs)
wavenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

wavenet_model.summary()
```

---

## Implementation Checklist

### Phase 1: Concurrency & Weights ✓
- [ ] Create concurrency_weights.ipynb
- [ ] Compute concurrent events count
- [ ] Calculate uniqueness weights
- [ ] Calculate return attribution weights (for comparison)
- [ ] Calculate time decay weights
- [ ] Compare all weighting methods
- [ ] Save weights for model training

### Phase 2: Model Training ✓
- [ ] Create models.ipynb
- [ ] Implement PurgedKFold CV class
- [ ] Train baseline Random Forest
- [ ] Train weighted Random Forest
- [ ] Train Bagging classifier with constrained samples
- [ ] Save all models

### Phase 3: Evaluation ✓
- [ ] Create evaluation.ipynb
- [ ] Implement PWA metric
- [ ] Perform cross-validation with all metrics
- [ ] Create performance comparison table
- [ ] Generate visualizations (boxplots, confusion matrices)
- [ ] Document key findings

### Phase 4: Feature Importance ✓
- [ ] Add feature importance section to models.ipynb
- [ ] Calculate Random Forest feature importances
- [ ] Calculate permutation importances
- [ ] Optional: SHAP analysis
- [ ] Save importance plots

### Phase 5: Deep Learning (Optional)
- [ ] Create deep_learning.ipynb
- [ ] Prepare sequential data
- [ ] Implement LSTM architecture
- [ ] Implement WaveNet architecture
- [ ] Generate prediction heatmaps

---

## Key Takeaways from Article 19850

1. **Concurrency is a Universal Problem**: Affects all financial ML regardless of strategy
2. **Use Uniqueness Weighting**: Consistently improves F1 and PWA scores
3. **Avoid Return Attribution for Meta-Labeling**: Can cause model collapse
4. **Set max_samples Intelligently**: Use mean uniqueness value
5. **Always Use Purged Cross-Validation**: Prevents temporal data leakage
6. **Evaluate with Multiple Metrics**: Accuracy alone is misleading

---

## Expected Timeline

- **Phase 1**: 2-3 hours (straightforward, uses existing functions)
- **Phase 2**: 3-4 hours (model training and CV implementation)
- **Phase 3**: 2-3 hours (evaluation and visualization)
- **Phase 4**: 1-2 hours (feature analysis)
- **Phase 5**: 6-8 hours (deep learning, optional)

**Total Core Pipeline (Phases 1-4)**: ~8-12 hours

---

## Next Steps

1. Start with Phase 1 (concurrency_weights.ipynb)
2. Ensure you have labeled events from sides.ipynb saved to CSV
3. Run each cell sequentially and verify outputs
4. Save all intermediate results (weights, models, metrics)
5. Document any deviations from article methodology

Let me know when you're ready to begin Phase 1!
