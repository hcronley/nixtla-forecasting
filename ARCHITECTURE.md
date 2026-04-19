# 🏗️ System Architecture: Time Series Forecasting App

## Overview

This document describes the technical architecture, design decisions, and implementation details of the Time Series Forecasting App.

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                       │
│  (streamlit_app.py - 40KB, 1000+ lines)                     │
│                                                              │
│  ├─ Data Loading (CSV/Sample)                              │
│  ├─ Configuration Management                               │
│  ├─ Model Selection & Parameters                           │
│  ├─ Forecast Execution                                     │
│  ├─ Visualization & Results Display                        │
│  └─ Error Handling & User Feedback                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
┌───────▼──────┐  │  ┌───────▼──────────┐
│  VALIDATION  │  │  │   CONFIGURATION  │
│              │  │  │                  │
│ validators.py│  │  │config_manager.py │
│   (18 KB)    │  │  │    (12 KB)       │
│              │  │  │                  │
│ ├─ CSV Check │  │  │ ├─ Save Config   │
│ ├─ Date Parse│  │  │ ├─ Load Config   │
│ ├─ Data QA   │  │  │ ├─ JSON Export   │
│ └─ Frequency │  │  │ └─ Manage Files  │
└──────────────┘  │  └──────────────────┘
                  │
        ┌─────────▼────────────────────┐
        │   FORECASTING PIPELINE       │
        │                              │
        │  ┌──────────────────────┐    │
        │  │   BACKTESTING LAYER  │    │
        │  │   backtesting.py     │    │
        │  │      (18 KB)         │    │
        │  │                      │    │
        │  │ • Create Windows     │    │
        │  │ • Run Models         │    │
        │  │ • Aggregate Results  │    │
        │  │ • Generate Rankings  │    │
        │  └──────┬───────────────┘    │
        │         │                    │
        │  ┌──────▼──────────────────┐ │
        │  │  FORECASTING MODULES    │ │
        │  │                         │ │
        │  │ ┌─ df_statsforecast.py │ │
        │  │ │  • ARIMA, ETS, etc  │ │
        │  │ │  • One-step         │ │
        │  │ │  • Multi-step       │ │
        │  │ │                     │ │
        │  │ ├─ df_mlforecast.py   │ │
        │  │ │  • XGBoost, etc     │ │
        │  │ │  • Lag Features     │ │
        │  │ │  • Multi-output     │ │
        │  │ │                     │ │
        │  │ └─ df_neuralforecast  │ │
        │  │    • LSTM, MLP, etc  │ │
        │  │    • PyTorch-based   │ │
        │  │    • GPU-friendly    │ │
        │  │                     │ │
        │  └─────────────────────┘ │
        │                          │
        │  ┌──────────────────────┐ │
        │  │   REPORTING LAYER    │ │
        │  │   reporting.py       │ │
        │  │      (13 KB)         │ │
        │  │                      │ │
        │  │ • Rankings Table     │ │
        │  │ • Heatmaps          │ │
        │  │ • Comparisons       │ │
        │  │ • Summaries         │ │
        │  └──────────────────────┘ │
        └─────────────────────────────┘
                  │
        ┌─────────▼────────────┐
        │   DATA LAYER         │
        │                      │
        │  sample_data.py      │
        │  • AirPassengers     │
        │  • Energy Usage      │
        │  • Retail Sales      │
        │  • Temperature       │
        └──────────────────────┘
```

## Module Responsibilities

### `streamlit_app.py` (40 KB)
**Purpose**: Main application interface and orchestration

**Key Functions**:
- `main()` - App entry point
- `load_uploaded_file()` - Handle CSV uploads with validation
- `validate_and_prepare_data_wrapper()` - Data prep with error handling
- `build_backtest_models()` - Auto-select 7 models from all libraries
- `display_backtest_results()` - Visualizations & metrics
- `run_forecast()` - Execute single/batch forecasting
- `initialize_session_state()` - Streamlit state management

**UI Components**:
- Sidebar: Configuration controls
- Main area: Data overview, results, visualizations
- Expanders: Advanced options, debug views

**Dependencies**:
- `streamlit` - UI framework
- `plotly` - Interactive visualizations
- `pandas`, `numpy` - Data handling
- All three Nixtla libraries

### `validators.py` (18 KB)
**Purpose**: Input validation and data quality checks

**Classes**:
- `DataValidator` - Static methods for all validation checks
- `ValidationError` - Custom exception type

**Key Methods**:
- `validate_csv_structure()` - Check format and columns
- `validate_date_column()` - Parse and verify dates
- `validate_value_column()` - Convert and check numeric values
- `validate_data_length()` - Ensure minimum observations
- `validate_missing_values()` - Check completeness
- `validate_time_series_structure()` - Comprehensive check
- `detect_frequency()` - Auto-detect time series frequency
- `validate_forecast_parameters()` - Check horizon/train/test consistency
- `check_data_quality()` - Generate quality report

**Strategy**:
- Fail-fast on critical errors (return False immediately)
- Collect warnings for non-critical issues
- Provide actionable error messages
- Support graceful degradation (fallbacks)

### `backtesting.py` (18 KB)
**Purpose**: Rolling window cross-validation framework

**Classes**:
- `RollingWindowBacktester` - Main backtesting engine

**Key Methods**:
- `create_windows()` - Generate 5 rolling train/test splits
- `run_backtest()` - Execute all models on all windows
- `run_statsforecast_model()` - Run StatsForecast variant
- `run_mlforecast_model()` - Run MLForecast variant
- `run_neuralforecast_model()` - Run NeuralForecast variant
- `aggregate_results()` - Average metrics across windows
- `get_rankings()` - Sort models by performance
- `get_summary_stats()` - Generate summary table
- `export_results()` - Save to CSV

**Algorithm**:
```
For each window i in 1..5:
  train_set = data[0:train_start_i]
  test_set = data[train_start_i:train_end_i]
  
  For each model in model_list:
    metrics = run_model(train_set, test_set)
    store_metrics(model, window_i, metrics)

Aggregate metrics across all windows:
  For each model:
    mae_mean = avg(mae across 5 windows)
    mae_std = std(mae across 5 windows)
    Rank by mae_mean
```

**Window Strategy**:
- 5 fixed windows with sliding stride
- Each window gets progressively more training data
- No data leakage (test never in training)
- Realistic backtest simulation

### `reporting.py` (13 KB)
**Purpose**: Analysis and visualization of backtesting results

**Classes**:
- `ModelComparator` - Analyze and visualize results

**Key Methods**:
- `get_ranking_table()` - Formatted rankings
- `get_summary_table()` - Best model statistics
- `create_comparison_plot()` - Bar chart of top N models
- `create_window_performance_heatmap()` - Window × Model heatmap
- `create_metric_heatmap()` - Detailed metric comparison
- `create_ranking_bar_chart()` - Ranked performance
- `create_multi_metric_comparison()` - Normalized comparison
- `export_rankings_csv()` - Save rankings
- `get_best_model_config()` - Extract best model info

**Visualizations**:
- Plotly interactive charts
- Color-coded heatmaps (red=high error, green=low)
- Multi-metric comparisons
- Performance across windows

### `config_manager.py` (12 KB)
**Purpose**: Configuration persistence and reproducibility

**Classes**:
- `ConfigManager` - File-based config management

**Key Methods**:
- `save_config()` - Write to JSON
- `load_config()` - Read from JSON
- `list_configs()` - List all saved configs
- `delete_config()` - Remove config file
- `export_best_model_config()` - Extract from backtest results
- `get_config_summary()` - Human-readable output

**Utility Functions**:
- `create_config_from_params()` - Build config dict
- `load_and_validate_config()` - Load with validation

**Config Structure**:
```json
{
  "dataset": string,
  "model_name": string,
  "module": "StatsForecast|MLForecast|NeuralForecast",
  "model_type": string,
  "params": {
    "freq": string,
    "horizon": int,
    "season_length": int,
    "lags": [int],
    "input_size": int
  },
  "metrics": {
    "mae_mean": float,
    "rmse_mean": float,
    "mape_mean": float,
    "success_rate": float
  },
  "timestamp": ISO-8601,
  "description": string
}
```

### `df_statsforecast.py` (16 KB)
**Purpose**: Statistical forecasting models wrapper

**Classes**:
- `StatsforecastForecaster` - Interface to StatsForecast

**Models**:
- ARIMA, AutoARIMA - Autoregressive integrated moving average
- AutoETS - Exponential smoothing
- Naive, SeasonalNaive - Baseline models
- RandomWalkWithDrift - Random walk with trend

**Forecasting Strategies**:
1. **One-step** - Refit for each prediction (h=1 iterative)
2. **Multi-step** - Recursive forecasting (default StatsForecast)
3. **Multi-output** - NOT SUPPORTED (raises NotImplementedError)

**Why No Multi-Output?**
- Statistical models (ARIMA/ETS) are conditional on history
- Can't train separate models for each horizon step
- Recursive is the natural approach for these models

### `df_mlforecast.py` (21 KB)
**Purpose**: Machine learning forecasting models wrapper

**Classes**:
- `MLForecastForecaster` - Interface to MLForecast

**Models**:
- XGBoost, LightGBM, CatBoost - Tree-based ensemble
- RandomForest - Ensemble of trees
- LinearRegression - Baseline linear model

**Features**:
- Automatic lag feature generation
- Date feature extraction (month, dayofweek, etc.)
- Target transformations (differencing)
- Supports multi-output via `max_horizon` parameter

**Lag Strategy**:
- Default: [1, 12] (t-1 and t-12)
- Customizable for domain knowledge
- Combined with date features for rich representation

### `df_neuralforecast.py` (22 KB)
**Purpose**: Deep learning forecasting models wrapper

**Classes**:
- `NeuralForecastForecaster` - Interface to NeuralForecast

**Models**:
- RNN, LSTM, GRU - Recurrent architectures
- MLP - Feedforward network
- NBEATS, NHITS - State-of-the-art architectures
- TCN - Temporal convolutional network

**Forecasting Strategies**:
1. **One-step** - Refit h=1 model for each step
2. **Multi-step** - Recursive (RNN/LSTM/GRU only)
3. **Multi-output** - Direct (default for NBEATS/NHITS/MLP)

**Training**:
- PyTorch backend
- PyTorch Lightning for training
- GPU-compatible
- Configurable epochs, learning rate, batch size

### `sample_data.py` (7.6 KB)
**Purpose**: Pre-loaded datasets for testing

**Datasets**:
1. **AirPassengers** - Classical dataset
   - 144 monthly observations (1949-1960)
   - Clear trend and seasonality
   - Good for introducing time series concepts

2. **Energy Consumption** - Daily data
   - 365 observations (2023)
   - Weekly pattern
   - Realistic household data

3. **Retail Sales** - Weekly data
   - 156 observations (2021-2023)
   - Yearly seasonality with holiday spikes
   - Business forecasting application

4. **Temperature** - Daily data
   - 730 observations (2022-2023)
   - Strong yearly seasonality
   - Stationary-like after differencing

**Metadata**:
- Frequency, season_length, recommended_horizon
- Description for UI

## Data Flow

### 1. Data Loading
```
CSV Upload or Sample Dataset Selection
    ↓
load_uploaded_file() / get_sample_data()
    ↓
Validate with validators.py
    ↓
Convert to standard format (ds, y columns)
    ↓
Store in session_state.current_data
```

### 2. Single Forecast Mode
```
User configures: module, model, strategy, parameters
    ↓
Click "Run Forecast"
    ↓
run_forecast() routes to appropriate module
    ↓
Module trains on train_df, forecasts on test_df
    ↓
Calculate metrics (MAE, RMSE, MAPE)
    ↓
plot_forecast_results() visualizes
    ↓
Display metrics and downloadable results
```

### 3. Backtesting Mode
```
User checks "Compare all models" checkbox
    ↓
Click "Run Backtesting Comparison"
    ↓
build_backtest_models() selects 7 models
    ↓
RollingWindowBacktester.create_windows()
    ↓
For each window:
  For each model:
    Run through one of three modules
    Collect metrics
    ↓
Aggregate results across all windows
    ↓
ModelComparator generates visualizations
    ↓
Display rankings, heatmaps, comparisons
    ↓
User can save best model config to JSON
```

## Key Design Decisions

### 1. Three Separate Pipeline Modules
**Why?**: Each Nixtla library has different APIs and use cases
- StatsForecast: Statistical models, simple API
- MLForecast: Feature engineering, tree-based
- NeuralForecast: Deep learning, PyTorch

**Alternative Considered**: Single unified wrapper
- **Rejected**: Would hide library-specific capabilities
- **Current**: Pros outweigh cons for educational app

### 2. 5 Rolling Windows for Backtesting
**Why?**: 
- Sufficient for reliable averages (n=5)
- Balances coverage and runtime (2-3 minutes)
- Common in time series literature

**Alternative Considered**: 10 windows
- **Rejected**: Would double runtime
- **Alternative Available**: Configurable in code

### 3. Automatic Model Selection for Backtesting
**Why?**:
- User can compare paradigms automatically
- No need to manually pick each model
- Balanced representation (3 StatsForecast, 2 ML, 2 Neural)

**Alternative Considered**: User selects each model
- **Rejected**: Too many options, analysis paralysis

### 4. JSON Configs Instead of Database
**Why?**:
- Simple, portable, human-readable
- No database dependency
- Works with free Streamlit Cloud
- Easy to version control

**Alternative Considered**: SQLite database
- **Rejected**: Overkill for this use case
- **Future**: Could migrate if needed

### 5. Separate Validation Module
**Why?**:
- Reusable validation logic
- Clear separation of concerns
- Comprehensive error handling
- User-friendly feedback

**Coupled Validation**:
- **Rejected**: Would clutter main app
- **Anti-pattern**: Validation scattered everywhere

## Performance Considerations

### Memory Usage
- **Single Forecast**: ~100-200 MB
- **Backtesting**: ~500 MB - 1 GB
- **Limiting Factor**: Neural models (PyTorch)

### Runtime
- **Single Forecast**: 30-60 seconds
- **Backtesting 7 models × 5 windows**: 2-3 minutes
- **First Run**: Slower (model loading)
- **Subsequent Runs**: Faster (caching)

### Optimization Strategies
1. **Caching**: Streamlit's @st.cache_data for sample data
2. **Lazy Loading**: Models loaded only when needed
3. **Batch Processing**: Run multiple windows in sequence
4. **Early Stopping**: Stop training if convergence achieved

## Error Handling Strategy

### Levels of Validation

1. **Data Entry Level**
   - CSV format checks
   - Column existence
   - Data type validation

2. **Data Quality Level**
   - Missing value detection
   - Outlier identification
   - Frequency consistency

3. **Forecasting Parameters Level**
   - Horizon vs test size
   - Input size vs train size
   - Minimum observations

4. **Model Execution Level**
   - Try-except in each forecasting function
   - Graceful degradation for unavailable models
   - Clear error messages to users

### Error Message Strategy
- **❌ Critical**: Block execution
- **⚠️ Warning**: Allow but inform user
- **✅ Success**: Confirm completion

## Testing Strategy

### Unit Testing (Not in Current Version)
- Could test validators independently
- Could test config_manager CRUD operations
- Could test backtesting window creation

### Integration Testing
- End-to-end forecast execution
- Full backtesting pipeline
- Config save/load roundtrip

### Manual Testing Scenarios
1. Upload CSV with various data quality issues
2. Run single forecast with each module
3. Run backtesting with all 7 models
4. Save and load configurations
5. Test on all sample datasets

## Future Enhancement Opportunities

### Short Term
1. Additional metrics (sMAPE, MASE, RMSE)
2. Confidence intervals/prediction bands
3. Automated hyperparameter tuning
4. More sample datasets

### Medium Term
1. Multivariate forecasting
2. Exogenous variables support
3. Model interpretability (SHAP, feature importance)
4. Automated model selection
5. Prophet and AutoML support

### Long Term
1. Real-time data ingestion
2. Model serving/REST API
3. A/B testing framework
4. Ensemble forecasting
5. Transfer learning for small datasets

## Dependencies & Versions

### Critical Dependencies
- Python 3.8+ (type hints, walrus operator)
- Streamlit 1.28+ (new features)
- Nixtla libraries 1.5+ (stable APIs)

### Breaking Changes to Watch
- PyTorch 2.0+ changed module paths
- PyTorch-Lightning 2.0+ changed training API
- Nixtla libraries add models frequently

## Deployment Considerations

### Streamlit Cloud
- Free tier: 1 GB RAM per app
- May timeout on complex backtesting
- Large dataset size limited

### Private Deployment
- Consider Docker containerization
- Use environment variables for configuration
- Implement authentication if needed
- Monitor resource usage

## Conclusion

This architecture provides:
- ✅ **Modularity**: Each component has single responsibility
- ✅ **Extensibility**: Easy to add new models/strategies
- ✅ **Robustness**: Comprehensive validation and error handling
- ✅ **Usability**: Intuitive UI with helpful feedback
- ✅ **Reproducibility**: Config management for repeatability
- ✅ **Performance**: Optimized for Streamlit Cloud free tier
- ✅ **Maintainability**: Clear code organization and documentation

The system successfully combines three major forecasting paradigms into a single, user-friendly application suitable for education, research, and production use.

---

For deployment details, see [DEPLOYMENT.md](DEPLOYMENT.md)  
For user guide, see [README.md](README.md)
