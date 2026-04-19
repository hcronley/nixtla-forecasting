# 📈 Time Series Forecasting App with Nixtla

A production-ready web application for comparing and evaluating time series forecasting models using **StatsForecast**, **MLForecast**, and **NeuralForecast** from the Nixtla ecosystem.

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Overview

This application provides an intuitive interface for:
- **Comparing forecasting paradigms**: Statistical (ARIMA/ETS), Machine Learning (XGBoost/LightGBM), and Deep Learning (LSTM/NBEATS)
- **Backtesting models**: 5-window rolling cross-validation across 7+ models
- **Evaluating performance**: MAE, RMSE, MAPE metrics with standard deviations
- **Saving configurations**: Export best model setups for reproducibility
- **Validating data**: Comprehensive input validation with helpful error messages

Perfect for:
- 📊 **Data Scientists** exploring forecasting approaches
- 🎓 **Students** learning time series analysis
- 🏢 **Professionals** building production forecasting systems
- 📈 **Businesses** comparing model performance before deployment

## ✨ Features

### 🔄 Three Forecasting Paradigms

| Framework | Models | Use Case |
|-----------|--------|----------|
| **StatsForecast** | AutoARIMA, AutoETS, SeasonalNaive | Classical time series with clear seasonality |
| **MLForecast** | XGBoost, LightGBM, RandomForest | Structured data with external features |
| **NeuralForecast** | LSTM, MLP, NBEATS | Complex patterns, deep learning approach |

### 📊 Three Forecasting Strategies

1. **One-step Forecast** - Iterative refitting (most accurate but slowest)
2. **Multi-step Recursive** - Uses predictions iteratively (realistic deployment)
3. **Multi-output Direct** - All predictions at once (fastest, less error accumulation)

### 🎨 Rich User Interface

- **Data Loading**: Upload CSV or use sample datasets
- **Configuration Management**: Save/load model configs as JSON
- **Backtesting**: Run all models across 5 rolling windows
- **Visualizations**: 
  - Model comparison bar charts
  - Performance heatmaps
  - Combined forecast plots
  - Normalized metric comparisons
- **Metrics Dashboard**: MAE, RMSE, MAPE with error bands
- **Export Options**: CSV rankings, TXT reports, JSON configs

### ✅ Robust Validation

- CSV format verification
- Date/time parsing
- Numeric value conversion
- Missing value detection
- Data quality assessment
- Frequency auto-detection
- Parameter consistency checking

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone or navigate to project directory
cd path/to/nixtla

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Usage Flow

1. **Load Data**
   - Upload CSV with date and value columns
   - Or select a pre-loaded sample dataset

2. **Configure**
   - Choose forecasting module (StatsForecast/MLForecast/NeuralForecast)
   - Select model type
   - Choose forecasting strategy
   - Set parameters (horizon, train/test split, etc.)

3. **Run Forecast**
   - Single model: Click "Run Forecast"
   - Multiple models: Enable "Compare all models" and click "Run Backtesting Comparison"

4. **Analyze Results**
   - View performance metrics
   - Compare models with visualizations
   - Check data quality warnings
   - Export results

5. **Save Configuration**
   - After backtesting, save best model config
   - Load saved configs for reproducible runs

## 📊 Sample Datasets

Pre-loaded datasets for quick testing:

- **AirPassengers** (Monthly, 1949-1960)
  - Classic dataset with trend and seasonality
  - 144 observations

- **Energy Consumption** (Daily, 2023)
  - Household energy usage
  - Weekly pattern, 365 observations

- **Retail Sales** (Weekly, 2021-2023)
  - Retail store sales
  - Yearly seasonality, 156 observations

- **Temperature** (Daily, 2022-2023)
  - Daily temperature
  - Strong yearly pattern, 730 observations

## 🏗️ Architecture

```
streamlit_app.py (Main UI)
├── Config Management
│   └── config_manager.py
├── Data Validation
│   └── validators.py
├── Sample Data
│   └── sample_data.py
└── Forecasting Pipelines
    ├── backtesting.py
    ├── reporting.py
    └── full_pipeline/
        ├── df_statsforecast.py
        ├── df_mlforecast.py
        └── df_neuralforecast.py
```

## 📚 How Nixtla Packages Work Together

```
Input Data
    ↓
Validation (validators.py)
    ↓
Configuration (config_manager.py)
    ↓
Forecasting Pipeline Selection
    ├─ StatsForecast
    │  └─ ARIMA/ETS models
    ├─ MLForecast
    │  └─ Tree-based models
    └─ NeuralForecast
       └─ Deep learning models
    ↓
Backtesting (backtesting.py)
    ├─ 5-window rolling CV
    └─ Aggregate results
    ↓
Reporting (reporting.py)
    ├─ Visualizations
    ├─ Rankings
    └─ Metrics
    ↓
Results Display & Export
```

### StatsForecast
- Handles classical time series decomposition
- Supports ARIMA, ETS with seasonal adjustments
- Fast inference for baseline models

### MLForecast
- Builds lag features automatically
- Trains ensemble models
- Implements multi-output direct forecasting

### NeuralForecast
- Encoder-decoder architectures
- State-of-the-art models (NBEATS, NHITS)
- GPU-compatible for large datasets

## 📈 Performance Metrics

The app evaluates models using:

- **MAE (Mean Absolute Error)** - Average prediction error magnitude
- **RMSE (Root Mean Squared Error)** - Penalizes larger errors more
- **MAPE (Mean Absolute Percentage Error)** - Scale-independent metric

All metrics are calculated with:
- Mean values across all backtest windows
- Standard deviation for uncertainty quantification
- Success rate (% of windows completed successfully)

## 🛠️ Configuration Files

Model configurations are saved as JSON:

```json
{
  "dataset": "AirPassengers",
  "model_name": "StatsForecast-auto_arima",
  "module": "StatsForecast",
  "model_type": "auto_arima",
  "params": {
    "freq": "MS",
    "horizon": 12,
    "season_length": 12
  },
  "metrics": {
    "mae_mean": 12.34,
    "rmse_mean": 15.67,
    "mape_mean": 2.45,
    "success_rate": 1.0
  },
  "timestamp": "2026-04-19T11:40:00.123456"
}
```

## 📦 Dependencies

Core Nixtla Packages:
- `statsforecast>=1.5.0` - Statistical models
- `mlforecast>=0.10.0` - Machine learning models
- `neuralforecast>=1.6.0` - Deep learning models
- `utilsforecast>=0.0.10` - Utilities

ML/DL Libraries:
- `xgboost>=2.0.0` - Gradient boosting
- `lightgbm>=4.0.0` - Light gradient boosting
- `catboost>=1.2.0` - Categorical boosting
- `torch>=2.0.0` - Deep learning
- `pytorch-lightning>=2.0.0` - Training framework

See `requirements.txt` for complete dependencies.

## 🚀 Deployment

### Streamlit Cloud (Recommended)

See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions.

Quick summary:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click
4. Get shareable URL

### Local/Private Deployment

```bash
# Using Gunicorn
gunicorn --workers 1 --worker-class sync --bind 0.0.0.0:8000 streamlit run streamlit_app.py

# Using Docker
docker build -t forecasting-app .
docker run -p 8501:8501 forecasting-app
```

## 📖 Technical Documentation

For detailed architecture and design decisions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🎓 Learning Resources

### Time Series Forecasting
- [Nixtla Documentation](https://nixtla.github.io/statsforecast/)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Deep Learning for Time Series Forecasting](https://www.deeplearningbook.org/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Additional forecasting models (Prophet, AutoML)
- [ ] Multivariate time series support
- [ ] Exogenous variables handling
- [ ] Confidence intervals/prediction bands
- [ ] Additional metrics (sMAPE, MASE, etc.)
- [ ] Model interpretability (SHAP, feature importance)
- [ ] Automated model selection
- [ ] Hyperparameter tuning integration

## 📝 License

MIT License - See LICENSE file for details

## 🏆 Built For

**DATA 5630: Deep Forecasting (Utah State University)**

A production-ready application demonstrating:
- ✅ Multi-paradigm forecasting comparison
- ✅ Comprehensive backtesting framework
- ✅ Reproducible model configurations
- ✅ Professional data validation
- ✅ Publication-ready visualizations
- ✅ Cloud-native deployment

---

**Status**: Production Ready | **Last Updated**: April 2026 | **Maintainer**: Data Science Team
