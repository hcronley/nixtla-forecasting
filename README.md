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

Perfect for data scientists, students, professionals, and businesses comparing forecasting approaches.

## ✨ Features

### 🔄 Three Forecasting Paradigms

| Framework | Models | Use Case |
|-----------|--------|----------|
| **StatsForecast** | AutoARIMA, AutoETS, SeasonalNaive | Classical time series with clear seasonality |
| **MLForecast** | XGBoost, LightGBM, RandomForest | Structured data with external features |
| **NeuralForecast** | LSTM, MLP, NBEATS | Complex patterns, deep learning approach |

### 📊 Three Forecasting Strategies
- **One-step**: Iterative refitting (most accurate but slowest)
- **Multi-step Recursive**: Uses predictions iteratively (realistic deployment)
- **Multi-output Direct**: All predictions at once (fastest)

### 🎨 Rich User Interface
- Data loading (CSV upload or pre-loaded datasets)
- Configuration management (save/load as JSON)
- Backtesting with 5 rolling windows
- Interactive visualizations (Plotly charts, heatmaps, comparisons)
- Performance metrics dashboard (MAE, RMSE, MAPE with error bands)
- Export options (CSV rankings, JSON configs)

### ✅ Robust Validation
CSV format checks, date/time parsing, missing value detection, frequency auto-detection, and parameter consistency checking.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/your-repo.git
cd your-repo/nixtla
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

### Usage Flow

1. **Load Data** - Upload CSV with date and value columns, or select a pre-loaded sample
2. **Configure** - Choose forecasting module, model type, strategy, and parameters
3. **Run Forecast** - Single model or compare all models with backtesting
4. **Analyze Results** - View metrics, compare models, check data quality
5. **Save Configuration** - Export best model config for reproducible runs

## 📊 Sample Datasets

Pre-loaded datasets for quick testing:
- **AirPassengers** (Monthly, 1949-1960): 144 observations with trend and seasonality
- **Energy Consumption** (Daily, 2023): 365 observations with weekly pattern
- **Retail Sales** (Weekly, 2021-2023): 156 observations with yearly seasonality
- **Temperature** (Daily, 2022-2023): 730 observations with yearly pattern

## 📦 Dependencies

**Nixtla Core**: statsforecast, mlforecast, neuralforecast  
**ML/DL**: scikit-learn, xgboost, lightgbm, catboost, torch, pytorch-lightning  
**Data**: pandas, numpy, plotly, python-dateutil  
**Web**: streamlit

See `requirements.txt` for complete dependencies.

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub and click "New app"
4. Select repository, branch, and `streamlit_app.py`
5. Deploy!

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed deployment instructions.

### Local/Private Deployment

```bash
docker build -t forecasting-app .
docker run -p 8501:8501 forecasting-app
```

## 📚 Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Architecture, local testing, deployment guide
- **Code Architecture**: Modular design with separate pipelines for each Nixtla library
- **Rolling Window Backtesting**: 5-window CV for realistic model comparison

## 🎓 Technical Highlights

**Architecture**:
- `backtesting.py`: Rolling window cross-validation (5 windows, all models)
- `reporting.py`: Visualization and model comparison
- `config_manager.py`: Configuration persistence (JSON)
- `validators.py`: Input validation and data quality checks
- `streamlit_app.py`: Interactive UI and orchestration
- `df_statsforecast.py`, `df_mlforecast.py`, `df_neuralforecast.py`: Model wrappers

**Performance**:
- Single forecast: 30-60 seconds
- Backtesting (7 models × 5 windows): 2-3 minutes
- Memory: ~500MB-1GB during backtesting

## 📝 License

MIT License

## 🏆 Built For

**DATA 5630: Deep Forecasting (Utah State University)**

Demonstrates:
- Multi-paradigm forecasting comparison
- Comprehensive backtesting framework
- Reproducible model configurations
- Professional data validation
- Publication-ready visualizations
- Cloud-native deployment

---

**Status**: Production Ready | **Last Updated**: April 2026

For detailed development and deployment instructions, see [DEVELOPMENT.md](DEVELOPMENT.md).
