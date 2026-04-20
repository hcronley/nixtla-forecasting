# 🛠️ Development & Deployment Guide

Complete reference for local development, testing, and deploying the Time Series Forecasting App.

---

## Table of Contents

1. [Local Development](#local-development)
2. [System Architecture](#system-architecture)
3. [Testing Guide](#testing-guide)
4. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
5. [Troubleshooting](#troubleshooting)

---

# Local Development

## Running the App Locally

### Prerequisites

All dependencies installed via `requirements.txt`:

```bash
cd /path/to/nixtla
pip install -r requirements.txt
```

### Start the App

```bash
streamlit run streamlit_app.py
```

Opens at: `http://localhost:8501`

### Custom Port

If port 8501 is busy:

```bash
streamlit run streamlit_app.py --server.port 8502
```

---

# System Architecture

## Overview

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
        │  │ ├─ df_statsforecast.py │ │
        │  │ │  • ARIMA, ETS        │ │
        │  │ │  • One/Multi-step    │ │
        │  │ │                      │ │
        │  │ ├─ df_mlforecast.py    │ │
        │  │ │  • XGBoost, etc      │ │
        │  │ │  • All strategies    │ │
        │  │ │                      │ │
        │  │ └─ df_neuralforecast   │ │
        │  │    • LSTM, MLP, NBEATS │ │
        │  │    • All strategies    │ │
        │  │                        │ │
        │  └──────────────────────┘ │
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
        │  • 4 sample datasets │
        └──────────────────────┘
```

## Module Responsibilities

### Core Modules

**`streamlit_app.py`** (40 KB)
- Main application interface and orchestration
- UI components: sidebar controls, data preview, results display
- Session state management for Streamlit

**`validators.py`** (18 KB)
- `DataValidator` class with static validation methods
- CSV format checks, date parsing, data quality assessment
- Frequency auto-detection and parameter validation
- Strategy: fail-fast on critical errors, collect warnings for non-critical issues

**`backtesting.py`** (18 KB)
- `RollingWindowBacktester` class for 5-window cross-validation
- Creates non-overlapping train/test splits
- Routes to appropriate forecasting module
- Aggregates results and generates rankings

**`reporting.py`** (13 KB)
- `ModelComparator` class for analyzing results
- Generates ranking tables, comparison plots, heatmaps
- Creates performance visualizations with Plotly
- Extracts best model configuration

**`config_manager.py`** (12 KB)
- `ConfigManager` class for JSON-based persistence
- Save/load/list/delete configuration operations
- Configuration structure: dataset, model, params, metrics, timestamp

### Forecasting Modules

**`df_statsforecast.py`** (16 KB)
- `StatsforecastForecaster` class
- Models: ARIMA, AutoARIMA, AutoETS, Naive, SeasonalNaive, RandomWalkWithDrift
- Strategies: one-step, multi-step (recursive)
- Multi-output NOT SUPPORTED (statistical models are conditional on history)

**`df_mlforecast.py`** (21 KB)
- `MLForecastForecaster` class
- Models: XGBoost, LightGBM, CatBoost, RandomForest, LinearRegression
- All strategies: one-step, multi-step, multi-output
- Features: automatic lag generation, date features, target transformations

**`df_neuralforecast.py`** (22 KB)
- `NeuralForecastForecaster` class
- Models: RNN, LSTM, GRU, MLP, NBEATS, NHITS, TCN
- All strategies: one-step, multi-step, multi-output
- Training: PyTorch + PyTorch Lightning, GPU-compatible

### Data Layer

**`sample_data.py`** (7.6 KB)
- Pre-loaded datasets for quick testing
- AirPassengers, Energy Consumption, Retail Sales, Temperature
- Includes metadata: frequency, season_length, recommended_horizon

## Data Flow

### Single Forecast

```
User Config (Module, Model, Strategy, Parameters)
    ↓
run_forecast() routes to appropriate module
    ↓
Module trains on train_df, forecasts on test_df
    ↓
Calculate metrics (MAE, RMSE, MAPE)
    ↓
Visualize and display results
```

### Backtesting Comparison

```
User enables "Compare all models" + clicks "Run Forecast"
    ↓
build_backtest_models() selects 7 models
    ↓
RollingWindowBacktester.create_windows() → 5 windows
    ↓
For each window:
  For each model:
    Run through appropriate module
    Collect metrics (MAE, RMSE, MAPE)
    ↓
Aggregate metrics across windows
    ↓
ModelComparator generates visualizations
    ↓
Display rankings, heatmaps, comparisons
```

---

# Testing Guide

## Comprehensive Testing Checklist

### Test 1: UI and Navigation
- [ ] App loads without errors
- [ ] Sidebar displays correctly
- [ ] Page title and description visible
- [ ] Sample data loads when clicked

### Test 2: Data Quality Validation (with sample data)
- [ ] Data preview shows correct number of observations
- [ ] Quality warnings display if issues exist
- [ ] Validation checks pass (dates, values, length)
- [ ] Frequency detected correctly

### Test 3: Single Forecast (StatsForecast)
**Steps:**
1. Load sample data (AirPassengers)
2. Keep default settings: StatsForecast - Auto ETS, Horizon: 12
3. Click "Run Forecast"

**Expected Results:**
- Executes without error (5-10 seconds)
- Displays forecast table with dates and predictions
- Plots actual vs predicted
- Shows metrics: MAE, RMSE, MAPE

### Test 4: Single Forecast (MLForecast)
**Steps:**
1. Change model to "MLForecast - XGBoost"
2. Click "Run Forecast"

**Expected Results:**
- Different forecast from StatsForecast
- Metrics displayed
- Visualization updates

### Test 5: Backtesting Comparison
**Steps:**
1. Check "Run Full Model Comparison (Backtesting)"
2. Click "Run Forecast"

**Expected Results (60-90 seconds):**
- Model Rankings Table (top performers by MAE)
- Comparison bar chart
- Metric heatmap (window × model performance)
- Performance summary for best model

### Test 6: Config Save/Load
**Steps:**
1. After running forecast, click "Save Configuration"
2. Download JSON file
3. Click "Load Previous Configuration"
4. Upload the JSON file

**Expected Results:**
- JSON downloads successfully
- Config loads with all parameters filled
- Model type and parameters visible

### Test 7: CSV Upload
**Steps:**
1. Prepare CSV with columns: `ds`, `y`
2. Click "Upload CSV File"
3. Upload and run forecast

**Expected Results:**
- CSV loads without error
- Data quality validation runs
- Forecast works on custom data
- Metrics calculated correctly

### Test 8: Error Handling
**Steps:**
1. Upload CSV missing 'ds' column → clear error message
2. Upload CSV with only 5 rows → warning about data length
3. Upload non-CSV file → graceful rejection

**Expected Results:**
- Clear error messages
- App doesn't crash
- User can retry

## Performance Benchmarks

**Single Forecast Times:**
- StatsForecast: 3-8 seconds
- MLForecast: 5-15 seconds
- NeuralForecast: 10-30 seconds

**Backtesting (7 models, 5 windows):**
- Runtime: 60-120 seconds
- Memory: 600MB-1GB
- First run: Slower (model loading)
- Subsequent: Faster (caching)

## Running Test Suite

```bash
python test_app.py
```

Tests:
- Data loading and validation
- Single model forecasting
- Backtesting framework
- Reporting and visualization
- Configuration persistence
- Module imports

---

# Streamlit Cloud Deployment

## Prerequisites

- GitHub account
- Streamlit Cloud account (sign up at [streamlit.io/cloud](https://streamlit.io/cloud))
- Code pushed to GitHub repository

## Step 1: GitHub Repository Setup

### Option A: Create Personal Repository (Recommended)

```bash
# Create new repo at https://github.com/new
# Name: nixtla-forecasting

# Add remote and push
git remote set-url origin https://github.com/YOUR_USERNAME/nixtla-forecasting.git
git push -u origin main

# Verify
git remote -v
```

### Option B: Fork Existing Repository

```bash
# Fork on GitHub, then:
git remote set-url origin https://github.com/YOUR_USERNAME/Deep_forecasting-USU.git
git push -u origin main
```

## Step 2: Verify Files on GitHub

```bash
git status
git log --oneline -5
```

**Required files:**
- streamlit_app.py
- requirements.txt
- full_pipeline/*.py (all modules)
- README.md

## Step 3: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up"
3. Choose "Continue with GitHub"
4. Authorize Streamlit

## Step 4: Deploy Application

1. Click **"New app"** on Streamlit Cloud dashboard
2. Fill deployment form:
   - **Repository**: `YOUR_USERNAME/nixtla-forecasting`
   - **Branch**: `main`
   - **File path**: `streamlit_app.py` (or full path if nested)
3. Click **"Deploy"**
4. Wait 2-3 minutes for deployment

## Step 5: Access Your App

Once deployed (green checkmark appears):

```
https://YOUR_USERNAME-REPO_NAME.streamlit.app
```

## Step 6: Test Live Application

- ✅ Load sample data
- ✅ Single forecast
- ✅ Backtesting comparison
- ✅ Config save/load
- ✅ CSV upload

## Advanced Configuration

### Environment Secrets

For sensitive information:

1. Click app menu (⋮) → Settings → Secrets
2. Add TOML format:

```toml
api_key = "your_secret_key"
database_url = "postgresql://..."
```

Access in code:

```python
import streamlit as st
secret = st.secrets["api_key"]
```

### Custom Configuration

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"

[server]
maxUploadSize = 200
```

### Resource Limits (Free Tier)

- 3 concurrent apps
- 1 GB memory per app
- Shared CPU resources
- Public URL

For production: Consider Streamlit Cloud Pro or private deployment.

## Monitoring & Updates

### View Logs

1. Click app menu (⋮) → Manage app
2. Scroll to "Logs" section

### Push Updates

```bash
git add .
git commit -m "Update feature X"
git push origin main
```

Streamlit redeploys automatically within seconds!

### Clear Cache

```bash
# In code:
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()
```

Or click app menu (⋮) → Settings → Reboot app

## Sharing Your App

### Public URL
```
https://YOUR_USERNAME-REPO_NAME.streamlit.app
```

### Embed in Website

```html
<iframe
  src="https://YOUR_USERNAME-REPO_NAME.streamlit.app"
  height="500"
  width="100%">
</iframe>
```

### Add to Portfolio

```markdown
## Time Series Forecasting App

**Live Demo**: [Nixtla Forecasting App](https://YOUR_USERNAME-REPO_NAME.streamlit.app)

**Technologies**: Python, Streamlit, Nixtla (StatsForecast, MLForecast, NeuralForecast)

**Features**:
- Multi-model forecasting with 17+ models
- 5-window rolling cross-validation
- Interactive visualizations
- Model configuration persistence
- Production-ready architecture
```

---

# UI/Formatting

## Applied Fixes

### Deprecated Streamlit Parameters
- ✅ Replaced `width=True` with `use_container_width=True` (Streamlit 1.28+)
- ✅ Applied to: plotly_chart, dataframe, buttons, download buttons

### Responsive Layout
- ✅ All Plotly charts render with proper sizing
- ✅ Buttons stretch to container width
- ✅ DataFrames display full width
- ✅ No overlapping elements

### Compatibility
- ✅ Streamlit >= 1.28.0
- ✅ Python 3.8+
- ✅ All major browsers

---

# Troubleshooting

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"

**Solution**: Use different port
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: "torch or pytorch-lightning not found"

**Solution**: Install PyTorch
```bash
pip install torch pytorch-lightning
```

### Issue: App takes >1 minute to load

**Solution**: 
- First run installs packages (normal)
- Subsequent runs use cache
- Can't optimize further on free tier

### Issue: Backtesting takes too long or crashes

**Solution**:
- Use smaller dataset
- Reduce number of windows
- Disable NeuralForecast (heaviest)
- Check available memory

### Issue: "No such file or directory" during deployment

**Solution**: Verify file path in Streamlit settings includes correct directory structure

### Issue: Changes not appearing after push

**Solution**:
- Force refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- Wait 30 seconds for deployment
- Clear browser cache

### Issue: Memory exceeded on Streamlit Cloud

**Solution**:
- Use smaller sample dataset
- Reduce backtest windows
- Skip neural models temporarily

## Getting Help

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Nixtla Docs**: [nixtla.github.io](https://nixtla.github.io)
- **GitHub Issues**: Create issue in your repo

---

**Last Updated**: April 2026

For quick reference, see [README.md](README.md).
