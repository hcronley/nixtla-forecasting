"""
Time Series Forecasting App with Streamlit
Supports StatsForecast, MLForecast, and NeuralForecast modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add the full_pipeline directory to path to import forecasting modules
pipeline_path = Path(__file__).parent / 'full_pipeline'
sys.path.insert(0, str(pipeline_path))

# Import forecasting modules
try:
    from df_statsforecast import StatsforecastForecaster, train_test_split_ts
    STATSFORECAST_AVAILABLE = True
except ImportError as e:
    STATSFORECAST_AVAILABLE = False
    st.error(f"StatsForecast module not available: {e}")

try:
    from df_mlforecast import MLForecastForecaster
    MLFORECAST_AVAILABLE = True
except ImportError as e:
    MLFORECAST_AVAILABLE = False
    st.error(f"MLForecast module not available: {e}")

try:
    from df_neuralforecast import NeuralForecastForecaster
    NEURALFORECAST_AVAILABLE = True
except ImportError as e:
    NEURALFORECAST_AVAILABLE = False
    st.error(f"NeuralForecast module not available: {e}")

# Import sample data
try:
    from sample_data import DATASETS, get_sample_data
    SAMPLE_DATA_AVAILABLE = True
except ImportError:
    SAMPLE_DATA_AVAILABLE = False
    st.warning("Sample data module not available. Only file upload will work.")

# Import stock data client (yfinance - free)
try:
    from yfinance_client import fetch_stock_data
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Import backtesting and reporting modules
try:
    from backtesting import RollingWindowBacktester
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from reporting import ModelComparator, create_summary_report
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False

# Import config manager
try:
    from config_manager import ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

# Import validators
try:
    from validators import DataValidator, validate_upload_file, validate_and_prepare_data, get_validation_warnings
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
STATSFORECAST_MODELS = [
    'arima', 'auto_arima', 'auto_ets', 'naive',
    'seasonal_naive', 'random_walk_with_drift', 'window_average', 'seasonal_window_average'
]

MLFORECAST_MODELS = [
    'xgboost', 'lightgbm', 'random_forest', 'catboost', 'linear'
]

NEURALFORECAST_MODELS = [
    'mlp', 'rnn', 'lstm', 'gru'
]

RECURRENT_MODELS = ['rnn', 'lstm', 'gru']

# Frequency options
FREQUENCY_OPTIONS = {
    'Hourly': 'H',
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly (Start)': 'MS',
    'Monthly (End)': 'M',
    'Quarterly (Start)': 'QS',
    'Quarterly (End)': 'Q',
    'Yearly (Start)': 'YS',
    'Yearly (End)': 'Y'
}


def initialize_session_state():
    """Initialize session state variables"""
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'loaded_config' not in st.session_state:
        st.session_state.loaded_config = None
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager() if CONFIG_MANAGER_AVAILABLE else None


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate uploaded CSV file"""
    if VALIDATORS_AVAILABLE:
        is_valid, message, df = validate_upload_file(uploaded_file)
        if is_valid:
            st.sidebar.success(message)
            return df
        else:
            st.sidebar.error(message)
            return None
    else:
        # Fallback to basic loading
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None


def validate_and_prepare_data_wrapper(df: pd.DataFrame, date_col: str, value_col: str) -> Optional[pd.DataFrame]:
    """Validate and prepare data with comprehensive checks"""
    if VALIDATORS_AVAILABLE:
        is_valid, message, prepared_df = validate_and_prepare_data(df, date_col, value_col)

        # Display validation messages
        if is_valid:
            st.sidebar.success(message.split('\n')[0])  # Show first message (success)
            # Show warnings as info
            for line in message.split('\n')[1:]:
                if line.startswith('⚠️'):
                    st.sidebar.info(line)
        else:
            # Show critical errors
            for line in message.split('\n'):
                if line.startswith('❌'):
                    st.sidebar.error(line)
                elif line.startswith('⚠️'):
                    st.sidebar.info(line)

        return prepared_df

    else:
        # Fallback to basic validation
        try:
            if date_col not in df.columns:
                st.error(f"Column '{date_col}' not found in data")
                return None
            if value_col not in df.columns:
                st.error(f"Column '{value_col}' not found in data")
                return None

            prepared_df = pd.DataFrame({
                'ds': pd.to_datetime(df[date_col]),
                'y': pd.to_numeric(df[value_col])
            })

            prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)

            if prepared_df['y'].isna().any():
                st.warning(f"Found {prepared_df['y'].isna().sum()} missing values. They will be removed.")
                prepared_df = prepared_df.dropna()

            return prepared_df

        except Exception as e:
            st.error(f"Error preparing data: {e}")
            return None


def plot_time_series(df: pd.DataFrame, title: str = "Time Series Data"):
    """Plot time series using Plotly"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_forecast_results(results: Dict[str, Any], title: str = "Forecast Results"):
    """Plot forecast vs actual values"""
    forecasts_df = results['forecasts'].copy()

    fig = go.Figure()

    # Only plot if we have actual values (not NaN)
    if 'y_true' in forecasts_df.columns and not forecasts_df['y_true'].isna().all():
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=forecasts_df['ds'],
            y=forecasts_df['y_true'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

    # Plot predictions
    fig.add_trace(go.Scatter(
        x=forecasts_df['ds'],
        y=forecasts_df['y_pred'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def display_metrics(metrics: Dict[str, float]):
    """Display metrics in columns"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value=f"{metrics['mae']:.4f}",
            help="Average absolute difference between predicted and actual values"
        )

    with col2:
        st.metric(
            label="RMSE (Root Mean Squared Error)",
            value=f"{metrics['rmse']:.4f}",
            help="Square root of average squared differences (penalizes larger errors)"
        )

    with col3:
        st.metric(
            label="MAPE (Mean Absolute % Error)",
            value=f"{metrics['mape']:.2f}%",
            help="Average percentage error (scale-independent metric)"
        )


def build_backtest_models() -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary of models for backtesting from all available modules.

    Returns
    -------
    dict
        Dictionary mapping model names to their config
    """
    models = {}

    if STATSFORECAST_AVAILABLE:
        stats_models = ['auto_arima', 'auto_ets', 'seasonal_naive']
        for model in stats_models:
            models[f"StatsForecast-{model}"] = {
                'module': 'StatsForecast',
                'type': model,
                'params': {}
            }

    if MLFORECAST_AVAILABLE:
        ml_models = ['random_forest', 'lightgbm']
        for model in ml_models:
            models[f"MLForecast-{model}"] = {
                'module': 'MLForecast',
                'type': model,
                'params': {}
            }

    if NEURALFORECAST_AVAILABLE:
        neural_models = ['mlp', 'lstm']
        for model in neural_models:
            models[f"NeuralForecast-{model}"] = {
                'module': 'NeuralForecast',
                'type': model,
                'params': {}
            }

    return models


def display_backtest_results(backtest_results: Dict[str, Any]):
    """
    Display backtesting results with visualizations and tables.

    Parameters
    ----------
    backtest_results : dict
        Results from RollingWindowBacktester
    """
    if not REPORTING_AVAILABLE:
        st.error("Reporting module not available")
        return

    comparator = ModelComparator(backtest_results)

    # Summary section
    st.subheader("📊 Model Comparison Summary")

    # Best model summary
    best_summary = comparator.get_summary_table()

    # Top row: Best model and MAE
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(
            "🥇 Best Model",
            best_summary['Model'],
            help="Top-ranked model by MAE"
        )

    with col2:
        st.metric(
            "MAE",
            f"{best_summary['MAE']:.4f}",
            f"±{best_summary['Std (MAE)']:.4f}"
        )

    # Bottom row: RMSE and Success Rate
    col3, col4 = st.columns([1, 1])
    with col3:
        st.metric(
            "RMSE",
            f"{best_summary['RMSE']:.4f}",
            f"±{best_summary['Std (RMSE)']:.4f}"
        )

    with col4:
        st.metric(
            "Success Rate",
            f"{best_summary['Success Rate']*100:.0f}%",
            help="% of windows completed successfully"
        )

    # Model Rankings Table
    st.subheader("📋 Model Rankings")
    ranking_table = comparator.create_metric_comparison_table(top_n=10)
    st.dataframe(ranking_table, use_container_width=True, hide_index=True)

    # Save best model config
    if CONFIG_MANAGER_AVAILABLE:
        st.subheader("💾 Save Best Model Configuration")
        col1, col2 = st.columns([3, 1])

        with col1:
            config_name = st.text_input(
                "Configuration name (for saving):",
                value="best_model_config",
                help="Give this configuration a memorable name"
            )

        with col2:
            if st.button("💾 Save Config", use_container_width=True):
                if st.session_state.config_manager.export_best_model_config(
                    backtest_results,
                    filename=config_name,
                    dataset_name="Backtest Dataset"
                ):
                    st.success(f"✅ Configuration saved as '{config_name}'")
                else:
                    st.error("❌ Failed to save configuration")

    # Download results
    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        ranking_csv = ranking_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Rankings (CSV)",
            data=ranking_csv,
            file_name="backtest_rankings.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        report = create_summary_report(backtest_results)
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report,
            file_name="backtest_summary_report.txt",
            mime="text/plain",
            use_container_width=True
        )



    # 🏁 HORSE RACE VISUALIZATIONS - TOP 5 MODEL COMPARISON
    st.header("🏁 Top 5 Model Horse Race")
    st.markdown("Compare the top 5 forecasting models across metrics and windows")
    
    # Create 2-column layout for first two visualizations
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Rankings with Confidence Bands")
        fig_ranked = comparator.create_top5_ranked_comparison(top_n=5)
        st.plotly_chart(fig_ranked, use_container_width=True)
    
    with col2:
        st.subheader("How Rankings Changed Across Windows")
        fig_progression = comparator.create_ranking_progression_heatmap(top_n=5)
        st.plotly_chart(fig_progression, use_container_width=True)
    
    # Full-width waterfall chart
    st.subheader("Performance Gap Analysis")
    st.caption("Shows how much each model's performance differs from the top model")
    fig_waterfall = comparator.create_metric_race_waterfall(metric='mae_mean', top_n=5)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Expandable portfolio dashboard for presentations
    with st.expander("📊 Comprehensive Portfolio Dashboard (All Metrics)"):
        st.info("Use this dashboard for presentations and portfolio showcase - combines all key visualizations")
        fig_portfolio = comparator.create_top5_portfolio_summary()
        st.plotly_chart(fig_portfolio, use_container_width=True, height=900)

def fix_forecast_actuals(results: Dict[str, Any], test_df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """
    Fix actual values in forecast results by properly aligning with test data.
    This completely rebuilds the forecasts DataFrame to ensure correct alignment.
    """
    if results is None or test_df is None:
        return results

    forecasts_df = results['forecasts'].copy()

    # Get the first 'horizon' rows from test data
    test_subset = test_df.head(horizon).reset_index(drop=True)

    # Determine how many values we can match
    n_match = min(len(forecasts_df), len(test_subset))

    # Create a new dataframe with correct alignment
    new_forecasts = pd.DataFrame({
        'ds': test_subset['ds'].iloc[:n_match].values,
        'y_true': test_subset['y'].iloc[:n_match].values,
        'y_pred': forecasts_df['y_pred'].iloc[:n_match].values
    })

    # Add unique_id if it exists
    if 'unique_id' in forecasts_df.columns:
        new_forecasts['unique_id'] = forecasts_df['unique_id'].iloc[:n_match].values

    # Recalculate metrics with correct alignment
    actuals = new_forecasts['y_true'].values
    predictions = new_forecasts['y_pred'].values

    errors = actuals - predictions
    metrics = {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mape': float(np.mean(np.abs(errors / actuals)) * 100) if np.all(actuals != 0) else np.nan
    }

    return {
        'forecasts': new_forecasts,
        'metrics': metrics
    }


def run_forecast(
    module_type: str,
    model_type: str,
    strategy: str,
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run forecasting based on selected module and strategy"""

    try:
        # Create forecaster based on module type
        if module_type == "StatsForecast":
            if not STATSFORECAST_AVAILABLE:
                st.error("StatsForecast module is not available")
                return None

            forecaster = StatsforecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                season_length=params['season_length'],
                **params.get('model_params', {})
            )

        elif module_type == "MLForecast":
            if not MLFORECAST_AVAILABLE:
                st.error("MLForecast module is not available")
                return None

            forecaster = MLForecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                lags=params.get('lags'),
                **params.get('model_params', {})
            )

        elif module_type == "NeuralForecast":
            if not NEURALFORECAST_AVAILABLE:
                st.error("NeuralForecast module is not available")
                return None

            forecaster = NeuralForecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                input_size=params.get('input_size', 12),
                horizon=params['horizon'],
                **params.get('model_params', {})
            )

        else:
            st.error(f"Unknown module type: {module_type}")
            return None

        # Run forecast based on strategy
        # Note: Data is already in standard format with columns 'ds' and 'y'
        if strategy == "One-step forecast":
            if test_df is None:
                st.error("Test data is required for one-step forecasting")
                return None
            results = forecaster.one_step_forecast(
                train_df,
                test_df,
                target_col='y',
                date_col='ds',
                unique_id='1'
            )

        elif strategy == "Multi-step recursive":
            if module_type == "NeuralForecast" and model_type in RECURRENT_MODELS:
                results = forecaster.multi_step_forecast(
                    train_df,
                    params['horizon'],
                    target_col='y',
                    date_col='ds',
                    unique_id='1',
                    test_df=test_df,
                    use_recurrent=True
                )
            else:
                results = forecaster.multi_step_forecast(
                    train_df,
                    params['horizon'],
                    target_col='y',
                    date_col='ds',
                    unique_id='1',
                    test_df=test_df
                )

        elif strategy == "Multi-output direct":
            # Check if multi-output is supported
            if module_type == "StatsForecast":
                st.error("❌ Multi-output forecasting is not supported for StatsForecast models. "
                        "Statistical models (ARIMA/ETS) can only do recursive forecasting. "
                        "Please select 'Multi-step recursive' or use ML/Neural models.")
                return None

            results = forecaster.multi_output_forecast(
                train_df,
                params['horizon'],
                target_col='y',
                date_col='ds',
                unique_id='1',
                test_df=test_df
            )

        else:
            st.error(f"Unknown strategy: {strategy}")
            return None

        # Fix actual values alignment for multi-step and multi-output strategies
        if strategy in ["Multi-step recursive", "Multi-output direct"] and test_df is not None:
            results = fix_forecast_actuals(results, test_df, params['horizon'])

        return results

    except NotImplementedError as e:
        st.error(f"❌ {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        st.exception(e)
        return None


def main():
    """Main Streamlit app"""
    initialize_session_state()

    # Header
    st.title("📈 Time Series Forecasting App")
    st.markdown("""
    This app supports three forecasting paradigms using Nixtla's ecosystem:
    - **StatsForecast**: Statistical models (ARIMA, ETS, etc.)
    - **MLForecast**: Machine learning models (XGBoost, LightGBM, etc.)
    - **NeuralForecast**: Deep learning models (LSTM, NBEATS, etc.)
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Config Management
    if CONFIG_MANAGER_AVAILABLE:
        st.sidebar.subheader("0. Load Previous Config (Optional)")
        with st.sidebar.expander("📂 Load Configuration"):
            config_list = st.session_state.config_manager.list_configs()
            if not config_list.empty:
                selected_config = st.selectbox(
                    "Select a saved configuration:",
                    options=config_list['Filename'].tolist(),
                    key="config_selector"
                )

                if st.button("Load Selected Config", use_container_width=True):
                    loaded = st.session_state.config_manager.load_config(selected_config)
                    if loaded:
                        st.session_state.loaded_config = loaded
                        st.success(f"✅ Loaded config: {selected_config}")
                        st.info(st.session_state.config_manager.get_config_summary(loaded))
            else:
                st.info("No saved configurations yet. Run backtesting and save the best model config!")

    # Data Source Selection
    st.sidebar.subheader("1. Data Source")
    data_source_options = ["Upload CSV", "Sample Dataset"]

    if YFINANCE_AVAILABLE:
        data_source_options.append("Stock Data (Yahoo Finance)")

    data_source = st.sidebar.radio(
        "Choose data source:",
        data_source_options,
        help="Upload CSV, use sample data, or fetch real stock data"
    )

    data = None
    metadata = {}

    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV file with datetime and numeric columns"
        )

        if uploaded_file is not None:
            data = load_uploaded_file(uploaded_file)

            if data is not None:
                st.sidebar.success(f"✅ Loaded {len(data)} rows")

                # Column selection
                date_col = st.sidebar.selectbox(
                    "Select date column:",
                    options=data.columns.tolist()
                )
                value_col = st.sidebar.selectbox(
                    "Select value column:",
                    options=data.columns.tolist()
                )

                # Prepare data
                data = validate_and_prepare_data_wrapper(data, date_col, value_col)

                if data is not None:
                    st.session_state.current_data = data
                    st.session_state.data_loaded = True

    elif data_source == "Sample Dataset":
        if SAMPLE_DATA_AVAILABLE:
            dataset_name = st.sidebar.selectbox(
                "Select dataset:",
                options=list(DATASETS.keys()),
                help="Pre-loaded example datasets"
            )

            if dataset_name:
                data, metadata = get_sample_data(dataset_name)
                st.sidebar.success(f"✅ Loaded {len(data)} rows")
                st.sidebar.info(f"📝 {metadata['description']}")
                st.session_state.current_data = data
                st.session_state.data_loaded = True
        else:
            st.sidebar.error("Sample data module not available")
            
    elif data_source == "Stock Data (Yahoo Finance)":
        st.sidebar.info("📊 Historical stock data from Yahoo Finance. No API key required.")

        symbol = st.sidebar.text_input("Stock Symbol:", value="AAPL", help="e.g., AAPL, TSLA, MSFT, GOOGL")

        resolution_map = {
            'Daily': 'D',
            'Weekly': 'W',
            'Monthly': 'M'
        }

        res_label = st.sidebar.selectbox("Resolution:", options=list(resolution_map.keys()), index=0)
        resolution = resolution_map[res_label]

        # History period selection
        period = st.sidebar.radio(
            "📅 History Period:",
            ["1 Month", "3 Months", "1 Year", "5 Years", "10 Years", "Custom"],
            horizontal=True
        )

        period_map = {
            "1 Month": 30,
            "3 Months": 90,
            "1 Year": 365,
            "5 Years": 1825,
            "10 Years": 3650
        }

        if period == "Custom":
            days_back = st.sidebar.slider("Days:", 7, 3650, 365)
        else:
            days_back = period_map[period]

        if st.sidebar.button("Fetch Stock Data"):
            try:
                with st.spinner(f"Fetching data for {symbol}..."):
                    data = fetch_stock_data(symbol, resolution, days_back)

                    if data is not None and not data.empty:
                        st.sidebar.success(f"✅ Loaded {len(data)} rows for {symbol}")

                        # Set metadata for frequency
                        freq_map = {'D': 'D', 'W': 'W', 'M': 'M'}
                        metadata = {
                            'freq': freq_map.get(resolution, 'D'),
                            'season_length': 7 if resolution == 'D' else (52 if resolution == 'W' else 12),
                            'description': f"Historical {res_label} data for {symbol}",
                            'recommended_horizon': 14 if resolution == 'D' else (13 if resolution == 'W' else 12)
                        }

                        st.session_state.current_data = data
                        st.session_state.data_loaded = True
                    else:
                        st.sidebar.error("No data returned.")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

        # Keep data if already loaded
        if st.session_state.data_loaded and st.session_state.current_data is not None:
            data = st.session_state.current_data

    # Only show configuration if data is loaded
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        data = st.session_state.current_data

        # Module Selection
        st.sidebar.subheader("2. Forecasting Module")
        available_modules = []
        if STATSFORECAST_AVAILABLE:
            available_modules.append("StatsForecast")
        if MLFORECAST_AVAILABLE:
            available_modules.append("MLForecast")
        if NEURALFORECAST_AVAILABLE:
            available_modules.append("NeuralForecast")

        if not available_modules:
            st.error("No forecasting modules available. Please install required packages.")
            return

        module_type = st.sidebar.radio(
            "Select module:",
            options=available_modules,
            help="StatsForecast=Statistical, MLForecast=ML, NeuralForecast=DL"
        )

        # Model Selection
        st.sidebar.subheader("3. Model Selection")
        if module_type == "StatsForecast":
            model_type = st.sidebar.selectbox("Select model:", STATSFORECAST_MODELS)
        elif module_type == "MLForecast":
            model_type = st.sidebar.selectbox("Select model:", MLFORECAST_MODELS)
        else:  # NeuralForecast
            model_type = st.sidebar.selectbox("Select model:", NEURALFORECAST_MODELS)

        # Strategy Selection
        st.sidebar.subheader("4. Forecasting Strategy")
        strategy = st.sidebar.radio(
            "Select strategy:",
            [
                "One-step forecast",
                "Multi-step recursive",
                "Multi-output direct"
            ],
            help="One-step=iterative (slowest, most accurate), Recursive=default, Direct=all at once"
        )

        # Show warning for StatsForecast + Multi-output
        if module_type == "StatsForecast" and strategy == "Multi-output direct":
            st.sidebar.warning("⚠️ Multi-output is NOT supported for StatsForecast models")

        # Backtesting Option
        st.sidebar.subheader("5. Model Comparison (Optional)")
        enable_backtesting = st.sidebar.checkbox(
            "🔄 Compare all models across 5 windows",
            value=False,
            help="Run backtesting to compare all available models from all three libraries"
        )

        # Parameters
        st.sidebar.subheader("6. Parameters" if enable_backtesting else "5. Parameters")

        # Train/Test Split
        test_size = st.sidebar.slider(
            "Test set size (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )

        test_size_n = int(len(data) * test_size / 100)
        train_df, test_df = train_test_split_ts(data, test_size=test_size_n)

        st.sidebar.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

        # Frequency
        if metadata and 'freq' in metadata:
            default_freq = metadata['freq']
        else:
            default_freq = 'MS'

        # Find the index of default frequency
        freq_values = list(FREQUENCY_OPTIONS.values())
        try:
            freq_index = freq_values.index(default_freq)
        except ValueError:
            freq_index = 3  # Default to Monthly Start

        freq_label = st.sidebar.selectbox(
            "Frequency:",
            options=list(FREQUENCY_OPTIONS.keys()),
            index=freq_index,
            help="Time series frequency"
        )
        freq = FREQUENCY_OPTIONS[freq_label]

        # Season length
        if metadata and 'season_length' in metadata:
            default_season = metadata['season_length']
        else:
            default_season = 12

        season_length = st.sidebar.number_input(
            "Season length:",
            min_value=1,
            max_value=365,
            value=default_season,
            help="Number of periods in a season (e.g., 12 for monthly data with yearly seasonality)"
        )

        # Forecast horizon
        if metadata and 'recommended_horizon' in metadata:
            default_horizon = metadata['recommended_horizon']
        else:
            default_horizon = min(12, len(test_df))

        horizon = st.sidebar.number_input(
            "Forecast horizon (h):",
            min_value=1,
            max_value=len(test_df),
            value=default_horizon,
            help="Number of periods to forecast ahead"
        )

        # Module-specific parameters
        model_params = {}
        input_size = min(12, len(train_df) // 2)  # Default for all modules, can be overridden

        if module_type == "MLForecast":
            with st.sidebar.expander("ML-Specific Parameters"):
                lags_input = st.text_input(
                    "Lags (comma-separated):",
                    value="1,12",
                    help="Lag features to use (e.g., 1,12 for lag-1 and lag-12)"
                )
                try:
                    lags = [int(x.strip()) for x in lags_input.split(',')]
                except:
                    lags = [1, 12]

        elif module_type == "NeuralForecast":
            with st.sidebar.expander("Neural-Specific Parameters"):
                input_size = st.number_input(
                    "Input size (lookback window):",
                    min_value=1,
                    max_value=min(100, len(train_df)),
                    value=min(12, len(train_df) // 2),
                    help="Number of past observations to use"
                )

                hidden_size = st.number_input(
                    "Hidden size:",
                    min_value=4,
                    max_value=128,
                    value=16,
                    help="Size of hidden layers"
                )

                max_steps = st.number_input(
                    "Max training steps:",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Number of training epochs"
                )

                model_params = {
                    'encoder_hidden_size': hidden_size,
                    'max_steps': max_steps
                }

        # Prepare parameters dictionary
        params = {
            'freq': freq,
            'season_length': season_length,
            'horizon': horizon,
            'model_params': model_params
        }

        if module_type == "MLForecast":
            params['lags'] = lags

        if module_type == "NeuralForecast":
            params['input_size'] = input_size

        # Main area - Data Preview
        st.header("📊 Data Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Summary")
            st.write(f"**Total observations:** {len(data)}")
            st.write(f"**Date range:** {data['ds'].min().date()} to {data['ds'].max().date()}")
            st.write(f"**Value range:** {data['y'].min():.2f} to {data['y'].max():.2f}")
            st.write(f"**Mean:** {data['y'].mean():.2f}")
            st.write(f"**Std Dev:** {data['y'].std():.2f}")

        with col2:
            st.subheader("Train/Test Split")
            st.write(f"**Training set:** {len(train_df)} observations")
            st.write(f"**Test set:** {len(test_df)} observations")
            st.write(f"**Split ratio:** {100-test_size}/{test_size}")

        # Plot time series
        st.subheader("Time Series Plot")
        fig = plot_time_series(data)
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("View Data Table"):
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**First 10 rows:**")
                st.dataframe(data.head(10))
            with col_right:
                st.write("**Last 10 rows:**")
                st.dataframe(data.tail(10))

        # Data Quality Warnings
        if VALIDATORS_AVAILABLE:
            warnings = get_validation_warnings(
                data, 'ds', 'y',
                len(train_df), len(test_df),
                horizon
            )
            if warnings:
                with st.expander("⚠️ Data Quality Warnings"):
                    for warning in warnings:
                        st.warning(warning)

        # Run Forecast Button
        st.header("🚀 Run Forecast")

        if enable_backtesting:
            col1, col2 = st.columns([2, 1])
            run_forecast_btn = col1.button("Run Backtesting Comparison", type="primary", use_container_width=True)
            clear_backtest_btn = col2.button("Clear Results", use_container_width=True)

            if clear_backtest_btn:
                st.session_state.backtest_results = None
                st.session_state.forecast_results = None
                st.rerun()

            if run_forecast_btn:
                if not BACKTESTING_AVAILABLE:
                    st.error("❌ Backtesting module not available")
                else:
                    # Validate data before running
                    if VALIDATORS_AVAILABLE:
                        is_valid, param_msg = DataValidator.validate_forecast_parameters(
                            len(train_df), len(test_df), horizon
                        )
                        if not is_valid:
                            st.error("❌ Invalid forecast parameters:\n" + param_msg)
                            st.stop()

                    backtest_models = build_backtest_models()

                    if not backtest_models:
                        st.error("❌ No models available for backtesting")
                    else:
                        with st.spinner(f"⏳ Running backtesting with {len(backtest_models)} models across 5 windows... (This may take 2-3 minutes)"):
                            try:
                                backtester = RollingWindowBacktester(
                                    data=data,
                                    n_windows=5,
                                    test_size=test_size / 100
                                )

                                backtest_results = backtester.run_backtest(
                                    models=backtest_models,
                                    freq=metadata.get('freq', 'MS'),
                                    season_length=metadata.get('season_length', 12),
                                    horizon=min(horizon, len(test_df)),
                                    input_size=min(input_size, len(train_df) // 2),
                                    lags=[1, 12] if module_type == "MLForecast" else None
                                )

                                st.session_state.backtest_results = backtest_results
                                st.success("✅ Backtesting completed successfully!")

                            except Exception as e:
                                st.error(f"❌ Error during backtesting: {str(e)}")

        else:
            if st.button("Run Forecast", type="primary", use_container_width=True):
                # Validate data before running
                if VALIDATORS_AVAILABLE:
                    is_valid, param_msg = DataValidator.validate_forecast_parameters(
                        len(train_df), len(test_df), horizon, input_size
                    )
                    if not is_valid:
                        for line in param_msg.split('\n'):
                            if line.startswith('❌'):
                                st.error(line)
                        st.stop()

                with st.spinner(f"Running {strategy} with {module_type} - {model_type}..."):
                    results = run_forecast(
                        module_type=module_type,
                        model_type=model_type,
                        strategy=strategy,
                        train_df=train_df,
                        test_df=test_df if strategy == "One-step forecast" else test_df,
                        params=params
                    )

                    if results is not None:
                        st.session_state.forecast_results = results
                        st.success("✅ Forecast completed successfully!")

        # Display Backtesting Results
        if st.session_state.backtest_results is not None:
            st.divider()
            display_backtest_results(st.session_state.backtest_results)

        # Display Single Model Results
        elif st.session_state.forecast_results is not None:
            results = st.session_state.forecast_results

            st.header("📈 Forecast Results")

            # Metrics
            st.subheader("Performance Metrics")
            display_metrics(results['metrics'])

            # Forecast Plot
            st.subheader("Forecast vs Actual")
            fig_forecast = plot_forecast_results(
                results,
                title=f"{module_type} - {model_type} ({strategy})"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Results table
            with st.expander("View Forecast Table"):
                st.dataframe(results['forecasts'])

            # Debug view - show alignment verification
            with st.expander("🔍 Debug: Verify Data Alignment"):
                st.write("**First 5 rows of Test Set:**")
                st.dataframe(test_df.head(5))
                st.write("**First 5 rows of Forecast Results:**")
                st.dataframe(results['forecasts'].head(5))
                st.write("**Note:** The 'y' values from Test Set should match 'y_true' values in Forecast Results")

            # Download results
            st.subheader("💾 Download Results")
            csv = results['forecasts'].to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{module_type}_{model_type}_{strategy.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        # Show instructions if no data loaded
        st.info("👈 Please load data from the sidebar to get started")

        st.markdown("""
        ### Getting Started

        1. **Choose a data source:**
           - Upload your own CSV file with datetime and value columns
           - Or select a pre-loaded sample dataset

        2. **Configure forecasting:**
           - Select forecasting module (Statistical, ML, or Neural)
           - Choose a specific model
           - Pick a forecasting strategy
           - Adjust parameters as needed

        3. **Run forecast and analyze results:**
           - View performance metrics (MAE, RMSE, MAPE)
           - Visualize forecast vs actual values
           - Download results as CSV

        ### Forecasting Strategies

        - **One-step forecast**: Refits model for each prediction (slowest, most accurate)
        - **Multi-step recursive**: Uses previous predictions to forecast ahead (balanced)
        - **Multi-output direct**: Generates all predictions at once (fastest)

        ### Module Capabilities

        | Module | One-step | Recursive | Direct |
        |--------|----------|-----------|--------|
        | StatsForecast | ✅ | ✅ | ❌ |
        | MLForecast | ✅ | ✅ | ✅ |
        | NeuralForecast | ✅ | ✅ | ✅ |
        """)


if __name__ == "__main__":
    main()
