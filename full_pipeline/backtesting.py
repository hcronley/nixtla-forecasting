"""
backtesting.py
Rolling window backtesting framework for comparing forecasting models.

This module provides rolling cross-validation functionality to compare
multiple forecasting models from StatsForecast, MLForecast, and NeuralForecast
across multiple time windows, simulating realistic deployment scenarios.

Key Features:
- 5-window rolling cross-validation (or custom window count)
- Supports all three Nixtla libraries simultaneously
- Aggregates metrics across windows
- Ranks models by performance
- Handles per-window results for detailed analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

from df_statsforecast import StatsforecastForecaster, train_test_split_ts as sf_split
from df_mlforecast import MLForecastForecaster
from df_neuralforecast import NeuralForecastForecaster


class RollingWindowBacktester:
    """
    Implements rolling window cross-validation for time series forecasting.

    This class splits time series data into multiple overlapping train/test windows,
    runs specified models on each window, and aggregates results for comprehensive
    model comparison.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with columns 'ds' (datetime) and 'y' (values)
    n_windows : int, default=5
        Number of rolling windows to create
    test_size : float or int, default=0.2
        Size of test window (proportion or number of observations)

    Attributes
    ----------
    windows : list of dict
        Each dict contains 'train' and 'test' DataFrames
    results : dict
        Stores results from backtesting runs
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_windows: int = 5,
        test_size: Union[float, int] = 0.2
    ):
        """Initialize the backtester with data and window configuration."""
        self.data = data.copy().sort_values('ds').reset_index(drop=True)
        self.n_windows = n_windows
        self.test_size = test_size
        self.windows = []
        self.results = {}

        # Validate data
        if 'ds' not in self.data.columns or 'y' not in self.data.columns:
            raise ValueError("Data must have 'ds' (datetime) and 'y' (values) columns")

        if len(self.data) < n_windows * 2:
            raise ValueError(
                f"Data too short for {n_windows} windows. "
                f"Need at least {n_windows * 2} observations."
            )

    def create_windows(self) -> List[Dict[str, pd.DataFrame]]:
        """
        Create rolling train/test windows.

        Uses rolling window approach where each window shifts forward by a fixed stride.
        Ensures no data leakage (test data never used in training).

        Returns
        -------
        list of dict
            Each dict has keys 'train' and 'test' with DataFrame values
        """
        total_length = len(self.data)

        # Calculate window size and stride
        if isinstance(self.test_size, float):
            test_window_size = int(total_length * self.test_size)
        else:
            test_window_size = int(self.test_size)

        # Ensure we have enough data
        min_train_size = int(total_length * 0.1)  # At least 10% for training
        if test_window_size > total_length - min_train_size:
            test_window_size = total_length - min_train_size

        # Calculate stride: distribute windows across available data
        stride = (total_length - test_window_size) // (self.n_windows - 1) if self.n_windows > 1 else total_length - test_window_size

        self.windows = []
        for i in range(self.n_windows):
            test_start = i * stride
            test_end = test_start + test_window_size

            # Ensure test_end doesn't exceed data length
            if test_end > total_length:
                test_end = total_length
                test_start = max(0, test_end - test_window_size)

            train_df = self.data.iloc[:test_start].copy()
            test_df = self.data.iloc[test_start:test_end].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                self.windows.append({
                    'window_id': i,
                    'train': train_df,
                    'test': test_df,
                    'train_size': len(train_df),
                    'test_size': len(test_df)
                })

        return self.windows

    def run_statsforecast_model(
        self,
        model_type: str,
        window: Dict,
        freq: str,
        season_length: int,
        horizon: int,
        **model_params
    ) -> Optional[Dict[str, float]]:
        """
        Run a StatsForecast model on a single window.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., 'auto_arima', 'auto_ets')
        window : dict
            Window with 'train' and 'test' DataFrames
        freq : str
            Time series frequency
        season_length : int
            Seasonal period
        horizon : int
            Forecast horizon
        **model_params
            Additional model parameters

        Returns
        -------
        dict or None
            Metrics dict with mae, rmse, mape (or None if failed)
        """
        try:
            forecaster = StatsforecastForecaster(
                model_type=model_type,
                freq=freq,
                season_length=season_length,
                **model_params
            )

            # Use multi-step forecast for realistic backtesting
            results = forecaster.multi_step_forecast(
                window['train'],
                horizon=horizon,
                test_df=window['test']
            )

            return results.get('metrics')

        except Exception as e:
            print(f"  ⚠️  StatsForecast ({model_type}) failed on window: {str(e)[:50]}")
            return None

    def run_mlforecast_model(
        self,
        model_type: str,
        window: Dict,
        freq: str,
        horizon: int,
        lags: Optional[List[int]] = None,
        **model_params
    ) -> Optional[Dict[str, float]]:
        """
        Run an MLForecast model on a single window.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., 'xgboost', 'lightgbm')
        window : dict
            Window with 'train' and 'test' DataFrames
        freq : str
            Time series frequency
        horizon : int
            Forecast horizon
        lags : list, optional
            Lag features to use
        **model_params
            Additional model parameters

        Returns
        -------
        dict or None
            Metrics dict with mae, rmse, mape (or None if failed)
        """
        try:
            if lags is None:
                lags = [1, 12]

            forecaster = MLForecastForecaster(
                model_type=model_type,
                freq=freq,
                lags=lags,
                **model_params
            )

            results = forecaster.multi_step_forecast(
                window['train'],
                horizon=horizon,
                test_df=window['test']
            )

            return results.get('metrics')

        except Exception as e:
            print(f"  ⚠️  MLForecast ({model_type}) failed on window: {str(e)[:50]}")
            return None

    def run_neuralforecast_model(
        self,
        model_type: str,
        window: Dict,
        freq: str,
        input_size: int,
        horizon: int,
        **model_params
    ) -> Optional[Dict[str, float]]:
        """
        Run a NeuralForecast model on a single window.

        Parameters
        ----------
        model_type : str
            Type of model (e.g., 'lstm', 'nbeats')
        window : dict
            Window with 'train' and 'test' DataFrames
        freq : str
            Time series frequency
        input_size : int
            Input window size (lookback)
        horizon : int
            Forecast horizon
        **model_params
            Additional model parameters

        Returns
        -------
        dict or None
            Metrics dict with mae, rmse, mape (or None if failed)
        """
        try:
            forecaster = NeuralForecastForecaster(
                model_type=model_type,
                freq=freq,
                input_size=input_size,
                horizon=horizon,
                **model_params
            )

            results = forecaster.multi_output_forecast(
                window['train'],
                horizon=horizon,
                test_df=window['test']
            )

            return results.get('metrics')

        except Exception as e:
            print(f"  ⚠️  NeuralForecast ({model_type}) failed on window: {str(e)[:50]}")
            return None

    def run_backtest(
        self,
        models: Dict[str, Dict[str, Any]],
        freq: str,
        season_length: int = 12,
        horizon: int = 12,
        input_size: int = 12,
        lags: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run all models across all windows.

        Parameters
        ----------
        models : dict
            Dictionary mapping model names to configs.
            Format: {
                'model_name': {
                    'module': 'StatsForecast|MLForecast|NeuralForecast',
                    'type': 'auto_arima|xgboost|lstm|etc',
                    'params': {...}  # optional model parameters
                }
            }
        freq : str
            Time series frequency (e.g., 'MS', 'D')
        season_length : int
            Seasonal period
        horizon : int
            Forecast horizon
        input_size : int
            Input window size for neural models
        lags : list, optional
            Lag features for ML models

        Returns
        -------
        dict
            Backtesting results with structure:
            {
                'window_results': [...],
                'model_rankings': {...},
                'summary': {...}
            }
        """
        # Create windows if not already done
        if not self.windows:
            self.create_windows()

        print(f"\n{'='*80}")
        print(f"BACKTESTING: {len(models)} models × {len(self.windows)} windows")
        print(f"{'='*80}\n")

        # Store results for each model
        all_window_results = []

        # Run each model on each window
        for window in self.windows:
            window_id = window['window_id']
            print(f"Window {window_id + 1}/{len(self.windows)} (Train: {window['train_size']}, Test: {window['test_size']})")

            window_results = {
                'window_id': window_id,
                'train_size': window['train_size'],
                'test_size': window['test_size'],
                'models': {}
            }

            for model_name, model_config in models.items():
                module = model_config['module']
                model_type = model_config['type']
                params = model_config.get('params', {})

                # Route to appropriate module
                if module == 'StatsForecast':
                    metrics = self.run_statsforecast_model(
                        model_type=model_type,
                        window=window,
                        freq=freq,
                        season_length=season_length,
                        horizon=horizon,
                        **params
                    )
                    status = "✅" if metrics else "❌"
                    print(f"  {status} {model_name:30} (StatsForecast - {model_type})")

                elif module == 'MLForecast':
                    metrics = self.run_mlforecast_model(
                        model_type=model_type,
                        window=window,
                        freq=freq,
                        horizon=horizon,
                        lags=lags,
                        **params
                    )
                    status = "✅" if metrics else "❌"
                    print(f"  {status} {model_name:30} (MLForecast - {model_type})")

                elif module == 'NeuralForecast':
                    metrics = self.run_neuralforecast_model(
                        model_type=model_type,
                        window=window,
                        freq=freq,
                        input_size=input_size,
                        horizon=horizon,
                        **params
                    )
                    status = "✅" if metrics else "❌"
                    print(f"  {status} {model_name:30} (NeuralForecast - {model_type})")

                else:
                    metrics = None
                    print(f"  ❌ {model_name:30} (Unknown module: {module})")

                window_results['models'][model_name] = metrics

            all_window_results.append(window_results)
            print()

        # Aggregate results
        rankings = self.aggregate_results(all_window_results, models)

        self.results = {
            'window_results': all_window_results,
            'model_rankings': rankings,
            'windows': self.windows,
            'models': models,
            'original_data': self.data
        }

        return self.results

    def aggregate_results(
        self,
        window_results: List[Dict],
        models: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all windows.

        Parameters
        ----------
        window_results : list
            Results from each window
        models : dict
            Model configuration dictionary

        Returns
        -------
        dict
            Aggregated rankings with average metrics per model
        """
        aggregated = {}

        for model_name in models.keys():
            mae_values = []
            rmse_values = []
            mape_values = []
            success_count = 0

            for window in window_results:
                metrics = window['models'].get(model_name)
                if metrics is not None:
                    mae_values.append(metrics.get('mae', np.nan))
                    rmse_values.append(metrics.get('rmse', np.nan))
                    mape_values.append(metrics.get('mape', np.nan))
                    success_count += 1

            if success_count > 0:
                aggregated[model_name] = {
                    'module': models[model_name]['module'],
                    'type': models[model_name]['type'],
                    'success_rate': success_count / len(window_results),
                    'mae_mean': float(np.nanmean(mae_values)),
                    'mae_std': float(np.nanstd(mae_values)),
                    'rmse_mean': float(np.nanmean(rmse_values)),
                    'rmse_std': float(np.nanstd(rmse_values)),
                    'mape_mean': float(np.nanmean(mape_values)),
                    'mape_std': float(np.nanstd(mape_values)),
                    'window_count': success_count
                }

        return aggregated

    def get_rankings(
        self,
        metric: str = 'mae_mean',
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Get model rankings sorted by specified metric.

        Parameters
        ----------
        metric : str
            Metric to sort by (e.g., 'mae_mean', 'rmse_mean', 'mape_mean')
        ascending : bool
            Whether to sort ascending (lower is better for errors)

        Returns
        -------
        pd.DataFrame
            Sorted ranking table
        """
        if not self.results:
            raise ValueError("No results available. Run backtest() first.")

        rankings = self.results['model_rankings']

        # Convert to DataFrame
        df = pd.DataFrame(rankings).T
        df.index.name = 'Model'
        df = df.reset_index()

        # Sort by metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=ascending)

        return df

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for all models.

        Returns
        -------
        pd.DataFrame
            Summary statistics including mean, std, rank for all metrics
        """
        if not self.results:
            raise ValueError("No results available. Run backtest() first.")

        rankings = self.get_rankings('mae_mean')

        # Add ranking column
        rankings['rank'] = range(1, len(rankings) + 1)

        return rankings

    def export_results(self, filepath: str = 'backtest_results.csv') -> None:
        """
        Export backtest results to CSV.

        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        if not self.results:
            raise ValueError("No results available. Run backtest() first.")

        summary = self.get_summary_stats()
        summary.to_csv(filepath, index=False)
        print(f"\n✅ Results exported to {filepath}")


# Convenience function for quick backtesting
def run_quick_backtest(
    data: pd.DataFrame,
    models: Dict[str, Dict[str, Any]],
    n_windows: int = 5,
    test_size: float = 0.2,
    freq: str = 'MS',
    season_length: int = 12,
    horizon: int = 12,
    input_size: int = 12,
    lags: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Quick backtesting function for testing multiple models.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with 'ds' and 'y' columns
    models : dict
        Models to test
    n_windows : int
        Number of rolling windows
    test_size : float
        Test window size
    freq : str
        Time series frequency
    season_length : int
        Seasonal period
    horizon : int
        Forecast horizon
    input_size : int
        Input size for neural models
    lags : list, optional
        Lags for ML models

    Returns
    -------
    dict
        Complete backtesting results
    """
    backtester = RollingWindowBacktester(
        data=data,
        n_windows=n_windows,
        test_size=test_size
    )

    results = backtester.run_backtest(
        models=models,
        freq=freq,
        season_length=season_length,
        horizon=horizon,
        input_size=input_size,
        lags=lags
    )

    return results
