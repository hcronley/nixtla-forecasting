"""
config_manager.py
Configuration persistence module for reproducible forecasting.

This module handles saving and loading forecasting model configurations
to/from JSON files, enabling reproducible results and model reuse.

Features:
- Save complete model configurations to JSON
- Load configurations for reproducible forecasting
- Extract best model config from backtesting results
- Config validation and error handling
- Human-readable configuration files
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd


class ConfigManager:
    """
    Manages saving and loading forecasting model configurations.

    Provides utilities for persisting model setups to enable reproducibility
    and easy model reuse across different datasets and sessions.

    Parameters
    ----------
    config_dir : str, optional
        Directory to store configuration files (default: current directory)
    """

    def __init__(self, config_dir: str = "."):
        """Initialize config manager."""
        self.config_dir = config_dir
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

    def save_config(
        self,
        config: Dict[str, Any],
        filename: str,
        description: str = ""
    ) -> bool:
        """
        Save a model configuration to JSON file.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            {
                'dataset': str,
                'module': str,
                'model_type': str,
                'params': dict,
                'metrics': dict (optional),
                'timestamp': str (auto-added)
            }
        filename : str
            Name of the JSON file to save (without .json extension)
        description : str, optional
            Human-readable description of the configuration

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Add metadata
            full_config = config.copy()
            full_config['timestamp'] = datetime.now().isoformat()
            full_config['description'] = description

            # Validate required fields
            required_fields = ['module', 'model_type']
            for field in required_fields:
                if field not in full_config:
                    raise ValueError(f"Missing required field: {field}")

            # Build file path
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            filepath = os.path.join(self.config_dir, filename)

            # Write to JSON
            with open(filepath, 'w') as f:
                json.dump(full_config, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False

    def load_config(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a model configuration from JSON file.

        Parameters
        ----------
        filename : str
            Name of the JSON file to load (with or without .json extension)

        Returns
        -------
        dict or None
            Configuration dictionary if successful, None otherwise
        """
        try:
            # Build file path
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            filepath = os.path.join(self.config_dir, filename)

            # Check if file exists
            if not os.path.exists(filepath):
                print(f"❌ Config file not found: {filepath}")
                return None

            # Read from JSON
            with open(filepath, 'r') as f:
                config = json.load(f)

            return config

        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return None

    def list_configs(self) -> pd.DataFrame:
        """
        List all saved configurations.

        Returns
        -------
        pd.DataFrame
            DataFrame with filename, timestamp, module, model_type, description
        """
        configs = []

        try:
            for filename in os.listdir(self.config_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.config_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            config = json.load(f)

                        configs.append({
                            'Filename': filename.replace('.json', ''),
                            'Timestamp': config.get('timestamp', 'N/A'),
                            'Module': config.get('module', 'N/A'),
                            'Model Type': config.get('model_type', 'N/A'),
                            'Dataset': config.get('dataset', 'N/A'),
                            'Description': config.get('description', '')
                        })
                    except:
                        continue

            if not configs:
                print("No configurations found")
                return pd.DataFrame()

            df = pd.DataFrame(configs)
            return df.sort_values('Timestamp', ascending=False)

        except Exception as e:
            print(f"❌ Error listing configs: {e}")
            return pd.DataFrame()

    def delete_config(self, filename: str) -> bool:
        """
        Delete a saved configuration.

        Parameters
        ----------
        filename : str
            Name of the JSON file to delete

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            filepath = os.path.join(self.config_dir, filename)

            if not os.path.exists(filepath):
                print(f"❌ Config file not found: {filepath}")
                return False

            os.remove(filepath)
            print(f"✅ Deleted: {filename}")
            return True

        except Exception as e:
            print(f"❌ Error deleting config: {e}")
            return False

    def export_best_model_config(
        self,
        backtest_results: Dict[str, Any],
        filename: str = "best_model_config",
        dataset_name: str = "Unknown"
    ) -> bool:
        """
        Extract and save best model configuration from backtesting results.

        Parameters
        ----------
        backtest_results : dict
            Results from RollingWindowBacktester.run_backtest()
        filename : str
            Name for the saved configuration file
        dataset_name : str
            Name of the dataset used for identification

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Extract best model from results
            model_rankings = backtest_results.get('model_rankings', {})
            models_config = backtest_results.get('models', {})

            if not model_rankings:
                print("❌ No model rankings found in backtest results")
                return False

            # Find best model (lowest MAE)
            best_model_name = min(
                model_rankings.keys(),
                key=lambda k: model_rankings[k].get('mae_mean', float('inf'))
            )

            best_stats = model_rankings[best_model_name]
            best_model_config = models_config[best_model_name]

            # Build configuration
            config = {
                'dataset': dataset_name,
                'model_name': best_model_name,
                'module': best_model_config['module'],
                'model_type': best_model_config['type'],
                'params': best_model_config.get('params', {}),
                'metrics': {
                    'mae_mean': float(best_stats['mae_mean']),
                    'mae_std': float(best_stats['mae_std']),
                    'rmse_mean': float(best_stats['rmse_mean']),
                    'rmse_std': float(best_stats['rmse_std']),
                    'mape_mean': float(best_stats['mape_mean']),
                    'mape_std': float(best_stats['mape_std']),
                    'success_rate': float(best_stats['success_rate']),
                    'window_count': int(best_stats['window_count'])
                }
            }

            # Save configuration
            description = (
                f"Best model from backtesting: {best_model_name}\n"
                f"MAE: {config['metrics']['mae_mean']:.4f} (±{config['metrics']['mae_std']:.4f})"
            )

            return self.save_config(config, filename, description)

        except Exception as e:
            print(f"❌ Error exporting best model config: {e}")
            return False

    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of a configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary

        Returns
        -------
        str
            Formatted summary string
        """
        summary = f"""
Configuration Summary
{'='*60}
Dataset: {config.get('dataset', 'N/A')}
Model Name: {config.get('model_name', config.get('model_type', 'N/A'))}
Module: {config.get('module', 'N/A')}
Type: {config.get('model_type', 'N/A')}
Timestamp: {config.get('timestamp', 'N/A')}

Parameters:
{json.dumps(config.get('params', {}), indent=2)}

Metrics:
{json.dumps(config.get('metrics', {}), indent=2)}

Description:
{config.get('description', 'No description provided')}

{'='*60}
"""
        return summary


def create_config_from_params(
    module: str,
    model_type: str,
    dataset: str = "Unknown",
    freq: str = "MS",
    horizon: int = 12,
    season_length: int = 12,
    lags: Optional[list] = None,
    input_size: int = 12,
    **additional_params
) -> Dict[str, Any]:
    """
    Create a configuration dictionary from individual parameters.

    Parameters
    ----------
    module : str
        Forecasting module ('StatsForecast', 'MLForecast', 'NeuralForecast')
    model_type : str
        Model type (e.g., 'auto_arima', 'xgboost', 'lstm')
    dataset : str
        Dataset name
    freq : str
        Time series frequency
    horizon : int
        Forecast horizon
    season_length : int
        Seasonal period
    lags : list, optional
        Lag features for ML models
    input_size : int
        Input window size for neural models
    **additional_params
        Additional model-specific parameters

    Returns
    -------
    dict
        Configuration dictionary
    """
    config = {
        'dataset': dataset,
        'module': module,
        'model_type': model_type,
        'params': {
            'freq': freq,
            'horizon': horizon,
            'season_length': season_length,
            'input_size': input_size,
        }
    }

    # Add lags for ML models
    if lags is not None:
        config['params']['lags'] = lags

    # Add additional parameters
    config['params'].update(additional_params)

    return config


def load_and_validate_config(
    filepath: str,
    required_fields: Optional[list] = None
) -> Optional[Dict[str, Any]]:
    """
    Load and validate a configuration from file.

    Parameters
    ----------
    filepath : str
        Path to configuration JSON file
    required_fields : list, optional
        Fields that must be present in the config

    Returns
    -------
    dict or None
        Configuration if valid, None otherwise
    """
    if required_fields is None:
        required_fields = ['module', 'model_type']

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        # Validate required fields
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return None

        return config

    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {filepath}")
        return None
    except FileNotFoundError:
        print(f"❌ Config file not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None
