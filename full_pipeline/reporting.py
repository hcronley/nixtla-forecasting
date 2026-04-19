"""
reporting.py
Visualization and reporting module for backtesting results.

This module provides utilities for visualizing and analyzing backtesting results,
including ranking tables, comparison plots, and performance heatmaps.

Features:
- Model ranking tables with sorted metrics
- Combined forecast comparison plots
- Performance heatmaps across windows
- Aggregated statistics and summaries
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any


class ModelComparator:
    """
    Analyzes and visualizes forecasting model comparison results.

    Parameters
    ----------
    backtest_results : dict
        Results from RollingWindowBacktester.run_backtest()

    Attributes
    ----------
    results : dict
        Stored backtest results
    rankings : pd.DataFrame
        Model rankings by performance
    """

    def __init__(self, backtest_results: Dict[str, Any]):
        """Initialize comparator with backtest results."""
        self.results = backtest_results
        self.rankings = self._compute_rankings()

    def _compute_rankings(self) -> pd.DataFrame:
        """Compute model rankings from results."""
        if not self.results or 'model_rankings' not in self.results:
            raise ValueError("Invalid backtest results structure")

        rankings = self.results['model_rankings']
        df = pd.DataFrame(rankings).T
        df.index.name = 'Model'
        df = df.reset_index()

        # Ensure numeric columns are proper dtype
        numeric_cols = ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'mape_mean', 'mape_std', 'success_rate', 'window_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by MAE (lower is better)
        df = df.sort_values('mae_mean')
        df['rank'] = range(1, len(df) + 1)

        return df

    def get_ranking_table(
        self,
        top_n: Optional[int] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get formatted ranking table.

        Parameters
        ----------
        top_n : int, optional
            Show only top N models
        metrics : list, optional
            Specific metrics to include (default: all error metrics)

        Returns
        -------
        pd.DataFrame
            Formatted ranking table
        """
        df = self.rankings.copy()

        if top_n:
            df = df.head(top_n)

        # Select columns to display
        if metrics is None:
            metrics = ['rank', 'module', 'type', 'mae_mean', 'rmse_mean', 'mape_mean',
                       'success_rate']

        available_cols = [col for col in metrics if col in df.columns]
        return df[['Model'] + available_cols]

    def get_summary_table(self) -> Dict[str, float]:
        """
        Get summary statistics for best model.

        Returns
        -------
        dict
            Summary stats for top-ranked model
        """
        best = self.rankings.iloc[0]

        return {
            'Model': best['Model'],
            'Module': best['module'],
            'Type': best['type'],
            'MAE': float(best['mae_mean']),
            'RMSE': float(best['rmse_mean']),
            'MAPE': float(best['mape_mean']),
            'Std (MAE)': float(best['mae_std']),
            'Std (RMSE)': float(best['rmse_std']),
            'Success Rate': float(best['success_rate'])
        }

    def create_comparison_plot(
        self,
        top_n: int = 3,
        metric: str = 'mae_mean',
        error_bars: bool = True
    ) -> go.Figure:
        """
        Create bar plot comparing top N models.

        Parameters
        ----------
        top_n : int
            Number of top models to compare
        metric : str
            Metric to visualize
        error_bars : bool
            Whether to show error bars (std dev)

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        top_models = self.rankings.head(top_n)

        error_col = metric.replace('_mean', '_std')
        has_error = error_col in top_models.columns

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_models['Model'],
            y=top_models[metric],
            error_y=dict(
                type='data',
                array=top_models[error_col] if has_error else None,
                visible=error_bars and has_error
            ),
            marker=dict(
                color=top_models['rank'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rank")
            ),
            text=top_models['type'],
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>' +
                          f'{metric}: %{{y:.4f}}<br>' +
                          'Type: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Top {top_n} Models by {metric}',
            xaxis_title='Model',
            yaxis_title=metric,
            template='plotly_white',
            height=400,
            showlegend=False
        )

        return fig

    def create_metric_comparison_table(self, top_n: int = 5) -> pd.DataFrame:
        """
        Create detailed metric comparison table for top N models.

        Parameters
        ----------
        top_n : int
            Number of top models to include

        Returns
        -------
        pd.DataFrame
            Detailed comparison table
        """
        top_models = self.rankings.head(top_n)

        comparison = pd.DataFrame({
            'Rank': top_models['rank'],
            'Model': top_models['Model'],
            'Module': top_models['module'],
            'Type': top_models['type'],
            'MAE': top_models['mae_mean'].round(4),
            'MAE (±)': top_models['mae_std'].round(4),
            'RMSE': top_models['rmse_mean'].round(4),
            'RMSE (±)': top_models['rmse_std'].round(4),
            'MAPE (%)': (top_models['mape_mean']).round(2),
            'Success': (top_models['success_rate'] * 100).round(0).astype(int),
        })

        return comparison

    def create_window_performance_heatmap(self) -> go.Figure:
        """
        Create heatmap of model performance across windows.

        Returns
        -------
        go.Figure
            Plotly heatmap figure
        """
        window_results = self.results['window_results']

        # Extract MAE for each model across windows
        models = list(self.results['models'].keys())
        windows = [f"W{i+1}" for i in range(len(window_results))]

        data_matrix = []
        for model in models:
            model_mae = []
            for window in window_results:
                metrics = window['models'].get(model)
                if metrics:
                    model_mae.append(metrics.get('mae', np.nan))
                else:
                    model_mae.append(np.nan)
            data_matrix.append(model_mae)

        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=windows,
            y=models,
            colorscale='RdYlGn_r',  # Red = high error, Green = low error
            text=np.round(data_matrix, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="MAE")
        ))

        fig.update_layout(
            title='Model Performance Across Windows (MAE)',
            xaxis_title='Window',
            yaxis_title='Model',
            height=400,
            width=800
        )

        return fig

    def create_metric_heatmap(self, metric: str = 'mae_mean') -> go.Figure:
        """
        Create heatmap of specified metric across models and windows.

        Parameters
        ----------
        metric : str
            Metric to visualize ('mae_mean', 'rmse_mean', 'mape_mean')

        Returns
        -------
        go.Figure
            Plotly heatmap figure
        """
        window_results = self.results['window_results']
        models = list(self.results['models'].keys())
        windows = [f"W{i+1}" for i in range(len(window_results))]

        metric_map = {
            'mae_mean': 'mae',
            'rmse_mean': 'rmse',
            'mape_mean': 'mape'
        }

        metric_key = metric_map.get(metric, metric)

        # Build matrix
        data_matrix = []
        for model in models:
            model_metrics = []
            for window in window_results:
                metrics = window['models'].get(model)
                if metrics:
                    model_metrics.append(metrics.get(metric_key, np.nan))
                else:
                    model_metrics.append(np.nan)
            data_matrix.append(model_metrics)

        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=windows,
            y=models,
            colorscale='RdYlGn_r',
            text=np.round(data_matrix, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title=metric)
        ))

        fig.update_layout(
            title=f'Model Performance Across Windows ({metric})',
            xaxis_title='Window',
            yaxis_title='Model',
            height=400,
            width=800
        )

        return fig

    def create_ranking_bar_chart(self, top_n: int = 5) -> go.Figure:
        """
        Create bar chart ranking models by MAE.

        Parameters
        ----------
        top_n : int
            Number of top models to show

        Returns
        -------
        go.Figure
            Plotly figure
        """
        top_models = self.rankings.head(top_n)

        fig = go.Figure()

        # Add bars for MAE
        fig.add_trace(go.Bar(
            name='MAE',
            x=top_models['Model'],
            y=top_models['mae_mean'],
            error_y=dict(
                type='data',
                array=top_models['mae_std'],
                visible=True
            ),
            marker_color='#1f77b4'
        ))

        fig.update_layout(
            title=f'Top {top_n} Models - Mean Absolute Error',
            xaxis_title='Model',
            yaxis_title='MAE (lower is better)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )

        return fig

    def create_multi_metric_comparison(self, top_n: int = 3) -> go.Figure:
        """
        Create multi-metric comparison for top models.

        Parameters
        ----------
        top_n : int
            Number of top models

        Returns
        -------
        go.Figure
            Plotly figure with multiple metrics
        """
        top_models = self.rankings.head(top_n)

        fig = go.Figure()

        # Normalize metrics to 0-1 scale for comparison
        metrics_dict = {
            'MAE': top_models['mae_mean'].values,
            'RMSE': top_models['rmse_mean'].values,
            'MAPE': top_models['mape_mean'].values
        }

        for metric_name, values in metrics_dict.items():
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
            fig.add_trace(go.Scatter(
                x=top_models['Model'],
                y=normalized,
                mode='lines+markers',
                name=metric_name,
                marker=dict(size=10)
            ))

        fig.update_layout(
            title=f'Top {top_n} Models - Normalized Metric Comparison',
            xaxis_title='Model',
            yaxis_title='Normalized Score (0=best)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )

        return fig

    def export_rankings_csv(self, filepath: str = 'model_rankings.csv') -> None:
        """
        Export rankings to CSV file.

        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        self.get_ranking_table().to_csv(filepath, index=False)
        print(f"✅ Rankings exported to {filepath}")

    def get_best_model_config(self) -> Dict[str, Any]:
        """
        Get configuration of best-performing model.

        Returns
        -------
        dict
            Configuration dict for best model
        """
        best_model_name = self.rankings.iloc[0]['Model']
        best_config = self.results['models'][best_model_name]

        return {
            'model_name': best_model_name,
            'module': best_config['module'],
            'type': best_config['type'],
            'params': best_config.get('params', {}),
            'metrics': {
                'mae': float(self.rankings.iloc[0]['mae_mean']),
                'rmse': float(self.rankings.iloc[0]['rmse_mean']),
                'mape': float(self.rankings.iloc[0]['mape_mean'])
            }
        }


def create_summary_report(backtest_results: Dict[str, Any]) -> str:
    """
    Create a text summary report of backtesting results.

    Parameters
    ----------
    backtest_results : dict
        Results from RollingWindowBacktester

    Returns
    -------
    str
        Formatted text report
    """
    comparator = ModelComparator(backtest_results)

    summary = comparator.get_summary_table()
    ranking_table = comparator.get_ranking_table(top_n=5)

    report = f"""
BACKTESTING SUMMARY REPORT
{'='*60}

BEST MODEL
----------
Model: {summary['Model']}
Module: {summary['Module']}
Type: {summary['Type']}

Performance Metrics:
  • MAE:  {summary['MAE']:.4f} (±{summary['Std (MAE)']:.4f})
  • RMSE: {summary['RMSE']:.4f} (±{summary['Std (RMSE)']:.4f})
  • MAPE: {summary['MAPE']:.2f}% 
  • Success Rate: {summary['Success Rate']*100:.1f}%

TOP 5 MODELS
{'-'*60}
{ranking_table.to_string(index=False)}

{'='*60}
"""

    return report


