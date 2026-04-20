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


    def create_top5_ranked_comparison(self, top_n: int = 5) -> go.Figure:
        """
        Create ranked comparison for top N models with three metrics and confidence bands.

        Parameters
        ----------
        top_n : int
            Number of top models to display

        Returns
        -------
        go.Figure
            Plotly figure with ranked bars and metrics
        """
        top_models = self.rankings.head(top_n).copy()
        top_models['rank_label'] = range(1, len(top_models) + 1)
        
        # Medal emojis for visual appeal
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        top_models['medal'] = medals[:len(top_models)]
        
        # Model labels with module type
        top_models['label'] = top_models['medal'] + ' ' + top_models['Model'] + ' (' + top_models['module'].str[:2] + ')'
        
        # Create figure with confidence bands for MAE
        fig = go.Figure()
        
        # MAE bars (primary, largest)
        fig.add_trace(go.Bar(
            y=top_models['label'],
            x=top_models['mae_mean'],
            error_x=dict(
                type='data',
                array=top_models['mae_std'],
                visible=True
            ),
            name='MAE',
            marker=dict(color='#1f77b4', opacity=0.8),
            orientation='h',
            hovertemplate='<b>%{y}</b><br>MAE: %{x:.2f} ± %{error_x.array:.2f}<extra></extra>',
            width=0.6
        ))
        
        fig.update_layout(
            title='Top 5 Forecasting Models - Rankings with Confidence Bands',
            xaxis_title='MAE (Mean Absolute Error) - Lower is Better',
            yaxis_title='Model Ranking',
            template='plotly_white',
            height=450,
            hovermode='closest',
            showlegend=False,
            xaxis=dict(zeroline=False),
            yaxis=dict(tickfont=dict(size=12))
        )
        
        return fig

    def create_ranking_progression_heatmap(self, top_n: int = 5) -> go.Figure:
        """
        Create heatmap showing how model rankings change across windows.

        Parameters
        ----------
        top_n : int
            Number of top models to show

        Returns
        -------
        go.Figure
            Plotly heatmap figure
        """
        top_models = self.rankings.head(top_n)['Model'].tolist()
        window_results = self.results['window_results']
        
        # Build ranking matrix for each window
        rank_matrix = []
        for model in top_models:
            model_ranks = []
            for window in window_results:
                metrics = window['models'].get(model)
                if metrics and 'mae' in metrics:
                    model_ranks.append(metrics['mae'])
                else:
                    model_ranks.append(np.nan)
            rank_matrix.append(model_ranks)
        
        # Create annotations showing rank position (1-7) in each window
        annotations_text = []
        for i, model in enumerate(top_models):
            row_annotations = []
            mae_values = rank_matrix[i]
            # Get ranking for this window among all models
            for window_idx, window in enumerate(window_results):
                if not np.isnan(mae_values[window_idx]):
                    # Count how many models in this window beat this one
                    rank_in_window = 1
                    for other_model in self.results['models'].keys():
                        other_metrics = window['models'].get(other_model)
                        if other_metrics and other_metrics.get('mae', np.inf) < mae_values[window_idx]:
                            rank_in_window += 1
                    row_annotations.append(str(rank_in_window))
                else:
                    row_annotations.append('—')
            annotations_text.append(row_annotations)
        
        windows = [f"W{i+1}" for i in range(len(window_results))]
        
        fig = go.Figure(data=go.Heatmap(
            z=rank_matrix,
            x=windows,
            y=top_models,
            colorscale='RdYlGn_r',
            text=annotations_text,
            texttemplate='Rank: %{text}',
            textfont={"size": 11},
            colorbar=dict(title="MAE"),
            hovertemplate='Model: %{y}<br>Window: %{x}<br>MAE: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='How Rankings Changed Across Windows',
            xaxis_title='Backtesting Window',
            yaxis_title='Model',
            height=400,
            width=700
        )
        
        return fig

    def create_metric_race_waterfall(self, metric: str = 'mae_mean', top_n: int = 5) -> go.Figure:
        """
        Create waterfall chart showing performance gaps from top model.

        Parameters
        ----------
        metric : str
            Metric to visualize ('mae_mean', 'rmse_mean', 'mape_mean')
        top_n : int
            Number of top models to show

        Returns
        -------
        go.Figure
            Plotly waterfall figure
        """
        top_models = self.rankings.head(top_n).copy()
        
        # Get metric values
        metric_values = top_models[metric].values
        best_value = metric_values[0]
        gaps = metric_values - best_value
        
        # Create labels with gaps
        labels = []
        x_values = []
        for i, (model, gap, value) in enumerate(zip(top_models['Model'], gaps, metric_values)):
            if i == 0:
                labels.append(f"1st: {model} ({value:.2f})")
                x_values.append(value)
            else:
                labels.append(f"{i+1}. {model}")
                x_values.append(gap)
        
        # Waterfall measures
        measures = ['relative'] * len(labels)
        measures[0] = 'absolute'
        
        fig = go.Figure(go.Waterfall(
            x=labels,
            y=[x_values[0]] + gaps[1:].tolist(),
            measure=measures,
            text=[f"{v:.2f}" for v in [x_values[0]] + gaps[1:].tolist()],
            textposition='outside',
            connector={"line": {"color": "rgba(0,0,0,0.4)"}},
            decreasing={"marker": {"color": '#1f77b4'}},
            increasing={"marker": {"color": '#d62728'}},
            totals={"marker": {"color": '#ff7f0e'}},
            hovertemplate='%{x}<br>' + metric.replace('_mean', '').upper() + ': %{y:.2f}<extra></extra>'
        ))
        
        metric_label = {'mae_mean': 'MAE', 'rmse_mean': 'RMSE', 'mape_mean': 'MAPE'}.get(metric, metric)
        
        fig.update_layout(
            title=f'Performance Gap Analysis - {metric_label}',
            xaxis_title='Model Ranking',
            yaxis_title=f'{metric_label} (lower is better)',
            template='plotly_white',
            height=450,
            showlegend=False
        )
        
        return fig

    def create_top5_portfolio_summary(self) -> go.Figure:
        """
        Create comprehensive portfolio dashboard combining top 5 visualizations.

        Returns
        -------
        go.Figure
            Plotly subplot figure with multiple charts
        """
        from plotly.subplots import make_subplots
        
        top_models = self.rankings.head(5)
        
        # Prepare data for subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 5 Models by MAE',
                'Ranking Progression',
                'Performance Gap',
                'Multi-Metric Comparison'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # 1. Top 5 models by MAE (horizontal bar)
        fig.add_trace(
            go.Bar(
                y=top_models['Model'],
                x=top_models['mae_mean'],
                error_x=dict(array=top_models['mae_std']),
                name='MAE',
                marker=dict(color='#1f77b4'),
                orientation='h',
                hovertemplate='%{y}<br>MAE: %{x:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Ranking progression heatmap
        window_results = self.results['window_results']
        rank_matrix = []
        for model in top_models['Model']:
            model_ranks = []
            for window in window_results:
                metrics = window['models'].get(model)
                if metrics and 'mae' in metrics:
                    model_ranks.append(metrics['mae'])
                else:
                    model_ranks.append(np.nan)
            rank_matrix.append(model_ranks)
        
        windows = [f"W{i+1}" for i in range(len(window_results))]
        fig.add_trace(
            go.Heatmap(
                z=rank_matrix,
                x=windows,
                y=top_models['Model'],
                colorscale='RdYlGn_r',
                colorbar=dict(x=0.46, len=0.4),
                hovertemplate='%{y} - %{x}<br>MAE: %{z:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Performance gap waterfall
        gaps = top_models['mae_mean'].values - top_models['mae_mean'].values[0]
        gap_labels = top_models['Model'].tolist()
        fig.add_trace(
            go.Bar(
                x=gap_labels,
                y=gaps,
                name='Gap from Leader',
                marker_color=['gold', 'silver', '#CD7F32'] + ['#6C757D'] * (len(gaps) - 3),
                hovertemplate='%{x}<br>Gap: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Multi-metric comparison (normalized)
        for i, metric_name in enumerate(['MAE', 'RMSE', 'MAPE']):
            metric_col = f"{metric_name.lower()}_mean"
            values = top_models[metric_col].values
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
            fig.add_trace(
                go.Scatter(
                    x=top_models['Model'],
                    y=normalized,
                    mode='lines+markers',
                    name=metric_name,
                    marker=dict(size=8),
                    hovertemplate='%{x}<br>' + metric_name + ': %{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Window", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Gap from Leader", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Score", row=2, col=2)
        
        fig.update_layout(
            title_text="Top 5 Forecasting Models - Comprehensive Portfolio Dashboard",
            height=900,
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
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


