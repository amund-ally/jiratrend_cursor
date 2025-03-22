#from collections import namedtuple
import pandas as pd
import plotly.graph_objects as go

#Table_Figures = namedtuple('Tables', ['completed', 'stats'])

def create_tables(completed_df: pd.DataFrame):
    # Filter for completed issues with both estimate and actual time
    filtered_df = completed_df[completed_df['duedate'].notna()][['key', 'duedate', 'originalestimate', 'timespentworking']].copy()
    filtered_df.columns = ['Issue', 'Due Date', 'Est Time', 'Actual Time']

    metrics = calculate_estimate_accuracy_metrics(filtered_df)
    
    stats_table = create_stats_table(metrics)
    created_table = create_completed_table(filtered_df)
    return created_table, stats_table
    
def calculate_estimate_accuracy_metrics(filtered_df: pd.DataFrame) -> dict:
    """Calculate various metrics to analyze estimate accuracy."""
    metrics = {}
    
    # Basic statistics
    metrics['est_mean'] = filtered_df['Est Time'].mean()
    metrics['actual_mean'] = filtered_df['Actual Time'].mean()
    metrics['est_std'] = filtered_df['Est Time'].std()
    metrics['actual_std'] = filtered_df['Actual Time'].std()
    
    # Calculate error metrics
    errors = filtered_df['Actual Time'] - filtered_df['Est Time']
    metrics['mean_error'] = errors.mean()  # Positive means underestimated
    metrics['error_std'] = errors.std()
    
    # Calculate coefficient of variation (CV) - lower is better
    metrics['est_cv'] = metrics['est_std'] / metrics['est_mean'] if metrics['est_mean'] != 0 else float('inf')
    metrics['actual_cv'] = metrics['actual_std'] / metrics['actual_mean'] if metrics['actual_mean'] != 0 else float('inf')
    
    # Calculate percentage of estimates within 1 standard deviation
    within_1_std = abs(errors) <= metrics['error_std']
    metrics['percent_within_1_std'] = (within_1_std.sum() / len(filtered_df)) * 100
    
    return metrics


def create_stats_table(metrics: dict) -> go.Figure:
    """Create stats table data in Dash format."""
    return [
        {
            'Metric': 'Mean',
            'Estimates': f'{metrics["est_mean"]:.2f}',
            'Actual': f'{metrics["actual_mean"]:.2f}',
            'Interpretation': f'On average, tasks are estimated at {metrics["est_mean"]:.1f} days and take {metrics["actual_mean"]:.1f} days'
        },
        {
            'Metric': 'Standard Deviation',
            'Estimates': f'{metrics["est_std"]:.2f}',
            'Actual': f'{metrics["actual_std"]:.2f}',
            'Interpretation': f'Estimates vary by ±{metrics["est_std"]:.1f} days, actual time varies by ±{metrics["actual_std"]:.1f} days'
        },
        {
            'Metric': 'Error (Act-Est)',
            'Estimates': f'{metrics["mean_error"]:.2f}',
            'Actual': f'{metrics["error_std"]:.2f}',
            'Interpretation': f'{"Underestimated" if metrics["mean_error"] > 0 else "Overestimated"} by {abs(metrics["mean_error"]):.1f} days on average'
        },
        {
            'Metric': 'Coefficient of Variation',
            'Estimates': f'{metrics["est_cv"]:.2%}',
            'Actual': f'{metrics["actual_cv"]:.2%}',
            'Interpretation': f'{"Estimates" if metrics["est_cv"] > metrics["actual_cv"] else "Actual time"} shows more variation relative to mean'
        },
        {
            'Metric': 'Accuracy',
            'Estimates': f'{metrics["percent_within_1_std"]:.1f}%',
            'Actual': '',
            'Interpretation': f'{metrics["percent_within_1_std"]:.1f}% of estimates were within ±{metrics["error_std"]:.1f} days of actual time'
        }
    ]


def create_completed_table(filtered_df: pd.DataFrame) -> go.Figure:
    """Create completed issues table data in Dash format."""
    return filtered_df.to_dict('records')
