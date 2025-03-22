from collections import namedtuple
import pandas as pd
import plotly.graph_objects as go

Table_Figures = namedtuple('Tables', ['completed', 'stats'])

def create_tables(completed_df: pd.DataFrame) -> Table_Figures:
   # Filter for completed issues with both estimate and actual time
   filtered_df = completed_df[completed_df['duedate'].notna()][['key', 'duedate', 'originalestimate', 'timespentworking']].copy()
   filtered_df.columns = ['Issue', 'Due Date', 'Est Time', 'Actual Time']

   metrics = calculate_estimate_accuracy_metrics(filtered_df)

   stats_fig = create_stats_table(metrics)
   completed_fig = create_completed_table(filtered_df)
   return Table_Figures(completed=completed_fig, stats=stats_fig)
    
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
    """Create a table showing statistical analysis of estimates."""
    
    stats_df = pd.DataFrame([
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
    ])

    # Calculate dynamic height based on number of rows (header + data rows)
    row_height = 30 # matches cell height
    padding = 50 # extra space for title and margins
    table_height = (len(stats_df) + 1) * row_height + padding

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Estimates', 'Actual', 'Interpretation'],
            fill_color='rgba(200,200,200,0.3)',
            font=dict(color='black', size=14),
            align='left',
        ),
        cells=dict(
            values=[
                stats_df['Metric'],
                stats_df['Estimates'],
                stats_df['Actual'],
                stats_df['Interpretation']
            ],
            fill_color='rgba(200,200,200,0.1)',
            font=dict(color='black', size=12),
            align='left',
            height=row_height,
        ),
        columnwidth=[1, 0.7, 0.7, 3]  # Match header widths
    )])

    fig.update_layout(
        height=table_height,
        title='Estimation Analysis',
        margin=dict(t=30, b=0)  # Reduce margins to make it more compact
    )

    return fig


def create_completed_table(filtered_df: pd.DataFrame) -> go.Figure:
    """Create tables showing completed issues and statistical analysis."""    
    # Calculate dynamic height based on number of rows (header + data rows)
    row_height = 30 # matches cell height
    padding = 50 # extra space for title and margins
    table_height = (len(filtered_df) + 1) * row_height + padding

    # Create the completed issues table
    completed_issues_fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Issue', 'Due Date', 'Est Time(d)', 'Actual Time(d)'],
            fill_color='rgba(200,200,200,0.3)',
            font=dict(color='black', size=14),
            align='left'
        ),
        cells=dict(
            values=[
                filtered_df['Issue'],
                filtered_df['Due Date'].astype(str),
                filtered_df['Est Time'].apply(lambda x: f'{x:.2f}'),
                filtered_df['Actual Time'].apply(lambda x: f'{x:.2f}'),
            ],
            fill_color='rgba(200,200,200,0.1)',
            font=dict(color='black', size=12),
            align='left',
            height=row_height,
        )
    )])

    completed_issues_fig.update_layout(
        height=table_height,
        title='Completed Issues',
        margin=dict(t=30, b=0)  # Reduce margins to make it more compact

    )
    
    return completed_issues_fig
