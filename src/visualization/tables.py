from collections import namedtuple
from typing import Dict
import pandas as pd
import datetime as datetime
import numpy as np
import plotly.graph_objects as go
import logging

# Update the namedtuple to hold the scope changes DataFrame
Table_Data = namedtuple('TableData', ['completed_df', 'stats_df', 'scope_changes_df', 'completed_fig', 'stats_fig'])

def create_scope_changes_dataframe(estimate_changes: Dict, start_date: datetime.datetime) -> pd.DataFrame:
    """Create a DataFrame showing all estimate changes.
    
    Args:
        estimate_changes: Dictionary mapping dates to dictionaries of issue key -> estimate
        
    Returns:
        DataFrame with scope change details
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Creating scope changes dataframe from {len(estimate_changes)} date entries")
    
    # Parse the raw estimate changes into a flat list
    scope_changes = []
    sorted_dates = sorted(estimate_changes.keys())
    sorted_dates = [date for date in sorted_dates if date > start_date.date()]

    # Track previous estimates for calculating change
    previous_estimates = {}
    
    for date in sorted_dates:
        for issue_key, new_estimate in estimate_changes[date].items():
            old_estimate = previous_estimates.get(issue_key, 0)
            
            # Only include changes, not initial values from start date
            # unless there's a real change in value
            is_first_date = date == sorted_dates[0]
            if not is_first_date or (is_first_date and new_estimate != 0):
                scope_changes.append({
                    'Date': date,
                    'Issue': issue_key,
                    'Original Estimate': old_estimate,
                    'New Estimate': new_estimate,
                    'Change': new_estimate - old_estimate
                })
            
            # Update tracking of previous estimates
            previous_estimates[issue_key] = new_estimate
    
    # Create DataFrame
    if not scope_changes:
        logger.warning("No scope changes found to display")
        return pd.DataFrame()
    
    df = pd.DataFrame(scope_changes)
    
    # Sort by date (descending) and then by absolute change magnitude (descending)
    df['abs_change'] = df['Change'].abs()
    df = df.sort_values(['Date', 'abs_change'], ascending=[False, False])
    df = df.drop(columns=['abs_change'])
    
    # Add URL column for display
    df['Issue URL'] = df['Issue'].apply(
        lambda x: f"https://agrium.atlassian.net/browse/{x}"
    )

    # Reorder columns
    result_df = df[['Issue', 'Issue URL',  'Date', 'Original Estimate', 'New Estimate', 'Change']]
    
    logger.debug(f"Created scope changes dataframe with {len(result_df)} rows")
    return result_df

def create_tables(completed_df: pd.DataFrame, estimate_changes: Dict, start_date: datetime) -> Table_Data:
    """Create tables showing completed issues and statistical analysis.
    
    Args:
        completed_df: DataFrame with completed work data
        estimate_changes: Dictionary of raw estimate changes by date
        
    Returns:
        Table_Data: Namedtuple containing dataframes and plotly figures
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Creating tables from completed_df with {len(completed_df)} rows")
    
    # Filter for completed issues with both estimate and actual time
    filtered_df = completed_df[completed_df['duedate'].notna()][['key', 'duedate', 'originalestimate', 'timespentworking']].copy()
    filtered_df.columns = ['Issue', 'Due Date', 'Est Time', 'Actual Time']
    logger.debug(f"Filtered to {len(filtered_df)} rows with completed work")

    # Calculate metrics
    metrics = calculate_estimate_accuracy_metrics(filtered_df)
    logger.debug(f"Calculated metrics: mean error={metrics['mean_error']:.2f}, est_mean={metrics['est_mean']:.2f}")

    # Create stats dataframe
    stats_df = create_stats_dataframe(metrics)
    logger.debug(f"Created stats dataframe with {len(stats_df)} rows")

    # Create completed work dataframe
    completed_df = create_completed_dataframe(filtered_df)
    logger.debug(f"Created completed dataframe with {len(completed_df)} rows")
    
    # Create scope changes dataframe if estimate changes are provided
    scope_changes_df = None
    if estimate_changes:
        scope_changes_df = create_scope_changes_dataframe(estimate_changes, start_date)
        logger.debug(f"Created scope changes dataframe with {len(scope_changes_df)} rows")
    
    # For backward compatibility, also create the figures
    stats_fig = create_stats_table(metrics)
    completed_fig = create_completed_table(filtered_df)
    
    return Table_Data(
        completed_df=completed_df, 
        stats_df=stats_df, 
        scope_changes_df=scope_changes_df,
        completed_fig=completed_fig, 
        stats_fig=stats_fig
    )

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

def create_stats_dataframe(metrics: dict) -> pd.DataFrame:
    """Create a DataFrame showing statistical analysis of estimates."""
    
    stats_df = pd.DataFrame([
        {
            'Metric': 'Mean',
            'Estimates': metrics["est_mean"],
            'Actual': metrics["actual_mean"],
            'Interpretation': f'On average, tasks are estimated at {metrics["est_mean"]:.1f} days and take {metrics["actual_mean"]:.1f} days'
        },
        {
            'Metric': 'Standard Deviation',
            'Estimates': metrics["est_std"],
            'Actual': metrics["actual_std"],
            'Interpretation': f'Estimates vary by ±{metrics["est_std"]:.1f} days, actual time varies by ±{metrics["actual_std"]:.1f} days'
        },
        {
            'Metric': 'Error (Act-Est)',
            'Estimates': metrics["mean_error"],
            'Actual': metrics["error_std"],
            'Interpretation': f'{"Underestimated" if metrics["mean_error"] > 0 else "Overestimated"} by {abs(metrics["mean_error"]):.1f} days on average'
        },
        {
            'Metric': 'Coefficient of Variation',
            'Estimates': metrics["est_cv"],
            'Actual': metrics["actual_cv"],
            'Interpretation': f'{"Estimates" if metrics["est_cv"] > metrics["actual_cv"] else "Actual time"} shows more variation relative to mean'
        },
        {
            'Metric': 'Accuracy',
            'Estimates': metrics["percent_within_1_std"],
            'Actual': None,
            'Interpretation': f'{metrics["percent_within_1_std"]:.1f}% of estimates were within ±{metrics["error_std"]:.1f} days of actual time'
        }
    ])
    
    return stats_df

def create_completed_dataframe(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame showing completed issues with relevant metrics."""
    
    # Sort by completion date (descending)
    sorted_df = filtered_df.sort_values(by='Due Date', ascending=False).copy()

    # Add a column showing business days between completion dates
    # Create a shifted series for the previous date (next row's date)
    previous_dates = pd.to_datetime(sorted_df['Due Date'].shift(-1)).dt.date
    due_dates = pd.to_datetime(sorted_df['Due Date']).dt.date
    
    # Calculate business days between each row and the next
    days_between = []
    for current, previous in zip(due_dates, previous_dates):
        if pd.isna(previous):
            days_between.append(None)  # Last row has no "next" date
        else:
            days_between.append(np.busday_count(previous, current))
    sorted_df['Days Since Previous'] = days_between
    
    # Create an Issue URL column for display
    sorted_df['Issue URL'] = sorted_df['Issue'].apply(
        lambda x: f"https://agrium.atlassian.net/browse/{x}"
    )
    
    # Reorder columns to match desired display order
    result_df = sorted_df[['Issue', 'Issue URL', 'Due Date', 'Days Since Previous', 'Est Time', 'Actual Time']]
    
    return result_df

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

    sorted_df = filtered_df.sort_values(by='Due Date', ascending=False)

    ## Add a column showing business days between completion dates
    # Create a shifted series for the previous date (next row's date)
    previous_dates = pd.to_datetime(sorted_df['Due Date'].shift(-1)).dt.date
    due_dates = pd.to_datetime(sorted_df['Due Date']).dt.date
    
    # Calculate business days between each row and the next
    days_between = []
    for current, previous in zip(due_dates, previous_dates):
        if pd.isna(previous):
            days_between.append("")  # Last row has no "next" date
        else:
            days_between.append(np.busday_count(previous, current))
    sorted_df['Days Since Previous'] = days_between

    ## Add columns showing time in each state

    ## Create the completed issues table
    completed_issues_fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Issue', 'Date Completed', 'τ Passed', 'Est Time(d)', 'Actual Time(d)'],
            fill_color='rgba(200,200,200,0.3)',
            font=dict(color='black', size=14),
            align='left'
        ),
        cells=dict(
            values=[
                sorted_df['Issue'],
                sorted_df['Due Date'].astype(str),
                sorted_df['Days Since Previous'],
                sorted_df['Est Time'].apply(lambda x: f'{x:.2f}'),
                sorted_df['Actual Time'].apply(lambda x: f'{x:.2f}'),
            ],
            fill_color='rgba(200,200,200,0.1)',
            font=dict(color='black', size=12),
            align='left',
            height=row_height,
        ),
        columnwidth=[0.5, 0.7, 0.5, 0.5, 0.5]
    )])

    completed_issues_fig.update_layout(
        height=table_height,
        title='Completed Issues',
        margin=dict(t=30, b=0)  # Reduce margins to make it more compact

    )
    
    return completed_issues_fig

def create_state_time_table(state_time_df: pd.DataFrame) -> go.Figure:
    """Create a table showing time spent in each state for each issue."""
    # Check if there's data to display
    if state_time_df.empty:
        return go.Figure()
    
    # Sort by Issue key
    sorted_df = state_time_df.sort_values('Issue')
    
    # Define the desired state order
    desired_state_order = ['In Progress', 'In Review', 'In PO Review', 'Blocked', 'Done']
    
    # Ensure all desired columns exist, add them if they don't
    for state in desired_state_order:
        if state not in sorted_df.columns:
            sorted_df[state] = np.nan

    # Get all state columns (excluding 'Issue')
    state_columns = [col for col in sorted_df.columns if col != 'Issue']
    
    # Calculate column headers - Issue + all states
    headers = ['Issue'] + state_columns
    
    # Calculate column values
    values = [sorted_df['Issue']]
    for state in state_columns:
        values.append(sorted_df[state].apply(
            lambda x: f'{x:.2f}' if pd.notna(x) and x > 0 else ""
        ))
    
    # Calculate dynamic height based on number of rows
    row_height = 30  # matches cell height
    padding = 50     # extra space for title and margins
    table_height = (len(sorted_df) + 1) * row_height + padding
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='rgba(200,200,200,0.3)',
            font=dict(color='black', size=14),
            align='left'
        ),
        cells=dict(
            values=values,
            fill_color='rgba(200,200,200,0.1)',
            font=dict(color='black', size=12),
            align='left',
            height=row_height,
        ),
        editable=True
    )])
    
    fig.update_layout(
        height=table_height,
        title='Time Spent in Each State (Days)',
        margin=dict(t=30, b=0)
    )
    
    return fig

def create_state_time_dataframe(state_time_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame showing time spent in each state for each issue,
    with the same columns, names, values, and initial order as create_state_time_table.
    """
    # Check if there's data to display
    if state_time_df.empty:
        return pd.DataFrame()
    
    # Sort by Issue key
    sorted_df = state_time_df.sort_values('Issue').copy()
    
    sorted_df['Issue URL'] = sorted_df['Issue'].apply(
        lambda x: f"https://agrium.atlassian.net/browse/{x}"
    )

    # Define the desired state order
    desired_state_order = ['In Progress', 'In Review', 'In PO Review', 'Blocked']
    
    # Ensure all desired columns exist, add them if they don't
    for state in desired_state_order:
        if state not in sorted_df.columns:
            sorted_df[state] = np.nan
    
    # Get state columns in desired order (excluding 'Issue')
    state_columns = [col for col in desired_state_order if col in sorted_df.columns]
    
    # Reorder columns to put Issue first followed by state columns in desired order
    sorted_df = sorted_df[['Issue', 'Issue URL'] + state_columns]
    
    # Format the numeric values to 2 decimal places, but only for non-NaN values
    for state in state_columns:
        # Format the values but leave NaN values as is
        sorted_df[state] = sorted_df[state].apply(
            lambda x: round(x, 2) if pd.notna(x) and x > 0 else np.nan
        )
    
    return sorted_df