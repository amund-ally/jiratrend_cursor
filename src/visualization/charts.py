from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.widgets import Cursor
from src.config.project_config import ProjectConfig

# Constants
START_DATE = datetime(2025, 3, 12)
END_DATE = datetime(2025, 5, 6)
HOURS_PER_DAY = 12.8

def calculate_trend_line(valid_data: pd.DataFrame, project_config: ProjectConfig) -> Tuple[Optional[np.poly1d], Optional[datetime], Optional[Tuple[np.poly1d, np.poly1d]]]:
    """Calculate trend line based on total work completed over the period."""
    if len(valid_data) <= 1:
        return None, project_config.end_date, None
    
    # Get only the days where work was completed
    completed_days = valid_data[valid_data['originalestimate'] > 0].copy()
    if len(completed_days) < 2:
        return None, project_config.end_date, None
    
    # Calculate total work and days elapsed
    total_work = completed_days['originalestimate'].sum()
    first_day = completed_days['duedate'].min()
    last_day = completed_days['duedate'].max()
    days_elapsed = (last_day - first_day).days + 1  # +1 to include both start and end days
    
    # Calculate average velocity
    velocity = total_work / days_elapsed
    
    # Create the main trend line
    trend_line = np.poly1d([velocity, 0])
    
    # Calculate a reasonable range for daily velocity variation
    # Using 20% of the average velocity as a reasonable variation
    variation = velocity * 0.2
    upper_line = np.poly1d([velocity + variation, 0])
    lower_line = np.poly1d([velocity - variation, 0])
    
    return trend_line, None, (upper_line, lower_line)

def create_chart(df: pd.DataFrame, scope_df: pd.DataFrame, project_config: ProjectConfig) -> plt.Figure:
    """Create and return the progress chart."""
    df['duedate'] = pd.to_datetime(df['duedate'])
    df_with_dates = df.dropna(subset=['duedate'])
    
    daily_estimates = df_with_dates.groupby('duedate')['originalestimate'].sum().reset_index()
    daily_estimates = daily_estimates.sort_values('duedate')
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Create complete dataset
    date_range = pd.date_range(start=project_config.start_date, end=project_config.end_date, freq='D')
    complete_df = pd.DataFrame({'duedate': date_range})
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    complete_df['originalestimate'] = complete_df['originalestimate'].fillna(0)
    complete_df['cumulative_sum'] = complete_df['originalestimate'].cumsum()
    
    # Calculate completion metrics
    today_scope = scope_df['total_estimate'].max()
    days_needed = (today_scope * 8) / project_config.hours_per_day
    completion_date = project_config.start_date + timedelta(days=days_needed)
    
    today = datetime.now()
    work_days = (today - project_config.start_date).total_seconds() / (3600 * 24)
    expected_progress = (work_days * project_config.hours_per_day) / 8
    
    # Calculate trend line
    valid_data = complete_df[complete_df['cumulative_sum'] > 0].copy()
    trend_line, intersect_date, confidence_intervals = calculate_trend_line(valid_data, project_config)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    plot_data(ax, complete_df, scope_df, trend_line, valid_data, 
              completion_date, today_scope, today, expected_progress, intersect_date,
              confidence_intervals, project_config)
    
    plt.tight_layout()
    return fig

def plot_data(ax: plt.Axes, complete_df: pd.DataFrame, scope_df: pd.DataFrame,
              trend_line: Optional[np.poly1d], valid_data: pd.DataFrame,
              completion_date: datetime, today_scope: float,
              today: datetime, expected_progress: float,
              intersect_date: Optional[datetime],
              confidence_intervals: Optional[Tuple[np.poly1d, np.poly1d]],
              project_config: ProjectConfig) -> None:
    """Plot all data on the chart."""
    # Plot cumulative line
    line, = ax.plot(complete_df['duedate'], complete_df['cumulative_sum'], 
                    label='Completed', color='blue')
    ax.fill_between(complete_df['duedate'], complete_df['cumulative_sum'], 
                    alpha=0.3, color='blue')
    
    # Plot velocity trend line and confidence intervals
    if trend_line is not None:
        # Calculate days from start for each date
        days_from_start = [(d - project_config.start_date).days for d in complete_df['duedate']]
        trend_values = trend_line(days_from_start)
        
        # Plot the main trend line
        ax.plot(complete_df['duedate'], trend_values, 
                label='Current Velocity Projection', color='orange', linestyle='--', alpha=1.0)
        
        # Plot confidence intervals if available
        if confidence_intervals is not None:
            upper_line, lower_line = confidence_intervals
            upper_values = upper_line(days_from_start)
            lower_values = lower_line(days_from_start)
            
            # Fill between the confidence intervals
            ax.fill_between(complete_df['duedate'], lower_values, upper_values,
                          color='orange', alpha=0.1, label='Velocity Range (±20%)')
    
    # Plot scope line
    scope_line, = ax.plot(scope_df['date'], scope_df['total_estimate'], 
                         label=f'Total Scope ({scope_df["total_estimate"].max():.1f} days)', 
                         color='green', linestyle='--')
    ax.fill_between(scope_df['date'], scope_df['total_estimate'], 
                    alpha=0.1, color='green')
    
    # Plot intersection point if valid
    if intersect_date:
        ax.scatter(intersect_date, today_scope, color='red', s=100, 
                   label=f'Intersection ({intersect_date.strftime("%Y-%m-%d")})')
        ax.axvline(x=intersect_date, color='red', linestyle=':', alpha=0.5)
    
    # Plot completion point
    ax.scatter(completion_date, today_scope, color='purple', s=150, 
               marker='*', label=f'Ideal Done ({completion_date.strftime("%Y-%m-%d")})')
    ax.plot([project_config.start_date, completion_date], [0, today_scope], 
            color='purple', linestyle='--', alpha=0.5)
    
    # Plot today's expected progress
    ax.scatter(today, expected_progress, color='purple', s=50, 
               marker='o', label=f'Ideal Today ({expected_progress:.1f} days)')
    
    # Customize plot
    ax.set_title('Project Progress vs. Estimate')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    
    max_y = max(complete_df['cumulative_sum'].max(), scope_df['total_estimate'].max(), 
                expected_progress) + 2
    ax.set_ylim(0, max_y)
    plt.xticks(rotation=45) 