from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.config.project_config import ProjectConfig

def calculate_trend_line(valid_data: pd.DataFrame, project_config: ProjectConfig) -> Tuple[Optional[np.poly1d], Optional[float], Optional[Tuple[np.poly1d, np.poly1d]]]:
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
    
    return trend_line, velocity, (upper_line, lower_line)

def create_chart(df: pd.DataFrame, scope_df: pd.DataFrame, project_config: ProjectConfig) -> go.Figure:
    """Create and return the progress chart using Plotly."""
    df['duedate'] = pd.to_datetime(df['duedate'])
    df_with_dates = df.dropna(subset=['duedate'])
    
    # Sort by date and calculate cumulative sum
    daily_estimates = df_with_dates.groupby('duedate')['originalestimate'].sum().reset_index()
    daily_estimates = daily_estimates.sort_values('duedate')
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Create complete dataset with all dates
    date_range = pd.date_range(start=project_config.start_date, end=project_config.end_date, freq='D')
    complete_df = pd.DataFrame({'duedate': date_range})
    
    # Merge with daily estimates and forward fill
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    complete_df['originalestimate'] = complete_df['originalestimate'].fillna(0)
    
    # Calculate cumulative sum for the complete dataset
    complete_df['cumulative_sum'] = complete_df['originalestimate'].cumsum()
    completed_today = complete_df['cumulative_sum'].max()
    
    # Calculate completion metrics
    today_scope = scope_df['total_estimate'].max()
    days_needed = (today_scope * 8) / project_config.hours_per_day
    completion_date = project_config.start_date + timedelta(days=days_needed)
    
    today = datetime.now()
    work_days = (today - project_config.start_date).total_seconds() / (3600 * 24)
    expected_progress = (work_days * project_config.hours_per_day) / 8
    
    # Calculate trend line
    valid_data = complete_df[complete_df['cumulative_sum'] > 0].copy()
    trend_line, velocity, confidence_intervals = calculate_trend_line(valid_data, project_config)
    ideal_velocity = project_config.hours_per_day / 8

    # Create the Plotly figure
    fig = go.Figure()
    
    # Add completed work line
    fig.add_trace(go.Scatter(
        x=complete_df['duedate'].tolist(),
        y=complete_df['cumulative_sum'].tolist(),
        name=f'Completed ({completed_today:.1f} days)',
        mode='lines',
        line=dict(color='blue', width=2, shape='linear'),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.3)',
        connectgaps=True,
        hovertemplate='%{y:.1f} days<br>%{x}<extra></extra>'
    ))
        
    # Add today's expected progress point
    fig.add_trace(go.Scatter(
        x=[today],
        y=[expected_progress],
        name=f'Ideal Today ({expected_progress:.1f} days)',
        mode='markers',
        marker=dict(size=8, color='purple'),
        hovertemplate='%{y:.1f} days<extra></extra>'
    ))
    
    # Add velocity trend line and confidence intervals if available
    if trend_line is not None:
        # Calculate days from start for each date
        days_from_start = [(d - project_config.start_date).days for d in complete_df['duedate']]
        trend_values = trend_line(days_from_start)
        
        # Add the main trend line
        fig.add_trace(go.Scatter(
            x=complete_df['duedate'].tolist(),
            y=trend_values.tolist(),
            name=f'Velocity ({velocity:.1f}/{ideal_velocity:.1f} days)',
            mode='lines',
            line=dict(color='orange', dash='dash', width=2, shape='linear'),
            hoverinfo='skip',
#            customdata=[velocity] * len(complete_df),
#            hovertemplate='%{customdata:.1f} days<extra></extra>',
        ))
        
        # Add confidence intervals if available
        if confidence_intervals is not None:
            upper_line, lower_line = confidence_intervals
            upper_values = upper_line(days_from_start)
            lower_values = lower_line(days_from_start)
            
            # Add the confidence interval area
            fig.add_trace(go.Scatter(
                x=complete_df['duedate'].tolist() + complete_df['duedate'].tolist()[::-1],
                y=upper_values.tolist() + lower_values.tolist()[::-1],
                name='Velocity Range (±20%)',
                fill='toself',
                fillcolor='rgba(255,165,0,0.1)',
                line=dict(color='rgba(255,165,0,0)'),
                hoverinfo='skip'
            ))
    
    # Add ideal completion line
    fig.add_trace(go.Scatter(
        x=[project_config.start_date, completion_date],
        y=[0, today_scope],
        name=f'Ideal Done ({completion_date.strftime("%Y-%m-%d")})',
        line=dict(color='purple', dash='dash'),
        mode='lines+markers',
        marker=dict(
            size=10, 
            symbol='star',
            showscale=False,
            color=['rgba(0,0,0,0)', 'purple']  # First point transparent, second point purple
        ),
        hoverinfo='skip'
    ))

    # Add scope line
    fig.add_trace(go.Scatter(
        x=scope_df['date'].tolist(),
        y=scope_df['total_estimate'].tolist(),
        name=f'Total Scope ({today_scope:.1f} days)',
        mode='lines',
        line=dict(color='green', dash='dash', shape='linear'),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)',
        hovertemplate='%{y:.1f} day<br>%{x}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Project Progress vs. Estimate',
        xaxis_title='',
        yaxis_title='',
        hovermode='closest',
        showlegend=True,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    # Update axes
    fig.update_xaxes(
        tickangle=-45,
        gridcolor='rgba(128,128,128,0.3)',
        range=[project_config.start_date, project_config.end_date]  # Set x-axis range
    )
    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.3)'
    )
    
    return fig 