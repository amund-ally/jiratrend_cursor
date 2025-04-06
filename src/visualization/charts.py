from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.config.chart_config import ChartConfig

def get_empty_figure() -> go.Figure:
    """Return an empty figure with a message."""
    background = 'rgba(255, 255, 255, 0.0)'
    fig_none = go.Figure()
    fig_none.update_layout(
        paper_bgcolor=background,
        plot_bgcolor=background      
    )
    fig_none.update_layout(
        showlegend=False,
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
    )
    
    return fig_none

def calculate_velocity(valid_data: pd.DataFrame, start_date: datetime) -> float:
    """Calculate trend line based on total work completed over the period."""
    if len(valid_data) <= 1:
        return None
    
    # Get the total work completed so far
    completed_work = valid_data['cumulative_sum'].max()
    if completed_work <= 0:
        return None
    
    # Calculate business days between start date and today
    start_date = start_date.date()
    today_date = datetime.now().date()
    business_days_elapsed = np.busday_count(start_date, today_date)
    
    # Ensure we have at least one day to avoid division by zero
    business_days_elapsed = max(1, business_days_elapsed)
    
    # Normalize completed work to 8-hour days
    normalized_completed_work = completed_work
    
    # Calculate velocity as work completed per business day
    velocity = normalized_completed_work / business_days_elapsed
    
    # No trend line function needed; we'll just plot a straight line
    return velocity

def create_progress_chart(df: pd.DataFrame, scope_df: pd.DataFrame, chart_config: ChartConfig, 
                          what_if_days: float = 0.0, velocity_multiplier: float = 1.0) -> go.Figure:
    """Create and return the progress chart using Plotly."""
    df['duedate'] = pd.to_datetime(df['duedate'])
    df_with_dates = df.dropna(subset=['duedate'])
    
    # Sort by date and calculate cumulative sum
    daily_estimates = df_with_dates.groupby('duedate')['originalestimate'].sum().reset_index()
    daily_estimates = daily_estimates.sort_values('duedate')
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Create complete dataset with all dates
    date_range = pd.date_range(start=chart_config.start_date, end=chart_config.end_date, freq='D')
    complete_df = pd.DataFrame({'duedate': date_range})
    
    # Merge with daily estimates and forward fill
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    complete_df['originalestimate'] = complete_df['originalestimate'].fillna(0)
    
    # Calculate cumulative sum for the complete dataset
    complete_df['cumulative_sum'] = complete_df['originalestimate'].cumsum()

    # Calculated actual completed and what_if completed
    completed_today = complete_df['cumulative_sum'].max()
    what_if_completed = completed_today + what_if_days
    
    # Create a modified cumulative_sum column that includes the "what if" value
    complete_df['what_if_sum'] = complete_df['cumulative_sum']
    
     # Add the "what if" amount to today's value and all future values
    today = datetime.now().date()
    complete_df.loc[complete_df['duedate'].dt.date >= today, 'what_if_sum'] += what_if_days

    # Calculate completion metrics based on current scope
    today = datetime.now().date()
    today_scope = scope_df[scope_df['date'] <= today]['total_estimate'].iloc[-1]
    
    # Calculate total business days needed to complete the work
    # (Scope in days * 8 hours) / hours_per_day = total work days needed
    total_business_days_needed = (today_scope * 8) / chart_config.hours_per_day
    
    # Calculate how many business days have elapsed since the start
    start_date = chart_config.start_date.date()
    work_days_elapsed = np.busday_count(start_date, today)
    if np.is_busday(today):
        work_days_elapsed = max(1, work_days_elapsed)
    
    # Calculate how many days of work have been completed based on actual progress
    work_days_completed = completed_today * 8 / chart_config.hours_per_day
    
    # Calculate remaining work days needed
    remaining_days = total_business_days_needed - work_days_completed
    
    # Always round up to ensure we have enough time (ceiling function)
    business_days_remaining = int(np.ceil(remaining_days))
    
    # Alternative calculation for completion date (independent of current progress)
    # This will give a fixed completion date based only on start date, scope and rate
    total_days_needed = (today_scope * 8) / chart_config.hours_per_day
    pure_completion_date = np.busday_offset(start_date, int(np.ceil(total_days_needed)), roll='forward')
    pure_completion_date = pd.Timestamp(pure_completion_date).to_pydatetime().date()

    # Calculate the completion date by adding business days to today
    completion_date = np.busday_offset(today, business_days_remaining, roll='forward')
    completion_date = pd.Timestamp(completion_date).to_pydatetime().date()
    
    # Get total business days from start to completion
    total_plan_business_days = np.busday_count(start_date, pure_completion_date)
    if total_plan_business_days <= 0:
        total_plan_business_days = 1  # Avoid division by zero
    
    # Calculate business days elapsed
    elapsed_business_days = np.busday_count(start_date, today)
    if np.is_busday(today):
        elapsed_business_days = max(1, elapsed_business_days)
    
    # Calculate expected progress as a proportion of time elapsed on the ideal line
    # Using calendar days instead of business days to ensure point falls on the line
    days_from_start_to_today = (today - start_date).days
    days_from_start_to_completion = (pure_completion_date - start_date).days
    
    # Ensure we don't divide by zero
    if days_from_start_to_completion <= 0:
        days_from_start_to_completion = 1
        
    # Calculate expected progress using linear calendar day interpolation
    progress_ratio = days_from_start_to_today / days_from_start_to_completion
    expected_progress = progress_ratio * today_scope
    
    # Calculate trend line
    valid_data = complete_df[complete_df['cumulative_sum'] > 0].copy()

    # Calculate velocity based on valid data
    velocity = calculate_velocity(valid_data, chart_config.start_date)
    ideal_velocity = chart_config.hours_per_day / 8

    # Calculate what-if velocity separately
    what_if_velocity = None
    if what_if_days > 0:
        valid_data_copy = valid_data.copy()
        last_date_idx = valid_data_copy['duedate'].dt.date.idxmax()
        valid_data_copy.loc[last_date_idx, 'cumulative_sum'] += what_if_days
        what_if_velocity = calculate_velocity(valid_data_copy, chart_config.start_date)

    if what_if_velocity is not None:
        what_if_velocity = what_if_velocity * velocity_multiplier
    elif velocity_multiplier != 1.0:
        what_if_velocity = velocity * velocity_multiplier

    # Create the Plotly figure
    fig = go.Figure()
    
    # Add traces in REVERSE order of desired legend appearance
    
    # Add today's expected progress point
    fig.add_trace(go.Scatter(
        x=[today],
        y=[expected_progress],
        name=f'Ideal Today ({expected_progress:.1f} days)',
        mode='markers',
        marker=dict(size=8, color='purple'),
        hovertemplate='%{y:.1f} days<extra></extra>'
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
    
    # Add ideal completion line
    fig.add_trace(go.Scatter(
        x=[start_date, pure_completion_date],  # Using start_date to match calculation
        y=[0, today_scope],
        name=f'Ideal Done ({pure_completion_date.strftime("%Y-%m-%d")})',
        line=dict(color='purple', dash='dash'),
        mode='lines+markers',
        marker=dict(
            size=12, 
            symbol='circle',
            showscale=False,
            color=['rgba(0,0,0,0)', 'purple']  # First point transparent, second point purple
        ),
        hoverinfo='skip'
    ))
    
    # Add velocity trend line if we have calculated velocity
    if velocity is not None and velocity > 0:
        # Calculate projected completion date based on velocity
        remaining_work = today_scope - completed_today
        remaining_business_days = int(np.ceil(remaining_work / velocity))
        
        # Use numpy's busday_offset to add remaining business days to today
        projected_completion_date = np.busday_offset(today, remaining_business_days, roll='forward')
        projected_completion_date = pd.Timestamp(projected_completion_date).to_pydatetime().date()
        
        # Format date for display
        date_str = projected_completion_date.strftime('%Y-%m-%d')
        
        # Calculate +/- 20% velocity points for confidence interval
        velocity_high = velocity * 1.2  # 20% higher velocity
        velocity_low = velocity * 0.8   # 20% lower velocity
        
        # Calculate confidence interval completion dates
        remaining_days_optimistic = int(np.ceil(remaining_work / velocity_high))
        remaining_days_pessimistic = int(np.ceil(remaining_work / velocity_low))
        
        optimistic_completion = np.busday_offset(today, remaining_days_optimistic, roll='forward')
        pessimistic_completion = np.busday_offset(today, remaining_days_pessimistic, roll='forward')
        
        optimistic_completion = pd.Timestamp(optimistic_completion).to_pydatetime().date() 
        pessimistic_completion = pd.Timestamp(pessimistic_completion).to_pydatetime().date()
        
        # Add velocity range
        fig.add_trace(go.Scatter(
            x=[chart_config.start_date, optimistic_completion],
            y=[0, today_scope],
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[chart_config.start_date, pessimistic_completion],
            y=[0, today_scope],
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.1)',
            name='Velocity Range (±20%)',
            hovertemplate='Optimistic: %{text[0]}<br>Pessimistic: %{text[1]}<extra></extra>',
            text=[[optimistic_completion.strftime('%Y-%m-%d'), pessimistic_completion.strftime('%Y-%m-%d')]] * 2
        ))
        
        # Add marker at the projected completion point
        fig.add_trace(go.Scatter(
            x=[projected_completion_date],
            y=[today_scope],
            name=f'Projected Completion ({date_str})',
            mode='markers',
            marker=dict(
                size=12,
                symbol='star',
                color='orange',
                line=dict(color='black', width=1)
            ),
            hovertemplate='Projected completion date: %{x}<br>%{y:.1f} days<extra></extra>'
        ))
        
        # Add simple trend line from start to projected completion
        fig.add_trace(go.Scatter(
            x=[chart_config.start_date, projected_completion_date],
            y=[0, today_scope],
            name=f'Velocity ({velocity:.2f}/{ideal_velocity:.2f})',
            mode='lines',
            line=dict(color='orange', dash='dash', width=2),
            hovertemplate='Current velocity: %{text}<extra></extra>',
            text=[f"{velocity:.2f}/{ideal_velocity:.2f}"] * 2,
        ))

    # Add what-if velocity line if applicable
    if what_if_velocity is not None and what_if_velocity > 0:
        # Calculate projected completion date based on what-if velocity
        remaining_work = today_scope - what_if_completed
        remaining_business_days = int(np.ceil(remaining_work / what_if_velocity))
        
        # Use numpy's busday_offset to add remaining business days to today
        what_if_completion_date = np.busday_offset(today, remaining_business_days, roll='forward')
        what_if_completion_date = pd.Timestamp(what_if_completion_date).to_pydatetime().date()     

        # Add what-if trend line from start to projected completion
        fig.add_trace(go.Scatter(
            x=[chart_config.start_date, what_if_completion_date],
            y=[0, today_scope],
            name=f'Simulation ({what_if_completion_date.strftime("%Y-%m-%d")})',
            line=dict(color='deeppink', dash='dashdot', width=2),
            mode='lines+markers',
            marker=dict(
                size=12, 
                symbol='star',
                showscale=False,
                color=['rgba(0,0,0,0)', 'deeppink']  # First point transparent, second point purple
            ),            
            hovertemplate='Simluation: %{text}<extra></extra>',
            text=[f"{what_if_completion_date.strftime("%Y-%m-%d")}"] * 2,
        ))

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
    
    # Update layout
    fig.update_layout(
        title='Project Progress vs. Estimate',
        xaxis_title='',
        yaxis_title='',
        hovermode='closest',
        showlegend=True,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
    )
    
    # Update axes
    fig.update_xaxes(
        tickangle=-45,
        gridcolor='rgba(128,128,128,0.3)',
        range=[chart_config.start_date, chart_config.end_date],
    )

    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.3)',
        range=[0, max(today_scope, completed_today) * 1.08],  # Add 8% padding to the top
        showgrid=True,
        dtick=5,
        tick0=0,
    )
    
    return fig 

def create_scope_change_barchart() -> go.Figure:
    """Create a bar chart for scope changes."""
    fig = get_empty_figure()
    return fig

def create_state_time_chart(state_time_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing time spent in each state for all issues."""
    if state_time_df.empty:
        return get_empty_figure()
    
    state_columns = [col for col in state_time_df.columns if col != 'Issue']
    
    # Calculate averages for each state
    state_averages = {state: state_time_df[state].mean() for state in state_columns}
    
    # Calculate thresholds (2x the average)
    state_thresholds = {state: avg * 2 for state, avg in state_averages.items()}
    
    fig = go.Figure()
    
    # Create lists to collect all data points by category
    normal_x = []
    normal_y = []
    normal_text = []
    
    outlier_x = []
    outlier_y = []
    outlier_text = []
    
    avg_x = []
    avg_y = []
    
    # Process all states
    for state in state_columns:
        # Find which points are above threshold
        above_threshold = state_time_df[state] > state_thresholds[state]
        
        # Collect normal points
        normal_mask = ~above_threshold & ~state_time_df[state].isna()
        if normal_mask.any():
            normal_points = state_time_df[normal_mask]
            normal_x.extend([state] * len(normal_points))
            normal_y.extend(normal_points[state].values)
            normal_text.extend(normal_points['Issue'].values)
        
        # Collect outlier points
        if above_threshold.any():
            outlier_points = state_time_df[above_threshold]
            outlier_x.extend([state] * len(outlier_points))
            outlier_y.extend(outlier_points[state].values)
            outlier_text.extend(outlier_points['Issue'].values)
        
        # Collect average points
        if not pd.isna(state_averages[state]):
            avg_x.append(state)
            avg_y.append(state_averages[state])
    
    # Add normal points
    if normal_x:
        fig.add_trace(go.Scatter(
            x=normal_x,
            y=normal_y,
            mode='markers',
            name='Issues',
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=normal_text,
            hovertemplate='%{text}: %{y:.2f} days<extra></extra>'
        ))
    
    # Add outlier points
    if outlier_x:
        fig.add_trace(go.Scatter(
            x=outlier_x,
            y=outlier_y,
            mode='markers',
            name='Outliers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=outlier_text,
            hovertemplate='%{text}: %{y:.2f} days<extra></extra>'
        ))
    
    # Add average markers
    if avg_x:
        fig.add_trace(go.Scatter(
            x=avg_x,
            y=avg_y,
            mode='markers',
            marker=dict(size=15, symbol='line-ew', color='blue', line=dict(width=2,color='blue')),
            name='Averages',
            hovertemplate='Average: %{y:.2f} days<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Time Spent in Each State',
        xaxis_title='',
        yaxis_title='Days',
        hovermode='closest',
        height=600,
        xaxis=dict(
            categoryorder='array',
            categoryarray=state_columns
        ),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bordercolor="lightgray",
            borderwidth=1
        )
    )
    
    return fig

def create_state_time_boxplot(state_time_df: pd.DataFrame) -> go.Figure:
    """Create a box plot showing distribution of time spent in each state for all issues."""
    if state_time_df.empty:
        return get_empty_figure()

# Define the desired state order
    state_order = [
        'In Progress', 
        'In Review', 
        'In PO Review', 
        'Blocked', 
    ]

    state_columns = [state for state in state_order if state in state_time_df.columns]

    # Create a long-format DataFrame using melt
    melted_df = pd.melt(
        state_time_df, 
        id_vars=['Issue'],
        value_vars=state_columns,
        var_name='State', 
        value_name='Days',
    )
    
    # Remove rows with NaN values
    melted_df = melted_df.dropna(subset=['Days'])
    
    fig = go.Figure()

    
    # # Create box plots for each state
    for state in state_columns:
        state_data = melted_df[melted_df['State'] == state] 

        fig.add_trace(go.Box(
            y=state_data['Days'],
            name=state,
            boxpoints='all',  # Show outliers as individual points
            #jitter=0.3,            # Add jitter to points for better visibility
            #pointpos=-1.8,         # Position points to the left of the box
            marker=dict(
                color='rgba(0,0,255,0.5)',
                size=5,
                line=dict(width=1, color='black')
            ),
            line=dict(color='blue'),
            fillcolor='rgba(0,0,255,0.1)',
            hoverinfo='y+name',
            hovertemplate='%{y:.2f} days<extra></extra>'
        ))    

    # Update layout
    fig.update_layout(
        title='Distribution of Time Spent in Each State',
        xaxis_title='',
        yaxis_title='Days',
        hovermode='closest',
        height=600,
        # xaxis=dict(
        #     type='category',
        #     # categoryorder='array',
        #     # categoryarray=state_columns
        # ),
        plot_bgcolor='white',
        boxmode='overlay'
    )
    
    # Add grid lines for better readability
    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.3)',
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
    
    return fig