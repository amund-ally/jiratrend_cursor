from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.widgets import Cursor
import configparser
from jira import JIRA

# Constants
START_DATE = datetime(2025, 3, 12)
END_DATE = datetime(2025, 5, 6)
HOURS_PER_DAY = 12.8
JIRA_CONFIG_FILE = 'jira.config'

class JiraConfig:
    """Configuration class for JIRA credentials."""
    
    def __init__(self, username: str, api_key: str, server_url: str):
        self.username = username
        self.api_key = api_key
        self.server_url = server_url

    @classmethod
    def from_config_file(cls) -> 'JiraConfig':
        """Create JiraConfig instance from config file."""
        config = configparser.ConfigParser()
        config.read(JIRA_CONFIG_FILE)
        return cls(
            username=config['JIRA']['username'],
            api_key=config['JIRA']['api_key'],
            server_url=config['JIRA']['server_url']
        )

def get_jira_client(config: JiraConfig) -> JIRA:
    """Create and return a JIRA client instance."""
    return JIRA(config.server_url, basic_auth=(config.username, config.api_key))

def get_jira_query() -> str:
    """Return the JQL query for fetching issues."""
    return '''
    project = "DS" 
    and (status was in ("To Do", "In Progress", "In Review") after "2025-03-11"
         OR status changed to "Done" after "2025-03-11")
    and status != "Rejected"
    and type not in ("Feature", "Portfolio Epic")
    and (
        (issueKey in portfolioChildIssuesOf(DS-9557) 
        OR issueKey in portfolioChildIssuesOf(DS-12475)
        OR issueKey in (DS-9557, DS-12475))
        OR 
        (parent in (DS-9557, DS-12475))
    )
    and parent not in (DS-12484, DS-9866, DS-12111, DS-9009)   
    ORDER BY statusCategory ASC'''

def process_estimate_changes(issues: List) -> Tuple[Dict, Dict]:
    """Process estimate changes from issue changelogs."""
    estimate_changes: Dict = {}
    current_estimates: Dict = {}
    initial_estimates: Dict = {}
    
    for issue in issues:
        # Get the initial estimate from the issue's current state
        current_estimate = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        
        # Initialize the estimate changes with the current estimate
        if START_DATE.date() not in estimate_changes:
            estimate_changes[START_DATE.date()] = {}
        estimate_changes[START_DATE.date()][issue.key] = current_estimate
        current_estimates[issue.key] = current_estimate
        
        # Process changelog to find the estimate as of March 12th
        initial_estimate = 0  # Start with 0 by default
        
        # First, find the most recent estimate before March 12th
        pre_start_estimates = []
        for history in issue.changelog.histories:
            history_datetime = pd.to_datetime(history.created)
            if history_datetime.date() <= START_DATE.date():
                for item in history.items:
                    if item.field == 'timeoriginalestimate':
                        estimate = float(item.toString) / (3600 * 8) if item.toString else 0
                        pre_start_estimates.append((history_datetime, estimate))
        
        # Sort by datetime and get the most recent estimate before March 12th
        if pre_start_estimates:
            pre_start_estimates.sort(key=lambda x: x[0], reverse=True)
            initial_estimate = pre_start_estimates[0][1]
        
        # Then process changes after March 12th
        post_start_changes = []
        for history in issue.changelog.histories:
            history_datetime = pd.to_datetime(history.created)
            if history_datetime.date() > START_DATE.date():
                for item in history.items:
                    if item.field == 'timeoriginalestimate':
                        new_estimate = float(item.toString) / (3600 * 8) if item.toString else 0
                        post_start_changes.append((history_datetime, new_estimate))
        
        # Sort post-start changes by datetime and apply them
        post_start_changes.sort(key=lambda x: x[0])
        for change_datetime, new_estimate in post_start_changes:
            change_date = change_datetime.date()
            if change_date not in estimate_changes:
                estimate_changes[change_date] = {}
            estimate_changes[change_date][issue.key] = new_estimate
            current_estimates[issue.key] = new_estimate
        
        initial_estimates[issue.key] = initial_estimate
    
    # Set the initial estimates for March 12th
    estimate_changes[START_DATE.date()] = initial_estimates
    
    return estimate_changes, current_estimates

def create_scope_dataframe(estimate_changes: Dict) -> pd.DataFrame:
    """Create DataFrame with all estimate changes."""
    scope_data = []
    running_estimates = {}
    sorted_dates = sorted(estimate_changes.keys())
    
    # Initialize with the first date's estimates
    if sorted_dates:
        running_estimates = estimate_changes[sorted_dates[0]].copy()
    
    for date in sorted_dates:
        # Update running estimates with any changes for this date
        for issue_key, new_estimate in estimate_changes[date].items():
            running_estimates[issue_key] = new_estimate
        
        total = sum(running_estimates.values())
        scope_data.append({
            'date': date,
            'total_estimate': total
        })
    
    scope_df = pd.DataFrame(scope_data).sort_values('date')
    
    # Create complete dataset with all dates
    date_range = pd.date_range(start=START_DATE.date(), end=END_DATE.date(), freq='D').date
    complete_scope_df = pd.DataFrame({'date': date_range})
    complete_scope_df = complete_scope_df.merge(scope_df, on='date', how='left')
    complete_scope_df['total_estimate'] = complete_scope_df['total_estimate'].ffill()
    
    return complete_scope_df

def get_completed_work_data(issues: List) -> pd.DataFrame:
    """Get data for completed work."""
    data = []
    for issue in issues:
        originalestimate_days = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        duedate = pd.to_datetime(issue.fields.duedate).date() if issue.fields.duedate and issue.fields.status.name == "Done" else None
        data.append({
            'key': issue.key,
            'duedate': duedate,
            'originalestimate': originalestimate_days
        })
    return pd.DataFrame(data)

def get_jira_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process JIRA data."""
    config = JiraConfig.from_config_file()
    jira = get_jira_client(config)
    issues = jira.search_issues(get_jira_query(), maxResults=1000, expand='changelog')
    
    estimate_changes, _ = process_estimate_changes(issues)
    scope_df = create_scope_dataframe(estimate_changes)
    completed_df = get_completed_work_data(issues)
    
    return completed_df, scope_df

def calculate_trend_line(valid_data: pd.DataFrame) -> Tuple[Optional[np.poly1d], Optional[datetime], Optional[Tuple[np.poly1d, np.poly1d]]]:
    """Calculate trend line based on total work completed over the period."""
    if len(valid_data) <= 1:
        print("\nTrend calculation: Not enough data points")
        return None, END_DATE, None
    
    # Get only the days where work was completed
    completed_days = valid_data[valid_data['originalestimate'] > 0].copy()
    if len(completed_days) < 2:
        print("Not enough days with completed work")
        return None, END_DATE, None
    
    # Calculate total work and days elapsed
    total_work = completed_days['originalestimate'].sum()
    first_day = completed_days['duedate'].min()
    last_day = completed_days['duedate'].max()
    days_elapsed = (last_day - first_day).days + 1  # +1 to include both start and end days
    
    # Calculate average velocity
    velocity = total_work / days_elapsed
    
    print("\nVelocity calculation:")
    print(f"Total work completed: {total_work:.1f} days")
    print(f"Days elapsed: {days_elapsed}")
    print(f"Average velocity: {velocity:.2f} days/day")
    print(f"First completion: {first_day}")
    print(f"Last completion: {last_day}")
    
    # Create the main trend line
    trend_line = np.poly1d([velocity, 0])
    
    # Calculate a reasonable range for daily velocity variation
    # Using 20% of the average velocity as a reasonable variation
    variation = velocity * 0.2
    upper_line = np.poly1d([velocity + variation, 0])
    lower_line = np.poly1d([velocity - variation, 0])
    
    print(f"\nTrend line: y = {velocity:.4f}x")
    print(f"Upper bound: y = {(velocity + variation):.4f}x")
    print(f"Lower bound: y = {(velocity - variation):.4f}x")
    
    return trend_line, None, (upper_line, lower_line)

def create_chart(df: pd.DataFrame, scope_df: pd.DataFrame) -> None:
    """Create and display the progress chart."""
    df['duedate'] = pd.to_datetime(df['duedate'])
    df_with_dates = df.dropna(subset=['duedate'])
    
    daily_estimates = df_with_dates.groupby('duedate')['originalestimate'].sum().reset_index()
    daily_estimates = daily_estimates.sort_values('duedate')
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Create complete dataset
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    complete_df = pd.DataFrame({'duedate': date_range})
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    complete_df['originalestimate'] = complete_df['originalestimate'].fillna(0)
    complete_df['cumulative_sum'] = complete_df['originalestimate'].cumsum()
    
    # Calculate completion metrics
    today_scope = scope_df['total_estimate'].iloc[-1]
    days_needed = (today_scope * 8) / HOURS_PER_DAY
    completion_date = START_DATE + timedelta(days=days_needed)
    
    today = datetime.now()
    work_days = (today - START_DATE).total_seconds() / (3600 * 24)
    expected_progress = (work_days * HOURS_PER_DAY) / 8
    
    # Calculate trend line
    valid_data = complete_df[complete_df['cumulative_sum'] > 0].copy()
    print("\nData for trend line calculation:")
    print(f"Total rows: {len(complete_df)}")
    print(f"Rows with cumulative sum > 0: {len(valid_data)}")
    trend_line, intersect_date, confidence_intervals = calculate_trend_line(valid_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    plot_data(ax, complete_df, scope_df, trend_line, valid_data, 
              completion_date, today_scope, today, expected_progress, intersect_date,
              confidence_intervals)
    
    # Add interactivity
    add_interactive_annotations(ax, fig, complete_df, scope_df, 
                              completion_date, today_scope, expected_progress,
                              today)
    
    plt.tight_layout()
    plt.show()

def plot_data(ax: plt.Axes, complete_df: pd.DataFrame, scope_df: pd.DataFrame,
              trend_line: Optional[np.poly1d], valid_data: pd.DataFrame,
              completion_date: datetime, today_scope: float,
              today: datetime, expected_progress: float,
              intersect_date: Optional[datetime],
              confidence_intervals: Optional[Tuple[np.poly1d, np.poly1d]] = None) -> None:
    """Plot all data on the chart."""
    # Plot cumulative line
    line, = ax.plot(complete_df['duedate'], complete_df['cumulative_sum'], 
                    label='Completed', color='blue')
    ax.fill_between(complete_df['duedate'], complete_df['cumulative_sum'], 
                    alpha=0.3, color='blue')
    
    # Plot velocity trend line and confidence intervals
    if trend_line is not None:
        # Calculate days from start for each date
        days_from_start = [(d - START_DATE).days for d in complete_df['duedate']]
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
    ax.plot([START_DATE, completion_date], [0, today_scope], 
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

def add_interactive_annotations(ax: plt.Axes, fig: plt.Figure, 
                              complete_df: pd.DataFrame, scope_df: pd.DataFrame,
                              completion_date: datetime, today_scope: float,
                              expected_progress: float, today: datetime) -> None:
    """Add interactive annotations to the chart."""
    annotations = {
        'line': ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)),
        'scope': ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)),
        'completion': ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)),
        'progress': ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    }
    
    for annot in annotations.values():
        annot.set_visible(False)
    
    def update_annot(event):
        if event.inaxes != ax:
            for annot in annotations.values():
                annot.set_visible(False)
            fig.canvas.draw_idle()
            return
            
        x = mdates.num2date(event.xdata)
        x = x.replace(tzinfo=None)
        
        idx = (complete_df['duedate'].dt.tz_localize(None) - x).abs().idxmin()
        date = complete_df.loc[idx, 'duedate']
        value = complete_df.loc[idx, 'cumulative_sum']
        
        x_num = mdates.date2num(date)
        xdata_nums = mdates.date2num(complete_df['duedate'])
        
        y_line = np.interp(x_num, xdata_nums, complete_df['cumulative_sum'])
        distance = abs(event.ydata - y_line)
        
        scope_x = mdates.date2num(date)
        scope_y = np.interp(scope_x, mdates.date2num(scope_df['date']), scope_df['total_estimate'])
        scope_distance = abs(event.ydata - scope_y)
        
        completion_x = mdates.date2num(completion_date)
        completion_distance = np.sqrt((event.xdata - completion_x)**2 + (event.ydata - today_scope)**2)
        
        progress_x = mdates.date2num(today)
        progress_distance = np.sqrt((event.xdata - progress_x)**2 + (event.ydata - expected_progress)**2)
        
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        threshold = y_range * 0.01
        
        for annot in annotations.values():
            annot.set_visible(False)
        
        if distance < threshold:
            annotations['line'].xy = (x_num, value)
            annotations['line'].set_text(f"{date.strftime('%Y-%m-%d')}\n{value:.1f} days")
            annotations['line'].set_visible(True)
        elif scope_distance < threshold:
            annotations['scope'].xy = (scope_x, scope_y-7)
            annotations['scope'].set_text(f"Total Scope\n{scope_y:.1f} days")
            annotations['scope'].set_visible(True)
        elif completion_distance < threshold:
            annotations['completion'].xy = (completion_x, today_scope)
            annotations['completion'].set_text(f"Est Completion\n{completion_date.strftime('%Y-%m-%d %H:%M')}")
            annotations['completion'].set_visible(True)
        elif progress_distance < threshold:
            annotations['progress'].xy = (progress_x, expected_progress)
            annotations['progress'].set_text(f"Ideal Today\n{expected_progress:.1f} days")
            annotations['progress'].set_visible(True)
        
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', update_annot)

def main() -> None:
    """Main entry point for the script."""
    df, scope_df = get_jira_data()
    create_chart(df, scope_df)

if __name__ == "__main__":
    main() 