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
    
    for issue in issues:
        current_estimate = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        current_estimates[issue.key] = current_estimate

        if START_DATE.date() not in estimate_changes:
            estimate_changes[START_DATE.date()] = {}
        estimate_changes[START_DATE.date()][issue.key] = current_estimate
        
        for history in issue.changelog.histories:
            for item in history.items:
                if item.field == 'timeoriginalestimate':
                    change_date = pd.to_datetime(history.created).date()
                    new_estimate = float(item.toString) / (3600 * 8) if item.toString else 0
                    
                    if change_date not in estimate_changes:
                        estimate_changes[change_date] = {}
                    estimate_changes[change_date][issue.key] = new_estimate
                    current_estimates[issue.key] = new_estimate
    
    return estimate_changes, current_estimates

def create_scope_dataframe(estimate_changes: Dict) -> pd.DataFrame:
    """Create DataFrame with all estimate changes."""
    scope_data = []
    running_estimates = {}
    sorted_dates = sorted(estimate_changes.keys())
    
    for date in sorted_dates:
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

def calculate_trend_line(valid_data: pd.DataFrame) -> Tuple[Optional[np.poly1d], Optional[datetime]]:
    """Calculate trend line and intersection point."""
    if len(valid_data) <= 1:
        return None, END_DATE
    
    x = np.array([mdates.date2num(d) for d in valid_data['duedate']])
    y = valid_data['cumulative_sum'].values
    
    x_diff = x[-1] - x[0]
    y_diff = y[-1] - y[0]
    slope = y_diff / x_diff if x_diff != 0 else 0
    intercept = y[0] - slope * x[0]
    
    if slope <= 0 or slope >= 100:
        return None, END_DATE
    
    trend_line = np.poly1d([slope, intercept])
    return trend_line, None  # Intersection date will be calculated later

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
    trend_line, intersect_date = calculate_trend_line(valid_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    plot_data(ax, complete_df, scope_df, trend_line, valid_data, 
              completion_date, today_scope, today, expected_progress, intersect_date)
    
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
              intersect_date: Optional[datetime]) -> None:
    """Plot all data on the chart."""
    # Plot cumulative line
    line, = ax.plot(complete_df['duedate'], complete_df['cumulative_sum'], 
                    label='Completed', color='blue')
    ax.fill_between(complete_df['duedate'], complete_df['cumulative_sum'], 
                    alpha=0.3, color='blue')
    
    # Plot trend line
    if trend_line is not None:
        trend_x = np.array([mdates.date2num(d) for d in complete_df['duedate']])
        trend_y = trend_line(trend_x)
        ax.plot(complete_df['duedate'], trend_y, 
                label='Trend Line', color='orange', linestyle='--', alpha=1.0)
    
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
            annotations['scope'].xy = (scope_x-6.5, scope_y-7)
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