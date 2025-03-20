from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from jira import JIRA
from src.config.jira_config import JiraConfig
from src.config.project_config import ProjectConfig

# Constants
START_DATE = datetime(2025, 3, 12)
END_DATE = datetime(2025, 5, 6)
HOURS_PER_DAY = 12.8

def get_jira_client(config: JiraConfig) -> JIRA:
    """Create and return a JIRA client instance."""
    return JIRA(config.server_url, basic_auth=(config.username, config.api_key))

def get_jira_query(project_config: ProjectConfig) -> str:
    """Return the JQL query for fetching issues."""
    return project_config.jira_query

def process_estimate_changes(issues: List, project_config: ProjectConfig) -> Tuple[Dict, Dict]:
    """Process estimate changes from issue changelogs."""
    estimate_changes: Dict = {}
    current_estimates: Dict = {}
    initial_estimates: Dict = {}
    
    for issue in issues:
        # Get the initial estimate from the issue's current state
        current_estimate = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        
        # Initialize the estimate changes with the current estimate
        if project_config.start_date.date() not in estimate_changes:
            estimate_changes[project_config.start_date.date()] = {}
        estimate_changes[project_config.start_date.date()][issue.key] = current_estimate
        current_estimates[issue.key] = current_estimate
        
        # Process changelog to find the estimate as of start date
        initial_estimate = 0  # Start with 0 by default
        
        # First, find the most recent estimate before start date
        pre_start_estimates = []
        for history in issue.changelog.histories:
            history_datetime = pd.to_datetime(history.created)
            if history_datetime.date() <= project_config.start_date.date():
                for item in history.items:
                    if item.field == 'timeoriginalestimate':
                        estimate = float(item.toString) / (3600 * 8) if item.toString else 0
                        pre_start_estimates.append((history_datetime, estimate))
        
        # Sort by datetime and get the most recent estimate before start date
        if pre_start_estimates:
            pre_start_estimates.sort(key=lambda x: x[0], reverse=True)
            initial_estimate = pre_start_estimates[0][1]
        
        # Then process changes after start date
        post_start_changes = []
        for history in issue.changelog.histories:
            history_datetime = pd.to_datetime(history.created)
            if history_datetime.date() > project_config.start_date.date():
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
    
    # Set the initial estimates for start date
    estimate_changes[project_config.start_date.date()] = initial_estimates
    
    return estimate_changes, current_estimates

def create_scope_dataframe(estimate_changes: Dict, project_config: ProjectConfig) -> pd.DataFrame:
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
    date_range = pd.date_range(start=project_config.start_date.date(), 
                              end=project_config.end_date.date(), 
                              freq='D').date
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

def get_jira_data(jira_config: JiraConfig, project_config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process JIRA data."""
    jira = get_jira_client(jira_config)
    issues = jira.search_issues(get_jira_query(project_config), maxResults=1000, expand='changelog')
    
    estimate_changes, _ = process_estimate_changes(issues, project_config)
    scope_df = create_scope_dataframe(estimate_changes, project_config)
    completed_df = get_completed_work_data(issues)
    
    return completed_df, scope_df 