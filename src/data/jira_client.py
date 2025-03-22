from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from jira import JIRA
from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig

def get_jira_client(config: JiraConfig) -> JIRA:
    """Create and return a JIRA client instance."""
    return JIRA(config.server_url, basic_auth=(config.username, config.api_key))

def get_jira_query(chart_config: ChartConfig) -> str:
    """Return the JQL query for fetching issues."""
    return chart_config.jira_query

def process_estimate_changes(issues: dict, chart_config: ChartConfig) -> Tuple[Dict, Dict]:
    """Process estimate changes from issue changelogs."""
    estimate_changes: Dict = {}
    current_estimates: Dict = {}
    initial_estimates: Dict = {}
    
    for issue in issues:
        # Get the initial estimate from the issue's current state
        current_estimate = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        
        # Initialize the estimate changes with the current estimate
        if chart_config.start_date.date() not in estimate_changes:
            estimate_changes[chart_config.start_date.date()] = {}
        estimate_changes[chart_config.start_date.date()][issue.key] = current_estimate
        current_estimates[issue.key] = current_estimate
        
        # Process changelog to find the estimate as of start date
        initial_estimate = 0  # Start with 0 by default
        
        # First, find the most recent estimate before start date
        pre_start_estimates = []
        for history in issue.changelog.histories:
            history_datetime = pd.to_datetime(history.created)
            if history_datetime.date() <= chart_config.start_date.date():
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
            if history_datetime.date() > chart_config.start_date.date():
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
    estimate_changes[chart_config.start_date.date()] = initial_estimates
    
    return estimate_changes, current_estimates

def create_scope_dataframe(estimate_changes: Dict, chart_config: ChartConfig) -> pd.DataFrame:
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
    date_range = pd.date_range(start=chart_config.start_date.date(), 
                              end=chart_config.end_date.date(), 
                              freq='D').date
    complete_scope_df = pd.DataFrame({'date': date_range})
    complete_scope_df = complete_scope_df.merge(scope_df, on='date', how='left')
    complete_scope_df['total_estimate'] = complete_scope_df['total_estimate'].ffill()
    
    return complete_scope_df

def get_completed_work_data(issues: dict, actual_completed_df: pd.DataFrame) -> pd.DataFrame:
    """Get data for completed work."""
    data = []
    for issue in issues:
        originalestimate_days = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        duedate = pd.to_datetime(issue.fields.duedate).date() if issue.fields.duedate and issue.fields.status.name == "Done" else None
        data.append({
            'key': issue.key,
            'duedate': duedate,
            'originalestimate': originalestimate_days,
            'timespentworking': actual_completed_df.loc[issue.key, 'timespentworking'] if issue.key in actual_completed_df.index else 0,
        })
    return pd.DataFrame(data)

def get_actual_completion_data(issues: dict) -> pd.DataFrame:
    changelogList = {issue.key: issue.raw['changelog'] for issue in issues}
    issueList = {issue.key: issue.raw['fields'] for issue in issues}

    data = pd.DataFrame.from_dict(issueList, orient='index')
    data['changelog'] = changelogList
    data['issuehistory'] = [issueRecord.get('histories') for issueRecord in data['changelog']]
    data['status'] = [issueRecord.get('name') for issueRecord in data['status']]

    
    ## iterate through each row and create a data frame with an entry for each
    ## status change, inserted into a dictionary indexed by the issueKey
    statusChanges = defaultdict(pd.DataFrame)
    for index, issueRecord in data.iterrows():
        statusRecord = { "created":[issueRecord['created']], "fromStatusId":['0'], "fromStatusString" : ['Nothing'], "toStatusId": ['10000'], "toStatusString": ['To Do']}
        statusChanges[index] = pd.DataFrame(statusRecord, index=[index])
        statusChanges[index]['created'] = pd.to_datetime(statusChanges[index]['created'], utc=True)
        
        for issueHistory in issueRecord['issuehistory']:
            for historyItem in issueHistory['items']:
                if index == 'DS-11656':
                    print(historyItem)

                if historyItem['field'] == 'status':
                    statusRecord = { "created":[issueHistory['created']], "fromStatusId":[historyItem['from']], "fromStatusString" : [historyItem['fromString']], "toStatusId": [historyItem['to']], "toStatusString": [historyItem['toString']]}
                    
                    if index in statusChanges:
                        statusChanges[index] = pd.concat([statusChanges[index], pd.DataFrame(statusRecord, index=[index])], ignore_index=True)
                    else:
                        statusChanges[index] = pd.DataFrame(statusRecord, index=[index])

                    statusChanges[index]['created'] = pd.to_datetime(statusChanges[index]['created'], utc=True)

    ##using the dataframe from the above loop, compute the time difference between
    ##appropriate state changes and sum it up. Add to a list.
    workingTime = []
    for index, changes in statusChanges.items():
        changes.sort_values(by='created',ascending=True, inplace=True)
        #pydatetime = pd.Series(changes['created'].dt.to_pydatetime())
        changes['duration'] = (changes['created']-changes['created'].shift()).fillna(0)
        timeSpentWorking = 0.0
        for subindex, row in changes.iterrows():
            # 10744 = Backlog
            # 10000 = To Do
            # 10500 = In Progress
            # 10645 = In Review
            # 10001 = Done
            # 10100 = In PO Review
            fromStatusId = row['fromStatusId']
            #toStatusId = row['toStatusId']
            #moving from In Progress, In PO Review or In Review to anything
            if fromStatusId == '10500' or fromStatusId == '10100' or fromStatusId == '10645': 
                timeSpentWorking += row['duration'].total_seconds()
        
        workingTime.append(timeSpentWorking/60/60/24)

    ##create new row that represents the time spent on each story,
    ##computed from the history to account for flip-flopping state changes
    data['timespentworking'] = workingTime
    return data[['timespentworking', 'status']].copy()

def get_jira_data(jira_config: JiraConfig, chart_config: ChartConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process JIRA data."""
    jira = get_jira_client(jira_config)
    issues = jira.search_issues(get_jira_query(chart_config), maxResults=1000, expand='changelog')
    
    estimate_changes, _ = process_estimate_changes(issues, chart_config)
    scope_df = create_scope_dataframe(estimate_changes, chart_config)

    actual_completed_df = get_actual_completion_data(issues)
    completed_df = get_completed_work_data(issues, actual_completed_df)

    return completed_df, scope_df