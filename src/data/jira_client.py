"""Module for interacting with the JIRA API and processing JIRA data."""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, NamedTuple, Optional
import pandas as pd
from jira import JIRA
from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig
import logging


class JiraIssueData(NamedTuple):
    """Container for JIRA issue data."""
    completed_df: pd.DataFrame
    scope_df: pd.DataFrame
    state_time_df: pd.DataFrame


class JiraClient:
    """Client for JIRA API operations."""
    
    def __init__(self, config: JiraConfig):
        """Initialize JIRA client with the given configuration.
        
        Args:
            config: JIRA server configuration
        """
        self.config = config
        self.client = JIRA(config.server_url, basic_auth=(config.username, config.api_key))
    
    def get_issues(self, query: str) -> List:
        """Fetch issues from JIRA using the provided JQL query.
        
        Args:
            query: JQL query string
            
        Returns:
            List of JIRA issue objects
        """
        return self.client.search_issues(query, maxResults=1000, expand='changelog')


class EstimateProcessor:
    """Processes estimate changes from JIRA issues."""
    
    @staticmethod
    def process_estimate_changes(issues: List, chart_config: ChartConfig) -> Dict:
        """Process estimate changes from issue changelogs.
        
        Args:
            issues: List of JIRA issue objects
            chart_config: Chart configuration with start/end dates
            
        Returns:
            Tuple containing:
                - Dictionary of estimate changes by date
                - Dictionary of current estimates
        """
        estimate_changes: Dict = {}
        ########## there is a bug that more-or-less starts in this function
        # when Jira clones an issue, it takes the estimate of the issue it cloned
        # but doesn't show that estimate in the history. 
        # If the estimate for the issue is changed on the same day it was cloned,
        # this function will record it as a single change on that day to the final value
        # of the changes for that day. That means it'll miss recording the original estimate.
        # Ideas to fix:
        #  1) if estimate_changes was indexed by date and time rather than just date then all entries
        #     would be recorded
        #  2) this system could be changed to simply recording all changes across all days and not
        #.    even worry about start date. the later functioning could group and do what it needs
        #     relative to start date. 
        # This method and how this is designed is a prime example of poor code from AI when compared
        #    to what a good engineer would do. since the prompt included the notion of a start date
        #    and time, the AI spread it everywhere.
        ########## end bug talk

        # Initialize the estimate changes dict with an empty dict for start date
        start_date = chart_config.start_date.date()
        if start_date not in estimate_changes:
            estimate_changes[start_date] = {}
        
        for issue in issues:
            current_estimate = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
            issue_created = pd.to_datetime(issue.fields.created).date()
            
            # First, find if this issue had an estimate before the start date
            estimate_on_start_date = 0  # Default to zero - only include if there's evidence it existed with an estimate
            existed_before_start = issue_created <= start_date
            
            if existed_before_start:
                # Check for estimate history before start date
                pre_start_estimates = []
                for history in issue.changelog.histories:
                    history_datetime = pd.to_datetime(history.created)
                    if history_datetime.date() <= start_date:
                        for item in history.items:
                            if item.field == 'timeoriginalestimate':
                                estimate = float(item.toString) / (3600 * 8) if item.toString else 0
                                pre_start_estimates.append((history_datetime, estimate))
                
                # Sort by datetime and get the most recent estimate before start date
                if pre_start_estimates:
                    pre_start_estimates.sort(key=lambda x: x[0], reverse=True)
                    estimate_on_start_date = pre_start_estimates[0][1]
                elif existed_before_start:
                    # If the issue existed before start date but has no estimate history,
                    # check if it was created with an estimate
                    if issue.fields.timeoriginalestimate and issue_created <= start_date:
                        # Assume the current estimate was set on creation if there's no history
                        estimate_on_start_date = current_estimate
            else:
                # Issue created after start_date: record its initial estimate on its creation date
                # this has to be pulled from the history record for when the estimate changed
                found_initial_estimate = False
                for history in issue.changelog.histories:
                    history_datetime = pd.to_datetime(history.created)
                    if history_datetime.date() == issue_created:
                        for item in reversed(history.items):
                            if item.field == 'timeoriginalestimate':
                                if issue_created not in estimate_changes:
                                    estimate_changes[issue_created] = {}
                                estimate_changes[issue_created][issue.key] = item.fromString
                                found_initial_estimate = True
                                break
                    if found_initial_estimate:
                        break
            
            # Add this issue's initial estimate to the start date ONLY if it had a value then
            if estimate_on_start_date > 0:
                estimate_changes[start_date][issue.key] = estimate_on_start_date
            
            # Process changes after start date
            post_start_changes = []
            
            # Do NOT add creation as a change event - only actual estimate changes
            # from the changelog should be considered
            for history in issue.changelog.histories:
                history_datetime = pd.to_datetime(history.created)
                if history_datetime.date() > start_date:
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
        
        return estimate_changes
    
    @staticmethod
    def create_scope_dataframe(estimate_changes: Dict, chart_config: ChartConfig) -> pd.DataFrame:
        """Create DataFrame with all estimate changes."""
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Starting create_scope_dataframe with {len(estimate_changes)} change dates")
        logger.debug(f"Date range: {chart_config.start_date.date()} to {chart_config.end_date.date()}")
        
        scope_data = []
        running_estimates = {}
        sorted_dates = sorted(estimate_changes.keys())
        
        logger.debug(f"Sorted dates range: {sorted_dates[0]} to {sorted_dates[-1]}" if sorted_dates else "No dates found")
        
        # Initialize with the first date's estimates
        if sorted_dates:
            running_estimates = estimate_changes[sorted_dates[0]].copy()
            logger.debug(f"Initial estimates on {sorted_dates[0]}: {len(running_estimates)} issues, "
                        f"total estimate: {sum(running_estimates.values()):.2f} days")
        
        for date in sorted_dates:
            # Update running estimates with any changes for this date
            if date != sorted_dates[0]:  # Skip first date as we've already initialized with it
                changes_count = len(estimate_changes[date])
                if changes_count > 0:
                    logger.debug(f"Processing {changes_count} estimate changes for {date}")
                    logger.debug(f"Changes: {estimate_changes[date]}")
                    
                for issue_key, new_estimate in estimate_changes[date].items():
                    old_estimate = running_estimates.get(issue_key, 0)
                    running_estimates[issue_key] = new_estimate
                    logger.debug(f"Issue {issue_key} estimate changed: {old_estimate:.2f} -> {new_estimate:.2f}")
            
            total = sum(running_estimates.values())
            logger.debug(f"Total estimate for {date}: {total:.2f} days")
            
            scope_data.append({
                'date': date,
                'total_estimate': total
            })
        
        scope_df = pd.DataFrame(scope_data).sort_values('date')
        logger.debug(f"Created initial scope_df with {len(scope_df)} rows")
        
        # Create complete dataset with all dates
        date_range = pd.date_range(start=chart_config.start_date.date(), 
                                  end=chart_config.end_date.date(), 
                                  freq='D').date
        logger.debug(f"Creating complete date range with {len(date_range)} days")
        
        complete_scope_df = pd.DataFrame({'date': date_range})
        complete_scope_df = complete_scope_df.merge(scope_df, on='date', how='left')
        complete_scope_df['total_estimate'] = complete_scope_df['total_estimate'].ffill()
        
        logger.debug(f"Final complete_scope_df: {len(complete_scope_df)} rows")
        logger.debug(f"First row: {complete_scope_df.iloc[0].to_dict()}")
        logger.debug(f"Last row: {complete_scope_df.iloc[-1].to_dict()}")
        
        return complete_scope_df


class CompletedWorkProcessor:
    """Processes completed work data from JIRA issues."""
    
    @staticmethod
    def get_completed_work_data(issues: List, actual_completed_df: pd.DataFrame) -> pd.DataFrame:
        """Get data for completed work.
        
        Args:
            issues: List of JIRA issue objects
            actual_completed_df: DataFrame with actual completion data
            
        Returns:
            DataFrame with completed work data
        """
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
    
    @staticmethod
    def get_actual_completion_data(issues: List) -> pd.DataFrame:
        """Process actual time spent data from issue changelog.
        
        Args:
            issues: List of JIRA issue objects
            
        Returns:
            DataFrame with actual completion data
        """
        changelogList = {issue.key: issue.raw['changelog'] for issue in issues}
        issueList = {issue.key: issue.raw['fields'] for issue in issues}

        data = pd.DataFrame.from_dict(issueList, orient='index')
        data['changelog'] = changelogList
        data['issuehistory'] = [issueRecord.get('histories') for issueRecord in data['changelog']]
        data['status'] = [issueRecord.get('name') for issueRecord in data['status']]

        # Status IDs for working states
        working_states = ['10500', '10100', '10645']  # In Progress, In PO Review, In Review
        
        # iterate through each row and create a data frame with an entry for each
        # status change, inserted into a dictionary indexed by the issueKey
        statusChanges = defaultdict(pd.DataFrame)
        for index, issueRecord in data.iterrows():
            statusRecord = { 
                "created": [issueRecord['created']], 
                "fromStatusId": ['0'], 
                "fromStatusString": ['Nothing'], 
                "toStatusId": ['10000'], 
                "toStatusString": ['To Do']
            }
            statusChanges[index] = pd.DataFrame(statusRecord, index=[index])
            statusChanges[index]['created'] = pd.to_datetime(statusChanges[index]['created'], utc=True)
            
            for issueHistory in issueRecord['issuehistory']:
                for historyItem in issueHistory['items']:
                    if historyItem['field'] == 'status':
                        statusRecord = { 
                            "created": [issueHistory['created']], 
                            "fromStatusId": [historyItem['from']], 
                            "fromStatusString": [historyItem['fromString']], 
                            "toStatusId": [historyItem['to']], 
                            "toStatusString": [historyItem['toString']]
                        }
                        
                        if index in statusChanges:
                            statusChanges[index] = pd.concat([statusChanges[index], pd.DataFrame(statusRecord, index=[index])], ignore_index=True)
                        else:
                            statusChanges[index] = pd.DataFrame(statusRecord, index=[index])

                        statusChanges[index]['created'] = pd.to_datetime(statusChanges[index]['created'], utc=True)

        # using the dataframe from the above loop, compute the time difference between
        # appropriate state changes and sum it up
        workingTime = []
        for index, changes in statusChanges.items():
            changes.sort_values(by='created', ascending=True, inplace=True)
            # Fix the warning by explicitly specifying the dtype
            duration = (changes['created'] - changes['created'].shift())
            changes['duration'] = duration.fillna(pd.Timedelta(seconds=0))
            timeSpentWorking = 0.0
            
            for _, row in changes.iterrows():
                # Only count time spent in working states
                if row['fromStatusId'] in working_states:
                    timeSpentWorking += row['duration'].total_seconds()
            
            workingTime.append(timeSpentWorking / 60 / 60 / 24)  # Convert to days

        # Add computed working time to the data
        data['timespentworking'] = workingTime
        return data[['timespentworking', 'status']].copy()


class StateTimeProcessor:
    """Processes state time data from JIRA issues."""
    
    @staticmethod
    def get_state_time_analysis(issues: List) -> pd.DataFrame:
        """Calculate how much time each issue spent in each Jira state.
        
        Args:
            issues: List of JIRA issue objects
            
        Returns:
            DataFrame with issue keys and time spent in each state
        """
        changelogList = {issue.key: issue.raw['changelog'] for issue in issues}
        issueList = {issue.key: issue.raw['fields'] for issue in issues}

        data = pd.DataFrame.from_dict(issueList, orient='index')
        data['changelog'] = changelogList
        data['issuehistory'] = [issueRecord.get('histories') for issueRecord in data['changelog']]
        data['status'] = [issueRecord.get('name') for issueRecord in data['status']]
        
        # Define tracked states
        tracked_status_ids = {
            '10500': 'In Progress', 
            '10645': 'In Review',
            '10100': 'In PO Review',
            '10600': 'Blocked'
        }

        # Process status changes for each issue
        state_time_data = []
        
        for index, issueRecord in data.iterrows():
            # Create a list to store all status transitions
            transitions = []
            
            # Add initial state when created
            transitions.append({
                'timestamp': pd.to_datetime(issueRecord['created'], utc=True),
                'toStatus': '10000',  # Assuming 'To Do' is the initial state
                'toStatusName': 'To Do'
            })

            # Track if issue has any transitions to tracked states
            has_tracked_state_transitions = False        
            
            # Add all transitions from history
            for issueHistory in issueRecord['issuehistory']:
                for historyItem in issueHistory['items']:
                    if historyItem['field'] == 'status':
                        # Check if this transition is to a tracked state
                        if historyItem['to'] in tracked_status_ids:
                            has_tracked_state_transitions = True

                        transitions.append({
                            'timestamp': pd.to_datetime(issueHistory['created'], utc=True),
                            'toStatus': historyItem['to'],
                            'toStatusName': historyItem['toString']
                        })

            # Skip issues with no transitions or only the initial state
            if len(transitions) <= 1 or not has_tracked_state_transitions:
                continue

            # Sort transitions by timestamp
            transitions.sort(key=lambda x: x['timestamp'])
            
            # Calculate time spent in each state
            state_times = {state: 0 for state in tracked_status_ids.values()}
            
            # Process each transition
            for i in range(1, len(transitions)):
                prev_timestamp = transitions[i-1]['timestamp']
                curr_timestamp = transitions[i]['timestamp']
                
                # The state we're transitioning FROM is the state we were IN
                from_state = transitions[i-1]['toStatus']
                
                # Only count time for tracked states
                if from_state in tracked_status_ids:
                    state_name = tracked_status_ids[from_state]
                    duration_seconds = (curr_timestamp - prev_timestamp).total_seconds()
                    state_times[state_name] += duration_seconds / (60*60*24)  # Convert to days
            
            # For the current (final) state, calculate time until now
            final_state = transitions[-1]['toStatus']
            final_timestamp = transitions[-1]['timestamp']
            now = pd.Timestamp.utcnow()
            
            if final_state in tracked_status_ids:
                state_name = tracked_status_ids[final_state]
                duration_seconds = (now - final_timestamp).total_seconds()
                state_times[state_name] += duration_seconds / (60*60*24)  # Convert to days
            
            # Skip issues where no time was spent in any tracked state
            if all(time_spent == 0 for time_spent in state_times.values()):
                continue

            # Create issue data record
            issue_data = {'Issue': index}
            
            # Add time for each tracked state
            for state, time_spent in state_times.items():
                if time_spent >= 0.01:  # Only add states where time was spent
                    issue_data[state] = time_spent
            
            state_time_data.append(issue_data)
        
        # Create DataFrame
        state_time_df = pd.DataFrame(state_time_data)
        
        return state_time_df


def get_jira_data(jira_config: JiraConfig, chart_config: ChartConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch and process JIRA data.
    
    Args:
        jira_config: JIRA server configuration
        chart_config: Chart configuration with JQL query and date range
        
    Returns:
        Tuple containing:
            - DataFrame with completed work data
            - DataFrame with scope changes data
            - DataFrame with state time analysis data
            - Dictionary with raw estimate changes
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting JIRA data retrieval and processing")
    
    try:
        # Initialize JIRA client
        jira_client = JiraClient(jira_config)
        
        # Fetch issues from JIRA
        issues = jira_client.get_issues(chart_config.jira_query)
        
        # Process estimate changes
        logger.debug("Processing estimate changes")
        estimate_changes = EstimateProcessor.process_estimate_changes(issues, chart_config)
        scope_df = EstimateProcessor.create_scope_dataframe(estimate_changes, chart_config)
        logger.debug(f"Created scope dataframe with {len(scope_df)} rows")

        # Process completed work
        logger.debug("Processing completed work")
        actual_completed_df = CompletedWorkProcessor.get_actual_completion_data(issues)
        completed_df = CompletedWorkProcessor.get_completed_work_data(issues, actual_completed_df)
        logger.debug(f"Created completed work dataframe with {len(completed_df)} rows")

        # Get state time analysis data
        logger.debug("Processing state time analysis")
        state_time_df = StateTimeProcessor.get_state_time_analysis(issues)
        logger.debug(f"Created state time dataframe with {len(state_time_df)} rows")

        logger.info("Successfully processed all JIRA data")
        return completed_df, scope_df, state_time_df, estimate_changes
        
    except Exception as e:
        logger.error(f"Failed to process JIRA data: {str(e)}", exc_info=True)
        raise