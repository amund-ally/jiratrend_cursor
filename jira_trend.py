import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import configparser
from jira import JIRA
import numpy as np

def read_config():
    config = configparser.ConfigParser()
    config.read('jira.config')
    return {
        'username': config['JIRA']['username'],
        'api_key': config['JIRA']['api_key']
    }

def get_jira_data():
    config = read_config()
    jira = JIRA('https://your-jira-instance.atlassian.net', 
                basic_auth=(config['username'], config['api_key']))
    
    # Replace with your actual JQL query
    jql_query = 'project = YOUR_PROJECT'
    issues = jira.search_issues(jql_query, maxResults=1000)
    
    data = []
    for issue in issues:
        data.append({
            'key': issue.key,
            'duedate': issue.fields.duedate,
            'originalestimate': issue.fields.timeoriginalestimate / 3600 if issue.fields.timeoriginalestimate else 0
        })
    
    return pd.DataFrame(data)

def calculate_engineer_progress():
    start_date = datetime(2025, 3, 12)
    end_date = datetime(2025, 5, 6)
    today = datetime.now()
    
    # Calculate total work days between start and today
    work_days = 0
    current_date = start_date
    while current_date <= today:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            work_days += 1
        current_date += timedelta(days=1)
    
    # Calculate total hours completed (2 engineers * 6.4 hours * work_days)
    total_hours = 2 * 6.4 * work_days
    
    return today, total_hours

def create_chart(df):
    # Convert duedate to datetime
    df['duedate'] = pd.to_datetime(df['duedate'])
    
    # Group by duedate and sum originalestimate
    daily_estimates = df.groupby('duedate')['originalestimate'].sum().reset_index()
    
    # Sort by date
    daily_estimates = daily_estimates.sort_values('duedate')
    
    # Calculate cumulative sum
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Create date range for the entire project period
    start_date = datetime(2025, 3, 12)
    end_date = datetime(2025, 5, 6)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a complete dataset with all dates
    complete_df = pd.DataFrame({'duedate': date_range})
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    
    # Forward fill the cumulative sum
    complete_df['cumulative_sum'] = complete_df['cumulative_sum'].ffill()
    
    # Get today's progress
    today, total_hours = calculate_engineer_progress()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the cumulative line
    plt.plot(complete_df['duedate'], complete_df['cumulative_sum'], 
             label='Cumulative Original Estimate', color='blue')
    plt.fill_between(complete_df['duedate'], complete_df['cumulative_sum'], 
                     alpha=0.3, color='blue')
    
    # Plot today's progress point
    plt.scatter(today, total_hours, color='red', s=100, 
                label=f'Today\'s Progress ({total_hours:.1f} hours)')
    
    # Customize the plot
    plt.title('Project Progress vs. Original Estimates')
    plt.xlabel('Date')
    plt.ylabel('Hours')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('project_progress.png')
    plt.close()

def main():
    df = get_jira_data()
    create_chart(df)

if __name__ == "__main__":
    main() 