import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import configparser
from jira import JIRA
import numpy as np
import sys
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates

def read_config():
    config = configparser.ConfigParser()
    config.read('jira.config')
    return {
        'username': config['JIRA']['username'],
        'api_key': config['JIRA']['api_key'],
        'server_url': config['JIRA']['server_url']
    }

def get_jira_data():
    config = read_config()
    jira = JIRA(config['server_url'], 
                basic_auth=(config['username'], config['api_key']))
    
    jql_query = '''
    project = "DS" 
    and status was in ("To Do", "In Progress", "In Review") after "2025-03-11"
    and status != "Rejected"
    and type not in ("Feature", "Portfolio Epic")
    and (
        (issueKey in portfolioChildIssuesOf(DS-9557) 
        OR issueKey in portfolioChildIssuesOf(DS-12475)
        OR issueKey in (DS-9557, DS-12475))
        OR 
        (parent in (DS-9557, DS-12475))
    )
    ORDER BY statusCategory ASC'''
    issues = jira.search_issues(jql_query, maxResults=1000)
    
    data = []
    for issue in issues:
        # Convert seconds to days (assuming 8 hours per day for estimates)
        originalestimate_days = issue.fields.timeoriginalestimate / (3600 * 8) if issue.fields.timeoriginalestimate else 0
        # Convert duedate to date-only if it exists
        duedate = pd.to_datetime(issue.fields.duedate).date() if issue.fields.duedate and issue.fields.status.name == "Done" else None
        data.append({
            'key': issue.key,
            'duedate': duedate,
            'originalestimate': originalestimate_days
        })
    
    return pd.DataFrame(data)

def calculate_engineer_progress():
    start_date = datetime(2025, 3, 12).date()
    end_date = datetime(2025, 5, 6).date()
    today = datetime.now().date()
    
    # Calculate total work days between start and today
    work_days = 0
    current_date = start_date
    while current_date <= today:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            work_days += 1
        current_date += timedelta(days=1)
    print(f"Work days: {work_days}")
    
    # Calculate total hours completed (2 engineers * 6.4 hours * work_days)
    total_hours = 2 * 6.4 * work_days
    print(f"Total hours: {total_hours}")
    
    # Convert hours to 8-hour days (to match the y-axis scale)
    total_days = total_hours / 8
    print(f"Total days: {total_days}")
    
    return today, total_days

def create_chart(df):
    # Convert duedate to datetime (date-only)
    df['duedate'] = pd.to_datetime(df['duedate']).dt.date
    
    # Filter out issues without due dates
    df_with_dates = df.dropna(subset=['duedate'])
    for idx, row in df_with_dates.iterrows():
        print(f"{row['key']} | {row['duedate'].strftime('%Y-%m-%d')} | {row['originalestimate']:16.2f}")
    # Group by duedate and sum originalestimate
    daily_estimates = df_with_dates.groupby('duedate')['originalestimate'].sum().reset_index()
    
    # Sort by date
    daily_estimates = daily_estimates.sort_values('duedate')
    
    # Calculate cumulative sum
    daily_estimates['cumulative_sum'] = daily_estimates['originalestimate'].cumsum()
    
    # Print daily estimates data
    print("\nDaily Estimates Data:")
    print("Index | Due Date | Original Estimate | Cumulative Sum")
    print("-" * 55)
    for idx, row in daily_estimates.iterrows():
        print(f"{idx:5d} | {row['duedate'].strftime('%Y-%m-%d')} | {row['originalestimate']:16.2f} | {row['cumulative_sum']:13.2f}")
    
    # Create date range for the entire project period
    start_date = datetime(2025, 3, 12).date()
    end_date = datetime(2025, 5, 6).date()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D').date
    
    # Create a complete dataset with all dates
    complete_df = pd.DataFrame({'duedate': date_range})
    complete_df = complete_df.merge(daily_estimates, on='duedate', how='left')
    
    # Fill NaN values with 0 for originalestimate
    complete_df['originalestimate'] = complete_df['originalestimate'].fillna(0)
    
    # Calculate cumulative sum starting from 0
    complete_df['cumulative_sum'] = complete_df['originalestimate'].cumsum()
    
    # Get today's progress
    today, total_days = calculate_engineer_progress()
    
    # Calculate total days from all issues
    total_estimate_days = df['originalestimate'].sum()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the cumulative line
    line, = ax.plot(complete_df['duedate'], complete_df['cumulative_sum'], 
                    label='Cumulative Original Estimate', color='blue')
    ax.fill_between(complete_df['duedate'], complete_df['cumulative_sum'], 
                    alpha=0.3, color='blue')
    
    # Plot today's progress point
    ax.scatter(today, total_days, color='red', s=100, 
               label=f'Ideal Progress ({total_days:.1f} days)')
    
    # Customize the plot
    ax.set_title('Project Progress vs. Original Estimates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Days')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set y-axis maximum to total estimate days
    ax.set_ylim(0, total_estimate_days)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Create annotation for tooltips
    line_annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    line_annot.set_visible(False)
    
    progress_annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    progress_annot.set_visible(False)
    
    def update_annot(event):
        if event.inaxes == ax:
            # Get the x and y data
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            
            # Find the nearest point
            x = mdates.num2date(event.xdata)
            x = x.date()
            
            # Find the index of the nearest date
            idx = (complete_df['duedate'] - x).abs().idxmin()
            
            # Get the data for that date
            date = complete_df.loc[idx, 'duedate']
            value = complete_df.loc[idx, 'cumulative_sum']
            
            # Convert dates to numerical values for interpolation
            x_num = mdates.date2num(date)
            xdata_nums = mdates.date2num(complete_df['duedate'])
            
            # Calculate the distance from the mouse to the line at this x position
            y_line = np.interp(x_num, xdata_nums, complete_df['cumulative_sum'])
            distance = abs(event.ydata - y_line)
            
            # Calculate distance to progress point
            progress_x = mdates.date2num(today)
            progress_distance = np.sqrt((event.xdata - progress_x)**2 + (event.ydata - total_days)**2)
            
            # Only show line tooltip if within 5% of the y-axis range
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            threshold = y_range * 0.05
            
            # Show progress point tooltip if within 5% of the y-axis range
            progress_threshold = y_range * 0.05
            
            if distance < threshold:
                # Update line annotation
                line_annot.xy = (x_num, value)
                line_annot.set_text(f"{date.strftime('%Y-%m-%d')}\n{value:.1f} days")
                line_annot.set_visible(True)
                progress_annot.set_visible(False)
            elif progress_distance < progress_threshold:
                # Update progress annotation
                progress_annot.xy = (progress_x, total_days)
                progress_annot.set_text(f"Ideal Progress\n{total_days:.1f} days")
                progress_annot.set_visible(True)
                line_annot.set_visible(False)
            else:
                line_annot.set_visible(False)
                progress_annot.set_visible(False)
            
            fig.canvas.draw_idle()
        else:
            line_annot.set_visible(False)
            progress_annot.set_visible(False)
            fig.canvas.draw_idle()
    
    # Connect the event handler
    fig.canvas.mpl_connect('motion_notify_event', update_annot)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def main():
    df = get_jira_data()
    create_chart(df)

if __name__ == "__main__":
    main() 