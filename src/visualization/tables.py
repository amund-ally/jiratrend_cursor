import pandas as pd
from src.config.jira_config import JiraConfig
import plotly.graph_objects as go


def create_completed_table(completed_df: pd.DataFrame, jira_config: JiraConfig):
   # Assuming completed_df is your DataFrame
   filtered_df = completed_df[completed_df['duedate'].notna()][['key', 'duedate', 'originalestimate']].copy()

   # Select the relevant columns for display
   filtered_df = filtered_df[['key', 'duedate', 'originalestimate']].copy()
   filtered_df.columns = ['Issue', 'Due Date', 'Original Estimate']  # Custom titles

   fig = go.Figure(data=[go.Table(
      header=dict(values=['Issue', 'Due Date', 'Original Estimate'],
                  fill_color='rgba(0,0,255,0.3)',
                  font=dict(color='black', size=14),  # Header font color and size
                  align='left'),
      cells=dict(values=[
         filtered_df['Issue'],  # Use the formatted Issue column
         filtered_df['Due Date'].astype(str),  # Convert dates to string for display
         filtered_df['Original Estimate']
      ],
      fill_color='rgba(0,255,0,0.1)',
      font=dict(color='black', size=12),  # Cell font color and size
      align='left',
      height=30)
   )])

   fig.update_layout(
      height=400,
      title='Completed Issues',  # Optional title for the table
   )

   return fig
