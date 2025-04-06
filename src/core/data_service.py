"""Business logic for fetching and processing JIRA data."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.chart_config import ChartConfig
from src.config.jira_config import JiraConfig
from src.data.jira_client import get_jira_data


class ProjectData(NamedTuple):
    """Container for all project data needed for visualization."""
    completed_df: pd.DataFrame
    scope_df: pd.DataFrame
    state_time_df: pd.DataFrame


@dataclass
class SimulationParams:
    """Parameters for simulation scenarios."""
    what_if_days: float = 0.0
    velocity_multiplier: float = 1.0


class DataService:
    """Service for fetching and processing project data."""
    
    @staticmethod
    def fetch_project_data(jira_config: JiraConfig, chart_config: ChartConfig) -> ProjectData:
        """Fetch all required data for project visualization."""
        completed_df, scope_df, state_time_df = get_jira_data(jira_config, chart_config)
        return ProjectData(completed_df, scope_df, state_time_df)
    
    @staticmethod
    def calculate_metrics(project_data: ProjectData, simulation: Optional[SimulationParams] = None) -> Dict:
        """Calculate key project metrics."""
        if simulation is None:
            simulation = SimulationParams()
            
        completed_df = project_data.completed_df
        scope_df = project_data.scope_df
        
        # Calculate metrics
        total_scope = scope_df['total_estimate'].max()
        completed_work = completed_df[completed_df['duedate'].notna()]['originalestimate'].sum()
        what_if_completed = completed_work + simulation.what_if_days
        remaining_work = total_scope - completed_work
        what_if_remaining = total_scope - what_if_completed
        
        # Only calculate days since last completed if there are completed items
        days_since_last = 0
        if not completed_df.empty and completed_df['duedate'].notna().any():
            days_since_last = np.busday_count(
                pd.Timestamp(completed_df['duedate'].max()).date(), 
                datetime.now().date()
            )
        
        return {
            'total_scope': total_scope,
            'completed_work': completed_work,
            'what_if_completed': what_if_completed,
            'remaining_work': remaining_work,
            'what_if_remaining': what_if_remaining,
            'days_since_last_completed': days_since_last
        }