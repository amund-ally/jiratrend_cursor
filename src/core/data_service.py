"""Business logic for fetching and processing JIRA data."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Tuple
import logging

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
    estimate_changes: Dict  # Raw estimate changes by date


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
        completed_df, scope_df, state_time_df, estimate_changes = get_jira_data(jira_config, chart_config)
        return ProjectData(completed_df, scope_df, state_time_df, estimate_changes)
    
    @staticmethod
    def calculate_metrics(project_data: ProjectData, computed_values: Dict, simulation: Optional[SimulationParams] = None) -> Dict:
        """Calculate key project metrics."""
        logger = logging.getLogger(__name__)
        logger.debug("Starting metrics calculation")
        
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
        
        logger.debug(f"Basic metrics: total_scope={total_scope:.2f}, completed_work={completed_work:.2f}, remaining_work={remaining_work:.2f}")
        logger.debug(f"What-if metrics: what_if_completed={what_if_completed:.2f}, what_if_remaining={what_if_remaining:.2f}")
        
        # Only calculate days since last completed if there are completed items
        days_since_last = 0
        if not completed_df.empty and completed_df['duedate'].notna().any():
            days_since_last = np.busday_count(
                pd.Timestamp(completed_df['duedate'].max()).date(), 
                datetime.now().date()
            )
            logger.debug(f"Days since last completed issue: {days_since_last}")

        projected_date = computed_values.get('projected_completion_date')
        ideal_date = computed_values.get('ideal_completion_date')
        configured_hours_per_day = computed_values.get('total_hours_per_day', 8.0)
        hours_per_person_per_day = computed_values.get('hours_per_person_per_day', 8.0)
        team_size = computed_values.get('team_size', 1)
        
        logger.debug(f"Dates: projected_date={projected_date}, ideal_date={ideal_date}")
        logger.debug(f"Configuration: hours_per_person_per_day={hours_per_person_per_day}, team_size={team_size}, total_hours={configured_hours_per_day}")
        
        # Calculate difference in business days
        days_off = 0
        required_hours_per_day = configured_hours_per_day  # Default to configured value
        
        if projected_date and ideal_date:
            days_off = np.busday_count(ideal_date, projected_date)
            logger.debug(f"Schedule variance: {days_off} business days off from ideal")
            
            # Calculate required hours per day to meet ideal date
            today = datetime.now().date()
            logger.debug(f"Today's date: {today}")
            
            # Calculate business days between today and ideal date
            business_days_remaining = np.busday_count(today, ideal_date)
            logger.debug(f"Business days remaining until ideal date: {business_days_remaining}")
            
            if business_days_remaining > 0:
                # Daily work needed to complete remaining scope by ideal date
                daily_work_required = remaining_work / business_days_remaining
                logger.debug(f"Daily work required: {daily_work_required:.2f} days of work per calendar day")
                
                # Convert to hours per person per day
                logger.debug(f"Team configuration: hours_per_person_per_day={hours_per_person_per_day:.2f}, team_size={team_size}")

                # Calculate required hours per day
                required_hours_per_day = daily_work_required * 8
                logger.debug(f"Required hours per day (team total): {required_hours_per_day:.2f}")

                required_hours_per_person_per_day = required_hours_per_day / team_size
                logger.debug(f"Required hours per person per day: {required_hours_per_person_per_day:.2f}")
            else:
                # If already past ideal date, what would have been required
                logger.debug("Already past ideal date, calculating what would have been required")
                business_days_from_start = np.busday_count(computed_values.get('start_date', today), today)
                logger.debug(f"Business days elapsed since start: {business_days_from_start}")
                
                if business_days_from_start > 0:
                    required_hours_per_day = (total_scope * 8) / business_days_from_start
                    logger.debug(f"Required hours per day (historical): {required_hours_per_day:.2f}")
        else:
            logger.debug("Cannot calculate schedule variance - missing projected or ideal date")
        
        logger.debug("Completed metrics calculation")
        return {
            'total_scope': total_scope,
            'completed_work': completed_work,
            'what_if_completed': what_if_completed,
            'remaining_work': remaining_work,
            'what_if_remaining': what_if_remaining,
            'days_since_last_completed': days_since_last,
            'days_off': days_off,
            'required_hours_per_day': required_hours_per_day,
            'configured_hours_per_day': configured_hours_per_day,
            'team_size': team_size,
        }