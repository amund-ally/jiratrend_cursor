"""UI state management for Streamlit."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st

from src.config.chart_config import ChartConfig
from src.config.jira_config import JiraConfig
from src.core.data_service import SimulationParams


class StreamlitUIState:
    """Manages Streamlit session state for the application."""
    
    @staticmethod
    def initialize_chart_config_state():
        """Initialize session state for chart configuration."""
        if 'chart_name' not in st.session_state:
            st.session_state.chart_name = ""
        if 'hours_per_day' not in st.session_state:
            st.session_state.hours_per_day = 12.8
        if 'start_date' not in st.session_state:
            st.session_state.start_date = (datetime.today() - timedelta(days=7))
        if 'end_date' not in st.session_state:
            st.session_state.end_date = datetime.today()
        if 'jira_query' not in st.session_state:
            st.session_state.jira_query = ""
    
    @staticmethod
    def initialize_jira_config_state():
        """Initialize session state for JIRA configuration."""
        if 'server_url' not in st.session_state:
            st.session_state.server_url = "https://your-jira-instance.atlassian.net"
        if 'username' not in st.session_state:
            st.session_state.username = ""
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
    
    @staticmethod
    def initialize_simulation_state():
        """Initialize session state for simulation parameters."""
        if 'what_if_days' not in st.session_state:
            st.session_state.what_if_days = 0.0
        if 'velocity_multiplier' not in st.session_state:
            st.session_state.velocity_multiplier = 1.0
        
        # Ensure the initial values are valid
        st.session_state.what_if_days = max(0.0, st.session_state.what_if_days)
        st.session_state.velocity_multiplier = max(0.5, min(2.0, st.session_state.velocity_multiplier))
    
    @staticmethod
    def initialize_all():
        """Initialize all session state variables."""
        StreamlitUIState.initialize_chart_config_state()
        StreamlitUIState.initialize_jira_config_state()
        StreamlitUIState.initialize_simulation_state()
    
    @staticmethod
    def update_chart_config_state(config: ChartConfig):
        """Update session state from a ChartConfig object."""
        st.session_state.chart_name = config.name
        st.session_state.hours_per_day = config.hours_per_day
        st.session_state.start_date = config.start_date
        st.session_state.end_date = config.end_date
        st.session_state.jira_query = config.jira_query
    
    @staticmethod
    def update_jira_config_state(config: JiraConfig):
        """Update session state from a JiraConfig object."""
        st.session_state.server_url = config.server_url
        st.session_state.username = config.username
        st.session_state.api_key = config.api_key
    
    @staticmethod
    def get_simulation_params() -> SimulationParams:
        """Get current simulation parameters from session state."""
        return SimulationParams(
            what_if_days=st.session_state.get('what_if_days', 0.0),
            velocity_multiplier=st.session_state.get('velocity_multiplier', 1.0)
        )
    
    @staticmethod
    def create_chart_config_from_state() -> ChartConfig:
        """Create a ChartConfig object from current session state."""
        return ChartConfig(
            name=st.session_state.chart_name,
            hours_per_day=st.session_state.hours_per_day,
            start_date=datetime.combine(st.session_state.start_date, datetime.min.time()) 
                if isinstance(st.session_state.start_date, datetime.date) 
                else st.session_state.start_date,
            end_date=datetime.combine(st.session_state.end_date, datetime.min.time())
                if isinstance(st.session_state.end_date, datetime.date)
                else st.session_state.end_date,
            jira_query=st.session_state.jira_query
        )
    
    @staticmethod
    def create_jira_config_from_state() -> JiraConfig:
        """Create a JiraConfig object from current session state."""
        return JiraConfig(
            server_url=st.session_state.server_url,
            username=st.session_state.username,
            api_key=st.session_state.api_key
        )