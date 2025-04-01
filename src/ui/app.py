import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig
from src.data.jira_client import get_jira_data
from src.visualization.charts import create_progress_chart
from src.visualization.tables import create_tables

def get_metrics_explanation():
    """Returns the explanation text for the metrics panel."""
    return """
    ### Understanding Estimation Metrics
    
    #### Basic Statistics
    - **Mean**: The average value of estimates or actual time spent. Helps identify if you generally over or underestimate.
    - **Standard Deviation (σ)**: Measures how spread out your estimates/actuals are from their mean. 
      - Lower σ = more consistent estimates
      - Higher σ = more variable estimates
    
    #### Advanced Metrics
    - **Error (Act-Est)**: The difference between actual time and estimated time
      - Positive = You tend to underestimate
      - Negative = You tend to overestimate
      - The standard deviation of error tells you how consistent your estimation errors are
    
    - **Coefficient of Variation (CV = σ/μ)**: Standard deviation relative to the mean
      - More useful than raw standard deviation because it's relative to task size
      - Lower CV = more consistent estimates/actuals
      - If estimate CV > actual CV: Your estimates vary more than actual work time
      - If actual CV > estimate CV: The actual work time varies more than your estimates
    
    #### Interpreting Accuracy
    - The accuracy percentage shows what portion of estimates fall within one standard deviation of typical error
    - In a normal distribution, about 68% should fall within ±1 standard deviation
    - If your percentage is much lower: Your estimates are less reliable
    - If your percentage is higher: Your estimates are more consistent
    
    #### Using These Metrics
    1. Use Mean Error to adjust estimates up/down systematically
    2. Use Standard Deviation to understand your typical range of uncertainty
    3. Use CV to compare estimate consistency vs actual work consistency
    4. Use Accuracy % to set confidence intervals for future estimates
    """

def create_ui():
    """Create and run the Streamlit UI."""
    # Configure the page with optimized settings
    st.set_page_config(
        page_title="Reports",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    
    # Disable animations
    st.markdown("""
        <style>
        .stApp {
            animation: none !important;
            transition: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Charts")
    
    # Sidebar for configuration
    with st.sidebar:
        # Create tabs for different configurations
        tab1, tab2 = st.tabs(["Chart Settings", "JIRA Server"])
        
        with tab1:
            st.header("Chart Configuration")
            
            # Initialize session state for chart input fields if not exists
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
            
            # Load existing configurations
            available_configs = ChartConfig.list_available_configs()
            selected_config = st.selectbox(
                "Load Existing Configuration",
                [""] + available_configs,
                format_func=lambda x: "Select a configuration..." if x == "" else x
            )
            
            if selected_config:
                try:
                    config = ChartConfig.from_config_file(selected_config)
                    st.session_state.chart_name = config.name
                    st.session_state.hours_per_day = config.hours_per_day
                    st.session_state.start_date = config.start_date
                    st.session_state.end_date = config.end_date
                    st.session_state.jira_query = config.jira_query
                    st.toast(f"Loaded configuration: {selected_config}")
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
            
            # chart configuration input form
            with st.form(key="chart_config_form"):
                chart_name = st.text_input("Configuration Name", value=st.session_state.chart_name)
                st.subheader("Burnup Settings")
                hours_per_day = st.number_input("Hours per Day", min_value=0.0, max_value=24.0, 
                                            value=st.session_state.hours_per_day, step=0.1)
                start_date = st.date_input("Start Date", value=st.session_state.start_date)
                end_date = st.date_input("End Date", value=st.session_state.end_date)
                jira_query = st.text_area("JQL Query", value=st.session_state.jira_query, height=200)
                
                submitted = st.form_submit_button("Save Chart Config")
                if submitted:
                    if not chart_name:
                        st.error("Please enter a configuration name")
                    else:
                        try:
                            config = ChartConfig(
                                name=chart_name,
                                hours_per_day=hours_per_day,
                                start_date=datetime.combine(start_date, datetime.min.time()),
                                end_date=datetime.combine(end_date, datetime.min.time()),
                                jira_query=jira_query
                            )
                            config.save_to_config_file()
                            st.toast(f"Chart configuration '{chart_name}' saved!")
                        except Exception as e:
                            st.error(f"Error saving configuration: {str(e)}")
        
        with tab2:
            st.header("JIRA Server Configuration")
            
            # Initialize session state for JIRA input fields if not exists
            if 'server_url' not in st.session_state:
                st.session_state.server_url = "https://your-jira-instance.atlassian.net"
            if 'username' not in st.session_state:
                st.session_state.username = ""
            if 'api_key' not in st.session_state:
                st.session_state.api_key = ""
            
            # Input fields using session state
            server_url = st.text_input("JIRA Server URL", key="server_url_input", value=st.session_state.server_url)
            username = st.text_input("Username", key="username_input", value=st.session_state.username)
            api_key = st.text_input("API Key", key="api_key_input", value=st.session_state.api_key, type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Load JIRA Config"):
                    try:
                        config = JiraConfig.from_config_file()
                        # Update session state values
                        st.session_state.server_url = config.server_url
                        st.session_state.username = config.username
                        st.session_state.api_key = config.api_key
                        # Force a rerun to update the input fields
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading JIRA configuration: {str(e)}")
            
            with col2:
                if st.button("Save JIRA Config"):
                    config = JiraConfig(username=username, api_key=api_key, server_url=server_url)
                    config.save_to_config_file()
                    st.toast("JIRA configuration saved!")
    
    # Main content
    try:
        # Check if configurations are loaded
        if not st.session_state.chart_name:
            st.info("👈 Please select or create a chart configuration in the sidebar to view the chart.")
            return
            
        try:
            jira_config = JiraConfig.from_config_file()
        except Exception as e:
            st.error("⚠️ JIRA configuration not found. Please configure your JIRA connection in the sidebar.")
            return
            
        try:
            chart_config = ChartConfig.from_config_file(st.session_state.chart_name)
        except Exception as e:
            st.error("⚠️ Chart configuration not found. Please select or create a chart configuration in the sidebar.")
            return
        
        # Fetch and process data
        with st.spinner("Fetching JIRA data..."):
            completed_df, scope_df = get_jira_data(jira_config, chart_config)
        
        # Create and display the chart
        fig = create_progress_chart(completed_df, scope_df, chart_config)
        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_scope = scope_df['total_estimate'].max()
            st.metric(
                "Total Scope",
                f"{total_scope:.1f} days",
                help="Total estimated work in days",
            )
        
        with col2:
            # Only count completed issues (those with a duedate)
            completed_work = completed_df[completed_df['duedate'].notna()]['originalestimate'].sum()
            st.metric(
                "Completed Work",
                f"{completed_work:.1f} days",
                help="Total completed work in days"
            )
        
        with col3:
            remaining_work = total_scope - completed_work
            st.metric(
                "Remaining Work",
                f"{remaining_work:.1f} days",
                help="Remaining work in days"
            )

        with col4:
            days_since_last_completed = np.busday_count(pd.Timestamp(completed_df['duedate'].max()).date(), datetime.now().date())
            st.metric(
                "Work Days Since Last Completed",
                f"{days_since_last_completed} days",
                help="Weekdays since the last completed issue"
            )

        # Create and display tables
        tables = create_tables(completed_df)
        st.plotly_chart(tables.completed, use_container_width=True)
        st.plotly_chart(tables.stats, use_container_width=True)
        
        # Add collapsible metrics explanation
        with st.expander("📊 Understanding the Metrics", expanded=False):
            st.markdown(get_metrics_explanation())

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your configurations in the sidebar.")