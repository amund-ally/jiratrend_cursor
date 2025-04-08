"""Streamlit UI for the JiraTrend application."""
import streamlit as st
from datetime import datetime, timedelta

from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig
from src.core.data_service import DataService, ProjectData, SimulationParams
from src.ui.state_management import StreamlitUIState
from src.visualization.charts import create_progress_chart, create_state_time_boxplot, create_state_time_chart
from src.visualization.tables import create_tables, create_state_time_dataframe


def get_metrics_explanation() -> str:
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


def create_sidebar() -> tuple[JiraConfig, ChartConfig]:
    """Create and manage the sidebar UI components.
    
    Returns:
        tuple[JiraConfig, ChartConfig]: The current JIRA and chart configurations
    """
    jira_config = None
    chart_config = None
    
    # Create tabs for different configurations
    tab1, tab2 = st.sidebar.tabs(["Chart Settings", "JIRA Server"])
    
    with tab1:
        st.header("Chart Configuration")
        
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
                StreamlitUIState.update_chart_config_state(config)
                chart_config = config
                st.toast(f"Loaded configuration: {selected_config}")
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
        
        # Chart configuration input form
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
                        chart_config = config
                        st.toast(f"Chart configuration '{chart_name}' saved!")
                    except Exception as e:
                        st.error(f"Error saving configuration: {str(e)}")
    
    with tab2:
        st.header("JIRA Server Configuration")
        
        # Input fields using session state
        server_url = st.text_input("JIRA Server URL", key="server_url_input", value=st.session_state.server_url)
        username = st.text_input("Username", key="username_input", value=st.session_state.username)
        api_key = st.text_input("API Key", key="api_key_input", value=st.session_state.api_key, type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load JIRA Config"):
                try:
                    config = JiraConfig.from_config_file()
                    StreamlitUIState.update_jira_config_state(config)
                    jira_config = config
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading JIRA configuration: {str(e)}")
        
        with col2:
            if st.button("Save JIRA Config"):
                config = JiraConfig(username=username, api_key=api_key, server_url=server_url)
                config.save_to_config_file()
                jira_config = config
                st.toast("JIRA configuration saved!")
    
    # If config objects weren't created in the sidebar, create them from session state
    if chart_config is None and st.session_state.chart_name:
        try:
            chart_config = ChartConfig.from_config_file(st.session_state.chart_name)
        except Exception:
            pass
    
    if jira_config is None:
        try:
            jira_config = JiraConfig.from_config_file()
        except Exception:
            pass
    
    return jira_config, chart_config


def create_simulation_form() -> SimulationParams:
    """Create the simulation form UI.
    
    Returns:
        SimulationParams: The current simulation parameters
    """
    with st.form(key="simulation_form"):
        # Only use the key parameter to bind to session state, don't also set value
        st.slider(
            "What if the team completed additional work today?",
            min_value=0.0,
            max_value=20.0,
            key="what_if_days",
            step=1.0,
            help="Simulate additional completed work (in days) to see how it affects the project timeline"
        )
        
        st.slider(
            "Adjust velocity multiplier",
            min_value=0.5,
            max_value=2.0, 
            key="velocity_multiplier",
            step=0.1,
            help="Simulate increased/decreased team velocity (1.0 = no change)"
        )
        
        st.form_submit_button("Run Simulation")
    
    # Return the simulation parameters directly from session state
    # The form widget keys update session state automatically when changed
    return SimulationParams(
        what_if_days=st.session_state.what_if_days,
        velocity_multiplier=st.session_state.velocity_multiplier
    )


def display_metrics(metrics: dict, simulation: SimulationParams):
    """Display project metrics in the UI.
    
    Args:
        metrics: Dictionary of calculated metrics
        simulation: Current simulation parameters
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Scope",
            f"{metrics['total_scope']:.1f} days",
            help="Total estimated work in days",
        )

    with col2:
        label = "Completed Work" if simulation.what_if_days == 0 else f"Completed Work (with +{simulation.what_if_days:.1f})"
        value = f"{metrics['completed_work']:.1f} days" if simulation.what_if_days == 0 else f"{metrics['what_if_completed']:.1f} days"
        delta = f"+{simulation.what_if_days:.1f}" if simulation.what_if_days > 0 else None
        
        st.metric(
            label,
            value,
            delta=delta,
            help="Total completed work in days (including 'what if' scenario if applicable)"
        )

    with col3:
        label = "Remaining Work" if simulation.what_if_days == 0 else f"Remaining Work (with +{simulation.what_if_days:.1f})"
        value = f"{metrics['remaining_work']:.1f} days" if simulation.what_if_days == 0 else f"{metrics['what_if_remaining']:.1f} days"
        delta = f"-{simulation.what_if_days:.1f}" if simulation.what_if_days > 0 else None
        delta_color = "inverse" if simulation.what_if_days > 0 else "normal"
        
        st.metric(
            label,
            value,
            delta=delta,
            delta_color=delta_color,
            help="Remaining work in days (accounting for 'what if' scenario if applicable)"
        )

    with col4:
        st.metric(
            "Weekdays Since Last Completed",
            f"{metrics['days_since_last_completed']} days",
            help="Weekdays since the last completed issue"
        )


def display_burnup_chart(project_data: ProjectData, chart_config: ChartConfig, simulation: SimulationParams):
    """Display the burnup chart tab content.
    
    Args:
        project_data: Project data container
        chart_config: Current chart configuration
        simulation: Current simulation parameters
    """
    st.subheader("Burnup Chart")
    
    with st.expander("Simulation", expanded=False):
        simulation = create_simulation_form()
    
    # Create burnup chart
    fig = create_progress_chart(
        project_data.completed_df, 
        project_data.scope_df, 
        chart_config, 
        what_if_days=simulation.what_if_days,
        velocity_multiplier=simulation.velocity_multiplier
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display metrics
    metrics = DataService.calculate_metrics(project_data, simulation)
    display_metrics(metrics, simulation)
    
    # Create and display tables
    tables = create_tables(project_data.completed_df)
    st.plotly_chart(tables.completed, use_container_width=True)
    st.plotly_chart(tables.stats, use_container_width=True)
    
    # Add collapsible metrics explanation
    with st.expander("📊 Understanding the Metrics", expanded=False):
        st.markdown(get_metrics_explanation())


def display_issue_analysis(project_data: ProjectData):
    """Display the issue analysis tab content.
    
    Args:
        project_data: Project data container
    """
    st.subheader("Issue Analysis")
    
    # Add explanatory text
    st.info("""
    This analysis shows how much time issues spend in each state. 
    The chart below highlights issues that spend unusually long time in certain states.
    
    - Blue diamonds show average time in each state
    - Red points indicate issues that took more than twice the average time
    """)
    
    col1, col2 = st.columns(2)

    with col1:
        # Add state time chart and table
        state_time_boxplot = create_state_time_boxplot(project_data.state_time_df)
        st.plotly_chart(state_time_boxplot)
    with col2:
        state_time_chart = create_state_time_chart(project_data.state_time_df)
        st.plotly_chart(state_time_chart)
    
    # Add state time table
    state_time_table_df = create_state_time_dataframe(project_data.state_time_df)
    st.dataframe(
        state_time_table_df,
        column_config={
            "Issue": None,
            "Issue URL": st.column_config.LinkColumn(
                "Issue",
                help="Click to open in Jira",
                display_text=r"https:\/\/agrium\.atlassian\.net\/browse\/(.*)"
            ),
            **{col: st.column_config.NumberColumn(
                col, format="%.2f", width="small"
            ) for col in state_time_table_df.columns if col != "Issue" and col != "Issue URL"},
        },
        use_container_width=True,
        hide_index=True,
    )


def create_ui():
    """Create and run the Streamlit UI."""
    # Configure the page with optimized settings
    st.set_page_config(
        page_title="JiraTrend Reports",
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
    
    # Initialize all session state
    StreamlitUIState.initialize_all()
    
    # Create sidebar and get configurations
    jira_config, chart_config = create_sidebar()
    
    # Main content
    try:
        # Check if configurations are loaded
        if not chart_config or not chart_config.name:
            st.info("👈 Please select or create a chart configuration in the sidebar to view the chart.")
            return
            
        if not jira_config:
            st.error("⚠️ JIRA configuration not found. Please configure your JIRA connection in the sidebar.")
            return
        
        # Fetch and process data
        with st.spinner("Fetching JIRA data..."):
            project_data = DataService.fetch_project_data(jira_config, chart_config)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Burnup Chart", "Issue Analysis"])
        
        with tab1:
            display_burnup_chart(project_data, chart_config, StreamlitUIState.get_simulation_params())
        
        with tab2:
            display_issue_analysis(project_data)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your configurations in the sidebar.")