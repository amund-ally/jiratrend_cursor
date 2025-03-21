import streamlit as st
from datetime import datetime, timedelta
from src.config.jira_config import JiraConfig
from src.config.project_config import ProjectConfig
from src.data.jira_client import get_jira_data
from src.visualization.charts import create_chart
from src.visualization.tables import create_completed_table

def create_ui():
    """Create and run the Streamlit UI."""
    # Configure the page with optimized settings
    st.set_page_config(
        page_title="JIRA Reports",
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
            
            # Initialize session state for project input fields if not exists
            if 'project_name' not in st.session_state:
                st.session_state.project_name = ""
            if 'hours_per_day' not in st.session_state:
                st.session_state.hours_per_day = 12.8
            if 'start_date' not in st.session_state:
                st.session_state.start_date = (datetime.today() - timedelta(days=7))
            if 'end_date' not in st.session_state:
                st.session_state.end_date = datetime.today()
            if 'jira_query' not in st.session_state:
                st.session_state.jira_query = ""
            
            # Load existing configurations
            available_configs = ProjectConfig.list_available_configs()
            selected_config = st.selectbox(
                "Load Existing Configuration",
                [""] + available_configs,
                format_func=lambda x: "Select a configuration..." if x == "" else x
            )
            
            if selected_config:
                try:
                    config = ProjectConfig.from_config_file(selected_config)
                    st.session_state.project_name = config.name
                    st.session_state.hours_per_day = config.hours_per_day
                    st.session_state.start_date = config.start_date
                    st.session_state.end_date = config.end_date
                    st.session_state.jira_query = config.jira_query
                    st.success(f"Loaded configuration: {selected_config}")
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
            
            # Project configuration input fields
            project_name = st.text_input("Configuration Name", value=st.session_state.project_name)
            st.subheader("Burnup Settings")
            hours_per_day = st.number_input("Hours per Day", min_value=0.0, max_value=24.0, 
                                          value=st.session_state.hours_per_day, step=0.1)
            start_date = st.date_input("Start Date", value=st.session_state.start_date)
            end_date = st.date_input("End Date", value=st.session_state.end_date)
            jira_query = st.text_area("JQL Query", value=st.session_state.jira_query, height=200)
            
            if st.button("Save Chart Config"):
                if not project_name:
                    st.error("Please enter a configuration name")
                else:
                    try:
                        config = ProjectConfig(
                            name=project_name,
                            hours_per_day=hours_per_day,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            end_date=datetime.combine(end_date, datetime.min.time()),
                            jira_query=jira_query
                        )
                        config.save_to_config_file()
                        st.success(f"Chart configuration '{project_name}' saved!")
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
                    st.success("JIRA configuration saved!")
    
    # Main content
    try:
        # Check if configurations are loaded
        if not st.session_state.project_name:
            st.info("👈 Please select or create a chart configuration in the sidebar to view the chart.")
            return
            
        try:
            jira_config = JiraConfig.from_config_file()
        except Exception as e:
            st.error("⚠️ JIRA configuration not found. Please configure your JIRA connection in the sidebar.")
            return
            
        try:
            project_config = ProjectConfig.from_config_file(st.session_state.project_name)
        except Exception as e:
            st.error("⚠️ Chart configuration not found. Please select or create a chart configuration in the sidebar.")
            return
        
        # Fetch and process data
        with st.spinner("Fetching JIRA data..."):
            completed_df, scope_df = get_jira_data(jira_config, project_config)
        
        # Create and display the chart
        fig = create_chart(completed_df, scope_df, project_config)
        st.plotly_chart(fig, use_container_width=True)

        # Create and display a table of completed issues
        fig = create_completed_table(completed_df, jira_config)
        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_scope = scope_df['total_estimate'].max()
            st.metric(
                "Total Scope",
                f"{total_scope:.1f} days",
                help="Total estimated work in days"
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

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your configurations in the sidebar.") 