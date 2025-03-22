from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig
from src.data.jira_client import get_jira_data
from src.visualization.charts import create_chart
from src.visualization.tables_dash import create_tables

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f8f9fa;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td, 
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                padding: 10px;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            }
            .card {
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-radius: 8px;
                margin-bottom: 1rem;
            }
            .card-header {
                background-color: #f1f3f5;
                border-bottom: 1px solid #dee2e6;
                padding: 1rem;
            }
            .form-control {
                margin-bottom: 1rem;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 2rem;
                padding: 1rem 0;
            }

            /* Customize loading spinner */
            ._dash-loading-callback {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 1000;
            }
            
            ._dash-loading-callback::after {
                content: 'Loading...';
                margin-left: 10px;
                font-size: 14px;
                color: #2c3e50;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Sidebar configuration
    dbc.Row([
        # Left sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Chart Configuration", className="mb-0")),
                dbc.CardBody([
                    # Configuration selector
                    dcc.Dropdown(
                        id='config-selector',
                        options=[{'label': 'Select configuration...', 'value': ''}] + 
                                [{'label': c, 'value': c} for c in ChartConfig.list_available_configs()],
                        value='',
                        className='mb-3'
                    ),
                    
                    # Configuration form
                    dbc.Form([
                        dbc.Input(id='config-name', placeholder="Configuration Name", type='text'),
                        dbc.Input(id='hours-per-day', placeholder="Hours per Day", 
                                type='number', value=12.8, min=0, max=24, step=0.1),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=datetime.today() - timedelta(days=7),
                            end_date=datetime.today()
                        ),
                        dbc.Textarea(id='jql-query', placeholder="JQL Query", rows=5),
                        dbc.Button("Save Configuration", id='save-config', color="primary")
                    ])
                ], className="p-4")
            ], className="mb-4 shadow-sm"),
            
            # JIRA Configuration
            dbc.Card([
                dbc.CardHeader(html.H4("JIRA Configuration", className="mb-0")),
                dbc.CardBody([
                    dbc.Input(id='server-url', placeholder="JIRA Server URL", type='text'),
                    dbc.Input(id='username', placeholder="Username", type='text'),
                    dbc.Input(id='api-key', placeholder="API Key", type='password'),
                    dbc.Row([
                        dbc.Col(dbc.Button("Load Config", id='load-jira-config')),
                        dbc.Col(dbc.Button("Save Config", id='save-jira-config'))
                    ])
                ], className="p-4")
            ], className="shadow-sm")
        ], width=3, className="pe-4"),
        
        # Main content
        dbc.Col([
            dcc.Loading(
                id="loading-main",
                type="circle",
                children=[            
                    # Burnup Chart
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='burnup-chart')
                        ])
                    ], className="mb-4 shadow-sm"),
                    
                    # Metrics Cards
                    dbc.Row([
                        dbc.Col(dbc.Card(id='total-scope', className="text-center shadow-sm")),
                        dbc.Col(dbc.Card(id='completed-work', className="text-center shadow-sm")),
                        dbc.Col(dbc.Card(id='remaining-work', className="text-center shadow-sm"))
                    ], className="mb-4"),
                    
                    # Tables
                    dbc.Card([
                        dbc.CardHeader(html.H4("Completed Issues", className="mb-0")),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='completed-table',
                            columns=[
                                {'name': 'Issue', 'id': 'Issue'},
                                {'name': 'Due Date', 'id': 'Due Date'},
                                {'name': 'Est Time(d)', 'id': 'Est Time', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Actual Time(d)', 'id': 'Actual Time', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                            ],
                            style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'font-family': '-apple-system, BlinkMacSystemFont, sans-serif'
                                },
                                style_header={
                                    'backgroundColor': '#f1f3f5',
                                    'fontWeight': 'bold'
                                },
                                style_data={'backgroundColor': 'white'},
                                page_size=10
                            )
                        ], className="p-0")
                    ], className="mb-4 shadow-sm"),
                    
                    # Stats Table
                    dbc.Card([
                        dbc.CardHeader(html.H4("Statistics", className="mb-0")),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='stats-table',
                                columns=[
                                    {'name': 'Metric', 'id': 'Metric'},
                                    {'name': 'Estimates', 'id': 'Estimates'},
                                    {'name': 'Actual', 'id': 'Actual'},
                                    {'name': 'Interpretation', 'id': 'Interpretation'}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'font-family': '-apple-system, BlinkMacSystemFont, sans-serif'
                                },
                                style_header={
                                    'backgroundColor': '#f1f3f5',
                                    'fontWeight': 'bold'
                                },
                                style_data={'backgroundColor': 'white'}
                            )
                        ], className="p-0")
                    ], className="mb-4 shadow-sm")
                ]
            )
        ], width=9)
    ], className="g-4")
], fluid=True, className="px-4 py-4")

# Callbacks
@app.callback(
    [Output('burnup-chart', 'figure'),
     Output('completed-table', 'data'),
     Output('stats-table', 'data'),
     Output('total-scope', 'children'),
     Output('completed-work', 'children'),
     Output('remaining-work', 'children'),
     # form field outputs
     Output('config-name', 'value'),
     Output('hours-per-day', 'value'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date'),
     Output('jql-query', 'value')],
    [Input('config-selector', 'value')]
)
def update_charts(selected_config):
    """Update all dashboard components when configuration changes."""
    if not selected_config:
        empty_data = ({}, [], [], "", "", "")  # Dashboard outputs
        default_form = ("", 12.8, datetime.today() - timedelta(days=7), datetime.today(), "")  # Form outputs
        return empty_data + default_form
        
    try:
        # Load configurations
        jira_config = JiraConfig.from_config_file()
        chart_config = ChartConfig.from_config_file(selected_config)
        
        # Get data and create visualizations
        completed_df, scope_df = get_jira_data(jira_config, chart_config)
        chart_fig = create_chart(completed_df, scope_df, chart_config)
        completed_table, stats_table = create_tables(completed_df)

        # Calculate summary statistics
        total_scope = scope_df['total_estimate'].max()
        completed_work = completed_df[completed_df['duedate'].notna()]['originalestimate'].sum()
        remaining_work = total_scope - completed_work
        
        # Prepare all return values
        dashboard_data = (
            chart_fig,
            completed_table,
            stats_table,
            create_metric_card("Total Scope", f"{total_scope:.1f} days"),
            create_metric_card("Completed Work", f"{completed_work:.1f} days"),
            create_metric_card("Remaining Work", f"{remaining_work:.1f} days")
        )
        
        form_data = (
            chart_config.name,
            chart_config.hours_per_day,
            chart_config.start_date,
            chart_config.end_date,
            chart_config.jira_query
        )
        
        return dashboard_data + form_data
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        empty_data = ({}, [], [], error_msg, "", "")
        default_form = ("", 12.8, datetime.today() - timedelta(days=7), datetime.today(), "")
        return empty_data + default_form

# Callback for saving chart configuration
@app.callback(
    [Output('config-selector', 'options'),
     Output('config-selector', 'value')],
    [Input('save-config', 'n_clicks')],
    [State('config-name', 'value'),
     State('hours-per-day', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('jql-query', 'value')]
)
def save_configuration(n_clicks, name, hours_per_day, start_date, end_date, jql_query):
    """Save configuration and update dropdown options."""
    if n_clicks is None:
        raise PreventUpdate
        
    try:
        # Create and save new config
        config = ChartConfig(
            name=name,
            hours_per_day=hours_per_day,
            start_date=datetime.strptime(start_date.split('T')[0], '%Y-%m-%d'),
            end_date=datetime.strptime(end_date.split('T')[0], '%Y-%m-%d'),
            jira_query=jql_query
        )
        config.save_to_config_file()
        
        # Update dropdown options
        new_options = [{'label': 'Select configuration...', 'value': ''}] + \
                     [{'label': c, 'value': c} for c in ChartConfig.list_available_configs()]
        
        # Return new options and select the saved config
        return new_options, name
        
    except Exception as e:
        raise PreventUpdate
    
def create_metric_card(title, value):
    return dbc.CardBody([
        html.H4(title, className="card-title"),
        html.H2(value)
    ])