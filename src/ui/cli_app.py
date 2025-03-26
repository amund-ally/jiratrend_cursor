#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import plotly.io as pio

from src.config.jira_config import JiraConfig
from src.config.chart_config import ChartConfig
from src.data.jira_client import get_jira_data
from src.visualization.charts import create_chart


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate JIRA burnup charts from the command line'
    )
    
    parser.add_argument(
        '--chart-config',
        type=str,
        required=True,
        help='Name of the chart configuration to use (e.g., "CAWO Migration")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Optional: Save chart to HTML file instead of displaying it'
    )
    
    parser.add_argument(
        '--format',
        choices=['html', 'png', 'jpg', 'svg', 'pdf'],
        default='html',
        help='Output format when saving chart (default: html)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI application."""
    args = parse_args()
    
    try:
        # Load JIRA configuration
        jira_config = JiraConfig.from_config_file()
        print(f"Loaded JIRA configuration for server: {jira_config.server_url}")
        
        # Load chart configuration
        chart_config = ChartConfig.from_config_file(args.chart_config)
        print(f"Loaded chart configuration: {args.chart_config}")
        print(f"Query: {chart_config.jira_query}")
        print(f"Date range: {chart_config.start_date.date()} to {chart_config.end_date.date()}")
        
        # Fetch JIRA data
        print("Fetching data from JIRA...")
        completed_df, scope_df = get_jira_data(jira_config, chart_config)
        print(f"Retrieved {len(completed_df)} issues")
        
        # Create chart
        print("Generating chart...")
        fig = create_chart(completed_df, scope_df, chart_config)
        
        # Output handling
        if args.output:
            # Save to file in requested format
            output_path = args.output
            if not output_path.endswith(f'.{args.format}'):
                output_path = f"{output_path}.{args.format}"
            
            if args.format == 'html':
                fig.write_html(output_path)
            else:
                fig.write_image(output_path)
                
            print(f"Chart saved to: {output_path}")
        else:
            # Show interactive plot
            print("Opening interactive plot...")
            fig.show()
            
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()