import configparser
import os
from dataclasses import dataclass
from datetime import datetime

class ChartConfig:
    """Configuration for the chart display."""
    
    def __init__(self, name: str, hours_per_person_per_day: float, team_size: int,
                 start_date: datetime, end_date: datetime, jira_query: str):
        """Initialize chart configuration.
        
        Args:
            name: Configuration name
            hours_per_person_per_day: Productive hours one person works per day
            team_size: Number of people in the team
            start_date: Chart start date
            end_date: Chart end date
            jira_query: JQL query to fetch issues
        """
        self.name = name
        self.hours_per_person_per_day = hours_per_person_per_day
        self.team_size = team_size
        self.start_date = start_date
        self.end_date = end_date
        self.jira_query = jira_query
        
    @property
    def hours_per_day(self) -> float:
        """Total team hours per day (for backward compatibility)."""
        return self.hours_per_person_per_day * self.team_size

    @classmethod
    def from_config_file(cls, config_name: str) -> 'ChartConfig':
        """Create ChartConfig instance from config file."""
        config = configparser.ConfigParser()
        config_file = f'config/chart_{config_name}.config'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at {config_file}")
        
        config.read(config_file)
        return cls(
            name=config['CHART']['name'],
            hours_per_person_per_day=float(config['CHART']['hours_per_person_per_day']),
            team_size=int(config['CHART']['team_size']),
            start_date=datetime.strptime(config['CHART']['start_date'], '%Y-%m-%d'),
            end_date=datetime.strptime(config['CHART']['end_date'], '%Y-%m-%d'),
            jira_query=config['CHART']['jira_query']
        )

    def save_to_config_file(self) -> None:
        """Save the current configuration to a file."""
        # Ensure the config directory exists
        os.makedirs('config', exist_ok=True)
        
        config = configparser.ConfigParser()
        config['CHART'] = {
            'name': self.name,
            'hours_per_person_per_day': str(self.hours_per_person_per_day),
            'team_size': str(self.team_size),
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'jira_query': self.jira_query
        }
        
        config_file = f'config/chart_{self.name}.config'
        with open(config_file, 'w') as f:
            config.write(f)

    @staticmethod
    def list_available_configs() -> list[str]:
        """List all available chart configurations."""
        config_dir = 'config'
        if not os.path.exists(config_dir):
            return []
        
        configs = []
        for file in os.listdir(config_dir):
            if file.startswith('chart_') and file.endswith('.config'):
                config_name = file[6:-7]  # Remove 'chart_' prefix and '.config' suffix
                configs.append(config_name)
        return sorted(configs)