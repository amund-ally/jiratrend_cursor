import configparser
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChartConfig:
    """Configuration class for chart settings."""
    name: str
    hours_per_day: float
    start_date: datetime
    end_date: datetime
    jira_query: str

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
            hours_per_day=float(config['CHART']['hours_per_day']),
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
            'hours_per_day': str(self.hours_per_day),
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