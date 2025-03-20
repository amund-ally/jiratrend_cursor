import configparser
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ProjectConfig:
    """Configuration class for project settings."""
    name: str
    hours_per_day: float
    start_date: datetime
    end_date: datetime
    jira_query: str

    @classmethod
    def from_config_file(cls, config_name: str) -> 'ProjectConfig':
        """Create ProjectConfig instance from config file."""
        config = configparser.ConfigParser()
        config_file = f'config/project_{config_name}.config'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at {config_file}")
        
        config.read(config_file)
        return cls(
            name=config['PROJECT']['name'],
            hours_per_day=float(config['PROJECT']['hours_per_day']),
            start_date=datetime.strptime(config['PROJECT']['start_date'], '%Y-%m-%d'),
            end_date=datetime.strptime(config['PROJECT']['end_date'], '%Y-%m-%d'),
            jira_query=config['PROJECT']['jira_query']
        )

    def save_to_config_file(self) -> None:
        """Save the current configuration to a file."""
        # Ensure the config directory exists
        os.makedirs('config', exist_ok=True)
        
        config = configparser.ConfigParser()
        config['PROJECT'] = {
            'name': self.name,
            'hours_per_day': str(self.hours_per_day),
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'jira_query': self.jira_query
        }
        
        config_file = f'config/project_{self.name}.config'
        with open(config_file, 'w') as f:
            config.write(f)

    @staticmethod
    def list_available_configs() -> list[str]:
        """List all available project configurations."""
        config_dir = 'config'
        if not os.path.exists(config_dir):
            return []
        
        configs = []
        for file in os.listdir(config_dir):
            if file.startswith('project_') and file.endswith('.config'):
                config_name = file[8:-7]  # Remove 'project_' prefix and '.config' suffix
                configs.append(config_name)
        return sorted(configs) 