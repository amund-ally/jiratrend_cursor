import configparser
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class JiraConfig:
    """Configuration class for JIRA credentials."""
    username: str
    api_key: str
    server_url: str

    @classmethod
    def from_config_file(cls, config_file: str = 'config/jira.config') -> 'JiraConfig':
        """Create JiraConfig instance from config file."""
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at {config_file}. Please save your configuration first.")
        config.read(config_file)
        return cls(
            username=config['JIRA']['username'],
            api_key=config['JIRA']['api_key'],
            server_url=config['JIRA']['server_url']
        )

    def save_to_config_file(self, config_file: str = 'config/jira.config') -> None:
        """Save the current configuration to a file."""
        # Ensure the config directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        config = configparser.ConfigParser()
        config['JIRA'] = {
            'username': self.username,
            'api_key': self.api_key,
            'server_url': self.server_url
        }
        with open(config_file, 'w') as f:
            config.write(f) 