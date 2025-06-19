import yaml
from pathlib import Path

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)