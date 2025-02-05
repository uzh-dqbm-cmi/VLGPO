import yaml
from pathlib import Path

def load_config(setting: str):
    config_path = Path(__file__).resolve().parents[3] / 'configs' / f"{setting}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config