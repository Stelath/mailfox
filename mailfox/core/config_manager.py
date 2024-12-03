import os
import yaml

CONFIG_FILE = os.path.expanduser("~/.mailfox/config.yaml")

def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)

def read_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError("Configuration file not found.")
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)