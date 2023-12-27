import logging
from dataclasses import dataclass, fields
from typing import Optional, Dict
import yaml

@dataclass
class Net_Config():

    net_file: str
    expected_hash: int
    net: str
    nodes: Dict[str, int]
    device: str

def save_net_config_to_yaml(config, file_path):
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_net_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return Net_Config(**config_dict)