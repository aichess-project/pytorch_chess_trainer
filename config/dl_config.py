import logging
from dataclasses import dataclass, fields
import yaml

@dataclass
class DL_Config():
   
    data_directory: str
    filename: str 

def save_dl_config_to_yaml(config, file_path):
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_dl_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return DL_Config(**config_dict)
