import logging
from dataclasses import dataclass, fields
import yaml
from libs.file_lib import os_path

@dataclass
class Score:
    next_run: int
    best_score: float
    best_run: int

def save_score_to_yaml(score, file_path):
    file_path = os_path(file_path)
    print (file_path)
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(score, field.name) for field in fields(score)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_score_from_yaml(file_path):
    file_path = os_path(file_path)
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return Score(**config_dict)
