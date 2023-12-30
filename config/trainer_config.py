import logging
from dataclasses import dataclass, fields
import yaml
from typing import Dict

@dataclass
class Trainer_Config():
    learning_rate: float
    weight_decay: float
    epochs: int
    optimizer: str
    reduction: str
    criterion: str
    device: str
    test_threshold: float
    shuffle: bool
    training_steps: Dict[str, str] = None

    #
    # Special Lib for NVIDIA
    # Recommend to set to False, when reproducibility is importatnt
    # unclear, if it can be used on macos
    #
    cudnn_enabled: bool = False

def save_trainer_config_to_yaml(config, file_path):
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_trainer_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return Trainer_Config(**config_dict)