import logging
from dataclasses import dataclass, fields
import yaml
from typing import Dict, List
from itertools import product
from config.trainer_config import Trainer_Config

@dataclass
class Trainer_Config_List():
    learning_rate: List[float]
    weight_decay: List[float]
    epochs: List[int]
    optimizer: List[str]
    reduction: List[float]
    criterion: List[float]
    device: str
    test_threshold: float
    shuffle: bool
    training_steps: Dict[str, str]
    conv_config: str
    dl_config: str
    net_config: str
    #
    # Special Lib for NVIDIA
    # Recommend to set to False, when reproducibility is importatnt
    # unclear, if it can be used on macos
    #
    cudnn_enabled: bool = False
    
def save_trainer_config_list_to_yaml(config, file_path):
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_trainer_config_list_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return Trainer_Config_List(**config_dict)

def config_iterator(trainer_config_list):
    list_values = [
        trainer_config_list.learning_rate,
        trainer_config_list.weight_decay,
        trainer_config_list.epochs,
        trainer_config_list.optimizer,
        trainer_config_list.reduction,
        trainer_config_list.criterion,
    ]
    trainer_config = Trainer_Config(
        cudnn_enabled = trainer_config_list.cudnn_enabled,
        device = trainer_config_list.device,
        test_threshold = trainer_config_list.test_threshold,
        shuffle = trainer_config_list.shuffle,
        training_steps = trainer_config_list.training_steps,
        conv_config = trainer_config_list.conv_config,
        dl_config = trainer_config_list.dl_config,
        net_config = trainer_config_list.net_config,
        learning_rate = 0.0,
        weight_decay = 0.0,
        epochs = 0,
        optimizer = None,
        reduction = None,
        criterion = None
        )
    
    # Generate all combinations
    all_combinations = product(*list_values)

    # Iterate through combinations and yield Trainer_Config_List instances
    for combination in all_combinations:
        trainer_config.learning_rate = combination[0]
        trainer_config.weight_decay = combination[1]
        trainer_config.epochs = combination[2]
        trainer_config.optimizer = combination[3]
        trainer_config.reduction = combination[4]
        trainer_config.criterion = combination[5]
        logging.info(f"Config {trainer_config.learning_rate} {trainer_config.weight_decay} {trainer_config.epochs} {trainer_config.optimizer} {trainer_config.reduction} {trainer_config.criterion}")
        yield trainer_config