import logging
from dataclasses import dataclass, fields
import yaml
from libs.import_lib import get_class
from torch.utils.data import Dataset
from libs.file_lib import os_path

@dataclass
class DL_Config():
   
    data_directory: str
    filename: str
    delimiter: str
    lib: str
    class_name: str

def save_dl_config_to_yaml(config, file_path):
    file_path = os_path(file_path)
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}

    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_dl_config_from_yaml(file_path):
    file_path = os_path(file_path)
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return DL_Config(**config_dict)

def get_dl(dl_config):
    dl_lib = "data_loader." + dl_config.lib
    dl_class = dl_config.class_name
    return get_class(dl_lib, dl_class, Dataset)
