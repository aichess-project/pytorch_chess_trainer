import logging
from dataclasses import dataclass, fields
import yaml
import importlib
from converter.converter_base import Converter_Base
from libs.import_lib import get_class

@dataclass
class Conv_Config():
   
    lib: str
    class_name: str

def save_conv_config_to_yaml(config, file_path):
    # Extract the fields from the data class
    fields_dict = {field.name: getattr(config, field.name) for field in fields(config)}
    with open(file_path, 'w') as file:
        yaml.dump(fields_dict, file)

def load_conv_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Conv_Config(**config_dict)

def get_converter(con_config):
    converter_lib = "converter." + con_config.lib
    converter_class = con_config.class_name
    return get_class(converter_lib, converter_class, Converter_Base)