from libs.log_lib import setup_logging
from trainer.trainer import Chess_Trainer
from config.trainer_config import load_trainer_config_from_yaml
from config.net_config import get_net, load_net_config_from_yaml
from config.dl_config import load_dl_config_from_yaml
from config.conv_config import get_converter, load_conv_config_from_yaml
import logging


def create_net(converter):
    net_config = load_net_config_from_yaml("net_config.yaml")
    logging.info(f"Net Config: {net_config}")
    net_class = get_net(net_config)
    net_class.init_net(net_config, converter)
    return net_class

def create_converter():
    conv_config = load_conv_config_from_yaml("conv_config.yaml")
    logging.info(f"Converter Config: {conv_config}")
    return get_converter(conv_config)

def main():
    converter = create_converter()
    net = create_net(converter)
    trainer_config = load_trainer_config_from_yaml("trainer_config.yaml")
    logging.info(f"Trainer Config: {trainer_config}")
    
    dl_config = load_dl_config_from_yaml("dl_config.yaml")
    logging.info(f"DL Config: {dl_config}")
    
    #trainer = Chess_Trainer(trainer_config, data_loaders, machine)

if __name__ == '__main__':
    setup_logging()
    logging.info("Start Training")
    main()
    logging.info("End Training")
